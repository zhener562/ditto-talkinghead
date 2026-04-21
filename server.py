"""Ditto-TalkingHead FastAPI server.

hage プロジェクトの tts_server (tts_server/musetalk_service.py) と対称な
形で音声→リップシンク動画フレームを返す。MuseTalk と同じストリーミング
プロトコル ([4B big-endian length][JPEG bytes]) を採用しているので Rust
側プロキシから同じ扱いができる。

起動:
    cd ditto-talkinghead
    source .venv/bin/activate
    uv pip install fastapi uvicorn python-multipart
    uvicorn server:app --port 8881

環境変数:
    DITTO_DATA_ROOT   TRT engines のディレクトリ
    DITTO_CFG_PKL     cfg pkl のパス（デフォルト: pure TRT FP8）
    DITTO_AVATARS_DIR アバター画像の保管先
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import os
import queue as queue_mod
import shutil
import struct
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

import json
import re
import urllib.error
import urllib.request

from pydantic import BaseModel

from stream_pipeline_online import StreamSDK

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ditto_server")

REPO_ROOT = Path(__file__).parent
DATA_ROOT = os.environ.get(
    "DITTO_DATA_ROOT",
    str(REPO_ROOT / "checkpoints" / "ditto_trt_blackwell"),
)
CFG_PKL = os.environ.get(
    "DITTO_CFG_PKL",
    str(REPO_ROOT / "checkpoints" / "ditto_cfg" / "v0.4_hubert_cfg_trt_bw_pure.pkl"),
)
AVATARS_DIR = Path(os.environ.get("DITTO_AVATARS_DIR", str(REPO_ROOT / "avatars")))
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_CACHE_MAX = int(os.environ.get("DITTO_AVATAR_CACHE_MAX", "8"))
TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://localhost:8880").rstrip("/")
TYPE_AUDIO = b"\x00"
TYPE_FRAME = b"\x01"

FPS = 25
AUDIO_SR = 16000
CHUNKSIZE = (3, 5, 2)
SPLIT_LEN = int(sum(CHUNKSIZE) * 0.04 * AUDIO_SR) + 80  # 6480
AUDIO_STEP = CHUNKSIZE[1] * 640
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov")

# ─── StreamSDK subclass: divert writer to in-memory queue ─────────────────────


class _FrameCaptureSDK(StreamSDK):
    """StreamSDK that (a) diverts writer output into a queue and (b) caches
    avatar_registrar results per source, so per-sentence setup/close cycles
    don't repeat face detection/landmark/appearance extraction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_out_queue: Optional[queue_mod.Queue] = None
        self._source_info_cache: dict = {}
        self._install_registrar_cache()

    def _install_registrar_cache(self) -> None:
        real_registrar = self.avatar_registrar
        cache = self._source_info_cache

        # n_frames is intentionally ignored in the cache key: for image avatars
        # it has no effect, and for video avatars the smaller-N_d run just sees
        # the source_info from a longer previous run (mirror_index handles it).
        def cached(source_path, **kwargs):
            key = (
                source_path,
                kwargs.get("max_dim"),
                kwargs.get("crop_scale"),
                kwargs.get("crop_vx_ratio"),
                kwargs.get("crop_vy_ratio"),
                kwargs.get("crop_flag_do_rot"),
            )
            if key not in cache:
                logger.info("avatar_registrar cache miss: %s", source_path)
                cache[key] = real_registrar(source_path, **kwargs)
            return cache[key]

        self.avatar_registrar = cached  # type: ignore[assignment]

    def _writer_worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.writer_queue.get(timeout=1)
                except queue_mod.Empty:
                    continue
                if item is None:
                    break
                if self._frame_out_queue is not None:
                    self._frame_out_queue.put(item)
                self.writer_pbar.update()
        finally:
            if self._frame_out_queue is not None:
                self._frame_out_queue.put(None)


# ─── Globals ──────────────────────────────────────────────────────────────────

_sdk: Optional[_FrameCaptureSDK] = None
_sdk_lock = threading.Lock()  # serialize generate calls — workers are not reentrant
_ready = False
_avatars: "OrderedDict[str, str]" = OrderedDict()


# ─── SDK lifecycle ────────────────────────────────────────────────────────────


def _load_sdk() -> _FrameCaptureSDK:
    logger.info("loading StreamSDK: data_root=%s cfg_pkl=%s", DATA_ROOT, CFG_PKL)
    t0 = time.perf_counter()
    sdk = _FrameCaptureSDK(CFG_PKL, DATA_ROOT)
    logger.info("StreamSDK loaded in %.1fs", time.perf_counter() - t0)
    return sdk


def _warmup_if_hybrid(sdk: _FrameCaptureSDK) -> None:
    """Hybrid cfg (warp is PyTorch) must compile on the main thread to avoid
    torch.compile deadlocks when first invoked from a worker thread.
    Pure TRT cfg has a TRT engine for warp and needs no warmup.
    """
    warp_net = getattr(sdk.warp_f3d, "warp_net", None)
    if warp_net is None or not hasattr(warp_net, "model"):
        logger.info("warmup: warp is not PyTorch — skipping (pure TRT)")
        return
    if not isinstance(warp_net.model, torch.nn.Module):
        logger.info("warmup: warp.model is not nn.Module — skipping")
        return

    logger.info("warmup: torch.compile(warp) + dummy forward")
    t0 = time.perf_counter()
    warp_net.model = torch.compile(warp_net.model, dynamic=False)
    try:
        device = next(warp_net.model.parameters()).device
        dtype = next(warp_net.model.parameters()).dtype
    except StopIteration:
        device, dtype = torch.device("cuda"), torch.float32
    dummy_f3d = torch.zeros(1, 32, 16, 64, 64, device=device, dtype=dtype)
    dummy_kp = torch.zeros(1, 21, 3, device=device, dtype=dtype)
    with torch.no_grad():
        try:
            _ = sdk.warp_f3d(dummy_f3d, dummy_kp, dummy_kp)
        except Exception as e:
            logger.warning("warmup forward raised (non-fatal): %s", e)
    logger.info("warmup done in %.1fs", time.perf_counter() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _sdk, _ready
    _sdk = _load_sdk()
    try:
        _warmup_if_hybrid(_sdk)
    except Exception:
        logger.exception("warmup failed")
    _ready = True
    logger.info("ditto_server ready")
    yield
    _ready = False


app = FastAPI(lifespan=lifespan, title="Ditto-TalkingHead")


def _require_ready() -> _FrameCaptureSDK:
    if not _ready or _sdk is None:
        raise HTTPException(status_code=503, detail="Ditto SDK not ready")
    return _sdk


# ─── Avatar cache (LRU) ───────────────────────────────────────────────────────


def _cache_set(avatar_id: str, src_path: str) -> None:
    if avatar_id in _avatars:
        _avatars.move_to_end(avatar_id)
    _avatars[avatar_id] = src_path
    while len(_avatars) > AVATAR_CACHE_MAX:
        evicted_id, _ = _avatars.popitem(last=False)
        logger.info("[AvatarCache] evicted '%s'", evicted_id)


def _resolve_source(avatar_id: str) -> Optional[str]:
    if avatar_id in _avatars:
        _avatars.move_to_end(avatar_id)
        return _avatars[avatar_id]
    avatar_dir = AVATARS_DIR / avatar_id
    if not avatar_dir.is_dir():
        return None
    for suffix in IMAGE_EXTS:
        candidate = avatar_dir / f"source{suffix}"
        if candidate.exists():
            _cache_set(avatar_id, str(candidate))
            return str(candidate)
    return None


# ─── Audio chunk generator ────────────────────────────────────────────────────


def _iter_audio_chunks(audio: np.ndarray):
    """Split audio into online-mode chunks (chunksize=(3,5,2) → 6480 samples).

    Matches the streaming preroll logic in inference.py: pad the first
    chunksize[0] frames with silence so LMDM has warmup context.
    """
    prepad = np.zeros((CHUNKSIZE[0] * 640,), dtype=np.float32)
    padded = np.concatenate([prepad, audio.astype(np.float32)], axis=0)
    for i in range(0, len(padded), AUDIO_STEP):
        chunk = padded[i : i + SPLIT_LEN]
        if len(chunk) < SPLIT_LEN:
            chunk = np.pad(chunk, (0, SPLIT_LEN - len(chunk)))
        yield chunk


# ─── Core pipeline runner ─────────────────────────────────────────────────────


def _run_generate(
    avatar_id: str,
    audio: np.ndarray,
    on_frame: Callable[[np.ndarray], None],
) -> None:
    """Blocking: run the ditto pipeline and invoke on_frame(rgb_uint8) per frame.
    The lock serializes access to the single SDK instance.
    """
    sdk = _require_ready()
    src_path = _resolve_source(avatar_id)
    if src_path is None:
        raise HTTPException(404, f"Avatar '{avatar_id}' not prepared")

    with _sdk_lock:
        out_q: queue_mod.Queue = queue_mod.Queue()
        sdk._frame_out_queue = out_q
        tmp_out = tempfile.mktemp(suffix=".mp4")
        feeder_exc: list[BaseException] = []

        def feeder():
            try:
                for chunk in _iter_audio_chunks(audio):
                    sdk.run_chunk(chunk, CHUNKSIZE)
            except BaseException as e:  # noqa: BLE001
                feeder_exc.append(e)
            finally:
                try:
                    sdk.close()
                except BaseException as e:  # noqa: BLE001
                    feeder_exc.append(e)
                # belt-and-suspenders sentinel in case writer_worker crashed
                out_q.put(None)

        try:
            sdk.setup(src_path, tmp_out, online_mode=True)
            num_f = math.ceil(len(audio) / AUDIO_SR * FPS)
            sdk.setup_Nd(N_d=num_f)

            t = threading.Thread(target=feeder, daemon=True, name="ditto-feeder")
            t.start()

            while True:
                item = out_q.get()
                if item is None:
                    break
                on_frame(item)

            t.join(timeout=120)
            if feeder_exc:
                raise feeder_exc[0]
        finally:
            sdk._frame_out_queue = None
            # cleanup empty tmp mp4 (writer opened it but we diverted frames)
            for p in (tmp_out, tmp_out + ".tmp.mp4"):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


def _load_audio_16k(content: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(content))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SR)
    return audio


def _encode_jpeg(rgb: np.ndarray, quality: int) -> Optional[bytes]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ready" if _ready else "loading",
        "data_root": DATA_ROOT,
        "cfg_pkl": CFG_PKL,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/ditto/prepare")
async def ditto_prepare(file: UploadFile = File(...), avatar_id: str = Form(...)):
    content = await file.read()
    suffix = (Path(file.filename or "").suffix or ".png").lower()
    if suffix not in IMAGE_EXTS:
        raise HTTPException(400, f"unsupported extension: {suffix}")

    avatar_dir = AVATARS_DIR / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)
    # clear previous source files so we don't mix extensions
    for p in avatar_dir.glob("source.*"):
        p.unlink(missing_ok=True)
    src_path = avatar_dir / f"source{suffix}"
    src_path.write_bytes(content)

    _cache_set(avatar_id, str(src_path))
    return {
        "avatar_id": avatar_id,
        "source_path": str(src_path),
        "size_bytes": len(content),
        "kind": "video" if suffix in (".mp4", ".mov") else "image",
    }


@app.get("/ditto/avatars")
def ditto_list_avatars():
    out = []
    if not AVATARS_DIR.exists():
        return out
    for d in sorted(AVATARS_DIR.iterdir()):
        if not d.is_dir():
            continue
        for suffix in IMAGE_EXTS:
            if (d / f"source{suffix}").exists():
                out.append({"id": d.name, "source_ext": suffix})
                break
    return out


@app.delete("/ditto/avatars/{avatar_id}")
def ditto_delete_avatar(avatar_id: str):
    removed_cache = _avatars.pop(avatar_id, None) is not None
    avatar_dir = AVATARS_DIR / avatar_id
    removed_disk = False
    if avatar_dir.exists():
        shutil.rmtree(avatar_dir)
        removed_disk = True
    return {
        "avatar_id": avatar_id,
        "removed_cache": removed_cache,
        "removed_disk": removed_disk,
    }


@app.post("/ditto/generate")
async def ditto_generate(
    file: UploadFile = File(...),
    avatar_id: str = Form(...),
    fps: int = Form(FPS),  # noqa: ARG001 — ditto is fixed 25fps, accepted for symmetry
    jpeg_quality: int = Form(85),
):
    _require_ready()
    audio = _load_audio_16k(await file.read())

    def _collect() -> tuple[list[str], float]:
        frames: list[str] = []
        t0 = time.perf_counter()

        def on_frame(rgb: np.ndarray):
            payload = _encode_jpeg(rgb, jpeg_quality)
            if payload is not None:
                frames.append(base64.b64encode(payload).decode())

        _run_generate(avatar_id, audio, on_frame)
        return frames, (time.perf_counter() - t0) * 1000

    frames, elapsed_ms = await asyncio.to_thread(_collect)
    n = len(frames)
    return {
        "avatar_id": avatar_id,
        "fps": FPS,
        "num_frames": n,
        "total_ms": int(elapsed_ms),
        "ms_per_frame": round(elapsed_ms / max(n, 1), 1),
        "frames": frames,
    }


@app.post("/ditto/generate/stream")
async def ditto_generate_stream(
    file: UploadFile = File(...),
    avatar_id: str = Form(...),
    fps: int = Form(FPS),  # noqa: ARG001
    jpeg_quality: int = Form(85),
):
    _require_ready()
    audio = _load_audio_16k(await file.read())
    loop = asyncio.get_running_loop()
    chunk_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    def worker():
        def on_frame(rgb: np.ndarray):
            payload = _encode_jpeg(rgb, jpeg_quality)
            if payload is None:
                return
            framed = struct.pack(">I", len(payload)) + payload
            asyncio.run_coroutine_threadsafe(chunk_q.put(framed), loop)

        try:
            _run_generate(avatar_id, audio, on_frame)
        except BaseException:  # noqa: BLE001
            logger.exception("ditto generate failed")
        finally:
            asyncio.run_coroutine_threadsafe(chunk_q.put(None), loop)

    threading.Thread(target=worker, daemon=True, name="ditto-stream-worker").start()

    async def stream():
        while True:
            item = await chunk_q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        stream(),
        media_type="application/octet-stream",
        headers={
            "X-Ditto-FPS": str(FPS),
            "Access-Control-Expose-Headers": "X-Ditto-FPS",
        },
    )


@app.get("/ditto/benchmark")
async def ditto_benchmark(avatar_id: str = "default", duration_s: float = 2.0):
    _require_ready()
    audio = np.zeros(int(AUDIO_SR * duration_s), dtype=np.float32)

    def _count() -> tuple[int, float]:
        n = 0
        t0 = time.perf_counter()

        def on_frame(_rgb: np.ndarray):
            nonlocal n
            n += 1

        _run_generate(avatar_id, audio, on_frame)
        return n, time.perf_counter() - t0

    n, elapsed = await asyncio.to_thread(_count)
    realtime = n / FPS if n else 0
    ratio = elapsed / max(realtime, 1e-3)
    return {
        "num_frames": n,
        "fps": FPS,
        "total_ms": int(elapsed * 1000),
        "ms_per_frame": round(elapsed * 1000 / max(n, 1), 1),
        "realtime_ratio": round(ratio, 2),
        "is_realtime": ratio < 1.0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ─── Combined TTS + Ditto streaming ───────────────────────────────────────────
#
# Protocol (matches hage's MuseTalk /tts/stream/avatar):
#   [1 byte type: 0=audio WAV, 1=video JPEG frame]
#   [4 bytes big-endian length]
#   [payload bytes]
#
# Per sentence: the upstream tts_server yields one audio chunk (buffered per
# sentence). We forward it to the client, then feed the same audio (resampled
# to 16 kHz) into the ditto online pipeline via run_chunk. Frames produced by
# ditto are interleaved into the output stream as soon as they are ready — a
# separate frame-pump thread drains the SDK's out-queue so audio-of-next-
# sentence and frames-of-current-sentence can overlap.


class DittoTtsAvatarRequest(BaseModel):
    text: str
    avatar_id: str
    tts_backend: Optional[str] = "kokoro"  # kokoro | voicevox | xtts | sbv2 | irodori
    voice: Optional[str] = None
    speed: Optional[float] = 1.0
    lang: Optional[str] = "j"
    caption: Optional[str] = None  # irodori
    speaker: Optional[int] = None  # voicevox
    model_id: Optional[int] = None  # sbv2
    speaker_id: Optional[int] = None  # sbv2
    style: Optional[str] = None  # sbv2
    language: Optional[str] = None  # xtts
    num_steps: Optional[int] = None  # irodori
    fps: Optional[int] = FPS
    jpeg_quality: Optional[int] = 85


def _build_tts_request(req: DittoTtsAvatarRequest) -> tuple[str, dict, int]:
    """Return (url, json_payload, upstream_sample_rate_hint) for the chosen TTS backend.

    The sample rate hint is only used for logging; actual rate is parsed from
    the WAV header per chunk. tts_server always prepends a WAV header so
    soundfile can recover the true rate regardless.
    """
    backend = (req.tts_backend or "kokoro").lower()
    if backend == "kokoro":
        return (
            f"{TTS_SERVER_URL}/tts/stream",
            {
                "text": req.text,
                "voice": req.voice or "jf_alpha",
                "speed": req.speed or 1.0,
                "lang": req.lang or "j",
            },
            24000,
        )
    if backend == "voicevox":
        return (
            f"{TTS_SERVER_URL}/voicevox/tts/stream",
            {
                "text": req.text,
                "speaker": int(req.speaker if req.speaker is not None else (req.voice or 3)),
                "speed": req.speed or 1.0,
            },
            24000,
        )
    if backend == "xtts":
        return (
            f"{TTS_SERVER_URL}/xtts/tts/stream",
            {
                "text": req.text,
                "voice": req.voice,
                "language": req.language or "ja",
                "speed": req.speed or 1.0,
            },
            24000,
        )
    if backend == "sbv2":
        return (
            f"{TTS_SERVER_URL}/sbv2/tts/stream",
            {
                "text": req.text,
                "model_id": req.model_id or 0,
                "speaker_id": req.speaker_id or 0,
                "style": req.style or "Neutral",
                "style_weight": 5.0,
                "speed": req.speed or 1.0,
                "split": False,
            },
            44100,
        )
    if backend == "irodori":
        return (
            f"{TTS_SERVER_URL}/irodori/tts/stream",
            {
                "text": req.text,
                "caption": req.caption,
                "speed": req.speed or 1.0,
                "num_steps": req.num_steps or 20,
            },
            48000,
        )
    raise HTTPException(400, f"unknown tts_backend: {backend}")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。．！？!?])\s*")


def _split_sentences_for_xtts(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _stream_xtts_per_sentence(req: DittoTtsAvatarRequest):
    """XTTS は inference_stream() が GPT トークン単位で多数のチャンクを返すため、
    そのまま ditto に流すとサブチャンク境界ごとに offline パイプラインが回って
    再生がカクつく。ここで文単位に分割して各文を別リクエストで投げ、返ってきた
    チャンク群を 1 本の WAV に結合してから yield する。
    """
    sentences = _split_sentences_for_xtts(req.text)
    if not sentences:
        return
    for sentence in sentences:
        payload = {
            "text": sentence,
            "voice": req.voice,
            "language": req.language or "ja",
            "speed": req.speed or 1.0,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{TTS_SERVER_URL}/xtts/tts/stream",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise HTTPException(502, f"tts_server xtts HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise HTTPException(502, f"tts_server unreachable: {e}") from e

        audio_parts: list[np.ndarray] = []
        sample_rate: Optional[int] = None
        with resp:
            buf = bytearray()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                while len(buf) >= 4:
                    length = int.from_bytes(bytes(buf[:4]), "big")
                    if len(buf) < 4 + length:
                        break
                    wav = bytes(buf[4 : 4 + length])
                    del buf[: 4 + length]
                    audio, sr = sf.read(io.BytesIO(wav))
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    audio_parts.append(audio.astype(np.float32))
                    sample_rate = sr
        if not audio_parts or sample_rate is None:
            continue
        merged = np.concatenate(audio_parts)
        out = io.BytesIO()
        sf.write(out, merged, sample_rate, format="WAV", subtype="PCM_16")
        yield out.getvalue()


def _stream_tts_wavs(req: DittoTtsAvatarRequest):
    """Yield (wav_bytes) per sentence from tts_server. Blocking generator."""
    backend = (req.tts_backend or "kokoro").lower()
    if backend == "xtts":
        yield from _stream_xtts_per_sentence(req)
        return

    url, payload, _sr_hint = _build_tts_request(req)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HTTPException(502, f"tts_server {req.tts_backend} HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise HTTPException(502, f"tts_server unreachable: {e}") from e

    with resp:
        buf = bytearray()
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while len(buf) >= 4:
                length = int.from_bytes(bytes(buf[:4]), "big")
                if len(buf) < 4 + length:
                    break
                wav = bytes(buf[4 : 4 + length])
                del buf[: 4 + length]
                yield wav


def _wav_bytes_to_16k(wav_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(wav_bytes))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SR)
    return audio


def _run_sentence_offline(
    sdk: _FrameCaptureSDK,
    src_path: str,
    audio16k: np.ndarray,
    jpeg_quality: int,
    push_frame: Callable[[bytes], None],
) -> int:
    """Run one ditto offline pipeline over the given 16kHz audio and push each
    produced JPEG to the client via push_frame. Returns frame count.

    Offline (not online_mode) gives us a fixed N_d frames = ceil(duration*fps),
    which means per-sentence audio:frames is 1:1 — matching MuseTalk behavior
    so the frontend's "buffer audio until first frame" logic syncs correctly.
    """
    num_f = max(1, math.ceil(len(audio16k) / AUDIO_SR * FPS))
    frame_q: queue_mod.Queue = queue_mod.Queue()
    sdk._frame_out_queue = frame_q
    tmp_mp4 = tempfile.mktemp(suffix=".mp4")

    try:
        sdk.setup(src_path, tmp_mp4, online_mode=False)
        sdk.setup_Nd(N_d=num_f)

        aud_feat = sdk.wav2feat.wav2feat(audio16k)
        sdk.audio2motion_queue.put(aud_feat)

        # close() is blocking (joins all worker threads). Run it in background
        # so we can drain frames from frame_q as they are produced, instead of
        # waiting for the whole pipeline to complete.
        closer_exc: list[BaseException] = []

        def closer():
            try:
                sdk.close()
            except BaseException as e:  # noqa: BLE001
                closer_exc.append(e)

        closer_t = threading.Thread(target=closer, daemon=True, name="ditto-closer")
        closer_t.start()

        count = 0
        while True:
            item = frame_q.get()
            if item is None:
                break
            payload = _encode_jpeg(item, jpeg_quality)
            if payload is None:
                continue
            push_frame(TYPE_FRAME + struct.pack(">I", len(payload)) + payload)
            count += 1

        closer_t.join(timeout=30)
        if closer_exc:
            raise closer_exc[0]
        return count
    finally:
        sdk._frame_out_queue = None
        for p in (tmp_mp4, tmp_mp4 + ".tmp.mp4"):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


@app.post("/ditto/tts/stream/avatar")
async def ditto_tts_stream_avatar(req: DittoTtsAvatarRequest):
    """TTS + Ditto 結合ストリーミング。MuseTalk の同名エンドポイントと同一プロトコル。

    Per-sentence offline pipeline: each TTS sentence runs a complete ditto
    setup→feed→close cycle. Frames are forwarded to the client as they are
    produced. avatar_registrar is cached in the SDK so only the first sentence
    pays for face registration; subsequent sentences setup is fast.
    """
    sdk = _require_ready()
    src_path = _resolve_source(req.avatar_id)
    if src_path is None:
        raise HTTPException(404, f"Avatar '{req.avatar_id}' not prepared")

    jpeg_quality = req.jpeg_quality or 85
    loop = asyncio.get_running_loop()
    output_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    def push(payload: bytes) -> None:
        asyncio.run_coroutine_threadsafe(output_q.put(payload), loop)

    def push_sentinel() -> None:
        asyncio.run_coroutine_threadsafe(output_q.put(None), loop)

    def worker():
        try:
            with _sdk_lock:
                t_start = time.perf_counter()
                total_audio_ms = 0
                total_frames = 0
                sentence_idx = 0

                for wav_bytes in _stream_tts_wavs(req):
                    push(TYPE_AUDIO + struct.pack(">I", len(wav_bytes)) + wav_bytes)

                    audio16k = _wav_bytes_to_16k(wav_bytes)
                    audio_ms = int(len(audio16k) / AUDIO_SR * 1000)
                    total_audio_ms += audio_ms

                    t_sentence = time.perf_counter()
                    n = _run_sentence_offline(sdk, src_path, audio16k, jpeg_quality, push)
                    total_frames += n
                    sentence_idx += 1
                    logger.info(
                        "combined: sentence#%d audio=%dms frames=%d render=%.0fms elapsed=%.0fms",
                        sentence_idx,
                        audio_ms,
                        n,
                        (time.perf_counter() - t_sentence) * 1000,
                        (time.perf_counter() - t_start) * 1000,
                    )

                logger.info(
                    "combined: done sentences=%d audio_total=%dms frames=%d elapsed=%.0fms",
                    sentence_idx,
                    total_audio_ms,
                    total_frames,
                    (time.perf_counter() - t_start) * 1000,
                )
        except BaseException as e:  # noqa: BLE001
            logger.exception("ditto+tts combined error: %s", e)
        finally:
            push_sentinel()

    threading.Thread(target=worker, daemon=True, name="ditto-combined-worker").start()

    async def stream():
        while True:
            item = await output_q.get()
            if item is None:
                break
            yield item

    # sample-rate header: per-backend hint only; each WAV chunk carries its own
    # true rate in the header.
    _, _, sr_hint = _build_tts_request(req)
    return StreamingResponse(
        stream(),
        media_type="application/octet-stream",
        headers={
            "X-Ditto-FPS": str(FPS),
            "X-Sample-Rate": str(sr_hint),
            "Access-Control-Expose-Headers": "X-Ditto-FPS, X-Sample-Rate",
        },
    )


@app.post("/ditto/load")
def ditto_load():
    sdk = _require_ready()
    return {
        "status": "loaded",
        "data_root": DATA_ROOT,
        "cfg_pkl": CFG_PKL,
        "sdk_type": type(sdk).__name__,
    }
