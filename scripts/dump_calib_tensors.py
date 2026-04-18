"""Run pipeline over (image, audio) pairs and dump decoder/warp inputs as .npz for FP8 calibration."""
import os, sys, math, glob, random, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, librosa, torch
from stream_pipeline_offline import StreamSDK
from core.models import warp_network as wn_mod, decoder as dec_mod

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "calib" / "images"
AUD_DIR = ROOT / "calib" / "audios"
OUT_DIR = ROOT / "calib" / "tensors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tuning
N_SAMPLES_PER_RUN = 20     # subsample frames per run
MAX_AUDIOS_PER_IMG = 2     # pair each image with this many audios
MAX_FRAMES_PER_RUN = 300   # cap to keep runtime manageable

# Collect buffers per run
warp_inputs = []   # list of dicts {feature_3d, kp_source, kp_driving}
decode_inputs = [] # list of feature arrays

# Wrap WarpNetwork / Decoder to record inputs
_orig_warp = wn_mod.WarpNetwork.__call__
def warp_tap(self, feature_3d, kp_source, kp_driving):
    warp_inputs.append((feature_3d.astype(np.float16), kp_source.astype(np.float16), kp_driving.astype(np.float16)))
    return _orig_warp(self, feature_3d, kp_source, kp_driving)
wn_mod.WarpNetwork.__call__ = warp_tap

_orig_dec = dec_mod.Decoder.__call__
def dec_tap(self, feature):
    decode_inputs.append(feature.astype(np.float16))
    return _orig_dec(self, feature)
dec_mod.Decoder.__call__ = dec_tap

def run_one(sdk, image_path, audio_path, max_frames=MAX_FRAMES_PER_RUN):
    warp_inputs.clear(); decode_inputs.clear()
    audio, _ = librosa.core.load(audio_path, sr=16000)
    num_f = min(max_frames, math.ceil(len(audio) / 16000 * 25))
    tmp_out = str(ROOT / "tmp" / f"calib_{os.getpid()}.mp4")
    sdk.setup(str(image_path), tmp_out)
    sdk.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
    aud_feat = sdk.wav2feat.wav2feat(audio)
    sdk.audio2motion_queue.put(aud_feat)
    sdk.close()
    return list(warp_inputs), list(decode_inputs)

def subsample(lst, n):
    if len(lst) <= n: return lst
    idxs = np.linspace(0, len(lst)-1, n).astype(int)
    return [lst[i] for i in idxs]

def main():
    random.seed(42)
    images = sorted(IMG_DIR.glob("*.png"))
    audios = sorted(AUD_DIR.glob("*.wav"))
    print(f"Images: {len(images)}, Audios: {len(audios)}")

    # Pair each image with MAX_AUDIOS_PER_IMG random audios
    pairs = []
    for img in images:
        chosen = random.sample(audios, min(MAX_AUDIOS_PER_IMG, len(audios)))
        for aud in chosen:
            pairs.append((img, aud))
    print(f"Total runs: {len(pairs)}")

    # Build SDK once, reuse across runs
    sdk = StreamSDK(str(ROOT / "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_bw_pure.pkl"),
                    str(ROOT / "checkpoints/ditto_trt_blackwell"))

    all_warp_feat3d, all_warp_kps, all_warp_kpd = [], [], []
    all_dec_feat = []
    t_all = time.time()
    for i, (img, aud) in enumerate(pairs):
        try:
            t0 = time.time()
            w, d = run_one(sdk, img, aud)
            w = subsample(w, N_SAMPLES_PER_RUN)
            d = subsample(d, N_SAMPLES_PER_RUN)
            for f3d, kps, kpd in w:
                all_warp_feat3d.append(f3d[0]); all_warp_kps.append(kps[0]); all_warp_kpd.append(kpd[0])
            for f in d:
                all_dec_feat.append(f[0])
            print(f"[{i+1:3d}/{len(pairs)}] {img.name} × {aud.name}: warp={len(w)}, dec={len(d)} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[{i+1:3d}/{len(pairs)}] FAIL {img.name} × {aud.name}: {e}", flush=True)

    wa = np.stack(all_warp_feat3d)
    ws = np.stack(all_warp_kps)
    wd = np.stack(all_warp_kpd)
    df = np.stack(all_dec_feat)
    np.savez(OUT_DIR / "warp_calib.npz", feature_3d=wa, kp_source=ws, kp_driving=wd)
    np.savez(OUT_DIR / "decoder_calib.npz", feature=df)
    print(f"\nDone in {time.time()-t_all:.1f}s")
    print(f"  warp_calib.npz: feature_3d={wa.shape} kp_source={ws.shape} kp_driving={wd.shape}")
    print(f"  decoder_calib.npz: feature={df.shape}")
    print(f"  Sizes: warp={(OUT_DIR/'warp_calib.npz').stat().st_size/1e6:.1f} MB, decoder={(OUT_DIR/'decoder_calib.npz').stat().st_size/1e6:.1f} MB")

if __name__ == "__main__":
    main()
