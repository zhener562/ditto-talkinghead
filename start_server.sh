#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "error: .venv not found. Set up ditto environment first (see BLACKWELL_NOTES.md)."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Install server deps if missing. Everything else (torch/trt/librosa/cv2) should
# already be in .venv from the inference setup.
python - <<'PY' >/dev/null 2>&1 || uv pip install fastapi uvicorn python-multipart
import fastapi, uvicorn, multipart  # noqa: F401
PY

PORT="${DITTO_PORT:-8881}"
HOST="${DITTO_HOST:-127.0.0.1}"  # LAN に晒さない。外から叩く必要があれば DITTO_HOST=0.0.0.0

# Pin to GPU 0 (音声聖域)。重バッチ (LLM / HiDream) は GPU1 に隔離してあるので
# SM 競合は起きない。2026-08-31 に GPU1 から移した: GPU1 の PCIe が x2 で
# 実測 3.0 GB/s しかなく、毎フレーム 8.9MB を CPU へ返す Ditto が最も損をするため。
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# TRT エンジンは **GPU アーキ固有のバイナリ**。Myelin fusion カーネルが SM 構成に
# 焼き込まれるので、違うカードで起動すると CUDA 720 (__myl_Inor kernel launch 失敗
# → 顔が黒くなる) になる。2026-06-01 に 4000 BW 用を別途ビルドしたのはこのため。
#
# **カード番号ではなく実際の型番で選ぶ (2026-08-31)**。それまでは「GPU1 = 4000 BW」
# という番号ベースの分岐だったが、GPU1 が 6000 BW に換装されて前提ごと崩れた。
# 壊れ方は無音で、4000bw エンジンは 6000 BW でも "動いてしまう" (黒フレーム 0) が
# 26.0 vs 19.7 ms/frame で 25% 遅い。番号で判定する限り換装のたびに再発するので、
# 判定材料をカード自身にした。
#
# DITTO_DATA_ROOT が明示指定されていればそれを尊重 (再ビルド検証等)。
if [ -z "${DITTO_DATA_ROOT:-}" ]; then
  _ditto_gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader \
                       -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null || true)"
  case "$_ditto_gpu_name" in
    *"RTX PRO 4000"*) _ditto_engines="ditto_trt_4000bw" ;;
    *"RTX PRO 6000"*) _ditto_engines="ditto_trt_blackwell" ;;
    *)
      # 未知のカード。engine が合わなければ黒フレームになるので、黙って
      # 進めずに気付けるようにしておく。
      echo "[start_server] warn: 未知の GPU '${_ditto_gpu_name:-?}' — blackwell エンジンで起動する" >&2
      _ditto_engines="ditto_trt_blackwell"
      ;;
  esac
  export DITTO_DATA_ROOT="$(pwd)/checkpoints/$_ditto_engines"
  echo "[start_server] gpu='${_ditto_gpu_name:-?}' engines=$_ditto_engines"
fi

exec uvicorn server:app --host "$HOST" --port "$PORT" --log-level info
