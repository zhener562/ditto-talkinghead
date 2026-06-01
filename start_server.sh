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
HOST="${DITTO_HOST:-0.0.0.0}"

# Pin to GPU 1 (4000 BW, voice sanctuary)。2026-06-01 に GPU1 へ復帰。
# 経緯: GPU1 では Ditto generate が CUDA 720 (TRT Myelin __myl_Inor kernel launch 失敗 →
# 顔が黒くなる) になっていた。真因は VRAM でも vLLM 競合でもなく (GPU1 を 19.8GB 空けて Ditto
# 単独でも 720 が出た)、checkpoints/ditto_trt_blackwell/ の TRT エンジンが 2026-05-03 に
# 6000 BW で焼かれた GPU 固有バイナリだったこと。4000 BW (SM 構成が違う) では Myelin fusion
# カーネルが起動できない。scripts/build_trt10_blackwell_4000.py + build_warp_4000.py
# (warp は TRT10 ABI の GridSample3D plugin 経由: grid-sample3d-trt-plugin/build_trt10/) で
# 4000 BW 用エンジンを checkpoints/ditto_trt_4000bw/ に再ビルドして解決。これで GPU0 の
# HiDream/LTX 重バーストとの SM 競合 (ノイジーネイバー → アバターのカクつき) を物理分離で根絶。
# GPU0 (6000 BW, 17 vs 32 ms/frame と高速) に逃がすなら CUDA_VISIBLE_DEVICES=0。
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# TRT エンジンは GPU 固有。GPU1→ditto_trt_4000bw / GPU0→ditto_trt_blackwell(6000) を出し分ける。
# DITTO_DATA_ROOT が明示指定されていればそれを尊重 (再ビルド検証等)。
if [ -z "${DITTO_DATA_ROOT:-}" ]; then
  if [ "$CUDA_VISIBLE_DEVICES" = "1" ]; then
    export DITTO_DATA_ROOT="$(pwd)/checkpoints/ditto_trt_4000bw"
  else
    export DITTO_DATA_ROOT="$(pwd)/checkpoints/ditto_trt_blackwell"
  fi
fi

exec uvicorn server:app --host "$HOST" --port "$PORT" --log-level info
