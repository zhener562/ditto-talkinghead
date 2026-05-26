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

# Pin to GPU 1 (4000 BW, voice sanctuary). Override with CUDA_VISIBLE_DEVICES=0.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec uvicorn server:app --host "$HOST" --port "$PORT" --log-level info
