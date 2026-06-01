"""Build TRT 10 engines for the RTX PRO 4000 Blackwell (GPU1).

build_trt10_blackwell.py のクローン。TRT エンジンはビルドした GPU 固有 (Myelin fusion が
SM 構成を焼き込む) なので、6000 BW で焼いた ./checkpoints/ditto_trt_blackwell/ は 4000 BW で
CUDA 720 になる。これは **CUDA_VISIBLE_DEVICES=1 で実行** して 4000 BW 用エンジンを別 dir に焼く。

6000 用 (ditto_trt_blackwell) は温存して GPU0 fallback に残す。
ライブ音声 (tts_server 8880 / CustomVoice 8890) と GPU1 を共有するので、builder の
workspace を 4GB に cap して OOM を防ぐ。warp は plugin 不要の warp_network_ori を使用。
"""
import os, subprocess, time

ONNX_DIR = "./checkpoints/ditto_onnx"
TRT_DIR = "./checkpoints/ditto_trt_4000bw"
os.makedirs(TRT_DIR, exist_ok=True)

POOL_LIMIT = ["--pool-limit", "workspace:4294967296"]  # 4GB: ライブ音声を OOM させない

FP32_MODELS = {"motion_extractor", "hubert", "wavlm"}

EXTRA_SHAPES = {
    "wavlm": ["--trt-min-shapes", "audio:[1,1000]", "--trt-max-shapes", "audio:[1,320080]", "--trt-opt-shapes", "audio:[1,320080]"],
    "hubert": ["--trt-min-shapes", "input_values:[1,3240]", "--trt-max-shapes", "input_values:[1,12960]", "--trt-opt-shapes", "input_values:[1,6480]"],
}

def convert(name, onnx_name=None):
    onnx_name = onnx_name or name
    onnx_path = f"{ONNX_DIR}/{onnx_name}.onnx"
    fp16 = not (name in FP32_MODELS or name.startswith("lmdm"))
    trt_path = f"{TRT_DIR}/{name}_fp{'16' if fp16 else '32'}.engine"
    if os.path.isfile(trt_path):
        print(f"[skip] {name}", flush=True)
        return
    cmd = ["polygraphy", "convert", onnx_path, "-o", trt_path, "--builder-optimization-level=5"]
    if fp16:
        cmd.append("--fp16")
    cmd += EXTRA_SHAPES.get(name, [])
    cmd += POOL_LIMIT
    print(f"[build] {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:], flush=True)
        print(f"[FAIL] {name} in {dt:.1f}s (rc={r.returncode})", flush=True)
        return False
    print(f"[ok] {name} in {dt:.1f}s", flush=True)
    return True

targets = [
    ("appearance_extractor", None),
    ("motion_extractor", None),
    ("stitch_network", None),
    ("decoder", None),
    ("warp_network", "warp_network_ori"),
    ("lmdm_v0.4_hubert", None),
    ("hubert", None),
    ("insightface_det", None),
    ("blaze_face", None),
    ("face_mesh", None),
    ("landmark106", None),
    ("landmark203", None),
]

for name, onnx_name in targets:
    convert(name, onnx_name)

print("\nDone. Engines in:", TRT_DIR, flush=True)
