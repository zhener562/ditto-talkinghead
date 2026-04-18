"""Build TRT 10 engines for Blackwell (sm_120). Uses warp_network_ori (standard GridSample, no custom plugin)."""
import os, subprocess, sys, time

ONNX_DIR = "./checkpoints/ditto_onnx"
TRT_DIR = "./checkpoints/ditto_trt_blackwell"
os.makedirs(TRT_DIR, exist_ok=True)

FP32_MODELS = {"motion_extractor", "hubert", "wavlm"}  # also lmdm* via prefix

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
        print(f"[skip] {name}")
        return
    cmd = ["polygraphy", "convert", onnx_path, "-o", trt_path, "--builder-optimization-level=5"]
    if fp16:
        cmd.append("--fp16")
    cmd += EXTRA_SHAPES.get(name, [])
    print(f"[build] {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(f"[FAIL] {name} in {dt:.1f}s (rc={r.returncode})")
        return False
    print(f"[ok] {name} in {dt:.1f}s")
    return True

targets = [
    # (engine_name, onnx_basename)
    ("appearance_extractor", None),
    ("motion_extractor", None),
    ("stitch_network", None),
    ("decoder", None),
    ("warp_network", "warp_network_ori"),  # Use standard GridSample version
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

print("\nDone. Engines in:", TRT_DIR)
