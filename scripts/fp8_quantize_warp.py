"""Insert FP8 Q/DQ nodes into warp_network.onnx (with GridSample3D custom plugin)."""
import numpy as np, pathlib
from modelopt.onnx.quantization import quantize

ROOT = pathlib.Path(__file__).resolve().parent.parent
data = np.load(str(ROOT / "calib/tensors/warp_calib.npz"))
f3d = data["feature_3d"].astype(np.float32)
kps = data["kp_source"].astype(np.float32)
kpd = data["kp_driving"].astype(np.float32)
print(f"warp calib: f3d={f3d.shape} kp_s={kps.shape} kp_d={kpd.shape}")

N = 256
if f3d.shape[0] > N:
    idxs = np.linspace(0, f3d.shape[0]-1, N).astype(int)
    f3d, kps, kpd = f3d[idxs], kps[idxs], kpd[idxs]

PLUGIN = "/home/zhener/ドキュメント/github/grid-sample3d-trt-plugin/build_trt10/libgrid_sample_3d_plugin.so"

quantize(
    onnx_path=str(ROOT / "checkpoints/ditto_onnx/warp_network.onnx"),
    quantize_mode="fp8",
    calibration_data={"feature_3d": f3d, "kp_source": kps, "kp_driving": kpd},
    calibration_method="max",
    calibration_eps=["trt"],  # TRT EP handles custom plugins
    output_path=str(ROOT / "checkpoints/ditto_onnx/warp_network_fp8_qdq.onnx"),
    high_precision_dtype="fp16",
    trt_plugins=[PLUGIN],
    log_level="INFO",
)
print("Done.")
