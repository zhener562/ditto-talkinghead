"""Insert FP8 Q/DQ nodes into decoder.onnx using modelopt + our calibration tensors."""
import numpy as np, pathlib
from modelopt.onnx.quantization import quantize

ROOT = pathlib.Path(__file__).resolve().parent.parent
data = np.load(str(ROOT / "calib/tensors/decoder_calib.npz"))["feature"].astype(np.float32)
print(f"decoder calib data: {data.shape} {data.dtype} range=[{data.min():.3f}, {data.max():.3f}]")

# Subsample to keep calibration fast (256 samples is plenty)
N = 256
if data.shape[0] > N:
    idxs = np.linspace(0, data.shape[0]-1, N).astype(int)
    data = data[idxs]
print(f"Using {data.shape[0]} samples for calibration")

quantize(
    onnx_path=str(ROOT / "checkpoints/ditto_onnx/decoder.onnx"),
    quantize_mode="fp8",
    calibration_data={"feature": data},
    calibration_method="max",  # FP8 typically uses max (range-based) over entropy
    output_path=str(ROOT / "checkpoints/ditto_onnx/decoder_fp8_qdq.onnx"),
    high_precision_dtype="fp16",
    log_level="INFO",
)
print("Done. Wrote decoder_fp8_qdq.onnx")
