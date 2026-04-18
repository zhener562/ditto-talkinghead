"""Decoder FP8 v3: entropy calibration, exclude all SPADE modulation layers."""
import numpy as np, pathlib
from modelopt.onnx.quantization import quantize
import onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Find all SPADE-related layers to exclude
m = onnx.load(str(ROOT / "checkpoints/ditto_onnx/decoder.onnx"))
exclude = []
for n in m.graph.node:
    name = n.name.lower()
    # SPADE modulation is very precision-sensitive
    if any(k in name for k in ["norm_s", "mlp_shared", "mlp_gamma", "mlp_beta", "conv_img"]):
        exclude.append(n.name)
print(f"Excluding {len(exclude)} sensitive nodes: {exclude[:5]}...")

data = np.load(str(ROOT / "calib/tensors/decoder_calib.npz"))["feature"].astype(np.float32)
# Use 256 samples (entropy can be slower)
idxs = np.linspace(0, data.shape[0]-1, 256).astype(int)
data = data[idxs]

quantize(
    onnx_path=str(ROOT / "checkpoints/ditto_onnx/decoder.onnx"),
    quantize_mode="fp8",
    calibration_data={"feature": data},
    calibration_method="entropy",  # switch from max → entropy
    nodes_to_exclude=exclude,
    output_path=str(ROOT / "checkpoints/ditto_onnx/decoder_fp8_qdq_v3.onnx"),
    high_precision_dtype="fp16",
    log_level="INFO",
)
print("Done.")
