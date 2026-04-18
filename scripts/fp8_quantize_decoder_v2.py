"""Decoder FP8 v2: exclude output conv, use full 640 samples."""
import numpy as np, pathlib
from modelopt.onnx.quantization import quantize

ROOT = pathlib.Path(__file__).resolve().parent.parent
data = np.load(str(ROOT / "calib/tensors/decoder_calib.npz"))["feature"].astype(np.float32)
print(f"decoder calib data: {data.shape}")

quantize(
    onnx_path=str(ROOT / "checkpoints/ditto_onnx/decoder.onnx"),
    quantize_mode="fp8",
    calibration_data={"feature": data},
    calibration_method="max",
    # Exclude final output conv + last normalization layer (typically sensitive)
    nodes_to_exclude=[
        "/conv_img/conv_img.0/Conv",
        "/up_1/conv_1/Conv",
        "/up_1/norm_1/mlp_shared/mlp_shared.0/Conv",
        "/up_1/norm_1/mlp_gamma/Conv",
        "/up_1/norm_1/mlp_beta/Conv",
    ],
    output_path=str(ROOT / "checkpoints/ditto_onnx/decoder_fp8_qdq_v2.onnx"),
    high_precision_dtype="fp16",
    log_level="INFO",
)
print("Done.")
