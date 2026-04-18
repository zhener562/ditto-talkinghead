"""Polygraphy data loaders for FP8 calibration."""
import numpy as np, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

def load_data_decoder():
    data = np.load(str(ROOT / "calib/tensors/decoder_calib.npz"))
    feat = data["feature"].astype(np.float32)
    for i in range(feat.shape[0]):
        yield {"feature": feat[i:i+1]}  # (1, 256, 64, 64)

def load_data_warp():
    data = np.load(str(ROOT / "calib/tensors/warp_calib.npz"))
    f3d = data["feature_3d"].astype(np.float32)
    kps = data["kp_source"].astype(np.float32)
    kpd = data["kp_driving"].astype(np.float32)
    for i in range(f3d.shape[0]):
        yield {
            "feature_3d": f3d[i:i+1],
            "kp_source": kps[i:i+1],
            "kp_driving": kpd[i:i+1],
        }
