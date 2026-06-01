"""4000 BW 用 warp_network エンジンを GridSample3D plugin 経由で焼く。

標準 polygraphy では warp の 3D GridSample (5D 入力) が
`addGridSample: nbDims == 4` で弾かれるため、cvt_onnx_to_trt.onnx_to_trt_for_gridsample
(libgrid_sample_3d_plugin.so をロードする経路) を使う。CUDA_VISIBLE_DEVICES=1 で実行。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from cvt_onnx_to_trt import onnx_to_trt_for_gridsample

ONNX = "./checkpoints/ditto_onnx/warp_network.onnx"          # 3D gridsample 版 (plugin 用)
PLUGIN = "./checkpoints/ditto_onnx/libgrid_sample_3d_plugin.so"
OUT = "./checkpoints/ditto_trt_4000bw/warp_network_fp16.engine"

assert os.path.isfile(ONNX), ONNX
assert os.path.isfile(PLUGIN), PLUGIN
print(f"[warp] building {OUT} via plugin {PLUGIN}", flush=True)
onnx_to_trt_for_gridsample(ONNX, OUT, fp16=True, plugin_file=PLUGIN)
print(f"[warp] done. exists={os.path.isfile(OUT)} size={os.path.getsize(OUT) if os.path.isfile(OUT) else 0}", flush=True)
