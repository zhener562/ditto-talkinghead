# Ditto-TalkingHead on Blackwell (RTX PRO 4000 / sm_120)

TensorRT 10 / CUDA 12.8 移植メモ。2026-04-19 時点。

## ベンチ結果 (example/audio.wav 15.76s, 394 frames @ 25fps, 1432×1432)

| 設定 | Total | RT | vs baseline |
|---|---:|---:|---:|
| baseline PyTorch (eager, cu128) | 23.04s | 0.68x | 1.00x |
| + torch.compile (warp+decode, warmup) | 19.15s | 0.82x | 1.20x |
| TRT 10 全体 + PyTorch warp | 19.25s | 0.82x | 1.20x |
| TRT 10 + torch.compile(warp) | 14.04s | 1.12x | 1.64x |
| pure TRT 10 (warp plugin 自前ビルド) | 9.26s | 1.70x | 2.49x |
| **TRT 10 FP8 (decoder+warp QDQ)** | **7.51s** | **2.10x** | **3.07x** (PSNR Y=39.5dB) |

ピーク VRAM は PyTorch 版で 1.45 GB。TRT 化後は計測未、ただし engines 合計 ~550MB。

## 環境

| コンポーネント | バージョン |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell (sm_120, 24GB GDDR7) |
| ドライバ | 580.126.09 |
| CUDA | 12.8 |
| Python | 3.10.20 (uv 管理) |
| PyTorch | 2.7.1+cu128 |
| TensorRT | 10.16.1.11 (`tensorrt-cu12` pip) |
| cuda-python | 12.9.6 (13.x は `from cuda import cuda` 非互換) |

## パイプライン解剖

```
audio (16kHz PCM)
  ↓ hubert (音声特徴)
  ↓ lmdm_v0.4_hubert (DDIM 50 step で motion 予測) ← ここは並列 thread
  ↓ motion_stitch (CPU)
  ↓ warp_network (5D feature volume を 3D grid で warp) ← 最大の律速
  ↓ decoder (SPADE, 512×512 RGB 出力)
  ↓ putback (元解像度に合成, CPU)
  ↓ writer (imageio-ffmpeg エンコード)
```

**重要な観測:** 6 段は `threading.Thread` で producer/consumer 並列動作。ボトルネックは **warp の 1 ステージだけ** が全体律速を決める。

### プロファイル (PyTorch baseline)

| ステージ | mean | 備考 |
|---|---:|---|
| warp | 50.42 ms/frame | GPU、~20 fps floor |
| decode | 31.27 ms/frame | GPU |
| putback | 6.15 ms/frame | CPU |
| writer | 1.94 ms/frame | CPU ffmpeg |

### DDIM ステップ削減は効かない

`sampling_timesteps` を 50→5 にしても 24s→21s (1.15x) のみ。LMDM は別スレッドで audio 全体を先行処理する producer。律速は writer 側なので意味なし。

## 何が動いた / 動かなかった

### ✅ TRT 10 + Polygraphy でビルド成功

以下のモデルは `polygraphy convert --fp16 --builder-optimization-level=5` で普通にビルド成功:
- `appearance_extractor`, `motion_extractor`, `decoder`
- `stitch_network`, `hubert`, `lmdm_v0.4_hubert`
- `insightface_det`, `blaze_face`, `face_mesh`, `landmark106`, `landmark203`

Blackwell native (sm_120) にしたいので `--hardware-compatibility-level=Ampere_Plus` は**外す**。Ampere_Plus をつけると sm_80-89 共通カーネルになり、sm_120 専用最適化が効かない。

### ❌ warp_network: TRT 10 で非対応

- `warp_network.onnx` は `GridSample3D` というカスタム op を含む（ditto 独自）
- `warp_network_ori.onnx` は標準 `GridSample` だが **5D 入力** (3D grid sample)
- TRT 10 の `IGridSampleLayer` は **4D 入力のみ**（2D grid sample）
  - エラー: `addGridSample: Parameter check failed, condition: input.getDimensions().nbDims == 4`
- 配布の `libgrid_sample_3d_plugin.so` は **TRT 8.6 ABI**（`libnvinfer.so.8` 依存）で TRT 10 runtime にロード不可

→ 暫定対処: warp_network だけ PyTorch `.pth` + `torch.compile` で動かす

### ❌ torch.compile を素のまま適用 → デッドロック

ditto のワーカーは **非メインスレッドから初回呼び出し** される。Dynamo は main thread optimization が強く、別スレッドからの初回 compile で数分止まる。

**回避策:** ロード直後、ワーカー起動前に **メインスレッドでダミー入力 1回通してコンパイル**。以降はキャッシュヒット。

```python
SDK.warp_f3d.warp_net.model = torch.compile(SDK.warp_f3d.warp_net.model, dynamic=False)
_ = SDK.warp_f3d(dummy_f3d, dummy_kp, dummy_kp)  # warmup in main thread
```

初回 9s 程度、以降は `~/.cache/torch/inductor/` にヒットして ~1s。

### ❌ onnxruntime-gpu フォールバックは使えない

`CUDAExecutionProvider` は `libcublasLt.so.12` を探して失敗 → CPU に落ちる（500ms/frame）。

PyTorch の bundle 版 cublas は pip の `nvidia-cublas-cu12` に入っているが ORT が見つけない。`LD_LIBRARY_PATH` 対応が必要だが、今回は回避。

### その他ハマりどころ

1. **`cuda-python` バージョン**: 13.x は `from cuda import cuda, cudart, nvrtc` が動かない。`core/utils/tensorrt_utils.py` が古い API を使っているので `cuda-python<13` を固定。
2. **TRT 8 prebuilt engine は Blackwell で読めない**: SM コード埋め込み時点で sm_80-89 のみ。`ditto_trt_Ampere_Plus/` は Blackwell ではロード段階でエラー。
3. **cfg_pkl はファイル名をハードコード**: hybrid 構成（warp だけ .pth）にするには cfg を書き換えて別 pkl を作る必要がある（`v0.4_hubert_cfg_trt_bw.pkl`）。

## 成果物

| ファイル | 用途 |
|---|---|
| `scripts/build_trt10_blackwell.py` | polygraphy 一括ビルド |
| `bench_compile.py` / `bench_hybrid.py` | ベンチ |
| `checkpoints/ditto_trt_blackwell/*.engine` | TRT 10 FP16/FP32 エンジン（要ローカルビルド、~550MB） |
| `checkpoints/ditto_trt_blackwell/warp_network.pth` | PyTorch フォールバック用 |
| `checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_bw.pkl` | hybrid cfg (warp だけ .pth) |

### 実行コマンド

```bash
cd ~/ドキュメント/github/ditto-talkinghead
source .venv/bin/activate
python inference.py \
  --data_root "./checkpoints/ditto_trt_blackwell" \
  --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_bw.pkl" \
  --audio_path "./example/audio.wav" \
  --source_path "./example/image.png" \
  --output_path "./tmp/result_trt.mp4"
```

ただしベンチでは `bench_hybrid.py` の方が torch.compile を明示的にウォームアップするため速い（14s vs 19s）。`inference.py` 側に compile 処理は未統合。

---

## 次の射程: GridSample3D プラグインを TRT 10 対応ビルド

warp が PyTorch のまま残っているのが最後のボトルネック。これを TRT 10 化すれば **~2x 短縮 → 1.5〜2x realtime** が射程に入る。

### プラグインソース

**https://github.com/SeanWangJS/grid-sample3d-trt-plugin** (Apache-2.0)

- 配布の `libgrid_sample_3d_plugin.so` の正体。これを TRT 10 / sm_120 で再ビルドすれば OK
- 実装: `IPluginV2DynamicExt` ベース（TRT 10 でも deprecated だが動く）
- CUDA カーネル: 標準的な trilinear interpolation on 5D tensor

### ビルド手順（Blackwell 向け）

```bash
git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git
cd grid-sample3d-trt-plugin

# TRT 10 の header/lib を pip wheel から拾う
TRT_ROOT=$(python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))")
TRT_INC=$(python -c "import tensorrt; import os; print(os.path.dirname(tensorrt.__file__))")/include

mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTensorRT_INCLUDE_DIR="$TRT_INC" \
  -DTensorRT_LIB_DIR="$TRT_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build . -j"$(nproc)"
# → build/libgrid_sample_3d_plugin.so
```

### 想定される修正点（TRT 10 移行）

1. **`CMakeLists.txt` の `CMAKE_CUDA_ARCHITECTURES`**: `"70;80;86;89"` ハードコードを外すか `120` を追加
2. **Plugin Creator Registration API**: TRT 8.6 → 10 で `IPluginCreatorInterface` → `IPluginCreator` の変更（Issue #13 参照）
3. **`libnvinfer.so.8` → `libnvinfer.so.10`**: リンカフラグ変更のみ
4. **`IPluginV2DynamicExt` 自体は TRT 10 で deprecated だが動く**。将来 `IPluginV3` 化するとさらに綺麗

### 統合手順

ビルド成功後:

```python
# core/utils/tensorrt_utils.py の冒頭に追加
import ctypes
ctypes.CDLL("/path/to/libgrid_sample_3d_plugin.so")
```

または `warp_network.onnx`（オリジナル、GridSample3D 版）を TRT でビルドする時に polygraphy の `--plugins` フラグでロード:

```bash
polygraphy convert warp_network.onnx -o warp_network_fp16.engine \
  --fp16 --builder-optimization-level=5 \
  --plugins /path/to/libgrid_sample_3d_plugin.so
```

その後 cfg_pkl の `warp_network_cfg.model_path` を `.engine` に戻せば純 TRT 化完了。

### 期待される性能

- warp PyTorch (現状): 40-50 ms/frame (torch.compile 後)
- warp TRT 10 FP16 (予測): 15-25 ms/frame

他ステージの TRT はすでに decode 10ms, putback 6ms 程度と推定されるので、フレームあたり 50+16+6 ≈ **72ms → ~14 fps**... 待てよ、writer は現状で 30 fps 出てるので実質もう warp が律速ではない。

**再検討:** 現状の 14.04s の内訳:
- 並列実行なので最も遅いステージが律速
- writer 30fps ≒ 33ms/frame が観測されている
- warp TRT 化で warp が 20ms 以下になれば **decoder (TRT 10ms) と合わせて ~30-35 ms/frame = 28-33 fps**
- 394 frame / 30 fps ≒ **13 秒 → 1.20x realtime**

つまり、warp を TRT 化しても **劇的な向上ではなく 1.12x → 1.20x 程度**の可能性が高い。律速は decoder や writer ffmpeg エンコードに移っている。

**真のブレークスルー狙いなら:**
1. **FP8 量子化**（Blackwell の第5世代 Tensor Core）: decoder/warp を 2x、`--fp8` でキャリブレーションデータを数百サンプル用意
2. **writer の GPU エンコード**: `imageio-ffmpeg` CPU → `nvenc` に置き換え
3. **putback も GPU 化**: 現状 CPU でアフィン変換、CUDA 実装に

### どれを最優先でやるか

コスパ順:
1. ~~**GridSample3D plugin ビルド + 統合**~~ → **完了**（1時間、+50% で 1.12→1.70x realtime）
2. **FP8 量子化**（1-2日、期待 +50-100%、Blackwell の真価、3-4x realtime 射程）
3. **GPU putback**（数時間、putback 6ms/frame を CUDA 化、律速じゃないので効果小）

## プラグインビルド手順（完了記録）

1. `nvcc 12.8` が必要。micromamba でサンドボックス:
   ```bash
   export MAMBA_ROOT_PREFIX=$HOME/.local/share/micromamba
   ~/.local/bin/micromamba create -n cuda128 -c conda-forge "cuda-nvcc=12.8" "cuda-cccl=12.8" -y
   ```
2. TRT 10 headers を公式 OSS から取得（pip wheel には入ってない）:
   ```bash
   git clone --depth=1 --branch release/10.16 https://github.com/NVIDIA/TensorRT.git /tmp/trt-oss
   cp /tmp/trt-oss/include/*.h ~/ドキュメント/github/grid-sample3d-trt-plugin/trt10_include/
   ```
3. `tensorrt_libs/` の `libnvinfer.so.10` に `libnvinfer.so` シンボリックリンク（リンカ用）
4. CMakeLists.txt を一部改修（`CMAKE_CUDA_ARCHITECTURES` 外部指定可、`target_link_directories` 追加、`test/` を opt-in）
5. cmake + build
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
     -DCMAKE_CUDA_COMPILER=$HOME/.local/share/micromamba/envs/cuda128/bin/nvcc \
     -DCUDAToolkit_ROOT=$HOME/.local/share/micromamba/envs/cuda128 \
     -DTensorRT_INCLUDE_DIR=.../trt10_include \
     -DTensorRT_LIB_DIR=.../tensorrt_libs
   cmake --build . -j
   ```
6. 生成された `libgrid_sample_3d_plugin.so` を `core/utils/tensorrt_utils.py` 冒頭で `ctypes.CDLL(...)` ロード（環境変数 `DITTO_GS3D_PLUGIN` で上書き可）
7. `warp_network.onnx`（カスタム GridSample3D op 版）を `polygraphy convert --plugins <so>` で engine 化
8. cfg を純 TRT 用に戻す（`v0.4_hubert_cfg_trt_bw_pure.pkl`）

ビルド注意点: `IPluginV2` / `registerCreator` は TRT 10 で deprecated 警告が出るが動作する。将来 `IPluginV3` 化すれば警告が消えて綺麗。

## FP8 量子化（3.07x baseline 達成）

### データセット

Seedream v4 (WaveSpeed API) で多様な日本人・アジア人の顔画像 15 枚を生成、VoiceVox で 9 話者 × 10 テキスト = 18 本の音声を用意。16×2 = 32 ラン × 20 フレーム subsample = **640 サンプル** を中間テンソルとして保存。

- `scripts/gen_calib_images.py`: seedream 画像生成
- `scripts/gen_calib_audios.py`: VoiceVox TTS 生成
- `scripts/dump_calib_tensors.py`: pipeline 走らせて warp/decoder 入力を `.npz` に dump
- 出力: `calib/tensors/{warp,decoder}_calib.npz`（3.8GB, gitignore）

### 発見したバグ (modelopt の fp16 オーバーフロー)

`modelopt.onnx.quantization.fp8.py:72` の:
```python
np_fp8_scale = (np_scale * 448.0) / 127.0
```
`np_scale` が fp16 だと `scale × 448 > 65504` で **inf 化** → 後続で nan 連鎖 → decoder 出力が真っ白に saturate。

修正パッチ（`.venv/` 内、要再適用）:
```python
np_fp8_scale = (np_scale.astype(np.float32) * 448.0) / 127.0
```

症状: 4つの異なる FP8 変種（max/entropy キャリブ × 全量子化/SPADE除外 × weight-only）で **全て同一の壊れた出力** (md5 一致)。これが「全層 inf スケール」を示す決定的証拠だった。

### QDQ 挿入手順

1. `scripts/fp8_quantize_decoder.py`: modelopt で decoder.onnx → decoder_fp8_qdq.onnx（109 ノード QDQ 挿入, 256 サンプル max キャリブ, 4分）
2. `scripts/fp8_quantize_warp.py`: 同様に warp_network.onnx → warp_network_fp8_qdq.onnx（18 ノード, GridSample3D カスタム op は `trt_plugins` で ORT に通知, 20秒）
3. `polygraphy convert *.onnx -o *_fp8.engine --strongly-typed --builder-optimization-level=5`（QDQ を尊重するために **strongly-typed 必須**、`--fp8 --fp16` だと TRT が勝手に全層 FP8 化）

### 品質検証

PSNR (Y channel) = **39.47 dB** (FP16 基準)。40dB 超はプロ水準の無劣化。視覚的に区別不可能。

## 新しい律速 (pure TRT)

| ステージ | mean | vs baseline |
|---|---:|---|
| warp (TRT plugin) | **18.94 ms** | 50→19ms (2.6x) |
| decode (TRT FP16) | 19.47 ms | 31→19ms (1.6x) = **現律速** |
| putback (CPU) | 6.35 ms | 不変 |
| writer (ffmpeg) | 2.05 ms | 不変 |

warp と decode が 19ms で均衡。次に効くのは **FP8 で decoder を半減**。

---

## 参考リンク

- 配布元: [antgroup/ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead)
- 重み: [digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead)
- GridSample3D プラグインソース: [SeanWangJS/grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin)
- DGX Spark (ARM64) 移植記事: [blog.mikihands.com](https://blog.mikihands.com/en/whitedec/2026/3/3/ditto-talkinghead-dgx-spark-arm64-tensorrt-porting/)
- NVIDIA TRT 3D grid_sample 非対応 issue: [NVIDIA/TensorRT#2612](https://github.com/NVIDIA/TensorRT/issues/2612)
