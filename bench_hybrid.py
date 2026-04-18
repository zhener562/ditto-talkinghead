"""Hybrid TRT + torch.compile(warp) benchmark."""
import time, torch, math, librosa, sys, numpy as np
torch.set_float32_matmul_precision('high')
from stream_pipeline_offline import StreamSDK

SDK = StreamSDK('./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_bw.pkl', './checkpoints/ditto_trt_blackwell')

# Only warp is PyTorch now. Compile + warmup.
print('>>> compile warp', flush=True)
SDK.warp_f3d.warp_net.model = torch.compile(SDK.warp_f3d.warp_net.model, dynamic=False)
# Warmup
t0 = time.time()
f3d = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)
kp = np.random.randn(1, 21, 3).astype(np.float32)
_ = SDK.warp_f3d(f3d, kp, kp)
print(f'>>> warp warmup {time.time()-t0:.2f}s', flush=True)

audio, sr = librosa.core.load('./example/audio.wav', sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)

SDK.setup('./example/image.png', './tmp/out_hybrid.mp4')
SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})

t1 = time.time()
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()
total = time.time() - t1
print(f'>>> HYBRID (TRT+compile): total={total:.2f}s, realtime={15.76/total:.2f}x', flush=True)
