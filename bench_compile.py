import time, torch, math, librosa, sys
torch.set_float32_matmul_precision('high')
from stream_pipeline_offline import StreamSDK

mode = sys.argv[1] if len(sys.argv) > 1 else 'none'  # none | compile

SDK = StreamSDK('./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl', './checkpoints/ditto_pytorch')

if mode == 'compile':
    print('>>> compiling warp + decode (default mode) + warmup', flush=True)
    import numpy as np
    SDK.warp_f3d.warp_net.model = torch.compile(SDK.warp_f3d.warp_net.model, dynamic=False)
    SDK.decode_f3d.decoder.model = torch.compile(SDK.decode_f3d.decoder.model, dynamic=False)
    # Warmup in main thread
    t_w = time.time()
    dummy_f3d = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)
    dummy_kp = np.random.randn(1, 21, 3).astype(np.float32)
    out = SDK.warp_f3d(dummy_f3d, dummy_kp, dummy_kp)
    print(f'>>> warp warmup: {time.time()-t_w:.2f}s, shape={out.shape}', flush=True)
    t_w = time.time()
    out = SDK.decode_f3d(out)
    print(f'>>> decode warmup: {time.time()-t_w:.2f}s, shape={out.shape}', flush=True)

audio, sr = librosa.core.load('./example/audio.wav', sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)

SDK.setup('./example/image.png', f'./tmp/out_{mode}.mp4')
SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})

t1 = time.time()
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()
total = time.time() - t1
print(f'>>> mode={mode}: total={total:.2f}s, realtime={15.76/total:.2f}x', flush=True)
