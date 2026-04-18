"""Profile each stage on the TRT+compile hybrid setup."""
import time, torch, math, librosa, numpy as np
torch.set_float32_matmul_precision('high')
from stream_pipeline_offline import StreamSDK
from core.models import warp_network as wn_mod, decoder as dec_mod

stats = {'warp': [], 'decode': [], 'putback': [], 'writer': []}
def wrap(cls, sk, sync=False):
    orig = cls.__call__
    def w(self, *a, **kw):
        if sync: torch.cuda.synchronize()
        t=time.time()
        r = orig(self, *a, **kw)
        if sync: torch.cuda.synchronize()
        stats[sk].append(time.time()-t); return r
    cls.__call__ = w
wrap(wn_mod.WarpNetwork, 'warp', sync=True)
wrap(dec_mod.Decoder, 'decode', sync=True)
from core.atomic_components.putback import PutBack
wrap(PutBack, 'putback')

SDK = StreamSDK('./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_bw.pkl', './checkpoints/ditto_trt_blackwell')

SDK.warp_f3d.warp_net.model = torch.compile(SDK.warp_f3d.warp_net.model, dynamic=False)
f3d = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)
kp = np.random.randn(1, 21, 3).astype(np.float32)
_ = SDK.warp_f3d(f3d, kp, kp); stats['warp'].clear()  # discard warmup
_ = SDK.decode_f3d(np.random.randn(1, 256, 64, 64).astype(np.float32)); stats['decode'].clear()

audio, sr = librosa.core.load('./example/audio.wav', sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)

SDK.setup('./example/image.png', './tmp/prof_trt.mp4')
SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})

# Also time writer
orig_writer = SDK.writer
def t_writer(frame, fmt='rgb'):
    t=time.time(); r = orig_writer(frame, fmt); stats['writer'].append(time.time()-t); return r
SDK.writer = t_writer

t1 = time.time()
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()
total = time.time() - t1

print(f'\n=== Profile (TRT+compile hybrid, {num_f} frames) ===', flush=True)
for k,v in stats.items():
    if v:
        print(f'{k:10s}: n={len(v):3d}, mean={1000*sum(v)/len(v):6.2f}ms, sum={sum(v):6.2f}s, max={1000*max(v):6.2f}ms')
print(f'wall clock: {total:.2f}s  → realtime={15.76/total:.2f}x')
