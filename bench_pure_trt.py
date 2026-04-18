"""Pure TRT 10 pipeline benchmark (warp_network also via TRT plugin)."""
import time, math, librosa
from stream_pipeline_offline import StreamSDK

SDK = StreamSDK('./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_bw_pure.pkl', './checkpoints/ditto_trt_blackwell')

audio, sr = librosa.core.load('./example/audio.wav', sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)

SDK.setup('./example/image.png', './tmp/out_pure_trt.mp4')
SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})

t1 = time.time()
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()
total = time.time() - t1
print(f'>>> PURE TRT: total={total:.2f}s, realtime={15.76/total:.2f}x', flush=True)
