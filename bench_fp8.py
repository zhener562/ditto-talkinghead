"""FP8 pipeline benchmark."""
import time, math, librosa, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stream_pipeline_offline import StreamSDK

SDK = StreamSDK('./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_bw_fp8.pkl', './checkpoints/ditto_trt_blackwell')

audio, sr = librosa.core.load('./example/audio.wav', sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)

SDK.setup('./example/image.png', './tmp/out_fp8.mp4')
SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})

t1 = time.time()
aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()
total = time.time() - t1
print(f'>>> FP8 pipeline: total={total:.2f}s, realtime={15.76/total:.2f}x', flush=True)
