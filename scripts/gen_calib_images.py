"""Generate diverse portrait images via WaveSpeed seedream-v4 for FP8 calibration."""
import os, sys, re, time, json, pathlib, base64, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "https://api.wavespeed.ai/api/v3"
SECRETS = pathlib.Path.home() / "ドキュメント/github/hage/Secrets.toml"
OUT = pathlib.Path(__file__).resolve().parent.parent / "calib" / "images"
OUT.mkdir(parents=True, exist_ok=True)

def api_key():
    for line in SECRETS.read_text().splitlines():
        m = re.match(r'^WAVESPEEDAI_API_KEY\s*=\s*"([^"]+)"', line)
        if m: return m.group(1)
    raise SystemExit("WAVESPEEDAI_API_KEY not found")

# Diverse portraits for talking-head calibration: varied age/gender/lighting/angle/ethnicity
PROMPTS = [
    ("jp_f_20s_warm",    "beautiful Japanese woman, 25 years old, natural makeup, soft warm sunset lighting, shoulder-length black hair, white blouse, head-and-shoulders portrait, shallow depth of field, photorealistic"),
    ("jp_m_30s_studio",  "handsome Japanese man, 32 years old, short black hair, navy blue suit, clean studio lighting, front-facing portrait, professional photo, photorealistic"),
    ("jp_f_40s_soft",    "Japanese woman, 45 years old, elegant, shoulder-length hair with gentle waves, soft window light, beige turtleneck, gentle smile, photorealistic portrait"),
    ("jp_m_50s_outdoor", "Japanese man, 55 years old, salt-and-pepper hair, glasses, outdoor natural light, casual cardigan, kind eyes, photorealistic portrait"),
    ("jp_f_teen_school", "Japanese teenage girl, 17 years old, long straight black hair, school uniform, bright afternoon light, friendly expression, photorealistic portrait"),
    ("jp_m_20s_casual",  "Japanese young man, 23 years old, tousled hair, casual t-shirt, cafe window light, relaxed pose, slight smile, photorealistic"),
    ("asian_f_30s_side", "Chinese woman, 30 years old, soft side lighting, shoulder-length brown hair, cream blouse, three-quarter portrait, looking at camera, photorealistic"),
    ("asian_m_40s_glass","Korean man, 40 years old, clean-shaven, round glasses, gray sweater, soft overhead light, thoughtful expression, front portrait, photorealistic"),
    ("jp_f_30s_night",   "Japanese woman, 30 years old, straight bob haircut, evening indoor light, warm lamp glow, navy sweater, subtle earrings, photorealistic portrait"),
    ("jp_m_20s_dark",    "Japanese man, 25 years old, stylish short hair, charcoal hoodie, moody low-key lighting, strong jawline, direct gaze, photorealistic"),
    ("multi_f_40s",      "mixed-race Asian woman, 40 years old, curly auburn hair, natural daylight, earth-tone blouse, warm smile, photorealistic portrait"),
    ("jp_elder_m",       "elderly Japanese man, 70 years old, white hair, gentle wrinkles, soft diffused light, traditional collared shirt, kind smile, photorealistic portrait"),
    ("jp_child_f",       "Japanese girl, 10 years old, shoulder-length pigtails, bright outdoor afternoon sunshine, pastel yellow dress, cheerful smile, photorealistic portrait"),
    ("jp_f_30s_glasses", "Japanese woman, 33 years old, oval glasses, chin-length bob, white shirt, clean studio lighting, neutral expression, photorealistic portrait"),
    ("jp_m_60s_stern",   "Japanese businessman, 62 years old, short gray hair, dark formal suit, serious expression, studio lighting, photorealistic portrait"),
]

def request_image(key, prompt, name):
    req = urllib.request.Request(
        f"{API_BASE}/bytedance/seedream-v4",
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps({
            "prompt": prompt,
            "size": "1024*1024",
            "enable_sync_mode": False,
        }).encode(),
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    task_id = resp["data"]["id"]
    result_url = resp["data"]["urls"]["get"]
    # Poll
    start = time.time()
    while time.time() - start < 90:
        time.sleep(2)
        with urllib.request.urlopen(urllib.request.Request(result_url, headers={"Authorization": f"Bearer {key}"})) as r:
            data = json.loads(r.read())["data"]
        if data["status"] == "completed":
            url = data["outputs"][0]
            urllib.request.urlretrieve(url, str(OUT / f"{name}.png"))
            return name, True, None
        if data["status"] == "failed":
            return name, False, data.get("error", "failed")
    return name, False, "timeout"

def main():
    key = api_key()
    # Keep one real example too (for continuity)
    print(f"Writing to: {OUT}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(request_image, key, p, n): n for n, p in PROMPTS}
        for f in as_completed(futs):
            n, ok, err = f.result()
            print(f"  {'OK ' if ok else 'FAIL'} {n}" + (f"  ({err})" if err else ""), flush=True)

if __name__ == "__main__":
    main()
