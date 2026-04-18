"""Generate varied Japanese TTS via VoiceVox for FP8 calibration."""
import os, pathlib, urllib.request, urllib.parse, json

VV = "http://127.0.0.1:50021"
OUT = pathlib.Path(__file__).resolve().parent.parent / "calib" / "audios"
OUT.mkdir(parents=True, exist_ok=True)

# speaker id → label (diverse voices)
SPEAKERS = {
    2: "metan_normal",
    3: "zundamon_normal",
    8: "tsumugi_normal",
    10: "hau_normal",
    11: "genno_normal",
    12: "shirakami_futsuu",
    13: "aoyama_normal",
    14: "meimei_normal",
    16: "kyusyu_normal",
}

TEXTS = [
    "こんにちは、今日はとてもいい天気ですね。散歩に出かけるのにぴったりです。",
    "先週、新しい本を読み終えました。主人公の成長がとても印象的でした。",
    "えっ、本当ですか？それは驚きました！信じられないです。",
    "朝ごはんにトーストと目玉焼きを食べました。コーヒーも飲みました。",
    "電車が遅れていて、約束の時間に間に合いませんでした。本当に申し訳ありません。",
    "ねえ、聞いて！昨日、駅前で懐かしい友達に会ったんだ！すごく嬉しかった。",
    "このプロジェクトの期限は来週の金曜日です。進捗状況を共有してください。",
    "雨が降り始めました。傘を持ってくればよかったです。",
    "ありがとうございます。本当に助かりました。これからもよろしくお願いします。",
    "エンジンを切って、車から降りる時は、貴重品を忘れないでください。",
]

def request(path, payload=None, is_json=True, method="POST"):
    url = f"{VV}{path}"
    data = json.dumps(payload).encode() if payload else None
    headers = {"Content-Type": "application/json"} if is_json else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def synth(text, speaker_id):
    # 1. /audio_query?speaker=...
    q = urllib.request.urlopen(
        f"{VV}/audio_query?speaker={speaker_id}&text={urllib.parse.quote(text)}",
        data=b"", timeout=30
    ).read()
    # 2. /synthesis?speaker=...
    req = urllib.request.Request(
        f"{VV}/synthesis?speaker={speaker_id}",
        data=q, method="POST",
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def main():
    # pair: (speaker_id, text_idx)  — 一方を主体、もう一方を選んで変化をつける
    pairs = []
    spk_list = list(SPEAKERS.items())
    for i, text in enumerate(TEXTS):
        sid, sname = spk_list[i % len(spk_list)]
        pairs.append((sid, sname, i, text))
    # さらに追加で色々混ぜる
    import random
    random.seed(0)
    for _ in range(8):
        sid, sname = random.choice(spk_list)
        i = random.randrange(len(TEXTS))
        pairs.append((sid, sname, i, TEXTS[i]))

    for idx, (sid, sname, tidx, text) in enumerate(pairs):
        out = OUT / f"{idx:02d}_{sname}_t{tidx}.wav"
        try:
            wav = synth(text, sid)
            out.write_bytes(wav)
            print(f"  OK  {out.name}  ({len(wav)/1024:.0f} KB, '{text[:20]}…')", flush=True)
        except Exception as e:
            print(f"  FAIL {out.name}  ({e})", flush=True)

if __name__ == "__main__":
    main()
