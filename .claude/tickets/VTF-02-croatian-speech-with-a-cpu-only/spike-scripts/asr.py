"""Transcribe wavs with faster-whisper (run inside asrvenv). usage: asr.py <json from spike> [model] """
import sys, json, re, difflib
import soundfile as sf, soxr, numpy as np
from faster_whisper import WhisperModel

res = json.load(open(sys.argv[1], encoding="utf-8"))["res"]
size = sys.argv[2] if len(sys.argv) > 2 else "small"
m = WhisperModel(size, device="cpu", compute_type="int8")


def norm(s):
    return re.sub(r"[^\w\s]", "", s.lower()).split()


tot = 0
n = 0
out = []
for r in res:
    lang = "en" if r["text"].startswith("Hello") else "hr"
    x, sr = sf.read(r["wav"], dtype="float32"); x = soxr.resample(x, sr, 16000)
    segs, info = m.transcribe(x, language=lang, beam_size=5)
    tr = " ".join(s.text.strip() for s in segs)
    a, b = norm(r["text"]), norm(tr)
    sm = difflib.SequenceMatcher(None, a, b)
    wacc = sm.ratio()
    r["asr"] = tr
    r["wratio"] = wacc
    out.append(r)
    print(f"#{r['i']} steps={r['steps']} ratio={wacc:.2f}\n   IN : {r['text']}\n   ASR: {tr}", flush=True)
json.dump(out, open(sys.argv[1].replace(".json", ".asr.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for st in sorted({r["steps"] for r in out}):
    xs = [r["wratio"] for r in out if r["steps"] == st]
    print(f"steps={st} mean word-match ratio={sum(xs)/len(xs):.2f}")
