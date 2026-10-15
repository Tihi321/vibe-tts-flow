"""Cloning spike: encode an auto-voice clip with the fp32 Higgs encoders and reuse as reference."""
import json, os, sys, time
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import Engine, SENTS, SR, peak_mb

model_dir, llm, out = sys.argv[1:4]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 8
cfg = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
eng = Engine(model_dir, "fp16", 0, llm)
t0 = time.perf_counter()
eng.load_encoder(fp32=True)
print(f"encoder load {time.perf_counter()-t0:.2f}s")
ref_wav_path = os.path.join(out, "bd_fp32_cfg2_s16_cfg2_03.wav")
ref_text = SENTS[3]
w, sr = sf.read(ref_wav_path, dtype="float32")
assert sr == SR
w = w[int(0.1 * SR):-int(0.1 * SR)]
t0 = time.perf_counter()
codes = eng.encode_ref(w)
print(f"encode ref ({len(w)/SR:.1f}s) -> codes {codes.shape} in {time.perf_counter()-t0:.2f}s")
np.save(os.path.join(out, "ref.codes.npy"), codes)
res = []
for i in (0, 1, 2, 4, 5):
    t = SENTS[i]
    wav, dt, T = eng.synth(t, "hr", steps, cfg, ref_text=ref_text, ref_codes=codes)
    dur = len(wav) / SR - 0.2
    fn = os.path.join(out, f"clone_s{steps}_cfg{cfg:g}_{i:02d}.wav")
    sf.write(fn, wav, SR, subtype="PCM_16")
    res.append(dict(i=i, steps=steps, T=T, audio_s=dur, synth_s=dt, rtf=dt / max(dur, 1e-3), text=t, wav=fn))
    print(f"clone #{i} T={T} audio={dur:.1f}s synth={dt:.2f}s RTF={dt/max(dur,1e-3):.2f}", flush=True)
json.dump(dict(res=res), open(os.path.join(out, f"clone_s{steps}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("peak MB", peak_mb()[0])
