import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import Engine, MASK_ID

eng = Engine(sys.argv[1], "fp16")
ids, am, t0 = eng.build("Hello there.", 20, "en")
h1 = eng.hidden(ids, am).astype(np.float32)
ids2 = ids.copy()
ids2[0, :, t0:] = (np.arange(ids.shape[2] - t0) * 37 % 1000)[None, :]
am2 = am.copy()
h2 = eng.hidden(ids2, am2).astype(np.float32)
d = np.abs(h1 - h2)[0].max(-1)
print("S", ids.shape[2], "audio starts", t0)
print("max |dh| at positions BEFORE audio region (text prefix):", d[:t0].max(), " (0 => causal, >0 => bidirectional)")
print("per-pos dh first 5:", d[:5])
print("dh in audio region mean:", d[t0:].mean())
