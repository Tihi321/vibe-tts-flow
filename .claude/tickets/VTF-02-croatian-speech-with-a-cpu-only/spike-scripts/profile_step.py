import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import Engine, log_softmax, MASK_ID

llm = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
variant = sys.argv[3] if len(sys.argv) > 3 else "fp16"
eng = Engine(sys.argv[1], variant, int(os.environ.get("THR","0")), llm)
ids, am, t0 = eng.build("Dobar dan, ovo je test čitanja na hrvatskom jeziku.", 77, "hr")
print("S =", ids.shape[2])


def t(f, n=5):
    f()
    a = time.perf_counter()
    for _ in range(n):
        r = f()
    return (time.perf_counter() - a) / n, r


te, e = t(lambda: eng.emb.run(None, {"input_ids": ids, "audio_mask": am})[0])
feed = {"inputs_embeds": e.astype(np.float32)}
if "attention_mask" in eng.llm_inputs:
    feed["attention_mask"] = np.ones((1, ids.shape[2]), np.int64)
for i in eng.llm.get_inputs():
    if "past" in i.name:
        feed[i.name] = np.zeros((1, 8, 0, 128), np.float32)
tl, h = t(lambda: eng.llm.run(["hidden_states"], feed)[0])
th, lg = t(lambda: eng.heads.run(None, {"hidden_states": h.astype(eng.heads_dt)})[0])
tn, _ = t(lambda: log_softmax(lg[0, :, t0:t0 + 77, :].astype(np.float32)))
print(f"embeddings {te*1000:.0f} ms | llm {tl*1000:.0f} ms | heads {th*1000:.0f} ms | numpy softmax {tn*1000:.0f} ms")
