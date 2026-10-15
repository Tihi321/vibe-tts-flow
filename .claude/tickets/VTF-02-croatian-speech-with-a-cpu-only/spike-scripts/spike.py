"""Phase 0 spike: minimal numpy + onnxruntime + tokenizers OmniVoice ONNX synth (no torch/transformers)."""
import argparse, ctypes, json, math, os, sys, time
from ctypes import wintypes
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duration_upstream import RuleDurationEstimator  # copy of upstream omnivoice/utils/duration.py

SR = 24000
HOP = 960
MASK_ID = 1024
NCB = 8


def peak_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    h = k32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(h), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 1e6, pmc.WorkingSetSize / 1e6


def mk_sess(path, threads=0):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


class Engine:
    def __init__(self, model_dir, variant="fp16", threads=0, llm_path=None):
        t0 = time.perf_counter()
        d = model_dir if variant == "fp16" else os.path.join(model_dir, "int4")
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.emb = mk_sess(os.path.join(d, "audio_embeddings_encoder.onnx"), threads)
        self.llm = mk_sess(llm_path or os.path.join(d, "llm_decoder.onnx"), threads)
        self.llm_inputs = {i.name for i in self.llm.get_inputs()}
        self.heads = mk_sess(os.path.join(d, "audio_heads_decoder.onnx"), threads)
        self.dec = mk_sess(os.path.join(model_dir, "audio_tokenizer", "fp16", "higgs_decoder.onnx"), threads)
        self.enc = None
        self.model_dir = model_dir
        self.est = RuleDurationEstimator()
        self.heads_dt = np.float16 if "float16" in self.heads.get_inputs()[0].type else np.float32
        self.load_s = time.perf_counter() - t0
        self.rng = np.random.default_rng(0)

    # ---- tokenization (upstream _prepare_inference_inputs) ----
    def ids(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def build(self, text, T, lang, ref_text=None, ref_codes=None, denoise=True):
        style = ""
        if denoise and ref_codes is not None:
            style += "<|denoise|>"
        style += f"<|lang_start|>{lang if lang else 'None'}<|lang_end|><|instruct_start|>None<|instruct_end|>"
        full = (ref_text.strip() + " " + text.strip()) if ref_text else text.strip()
        wrapped = f"<|text_start|>{full}<|text_end|>"
        sid, tid = self.ids(style), self.ids(wrapped)
        n_pre = len(sid) + len(tid)
        R = ref_codes.shape[1] if ref_codes is not None else 0
        S = n_pre + R + T
        ids = np.zeros((1, NCB, S), np.int64)
        ids[0, :, :len(sid)] = sid
        ids[0, :, len(sid):n_pre] = tid
        if R:
            ids[0, :, n_pre:n_pre + R] = ref_codes
        ids[0, :, n_pre + R:] = MASK_ID
        am = np.zeros((1, S), bool)
        am[0, n_pre:] = True  # ref + target are "audio" positions (upstream)
        return ids, am, n_pre + R

    def hidden(self, ids, am):
        e = self.emb.run(None, {"input_ids": ids, "audio_mask": am})[0]
        B, S, _ = e.shape
        feed = {"inputs_embeds": e.astype(np.float32, copy=False)}
        if "attention_mask" in self.llm_inputs:
            feed["attention_mask"] = np.ones((B, S), np.int64)
        for i in self.llm.get_inputs():
            if "past" in i.name:
                feed[i.name] = np.zeros((B, 8, 0, 128), np.float32)
        return self.llm.run(["hidden_states"], feed)[0]

    def forward(self, ids, am):
        h = self.hidden(ids, am)
        return self.heads.run(None, {"hidden_states": h.astype(self.heads_dt, copy=False)})[0]  # (B,8,S,1025)

    def estimate_T(self, text, ref_text=None, ref_T=None, speed=1.0, cps=None):
        if cps is not None:  # simple chars/sec override
            return max(1, int(len(text) / cps * 25 / speed))
        if ref_text and ref_T:
            est = self.est.estimate_duration(text, ref_text, ref_T)
        else:
            est = self.est.estimate_duration(text, "Nice to meet you.", 25)
        return max(1, int(est / speed))

    def unmask(self, text, T, lang="hr", steps=16, cfg=0.0, t_shift=0.1, layer_pen=5.0, pos_temp=5.0,
               ref_text=None, ref_codes=None, cfg_frac=1.0):
        ids, am, tgt0 = self.build(text, T, lang, ref_text, ref_codes)
        if cfg:
            uids = np.full((1, NCB, T), MASK_ID, np.int64)
            uam = np.ones((1, T), bool)
        ts = np.linspace(0, 1, steps + 1)
        ts = t_shift * ts / (1 + (t_shift - 1) * ts)
        total = T * NCB
        rem = total
        sched = []
        for s in range(steps):
            k = rem if s == steps - 1 else min(math.ceil(total * (ts[s + 1] - ts[s])), rem)
            sched.append(int(k))
            rem -= int(k)
        tokens = np.full((NCB, T), MASK_ID, np.int64)
        layer = (np.arange(NCB) * layer_pen)[:, None]
        for s in range(steps):
            k = sched[s]
            if k <= 0:
                continue
            lg = self.forward(ids, am)[0, :, tgt0:tgt0 + T, :].astype(np.float32)  # (8,T,1025)
            lp = log_softmax(lg)
            if cfg and s < max(1, round(cfg_frac * steps)):
                up = log_softmax(self.forward(uids, uam)[0].astype(np.float32))
                lp = log_softmax(lp + cfg * (lp - up))
            lp[..., MASK_ID] = -np.inf
            pred = lp.argmax(-1)
            conf = lp.max(-1) - layer
            if pos_temp > 0:
                u = self.rng.random(conf.shape)
                conf = conf / pos_temp + (-np.log(-np.log(u + 1e-10) + 1e-10))
            conf = np.where(tokens != MASK_ID, -np.inf, conf)
            top = np.argpartition(-conf.ravel(), k - 1)[:k]
            flat = tokens.ravel().copy()
            flat[top] = pred.ravel()[top]
            tokens = flat.reshape(NCB, T)
            ids[0, :, tgt0:tgt0 + T] = tokens
            if cfg:
                uids[0] = tokens
        return tokens

    def decode(self, tokens):
        w = self.dec.run(None, {"codes": tokens[:, None, :]})[0]
        return w.astype(np.float32).reshape(-1)

    @staticmethod
    def post(w, thr_db=-50.0, keep_ms=100):
        # simplified upstream post-process: trim edge silence, peak-normalise to 0.5, fade + pad 0.1 s
        thr = 10 ** (thr_db / 20)
        idx = np.where(np.abs(w) > thr)[0]
        if len(idx):
            keep = int(SR * keep_ms / 1000)
            w = w[max(0, idx[0] - keep): idx[-1] + keep]
        pk = np.abs(w).max()
        if pk > 1e-6:
            w = w / pk * 0.5
        k = min(int(0.1 * SR), len(w) // 2)
        if k:
            w = w.copy()
            w[:k] *= np.linspace(0, 1, k, dtype=np.float32)
            w[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
        pad = np.zeros(int(0.1 * SR), np.float32)
        return np.concatenate([pad, w, pad])

    def synth(self, text, lang="hr", steps=16, cfg=0.0, cps=None, speed=1.0, **kw):  # kw: cfg_frac, ref_text, ref_codes
        t0 = time.perf_counter()
        rc = kw.get("ref_codes")
        T = self.estimate_T(text, speed=speed, cps=cps, ref_text=kw.get("ref_text"),
                            ref_T=rc.shape[1] if rc is not None else None)
        tokens = self.unmask(text, T, lang, steps, cfg, **kw)
        w = self.post(self.decode(tokens))
        dt = time.perf_counter() - t0
        return w, dt, T

    # ---- cloning ----
    def load_encoder(self, fp32=False):
        base = os.path.join(self.model_dir, "audio_tokenizer", "" if fp32 else "fp16")
        self.enc = {n: mk_sess(os.path.join(base, n + ".onnx")) for n in ("acoustic_encoder", "semantic_encoder", "quantizer_encoder")}
        self.enc_dt = np.float32 if fp32 else np.float16

    def encode_ref(self, wav24):
        import soxr
        n = len(wav24) - len(wav24) % HOP
        wav24 = wav24[:n].astype(np.float32)
        wav16 = soxr.resample(wav24, SR, 16000).astype(np.float32)
        dt = self.enc_dt
        a = self.enc["acoustic_encoder"].run(None, {"waveform_24k": wav24[None, None, :].astype(dt)})[0]
        s = self.enc["semantic_encoder"].run(None, {"waveform_16k": wav16[None, :].astype(dt)})[0]
        T = min(a.shape[2], s.shape[2])
        c = self.enc["quantizer_encoder"].run(None, {"acoustic_features": a[:, :, :T], "semantic_features": s[:, :, :T]})[0]
        return c[:, 0, :]  # (8,T)


SENTS = [
    "Dobar dan, ovo je test čitanja na hrvatskom jeziku.",
    "Šećer, čaj i đumbir kupila sam u dućanu.",
    "Danas je 15. listopada 2026. godine.",
    "Zagreb je glavni grad Republike Hrvatske, a nalazi se na sjeveru zemlje, između Medvednice i rijeke Save, gdje živi gotovo osamsto tisuća stanovnika.",
    "Što radiš večeras, hoćeš li doći na kavu ili ćemo se naći sutra poslijepodne?",
    "Hrvatska ima više od tisuću otoka, a najpoznatiji su Hvar, Brač, Korčula i Krk.",
    "Ljeto je bilo toplo, ali jesen je donijela kišu i hladne jutarnje vjetrove.",
    "Molim vas, možete li mi reći gdje je najbliža ljekarna?",
    "Učenici su napisali zadaću, a nastavnica ih je pohvalila za trud i urednost.",
    "Tehnologija se razvija brzo, pa je važno stalno učiti nove stvari.",
]
EN = "Hello, this is an English sentence for comparison."

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--steps", type=int, nargs="+", default=[16])
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--lang", default="hr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="limit number of sentences")
    ap.add_argument("--cfg_frac", type=float, default=1.0)
    ap.add_argument("--cps", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--llm", default=None, help="override llm onnx (bidirectional export)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("onnxruntime", ort.__version__, "threads", a.threads or "default")
    eng = Engine(a.model_dir, a.variant, a.threads, a.llm)
    print(f"load {eng.load_s:.2f}s")
    sents = SENTS + [EN]
    if a.n:
        sents = sents[:a.n]
    res = []
    for steps in a.steps:
        for i, t in enumerate(sents):
            lang = "en" if t == EN and a.lang == "hr" else (a.lang if a.lang != "none" else None)
            w, dt, T = eng.synth(t, lang, steps, a.cfg, a.cps, cfg_frac=a.cfg_frac)
            dur = len(w) / SR - 0.2
            fn = os.path.join(a.out, f"{a.tag}_s{steps}_cfg{a.cfg:g}_{i:02d}.wav")
            sf.write(fn, w, SR, subtype="PCM_16")
            r = dict(i=i, steps=steps, T=T, est_audio=T / 25, audio_s=dur, synth_s=dt, rtf=dt / max(dur, 1e-3), text=t, wav=fn)
            res.append(r)
            print(f"[{a.tag} steps={steps} cfg={a.cfg}] #{i} T={T} ({T/25:.1f}s est) audio={dur:.1f}s synth={dt:.2f}s RTF={r['rtf']:.2f}", flush=True)
    pk, cur = peak_mb()
    print(f"peak working set {pk:.0f} MB")
    json.dump(dict(load_s=eng.load_s, peak_mb=pk, ort=ort.__version__, res=res), open(os.path.join(a.out, f"{a.tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
