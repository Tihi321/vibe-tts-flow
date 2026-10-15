"""OmniVoice speech engine: English, Croatian and many other languages on CPU without torch.

A numpy + onnxruntime + tokenizers port of the k2-fsa/OmniVoice generation loop: a masked
diffusion language model that fills 8 audio codebooks over a fixed number of frames in a few
unmasking steps, with classifier-free guidance. The model files are built once by
vibetts/omnivoice_setup.py (Settings > OmniVoice) into models/omnivoice/ (see manifest.json there).

Every chunk is generated with a *voice preset* as its reference prefix (codes + text of a short
clip the model produced earlier), so the speaker stays the same from sentence to sentence.
Create one with `python -m vibetts.omnivoice make-voice <name> --seed N`, design one from a
description (`design-voice`), or clone one from a recording (`clone-voice`).

The model weights are CC-BY-NC (non-commercial); the code is Apache-2.0.
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import gc
import json
import logging
import math
import re
import struct
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
OMNI_PREFIX = "omni"
DEFAULT_VOICE = "omni:default"
DEFAULT_DIR = Path("models/omnivoice")
MANIFEST = "manifest.json"
DEFAULT_TEXT = "Dobar dan, ovo je test čitanja na hrvatskom jeziku."

HOP = 960  # samples per audio frame: 25 frames per second
FPS = SAMPLE_RATE // HOP
MASK_ID = 1024
CODEBOOKS = 8
CFG_SCALE = 2.0  # upstream default; without it few-step output is not intelligible
T_SHIFT = 0.1
LAYER_PENALTY = 5.0
POSITION_TEMP = 5.0

LLM_FILE = "llm_bidir_int4.onnx"
EMBED_FILE = "audio_embeddings_encoder.onnx"
HEADS_FILE = "audio_heads_decoder.onnx"
DECODER_FILE = "higgs_decoder.onnx"
TOKENIZER_FILE = "tokenizer.json"
ENCODERS_DIR = "encoders"
ENCODER_NAMES = ("acoustic_encoder", "semantic_encoder", "quantizer_encoder")

RESERVED_NAME = "default"
MAX_NAME = 32
MIN_CLONE_S = 2.0  # shorter recordings do not carry enough of the voice
MAX_CLONE_S = 8.0  # longer references slow down every chunk (and can't be cut: the transcript must match)
DESIGN_STEPS = 16
# Voice design is trained on zh/en only, so the design clip is generated from this sentence.
# The experiment that picked the language is in the VTF-02 ticket notes.
DESIGN_TEXTS = {"hr": DEFAULT_TEXT, "en": "Hello, this is a short sample of how my voice sounds."}
DESIGN_LANG = "hr"


# ---------------------------------------------------------------- voice ids and files

def is_omni(voice: str) -> bool:
    """'omni:<preset>', or the legacy 'omni_<lang>[:<preset>]' / bare 'omni'."""
    return voice == OMNI_PREFIX or voice.startswith((OMNI_PREFIX + ":", OMNI_PREFIX + "_"))


def _manifest_files(model_dir: Path) -> list[str] | None:
    try:
        data = json.loads((Path(model_dir) / MANIFEST).read_text(encoding="utf-8"))
        return list(data["files"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def models_present(model_dir: Path) -> bool:
    """True when manifest.json exists and every file it lists is there."""
    names = _manifest_files(model_dir)
    return bool(names) and all((Path(model_dir) / n).is_file() for n in names)


def _voices_dir(model_dir: Path) -> Path:
    return Path(model_dir) / "voices"


def list_presets(model_dir: Path) -> list[str]:
    """Voice ids ('omni:<name>') of the presets in <model_dir>/voices. No model load."""
    d = _voices_dir(model_dir)
    if not d.is_dir():
        return []
    return [f"omni:{p.stem}" for p in sorted(d.glob("*.npy"))
            if p.with_suffix(".txt").is_file()]


def preset_info(model_dir: Path, name: str) -> dict:
    """What the Voices list shows for preset `name`: {'name', 'voice', 'kind', 'description',
    'lang'}. kind is 'default', 'description' (designed), 'recording' (cloned) or 'random';
    lang is the language id of the preset's sample ('' when unknown)."""
    meta: dict = {}
    try:
        meta = json.loads(_preset_files(model_dir, name)[3].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if not isinstance(meta, dict):
        meta = {}
    kind = {"design": "description", "clone": "recording"}.get(meta.get("kind"), "random")
    if name == RESERVED_NAME:
        kind = "default"
    description = str(meta.get("instruct") or meta.get("source") or "")
    if kind == "random" and "seed" in meta:
        description = f"seed {meta['seed']}"
    lang = str(meta.get("lang") or ("hr" if name == RESERVED_NAME else ""))
    return {"name": name, "voice": f"omni:{name}", "kind": kind, "description": description,
            "lang": lang}


def list_preset_info(model_dir: Path) -> list[dict]:
    """preset_info of every listable preset ('default' first)."""
    names = [v.partition(":")[2] for v in list_presets(model_dir)]
    names.sort(key=lambda n: (n != RESERVED_NAME, n))
    return [preset_info(model_dir, n) for n in names]


def parse_voice(voice: str) -> tuple[str | None, str]:
    """(language, preset) of a voice id. 'omni:default' -> (None, 'default'): the language
    comes from the request. Legacy ids carry one: 'omni_hr:default' -> ('hr', 'default'), a
    bare 'omni_hr' -> ('hr', '') (no preset)."""
    head, _, preset = voice.partition(":")
    legacy = head[len(OMNI_PREFIX) + 1:] if head.startswith(OMNI_PREFIX + "_") else ""
    return (legacy or None), preset


def load_preset(model_dir: Path, name: str) -> tuple[np.ndarray, str]:
    """(codes of shape (8, T), text) of a preset."""
    base = _voices_dir(model_dir) / name
    codes = np.load(base.with_suffix(".npy")).astype(np.int64)
    if codes.ndim != 2 or codes.shape[0] != CODEBOOKS or codes.shape[1] < 1:
        raise ValueError(f"voice preset {name!r} must hold {CODEBOOKS} x T codes")
    return codes, base.with_suffix(".txt").read_text(encoding="utf-8").strip()


def encoders_present(model_dir: Path) -> bool:
    """True when the fp32 Higgs encoders (needed to clone from a recording) are installed."""
    d = Path(model_dir) / ENCODERS_DIR
    return all((d / f"{n}.onnx").is_file() for n in ENCODER_NAMES)


# ---------------------------------------------------------------- voice design instruct

# English items of upstream `_resolve_instruct` (accents are left out: they are English accents
# and these voices speak Croatian). One item per category at most.
INSTRUCT_OPTIONS: dict[str, list[str]] = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "style": ["whisper"],
}
_INSTRUCT_VALID = [item for items in INSTRUCT_OPTIONS.values() for item in items]


def validate_instruct(s: str) -> str:
    """Normalise a voice description ('Female,  young adult') to 'female, young adult'.

    Raises ValueError for unknown items (with a "did you mean" hint) and for two items of the
    same category. An empty description gives '' (the model then picks any voice)."""
    items = [x.strip().lower() for x in re.split(r"\s*[,，]\s*", (s or "").strip())]
    items = [x for x in items if x]
    unknown = []
    for n in items:
        if n not in _INSTRUCT_VALID:
            sug = difflib.get_close_matches(n, _INSTRUCT_VALID, n=1, cutoff=0.6)
            unknown.append(f"  '{n}' (unsupported; did you mean '{sug[0]}'?)" if sug
                           else f"  '{n}' (unsupported)")
    if unknown:
        raise ValueError("Unsupported voice description items:\n" + "\n".join(unknown)
                         + "\n\nValid items: " + ", ".join(_INSTRUCT_VALID))
    items = list(dict.fromkeys(items))  # a repeated item is not a conflict
    conflicts = []
    for cat, options in INSTRUCT_OPTIONS.items():
        hits = [n for n in items if n in options]
        if len(hits) > 1:
            conflicts.append(f"{cat}: " + " vs ".join(f"'{x}'" for x in hits))
    if conflicts:
        raise ValueError("Conflicting voice description items in the same category ("
                         + "; ".join(conflicts) + "). Use at most one item per category.")
    return ", ".join(items)


def sanitize_name(s: str) -> str:
    """Preset file name: lower case [a-z0-9_-], at most 32 characters. 'default' is reserved.
    Raises ValueError when nothing usable is left."""
    s = (s or "").replace("đ", "dj").replace("Đ", "Dj")  # no decomposition for d-stroke
    ascii_ = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9_-]+", "-", ascii_.lower()).strip("-_")[:MAX_NAME].strip("-_")
    if not name:
        raise ValueError("The voice name needs at least one letter or digit.")
    if name == RESERVED_NAME:
        raise ValueError(f"'{RESERVED_NAME}' is reserved; choose another name.")
    return name


# ---------------------------------------------------------------- voice clips and presets

@dataclass
class VoiceClip:
    """A candidate voice: reference codes + the text they say, ready to be saved as a preset."""
    codes: np.ndarray  # (8, T) int16
    text: str
    audio: np.ndarray  # float32 mono 24 kHz; for a clone this is the decoded codes (preview)
    meta: dict = field(default_factory=dict)  # kind 'design'|'clone', instruct/seed or source
    warning: str = ""  # shown to the user (e.g. the recording was cut)


def _preset_files(model_dir: Path, name: str) -> list[Path]:
    base = _voices_dir(model_dir) / name
    return [base.with_suffix(e) for e in (".npy", ".txt", ".wav", ".json")]


def save_preset(model_dir: Path, name: str, clip: VoiceClip, overwrite: bool = False) -> str:
    """Write a clip as preset `name` (sanitised, never 'default'); returns the voice id.
    Raises ValueError for a bad name and FileExistsError when the preset exists."""
    from .tts import write_wav

    name = sanitize_name(name)
    d = _voices_dir(model_dir)
    if not overwrite and any(f.exists() for f in _preset_files(model_dir, name)):
        raise FileExistsError(f"A voice named '{name}' already exists.")
    d.mkdir(parents=True, exist_ok=True)
    npy, txt, wav, meta = _preset_files(model_dir, name)
    write_wav(wav, clip.audio)
    txt.write_text(clip.text.strip() + "\n", encoding="utf-8")
    meta.write_text(json.dumps({**clip.meta, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(npy, np.asarray(clip.codes).astype(np.int16))  # last: this makes it listable
    return f"omni:{name}"


def delete_preset(model_dir: Path, name: str) -> None:
    """Remove a preset's files. Refuses 'default'; FileNotFoundError if there is no such preset."""
    if name == RESERVED_NAME:
        raise ValueError(f"The '{RESERVED_NAME}' voice cannot be deleted.")
    if not name or name != Path(name).name or name.startswith("."):
        raise ValueError(f"Invalid voice name {name!r}.")
    files = [f for f in _preset_files(model_dir, name) if f.exists()]
    if not files:
        raise FileNotFoundError(f"No voice named '{name}'.")
    for f in files:
        f.unlink()


# ---------------------------------------------------------------- reference recordings

_FMT_PCM, _FMT_FLOAT, _FMT_EXT = 1, 3, 0xFFFE


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """(mono float32 samples, sample rate) of a WAV: PCM 8/16/24/32-bit or 32/64-bit float,
    any channel count (mixed down). Raises ValueError for anything else."""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise ValueError(f"Cannot read {path}: {e}") from e
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Not a WAV file.")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == _FMT_EXT and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]  # the sub-format GUID starts with the tag
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            data = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("The WAV file is incomplete (no fmt or data chunk).")
    tag, ch, rate, bits = fmt
    if ch < 1 or rate < 8000:
        raise ValueError("Unsupported WAV header.")
    if tag == _FMT_PCM and bits in (8, 16, 32):
        step = bits // 8
        x = np.frombuffer(data[:len(data) // step * step], {8: "u1", 16: "<i2", 32: "<i4"}[bits])
        x = ((x.astype(np.float32) - 128) / 128 if bits == 8
             else x.astype(np.float32) / 2 ** (bits - 1))
    elif tag == _FMT_PCM and bits == 24:
        b = np.frombuffer(data[:len(data) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / 2 ** 23
    elif tag == _FMT_FLOAT and bits in (32, 64):
        step = bits // 8
        x = np.frombuffer(data[:len(data) // step * step], "<f4" if bits == 32 else "<f8")
        x = np.nan_to_num(x.astype(np.float32))
    else:
        raise ValueError(f"Unsupported WAV format (tag {tag}, {bits} bit); use 16-bit PCM or float.")
    x = x[:len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(rate)


def _frame_rms(x: np.ndarray, rate: int, win_s: float = 0.02) -> np.ndarray:
    n = max(1, int(rate * win_s))
    k = len(x) // n
    return np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(axis=1)) if k else np.zeros(0)


def trim_silence(x: np.ndarray, rate: int, margin_s: float = 0.1) -> np.ndarray:
    """Cut leading/trailing silence (below 35 dB under the loudest frame), keeping a margin."""
    rms = _frame_rms(x, rate)
    if not len(rms) or rms.max() < 1e-4:
        return x[:0]
    loud = np.where(rms > rms.max() * 10 ** (-35 / 20))[0]
    n, m = int(rate * 0.02), int(rate * margin_s)
    return x[max(0, loud[0] * n - m): min(len(x), (loud[-1] + 1) * n + m)]


# ---------------------------------------------------------------- duration estimate

# Port of upstream omnivoice/utils/duration.py (Apache-2.0, Xiaomi Corp.): speaking time of a
# character relative to a Latin letter, by script.
_WEIGHTS = {"cjk": 3.0, "hangul": 2.5, "kana": 2.2, "ethiopic": 3.0, "yi": 3.0, "indic": 1.8,
            "thai_lao": 1.5, "khmer_myanmar": 1.8, "arabic": 1.5, "hebrew": 1.5, "latin": 1.0,
            "cyrillic": 1.0, "greek": 1.0, "armenian": 1.0, "georgian": 1.0,
            "punctuation": 0.5, "space": 0.2, "digit": 3.5, "mark": 0.0, "default": 1.0}

# (last code point, script) in ascending order, searched with bisect
_RANGES = [
    (0x02AF, "latin"), (0x03FF, "greek"), (0x052F, "cyrillic"), (0x058F, "armenian"),
    (0x05FF, "hebrew"), (0x077F, "arabic"), (0x089F, "arabic"), (0x08FF, "arabic"),
    (0x097F, "indic"), (0x09FF, "indic"), (0x0A7F, "indic"), (0x0AFF, "indic"),
    (0x0B7F, "indic"), (0x0BFF, "indic"), (0x0C7F, "indic"), (0x0CFF, "indic"),
    (0x0D7F, "indic"), (0x0DFF, "indic"), (0x0EFF, "thai_lao"), (0x0FFF, "indic"),
    (0x109F, "khmer_myanmar"), (0x10FF, "georgian"), (0x11FF, "hangul"), (0x137F, "ethiopic"),
    (0x139F, "ethiopic"), (0x13FF, "default"), (0x167F, "default"), (0x169F, "default"),
    (0x16FF, "default"), (0x171F, "default"), (0x173F, "default"), (0x175F, "default"),
    (0x177F, "default"), (0x17FF, "khmer_myanmar"), (0x18AF, "default"), (0x18FF, "default"),
    (0x194F, "indic"), (0x19DF, "indic"), (0x19FF, "khmer_myanmar"), (0x1A1F, "indic"),
    (0x1AAF, "indic"), (0x1B7F, "indic"), (0x1BBF, "indic"), (0x1BFF, "indic"),
    (0x1C4F, "indic"), (0x1C7F, "indic"), (0x1C8F, "cyrillic"), (0x1CBF, "georgian"),
    (0x1CCF, "indic"), (0x1CFF, "indic"), (0x1D7F, "latin"), (0x1DBF, "latin"),
    (0x1DFF, "default"), (0x1EFF, "latin"), (0x309F, "kana"), (0x30FF, "kana"),
    (0x312F, "cjk"), (0x318F, "hangul"), (0x9FFF, "cjk"), (0xA4CF, "yi"), (0xA4FF, "default"),
    (0xA63F, "default"), (0xA69F, "cyrillic"), (0xA6FF, "default"), (0xA7FF, "latin"),
    (0xA82F, "indic"), (0xA87F, "default"), (0xA8DF, "indic"), (0xA8FF, "indic"),
    (0xA92F, "indic"), (0xA95F, "indic"), (0xA97F, "hangul"), (0xA9DF, "indic"),
    (0xA9FF, "khmer_myanmar"), (0xAA5F, "indic"), (0xAA7F, "khmer_myanmar"),
    (0xAADF, "indic"), (0xAAFF, "indic"), (0xAB2F, "ethiopic"), (0xAB6F, "latin"),
    (0xABBF, "default"), (0xABFF, "indic"), (0xD7AF, "hangul"), (0xFAFF, "cjk"),
    (0xFDFF, "arabic"), (0xFE6F, "default"), (0xFEFF, "arabic"), (0xFFEF, "latin"),
]
_BREAKS = [r[0] for r in _RANGES]


@lru_cache(maxsize=4096)
def _char_weight(ch: str) -> float:
    code = ord(ch)
    if 65 <= code <= 90 or 97 <= code <= 122:
        return _WEIGHTS["latin"]
    if code == 32:
        return _WEIGHTS["space"]
    if code == 0x0640:  # Arabic tatweel
        return _WEIGHTS["mark"]
    cat = unicodedata.category(ch)
    if cat[0] == "M":
        return _WEIGHTS["mark"]
    if cat[0] in "PS":
        return _WEIGHTS["punctuation"]
    if cat[0] == "Z":
        return _WEIGHTS["space"]
    if cat[0] == "N":
        return _WEIGHTS["digit"]
    i = bisect.bisect_left(_BREAKS, code)
    if i < len(_RANGES):
        return _WEIGHTS.get(_RANGES[i][1], _WEIGHTS["default"])
    return _WEIGHTS["cjk"] if code > 0x20000 else _WEIGHTS["default"]


def text_weight(text: str) -> float:
    return sum(_char_weight(c) for c in text)


def estimate_frames(text: str, ref_text: str, ref_frames: float, low: float = 50.0,
                    boost: float = 3.0) -> float:
    """Frames of speech for `text`, scaled from a reference (`ref_text` took `ref_frames`).
    Estimates under `low` frames are boosted: very short texts need proportionally more room."""
    ref_w = text_weight(ref_text)
    if ref_frames <= 0 or ref_w == 0:
        return 0.0
    est = text_weight(text) / (ref_w / ref_frames)
    if est < low:
        return low * (est / low) ** (1.0 / boost)
    return est


# ---------------------------------------------------------------- generation maths

def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def cfg_combine(cond: np.ndarray, uncond: np.ndarray, scale: float) -> np.ndarray:
    """Classifier-free guidance on log-probabilities, renormalised."""
    return log_softmax(cond + scale * (cond - uncond))


def unmask_schedule(frames: int, steps: int, t_shift: float = T_SHIFT) -> list[int]:
    """Cells (of frames x codebooks) to unmask per step; sums to all of them."""
    ts = np.linspace(0, 1, steps + 1)
    ts = t_shift * ts / (1 + (t_shift - 1) * ts)  # warp: few cells early, many late
    total = frames * CODEBOOKS
    remaining = total
    plan = []
    for s in range(steps):
        k = remaining if s == steps - 1 else min(math.ceil(total * (ts[s + 1] - ts[s])), remaining)
        plan.append(int(k))
        remaining -= int(k)
    return plan


def post_process(w: np.ndarray, thr_db: float = -50.0, keep_ms: int = 100) -> np.ndarray:
    """Trim edge silence, peak-normalise to 0.5, 0.1 s fade in/out, 0.1 s of padding."""
    idx = np.where(np.abs(w) > 10 ** (thr_db / 20))[0]
    if len(idx):
        keep = SAMPLE_RATE * keep_ms // 1000
        w = w[max(0, idx[0] - keep): idx[-1] + keep]
    peak = np.abs(w).max() if len(w) else 0.0
    if peak > 1e-6:
        w = w / peak * 0.5
    k = min(SAMPLE_RATE // 10, len(w) // 2)
    if k:
        w = w.copy()
        w[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        w[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    pad = np.zeros(SAMPLE_RATE // 10, np.float32)
    return np.concatenate([pad, w.astype(np.float32), pad])


# ---------------------------------------------------------------- engine

def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class OmniVoiceEngine:
    """Synthesizes one chunk of text per call; `sessions`/`tokenizer` can be injected (tests)."""

    def __init__(self, model_dir: Path, steps: int = 6, *, sessions: dict[str, Any] | None = None,
                 tokenizer: Any = None, seed: int = 0,
                 encoder_loader: Callable[[], dict[str, Any]] | None = None):
        self.model_dir = Path(model_dir)
        self.steps = max(1, int(steps))
        self.seed = seed
        self._lock = threading.Lock()
        t0 = time.perf_counter()
        if sessions is None:
            from tokenizers import Tokenizer  # lazy: only needed for this engine

            d = self.model_dir
            tokenizer = Tokenizer.from_file(str(d / TOKENIZER_FILE))
            sessions = {"embed": _session(d / EMBED_FILE), "llm": _session(d / LLM_FILE),
                        "heads": _session(d / HEADS_FILE), "decoder": _session(d / DECODER_FILE)}
        self._tok = tokenizer
        self._encoder_loader = encoder_loader or self._load_encoders
        self._embed, self._llm = sessions["embed"], sessions["llm"]
        self._heads, self._decoder = sessions["heads"], sessions["decoder"]
        self._heads_dtype = (np.float16 if "float16" in self._heads.get_inputs()[0].type
                             else np.float32)
        log.info("loaded OmniVoice in %.2fs", time.perf_counter() - t0)

    def voices(self) -> list[str]:
        return list_presets(self.model_dir)

    def synth(self, text: str, voice: str, speed: float, lang: str | None = None) -> np.ndarray:
        """`lang` is the language id of `text` ('hr', 'en', ...); default English (a legacy
        'omni_hr' voice id keeps Croatian)."""
        legacy, preset = parse_voice(voice)
        lang = lang or legacy or "en"
        ref_codes, ref_text = load_preset(self.model_dir, preset) if preset else (None, "")
        with self._lock:
            return self._synth(text.strip(), lang, ref_codes, ref_text, speed)

    def generate_codes(self, text: str, lang: str = "hr", speed: float = 1.0,
                       ref_codes: np.ndarray | None = None, ref_text: str = "") -> np.ndarray:
        """Codes (8, T) for `text`, without decoding them to audio."""
        with self._lock:
            return self._codes(text.strip(), lang, ref_codes, ref_text, speed)[0]

    def _synth(self, text, lang, ref_codes, ref_text, speed) -> np.ndarray:
        tokens, _ = self._codes(text, lang, ref_codes, ref_text, speed)
        return post_process(self._decode(tokens))

    def _codes(self, text, lang, ref_codes, ref_text, speed, instruct="", seed=None, steps=None):
        if ref_codes is not None:
            est = estimate_frames(text, ref_text, ref_codes.shape[1])
        else:
            est = estimate_frames(text, "Nice to meet you.", 25)  # upstream default reference
        frames = max(1, int(est / max(speed, 0.1)))
        return self._unmask(text, frames, lang, ref_codes, ref_text, instruct, seed, steps), frames

    # -- custom voices (these share the engine, and its lock, with reading)
    def design_clip(self, instruct: str, seed: int, text: str | None = None,
                    lang: str | None = None) -> VoiceClip:
        """A ~3 s voice described in words ('female, young adult, low pitch'), no reference.

        `text` is what the clip says (default: DESIGN_TEXTS[lang]); `lang` is its language
        id (default: DESIGN_LANG for the default text, else 'hr'; 'hr' and 'en' have built-in
        texts). Takes a few seconds (16 steps). Raises ValueError for a bad description."""
        instruct = validate_instruct(instruct)
        if text is None:
            lang = lang or DESIGN_LANG
            if lang not in DESIGN_TEXTS:
                raise ValueError(f"No built-in sample text for language {lang!r}; "
                                 f"choose one of {', '.join(DESIGN_TEXTS)}.")
            text = DESIGN_TEXTS[lang]
        text = text.strip()
        lang = lang or "hr"
        if not text:
            raise ValueError("The sample text is empty.")
        with self._lock:
            codes, _ = self._codes(text, lang, None, "", 1.0, instruct, seed, DESIGN_STEPS)
            audio = post_process(self._decode(codes))
        meta = {"kind": "design", "instruct": instruct, "seed": int(seed), "lang": lang}
        return VoiceClip(codes.astype(np.int16), text, audio, meta)

    def clone_clip(self, wav_path: Path, ref_text: str, lang: str | None = None) -> VoiceClip:
        """A voice cloned from a recording of someone saying `ref_text`.

        The recording is mixed to mono and silence-trimmed; under 2 s or over 8 s of speech is
        rejected (a cut clip would no longer match its transcript).
        The fp32 encoders are loaded for this call only. Raises ValueError (bad WAV, too short,
        empty text) or FileNotFoundError (encoders not installed). Takes a few seconds.
        `lang` (language id of the recording) is only stored with the preset."""
        import soxr

        ref_text = ref_text.strip()
        if not ref_text:
            raise ValueError("The transcript of the recording is empty.")
        x, rate = read_wav(wav_path)
        x = trim_silence(x, rate)
        if len(x) < MIN_CLONE_S * rate:
            raise ValueError(f"The recording is too short ({len(x) / rate:.1f} s of speech); "
                             f"use at least {MIN_CLONE_S:g} s.")
        if len(x) > MAX_CLONE_S * rate:
            raise ValueError(f"The recording is too long ({len(x) / rate:.1f} s of speech); "
                             f"trim it to 3-{MAX_CLONE_S:g} s and type what is said in it.")
        warning = ""
        if rate != SAMPLE_RATE:
            x = soxr.resample(x, rate, SAMPLE_RATE)
        rms = float(np.sqrt((x ** 2).mean()))
        if 1e-6 < rms < 0.1:  # quiet recordings are brought up to a normal level, like upstream
            x = np.clip(x * (0.1 / rms), -1.0, 1.0)
        x = x[:len(x) - len(x) % HOP].astype(np.float32)
        with self._lock:
            enc = self._encoder_loader()
            try:
                codes = self._encode(enc, x)
            finally:
                del enc
                self._release_encoders()
            audio = post_process(self._decode(codes))
        meta = {"kind": "clone", "source": Path(wav_path).name,
                "seconds": round(len(x) / SAMPLE_RATE, 2)}
        if lang:
            meta["lang"] = lang
        if warning:
            meta["warning"] = warning  # also in clip.warning
        return VoiceClip(codes.astype(np.int16), ref_text, audio, meta, warning)

    def _load_encoders(self) -> dict[str, Any]:
        d = self.model_dir / ENCODERS_DIR
        if not encoders_present(self.model_dir):
            raise FileNotFoundError("The voice cloning encoders are not installed; "
                                    "install OmniVoice with voice cloning (Settings > OmniVoice).")
        return {n: _session(d / f"{n}.onnx") for n in ENCODER_NAMES}

    def _release_encoders(self) -> None:
        gc.collect()  # the sessions are only referenced by the caller's local dict

    @staticmethod
    def _encode(enc: dict[str, Any], wav24: np.ndarray) -> np.ndarray:
        """Codes (8, T) of 24 kHz audio via the acoustic, semantic and quantizer encoders."""
        import soxr

        wav16 = soxr.resample(wav24, SAMPLE_RATE, 16000).astype(np.float32)
        a = enc["acoustic_encoder"].run(None, {"waveform_24k": wav24[None, None, :]})[0]
        s = enc["semantic_encoder"].run(None, {"waveform_16k": wav16[None, :]})[0]
        t = min(a.shape[2], s.shape[2])
        c = enc["quantizer_encoder"].run(None, {"acoustic_features": a[:, :, :t],
                                                "semantic_features": s[:, :, :t]})[0]
        return np.asarray(c)[:, 0, :].astype(np.int64)

    # -- prompt
    def _ids(self, s: str) -> list[int]:
        return list(self._tok.encode(s, add_special_tokens=False).ids)

    def _build(self, text: str, frames: int, lang: str, ref_codes, ref_text: str,
               instruct: str = ""):
        """Input ids (1, 8, S), audio mask (1, S) and the index where the target frames start."""
        style = "<|denoise|>" if ref_codes is not None else ""
        style += f"<|lang_start|>{lang or 'None'}<|lang_end|><|instruct_start|>{instruct or 'None'}<|instruct_end|>"
        full = f"{ref_text.strip()} {text}" if ref_text else text
        style_ids, text_ids = self._ids(style), self._ids(f"<|text_start|>{full}<|text_end|>")
        n_pre = len(style_ids) + len(text_ids)
        n_ref = ref_codes.shape[1] if ref_codes is not None else 0
        ids = np.zeros((1, CODEBOOKS, n_pre + n_ref + frames), np.int64)
        ids[0, :, :len(style_ids)] = style_ids  # text rows are copied to all codebooks
        ids[0, :, len(style_ids):n_pre] = text_ids
        if n_ref:
            ids[0, :, n_pre:n_pre + n_ref] = ref_codes
        ids[0, :, n_pre + n_ref:] = MASK_ID
        audio = np.zeros((1, ids.shape[2]), bool)
        audio[0, n_pre:] = True  # reference and target positions are audio, style and text are not
        return ids, audio, n_pre + n_ref

    def _logits(self, ids: np.ndarray, audio: np.ndarray, start: int, frames: int) -> np.ndarray:
        """Float32 logits (8, frames, 1025) for the positions start..start+frames."""
        emb = self._embed.run(None, {"input_ids": ids, "audio_mask": audio})[0]
        hidden = self._llm.run(["hidden_states"],
                               {"inputs_embeds": emb.astype(np.float32, copy=False)})[0]
        # the heads act per position, so only the target positions are projected
        hidden = hidden[:, start:start + frames].astype(self._heads_dtype, copy=False)
        return self._heads.run(None, {"hidden_states": hidden})[0][0].astype(np.float32)

    def _unmask(self, text, frames, lang, ref_codes, ref_text, instruct="", seed=None,
                steps=None) -> np.ndarray:
        ids, audio, start = self._build(text, frames, lang, ref_codes, ref_text, instruct)
        # unconditional pass for guidance: the target frames alone, no text or reference
        uids = np.full((1, CODEBOOKS, frames), MASK_ID, np.int64)
        uaudio = np.ones((1, frames), bool)
        rng = np.random.default_rng(self.seed if seed is None else seed)  # same text, same audio
        tokens = np.full((CODEBOOKS, frames), MASK_ID, np.int64)
        layer = (np.arange(CODEBOOKS) * LAYER_PENALTY)[:, None]
        for k in unmask_schedule(frames, steps or self.steps):
            if k <= 0:
                continue
            cond = log_softmax(self._logits(ids, audio, start, frames))
            uncond = log_softmax(self._logits(uids, uaudio, 0, frames))
            lp = cfg_combine(cond, uncond, CFG_SCALE)
            lp[..., MASK_ID] = -np.inf
            pred = lp.argmax(-1)
            conf = lp.max(-1) - layer  # earlier codebooks first
            gumbel = -np.log(-np.log(rng.random(conf.shape) + 1e-10) + 1e-10)
            conf = conf / POSITION_TEMP + gumbel
            conf = np.where(tokens != MASK_ID, -np.inf, conf)
            top = np.argpartition(-conf.ravel(), k - 1)[:k]
            flat = tokens.ravel().copy()
            flat[top] = pred.ravel()[top]
            tokens = flat.reshape(CODEBOOKS, frames)
            ids[0, :, start:start + frames] = tokens
            uids[0] = tokens
        return tokens

    def _decode(self, tokens: np.ndarray) -> np.ndarray:
        w = self._decoder.run(None, {"codes": tokens[:, None, :]})[0]
        return np.asarray(w, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------- command line

def make_voice(engine: OmniVoiceEngine, name: str, seed: int, text: str = DEFAULT_TEXT,
               lang: str = "hr") -> Path:
    """Generate a short clip without a reference and save it as a preset; returns the .wav path."""
    from .tts import write_wav

    engine.seed = seed
    codes = engine.generate_codes(text, lang)
    d = _voices_dir(engine.model_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    write_wav(d / f"{name}.wav", post_process(engine._decode(codes)))
    meta = {"kind": "seed", "seed": int(seed), "lang": lang,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    np.save(d / f"{name}.npy", codes.astype(np.int16))  # last: this makes it listable
    return d / f"{name}.wav"


def _report_clip(engine: OmniVoiceEngine, name: str, clip: VoiceClip, t0: float) -> None:
    voice = save_preset(engine.model_dir, name, clip, overwrite=True)
    wav = _voices_dir(engine.model_dir) / f"{sanitize_name(name)}.wav"
    print(f"saved {voice} ({len(clip.audio) / SAMPLE_RATE:.1f} s) in "
          f"{time.perf_counter() - t0:.1f}s: {wav}")
    if clip.warning:
        print(f"warning: {clip.warning}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vibetts.omnivoice")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-voice", help="create a voice preset from a short generated clip")
    mk.add_argument("name")
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--text", default=DEFAULT_TEXT)
    mk.add_argument("--steps", type=int, default=16)
    items = "; ".join(f"{k}: {', '.join(v)}" for k, v in INSTRUCT_OPTIONS.items())
    ds = sub.add_parser("design-voice", help="create a voice preset from a description")
    ds.add_argument("name")
    ds.add_argument("--instruct", required=True,
                    help=f"e.g. 'female, young adult, low pitch' (items {items})")
    ds.add_argument("--seed", type=int, default=0)
    ds.add_argument("--text", default=None, help="sentence the clip says (default: built in)")
    ds.add_argument("--lang", default=None, help="language id of the sentence")
    cl = sub.add_parser("clone-voice", help="create a voice preset from a recording (WAV)")
    cl.add_argument("name")
    cl.add_argument("--wav", required=True)
    cl.add_argument("--text", required=True, help="transcript of the recording")
    say = sub.add_parser("say", help="synthesize text, print synth time and real-time factor")
    say.add_argument("text")
    say.add_argument("--voice", default=DEFAULT_VOICE)
    say.add_argument("--steps", type=int, default=6)
    say.add_argument("--speed", type=float, default=1.0)
    say.add_argument("--lang", default="hr", help="language id of the text (default hr)")
    say.add_argument("--out", default="")
    for p in (mk, ds, cl, say):
        p.add_argument("--model-dir", default=str(DEFAULT_DIR))
    a = ap.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = Path(a.model_dir)
    if not models_present(model_dir):
        print(f"OmniVoice model files missing in {model_dir}; install them from Settings > OmniVoice",
              file=sys.stderr)
        return 1
    engine = OmniVoiceEngine(model_dir, getattr(a, "steps", 6))
    if a.cmd in ("design-voice", "clone-voice"):
        t0 = time.perf_counter()
        try:
            sanitize_name(a.name)
            clip = (engine.design_clip(a.instruct, a.seed, a.text, a.lang)
                    if a.cmd == "design-voice" else engine.clone_clip(Path(a.wav), a.text))
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        _report_clip(engine, a.name, clip, t0)
        return 0
    if a.cmd == "make-voice":
        t0 = time.perf_counter()
        wav = make_voice(engine, a.name, a.seed, a.text)
        print(f"saved {wav} in {time.perf_counter() - t0:.1f}s")
        return 0
    t0 = time.perf_counter()
    samples = engine.synth(a.text, a.voice, a.speed, a.lang)
    dt = time.perf_counter() - t0
    audio = len(samples) / SAMPLE_RATE - 0.2  # without the 0.1 s pads
    print(f"synth {dt:.2f}s for {audio:.2f}s of audio, RTF {dt / max(audio, 1e-3):.2f}")
    if a.out:
        from .tts import write_wav

        write_wav(Path(a.out), samples)
        print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
