"""Supertonic 3 speech engine: multilingual ONNX TTS (31 languages), 44.1 kHz mono.

The inference path is a port of the `supertonic` SDK 1.3.1 (MIT License, Copyright (c)
Supertone Inc., https://github.com/supertone-inc/supertonic-py): text normalisation, language
tokens, duration predictor, text encoder, the vector-estimator denoising loop and the vocoder.
Only numpy and onnxruntime are needed; the SDK's downloader, chunking and silence are dropped
because VibeTTS downloads and chunks itself. The model weights are OpenRAIL-M licensed.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from unicodedata import normalize

import numpy as np

from .config import SUPERTONIC_LANGUAGES, SUPERTONIC_PREFIX, SUPERTONIC_VOICES

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
REPO = "Supertone/supertonic-3"
REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"

# (path relative to the model dir, size in bytes) at REVISION
MODEL_FILES: tuple[tuple[str, int], ...] = (
    ("onnx/duration_predictor.onnx", 3_700_147),
    ("onnx/text_encoder.onnx", 36_416_150),
    ("onnx/vector_estimator.onnx", 256_534_781),
    ("onnx/vocoder.onnx", 101_424_195),
    ("onnx/tts.json", 8_253),
    ("onnx/unicode_indexer.json", 277_676),
    ("voice_styles/F1.json", 292_046),
    ("voice_styles/F2.json", 292_423),
    ("voice_styles/F3.json", 290_794),
    ("voice_styles/F4.json", 291_808),
    ("voice_styles/F5.json", 291_479),
    ("voice_styles/M1.json", 291_748),
    ("voice_styles/M2.json", 292_055),
    ("voice_styles/M3.json", 290_198),
    ("voice_styles/M4.json", 291_522),
    ("voice_styles/M5.json", 291_469),
    ("LICENSE", 15_007),
)

MIN_SPEED = 0.7
MAX_SPEED = 2.0
SPEED_FACTOR = 1.05  # the SDK default: app speed 1.0 sounds slow at model speed 1.0
STEPS_RANGE = (1, 100)


def models_present(model_dir: Path) -> bool:
    """True when every model file exists with the expected size."""
    base = Path(model_dir)
    for rel, size in MODEL_FILES:
        try:
            if (base / rel).stat().st_size != size:
                return False
        except OSError:
            return False
    return True


# ---------------------------------------------------------------- text processing

_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+")
_SYMBOLS = {
    "–": "-", "‑": "-", "—": "-", "¯": " ", "_": " ",
    "“": '"', "”": '"', "‘": "'", "’": "'", "´": "'", "`": "'",
    "[": " ", "]": " ", "|": " ", "/": " ", "#": " ", "→": " ", "←": " ",
}
_DECORATIVE = re.compile(r"[♥☆♡©\\]")
_ABBREVIATIONS = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
_PUNCT_SPACING = [(re.compile(re.escape(" " + c)), c) for c in ",.!?;:'"]
_DUP_QUOTES = re.compile(r"([\"'`])\1+")
_WHITESPACE = re.compile(r"\s+")
_ENDING = re.compile(r"[.!?;:,'\")\]}…。」』】〉》›»]$")


def preprocess(text: str, lang: str) -> str:
    """Normalise text and wrap it in the language token: '<hr>Dobar dan.</hr>'."""
    text = normalize("NFKD", text)
    text = _EMOJI.sub("", text)
    for old, new in _SYMBOLS.items():
        text = text.replace(old, new)
    text = _DECORATIVE.sub("", text)
    for old, new in _ABBREVIATIONS.items():
        text = text.replace(old, new)
    for pattern, new in _PUNCT_SPACING:
        text = pattern.sub(new, text)
    text = _DUP_QUOTES.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not _ENDING.search(text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


def model_speed(speed: float) -> float:
    """App speed (1.0 = normal) -> model speed, clamped to the range the model accepts."""
    return min(MAX_SPEED, max(MIN_SPEED, SPEED_FACTOR * float(speed)))


def _load_style(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for key in ("style_ttl", "style_dp"):
        part = data[key]
        out.append(np.array(part["data"], dtype=np.float32).reshape(*part["dims"]))
    return out[0], out[1]


# ---------------------------------------------------------------- engine

class SupertonicEngine:
    """The four Supertonic ONNX graphs on CPU. Voice styles are read on first use."""

    def __init__(self, model_dir: Path, steps: int = 8):
        import onnxruntime as ort  # lazy, like Kokoro

        t0 = time.perf_counter()
        self._dir = Path(model_dir)
        self._steps = max(STEPS_RANGE[0], min(STEPS_RANGE[1], int(steps)))
        self._lock = threading.Lock()
        self._styles: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        cfg = json.loads((self._dir / "onnx" / "tts.json").read_text(encoding="utf-8"))
        self._rate = int(cfg["ae"]["sample_rate"])
        self._chunk = int(cfg["ae"]["base_chunk_size"]) * int(cfg["ttl"]["chunk_compress_factor"])
        self._latent_dim = int(cfg["ttl"]["latent_dim"]) * int(cfg["ttl"]["chunk_compress_factor"])
        if self._rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected Supertonic sample rate {self._rate}")

        indexer = json.loads((self._dir / "onnx" / "unicode_indexer.json").read_text(encoding="utf-8"))
        self._indexer = np.array(indexer, dtype=np.int64)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        def load(name: str):
            return ort.InferenceSession(str(self._dir / "onnx" / name), sess_options=opts,
                                        providers=["CPUExecutionProvider"])

        self._dp = load("duration_predictor.onnx")
        self._enc = load("text_encoder.onnx")
        self._est = load("vector_estimator.onnx")
        self._voc = load("vocoder.onnx")
        log.info("loaded Supertonic 3 in %.2fs (%d steps)", time.perf_counter() - t0, self._steps)

    def voices(self) -> list[str]:
        return [SUPERTONIC_PREFIX + v for v in SUPERTONIC_VOICES]

    def _style(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        style = self._styles.get(name)
        if style is None:
            style = self._styles[name] = _load_style(self._dir / "voice_styles" / f"{name}.json")
        return style

    def _encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        codes = np.fromiter((ord(c) for c in text), dtype=np.int64, count=len(text))
        codes = codes[codes < len(self._indexer)]
        ids = self._indexer[codes]
        ids = ids[ids >= 0]  # characters the model does not know are dropped
        return ids.reshape(1, -1), np.ones((1, 1, ids.size), dtype=np.float32)

    def synth(self, text: str, voice: str, speed: float, lang: str | None = None) -> np.ndarray:
        name = voice[len(SUPERTONIC_PREFIX):] if voice.startswith(SUPERTONIC_PREFIX) else ""
        if name not in SUPERTONIC_VOICES:
            raise ValueError(f"Not a Supertonic voice: {voice!r}")
        lang = lang or "en"
        if lang not in SUPERTONIC_LANGUAGES:
            raise ValueError(f"Unsupported Supertonic language: {lang!r}")
        spd = model_speed(speed)
        with self._lock:
            ttl, dp = self._style(name)
            ids, mask = self._encode(preprocess(text, lang))
            dur, *_ = self._dp.run(None, {"text_ids": ids, "style_dp": dp, "text_mask": mask})
            dur = dur / spd
            emb, *_ = self._enc.run(None, {"text_ids": ids, "style_ttl": ttl, "text_mask": mask})

            wav_len = (dur * self._rate).astype(np.int64)
            latent_len = max(1, int((float(dur.max()) * self._rate + self._chunk - 1) // self._chunk))
            latent_mask = (np.arange(latent_len)[None, :]
                           < ((wav_len + self._chunk - 1) // self._chunk)[:, None]
                           ).astype(np.float32).reshape(-1, 1, latent_len)
            xt = np.random.randn(1, self._latent_dim, latent_len).astype(np.float32) * latent_mask
            total = np.array([self._steps], dtype=np.float32)
            for step in range(self._steps):
                xt, *_ = self._est.run(None, {
                    "noisy_latent": xt, "text_emb": emb, "style_ttl": ttl, "text_mask": mask,
                    "latent_mask": latent_mask, "current_step": np.array([step], dtype=np.float32),
                    "total_step": total})
            wav, *_ = self._voc.run(None, {"latent": xt})
        return np.asarray(wav, dtype=np.float32).reshape(-1)
