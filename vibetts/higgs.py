"""Higgs Audio v3 TTS engine: multilingual voice-cloning ONNX TTS (4B parameters), 44.1 kHz mono.

The inference path is a port of `inference.py` from the community ONNX conversion
https://huggingface.co/onnx-community/higgs-audio-v3-tts-4b (the `cpu_int4/` graphs of Boson AI's
Higgs Audio v3 TTS 4B): the Qwen3 decoder loop with its KV cache, the 8-codebook delay-pattern
sampler and the neural codec. Only numpy, onnxruntime and `tokenizers` are needed (the original
also pulls in transformers and librosa); WAV clips (integer PCM or float) are parsed by hand and
resampled with soxr.

LICENSE: the weights are under the Boson Higgs Audio v3 Research and Non-Commercial License.
Personal and hobbyist use is allowed; an application or product made available to end users, or any
use that earns revenue, needs a commercial licence from Boson AI. The downloaded model folder keeps
the LICENSE file next to the weights.

Voices are reference clips (a few seconds of speech plus its transcript), one per language:
bundled presets as pre-encoded codes in `vibetts/voices/higgs/<VOICE>/<lang>.codes.npy` (int16
[8, T]) + `<lang>.txt`, and the user's own clips as `<voices_dir>/<name>/<lang>.wav` + `.txt`
whose codes are cached beside them the first time they are used.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import (HIGGS_LANGUAGES, HIGGS_PREFIX, HIGGS_PRESETS, HIGGS_VOICE_NAME_RE,
                     RESOURCES, higgs_voice_name, is_higgs_voice)
from .tts import SynthCancelled  # noqa: F401  (raised by synth(cancel=); defined in tts so the Speaker can catch it)

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000          # the model's own rate; synth() returns 44.1 kHz
OUTPUT_RATE = 44100
REPO = "onnx-community/higgs-audio-v3-tts-4b"
REVISION = "ecc25d6cc6e48909401bfa7ce5ac95e84f41f331"
SUBDIR = "cpu_int4"

# (path relative to the model dir, size in bytes) at REVISION
MODEL_FILES: tuple[tuple[str, int], ...] = (
    (f"{SUBDIR}/audio_embed.onnx", 13_461_914),
    (f"{SUBDIR}/audio_encoder.onnx", 654_423_379),
    (f"{SUBDIR}/audio_heads.onnx", 13_462_729),
    (f"{SUBDIR}/audio_tokenizer.onnx", 86_516_167),
    (f"{SUBDIR}/llm_decoder.onnx", 384_495),
    (f"{SUBDIR}/llm_decoder.onnx.data", 2_291_924_992),
    (f"{SUBDIR}/manifest.json", 2_771),
    (f"{SUBDIR}/text_embed.onnx", 777_912_621),
    (f"{SUBDIR}/tokenizer.json", 11_433_924),
    (f"{SUBDIR}/tokenizer_config.json", 2_044),
    ("LICENSE", 25_685),
)
DOWNLOAD_BYTES = sum(size for _, size in MODEL_FILES)

# A voice without a clip for the requested language borrows the clip of a close language, then
# English. Only used when the voice has no clip of its own for that language (the bundled presets
# have one for each of the 31 Supertonic languages unless a clip was dropped by the ASR check).
CLIP_FALLBACK: dict[str, str] = {
    "bs": "hr", "sr": "hr", "cnr": "hr", "sl": "hr",       # South Slavic neighbours of Croatian
    "mk": "bg", "be": "ru", "sk": "cs", "cs": "sk", "uk": "ru",
    "ca": "es", "gl": "es", "ast": "es", "oc": "fr", "lb": "de",
    "no": "da", "nb": "da", "nn": "da", "af": "nl", "fy": "nl",
}
DEFAULT_CLIP_LANGUAGE = "en"

MIN_CLIP_SECONDS, MAX_CLIP_SECONDS = 3.0, 30.0

# model constants
HEAD_DIM = 128
N_CODEBOOKS = 8
BOC_ID = 1024                 # begin-of-code (delay fill)
EOC_ID = 1025                 # end-of-code
FRAME_SAMPLES = 960           # 24 kHz / 25 frames per second
FRAMES_PER_SECOND = 25
TOP_K = 50
MAX_REPEAT = 32               # codebook-0 repeated this often in a row: the loop has degenerated
MAX_FRAMES = 2000             # the reference default (80 s)

# frame cap and "far fewer than expected" thresholds
FRAMES_PER_CHAR = 2.0         # typical clone read: ~12 characters per second at 25 frames/s
CAP_FRAMES_PER_CHAR = 4.0     # the cap is twice the typical length
CAP_BASE_FRAMES = 75
CJK_CHAR_WEIGHT = 4.0         # one ideograph / kana / hangul carries several Latin characters
SHORT_FRACTION = 0.5          # a repeat-guard stop below this share of the expected frames retries
MIN_FRAMES = 5                # fewer frames than this is never usable
ATTEMPTS = 3

SPEED_TAG_DELTA = 0.15        # |speed - 1| below this adds no tag
SPEED_TAG_STRONG = (0.75, 1.35)

_TAG_FRAGMENTS = re.compile(r"<\||\|>")
_LANG_FILE = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{2,8})?$")


# ---------------------------------------------------------------- text helpers

def _check_cancel(cancel: Any) -> None:
    """Raise SynthCancelled when `cancel` (an Event or a zero-argument callable) says so."""
    if cancel is None:
        return
    done = cancel.is_set() if hasattr(cancel, "is_set") else cancel()
    if done:
        raise SynthCancelled()


def sanitize_text(text: str) -> str:
    """Remove `<|` and `|>` so read text cannot inject Higgs control tags."""
    return " ".join(_TAG_FRAGMENTS.sub("", text).split())


def speed_tag(speed: float) -> str:
    """The (approximate, weak) prosody tag for an app speed, or '' close to normal speed."""
    if abs(speed - 1.0) < SPEED_TAG_DELTA - 1e-9:   # tolerate float noise at exactly 0.15
        return ""
    slow = speed < 1.0
    strong = speed <= SPEED_TAG_STRONG[0] if slow else speed >= SPEED_TAG_STRONG[1]
    return f"<|prosody:speed_{'very_' if strong else ''}{'slow' if slow else 'fast'}|>"


def _weighted_length(text: str) -> float:
    return sum(CJK_CHAR_WEIGHT if ord(c) >= 0x2E80 else 1.0 for c in text)


def max_frames_for(text: str) -> int:
    """Frame cap for a text: generous, so a runaway stops early, never above the reference 2000."""
    return int(min(MAX_FRAMES, CAP_BASE_FRAMES + CAP_FRAMES_PER_CHAR * _weighted_length(text)))


def expected_frames(text: str) -> int:
    return max(MIN_FRAMES, int(FRAMES_PER_CHAR * _weighted_length(text)))


def needs_retry(frames: int, stop: str, text: str) -> bool:
    """True when a result is unusable: (almost) empty, or the repeat guard stopped the loop far
    below the length the text should produce."""
    if frames < MIN_FRAMES:
        return True
    return stop == "repeat" and frames < SHORT_FRACTION * expected_frames(text)


def stable_seed(voice: str, lang: str) -> int:
    """Same voice + language -> same seed in every process (Python's hash() is randomised)."""
    digest = hashlib.sha256(f"{voice}\0{lang}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


# ---------------------------------------------------------------- delay pattern

def apply_delay_pattern(codes_TN: np.ndarray) -> np.ndarray:
    """[T, N] -> [T+N-1, N]: codebook c shifted down by c, the gap is BOC, the tail EOC."""
    T, N = codes_TN.shape
    out = np.full((T + N - 1, N), EOC_ID, dtype=np.int64)
    for c in range(N):
        out[c:c + T, c] = codes_TN[:, c]
        out[:c, c] = BOC_ID
    return out


def reverse_delay_pattern(delayed_LN: np.ndarray) -> np.ndarray:
    """[L, N] -> [L-(N-1), N]: undo the per-codebook shift."""
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        return np.zeros((0, N), dtype=np.int64)
    out = np.zeros((T, N), dtype=np.int64)
    for c in range(N):
        out[:, c] = delayed_LN[c:c + T, c]
    return out


# ---------------------------------------------------------------- model files

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


# ---------------------------------------------------------------- voices and clips

def presets_dir() -> Path:
    """Bundled preset clips: <repo>/vibetts/voices/higgs, or <_MEIPASS>/vibetts/voices/higgs frozen."""
    return RESOURCES / "vibetts" / "voices" / "higgs"


def _bare(voice: str) -> str:
    return higgs_voice_name(voice) if is_higgs_voice(voice) else voice


def _is_preset(name: str) -> bool:
    return name.lower() in {p.lower() for p in HIGGS_PRESETS}


def _voice_dir(name: str, voices_dir: Path) -> Path:
    return (presets_dir() if name in HIGGS_PRESETS else Path(voices_dir)) / name


def _clip_languages(folder: Path, preset: bool) -> list[str]:
    """Languages with a usable clip in `folder` (a transcript plus codes, or plus a WAV)."""
    langs = []
    try:
        entries = sorted(folder.glob("*.txt"))
    except OSError:
        return []
    for txt in entries:
        lang = txt.stem
        if not _LANG_FILE.match(lang):
            continue
        has_codes = (folder / f"{lang}.codes.npy").is_file()
        if has_codes or (not preset and (folder / f"{lang}.wav").is_file()):
            langs.append(lang)
    return langs


def custom_voices(voices_dir: Path) -> list[str]:
    """Names of the user's own voices (sub-folders with at least one clip), sorted."""
    base = Path(voices_dir)
    names = []
    try:
        subs = [p for p in base.iterdir() if p.is_dir()]
    except OSError:
        return []
    for sub in subs:
        if HIGGS_VOICE_NAME_RE.match(sub.name) and not _is_preset(sub.name) \
                and _clip_languages(sub, preset=False):
            names.append(sub.name)
    return sorted(names)


def voice_ids(voices_dir: Path) -> list[str]:
    return [HIGGS_PREFIX + p for p in HIGGS_PRESETS] + \
           [HIGGS_PREFIX + n for n in custom_voices(voices_dir)]


def voice_languages(voice: str, voices_dir: Path) -> list[str]:
    """Languages the voice has an own clip for (`voice` is 'higgs:F1' or 'F1')."""
    name = _bare(voice)
    if not HIGGS_VOICE_NAME_RE.match(name):
        return []
    return _clip_languages(_voice_dir(name, voices_dir), preset=name in HIGGS_PRESETS)


_WAV_PCM, _WAV_FLOAT, _WAV_EXTENSIBLE = 1, 3, 0xFFFE


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """PCM (8/16/24/32-bit) or float (32/64-bit) WAV -> (mono float32, sample rate).
    ValueError with a user-facing message. The RIFF chunks are parsed by hand because the stdlib
    `wave` module reads integer PCM only, and recorders often save 32-bit float."""
    try:
        blob = Path(path).read_bytes()
    except OSError as e:
        raise ValueError(f"Could not read {path}: {e}") from e
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("Not a readable WAV file: no RIFF/WAVE header.")
    fmt = raw = None
    pos = 12
    while pos + 8 <= len(blob):
        cid, size = blob[pos:pos + 4], int.from_bytes(blob[pos + 4:pos + 8], "little")
        body = blob[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            raw = body   # a truncated last chunk keeps what is there
            break
        pos += 8 + size + (size & 1)
    if fmt is None or len(fmt) < 16 or raw is None:
        raise ValueError("Not a readable WAV file: missing fmt or data chunk.")
    tag, channels, rate = (int.from_bytes(fmt[0:2], "little"), int.from_bytes(fmt[2:4], "little"),
                           int.from_bytes(fmt[4:8], "little"))
    bits = int.from_bytes(fmt[14:16], "little")
    if tag == _WAV_EXTENSIBLE and len(fmt) >= 26:
        tag = int.from_bytes(fmt[24:26], "little")   # first two bytes of the sub-format GUID
    if rate <= 0 or channels <= 0:
        raise ValueError("Not a readable WAV file.")
    dtypes = {(_WAV_PCM, 8): "u1", (_WAV_PCM, 16): "<i2", (_WAV_PCM, 32): "<i4",
              (_WAV_FLOAT, 32): "<f4", (_WAV_FLOAT, 64): "<f8"}
    width = (bits + 7) // 8
    raw = raw[:len(raw) - len(raw) % (width * channels)] if width else b""
    if (tag, bits) == (_WAV_PCM, 24):
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        data = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 2**23
    elif (tag, bits) in dtypes:
        data = np.frombuffer(raw, dtype=dtypes[(tag, bits)])
        if tag == _WAV_FLOAT:
            data = np.nan_to_num(data.astype(np.float32))
        elif bits == 8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32) / float(2 ** (bits - 1))
    else:
        raise ValueError("Unsupported WAV encoding: use PCM (8, 16, 24 or 32-bit) or "
                         "32-bit float (re-save the clip as 16-bit WAV).")
    return data.reshape(-1, channels).mean(axis=1), rate


def validate_clip(wav_path: Path) -> float:
    """Length of a usable reference clip in seconds; ValueError with a user-facing message."""
    samples, rate = _read_wav(Path(wav_path))
    seconds = len(samples) / rate
    if seconds < MIN_CLIP_SECONDS:
        raise ValueError(f"The clip is too short ({seconds:.1f} s): use {MIN_CLIP_SECONDS:g} to "
                         f"{MAX_CLIP_SECONDS:g} seconds of clean speech.")
    if seconds > MAX_CLIP_SECONDS:
        raise ValueError(f"The clip is too long ({seconds:.1f} s): use {MIN_CLIP_SECONDS:g} to "
                         f"{MAX_CLIP_SECONDS:g} seconds of clean speech.")
    return seconds


def _check_name(name: str) -> None:
    if not HIGGS_VOICE_NAME_RE.match(name or ""):
        raise ValueError("Voice name: 1 to 32 letters, digits, '-' or '_'.")
    if _is_preset(name):
        raise ValueError(f"'{name}' is a built-in voice name: pick another name.")


def add_custom_voice(voices_dir: Path, name: str, wav_path: Path, transcript: str,
                     lang: str) -> None:
    """Store a clip (WAV + transcript) as `<voices_dir>/<name>/<lang>.wav` + `.txt`.
    Adding again for the same name and language replaces that clip."""
    _check_name(name)
    if lang not in HIGGS_LANGUAGES:
        raise ValueError(f"Language {lang!r} is not supported by Higgs.")
    transcript = " ".join((transcript or "").split())
    if not transcript:
        raise ValueError("The transcript is empty: type what is said in the clip.")
    validate_clip(Path(wav_path))
    folder = Path(voices_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{lang}.wav"
    if Path(wav_path).resolve() != target.resolve():
        shutil.copyfile(wav_path, target)
    (folder / f"{lang}.codes.npy").unlink(missing_ok=True)   # re-encode the new audio
    (folder / f"{lang}.txt").write_text(transcript, encoding="utf-8")


def delete_custom_voice(voices_dir: Path, name: str) -> None:
    """Remove a custom voice with all its clips. A preset cannot be deleted."""
    if _is_preset(name):
        raise ValueError(f"'{name}' is a built-in voice and cannot be deleted.")
    if not HIGGS_VOICE_NAME_RE.match(name or ""):
        raise ValueError("Invalid voice name.")
    shutil.rmtree(Path(voices_dir) / name, ignore_errors=True)


def clip_candidates(lang: str) -> list[str]:
    """Languages to try for `lang`, in order: itself, its CLIP_FALLBACK, English."""
    out = [lang]
    for extra in (CLIP_FALLBACK.get(lang), DEFAULT_CLIP_LANGUAGE):
        if extra and extra not in out:
            out.append(extra)
    return out


# ---------------------------------------------------------------- engine

class HiggsEngine:
    accepts_cancel = True  # synth(..., cancel=) is supported (Speaker and export pass it)

    """The Higgs ONNX graphs on CPU. Sessions are created on first use; the 650 MB
    audio_encoder only to encode a clip that has no cached codes, then released."""

    def __init__(self, model_dir: Path, voices_dir: Path, temperature: float = 0.8,
                 threads: int = 0):
        self._dir = Path(model_dir)
        self._voices_dir = Path(voices_dir)
        self._temperature = float(temperature)
        self._threads = int(threads) if threads and threads > 0 else \
            max(1, min(8, (os.cpu_count() or 2) // 2))
        self._lock = threading.Lock()
        self._warned: set[tuple[str, str]] = set()
        self._loaded = False
        self._sess: dict = {}
        self._tok = None
        self._ids: dict[str, int] = {}

    def voices(self) -> list[str]:
        return voice_ids(self._voices_dir)

    # ---- sessions

    def _model(self, rel: str) -> str:
        return str(self._dir / SUBDIR / rel)

    def _options(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        return ort, opts

    def _load(self) -> None:
        if self._loaded:
            return
        from tokenizers import Tokenizer

        t0 = time.perf_counter()
        manifest = json.loads((self._dir / SUBDIR / "manifest.json").read_text(encoding="utf-8"))
        llm_meta = manifest["sub_models"]["llm_decoder"]
        self._n_layers = int(llm_meta["num_layers"])
        self._n_kv = int(llm_meta["num_kv_heads"])
        ort, opts = self._options()

        def load(name: str):
            return ort.InferenceSession(self._model(name + ".onnx"), sess_options=opts,
                                        providers=["CPUExecutionProvider"])

        self._sess = {n: load(n) for n in ("llm_decoder", "text_embed", "audio_embed",
                                           "audio_heads", "audio_tokenizer")}
        self._out_names = [o.name for o in self._sess["llm_decoder"].get_outputs()]
        self._tok = Tokenizer.from_file(self._model("tokenizer.json"))
        added = {t.content: i for i, t in self._tok.get_added_tokens_decoder().items()}
        for tag in ("<|tts|>", "<|text|>", "<|audio|>", "<|ref_text|>", "<|ref_audio|>"):
            if tag not in added:
                raise RuntimeError(f"Higgs tokenizer lacks {tag}: the model files are wrong.")
        self._ids = added
        self._loaded = True
        log.info("loaded Higgs Audio v3 in %.2fs (%d threads)", time.perf_counter() - t0,
                 self._threads)

    # ---- clips

    def _encode_wav(self, wav_path: Path) -> np.ndarray:
        """WAV -> codes int16 [8, T] through audio_encoder (loaded now, released after)."""
        from .tts import resample

        samples, rate = _read_wav(wav_path)
        x = np.asarray(resample(samples, rate, SAMPLE_RATE), dtype=np.float32).reshape(-1)
        # the graph needs a whole number of codec frames: pad with silence
        pad_to = max(FRAME_SAMPLES, -(-len(x) // FRAME_SAMPLES) * FRAME_SAMPLES)
        x = np.pad(x, (0, pad_to - len(x))).reshape(1, 1, -1)
        ort, opts = self._options()
        encoder = ort.InferenceSession(self._model("audio_encoder.onnx"), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        try:
            codes = encoder.run(None, {"input_values": x})[0]
        finally:
            del encoder
            gc.collect()
        return np.asarray(codes[0], dtype=np.int16)

    def encode_clip(self, wav_path: Path) -> np.ndarray:
        """Codes int16 [8, T] of a reference WAV (used by scripts/make_higgs_presets.py)."""
        with self._lock:
            return self._encode_wav(Path(wav_path))

    def _clip_codes(self, folder: Path, lang: str, preset: bool) -> np.ndarray:
        """Codes [8, T] of folder/<lang>: the cache, or encoded from the WAV (and cached)."""
        cache = folder / f"{lang}.codes.npy"
        wav = folder / f"{lang}.wav"
        if cache.is_file() and not (not preset and wav.is_file()
                                    and wav.stat().st_mtime > cache.stat().st_mtime):
            return np.load(cache)
        codes = self._encode_wav(wav)
        try:
            np.save(cache, codes)
        except OSError as e:
            log.warning("could not cache clip codes %s: %s", cache, e)
        return codes

    def _find_clip(self, name: str, lang: str) -> tuple[np.ndarray, str]:
        """(codes [8, T], transcript) of the best clip of voice `name` for `lang`."""
        preset = name in HIGGS_PRESETS
        folder = _voice_dir(name, self._voices_dir)
        own = _clip_languages(folder, preset)
        if not own:
            raise ValueError(f"Unknown Higgs voice or no clips: {name!r}")
        for cand in clip_candidates(lang):
            if cand in own:
                if cand != lang and (name, lang) not in self._warned:
                    self._warned.add((name, lang))
                    log.warning("Higgs voice %s has no %s clip: using its %s clip", name, lang, cand)
                text = (folder / f"{cand}.txt").read_text(encoding="utf-8").strip()
                return self._clip_codes(folder, cand, preset), text
        # no English/fallback clip either: any clip is better than none
        cand = own[0]
        if (name, lang) not in self._warned:
            self._warned.add((name, lang))
            log.warning("Higgs voice %s has no %s clip: using its %s clip", name, lang, cand)
        text = (folder / f"{cand}.txt").read_text(encoding="utf-8").strip()
        return self._clip_codes(folder, cand, preset), text

    # ---- model steps

    def _embed_ids(self, ids: list[int]) -> np.ndarray:
        return self._sess["text_embed"].run(None, {"input_ids": np.asarray([ids], np.int64)})[0]

    def _prefill(self, text: str, ref_codes: np.ndarray, ref_text: str) -> np.ndarray:
        """<|tts|> <|ref_text|> tok(ref) <|ref_audio|> [ref codes] <|text|> tok(text) <|audio|>."""
        ids = self._ids
        enc = lambda s: self._tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        delayed_ref = apply_delay_pattern(np.asarray(ref_codes).T.astype(np.int64))
        ref_embeds = self._sess["audio_embed"].run(None, {"codes": delayed_ref[None]})[0]
        parts = [
            self._embed_ids([ids["<|tts|>"]]),
            self._embed_ids([ids["<|ref_text|>"], *enc(ref_text)]),
            self._embed_ids([ids["<|ref_audio|>"]]),
            np.asarray(ref_embeds, dtype=np.float32),
            self._embed_ids([ids["<|text|>"], *enc(text)]),
            self._embed_ids([ids["<|audio|>"]]),
        ]
        return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts], axis=1)

    def _empty_past(self) -> dict[str, np.ndarray]:
        z = np.zeros((1, self._n_kv, 0, HEAD_DIM), dtype=np.float32)
        return {f"past_key_values.{i}.{kv}": z
                for i in range(self._n_layers) for kv in ("key", "value")}

    def _llm_step(self, embeds: np.ndarray, attn_len: int, past: dict):
        feeds = {"inputs_embeds": embeds.astype(np.float32),
                 "attention_mask": np.ones((1, attn_len), dtype=np.int64), **past}
        out = dict(zip(self._out_names, self._sess["llm_decoder"].run(None, feeds)))
        hidden = np.asarray(out["hidden_states"], dtype=np.float32)
        new_past = {f"past_key_values.{i}.{kv}": out[f"present.{i}.{kv}"]
                    for i in range(self._n_layers) for kv in ("key", "value")}
        return hidden, new_past

    def _sample(self, logits_NV: np.ndarray, rng) -> np.ndarray:
        """Per-codebook temperature + top-k sampling [N, V] -> [N]."""
        if not np.isfinite(logits_NV).all():
            raise RuntimeError(
                "Higgs produced non-finite (NaN/Inf) values: the INT4 CPU kernels failed on this "
                "machine. Delete and download the Higgs model again, or use another engine.")
        if self._temperature <= 0:
            return logits_NV.argmax(-1).astype(np.int64)
        N, V = logits_NV.shape
        logits = logits_NV.astype(np.float64) / self._temperature
        if 0 < TOP_K < V:
            kth = np.partition(logits, V - TOP_K, axis=-1)[:, V - TOP_K][:, None]
            logits = np.where(logits < kth, -np.inf, logits)
        logits -= logits.max(-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(-1, keepdims=True)
        picks = (np.cumsum(probs, axis=-1) < rng.random((N, 1))).sum(-1)
        return np.minimum(picks, V - 1).astype(np.int64)

    def _generate(self, prefill: np.ndarray, max_frames: int, seed: int, cancel: Any = None):
        """The autoregressive loop. Returns (codes [T, 8] or None, stop, frames). `cancel` is
        checked before the prefill and every frame (SynthCancelled)."""
        rng = np.random.default_rng(seed)
        total = prefill.shape[1]
        _check_cancel(cancel)
        hidden, past = self._llm_step(prefill, total, self._empty_past())
        if float(np.abs(hidden).mean()) < 1e-8:
            raise RuntimeError("Higgs llm_decoder returned all-zero values: the INT4 kernels do "
                               "not work on this machine.")
        delayed: list[np.ndarray] = []
        delay_count = 0
        eoc_countdown = None
        last_cb0, repeat = None, 0
        stop = "cap"
        for _ in range(max_frames):
            _check_cancel(cancel)
            logits = self._sess["audio_heads"].run(None, {"hidden_states": hidden[:, -1:, :]})[0]
            codes = self._sample(np.asarray(logits)[0, 0], rng)
            if codes[0] == last_cb0:
                repeat += 1
                if repeat >= MAX_REPEAT:
                    stop = "repeat"
                    break
            else:
                repeat = 0
            last_cb0 = int(codes[0])
            if delay_count < N_CODEBOOKS:               # delay ramp: later codebooks wait
                if delay_count + 1 < N_CODEBOOKS:
                    codes[delay_count + 1:] = BOC_ID
                delay_count += 1
            elif eoc_countdown is not None:
                eoc_countdown -= 1
                if eoc_countdown <= 0:
                    delayed.append(codes)
                    stop = "eoc"
                    break
            elif int(codes[0]) == EOC_ID:
                eoc_countdown = N_CODEBOOKS - 2
            delayed.append(codes)
            emb = self._sess["audio_embed"].run(None, {"codes": codes[None, None]})[0]
            total += 1
            hidden, past = self._llm_step(emb, total, past)
        if stop == "repeat":
            del delayed[-MAX_REPEAT:]      # the tail is the degenerate buzz that tripped the guard
        if not delayed:
            return None, stop, 0
        codes_TN = reverse_delay_pattern(np.stack(delayed))
        if codes_TN.shape[0] == 0:
            return None, stop, 0
        return np.clip(codes_TN, 0, 1023), stop, codes_TN.shape[0]

    def _decode(self, codes_TN: np.ndarray) -> np.ndarray:
        wav = self._sess["audio_tokenizer"].run(
            None, {"audio_codes": codes_TN.T[None].astype(np.int64)})[0]
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    # ---- public

    def synth(self, text: str, voice: str, speed: float, lang: str | None = None,
              seed_offset: int = 0, cancel: Any = None) -> np.ndarray:
        """Float32 mono 44.1 kHz. `seed_offset` (default 0) shifts the fixed per-voice seed; only
        the preset check script uses it to try another take. `cancel` (a threading.Event or a
        zero-argument callable returning bool) stops a running synthesis within one frame:
        SynthCancelled is raised and the lock is released."""
        from .tts import resample

        name = _bare(voice) if is_higgs_voice(voice) else ""
        if not name:
            raise ValueError(f"Not a Higgs voice: {voice!r}")
        lang = lang or "en"
        if lang not in HIGGS_LANGUAGES:
            raise ValueError(f"Unsupported Higgs language: {lang!r}")
        clean = sanitize_text(text)
        if not clean:
            return np.zeros(0, dtype=np.float32)
        body = speed_tag(float(speed)) + clean
        with self._lock:
            _check_cancel(cancel)  # cancelled while waiting for the lock
            self._load()
            ref_codes, ref_text = self._find_clip(name, lang)
            prefill = self._prefill(body, ref_codes, ref_text)
            cap = max_frames_for(clean)
            seed = stable_seed(voice, lang) + int(seed_offset)
            best, best_frames = None, 0
            for attempt in range(ATTEMPTS):
                codes, stop, frames = self._generate(prefill, cap, seed + attempt, cancel)
                if stop == "cap":
                    log.warning("Higgs hit the frame cap (%d) for %d characters: audio is cut",
                                cap, len(clean))
                if frames > best_frames:
                    best, best_frames = codes, frames
                if not needs_retry(frames, stop, clean):
                    break
                log.warning("Higgs attempt %d/%d unusable (%d frames, stop=%s): retrying with "
                            "another seed", attempt + 1, ATTEMPTS, frames, stop)
            if best is None or best_frames < MIN_FRAMES:
                raise RuntimeError("Higgs produced no speech for this text after "
                                   f"{ATTEMPTS} attempts.")
            _check_cancel(cancel)
            wav = self._decode(best)
        if not np.isfinite(wav).all():
            raise RuntimeError("Higgs codec produced non-finite (NaN/Inf) audio.")
        return resample(wav, SAMPLE_RATE, OUTPUT_RATE)
