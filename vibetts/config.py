"""Configuration: dataclasses, TOML load/save, frozen-aware paths."""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

log = logging.getLogger(__name__)


def _app_root() -> Path:
    """Folder holding user data: next to the exe when frozen, else the repo root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _resources_root() -> Path:
    """Folder holding bundled read-only resources (PyInstaller _MEIPASS when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", None) or _app_root())
    return ROOT


ROOT: Path = _app_root()
RESOURCES: Path = _resources_root()


@dataclass
class HotkeyConfig:
    read: str = "alt+win"  # press: read the selection; press again (or Esc): stop


READ_AUTO = "auto"  # speech.read_language: detect the language of each read
SUPERTONIC_PREFIX = "supertonic:"
SUPERTONIC_VOICES = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
SUPERTONIC_STEPS_RANGE = (4, 16)
SUPERTONIC_LANGUAGE_NAMES = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "ar": "Arabic", "bg": "Bulgarian",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "es": "Spanish",
    "et": "Estonian", "fi": "Finnish", "fr": "French", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "lt": "Lithuanian", "lv": "Latvian",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "tr": "Turkish", "uk": "Ukrainian",
    "vi": "Vietnamese",
}
SUPERTONIC_LANGUAGES = tuple(SUPERTONIC_LANGUAGE_NAMES)  # the 31 codes Supertonic 3 reads
# Kokoro voice prefix (af_heart -> "a") -> the language it reads. Japanese (j) is left out
# (espeak cannot read kanji) and so is Chinese (z, Supertonic has no Chinese).
KOKORO_LANGUAGES = {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it", "p": "pt"}
DEFAULT_SUPERTONIC_VOICE = "supertonic:F1"
# The Kokoro voice that reads each Kokoro language when Supertonic is not installed.
DEFAULT_KOKORO_VOICES = {"en": "af_heart", "es": "ef_dora", "fr": "ff_siwis", "hi": "hf_alpha",
                         "it": "if_sara", "pt": "pf_dora"}
DEFAULT_LANGUAGE_VOICES = {c: ("af_heart" if c == "en" else DEFAULT_SUPERTONIC_VOICE)
                           for c in SUPERTONIC_LANGUAGES}


def is_supertonic_voice(voice: str) -> bool:
    """'supertonic:F1' .. 'supertonic:M5'."""
    return voice.startswith(SUPERTONIC_PREFIX) and voice[len(SUPERTONIC_PREFIX):] in SUPERTONIC_VOICES


def kokoro_voice_lang(voice: str) -> str | None:
    """'af_heart' -> 'en', 'ef_dora' -> 'es'; None when it is not a supported Kokoro voice id."""
    if len(voice) > 3 and voice[2:3] == "_":
        return KOKORO_LANGUAGES.get(voice[:1])
    return None


def voice_fits_language(voice: str, lang: str) -> bool:
    """Can `voice` read `lang`? Supertonic voices read all 31; a Kokoro voice only its own."""
    if is_supertonic_voice(voice):
        return lang in SUPERTONIC_LANGUAGES
    return kokoro_voice_lang(voice) == lang


def enabled_languages(s: "SpeechConfig") -> list[str]:
    """Codes whose voice is not Off: English first, then SUPERTONIC_LANGUAGES order."""
    return [c for c in SUPERTONIC_LANGUAGES if s.language_voices.get(c, "")]


@dataclass
class SpeechConfig:
    model: str = "models/kokoro-v1.0.int8.onnx"
    voices: str = "models/voices-v1.0.bin"
    speed: float = 1.0           # 0.5 .. 2.0
    sentence_pause: float = 0.25  # seconds of silence between sentences
    volume: float = 1.0          # 0 .. 1
    device: str = ""             # "" = Windows default output, or part of the device name
    preload: bool = True         # load the model at startup so the first read starts at once
    # language code -> voice: a Kokoro id (af_heart, ef_dora), "supertonic:F1" .. "supertonic:M5",
    # or "" = Off (never detected, never offered)
    language_voices: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LANGUAGE_VOICES))
    read_language: str = "auto"             # auto: detect the language of each read | a language code
    supertonic_dir: str = "models/supertonic-3"  # Supertonic model files, downloaded from Settings
    supertonic_steps: int = 8               # 4 .. 16: fewer = faster, more = better quality


@dataclass
class SelectionConfig:
    copy_timeout_ms: int = 400          # how long to wait for the app to answer Ctrl+C
    fallback_to_clipboard: bool = True  # nothing selected: read the clipboard text instead
    restore_clipboard: bool = True      # put the previous clipboard content back
    clean_markdown: bool = True         # drop markdown symbols, say "link" for URLs
    max_chars: int = 20000              # longer selections are cut


@dataclass
class UiConfig:
    show_pill: bool = True
    pill_position: str = "bottom"  # bottom | top
    show_settings_on_start: bool = True


EXPORT_FORMATS = ("wav", "mp3")
MP3_BITRATES = (32, 48, 64, 96, 128, 160, 192, 256, 320)  # kbps; MPEG-1 Layer III at 44.1 kHz
MCP_DEFAULT_PORT = 7723
READER_DEFAULT_PORT = 7724
EXTENSION_ID_RE = re.compile(r"^[a-p]{32}$")  # Chrome extension ids use the letters a..p
DEFAULT_EXTENSION_ID = "injaiiefkhoimpkeandbjfbnnjddaceh"  # pinned by extension/manifest.json


@dataclass
class ExportConfig:
    folder: str = ""        # "" = ~/Documents/VibeTTS; relative paths are next to the app
    format: str = "mp3"     # wav | mp3
    mp3_bitrate: int = 64   # one of MP3_BITRATES


@dataclass
class McpConfig:
    enabled: bool = False          # local MCP server for AI agents (opt-in)
    port: int = MCP_DEFAULT_PORT   # 1024 .. 65535, listens on 127.0.0.1 only
    max_chars: int = 200000        # longest text one tool call may synthesize; 0 = no limit


@dataclass
class ReaderConfig:
    enabled: bool = True              # local book reader + Chrome extension API (127.0.0.1 only)
    port: int = READER_DEFAULT_PORT   # 1024 .. 65535
    library_folder: str = "library"   # imported books; relative paths are next to the app
    max_chars: int = 500000           # longest text one read request may carry; 0 = no limit
    extension_ids: list[str] = field(default_factory=lambda: [DEFAULT_EXTENSION_ID])


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)


def config_path() -> Path:
    env = os.environ.get("VIBETTS_CONFIG")
    return Path(env) if env else ROOT / "config.toml"


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def export_folder(cfg: Config) -> Path:
    """Default folder for exported audio: [export] folder, else ~/Documents/VibeTTS."""
    if cfg.export.folder.strip():
        return resolve_path(cfg.export.folder.strip())
    return Path.home() / "Documents" / "VibeTTS"


def portable_path(p: str | Path) -> str:
    """Path relative to ROOT (forward slashes) if inside it, else absolute."""
    path = Path(p)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- dict <-> dataclass

def _coerce(value: Any, default: Any) -> Any:
    """Coerce value to the type of default; raise ValueError/TypeError if impossible."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise TypeError("expected a boolean")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected a number")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("expected an integer")
        return int(value)
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("expected a number")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise TypeError("expected a string")
        return value
    if isinstance(default, list):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise TypeError("expected a list of strings")
        return list(value)
    if isinstance(default, dict):
        if not isinstance(value, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise TypeError("expected a table of strings")
        return dict(value)
    raise TypeError("unsupported type")


def _build(cls: type, data: Any, where: str) -> Any:
    obj = cls()
    if not isinstance(data, dict):
        if data is not None:
            log.warning("config: [%s] should be a table, using defaults", where)
        return obj
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        current = getattr(obj, f.name)
        if dataclasses.is_dataclass(current):
            setattr(obj, f.name, _build(type(current), data[f.name], f"{where}.{f.name}"))
            continue
        try:
            setattr(obj, f.name, _coerce(data[f.name], current))
        except (TypeError, ValueError) as e:
            log.warning("config: invalid value for %s.%s (%s); using default %r",
                        where, f.name, e, current)
    return obj


def _validate_speech(s: SpeechConfig) -> None:
    voices = {}
    for code, voice in s.language_voices.items():
        if code not in SUPERTONIC_LANGUAGES:
            log.warning("config: speech.language_voices has unknown language %r; ignored", code)
            continue
        voices[code] = voice
    for code, default in DEFAULT_LANGUAGE_VOICES.items():
        voice = voices.get(code, default)
        if voice != "" and not voice_fits_language(voice, code):
            log.warning("config: speech.language_voices.%s %r is not a voice for that language; "
                        "using %r", code, voice, default)
            voice = default
        voices[code] = voice
    s.language_voices = {c: voices[c] for c in SUPERTONIC_LANGUAGES}
    if not enabled_languages(s):
        log.warning("config: every language is off; turning English on")
        s.language_voices["en"] = DEFAULT_LANGUAGE_VOICES["en"]
    if s.read_language != READ_AUTO and s.read_language not in enabled_languages(s):
        log.warning("config: speech.read_language %r is not an enabled language; using auto",
                    s.read_language)
        s.read_language = READ_AUTO
    lo, hi = SUPERTONIC_STEPS_RANGE
    steps = min(max(int(s.supertonic_steps), lo), hi)
    if steps != s.supertonic_steps:
        log.warning("config: speech.supertonic_steps %d outside %d..%d; using %d",
                    s.supertonic_steps, lo, hi, steps)
        s.supertonic_steps = steps


_OLD_SPEECH_KEYS = ("voice", "english_engine", "english_supertonic_voice", "other_language",
                    "other_voice", "language_mode")


def _migrate_speech(d: dict) -> dict:
    """Old [speech] keys (English profile + one other-language profile) -> language_voices and
    read_language. Returns `d` unchanged when it is already in the new shape."""
    sp = d.get("speech") if isinstance(d, dict) else None
    if not isinstance(sp, dict) or "language_voices" in sp or not any(k in sp for k in _OLD_SPEECH_KEYS):
        return d
    sp = dict(sp)
    voices: dict[str, str] = {}
    en = sp.get("voice", "af_heart")
    if sp.get("english_engine") == "supertonic":
        st = sp.get("english_supertonic_voice", "F1")
        en = SUPERTONIC_PREFIX + str(st)
    if isinstance(en, str):
        voices["en"] = en
    other = sp.get("other_language", "hr")
    if isinstance(other, str) and other in SUPERTONIC_LANGUAGES and other != "en":
        voices[other] = SUPERTONIC_PREFIX + str(sp.get("other_voice", "F1"))
    mode = sp.get("language_mode", "auto")
    if mode == "english":
        sp["read_language"] = "en"
    elif mode == "other" and isinstance(other, str):
        sp["read_language"] = other
    sp["language_voices"] = voices
    for k in _OLD_SPEECH_KEYS:
        sp.pop(k, None)
    log.info("config: migrated the old English / other-language voice settings")
    return {**d, "speech": sp}


def config_from_dict(d: dict) -> Config:
    d = _migrate_speech(d)
    cfg = _build(Config, d, "config")
    _validate_speech(cfg.speech)
    _validate_export(cfg.export)
    _validate_mcp(cfg.mcp)
    _validate_reader(cfg.reader)
    return cfg


def snap_bitrate(kbps: int) -> int:
    """The allowed MP3 bitrate nearest to `kbps`."""
    return min(MP3_BITRATES, key=lambda b: (abs(b - int(kbps)), b))


def _validate_export(e: ExportConfig) -> None:
    fmt = e.format.strip().lower()
    if fmt not in EXPORT_FORMATS:
        log.warning("config: export.format %r is not one of %s; using wav", e.format,
                    ", ".join(EXPORT_FORMATS))
        fmt = "wav"
    e.format = fmt
    snapped = snap_bitrate(e.mp3_bitrate)
    if snapped != e.mp3_bitrate:
        log.warning("config: export.mp3_bitrate %d is not one of %s; using %d",
                    e.mp3_bitrate, MP3_BITRATES, snapped)
        e.mp3_bitrate = snapped


def _validate_mcp(m: McpConfig) -> None:
    if not 1024 <= m.port <= 65535:
        log.warning("config: mcp.port %d out of range 1024..65535; using %d", m.port,
                    MCP_DEFAULT_PORT)
        m.port = MCP_DEFAULT_PORT
    if m.max_chars < 0:
        log.warning("config: mcp.max_chars %d is negative; using 0 (no limit)", m.max_chars)
        m.max_chars = 0


def _validate_reader(r: ReaderConfig) -> None:
    if not 1024 <= r.port <= 65535:
        log.warning("config: reader.port %d out of range 1024..65535; using %d", r.port,
                    READER_DEFAULT_PORT)
        r.port = READER_DEFAULT_PORT
    if r.max_chars < 0:
        log.warning("config: reader.max_chars %d is negative; using 0 (no limit)", r.max_chars)
        r.max_chars = 0
    ids: list[str] = []
    for raw in r.extension_ids:
        ext_id = raw.strip().lower()
        if not EXTENSION_ID_RE.match(ext_id):
            log.warning("config: reader.extension_ids entry %r is not a Chrome extension id "
                        "(32 letters a..p); ignored", raw)
        elif ext_id not in ids:
            ids.append(ext_id)
    r.extension_ids = ids


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


# ---------------------------------------------------------------- load / save

def load_config(path: Path | None = None) -> Config:
    path = Path(path) if path else config_path()
    if not path.exists():
        example = RESOURCES / "config.example.toml"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if example.exists():
                shutil.copyfile(example, path)
            else:
                save_config(Config(), path)
            log.info("created %s", path)
        except OSError as e:
            log.warning("could not create %s: %s", path, e)
            return Config()
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        log.warning("could not read %s (%s); using defaults", path, e)
        return Config()
    return config_from_dict(data)


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = Path(path) if path else config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(config_to_dict(cfg), fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
