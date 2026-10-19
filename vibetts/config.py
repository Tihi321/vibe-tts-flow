"""Configuration: dataclasses, TOML load/save, frozen-aware paths."""
from __future__ import annotations

import dataclasses
import logging
import os
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


@dataclass
class SpeechConfig:
    model: str = "models/kokoro-v1.0.int8.onnx"
    voices: str = "models/voices-v1.0.bin"
    voice: str = "af_heart"
    speed: float = 1.0           # 0.5 .. 2.0
    sentence_pause: float = 0.25  # seconds of silence between sentences
    volume: float = 1.0          # 0 .. 1
    device: str = ""             # "" = Windows default output, or part of the device name
    preload: bool = True         # load the model at startup so the first read starts at once


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
MP3_BITRATES = (32, 48, 64, 96, 128, 160)  # kbps; MPEG-2 Layer III at 24 kHz caps at 160
MCP_DEFAULT_PORT = 7723


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
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


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
    v = s.voice
    if not (len(v) > 3 and v[2:3] == "_"):  # Kokoro ids look like af_heart
        log.warning("config: speech.voice %r is not a Kokoro voice; using %s", v,
                    SpeechConfig().voice)
        s.voice = SpeechConfig().voice


def config_from_dict(d: dict) -> Config:
    cfg = _build(Config, d, "config")
    _validate_speech(cfg.speech)
    _validate_export(cfg.export)
    _validate_mcp(cfg.mcp)
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
