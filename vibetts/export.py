"""Export speech to an audio file (WAV or MP3), chunk by chunk."""
from __future__ import annotations

import os
import re
import threading
import wave
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .config import EXPORT_FORMATS as FORMATS, MP3_BITRATES  # noqa: F401  (re-exported)
from .textprep import Chunk
from .tts import SAMPLE_RATE


class ExportCancelled(Exception):
    """The export was cancelled; no output file is left behind."""


class ExportBusy(RuntimeError):
    """Another export is already running (one at a time, app-wide)."""


@dataclass
class ExportRequest:
    text: str
    out: Path | None = None        # None / relative / bare name -> resolved under the export folder
    voice: str | None = None       # None = route as configured (route_for)
    speed: float | None = None     # None = speech.speed
    lang: str | None = None        # None = auto ("hr" if the text looks Croatian, else "en")
    fmt: str | None = None         # None = from the out suffix, else [export] format
    bitrate: int | None = None     # None = [export] mp3_bitrate
    markdown: bool = True          # clean markdown symbols before speaking
    gain: float = 1.0              # sample multiplier (the CLI passes speech.volume)
    overwrite: bool = False
    title: str = ""                # default file stem (e.g. the document name)


@dataclass
class ExportProgress:
    done: int                 # chunks written
    total: int                # chunks in the request
    audio_seconds: float      # audio written so far
    paused: bool = False      # waiting while the app reads aloud


@dataclass
class ExportResult:
    path: Path
    fmt: str
    seconds: float
    chunks: int
    bytes: int
    voice: str
    lang: str


class AudioSink(Protocol):
    def write(self, samples: np.ndarray) -> None: ...
    def close(self) -> None: ...
    def abort(self) -> None: ...


def mp3_available() -> bool:
    try:
        import lameenc  # noqa: F401
    except ImportError:
        return False
    return True


def pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def format_for(path: str | Path, default: str | None = None) -> str:
    suffix = Path(path).suffix.lower()
    if not suffix:
        if default:
            return default
        raise ValueError("Cannot tell the format: add a .wav or .mp3 extension")
    if suffix in (".wav", ".mp3"):
        return suffix[1:]
    raise ValueError(f"Only .wav and .mp3 are supported (got {suffix})")


_BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_stem(title: str) -> str:
    stem = _BAD_NAME_CHARS.sub("", title or "")
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem[:120].strip(" .")


def resolve_output(out: str | Path | None, folder: Path, title: str, fmt: str,
                   overwrite: bool = False) -> Path:
    folder = Path(folder)
    if out is None or str(out).strip() == "":
        stem = _safe_stem(title) or datetime.now().strftime("vibetts-%Y%m%d-%H%M%S")
        path = folder / f"{stem}.{fmt}"
    else:
        path = Path(out)
        if not path.is_absolute():
            path = folder / path
        if not path.suffix:
            path = path.with_name(f"{path.name}.{fmt}")
        elif path.suffix.lower() not in (".wav", ".mp3"):
            raise ValueError(f"Only .wav and .mp3 are supported (got {path.suffix})")
    inside = folder.resolve() in (path.parent.resolve(), *path.parent.resolve().parents)
    if inside:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {path.parent}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")
    return path


class WavSink:
    def __init__(self, path: Path):
        self._f = open(path, "wb")
        self._w = wave.open(self._f, "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(SAMPLE_RATE)

    def write(self, samples: np.ndarray) -> None:
        self._w.writeframes(pcm16(samples))

    def close(self) -> None:
        try:
            self._w.close()  # fixes up the header sizes
        finally:
            self._f.close()

    def abort(self) -> None:
        try:
            self._w._file = None  # type: ignore[attr-defined]  # keep wave.__del__ from finalizing
            self._f.close()
        except Exception:
            pass


class Mp3Sink:
    def __init__(self, path: Path, bitrate: int):
        import lameenc
        self._enc = lameenc.Encoder()
        self._enc.set_bit_rate(int(bitrate))
        self._enc.set_in_sample_rate(SAMPLE_RATE)
        self._enc.set_channels(1)
        self._enc.set_quality(2)
        self._f = open(path, "wb")

    def write(self, samples: np.ndarray) -> None:
        self._f.write(self._enc.encode(pcm16(samples)))

    def close(self) -> None:
        try:
            self._f.write(self._enc.flush())
        finally:
            self._f.close()

    def abort(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def open_sink(path: Path, fmt: str, bitrate: int = 64) -> AudioSink:
    if fmt == "wav":
        return WavSink(path)
    if fmt == "mp3":
        return Mp3Sink(path, bitrate)
    raise ValueError(f"Unsupported format: {fmt}")


def render(chunks: list[Chunk], synth: Callable[[str], np.ndarray], out: Path, fmt: str, *,
           pause: float, gain: float = 1.0, bitrate: int = 64,
           progress: Callable[[ExportProgress], None] | None = None,
           cancel: threading.Event | None = None,
           before_chunk: Callable[[], None] | None = None) -> tuple[float, int]:
    if not chunks:
        raise ValueError("nothing to export")
    out = Path(out)
    part = out.with_name(out.name + ".part")
    total = len(chunks)
    frames = 0
    sink = None
    try:
        sink = open_sink(part, fmt, bitrate)
        if progress:
            progress(ExportProgress(0, total, 0.0))
        for i, chunk in enumerate(chunks):
            if before_chunk:
                before_chunk()
            if cancel is not None and cancel.is_set():
                raise ExportCancelled()
            audio = np.asarray(synth(chunk.text), dtype=np.float32) * np.float32(gain)
            sink.write(audio)
            frames += len(audio)
            gap = int(pause * chunk.pause_after * SAMPLE_RATE)
            if gap > 0:
                sink.write(np.zeros(gap, dtype=np.float32))
                frames += gap
            if progress:
                progress(ExportProgress(i + 1, total, frames / SAMPLE_RATE))
        sink.close()
        sink = None
        os.replace(part, out)
    except BaseException:
        if sink is not None:
            sink.abort()
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return frames / SAMPLE_RATE, out.stat().st_size
