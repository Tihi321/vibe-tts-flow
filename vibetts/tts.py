"""Kokoro speech engine and a streaming speaker.

The speaker generates one chunk ahead on a worker thread while the previous chunk
plays, so audio starts after the first (short) chunk and never waits afterwards:
Kokoro runs about 8x faster than real time on a modern CPU.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .textprep import Chunk

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
BLOCK_FRAMES = 2400  # 100 ms: how quickly Stop takes effect

# voice name prefix -> espeak language. Japanese (j) and Chinese (z) voices need extra
# phonemizer packages that are not bundled, so they are not offered.
VOICE_LANGS = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it",
               "p": "pt-br"}
LANG_LABELS = {"a": "American English", "b": "British English", "e": "Spanish",
               "f": "French", "h": "Hindi", "i": "Italian", "p": "Portuguese (BR)"}


def voice_lang(voice: str) -> str:
    return VOICE_LANGS.get(voice[:1], "en-us")


def voice_supported(voice: str) -> bool:
    return len(voice) > 3 and voice[:1] in VOICE_LANGS and voice[2:3] == "_"


def voice_label(voice: str) -> str:
    """'af_heart' -> 'Heart (American English, female)'."""
    if not voice_supported(voice):
        return voice
    gender = {"f": "female", "m": "male"}.get(voice[1:2], "")
    name = voice[3:].replace("_", " ").title()
    lang = LANG_LABELS.get(voice[:1], "")
    return f"{name} ({lang}, {gender})" if gender else f"{name} ({lang})"


class Engine(Protocol):
    def voices(self) -> list[str]: ...
    def synth(self, text: str, voice: str, speed: float) -> np.ndarray: ...


class KokoroEngine:
    """kokoro-onnx wrapper. Loading takes about a second; synthesis is thread-safe enough
    for one producer (the speaker never synthesizes two chunks at once)."""

    def __init__(self, model_path: Path, voices_path: Path):
        from kokoro_onnx import Kokoro  # lazy: pulls in onnxruntime

        t0 = time.perf_counter()
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self._lock = threading.Lock()
        log.info("loaded %s in %.2fs", Path(model_path).name, time.perf_counter() - t0)

    def voices(self) -> list[str]:
        return [v for v in self._kokoro.get_voices() if voice_supported(v)]

    def synth(self, text: str, voice: str, speed: float) -> np.ndarray:
        with self._lock:
            samples, _ = self._kokoro.create(text, voice=voice, speed=speed,
                                             lang=voice_lang(voice))
        return np.asarray(samples, dtype=np.float32)


# ---------------------------------------------------------------- audio output

class Output(Protocol):
    def write(self, block: np.ndarray) -> None: ...
    def drain(self) -> None: ...
    def abort(self) -> None: ...
    def close(self) -> None: ...


class SoundDeviceOutput:
    def __init__(self, device: str = ""):
        import sounddevice as sd

        self._stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                       device=find_output_device(device))
        self._stream.start()

    def write(self, block: np.ndarray) -> None:
        self._stream.write(block.reshape(-1, 1))

    def drain(self) -> None:
        self._stream.stop()  # returns once the buffered audio has played

    def abort(self) -> None:
        self._stream.abort()

    def close(self) -> None:
        self._stream.close()


def list_output_devices() -> list[str]:
    import sounddevice as sd

    names: list[str] = []
    for d in sd.query_devices():
        if d.get("max_output_channels", 0) > 0 and d["name"] not in names:
            names.append(d["name"])
    return names


def find_output_device(name: str) -> int | None:
    """Index of the first output device whose name contains `name`, None for default."""
    if not name.strip():
        return None
    import sounddevice as sd

    needle = name.strip().lower()
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0 and needle in d["name"].lower():
            return i
    log.warning("output device %r not found; using the default", name)
    return None


def write_wav(path: Path, samples: np.ndarray) -> None:
    from .export import pcm16  # lazy: export imports SAMPLE_RATE from this module

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16(samples))


# ---------------------------------------------------------------- speaker

class SpeakRequest:
    def __init__(self, chunks: list[Chunk], voice: str, speed: float, pause: float,
                 volume: float = 1.0, device: str = ""):
        self.chunks = chunks
        self.voice = voice
        self.speed = speed
        self.pause = pause
        self.volume = volume
        self.device = device
        self.done = threading.Event()  # set once playback has ended (any way)
        self.error: str | None = None  # why it failed (valid once done is set)
        self.interrupted = False  # stopped / replaced before every chunk was played


_END = object()


class Speaker:
    """Plays one SpeakRequest at a time; a new request or stop() ends the current one.

    Callbacks run on worker threads: on_start(req) when the first audio plays,
    on_done(req, error) when playback ends (error is None after a normal end or a stop).
    """

    def __init__(self, engine: Callable[[], Engine],
                 output: Callable[[str], Output] = SoundDeviceOutput,
                 on_start: Callable[[SpeakRequest], None] = lambda req: None,
                 on_done: Callable[[SpeakRequest, str | None], None] = lambda req, err: None):
        self._engine = engine
        self._output = output
        self.on_start = on_start
        self.on_done = on_done
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self._threads: list[threading.Thread] = []

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(t.is_alive() for t in self._threads)

    def stop(self, wait: float = 0.0) -> None:
        with self._lock:
            stop, threads = self._stop, list(self._threads)
        if stop is not None:
            stop.set()
        if wait:
            deadline = time.monotonic() + wait
            for t in threads:
                t.join(max(0.0, deadline - time.monotonic()))

    def speak(self, req: SpeakRequest) -> None:
        self.stop(wait=2.0)
        stop = threading.Event()
        audio: queue.Queue = queue.Queue(maxsize=2)  # one playing, one ready
        producer = threading.Thread(target=self._produce, args=(req, audio, stop),
                                    name="vibetts-synth", daemon=True)
        player = threading.Thread(target=self._play, args=(req, audio, stop),
                                  name="vibetts-play", daemon=True)
        with self._lock:
            self._stop = stop
            self._threads = [producer, player]
        producer.start()
        player.start()

    def speak_and_wait(self, req: SpeakRequest, timeout: float | None = None) -> None:
        self.speak(req)
        with self._lock:
            threads = list(self._threads)
        for t in threads:
            t.join(timeout)

    # -- workers
    @staticmethod
    def _put(q: queue.Queue, item: Any, stop: threading.Event) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self, req: SpeakRequest, q: queue.Queue, stop: threading.Event) -> None:
        try:
            engine = self._engine()
            for chunk in req.chunks:
                if stop.is_set():
                    return
                t0 = time.perf_counter()
                samples = engine.synth(chunk.text, req.voice, req.speed)
                log.debug("synth %d chars -> %.2fs audio in %.2fs", len(chunk.text),
                          len(samples) / SAMPLE_RATE, time.perf_counter() - t0)
                gap = int(req.pause * chunk.pause_after * SAMPLE_RATE)
                if gap > 0:
                    samples = np.concatenate([samples, np.zeros(gap, dtype=np.float32)])
                if not self._put(q, samples, stop):
                    return
        except Exception as e:  # noqa: BLE001
            log.exception("speech synthesis failed")
            self._put(q, e, stop)
            return
        self._put(q, _END, stop)

    def _play(self, req: SpeakRequest, q: queue.Queue, stop: threading.Event) -> None:
        out: Output | None = None
        error: str | None = None
        try:
            while not stop.is_set():
                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _END:
                    break
                if isinstance(item, Exception):
                    error = str(item) or item.__class__.__name__
                    break
                if out is None:
                    out = self._output(req.device)
                    self._safe(lambda: self.on_start(req))
                samples = item * float(req.volume) if req.volume != 1.0 else item
                for i in range(0, len(samples), BLOCK_FRAMES):
                    if stop.is_set():
                        break
                    out.write(samples[i:i + BLOCK_FRAMES])
        except Exception as e:  # noqa: BLE001
            log.exception("audio playback failed")
            error = str(e) or e.__class__.__name__
        finally:
            stopped = stop.is_set()
            interrupted = stopped or error is not None
            req.error, req.interrupted = error, stopped
            stop.set()  # also ends the producer
            if out is not None:
                try:
                    if interrupted:
                        out.abort()
                    else:
                        out.drain()
                    out.close()
                except Exception:
                    log.debug("closing audio output failed", exc_info=True)
            req.done.set()
            self._safe(lambda: self.on_done(req, error))

    @staticmethod
    def _safe(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("speaker callback failed")
