"""Reading session: what the web reader and the browser extension are playing.

One session at a time, on top of TTSApp's single playback slot. The clients send a list of
items (sentences / paragraphs they cut themselves); the session tells which item is playing, so
each client can highlight it in its own view. Status changes bump a version counter that
`wait` long-polls on. Text is never logged, only counts.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from . import textprep

log = logging.getLogger(__name__)

STATES = ("idle", "loading", "playing", "paused", "finished", "stopped", "error")
ACTIVE = ("loading", "playing", "paused")
PAUSES = ("sentence", "paragraph")
SOURCES = ("reader", "extension")
MAX_ITEMS = 20_000
MIN_SPEED, MAX_SPEED = 0.5, 2.0


class ReadingError(ValueError):
    """The request is not valid (HTTP 400)."""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class ReadingSession:
    """Thread-safe. `app` is the TTSApp (speak, pause, resume, stop, cfg.speech)."""

    def __init__(self, app: Any):
        self._app = app
        self._cond = threading.Condition()  # guards _st, _version and _gen; long-poll wakeups
        self._ctl = threading.RLock()  # one start / seek / stop at a time
        self._version = 0
        self._gen = 0  # bumped per request: callbacks of an older one are ignored
        self._started = False  # first audio of the current request has played
        self._items: list[tuple[str, str]] = []
        self._chunks: list[textprep.Chunk] = []
        self._st: dict[str, Any] = {
            "session": None, "source": None, "meta": {}, "state": "idle", "item": 0,
            "count": 0, "error": None, "voice": None, "lang": None, "speed": 1.0}

    # -- status
    def _status(self) -> dict:
        return {"version": self._version, **self._st, "meta": dict(self._st["meta"])}

    def status(self) -> dict:
        with self._cond:
            return self._status()

    def _set(self, **changes: Any) -> None:
        """Apply `changes`; a real change bumps the version and wakes the long-polls."""
        with self._cond:
            if all(self._st.get(k) == v for k, v in changes.items()):
                return
            self._st.update(changes)
            self._version += 1
            self._cond.notify_all()

    def wait(self, since: int | None, timeout: float) -> dict:
        """The status once its version differs from `since` (at once if `since` is None or
        stale); after `timeout` seconds the unchanged status."""
        with self._cond:
            if since is not None and since == self._version:
                self._cond.wait_for(lambda: self._version != since, max(0.0, timeout))
            return self._status()

    # -- control
    def start(self, items: list[dict], start: int = 0, voice: str | None = None,
              lang: str | None = None, speed: float | None = None, source: str = "reader",
              meta: dict | None = None, max_chars: int = 500_000) -> dict:
        """Read `items` from index `start` as a new session: {"session", "count"}."""
        from .app import missing_message, note_fallback, route_request, voice_ready

        pairs = self._check_items(items, max_chars)
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(pairs):
            raise ReadingError("start is out of range")
        if speed is not None:
            value = _number(speed)
            if value is None or not MIN_SPEED <= value <= MAX_SPEED:
                raise ReadingError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
            speed = value
        if source not in SOURCES:
            raise ReadingError("source must be 'reader' or 'extension'")
        if meta is not None and not isinstance(meta, dict):
            raise ReadingError("meta must be an object")
        for name, value in (("voice", voice), ("lang", lang)):
            if value is not None and not isinstance(value, str):
                raise ReadingError(f"{name} must be a string")
        s = self._app.cfg.speech
        try:
            route = route_request(s, "\n".join(text for text, _ in pairs), voice or None,
                                  lang or None)
        except ValueError as e:
            raise ReadingError(str(e)) from e
        if not voice_ready(s, route.voice):
            raise RuntimeError(missing_message(route.voice))
        chunks = textprep.prepare_items(pairs)
        if not any(c.tag >= start for c in chunks):
            raise ReadingError("Nothing to read in these items")
        note_fallback(route)
        with self._ctl:
            session = uuid.uuid4().hex
            self._items, self._chunks = pairs, chunks
            self._set(session=session, source=source, meta=dict(meta or {}), count=len(pairs),
                      voice=route.voice, lang=route.lang,
                      speed=speed if speed is not None else s.speed)
            self._speak_from(start, speed)
        log.info("reading session: %d items (%d chars) from %d with %s (%s)", len(pairs),
                 sum(len(text) for text, _ in pairs), start, route.voice, route.lang)
        return {"session": session, "count": len(pairs)}

    def seek(self, item: int) -> dict:
        """Continue the same session at `item`."""
        with self._ctl:
            st = self.status()
            if st["session"] is None:
                raise ReadingError("No reading session")
            if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < st["count"]:
                raise ReadingError("item is out of range")
            if not any(c.tag >= item for c in self._chunks):
                raise ReadingError("Nothing to read from this item")
            self._speak_from(item, st["speed"])
            return self.status()

    def pause(self) -> dict:
        if self.status()["state"] in ACTIVE:
            self._app.pause()  # calls notify_paused
        return self.status()

    def resume(self) -> dict:
        if self.status()["state"] == "paused":
            self._app.resume()
        return self.status()

    def stop(self) -> dict:
        with self._ctl:
            if self.status()["state"] in ACTIVE:
                with self._cond:
                    self._gen += 1  # the watcher of the stopped request stays quiet
                self._app.stop()
                self._set(state="stopped")
            return self.status()

    def notify_paused(self, paused: bool) -> None:
        """TTSApp tells that playback was paused / resumed (tray, hotkey or the API)."""
        state = self.status()["state"]
        if paused and state in ("loading", "playing"):
            self._set(state="paused")
        elif not paused and state == "paused":
            self._set(state="playing" if self._started else "loading")

    # -- internals
    @staticmethod
    def _check_items(items: Any, max_chars: int) -> list[tuple[str, str]]:
        if not isinstance(items, list) or not items:
            raise ReadingError("No items to read")
        if len(items) > MAX_ITEMS:
            raise ReadingError(f"Too many items (more than {MAX_ITEMS})")
        pairs: list[tuple[str, str]] = []
        total = 0
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("text"), str):
                raise ReadingError("Every item needs a text")
            pause = it.get("pause") or "sentence"
            if pause not in PAUSES:
                raise ReadingError("pause must be 'sentence' or 'paragraph'")
            total += len(it["text"])
            pairs.append((it["text"], pause))
        if total > max_chars:
            raise ReadingError(f"Too much text ({total} characters, the limit is {max_chars})")
        return pairs

    def _speak_from(self, index: int, speed: float | None) -> None:
        """(Re)start playback at item `index`; the session id stays."""
        chunks = [c for c in self._chunks if c.tag >= index]
        with self._cond:
            self._gen += 1
            gen = self._gen
        self._started = False
        self._set(state="loading", item=index, error=None)

        def on_chunk(chunk: textprep.Chunk) -> None:
            with self._cond:
                if gen != self._gen:
                    return
                self._started = True
            self._set(item=chunk.tag, state="paused" if self.status()["state"] == "paused"
                      else "playing")

        try:
            req = self._app.speak(chunks, self._app.cfg.speech, self._st["voice"],
                                  self._st["lang"], speed, on_chunk)
        except Exception as e:  # noqa: BLE001
            log.exception("starting the reading session failed")
            self._set(state="error", error=str(e) or e.__class__.__name__)
            return
        threading.Thread(target=self._watch, args=(req, gen), name="vibetts-reading",
                         daemon=True).start()

    def _watch(self, req: Any, gen: int) -> None:
        """Move the state on when `req` ends, unless a newer request took over."""
        req.done.wait()
        with self._cond:
            if gen != self._gen:
                return
        if req.error:
            self._set(state="error", error=req.error)
        elif req.interrupted:
            self._set(state="stopped")  # replaced (hotkey read, MCP speak) or Esc
        else:
            self._set(state="finished")
