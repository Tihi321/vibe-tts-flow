"""Single running instance via a named mutex; a second launch asks the first to show its window."""
from __future__ import annotations

import logging
import threading
from typing import Callable

import pywintypes
import win32api
import win32event
import winerror

log = logging.getLogger(__name__)


class SingleInstance:
    def __init__(self, name: str = "VibeTTS") -> None:
        self._mutex_name = f"Local\\{name}"
        self._event_name = f"Local\\{name}.Show"
        self._mutex = None
        self._event = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> bool:
        """True if this process is the first instance."""
        try:
            self._mutex = win32event.CreateMutex(None, False, self._mutex_name)
        except pywintypes.error as e:
            log.warning("single-instance mutex failed: %s", e)
            return True  # fail open: never block startup
        if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
            win32api.CloseHandle(self._mutex)
            self._mutex = None
            return False
        self._event = win32event.CreateEvent(None, False, False, self._event_name)
        return True

    def signal_show(self) -> None:
        """Ask the running instance to show its window."""
        ev = win32event.CreateEvent(None, False, False, self._event_name)
        try:
            win32event.SetEvent(ev)
        finally:
            win32api.CloseHandle(ev)

    def watch(self, callback: Callable[[], None]) -> None:
        if self._event is None or self._thread is not None:
            return
        event = self._event

        def run() -> None:
            while not self._stop.is_set():
                rc = win32event.WaitForSingleObject(event, 250)
                if rc == win32event.WAIT_OBJECT_0:
                    try:
                        callback()
                    except Exception:
                        log.exception("show-window callback failed")

        self._thread = threading.Thread(target=run, name="single-instance-watch", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        for attr in ("_event", "_mutex"):
            h = getattr(self, attr)
            if h is not None:
                setattr(self, attr, None)
                try:
                    win32api.CloseHandle(h)
                except pywintypes.error:
                    pass
