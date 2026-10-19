"""VibeTTS application: hotkey -> selected text -> Kokoro -> speakers, plus tray and settings."""
from __future__ import annotations

import argparse
import copy
import ctypes
import dataclasses
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .config import (
    ROOT,
    Config,
    SpeechConfig,
    config_path,
    export_folder,
    load_config,
    resolve_path,
    save_config,
)
from .logging_setup import setup_logging

if TYPE_CHECKING:  # pragma: no cover
    from .export import ExportProgress, ExportRequest, ExportResult
    from .textprep import Chunk

log = logging.getLogger(__name__)

_MCP_LOCK = threading.RLock()  # serializes _sync_mcp / stop (Tk thread, tray thread, quit)

MISSING_MODEL = "Download the voice model to start reading aloud"


def _short(msg: str, limit: int = 60) -> str:
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


def model_files(s: SpeechConfig) -> tuple[Path, Path]:
    return resolve_path(s.model), resolve_path(s.voices)


def models_present(s: SpeechConfig) -> bool:
    return all(p.is_file() for p in model_files(s))


def plan_speech(s: SpeechConfig, text: str, voice: str | None = None, markdown: bool = True,
                max_chars: int = 0) -> tuple[list[Chunk], str]:
    """Chunks and voice of one request (an explicit `voice`, else speech.voice)."""
    from . import textprep

    return textprep.prepare(text, markdown, max_chars), voice or s.voice


def run_export(engine: Any, s: SpeechConfig, req: ExportRequest, folder: Path,
               progress: Any = None, cancel: Any = None, before_chunk: Any = None,
               fmt_default: str = "wav", bitrate_default: int = 64) -> ExportResult:
    """Blocking: synthesize req.text with `engine` and write the audio file (CLI, Settings
    Export tab and MCP all come through here). Raises ExportCancelled, FileExistsError,
    ValueError (bad format / empty text) and RuntimeError (voice model missing)."""
    from . import export

    chunks, voice = plan_speech(s, req.text, req.voice, req.markdown)
    if not chunks:
        raise ValueError("Nothing to read in this text")
    speed = req.speed or s.speed
    if not models_present(s):
        raise RuntimeError(MISSING_MODEL)
    fmt = req.fmt
    if req.out is not None and str(req.out).strip():
        by_name = export.format_for(req.out, default=fmt or fmt_default)
        if req.fmt and Path(req.out).suffix and by_name != req.fmt:
            raise ValueError(f"Format {req.fmt} does not match the file name {Path(req.out).name}")
        fmt = by_name
    fmt = fmt or fmt_default
    if fmt not in export.FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    if fmt == "mp3" and not export.mp3_available():
        raise RuntimeError("MP3 export is unavailable: the lameenc package is not installed")
    path = export.resolve_output(req.out, folder, req.title, fmt, req.overwrite)
    seconds, size = export.render(
        chunks, lambda text: engine.synth(text, voice, speed), path, fmt,
        pause=s.sentence_pause, gain=req.gain, bitrate=req.bitrate or bitrate_default,
        progress=progress, cancel=cancel, before_chunk=before_chunk)
    return export.ExportResult(path, fmt, seconds, len(chunks), size, voice)


def voices_in_file(path: Path) -> list[str]:
    """Voice names stored in voices-v1.0.bin (an .npz archive) without loading the model."""
    import numpy as np

    from .tts import voice_supported

    with np.load(path) as data:
        return sorted(v for v in data.files if voice_supported(v))


class EngineHolder:
    """Loads the Kokoro engine once (lazily or in the background) and reloads it when the
    model or voices file changes."""

    def __init__(self, on_loading=lambda: None):
        self._lock = threading.Lock()
        self._engine: Any = None
        self._key: tuple[Path, Path] | None = None
        self.on_loading = on_loading

    @property
    def loaded(self) -> Any:
        return self._engine

    def get(self, s: SpeechConfig, announce: bool = True) -> Any:
        from .tts import KokoroEngine

        key = model_files(s)
        with self._lock:
            if self._engine is not None and self._key == key:
                return self._engine
            if not all(p.is_file() for p in key):
                raise RuntimeError(MISSING_MODEL)
            if announce:
                self.on_loading()
            self._engine = None
            engine = KokoroEngine(*key)
            engine.synth("Ready.", s.voice, 1.0)  # warm-up: first run is slower
            self._engine, self._key = engine, key
            return engine

    def preload(self, s: SpeechConfig) -> None:
        """Load the engine in the background."""
        if not models_present(s):
            return

        def run() -> None:
            try:
                self.get(s, announce=False)
            except Exception:
                log.exception("preloading the voice model failed")
        threading.Thread(target=run, name="vibetts-preload", daemon=True).start()


class TTSApp:
    def __init__(self, cfg: Config, start_hidden: bool = False, instance: Any = None):
        from . import autostart
        from .hotkeys import HotkeyListener, parse_chord
        from .overlay import Overlay
        from .tray import Tray, TrayHooks
        from .tts import SoundDeviceOutput, Speaker

        self.cfg = cfg
        self._start_hidden = start_hidden
        self._instance = instance
        self._autostart = autostart
        self.settings: Any = None
        self.enabled = True
        self._grabbing = False
        self._grab_lock = threading.Lock()
        self._current: Any = None  # the SpeakRequest whose callbacks we honour
        self._flash_until = 0.0
        self._quit_lock = threading.Lock()
        self._quit_done = False
        self._startup_errors: list[str] = []
        self._export_lock = threading.Lock()  # one export at a time (app, Settings, MCP)
        self._export_cancel: threading.Event | None = None
        self._mcp: Any = None  # the running McpServer
        self._mcp_key: tuple | None = None  # (enabled, port, max_chars) it was started with
        self._mcp_error: str | None = None

        try:
            chord = parse_chord(cfg.hotkey.read)
        except ValueError as e:
            log.error("invalid hotkey: %s; falling back to alt+win", e)
            self._startup_errors.append(_short(str(e)))
            chord = parse_chord("alt+win")

        self.overlay = Overlay(cfg.ui)
        self.engine = EngineHolder(on_loading=lambda: self.overlay.show(
            "processing", "Loading voice…"))
        self.speaker = Speaker(self._engine_for_speaker, SoundDeviceOutput,
                               on_start=self._on_speak_start, on_done=self._on_speak_done)
        self._speak_cfg: SpeechConfig = cfg.speech
        self.listener = HotkeyListener({"read": chord}, self._on_hotkey, self._on_escape)

        hooks = TrayHooks(
            is_enabled=lambda: self.enabled,
            set_enabled=self._set_enabled,
            is_speaking=lambda: self.speaker.busy,
            stop=self.stop,
            model_missing=lambda: not models_present(self.cfg.speech),
            download_model=lambda: self.open_settings("kokoro", download=True),
            voices=self._voices,
            current_voice=lambda: self.cfg.speech.voice,
            set_voice=self._set_voice,
            current_speed=lambda: self.cfg.speech.speed,
            set_speed=self._set_speed,
            is_autostart=autostart.is_enabled,
            set_autostart=self._set_autostart,
            open_settings=lambda: self.open_settings(),
            open_export=lambda: self.open_settings("export"),
            open_config=self._open_config,
            open_logs=self._open_logs,
            reload_config=self.reload_config,
            quit=self.quit,
        )
        self.tray = Tray(hooks)

    # -- run / quit
    def run(self) -> None:
        log.info("starting; hotkey=%s", self.cfg.hotkey.read)
        try:
            self.overlay.run(on_ready=self._start)
        finally:
            self.quit()

    def _start(self) -> None:
        try:
            self.listener.start()
        except Exception:
            log.exception("failed to start hotkey listener")
            self._startup_errors.append("Hotkey listener failed")
        try:
            self.tray.start()
        except Exception:
            log.exception("failed to start tray")
        try:
            self._autostart.sync()
        except Exception:
            log.exception("autostart sync failed")
        if self._instance is not None:
            try:
                self._instance.watch(lambda: self.open_settings())
            except Exception:
                log.exception("single-instance watcher failed")
        if not models_present(self.cfg.speech):
            self.open_settings("kokoro")  # the Kokoro tab has the Download panel
        else:
            if self.cfg.speech.preload:
                self.engine.preload(self.cfg.speech)
            if self.cfg.ui.show_settings_on_start and not self._start_hidden:
                self.open_settings()
        self._sync_mcp()
        for msg in self._startup_errors:
            self._flash("error", "Error: " + _short(msg, 53))
            self.tray.notify("VibeTTS", msg)
        self._startup_errors.clear()

    def quit(self) -> None:
        with self._quit_lock:
            if self._quit_done:
                return
            self._quit_done = True
        log.info("quitting")
        steps = [self.cancel_export, self._stop_mcp, self.speaker.stop, self.listener.stop,
                 self.tray.stop]
        if self._instance is not None:
            steps.append(self._instance.close)
        steps.append(self.overlay.quit)  # tears down the Tk root and the settings window with it
        for step in steps:
            try:
                step()
            except Exception:
                log.exception("error during quit")

    # -- hotkey (listener thread: keep it quick)
    def _on_hotkey(self, name: str) -> None:
        if name != "read" or not self.enabled:
            return
        if self.speaker.busy:
            self.stop()
            return
        with self._grab_lock:
            if self._grabbing:
                return
            self._grabbing = True
        threading.Thread(target=self._read_selection, name="vibetts-grab", daemon=True).start()

    def _on_escape(self) -> None:
        if self.speaker.busy:
            self.stop()

    def _read_selection(self) -> None:
        from . import selection, textprep
        from .hotkeys import wait_for_chord_release

        try:
            chord = self.listener.chord("read")
            if chord is not None and not wait_for_chord_release(chord, 3.0):
                self._flash("error", "Release the hotkey to read")
                return
            sel = self.cfg.selection
            grab = selection.grab_selection(sel.copy_timeout_ms, sel.restore_clipboard,
                                            sel.fallback_to_clipboard)
            if not grab.text:
                self._flash("ok", "Nothing selected")
                return
            s = self.cfg.speech
            if not models_present(s):
                self._flash("error", "Error: voice model not downloaded")
                self.open_settings("kokoro")
                return
            chunks = textprep.prepare(grab.text, sel.clean_markdown, sel.max_chars)
            if not chunks:
                self._flash("ok", "Nothing to read")
                return
            log.info("reading %d chars in %d chunks with %s%s",
                     sum(len(c.text) for c in chunks), len(chunks), s.voice,
                     " (from clipboard)" if grab.from_clipboard else "")
            self.speak(chunks, s)
        except Exception as e:
            log.exception("read selection failed")
            self._flash("error", "Error: " + _short(str(e) or e.__class__.__name__, 53))
        finally:
            with self._grab_lock:
                self._grabbing = False

    # -- speaking
    def speak(self, chunks: list, s: SpeechConfig, voice: str | None = None) -> Any:
        """Start reading `chunks`; returns the SpeakRequest (its `done` event, `error` and
        `interrupted` tell how it ended). Safe to call from any thread."""
        from .tts import SpeakRequest

        req = SpeakRequest(chunks, voice or s.voice, s.speed, s.sentence_pause, s.volume,
                           s.device)
        self._speak_cfg = s
        self._current = req
        self.overlay.show("processing", "Starting…")
        self.speaker.speak(req)
        return req

    def _engine_for_speaker(self) -> Any:
        return self.engine.get(self._speak_cfg)

    def stop(self) -> None:
        if self.speaker.busy:
            self._current = None
            self.speaker.stop()
            self.tray.set_speaking(False)
            self._flash("ok", "Stopped", 0.8)

    def _on_speak_start(self, req: Any) -> None:
        if req is not self._current:
            return
        self.overlay.show("speaking", "Reading · Esc to stop", ticking=True)
        self.tray.set_speaking(True)

    def _on_speak_done(self, req: Any, error: str | None) -> None:
        if req is not self._current:
            return
        self._current = None
        self.tray.set_speaking(False)
        if error:
            self._flash("error", "Error: " + _short(error, 53))
            if MISSING_MODEL in error:
                self.open_settings("kokoro")
        elif time.monotonic() >= self._flash_until:
            self.overlay.hide()

    # -- export (Settings Export tab, MCP, CLI share run_export)
    def export_audio(self, req: ExportRequest, progress: Any = None,
                     cancel: threading.Event | None = None) -> ExportResult:
        """Blocking (call from a worker thread): write req.text to an audio file. One export
        at a time (ExportBusy); it pauses while the app reads aloud and honours `cancel`
        (ExportCancelled) and cancel_export()."""
        from .export import ExportBusy, ExportProgress

        if not self._export_lock.acquire(blocking=False):
            raise ExportBusy("Another export is running")
        cancel = cancel if cancel is not None else threading.Event()
        self._export_cancel = cancel
        try:
            cfg = copy.deepcopy(self.cfg)  # a snapshot: Settings may change while we run
            s = cfg.speech
            engine = self.engine.get(s, announce=False)
            last = [ExportProgress(0, 0, 0.0)]

            def on_progress(p: ExportProgress) -> None:
                last[0] = p
                if progress is not None:
                    progress(p)

            log.info("export started: %d chars, voice=%s", len(req.text), req.voice or s.voice)
            result = run_export(
                engine, s, req, export_folder(cfg), on_progress, cancel,
                lambda: self._export_yield(last[0], progress, cancel),
                fmt_default=cfg.export.format, bitrate_default=cfg.export.mp3_bitrate)
            log.info("export finished: %.1fs audio, %d bytes (%s)", result.seconds,
                     result.bytes, result.fmt)
            return result
        except BaseException as e:
            log.info("export ended: %s", e.__class__.__name__)
            raise
        finally:
            self._export_cancel = None
            self._export_lock.release()

    def export_batch(self, reqs: list[ExportRequest], progress: Any = None,
                     cancel: threading.Event | None = None) -> list[dict]:
        """Blocking (call from a worker thread): write many short clips under ONE export slot
        (ExportBusy when another export runs). The model is loaded once and stays loaded.
        Returns one {'output', 'duration_s', 'ok', 'error'} per request, in order; a failing
        item does not stop the rest. After a cancel the remaining items report 'cancelled'.
        progress(done, total) is called after each item."""
        from .export import ExportBusy, ExportCancelled, ExportProgress

        if not self._export_lock.acquire(blocking=False):
            raise ExportBusy("Another export is running")
        cancel = cancel if cancel is not None else threading.Event()
        self._export_cancel = cancel
        results: list[dict] = []
        try:
            cfg = copy.deepcopy(self.cfg)
            s = cfg.speech
            idle = ExportProgress(0, 0, 0.0)
            total = len(reqs)
            for i, req in enumerate(reqs):
                row: dict = {"output": str(req.out), "duration_s": 0.0, "ok": False, "error": None}
                if cancel.is_set():
                    row["error"] = "cancelled"
                else:
                    try:
                        engine = self.engine.get(s, announce=False)
                        result = run_export(
                            engine, s, req, export_folder(cfg), None, cancel,
                            lambda: self._export_yield(idle, None, cancel),
                            fmt_default=cfg.export.format, bitrate_default=cfg.export.mp3_bitrate)
                        row.update(output=str(result.path), duration_s=round(result.seconds, 2),
                                   ok=True)
                    except ExportCancelled:
                        row["error"] = "cancelled"
                    except (ValueError, OSError, RuntimeError) as e:
                        row["error"] = str(e)
                    except Exception as e:
                        log.exception("batch item %d failed", i + 1)
                        row["error"] = f"{type(e).__name__}: {e}"
                results.append(row)
                log.info("batch export %d/%d: %s (%.1fs)", i + 1, total,
                         "ok" if row["ok"] else "failed", row["duration_s"])
                if progress is not None:
                    progress(i + 1, total)
            return results
        finally:
            self._export_cancel = None
            self._export_lock.release()

    def _export_yield(self, last: ExportProgress, progress: Any,
                      cancel: threading.Event) -> None:
        """Called before each chunk: reading aloud has priority, so wait while it plays."""
        from .export import ExportCancelled

        paused = False
        while self.speaker.busy:
            if cancel.is_set():
                raise ExportCancelled()
            if not paused:
                paused = True
                if progress is not None:
                    progress(dataclasses.replace(last, paused=True))
            time.sleep(0.1)
        if paused and progress is not None:
            progress(dataclasses.replace(last, paused=False))

    def cancel_export(self) -> None:
        cancel = self._export_cancel
        if cancel is not None:
            cancel.set()

    def _open_folder(self, path: Path) -> None:
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))  # type: ignore[attr-defined]

    # -- MCP server (agents)
    def _sync_mcp(self) -> None:
        """Start, stop or restart the MCP server to match [mcp] (no-op when unchanged)."""
        from . import mcp_server

        m = self.cfg.mcp
        key = (True, int(m.port), int(m.max_chars)) if m.enabled else None
        with _MCP_LOCK:
            if key == self._mcp_key:
                if key is None:
                    self._mcp_error = None
                return
            self._stop_mcp_locked()
            if key is None:
                self._mcp_error = None
                return
            try:
                server = mcp_server.McpServer(
                    mcp_server.build_tools(AgentBridge(self), key[2]), key[1],
                    version=__version__, instructions=mcp_server.INSTRUCTIONS)
                server.start()
            except OSError as e:
                self._mcp_error = f"Port {key[1]} is busy: {e}"
                log.error("MCP server did not start: %s", self._mcp_error)
                self._flash("error", "Error: " + _short(self._mcp_error, 53))
                return
            self._mcp, self._mcp_key, self._mcp_error = server, key, None

    def _stop_mcp_locked(self) -> None:
        server, self._mcp, self._mcp_key = self._mcp, None, None
        if server is not None:
            try:
                server.stop()
            except Exception:
                log.exception("stopping the MCP server failed")

    def _stop_mcp(self) -> None:
        with _MCP_LOCK:
            self._stop_mcp_locked()

    def mcp_status(self) -> dict:
        """{'enabled', 'running', 'url', 'error'} for the Settings Agents tab."""
        server = self._mcp
        running = bool(server is not None and server.running)
        url = server.url if server is not None else f"http://127.0.0.1:{self.cfg.mcp.port}/mcp"
        return {"enabled": bool(self.cfg.mcp.enabled), "running": running, "url": url,
                "error": self._mcp_error or ""}

    # -- overlay helper
    def _flash(self, kind: str, text: str, seconds: float = 1.5) -> None:
        self._flash_until = time.monotonic() + seconds
        try:
            self.overlay.flash(kind, text, seconds)
        except Exception:
            log.exception("overlay flash failed")

    # -- settings window
    def _settings_window(self):
        if self.settings is None:
            from .settings_ui import SettingsHooks, SettingsWindow
            from .tts import list_output_devices
            hooks = SettingsHooks(
                get_config=lambda: copy.deepcopy(self.cfg),
                apply=self._settings_apply,
                is_autostart=self._autostart.is_enabled,
                set_autostart=self._set_autostart,
                set_show_on_start=self._set_show_on_start,
                models_dir=lambda: resolve_path(self.cfg.speech.model).parent,
                voices=self._voices,
                output_devices=list_output_devices,
                play_test=self._play_test,
                stop=self.stop,
                open_logs=self._open_logs,
                open_config=self._open_config,
                load_document=self._load_document,
                export_audio=self.export_audio,
                cancel_export=self.cancel_export,
                mp3_available=self._mp3_available,
                mcp_status=self.mcp_status,
                open_folder=self._open_folder,
                version=__version__,
            )
            self.settings = SettingsWindow(self.overlay.root, hooks, self.overlay.call_soon)
        return self.settings

    @staticmethod
    def _load_document(path: str) -> Any:
        from . import docload  # lazy: pypdf is only imported for PDFs

        return docload.extract_text(path)

    @staticmethod
    def _mp3_available() -> bool:
        from . import export

        return export.mp3_available()

    def open_settings(self, tab: str | None = None, banner: str | None = None,
                      download: bool = False) -> None:
        """Thread-safe: show the settings window on the Tk thread (with `download`, also
        start downloading the missing model files)."""
        def show() -> None:
            try:
                self._settings_window().show(tab, banner, download)
            except Exception:
                log.exception("could not open settings window")
        self.overlay.call_soon(show)

    def _settings_apply(self, cfg: Config) -> str | None:
        save_config(cfg)
        return self.apply_config(cfg, "Settings saved")

    def _play_test(self, cfg: Config, text: str) -> None:
        from . import textprep

        chunks = textprep.prepare(text)
        if not chunks:
            return
        if not models_present(cfg.speech):
            raise RuntimeError(MISSING_MODEL)
        threading.Thread(target=self.speak, args=(chunks, cfg.speech),
                         name="vibetts-test", daemon=True).start()

    def _voices_changed(self) -> None:
        """Refresh the tray menus and the settings window."""
        try:
            self.tray.update_menu()
        except Exception:
            log.debug("tray refresh failed", exc_info=True)
        if self.settings is not None:
            self.overlay.call_soon(self.settings.refresh)

    def _set_show_on_start(self, value: bool) -> None:
        self.cfg.ui.show_settings_on_start = bool(value)
        self._save()

    # -- config
    def reload_config(self) -> None:
        try:
            new = load_config()
        except Exception:
            log.exception("reload failed")
            return
        self.apply_config(new, "Config reloaded")

    def apply_config(self, new: Config, ok_message: str = "Settings saved") -> str | None:
        from .hotkeys import parse_chord

        self.cfg = new
        self.overlay.cfg = new.ui
        self._sync_mcp()  # before any early return: MCP settings apply anyway
        try:
            self.listener.set_bindings({"read": parse_chord(new.hotkey.read)})
        except ValueError as e:
            log.error("invalid hotkey on reload: %s", e)
            self._flash("error", "Error: " + _short(str(e), 53))
            return str(e)
        if not models_present(new.speech):
            return MISSING_MODEL
        if new.speech.preload:
            self.engine.preload(new.speech)  # no-op when the same files are already loaded
        self._flash("ok", ok_message)
        try:
            if self.settings is not None:
                self.overlay.call_soon(self.settings.refresh)
            self.tray.update_menu()
        except Exception:
            log.debug("refresh after apply failed", exc_info=True)
        return None

    def _save(self) -> None:
        try:
            save_config(self.cfg)
        except Exception:
            log.exception("could not save config")

    # -- tray hooks
    def _set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.stop()
        self.tray.set_enabled(self.enabled)

    def _voices(self) -> list[str]:
        """Kokoro voices: from the loaded engine, else read from the voices file."""
        loaded = self.engine.loaded
        if loaded is not None:
            return list(loaded.voices())
        path = resolve_path(self.cfg.speech.voices)
        if path.is_file():
            try:
                return voices_in_file(path)
            except Exception:
                log.exception("could not read %s", path)
        return []

    def _set_voice(self, voice: str) -> None:
        self.cfg.speech.voice = voice
        self._save()
        self._voices_changed()

    def _set_speed(self, speed: float) -> None:
        self.cfg.speech.speed = float(speed)
        self._save()

    def _set_autostart(self, enabled: bool) -> None:
        try:
            if enabled:
                self._autostart.enable()
            else:
                self._autostart.disable()
        except Exception:
            log.exception("autostart change failed")

    def _open_config(self) -> None:
        os.startfile(str(config_path()))  # type: ignore[attr-defined]

    def _open_logs(self) -> None:
        folder = ROOT / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))  # type: ignore[attr-defined]


# ---------------------------------------------------------------- agents (MCP)

class AgentBridge:
    """What the MCP tools do inside the app (mcp_server.AgentApi on top of TTSApp). Methods run
    on the MCP server's request threads."""

    def __init__(self, app: TTSApp):
        self.app = app

    def _check_voice(self, s: SpeechConfig, voice: str | None) -> None:
        from .mcp_server import ToolError

        if not voice:
            return
        if not models_present(s):
            raise ToolError(MISSING_MODEL)
        if voice not in self.app._voices():
            raise ToolError(f"Unknown voice '{voice}'. Call list_voices to get the valid ids.")

    def list_voices(self) -> dict:
        from .tts import voice_label

        voices = [{"id": v, "label": voice_label(v)} for v in self.app._voices()]
        return {"voices": voices, "default_voice": self.app.cfg.speech.voice}

    def synthesize_batch(self, args: dict) -> dict:
        from .export import ExportRequest, format_for
        from .mcp_server import ToolError

        app = self.app
        items = args["items"]
        overwrite = bool(args.get("overwrite", False))
        limit = app.cfg.mcp.max_chars
        results: list[dict | None] = [None] * len(items)
        reqs: list[ExportRequest] = []
        slots: list[int] = []
        for i, it in enumerate(items):
            out = it.get("output")
            err = None
            try:
                if not isinstance(out, str) or not Path(out).is_absolute():
                    raise ValueError("'output' must be an absolute path")
                fmt = format_for(out)  # .wav / .mp3 only
                text = it.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("'text' must be a non-empty string")
                if limit > 0 and len(text) > limit:
                    raise ValueError(f"text is too long ({len(text)} characters); the limit is {limit}")
                voice, speed = it.get("voice"), it.get("speed")
                if voice is not None and not isinstance(voice, str):
                    raise ValueError("'voice' must be a string")
                if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float))
                                          or not 0.5 <= speed <= 2):
                    raise ValueError("'speed' must be a number between 0.5 and 2")
                try:
                    self._check_voice(app.cfg.speech, voice)
                except ToolError as e:
                    raise ValueError(str(e)) from e
            except ValueError as e:
                err = str(e)
            if err is not None:
                results[i] = {"output": out if isinstance(out, str) else "", "duration_s": 0.0,
                              "ok": False, "error": err}
                continue
            reqs.append(ExportRequest(text, Path(out), voice or None, speed, fmt, None, True, 1.0,
                                      overwrite, ""))
            slots.append(i)
        log.info("MCP synthesize_batch: %d items (%d valid)", len(items), len(reqs))
        if reqs:
            for slot, row in zip(slots, app.export_batch(reqs)):
                results[slot] = row
        done = [r for r in results if r is not None]
        return {"results": done, "ok_count": sum(1 for r in done if r["ok"]),
                "failed_count": sum(1 for r in done if not r["ok"]),
                "total_duration_s": round(sum(r["duration_s"] for r in done), 2)}

    def speak(self, text: str, voice: str | None, speed: float | None, wait: bool,
              timeout: float) -> dict:
        from .mcp_server import ToolError

        app = self.app
        if not app.enabled:
            raise ToolError("VibeTTS is turned off (tray menu > Enabled); "
                            "ask the user to turn it on")
        cfg = app.cfg
        s = cfg.speech
        self._check_voice(s, voice)
        chunks, voice = plan_speech(s, text, voice, cfg.selection.clean_markdown,
                                    cfg.selection.max_chars)
        if not chunks:
            raise ToolError("Nothing to read in this text")
        if not models_present(s):
            raise ToolError(MISSING_MODEL)
        if speed:
            s = dataclasses.replace(s, speed=float(speed))
        log.info("MCP speak: %d chars with %s", len(text), voice)
        req = app.speak(chunks, s, voice)
        done = bool(wait and req is not None and req.done.wait(timeout))
        error = req.error if done else None
        interrupted = bool(done and req.interrupted)
        return {"started": True, "completed": bool(done and not interrupted and not error),
                "interrupted": interrupted, "error": error, "voice": voice}

    def stop(self, cancel_export: bool) -> dict:
        app = self.app
        app.stop()
        cancelled = False
        if cancel_export and app._export_cancel is not None:
            app.cancel_export()
            cancelled = True
        return {"stopped": True, "export_cancelled": cancelled}

    def synthesize_to_file(self, args: dict) -> dict:
        from . import docload
        from .export import ExportRequest
        from .mcp_server import ToolError

        app = self.app
        title, markdown = "", True
        if args.get("text") is not None:
            text = args["text"]
        else:
            try:
                doc = docload.extract_text(args["path"])
            except docload.DocumentError as e:
                raise ToolError(str(e)) from e
            text, title, markdown = doc.text, doc.title, doc.markdown
            limit = app.cfg.mcp.max_chars
            if limit > 0 and len(text) > limit:
                raise ToolError(f"The document has {len(text)} characters; the limit is {limit}")
        self._check_voice(app.cfg.speech, args.get("voice"))
        output, fmt = args.get("output"), args.get("format")
        suffix = Path(output).suffix.lower() if output else ""
        if fmt and suffix in (".wav", ".mp3") and suffix[1:] != fmt:
            raise ToolError(f"format '{fmt}' does not match the output file name '{output}'")
        req = ExportRequest(text, Path(output) if output else None, args.get("voice"),
                            args.get("speed"), fmt, None, markdown, 1.0,
                            bool(args.get("overwrite", False)), title)
        log.info("MCP export: %d chars", len(text))
        result = app.export_audio(req)
        return {"path": str(result.path), "format": result.fmt,
                "duration_seconds": round(result.seconds, 2), "chunks": result.chunks,
                "bytes": result.bytes, "voice": result.voice}


# ---------------------------------------------------------------- headless MCP (agents)

class _NoSpeaker:
    busy = False

    def stop(self, wait: float = 0.0) -> None:
        pass


class HeadlessApp:
    """The agent-facing part of TTSApp (voices, exports) without the tray,
    hotkey, overlay or Settings window; what `--mcp-only` serves."""

    _voices = TTSApp._voices
    export_audio = TTSApp.export_audio
    export_batch = TTSApp.export_batch
    _export_yield = TTSApp._export_yield
    cancel_export = TTSApp.cancel_export

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = True
        self.engine = EngineHolder()
        self.speaker = _NoSpeaker()
        self._export_lock = threading.Lock()
        self._export_cancel: threading.Event | None = None

    def stop(self) -> None:
        pass

    def speak(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Speaking aloud is not available in headless (--mcp-only) mode; "
                           "use synthesize_to_file or synthesize_batch")


def run_mcp_only(cfg: Config, port: int | None = None) -> int:
    """Serve the MCP tools without any UI until interrupted (Ctrl+C). MCP is forced on; the
    configured [mcp] port is used unless `port` is given."""
    from . import mcp_server

    m = cfg.mcp
    m.enabled = True
    if port is not None:
        if not 1024 <= port <= 65535:
            print("error: --port must be between 1024 and 65535", file=sys.stderr)
            return 2
        m.port = port
    server = mcp_server.McpServer(
        mcp_server.build_tools(AgentBridge(HeadlessApp(cfg)), int(m.max_chars)),
        int(m.port), version=__version__, instructions=mcp_server.INSTRUCTIONS)
    try:
        server.start()
    except OSError as e:
        print(f"error: port {m.port} is busy (is VibeTTS already running?): {e}", file=sys.stderr)
        return 1
    print(f"VibeTTS MCP server (headless) listening on {server.url}; Ctrl+C to stop", flush=True)
    stop = threading.Event()
    try:
        while not stop.wait(0.5):
            if not server.running:
                return 1
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


# ---------------------------------------------------------------- one-shot CLI

def say(cfg: Config, text: str, out: str | None = None, title: str = "",
        markdown: bool | None = None) -> int:
    """Speak `text` (or write it to an audio file, WAV/MP3 by extension) without the tray;
    for testing and scripts."""
    from . import textprep
    from .export import ExportRequest
    from .tts import SpeakRequest, Speaker

    s = cfg.speech
    if markdown is None:
        markdown = cfg.selection.clean_markdown
    if not models_present(s):
        print(f"error: model files missing ({s.model}, {s.voices}); "
              "run: python -m vibetts.model_setup", file=sys.stderr)
        return 1
    chunks = textprep.prepare(text, markdown)
    if not chunks:
        print("error: nothing to say", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    engine = EngineHolder().get(s, announce=False)  # loads the engine, warms it up
    print(f"model loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    if out:
        req = ExportRequest(text, Path(out), s.voice, None, None, cfg.export.mp3_bitrate,
                            markdown, s.volume, True, title)
        t0 = time.perf_counter()
        try:
            result = run_export(engine, s, req, Path.cwd(), fmt_default=cfg.export.format,
                                bitrate_default=cfg.export.mp3_bitrate)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        took = max(time.perf_counter() - t0, 1e-6)
        print(f"wrote {result.path}: {result.seconds:.2f}s audio in {took:.2f}s "
              f"({result.seconds / took:.1f}x real time)", flush=True)
        return 0
    started = threading.Event()
    errors: list[str] = []
    t0 = time.perf_counter()

    def on_start(req: Any) -> None:
        print(f"first audio after {time.perf_counter() - t0:.2f}s", flush=True)
        started.set()

    speaker = Speaker(lambda: engine, on_start=on_start,
                      on_done=lambda req, err: errors.append(err) if err else None)
    speaker.speak_and_wait(SpeakRequest(chunks, s.voice, s.speed, s.sentence_pause, s.volume,
                                        s.device))
    if errors:
        print(f"error: {errors[0]}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- entry point

def _set_dpi_aware() -> None:
    """Per-monitor DPI awareness; must run before any Tk window exists."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vibetts", description="Read selected text aloud")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--console", action="store_true", help="also log to stderr")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--hidden", action="store_true",
                        help="start in the tray without opening the settings window")
    parser.add_argument("--say", metavar="TEXT", help="speak TEXT and exit (no tray)")
    parser.add_argument("--doc", metavar="PATH",
                        help="speak a document (.txt .md .html .docx .epub .pdf) and exit")
    parser.add_argument("--out", "--wav", dest="out", metavar="PATH",
                        help="with --say / --doc: write an audio file (.wav or .mp3) instead")
    parser.add_argument("--mcp-only", action="store_true",
                        help="headless: serve the MCP tools for agents (no tray, hotkey or UI; "
                             "MCP is forced on) until Ctrl+C")
    parser.add_argument("--port", type=int, metavar="N",
                        help="with --mcp-only: MCP port (default: [mcp] port)")
    args = parser.parse_args(argv)
    if args.port is not None and not args.mcp_only:
        parser.error("--port needs --mcp-only")
    if args.mcp_only and (args.say is not None or args.doc is not None):
        parser.error("--mcp-only cannot be combined with --say or --doc")
    if args.say is not None and args.doc is not None:
        parser.error("use either --say or --doc")
    if args.out and args.say is None and args.doc is None:
        parser.error("--out needs --say TEXT or --doc PATH")
    if args.version:
        print(f"vibetts {__version__}")
        return 0
    if args.config:
        os.environ["VIBETTS_CONFIG"] = str(Path(args.config).resolve())
    inst = None
    try:
        log_path = setup_logging(console=args.console)
        log.info("VibeTTS %s starting; log=%s", __version__, log_path)
        if args.mcp_only:
            return run_mcp_only(load_config(), args.port)
        if args.say is not None:
            return say(load_config(), args.say, args.out)
        if args.doc is not None:
            from . import docload

            try:
                doc = docload.extract_text(args.doc)
            except docload.DocumentError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            return say(load_config(), doc.text, args.out, doc.title, doc.markdown)
        _set_dpi_aware()
        from .single_instance import SingleInstance
        inst = SingleInstance()
        if not inst.acquire():
            log.info("another VibeTTS instance is running; asked it to show settings")
            inst.signal_show()
            inst.close()
            inst = None
            return 0
        cfg = load_config()
        log.info("effective hotkey=%s voice=%s speed=%s", cfg.hotkey.read, cfg.speech.voice,
                 cfg.speech.speed)
        TTSApp(cfg, start_hidden=args.hidden, instance=inst).run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        log.exception("fatal error")
        return 1
    finally:
        if inst is not None:
            try:
                inst.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
