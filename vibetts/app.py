"""VibeTTS application: hotkey -> selected text -> Kokoro -> speakers, plus tray and settings."""
from __future__ import annotations

import argparse
import copy
import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    ROOT,
    Config,
    SpeechConfig,
    config_path,
    load_config,
    resolve_path,
    save_config,
)
from .logging_setup import setup_logging

log = logging.getLogger(__name__)

MISSING_MODEL = "Download the voice model to start reading aloud"


def _short(msg: str, limit: int = 60) -> str:
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


def model_files(s: SpeechConfig) -> tuple[Path, Path]:
    return resolve_path(s.model), resolve_path(s.voices)


def models_present(s: SpeechConfig) -> bool:
    return all(p.is_file() for p in model_files(s))


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
            engine.synth("Ready.", s.voice or "af_heart", 1.0)  # warm-up: first run is slower
            self._engine = engine
            self._key = key
            return self._engine

    def preload(self, s: SpeechConfig) -> None:
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

        try:
            chord = parse_chord(cfg.hotkey.read)
        except ValueError as e:
            log.error("invalid hotkey: %s; falling back to alt+win", e)
            self._startup_errors.append(_short(str(e)))
            chord = parse_chord("alt+win")

        self.overlay = Overlay(cfg.ui)
        self.engine = EngineHolder(on_loading=lambda: self.overlay.show(
            "processing", "Loading voice…"))
        self.speaker = Speaker(lambda: self.engine.get(self._speak_cfg), SoundDeviceOutput,
                               on_start=self._on_speak_start, on_done=self._on_speak_done)
        self._speak_cfg: SpeechConfig = cfg.speech
        self.listener = HotkeyListener({"read": chord}, self._on_hotkey, self._on_escape)

        hooks = TrayHooks(
            is_enabled=lambda: self.enabled,
            set_enabled=self._set_enabled,
            is_speaking=lambda: self.speaker.busy,
            stop=self.stop,
            model_missing=lambda: not models_present(self.cfg.speech),
            download_model=lambda: self.open_settings("voice", download=True),
            voices=self._voices,
            current_voice=lambda: self.cfg.speech.voice,
            set_voice=self._set_voice,
            current_speed=lambda: self.cfg.speech.speed,
            set_speed=self._set_speed,
            is_autostart=autostart.is_enabled,
            set_autostart=self._set_autostart,
            open_settings=lambda: self.open_settings(),
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
            self.open_settings("voice")  # the Voice tab shows a Download panel
        else:
            if self.cfg.speech.preload:
                self.engine.preload(self.cfg.speech)
            if self.cfg.ui.show_settings_on_start and not self._start_hidden:
                self.open_settings()
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
        steps = [self.speaker.stop, self.listener.stop, self.tray.stop]
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
            if not models_present(self.cfg.speech):
                self._flash("error", "Error: voice model not downloaded")
                self.open_settings("voice")
                return
            sel = self.cfg.selection
            grab = selection.grab_selection(sel.copy_timeout_ms, sel.restore_clipboard,
                                            sel.fallback_to_clipboard)
            if not grab.text:
                self._flash("ok", "Nothing selected")
                return
            chunks = textprep.prepare(grab.text, sel.clean_markdown, sel.max_chars)
            if not chunks:
                self._flash("ok", "Nothing to read")
                return
            log.info("reading %d chars in %d chunks%s", sum(len(c.text) for c in chunks),
                     len(chunks), " (from clipboard)" if grab.from_clipboard else "")
            self.speak(chunks, self.cfg.speech)
        except Exception as e:
            log.exception("read selection failed")
            self._flash("error", "Error: " + _short(str(e) or e.__class__.__name__, 53))
        finally:
            with self._grab_lock:
                self._grabbing = False

    # -- speaking
    def speak(self, chunks: list, s: SpeechConfig) -> None:
        from .tts import SpeakRequest

        req = SpeakRequest(chunks, s.voice, s.speed, s.sentence_pause, s.volume, s.device)
        self._speak_cfg = s
        self._current = req
        self.overlay.show("processing", "Starting…")
        self.speaker.speak(req)

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
                self.open_settings("voice")
        elif time.monotonic() >= self._flash_until:
            self.overlay.hide()

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
                version=__version__,
            )
            self.settings = SettingsWindow(self.overlay.root, hooks, self.overlay.call_soon)
        return self.settings

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

        chunks = textprep.prepare(text, True, 0)
        if not chunks:
            return
        if not models_present(cfg.speech):
            raise RuntimeError(MISSING_MODEL)
        threading.Thread(target=self.speak, args=(chunks, cfg.speech), name="vibetts-test",
                         daemon=True).start()

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
        engine = self.engine.loaded
        if engine is not None:
            return engine.voices()
        path = resolve_path(self.cfg.speech.voices)
        if not path.is_file():
            return []
        try:
            return voices_in_file(path)
        except Exception:
            log.exception("could not read %s", path)
            return []

    def _set_voice(self, voice: str) -> None:
        self.cfg.speech.voice = voice
        self._save()

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


# ---------------------------------------------------------------- one-shot CLI

def say(cfg: Config, text: str, wav: str | None = None) -> int:
    """Speak `text` (or write it to a WAV file) without the tray; for testing and scripts."""
    import numpy as np

    from . import textprep
    from .tts import SAMPLE_RATE, KokoroEngine, SpeakRequest, Speaker, write_wav

    s = cfg.speech
    if not models_present(s):
        print(f"error: model files missing ({s.model}, {s.voices}); "
              "run: python -m vibetts.model_setup", file=sys.stderr)
        return 1
    chunks = textprep.prepare(text, cfg.selection.clean_markdown, 0)
    if not chunks:
        print("error: nothing to say", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    engine = KokoroEngine(*model_files(s))
    engine.synth("Ready.", s.voice, 1.0)  # warm-up, as the tray app does when preloading
    print(f"model loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    if wav:
        parts = []
        t0 = time.perf_counter()
        for c in chunks:
            audio = engine.synth(c.text, s.voice, s.speed)
            gap = np.zeros(int(s.sentence_pause * c.pause_after * SAMPLE_RATE), np.float32)
            parts += [audio, gap]
        samples = np.concatenate(parts)
        write_wav(Path(wav), samples * s.volume)
        took = time.perf_counter() - t0
        print(f"wrote {wav}: {len(samples) / SAMPLE_RATE:.2f}s audio in {took:.2f}s "
              f"({len(samples) / SAMPLE_RATE / took:.1f}x real time)", flush=True)
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
    parser.add_argument("--wav", metavar="PATH", help="with --say: write a WAV file instead")
    args = parser.parse_args(argv)
    if args.version:
        print(f"vibetts {__version__}")
        return 0
    if args.config:
        os.environ["VIBETTS_CONFIG"] = str(Path(args.config).resolve())
    inst = None
    try:
        log_path = setup_logging(console=args.console)
        log.info("VibeTTS %s starting; log=%s", __version__, log_path)
        if args.say is not None:
            return say(load_config(), args.say, args.wav)
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
