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
from typing import Any, NamedTuple

from . import __version__
from .config import (
    OMNI_DEFAULT_VOICE,
    ROOT,
    Config,
    SpeechConfig,
    config_path,
    is_omni_voice,
    load_config,
    normalize_omni_voice,
    resolve_path,
    save_config,
)
from .logging_setup import setup_logging

log = logging.getLogger(__name__)

MISSING_MODEL = "Download the voice model to start reading aloud"
MISSING_OMNI = "OmniVoice is not installed: install it in Settings > OmniVoice"


def _short(msg: str, limit: int = 60) -> str:
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


def model_files(s: SpeechConfig) -> tuple[Path, Path]:
    return resolve_path(s.model), resolve_path(s.voices)


def models_present(s: SpeechConfig) -> bool:
    return all(p.is_file() for p in model_files(s))


def _omni() -> Any:
    """The OmniVoice module, imported on first use (it is only needed for Croatian)."""
    from . import omnivoice

    return omnivoice


def omni_dir(s: SpeechConfig) -> Path:
    return resolve_path(s.omnivoice_dir)


def omni_present(s: SpeechConfig) -> bool:
    try:
        return bool(_omni().models_present(omni_dir(s)))
    except Exception:
        log.debug("OmniVoice check failed", exc_info=True)
        return False


def omni_presets(s: SpeechConfig) -> list[str]:
    """OmniVoice voice ids (omni:<preset>) when OmniVoice is installed, else []."""
    if not omni_present(s):
        return []
    try:
        return list(_omni().list_presets(omni_dir(s)))
    except Exception:
        log.exception("could not list OmniVoice voice presets")
        return []


def omni_preset_info(s: SpeechConfig) -> list[dict]:
    """Rows for the Settings voices list ({name, voice, kind, description, lang})."""
    if not omni_present(s):
        return []
    try:
        return list(_omni().list_preset_info(omni_dir(s)))
    except Exception:
        log.exception("could not read OmniVoice voice presets")
        return []


def voice_ready(s: SpeechConfig, voice: str) -> bool:
    """Are the model files for the engine that speaks `voice` present?"""
    return omni_present(s) if is_omni_voice(voice) else models_present(s)


def missing_message(voice: str) -> str:
    return MISSING_OMNI if is_omni_voice(voice) else MISSING_MODEL


class Route(NamedTuple):
    """How one request is read: the voice, the language of the text ('hr' or 'en': also
    picks the URL word) and whether OmniVoice was wanted but is not installed."""
    voice: str
    lang: str
    fell_back: bool

    @property
    def omni(self) -> bool:
        return is_omni_voice(self.voice)


def omni_voice_of(s: SpeechConfig) -> str:
    return normalize_omni_voice(s.omni_voice) if is_omni_voice(s.omni_voice) \
        else OMNI_DEFAULT_VOICE


def croatian_voice_of(s: SpeechConfig) -> str:
    """The voice for Croatian text: `croatian_voice`, or the OmniVoice voice when it is empty."""
    if s.croatian_voice and is_omni_voice(s.croatian_voice):
        return normalize_omni_voice(s.croatian_voice)
    return omni_voice_of(s)


def kokoro_voice_of(s: SpeechConfig) -> str:
    return s.voice if s.voice and not is_omni_voice(s.voice) else "af_heart"


def choose_voice(s: SpeechConfig, text: str, omni_available: bool) -> Route:
    """Voice and language of one request (decided once, up front):

    - Croatian text (auto_language on and looks_croatian): `croatian_voice` (or `omni_voice`),
      language hr, with either engine, when OmniVoice is installed;
    - other text: the Kokoro `voice` (engine kokoro) or `omni_voice` (engine omnivoice, lang en).

    Without OmniVoice installed everything falls back to the Kokoro voice (`fell_back` is
    True when OmniVoice was wanted: engine omnivoice, or Croatian text)."""
    from .textprep import looks_croatian

    kokoro = kokoro_voice_of(s)
    croatian = bool(s.auto_language and looks_croatian(text))
    if s.engine == "omnivoice":
        if not omni_available:
            return Route(kokoro, "en", True)
        return Route(croatian_voice_of(s), "hr", False) if croatian \
            else Route(omni_voice_of(s), "en", False)
    if croatian:
        if omni_available:
            return Route(croatian_voice_of(s), "hr", False)
        return Route(kokoro, "en", True)
    return Route(kokoro, "en", False)


def route_for(s: SpeechConfig, text: str) -> Route:
    """choose_voice with the OmniVoice check done only when it matters."""
    wanted = s.engine == "omnivoice" or s.auto_language
    return choose_voice(s, text, omni_present(s) if wanted else False)


def active_voice(s: SpeechConfig) -> str:
    """The voice normal (non-Croatian) text is read with: used for preloading and checks."""
    return route_for(dataclasses.replace(s, auto_language=False), "").voice


def voices_in_file(path: Path) -> list[str]:
    """Voice names stored in voices-v1.0.bin (an .npz archive) without loading the model."""
    import numpy as np

    from .tts import voice_supported

    with np.load(path) as data:
        return sorted(v for v in data.files if voice_supported(v))


class ClipEngine:
    """Engine stand-in that 'synthesizes' one ready-made clip (voice preview)."""

    def __init__(self, audio: Any):
        self.audio = audio

    def voices(self) -> list[str]:
        return []

    def synth(self, text: str, voice: str, speed: float, lang: str | None = None):
        return self.audio


class RoutingEngine:
    """Engine protocol over the loaded engines: synth goes to OmniVoice for omni voices,
    to Kokoro for the rest; voices() merges both lists."""

    def __init__(self, kokoro: Any = None, omni: Any = None):
        self.kokoro = kokoro
        self.omni = omni

    def voices(self) -> list[str]:
        out: list[str] = []
        for engine in (self.kokoro, self.omni):
            if engine is not None:
                out += list(engine.voices())
        return out

    def synth(self, text: str, voice: str, speed: float, lang: str | None = None):
        if is_omni_voice(voice):
            if self.omni is None:
                raise RuntimeError(MISSING_OMNI)
            return self.omni.synth(text, voice, speed, lang)
        if self.kokoro is None:
            raise RuntimeError(MISSING_MODEL)
        return self.kokoro.synth(text, voice, speed, lang)


class EngineHolder:
    """Loads one engine per family (Kokoro, OmniVoice) lazily or in the background and
    reloads a family when its files / settings change. get() loads the family that speaks
    `voice` (default: the configured engine's voice) and returns a RoutingEngine over
    everything loaded so far."""

    def __init__(self, on_loading=lambda: None):
        self._lock = threading.Lock()
        self._kokoro: Any = None
        self._kokoro_key: tuple | None = None
        self._omni: Any = None
        self._omni_key: tuple | None = None
        self.on_loading = on_loading

    @property
    def loaded(self) -> Any:
        """A RoutingEngine over the loaded families, or None when nothing is loaded."""
        with self._lock:
            if self._kokoro is None and self._omni is None:
                return None
            return RoutingEngine(self._kokoro, self._omni)

    def get(self, s: SpeechConfig, announce: bool = True, voice: str | None = None) -> Any:
        voice = voice or active_voice(s)
        with self._lock:
            if is_omni_voice(voice):
                self._ensure_omni(s, announce, voice)
            else:
                self._ensure_kokoro(s, voice, announce)
            return RoutingEngine(self._kokoro, self._omni)

    def _ensure_kokoro(self, s: SpeechConfig, voice: str, announce: bool) -> None:
        from .tts import KokoroEngine

        key = model_files(s)
        if self._kokoro is not None and self._kokoro_key == key:
            return
        if not all(p.is_file() for p in key):
            raise RuntimeError(MISSING_MODEL)
        if announce:
            self.on_loading()
        self._kokoro = None
        engine = KokoroEngine(*key)
        engine.synth("Ready.", voice, 1.0)  # warm-up: first run is slower
        self._kokoro, self._kokoro_key = engine, key

    def _ensure_omni(self, s: SpeechConfig, announce: bool, voice: str | None = None) -> None:
        key = (omni_dir(s), int(s.omnivoice_steps))
        if self._omni is not None and self._omni_key == key:
            return
        if not omni_present(s):
            raise RuntimeError(MISSING_OMNI)
        if announce:
            self.on_loading()
        self._omni = None
        engine = _omni().OmniVoiceEngine(key[0], key[1])
        try:
            engine.synth("Ready.", voice if voice and is_omni_voice(voice) else omni_voice_of(s),
                         1.0, "en")  # warm-up: first run is slower
        except Exception:
            log.warning("OmniVoice warm-up failed", exc_info=True)
        self._omni, self._omni_key = engine, key

    def get_omni(self, s: SpeechConfig, announce: bool = True) -> Any:
        """The shared OmniVoice engine, loaded on demand whatever engine is configured
        (used to design / clone voices). Never creates a second engine."""
        with self._lock:
            self._ensure_omni(s, announce)
            return self._omni

    def unload_omni(self) -> None:
        """Drop the OmniVoice engine (before its files are deleted, or after a reinstall)."""
        with self._lock:
            self._omni, self._omni_key = None, None

    def preload(self, s: SpeechConfig) -> None:
        """Load the engine of the configured voice in the background."""
        voice = active_voice(s)
        if not voice_ready(s, voice):
            return

        def run() -> None:
            try:
                self.get(s, announce=False, voice=voice)
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
        self._warned_omni_missing = False
        self._clip_engine: Any = None  # set while a voice preview plays

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
        self._speak_voice: str | None = None  # voice of the request being spoken
        self.listener = HotkeyListener({"read": chord}, self._on_hotkey, self._on_escape)

        hooks = TrayHooks(
            is_enabled=lambda: self.enabled,
            set_enabled=self._set_enabled,
            is_speaking=lambda: self.speaker.busy,
            stop=self.stop,
            model_missing=lambda: not voice_ready(self.cfg.speech,
                                                  active_voice(self.cfg.speech)),
            download_model=lambda: self.open_settings("kokoro", download=True),
            voices=self._voices,
            current_voice=lambda: self._current_voice(),
            set_voice=self._set_voice,
            current_engine=lambda: self.cfg.speech.engine,
            set_engine=self._set_engine,
            omni_available=lambda: omni_present(self.cfg.speech),
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
        voice = active_voice(self.cfg.speech)
        if not voice_ready(self.cfg.speech, voice):
            self.open_settings(self._setup_tab(voice))  # its tab has the Download panel
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
            sel = self.cfg.selection
            grab = selection.grab_selection(sel.copy_timeout_ms, sel.restore_clipboard,
                                            sel.fallback_to_clipboard)
            if not grab.text:
                self._flash("ok", "Nothing selected")
                return
            s = self.cfg.speech
            route = self._request_route(s, grab.text)  # decided once per request
            voice = route.voice
            if not voice_ready(s, voice):
                if route.omni:
                    self._flash("error", "Error: OmniVoice is not installed")
                else:
                    self._flash("error", "Error: voice model not downloaded")
                self.open_settings(self._setup_tab(voice))
                return
            chunks = textprep.prepare(grab.text, sel.clean_markdown, sel.max_chars,
                                      route.lang, long_chunks=route.omni)
            if not chunks:
                self._flash("ok", "Nothing to read")
                return
            log.info("reading %d chars in %d chunks with %s (%s)%s",
                     sum(len(c.text) for c in chunks), len(chunks), voice, route.lang,
                     " (from clipboard)" if grab.from_clipboard else "")
            self.speak(chunks, s, voice, route.lang)
        except Exception as e:
            log.exception("read selection failed")
            self._flash("error", "Error: " + _short(str(e) or e.__class__.__name__, 53))
        finally:
            with self._grab_lock:
                self._grabbing = False

    def _request_route(self, s: SpeechConfig, text: str) -> Route:
        """Voice and language of one request (see choose_voice). When OmniVoice is wanted but
        not installed the Kokoro voice is used instead (logged once)."""
        route = route_for(s, text)
        if route.fell_back and not self._warned_omni_missing:
            self._warned_omni_missing = True
            log.warning("OmniVoice is not installed (%s); using %s. Install it in "
                        "Settings > OmniVoice", s.omnivoice_dir, route.voice)
        return route

    @staticmethod
    def _setup_tab(voice: str) -> str:
        return "omnivoice" if is_omni_voice(voice) else "kokoro"

    # -- speaking
    def speak(self, chunks: list, s: SpeechConfig, voice: str | None = None,
              lang: str | None = None) -> None:
        from .tts import SpeakRequest

        voice = voice or active_voice(s)
        self._clip_engine = None
        req = SpeakRequest(chunks, voice, s.speed, s.sentence_pause, s.volume, s.device, lang)
        self._speak_cfg = s
        self._speak_voice = voice
        self._current = req
        self.overlay.show("processing", "Starting…")
        self.speaker.speak(req)

    def _engine_for_speaker(self) -> Any:
        clip = self._clip_engine
        if clip is not None:
            return clip
        return self.engine.get(self._speak_cfg, voice=self._speak_voice)

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
            elif MISSING_OMNI in error:
                self.open_settings("omnivoice")
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
                omni_present=lambda: omni_present(self.cfg.speech),
                omni_voices=lambda: omni_presets(self.cfg.speech),
                omni_voice_info=lambda: omni_preset_info(self.cfg.speech),
                omni_status=self._omni_status,
                omni_download_bytes=self._omni_download_bytes,
                omni_install=self._omni_install,
                omni_uninstall=self._omni_uninstall,
                play_preset=self._play_preset,
                design_clip=self._design_clip,
                clone_clip=self._clone_clip,
                save_voice=self._save_voice,
                delete_voice=self._delete_voice,
                play_clip=self._play_clip,
                encoders_present=self._encoders_present,
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

        route = self._request_route(cfg.speech, text)
        chunks = textprep.prepare(text, True, 0, route.lang, long_chunks=route.omni)
        if not chunks:
            return
        if not voice_ready(cfg.speech, route.voice):
            raise RuntimeError(missing_message(route.voice))
        threading.Thread(target=self.speak, args=(chunks, cfg.speech, route.voice, route.lang),
                         name="vibetts-test", daemon=True).start()

    # -- OmniVoice install (called from the settings window, on a worker thread)
    @staticmethod
    def _setup() -> Any:
        from . import omnivoice_setup  # lazy: the installer pulls in requests / numpy

        return omnivoice_setup

    def _omni_status(self) -> dict:
        """{'installed', 'encoders', 'bytes_on_disk'} of the OmniVoice files."""
        try:
            return dict(self._setup().status(omni_dir(self.cfg.speech)))
        except Exception:
            log.debug("OmniVoice status failed", exc_info=True)
            return {"installed": omni_present(self.cfg.speech),
                    "encoders": self._encoders_present(), "bytes_on_disk": 0}

    def _omni_download_bytes(self, include_encoders: bool) -> int:
        return int(self._setup().DOWNLOAD_BYTES(bool(include_encoders)))

    def _omni_install(self, include_encoders: bool, progress: Any, cancel: Any) -> None:
        """Blocking: download and convert OmniVoice into omnivoice_dir (raises
        model_setup.DownloadCancelled / SetupError)."""
        self.engine.unload_omni()
        try:
            self._setup().install(omni_dir(self.cfg.speech), include_encoders, progress, cancel)
        finally:
            self._voices_changed()  # also after a failure: files may be partly there
        if self.cfg.speech.preload and self.cfg.speech.engine == "omnivoice":
            self.engine.preload(self.cfg.speech)

    def _omni_uninstall(self) -> None:
        """Blocking: unload the engine and delete the OmniVoice files (voices are kept). An
        OmniVoice engine setting is switched back to Kokoro."""
        self.stop()
        self.engine.unload_omni()
        self._setup().uninstall(omni_dir(self.cfg.speech), keep_voices=True)
        if self.cfg.speech.engine == "omnivoice":
            self.cfg.speech.engine = "kokoro"
            self._save()
        self._voices_changed()

    # -- custom voices (called from the settings window)
    def _design_clip(self, instruct: str, seed: int, text: str | None = None,
                     lang: str | None = None) -> Any:
        """Blocking (call from a worker thread): a voice clip from a description."""
        engine = self.engine.get_omni(self.cfg.speech)
        return engine.design_clip(instruct, seed, text, lang)

    def _clone_clip(self, wav_path: Any, ref_text: str) -> Any:
        """Blocking (call from a worker thread): a voice clip cloned from a recording. The
        language stored with the preset is guessed from the transcript."""
        from .textprep import looks_croatian

        engine = self.engine.get_omni(self.cfg.speech)
        return engine.clone_clip(Path(wav_path), ref_text,
                                 "hr" if looks_croatian(ref_text) else "en")

    def _encoders_present(self) -> bool:
        try:
            return bool(_omni().encoders_present(omni_dir(self.cfg.speech)))
        except Exception:
            log.debug("encoder check failed", exc_info=True)
            return False

    def _save_voice(self, name: str, clip: Any, overwrite: bool = False) -> str:
        """Store `clip` as preset `name`; returns the voice id (omni:<name>). An existing
        preset is only replaced with overwrite=True (the settings window asks first)."""
        voice_id = _omni().save_preset(omni_dir(self.cfg.speech), name, clip,
                                       overwrite=overwrite)
        self._voices_changed()
        return voice_id

    def _delete_voice(self, name: str) -> None:
        """Delete preset `name` (or an omni:<name> id). A deleted voice that was selected falls
        back (omni_voice to omni:default, croatian_voice to 'same as omni_voice') and the
        config is saved."""
        from .tts import omni_preset

        preset = omni_preset(name) if is_omni_voice(name) else name
        _omni().delete_preset(omni_dir(self.cfg.speech), preset)
        voice_id = f"omni:{preset}"
        sp = self.cfg.speech
        changed = False
        if normalize_omni_voice(sp.omni_voice) == voice_id:
            sp.omni_voice = OMNI_DEFAULT_VOICE
            changed = True
        if sp.croatian_voice and normalize_omni_voice(sp.croatian_voice) == voice_id:
            sp.croatian_voice = ""
            changed = True
        if changed:
            self._save()
        self._voices_changed()

    def _play_preset(self, name: str) -> None:
        """Play the stored sample of preset `name` through the output device."""
        wav = omni_dir(self.cfg.speech) / "voices" / f"{name}.wav"
        audio, _ = _omni().read_wav(wav)
        self._play_clip(type("Sample", (), {"audio": audio})())

    def _play_clip(self, clip: Any) -> None:
        """Preview a VoiceClip through the configured output device (stop() ends it)."""
        from .textprep import Chunk
        from .tts import SpeakRequest

        s = self.cfg.speech
        self._clip_engine = ClipEngine(clip.audio)
        self._current = None  # no overlay callbacks for previews
        req = SpeakRequest([Chunk("preview", 0.0)], OMNI_DEFAULT_VOICE, 1.0, 0.0, s.volume,
                           s.device, "hr")
        self.speaker.speak(req)

    def _voices_changed(self) -> None:
        """Refresh the tray menus and the settings window (voices, engine radio, status)."""
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
        try:
            self.listener.set_bindings({"read": parse_chord(new.hotkey.read)})
        except ValueError as e:
            log.error("invalid hotkey on reload: %s", e)
            self._flash("error", "Error: " + _short(str(e), 53))
            return str(e)
        voice = active_voice(new.speech)
        if not voice_ready(new.speech, voice):
            return missing_message(voice)
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
        """Kokoro voices (from the loaded engine, else read from the voices file) plus the
        OmniVoice presets when OmniVoice is installed."""
        voices: list[str] = []
        loaded = self.engine.loaded
        if loaded is not None and loaded.kokoro is not None:
            voices = list(loaded.kokoro.voices())
        else:
            path = resolve_path(self.cfg.speech.voices)
            if path.is_file():
                try:
                    voices = voices_in_file(path)
                except Exception:
                    log.exception("could not read %s", path)
        return voices + omni_presets(self.cfg.speech)

    def _current_voice(self) -> str:
        sp = self.cfg.speech
        return omni_voice_of(sp) if sp.engine == "omnivoice" else kokoro_voice_of(sp)

    def _set_voice(self, voice: str) -> None:
        """Pick a voice from the tray: its engine becomes the active one."""
        sp = self.cfg.speech
        if is_omni_voice(voice):
            sp.engine, sp.omni_voice = "omnivoice", normalize_omni_voice(voice)
        else:
            sp.engine, sp.voice = "kokoro", voice
        self._save()
        self._voices_changed()

    def _set_engine(self, engine: str) -> None:
        if engine not in ("kokoro", "omnivoice"):
            return
        if engine == "omnivoice" and not omni_present(self.cfg.speech):
            return
        self.cfg.speech.engine = engine
        self._save()
        if self.cfg.speech.preload:
            self.engine.preload(self.cfg.speech)
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


# ---------------------------------------------------------------- one-shot CLI

def say(cfg: Config, text: str, wav: str | None = None) -> int:
    """Speak `text` (or write it to a WAV file) without the tray; for testing and scripts."""
    import numpy as np

    from . import textprep
    from .tts import SAMPLE_RATE, SpeakRequest, Speaker, write_wav

    s = cfg.speech
    route = route_for(s, text)
    voice, lang = route.voice, route.lang
    if route.fell_back:
        print("note: OmniVoice is not installed (Settings > OmniVoice); using "
              f"{voice}", file=sys.stderr)
    if not voice_ready(s, voice):
        if route.omni:
            print(f"error: {MISSING_OMNI} (expected in {s.omnivoice_dir})", file=sys.stderr)
        else:
            print(f"error: model files missing ({s.model}, {s.voices}); "
                  "run: python -m vibetts.model_setup", file=sys.stderr)
        return 1
    chunks = textprep.prepare(text, cfg.selection.clean_markdown, 0, lang,
                              long_chunks=route.omni)
    if not chunks:
        print("error: nothing to say", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    engine = EngineHolder().get(s, announce=False, voice=voice)  # loads the engine, warms it up
    print(f"model loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    if wav:
        parts = []
        t0 = time.perf_counter()
        for c in chunks:
            audio = engine.synth(c.text, voice, s.speed, lang)
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
    speaker.speak_and_wait(SpeakRequest(chunks, voice, s.speed, s.sentence_pause, s.volume,
                                        s.device, lang))
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
        log.info("effective hotkey=%s engine=%s voice=%s speed=%s", cfg.hotkey.read,
                 cfg.speech.engine, cfg.speech.voice, cfg.speech.speed)
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
