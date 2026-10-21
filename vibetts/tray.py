"""System tray icon and menu (pystray)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pystray
from PIL import Image, ImageDraw

from .config import (
    READ_AUTO,
    SUPERTONIC_LANGUAGE_NAMES,
    SUPERTONIC_PREFIX,
    is_supertonic_voice,
    kokoro_voice_lang,
)

log = logging.getLogger(__name__)

COLOR_ENABLED = "#e8e8e8"
COLOR_DISABLED = "#7a7a7a"
COLOR_SPEAKING = "#a78bfa"
TITLE = "VibeTTS – read aloud"
SPEEDS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75)


@dataclass
class TrayHooks:
    is_enabled: Callable[[], bool]
    set_enabled: Callable[[bool], None]
    is_speaking: Callable[[], bool]
    stop: Callable[[], None]
    model_missing: Callable[[], bool]
    download_model: Callable[[], None]
    voices: Callable[[], list[str]]
    current_speed: Callable[[], float]
    set_speed: Callable[[float], None]
    is_autostart: Callable[[], bool]
    set_autostart: Callable[[bool], None]
    open_settings: Callable[[], None]
    open_config: Callable[[], None]
    open_logs: Callable[[], None]
    reload_config: Callable[[], None]
    quit: Callable[[], None]
    open_export: Callable[[], None] = lambda: None         # Settings > Export
    # per-language voices: languages() lists the enabled language codes (English first),
    # voices() the Kokoro ids, then the Supertonic ids when installed
    languages: Callable[[], list[str]] = lambda: ["en"]
    current_voice: Callable[[str], str] = lambda lang: ""         # voice id of a language
    set_voice: Callable[[str, str], None] = lambda lang, voice: None  # voice "" = Off
    read_language: Callable[[], str] = lambda: "auto"             # auto | a language code
    set_read_language: Callable[[str], None] = lambda code: None
    supertonic_available: Callable[[], bool] = lambda: False
    # book reader (VTF-07): the menu item shows only while the reader server runs
    reader_running: Callable[[], bool] = lambda: False
    open_reader: Callable[[], None] = lambda: None
    is_paused: Callable[[], bool] = lambda: False
    toggle_pause: Callable[[], None] = lambda: None


def make_icon(color: str) -> Image.Image:
    """64x64 RGBA speaker glyph with sound waves in `color` on a transparent background."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((6, 23, 18, 41), fill=color)                       # magnet
    d.polygon([(18, 23), (32, 10), (32, 54), (18, 41)], fill=color)  # cone
    d.arc((26, 20, 46, 44), start=-50, end=50, fill=color, width=4)  # small wave
    d.arc((26, 10, 60, 54), start=-50, end=50, fill=color, width=4)  # large wave
    return img


class Tray:
    def __init__(self, hooks: TrayHooks, app_name: str = "VibeTTS"):
        self.hooks = hooks
        self.app_name = app_name
        self._enabled = True
        self._speaking = False
        self._icon = pystray.Icon(app_name, make_icon(COLOR_ENABLED), TITLE,
                                  pystray.Menu(self._menu_items))

    # -- menu ---------------------------------------------------------
    @staticmethod
    def _checked(fn: Callable[[], bool]) -> Callable:
        def check(item) -> bool:
            try:
                return bool(fn())
            except Exception:
                log.exception("Tray check failed")
                return False
        return check

    def _action(self, fn: Callable, *args) -> Callable:
        def act(icon, item):
            try:
                fn(*args)
            except Exception:
                log.exception("Tray action failed")
            try:
                self._icon.update_menu()
            except Exception:
                pass
        return act

    @staticmethod
    def _supertonic_text(v: str) -> str:
        name = v[len(SUPERTONIC_PREFIX):]
        return f"{name} ({'female' if name[:1] == 'F' else 'male'})"

    def _languages(self) -> list[str]:
        try:
            return list(self.hooks.languages())
        except Exception:
            log.exception("languages failed")
            return []

    def _language_voice_menu(self, lang: str, voices: list[str]) -> pystray.Menu:
        """Voices of one language: Off, its Kokoro voices, then the Supertonic voices."""
        h = self.hooks
        M = pystray.MenuItem

        def item(v: str, text: str) -> pystray.MenuItem:
            return M(text, self._action(h.set_voice, lang, v), radio=True,
                     checked=self._checked(lambda v=v: h.current_voice(lang) == v))

        items = [item("", "Off")]
        for v in voices:
            if is_supertonic_voice(v) or kokoro_voice_lang(v) != lang:
                continue
            gender = {"f": "female", "m": "male"}.get(v[1:2], "")
            items.append(item(v, f"{v[3:].title()} ({gender})" if gender else v))
        items += [item(v, self._supertonic_text(v)) for v in voices if is_supertonic_voice(v)]
        return pystray.Menu(*items)

    def _voice_menu(self) -> pystray.Menu | None:
        """One submenu per enabled language (Off, Kokoro voices of that language, Supertonic)."""
        try:
            voices = list(self.hooks.voices())
        except Exception:
            log.exception("voices failed")
            voices = []
        langs = self._languages()
        if not langs:
            return None
        M = pystray.MenuItem
        return pystray.Menu(*[
            M(SUPERTONIC_LANGUAGE_NAMES.get(lang, lang), self._language_voice_menu(lang, voices))
            for lang in langs])

    def _language_menu(self) -> pystray.Menu:
        """Auto (detect), or always read in one enabled language."""
        h = self.hooks
        M = pystray.MenuItem
        modes = [(READ_AUTO, "Auto (detect language)")]
        modes += [(c, f"Always {SUPERTONIC_LANGUAGE_NAMES.get(c, c)}") for c in self._languages()]
        return pystray.Menu(*[
            M(label, self._action(h.set_read_language, code), radio=True,
              checked=self._checked(lambda code=code: h.read_language() == code))
            for code, label in modes])

    def _menu_items(self):
        h = self.hooks
        M = pystray.MenuItem
        act = self._action
        chk = self._checked

        speed_items = [M(f"{s:g}x", act(h.set_speed, s), radio=True,
                         checked=chk(lambda s=s: abs(h.current_speed() - s) < 1e-6))
                       for s in SPEEDS]
        voice_menu = self._voice_menu()
        items = [
            M("Settings…", act(h.open_settings), default=True),
            M("Export audio…", act(h.open_export)),
            M("Open reader", act(h.open_reader), visible=lambda item: bool(h.reader_running())),
            M("Download voice model…", act(h.download_model),
              visible=lambda item: bool(h.model_missing())),
            M("Enabled", act(lambda: h.set_enabled(not h.is_enabled())), checked=chk(h.is_enabled)),
            M(lambda item: "Resume" if h.is_paused() else "Pause", act(h.toggle_pause),
              enabled=lambda item: bool(h.is_speaking())),
            M("Stop reading", act(h.stop), enabled=lambda item: bool(h.is_speaking())),
            pystray.Menu.SEPARATOR,
        ]
        items.append(M("Language", self._language_menu()))
        if voice_menu is not None:
            items.append(M("Voices", voice_menu))
        items += [
            M("Speed", pystray.Menu(*speed_items)),
            pystray.Menu.SEPARATOR,
            M("Launch at login", act(lambda: h.set_autostart(not h.is_autostart())),
              checked=chk(h.is_autostart)),
            M("Open config.toml", act(h.open_config)),
            M("Open log folder", act(h.open_logs)),
            M("Reload config", act(h.reload_config)),
            pystray.Menu.SEPARATOR,
            M("Quit", act(h.quit)),
        ]
        return items

    # -- lifecycle ----------------------------------------------------
    def start(self) -> None:
        self._icon.run_detached()

    def update_menu(self) -> None:
        try:
            self._icon.update_menu()
        except Exception:
            log.debug("Tray update_menu failed", exc_info=True)

    def stop(self) -> None:
        try:
            self._icon.stop()
        except Exception:
            log.exception("Tray stop failed")

    def _refresh_icon(self) -> None:
        if self._speaking:
            color = COLOR_SPEAKING
        else:
            color = COLOR_ENABLED if self._enabled else COLOR_DISABLED
        try:
            self._icon.icon = make_icon(color)
        except Exception:
            log.exception("Tray icon update failed")

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        self._refresh_icon()
        self.update_menu()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._refresh_icon()

    def notify(self, title: str, message: str) -> None:
        try:
            self._icon.notify(message, title)
        except Exception:
            log.debug("Tray notify failed", exc_info=True)
