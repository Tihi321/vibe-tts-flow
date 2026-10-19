"""System tray icon and menu (pystray)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pystray
from PIL import Image, ImageDraw

from .config import SUPERTONIC_PREFIX, is_supertonic_voice
from .tts import LANG_LABELS

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
    current_voice: Callable[[], str]
    set_voice: Callable[[str], None]
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
    # language profiles: voices() lists the English Kokoro ids, then the Supertonic ids when
    # installed; current_voice / set_voice are the English profile's voice
    current_language_mode: Callable[[], str] = lambda: "auto"      # auto | english | other
    set_language_mode: Callable[[str], None] = lambda mode: None
    other_language_name: Callable[[], str] = lambda: "Other language"
    supertonic_available: Callable[[], bool] = lambda: False
    current_other_voice: Callable[[], str] = lambda: ""            # supertonic:F1
    set_other_voice: Callable[[str], None] = lambda voice: None


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

    def _voice_menu(self) -> pystray.Menu | None:
        """English voice: Kokoro voices grouped en-US / en-GB, then the Supertonic voices."""
        h = self.hooks
        M = pystray.MenuItem
        try:
            voices = list(h.voices())
        except Exception:
            log.exception("voices failed")
            voices = []
        if not voices:
            return None
        groups: list[pystray.MenuItem] = []

        def item(v: str, text: str) -> pystray.MenuItem:
            return M(text, self._action(h.set_voice, v), radio=True,
                     checked=self._checked(lambda v=v: h.current_voice() == v))

        for prefix, label in LANG_LABELS.items():
            items = []
            for v in voices:
                if is_supertonic_voice(v) or v[:1] != prefix:
                    continue
                gender = {"f": "female", "m": "male"}.get(v[1:2], "")
                items.append(item(v, f"{v[3:].title()} ({gender})" if gender else v))
            if items:
                groups.append(M(label, pystray.Menu(*items)))
        supertonic = [item(v, self._supertonic_text(v)) for v in voices if is_supertonic_voice(v)]
        if supertonic:
            groups.append(M("Supertonic", pystray.Menu(*supertonic)))
        if not groups:
            return None
        return pystray.Menu(*groups)

    def _other_voice_menu(self) -> pystray.Menu | None:
        """Voices of the other-language profile: Supertonic F1..M5, when installed."""
        h = self.hooks
        M = pystray.MenuItem
        try:
            if not h.supertonic_available():
                return None
            voices = [v for v in h.voices() if is_supertonic_voice(v)]
        except Exception:
            log.exception("other voices failed")
            return None
        if not voices:
            return None
        return pystray.Menu(*[
            M(self._supertonic_text(v), self._action(h.set_other_voice, v), radio=True,
              checked=self._checked(lambda v=v: h.current_other_voice() == v))
            for v in voices])

    def _language_menu(self) -> pystray.Menu:
        h = self.hooks
        M = pystray.MenuItem
        try:
            other = h.other_language_name()
        except Exception:
            log.exception("language name failed")
            other = "Other language"
        modes = (("auto", "Auto (detect English)"), ("english", "English"), ("other", other))
        return pystray.Menu(*[
            M(label, self._action(h.set_language_mode, mode), radio=True,
              checked=self._checked(lambda mode=mode: h.current_language_mode() == mode))
            for mode, label in modes])

    def _menu_items(self):
        h = self.hooks
        M = pystray.MenuItem
        act = self._action
        chk = self._checked

        speed_items = [M(f"{s:g}x", act(h.set_speed, s), radio=True,
                         checked=chk(lambda s=s: abs(h.current_speed() - s) < 1e-6))
                       for s in SPEEDS]
        voice_menu = self._voice_menu()
        other_menu = self._other_voice_menu()
        items = [
            M("Settings…", act(h.open_settings), default=True),
            M("Export audio…", act(h.open_export)),
            M("Download voice model…", act(h.download_model),
              visible=lambda item: bool(h.model_missing())),
            M("Enabled", act(lambda: h.set_enabled(not h.is_enabled())), checked=chk(h.is_enabled)),
            M("Stop reading", act(h.stop), enabled=lambda item: bool(h.is_speaking())),
            pystray.Menu.SEPARATOR,
        ]
        items.append(M("Language", self._language_menu()))
        if voice_menu is not None:
            items.append(M("English voice", voice_menu))
        if other_menu is not None:
            try:
                other_name = h.other_language_name()
            except Exception:
                other_name = "Other language"
            items.append(M(f"{other_name} voice", other_menu))
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
