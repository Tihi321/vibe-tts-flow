"""System tray icon and menu (pystray)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pystray
from PIL import Image, ImageDraw

from .tts import LANG_LABELS, OMNI_LABEL, is_omni_voice, omni_preset

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
    set_voice: Callable[[str], None]  # also makes the voice's engine the active one
    current_speed: Callable[[], float]
    set_speed: Callable[[float], None]
    is_autostart: Callable[[], bool]
    set_autostart: Callable[[bool], None]
    open_settings: Callable[[], None]
    open_config: Callable[[], None]
    open_logs: Callable[[], None]
    reload_config: Callable[[], None]
    quit: Callable[[], None]
    current_engine: Callable[[], str] = lambda: "kokoro"   # "kokoro" | "omnivoice"
    set_engine: Callable[[str], None] = lambda engine: None
    omni_available: Callable[[], bool] = lambda: False     # OmniVoice installed?
    open_export: Callable[[], None] = lambda: None         # Settings > Export


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

    def _voice_menu(self) -> pystray.Menu | None:
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
                if is_omni_voice(v) or v[:1] != prefix:
                    continue
                gender = {"f": "female", "m": "male"}.get(v[1:2], "")
                items.append(item(v, f"{v[3:].title()} ({gender})" if gender else v))
            if items:
                groups.append(M(label, pystray.Menu(*items)))
        omni = [item(v, omni_preset(v).replace("_", " ").title())
                for v in voices if is_omni_voice(v)]
        if omni:
            groups.append(M(OMNI_LABEL, pystray.Menu(*omni)))
        return pystray.Menu(*groups)

    def _engine_menu(self) -> pystray.Menu:
        h = self.hooks
        M = pystray.MenuItem

        def entry(engine: str, text: str, enabled: Callable[[], bool]) -> pystray.MenuItem:
            return M(text, self._action(h.set_engine, engine), radio=True,
                     checked=self._checked(lambda: h.current_engine() == engine),
                     enabled=lambda item: bool(enabled()))
        return pystray.Menu(entry("kokoro", "Kokoro", lambda: True),
                            entry("omnivoice", OMNI_LABEL, h.omni_available))

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
            M("Download voice model…", act(h.download_model),
              visible=lambda item: bool(h.model_missing())),
            M("Enabled", act(lambda: h.set_enabled(not h.is_enabled())), checked=chk(h.is_enabled)),
            M("Stop reading", act(h.stop), enabled=lambda item: bool(h.is_speaking())),
            pystray.Menu.SEPARATOR,
        ]
        items.append(M("Engine", self._engine_menu()))
        if voice_menu is not None:
            items.append(M("Voice", voice_menu))
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
