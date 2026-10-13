"""Global hotkey handling: chord parsing, press/release tracking, pynput listener."""
from __future__ import annotations

import ctypes
import logging
import re
import string
import time
from ctypes import wintypes
from typing import Callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- key tables

_VK_BY_ID: dict[str, int] = {
    "ctrl_l": 0xA2, "ctrl_r": 0xA3, "alt_l": 0xA4, "alt_r": 0xA5,
    "shift_l": 0xA0, "shift_r": 0xA1, "cmd_l": 0x5B, "cmd_r": 0x5C,
    "caps_lock": 0x14, "space": 0x20, "esc": 0x1B, "tab": 0x09, "enter": 0x0D,
    "backspace": 0x08, "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23,
    "page_up": 0x21, "page_down": 0x22,
}
for _i in range(1, 25):
    _VK_BY_ID[f"f{_i}"] = 0x6F + _i
for _c in string.ascii_lowercase:
    _VK_BY_ID[_c] = ord(_c.upper())
for _c in string.digits:
    _VK_BY_ID[_c] = ord(_c)

_ID_BY_VK: dict[int, str] = {v: k for k, v in _VK_BY_ID.items()}
# generic modifier vks reported by some sources
_ID_BY_VK.update({0x11: "ctrl_l", 0x12: "alt_l", 0x10: "shift_l"})

_ALL_IDS = frozenset(_VK_BY_ID)

_LABELS = {
    "ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "cmd": "Win", "caps_lock": "Caps Lock",
    "esc": "Esc", "page_up": "Page Up", "page_down": "Page Down", "space": "Space",
}

_ALIASES: dict[str, frozenset[str]] = {}


def _alias(names: list[str], ids: set[str]) -> None:
    for n in names:
        _ALIASES[n] = frozenset(ids)


_alias(["ctrl", "control"], {"ctrl_l", "ctrl_r"})
_alias(["alt"], {"alt_l", "alt_r"})
_alias(["shift"], {"shift_l", "shift_r"})
_alias(["win", "super", "cmd", "meta"], {"cmd_l", "cmd_r"})
_alias(["capslock", "caps", "caps_lock"], {"caps_lock"})
_alias(["escape", "esc"], {"esc"})
_alias(["return", "enter"], {"enter"})
_alias(["del", "delete"], {"delete"})
_alias(["ins", "insert"], {"insert"})
_alias(["pageup", "pgup", "page_up"], {"page_up"})
_alias(["pagedown", "pgdn", "page_down"], {"page_down"})
for _base, _canon in (("ctrl", "ctrl"), ("control", "ctrl"), ("alt", "alt"), ("shift", "shift"),
                      ("win", "cmd"), ("cmd", "cmd"), ("super", "cmd"), ("meta", "cmd")):
    for _side, _s in (("l", "l"), ("left", "l"), ("r", "r"), ("right", "r")):
        for _n in (f"{_base}_{_side}", f"{_side}{_base}", f"{_side}_{_base}"):
            _alias([_n], {f"{_canon}_{_s}"})
_alias(["altgr", "alt_gr"], {"alt_r"})


def key_id_to_vk(key_id: str) -> int | None:
    if key_id in _VK_BY_ID:
        return _VK_BY_ID[key_id]
    m = re.fullmatch(r"vk:(\d+)", key_id)
    return int(m.group(1)) if m else None


def vk_to_key_id(vk: int) -> str:
    return _ID_BY_VK.get(vk, f"vk:{vk}")


def _label_for(group: frozenset[str]) -> str:
    if len(group) == 2:
        base = next(iter(group)).rsplit("_", 1)[0]
        return _LABELS.get(base, base.title())
    kid = next(iter(group))
    if kid.startswith("vk:"):
        return kid
    base = kid.rsplit("_", 1)[0]
    if kid[-2:] in ("_l", "_r") and base in ("ctrl", "alt", "shift", "cmd"):
        side = "Left" if kid.endswith("_l") else "Right"
        return f"{side} {_LABELS[base]}"
    if kid in _LABELS:
        return _LABELS[kid]
    return kid.upper()


class Chord:
    def __init__(self, groups: tuple[frozenset[str], ...], spec: str):
        self.groups = groups
        self.spec = spec

    @property
    def has_win(self) -> bool:
        return any(g & {"cmd_l", "cmd_r"} for g in self.groups)

    def describe(self) -> str:
        return " + ".join(_label_for(g) for g in self.groups)

    def __repr__(self) -> str:
        return f"Chord({self.spec!r})"


def parse_chord(spec: str) -> Chord:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('hotkey chord is empty; use e.g. "ctrl+win"')
    groups: list[frozenset[str]] = []
    for raw in spec.split("+"):
        name = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if not name:
            raise ValueError(f"hotkey chord {spec!r} has an empty key name (stray '+')")
        if name in _ALIASES:
            group = _ALIASES[name]
        elif name in _ALL_IDS:
            group = frozenset({name})
        elif re.fullmatch(r"vk:\d+", name):
            group = frozenset({name})
        else:
            raise ValueError(
                f"unknown key {raw.strip()!r} in hotkey chord {spec!r}; valid names include "
                "ctrl, alt, shift, win, caps_lock, space, f1..f24, a..z, 0..9, "
                "or sides such as ctrl_r / lwin"
            )
        if group not in groups:
            groups.append(group)
    return Chord(tuple(groups), spec.strip())


class ChordTracker:
    """Pure press/release tracking; no OS calls."""

    def __init__(self, chord: Chord):
        self._chord = chord
        self._pressed: set[str] = set()
        self._down = False

    def _satisfied(self) -> bool:
        return all(g & self._pressed for g in self._chord.groups)

    @property
    def is_down(self) -> bool:
        return self._down

    def set_chord(self, chord: Chord) -> None:
        self._chord = chord
        self._down = False
        self._pressed.clear()

    def press(self, key_id: str) -> bool:
        if key_id in self._pressed:
            return False  # key repeat
        self._pressed.add(key_id)
        if not self._down and self._satisfied():
            self._down = True
            return True
        return False

    def release(self, key_id: str) -> bool:
        self._pressed.discard(key_id)
        if self._down and not self._satisfied():
            self._down = False
            return True
        return False


# ---------------------------------------------------------------- OS helpers

_user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None
_ULONG_PTR = ctypes.c_size_t


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002


def _send_noop_key() -> None:
    """Press+release the unassigned VK 0xFF so Windows does not open Start on Win release."""
    if _user32 is None:
        return
    items = (_INPUT * 2)()
    for i, flags in enumerate((0, _KEYEVENTF_KEYUP)):
        items[i].type = _INPUT_KEYBOARD
        items[i].u.ki = _KEYBDINPUT(0xFF, 0, flags, 0, 0)
    _user32.SendInput(2, ctypes.byref(items), ctypes.sizeof(_INPUT))


def key_is_down(key_id: str) -> bool:
    if _user32 is None:
        return False
    vk = key_id_to_vk(key_id)
    if vk is None:
        return False
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


_MODIFIER_IDS = ("ctrl_l", "ctrl_r", "alt_l", "alt_r", "shift_l", "shift_r", "cmd_l", "cmd_r")


def wait_for_chord_release(chord: Chord, timeout: float = 1.0) -> bool:
    """Block until no chord key and no modifier is physically down (so a synthetic
    Ctrl+C is not turned into Win+Ctrl+C). True if released, False on timeout."""
    keys = {k for g in chord.groups for k in g} | set(_MODIFIER_IDS)
    deadline = time.monotonic() + timeout
    while True:
        if not any(key_is_down(k) for k in keys):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


# ---------------------------------------------------------------- listener

def _key_to_id(key) -> str | None:
    """Map a pynput Key/KeyCode to a canonical id (via its virtual-key code)."""
    vk = getattr(key, "vk", None)
    if vk is None:
        vk = getattr(getattr(key, "value", None), "vk", None)
    if vk is not None:
        return vk_to_key_id(int(vk))
    char = getattr(key, "char", None)
    if char and len(char) == 1 and char.isalnum() and char.isascii():
        return char.lower()
    return None


class HotkeyListener:
    """Passive global listener for several named chords (e.g. "read", later "translate")."""

    def __init__(self, bindings: dict[str, Chord], on_down: Callable[[str], None],
                 on_escape: Callable[[], None]):
        self._on_down = on_down
        self._on_escape = on_escape
        self._listener = None
        self._bindings: dict[str, tuple[Chord, ChordTracker]] = {}
        self.set_bindings(bindings)

    def set_bindings(self, bindings: dict[str, Chord]) -> None:
        self._bindings = {name: (chord, ChordTracker(chord)) for name, chord in bindings.items()}

    def chord(self, name: str) -> Chord | None:
        entry = self._bindings.get(name)
        return entry[0] if entry else None

    def start(self) -> None:
        from pynput import keyboard  # lazy

        if self._listener is not None:
            return
        # passive listener: nothing is suppressed, keys still reach the focused app
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()
        log.info("hotkey listener started: %s", ", ".join(
            f"{n}={c.describe()}" for n, (c, _) in self._bindings.items()))

    def stop(self) -> None:
        lst, self._listener = self._listener, None
        if lst is not None:
            try:
                lst.stop()
            except Exception:
                log.exception("error stopping hotkey listener")

    # pynput passes (key, injected); callbacks run on the listener thread.
    # Injected events (our own SendInput) are ignored.
    def _on_press(self, key, injected=False):
        if injected:
            return
        try:
            kid = _key_to_id(key)
            if kid is None:
                return
            if kid == "esc":
                self._on_escape()
            for name, (chord, tracker) in list(self._bindings.items()):
                if tracker.press(kid):
                    if chord.has_win:
                        try:
                            _send_noop_key()
                        except Exception:
                            log.exception("failed to send no-op key")
                    self._on_down(name)
        except Exception:
            log.exception("error in key press handler")

    def _on_release(self, key, injected=False):
        if injected:
            return
        try:
            kid = _key_to_id(key)
            if kid is None:
                return
            for _, tracker in list(self._bindings.values()):
                tracker.release(kid)
        except Exception:
            log.exception("error in key release handler")
