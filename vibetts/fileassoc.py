r"""Open .epub / .txt books with VibeTTS from Explorer: per-user registration under
HKCU\Software\Classes (no admin rights). Windows does not let an app make itself the default
silently; this only adds VibeTTS to "Open with", the user picks "Always" (or the default apps page)."""
from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

PROG_ID = "VibeTTS.Book"
PROG_NAME = "VibeTTS book"
EXTENSIONS = (".epub", ".txt")
EXE_NAME = "VibeTTS.exe"  # the frozen build; Applications\<exe> is only written for it
CLASSES = r"Software\Classes"
SHCNE_ASSOCCHANGED = 0x08000000
SHCNF_IDLIST = 0


class Registry(Protocol):
    """The few registry operations the registration needs (HKCU only); tests pass a fake."""

    def set_value(self, key: str, name: str, value: str) -> None: ...
    def get_value(self, key: str, name: str) -> str | None: ...
    def delete_value(self, key: str, name: str) -> None: ...
    def delete_tree(self, key: str) -> None: ...


class WinReg:
    """Registry over winreg, rooted at HKEY_CURRENT_USER."""

    def __init__(self) -> None:
        import winreg

        self._w = winreg

    def set_value(self, key: str, name: str, value: str) -> None:
        w = self._w
        with w.CreateKeyEx(w.HKEY_CURRENT_USER, key, 0, w.KEY_SET_VALUE) as k:
            w.SetValueEx(k, name, 0, w.REG_SZ, value)

    def get_value(self, key: str, name: str) -> str | None:
        w = self._w
        try:
            with w.OpenKey(w.HKEY_CURRENT_USER, key) as k:
                value, _ = w.QueryValueEx(k, name)
        except OSError:
            return None
        return value if isinstance(value, str) else None

    def delete_value(self, key: str, name: str) -> None:
        w = self._w
        try:
            with w.OpenKey(w.HKEY_CURRENT_USER, key, 0, w.KEY_SET_VALUE) as k:
                w.DeleteValue(k, name)
        except OSError:
            pass  # key or value is not there

    def delete_tree(self, key: str) -> None:
        w = self._w
        try:
            with w.OpenKey(w.HKEY_CURRENT_USER, key, 0, w.KEY_ALL_ACCESS) as k:
                while True:
                    try:
                        child = w.EnumKey(k, 0)
                    except OSError:
                        break
                    self.delete_tree(f"{key}\\{child}")
            w.DeleteKey(w.HKEY_CURRENT_USER, key)
        except OSError:
            pass  # not there


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _exe_name() -> str:
    return Path(sys.executable).name if _frozen() else ""


def open_command() -> str:
    """The shell command that opens one file: the exe itself when frozen, else pythonw + run.pyw."""
    if _frozen():
        return f'"{sys.executable}" --open "%1"'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    run = Path(__file__).resolve().parent.parent / "run.pyw"
    return f'"{pythonw}" "{run}" --open "%1"'


def _ext_key(ext: str) -> str:
    return f"{CLASSES}\\{ext}\\OpenWithProgids"


def _notify_changed() -> None:
    try:
        ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
    except Exception:
        log.debug("SHChangeNotify failed", exc_info=True)


def register(exts: tuple[str, ...] | list[str] = EXTENSIONS, reg: Registry | None = None) -> None:
    """Add VibeTTS to "Open with" for `exts` (a subset of .epub / .txt). The default value of an
    extension is never touched, other extensions already registered stay as they are."""
    bad = [e for e in exts if e not in EXTENSIONS]
    if bad:
        raise ValueError(f"Unsupported extension: {', '.join(bad)}")
    real = reg is None
    reg = reg or WinReg()
    prog = f"{CLASSES}\\{PROG_ID}"
    reg.set_value(prog, "", PROG_NAME)
    if _frozen():
        reg.set_value(f"{prog}\\DefaultIcon", "", f"{sys.executable},0")
    reg.set_value(f"{prog}\\shell\\open\\command", "", open_command())
    for ext in exts:
        reg.set_value(_ext_key(ext), PROG_ID, "")
    exe = _exe_name()
    if exe:
        app = f"{CLASSES}\\Applications\\{exe}"
        reg.set_value(f"{app}\\SupportedTypes", ".epub", "")
        reg.set_value(f"{app}\\SupportedTypes", ".txt", "")
        reg.set_value(app, "FriendlyAppName", "VibeTTS")
        reg.set_value(f"{app}\\shell\\open\\command", "", open_command())
    log.info("registered %s for %s", PROG_ID, ", ".join(exts) or "no extensions")
    if real:
        _notify_changed()


def unregister(reg: Registry | None = None) -> None:
    """Remove everything register() wrote."""
    real = reg is None
    reg = reg or WinReg()
    reg.delete_tree(f"{CLASSES}\\{PROG_ID}")
    for ext in EXTENSIONS:
        reg.delete_value(_ext_key(ext), PROG_ID)
    reg.delete_tree(f"{CLASSES}\\Applications\\{EXE_NAME}")
    log.info("unregistered %s", PROG_ID)
    if real:
        _notify_changed()


def registered_exts(reg: Registry | None = None) -> list[str]:
    """The extensions VibeTTS is currently listed for (needs the ProgID and the value)."""
    reg = reg or WinReg()
    if reg.get_value(f"{CLASSES}\\{PROG_ID}\\shell\\open\\command", "") is None:
        return []
    return [e for e in EXTENSIONS if reg.get_value(_ext_key(e), PROG_ID) is not None]


def open_default_apps() -> None:
    """Open Windows Settings > Default apps (the user picks VibeTTS there)."""
    os.startfile("ms-settings:defaultapps")  # type: ignore[attr-defined]
