"""Launch-at-login via the HKCU Run registry key."""
from __future__ import annotations

import logging
import sys
import winreg
from pathlib import Path

from vibetts.config import ROOT

log = logging.getLogger(__name__)

APP_KEY = "VibeTTS"
RUN_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


def launch_command() -> str:
    """Command stored in the Run key. Login starts are always --hidden (no window)."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --hidden'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = pythonw if pythonw.exists() else Path(sys.executable)
    return f'"{exe}" "{ROOT / "run.pyw"}" --hidden'


def current_command() -> str | None:
    """The command currently stored in the Run key, or None."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_PATH, 0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, APP_KEY)
        return str(value)
    except FileNotFoundError:
        return None
    except OSError:
        log.exception("Could not read Run key")
        return None


def sync() -> bool:
    """If autostart is enabled but stale (folder moved, old format), rewrite it."""
    try:
        stored = current_command()
        if stored is None or stored == launch_command():
            return False
        enable()
        return True
    except Exception:
        log.exception("Autostart sync failed")
        return False


def is_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_PATH, 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, APP_KEY)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        log.exception("Could not read Run key")
        return False


def enable() -> None:
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_PATH, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, APP_KEY, 0, winreg.REG_SZ, launch_command())
    log.info("Autostart enabled: %s", launch_command())


def disable() -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_PATH, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, APP_KEY)
    except FileNotFoundError:
        pass
    log.info("Autostart disabled")
