"""Read the selected text of the focused app: snapshot the clipboard, send a copy
shortcut, wait for the clipboard to change, then put the old content back.

Terminals get Ctrl+Insert instead of Ctrl+C, because Ctrl+C with nothing selected
interrupts the running program there.
"""
from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

VK_CONTROL = 0x11
VK_C = 0x43
VK_INSERT = 0x2D
KEYEVENTF_KEYUP = 0x2
KEYEVENTF_EXTENDEDKEY = 0x1
INPUT_KEYBOARD = 1
ULONG_PTR = ctypes.c_size_t

# window classes of terminals, where Ctrl+C means "interrupt"
TERMINAL_CLASSES = frozenset({
    "ConsoleWindowClass",              # conhost (cmd, PowerShell)
    "CASCADIA_HOSTING_WINDOW_CLASS",   # Windows Terminal
    "mintty",                          # Git Bash, MSYS2, Cygwin
    "VirtualConsoleClass",             # ConEmu / Cmder
    "PuTTY",
    "KiTTY",
    "org.wezfurlong.wezterm",
    "Alacritty",
})

# clipboard formats that hold GDI handles rather than memory; they cannot be copied
# as bytes (Windows synthesizes most of them again from the formats we do restore)
_HANDLE_FORMATS = frozenset({2, 3, 9, 14, 0x0082, 0x0083, 0x008E})  # BITMAP, METAFILEPICT,
# PALETTE, ENHMETAFILE, DSPBITMAP, DSPMETAFILEPICT, DSPENHMETAFILE
# OLE bookkeeping formats: they reference the source app's live object, not data
_SKIP_NAMES = frozenset({"DataObject", "Ole Private Data", "OLE Private Data"})
_CF_UNICODETEXT = 13
_CF_HDROP = 15
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    # MOUSEINPUT included so sizeof(INPUT) is correct (40 bytes on x64).
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _key(vk: int, up: bool) -> INPUT:
    flags = KEYEVENTF_KEYUP if up else 0
    if vk == VK_INSERT:
        flags |= KEYEVENTF_EXTENDEDKEY
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return inp


def send_copy(vk: int) -> None:
    """Ctrl + vk (C or Insert)."""
    user32 = ctypes.windll.user32
    for batch in ((_key(VK_CONTROL, False), _key(vk, False)),
                  (_key(vk, True), _key(VK_CONTROL, True))):
        arr = (INPUT * 2)(*batch)
        sent = user32.SendInput(2, arr, ctypes.sizeof(INPUT))
        if sent != 2:
            log.warning("SendInput sent %d/2 events", sent)
        time.sleep(0.01)


def copy_key_for(window_class: str) -> int:
    return VK_INSERT if window_class in TERMINAL_CLASSES else VK_C


def foreground_class() -> str:
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def sequence_number() -> int:
    return int(ctypes.windll.user32.GetClipboardSequenceNumber())


# ---------------------------------------------------------------- clipboard snapshot

def _open(win32clipboard) -> bool:
    for _ in range(10):
        try:
            win32clipboard.OpenClipboard()
            return True
        except Exception:
            time.sleep(0.02)
    return False


@dataclass
class Snapshot:
    formats: list[tuple[int, Any]]

    @property
    def empty(self) -> bool:
        return not self.formats


def _skip_name(win32clipboard, fmt: int) -> bool:
    if fmt < 0xC000:
        return False
    try:
        return win32clipboard.GetClipboardFormatName(fmt) in _SKIP_NAMES
    except Exception:
        return False


def snapshot_clipboard() -> Snapshot | None:
    """All memory-backed clipboard formats as (format, data); None if the clipboard is busy."""
    import win32clipboard

    if not _open(win32clipboard):
        log.debug("clipboard busy (snapshot)")
        return None
    formats: list[tuple[int, Any]] = []
    total = 0
    try:
        fmt = win32clipboard.EnumClipboardFormats(0)
        while fmt:
            if fmt not in _HANDLE_FORMATS and not _skip_name(win32clipboard, fmt):
                try:
                    data = win32clipboard.GetClipboardData(fmt)
                except Exception:
                    data = None
                if isinstance(data, (bytes, str)) or (fmt == _CF_HDROP and isinstance(data, tuple)):
                    total += len(data) if not isinstance(data, tuple) else 0
                    formats.append((fmt, data))
            fmt = win32clipboard.EnumClipboardFormats(fmt)
    finally:
        win32clipboard.CloseClipboard()
    if total > _MAX_SNAPSHOT_BYTES:
        log.info("clipboard too large to restore (%d bytes); keeping text only", total)
        formats = [(f, d) for f, d in formats if f == _CF_UNICODETEXT]
    return Snapshot(formats)


def _hdrop_bytes(paths: tuple) -> bytes:
    """DROPFILES struct + double-NUL-terminated UTF-16 path list."""
    import struct

    header = struct.pack("<IiiII", 20, 0, 0, 0, 1)  # pFiles=20, pt=(0,0), fNC=0, fWide=1
    body = "\0".join(paths) + "\0\0"
    return header + body.encode("utf-16-le")


def restore_clipboard(snap: Snapshot) -> bool:
    import win32clipboard

    if not _open(win32clipboard):
        log.warning("clipboard busy; previous content not restored")
        return False
    try:
        win32clipboard.EmptyClipboard()
        for fmt, data in snap.formats:
            try:
                if fmt == _CF_HDROP and isinstance(data, tuple):
                    data = _hdrop_bytes(data)
                win32clipboard.SetClipboardData(fmt, data)
            except Exception:
                log.debug("could not restore clipboard format %s", fmt, exc_info=True)
        # keep the restore out of Win+V clipboard history and cloud clipboard
        try:
            excl = win32clipboard.RegisterClipboardFormat(
                "ExcludeClipboardContentFromMonitorProcessing")
            win32clipboard.SetClipboardData(excl, b"\0\0\0\0")
        except Exception:
            pass
        return True
    finally:
        win32clipboard.CloseClipboard()


def clipboard_text() -> str | None:
    import win32clipboard
    import win32con

    if not _open(win32clipboard):
        return None
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        return None
    except Exception:
        log.debug("get clipboard failed", exc_info=True)
        return None
    finally:
        win32clipboard.CloseClipboard()


# ---------------------------------------------------------------- main entry

@dataclass
class Grab:
    text: str | None      # selected text, None when nothing was copied
    from_clipboard: bool  # True when `text` is the existing clipboard (fallback)


def grab_selection(timeout_ms: int = 400, restore: bool = True,
                   fallback_to_clipboard: bool = True) -> Grab:
    snap = snapshot_clipboard() if restore else None
    before = sequence_number()
    cls = foreground_class()
    vk = copy_key_for(cls)
    log.debug("copying selection from %r with Ctrl+%s", cls, "Insert" if vk == VK_INSERT else "C")
    send_copy(vk)
    deadline = time.monotonic() + timeout_ms / 1000.0
    changed = False
    while time.monotonic() < deadline:
        if sequence_number() != before:
            changed = True
            time.sleep(0.03)  # some apps write several formats one after another
            break
        time.sleep(0.01)
    if not changed:
        if fallback_to_clipboard:
            text = clipboard_text()
            return Grab(text if text and text.strip() else None, True)
        return Grab(None, False)
    text = clipboard_text()
    if restore and snap is not None:
        restore_clipboard(snap)
    return Grab(text if text and text.strip() else None, False)
