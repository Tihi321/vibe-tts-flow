"""Floating status pill (tkinter). Runs on the main thread; other threads post via a queue."""
from __future__ import annotations

import ctypes
import logging
import queue
import time
import tkinter as tk
import tkinter.font as tkfont
from typing import Callable

from vibetts.config import UiConfig

log = logging.getLogger(__name__)

TRANSPARENT = "#010203"
PILL_BG = "#1e1e1e"
PILL_H = 40
PILL_MIN_W = 220
EDGE_MARGIN = 48
FONT = ("Segoe UI", 10, "bold")

DOT_COLORS = {
    "recording": "#e5484d",
    "handsfree": "#f5a524",
    "processing": "#5aa9ff",
    "speaking": "#a78bfa",
    "ok": "#3ecf6e",
    "error": "#e5484d",
}
TEXT_COLORS = {"error": "#ff6b6b"}

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE, SWP_FRAMECHANGED = 0x1, 0x2, 0x4, 0x10, 0x20
GA_ROOT = 2


class Overlay:
    def __init__(self, cfg: UiConfig):
        self.cfg = cfg
        self._q: queue.Queue = queue.Queue()
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._font: tkfont.Font | None = None
        self._visible = False
        self._kind = "recording"
        self._text = ""
        self._ticking = False
        self._t0 = 0.0
        self._flash_token = 0
        self._dots = 0
        self._last_dots = 0.0
        self._last_render = ""
        self._scale = 1.0
        self._pill_h = PILL_H
        self._pill_min_w = PILL_MIN_W
        self._edge = EDGE_MARGIN
        self._width = PILL_MIN_W

    @property
    def root(self) -> tk.Tk | None:
        return self._root

    def _px(self, value: float) -> int:
        return int(round(value * self._scale))

    # ---- thread-safe API -------------------------------------------
    def show(self, kind: str, text: str, ticking: bool = False) -> None:
        if self.cfg.show_pill:
            self._q.put(("show", kind, text, ticking))

    def flash(self, kind: str, text: str, seconds: float = 1.5) -> None:
        if self.cfg.show_pill:
            self._q.put(("flash", kind, text, seconds))

    def hide(self) -> None:
        if self.cfg.show_pill:
            self._q.put(("hide",))

    def call_soon(self, fn: Callable[[], None]) -> None:
        self._q.put(("call", fn))

    def quit(self) -> None:
        self._q.put(("quit",))

    # ---- main loop ---------------------------------------------------
    def run(self, on_ready: Callable[[], None] | None = None) -> None:
        root = tk.Tk()
        self._root = root
        try:
            self._scale = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        except Exception:
            self._scale = 1.0
        self._pill_h = self._px(PILL_H)
        self._pill_min_w = self._px(PILL_MIN_W)
        self._edge = self._px(EDGE_MARGIN)
        self._width = self._pill_min_w
        root.title("VibeTTS")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg=TRANSPARENT)
        try:
            root.attributes("-transparentcolor", TRANSPARENT)
        except tk.TclError:
            log.warning("transparentcolor not supported")
        self._font = tkfont.Font(root=root, family=FONT[0], size=FONT[1], weight=FONT[2])
        self._canvas = tk.Canvas(root, width=self._pill_min_w, height=self._pill_h, bg=TRANSPARENT,
                                 highlightthickness=0, bd=0)
        self._canvas.pack()
        root.geometry(f"{self._pill_min_w}x{self._pill_h}+0+0")
        root.update()
        self._apply_window_styles()
        root.withdraw()
        self._visible = False

        root.after(50, self._poll)
        root.after(500, self._tick)
        if on_ready is not None:
            root.after(0, lambda: self._safe(on_ready))
        try:
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass
            self._root = None

    @staticmethod
    def _safe(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("Overlay callback failed")

    def _apply_window_styles(self) -> None:
        try:
            user32 = ctypes.windll.user32
            user32.GetWindowLongW.restype = ctypes.c_long
            hwnds = set()
            wid = self._root.winfo_id()
            hwnds.add(wid)
            parent = user32.GetParent(wid)
            if parent:
                hwnds.add(parent)
            root_hwnd = user32.GetAncestor(wid, GA_ROOT)
            if root_hwnd:
                hwnds.add(root_hwnd)
            for hwnd in hwnds:
                style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
                user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED)
        except Exception:
            log.exception("Could not apply window styles")

    # ---- command handling -------------------------------------------
    def _poll(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            while True:
                try:
                    cmd = self._q.get_nowait()
                except queue.Empty:
                    break
                try:
                    if self._handle(cmd) == "quit":
                        root.quit()
                        return
                except Exception:
                    log.exception("Overlay command failed: %r", cmd[:1])
        finally:
            if self._root is not None:
                self._root.after(50, self._poll)

    def _handle(self, cmd: tuple):
        op = cmd[0]
        if op == "show":
            self._flash_token += 1
            self._set(cmd[1], cmd[2], cmd[3])
        elif op == "flash":
            self._flash_token += 1
            token = self._flash_token
            self._set(cmd[1], cmd[2], False)
            self._root.after(int(cmd[3] * 1000), lambda: self._end_flash(token))
        elif op == "hide":
            self._flash_token += 1
            self._hide()
        elif op == "call":
            self._safe(cmd[1])
        elif op == "quit":
            return "quit"

    def _end_flash(self, token: int) -> None:
        if token == self._flash_token:
            self._hide()

    def _set(self, kind: str, text: str, ticking: bool) -> None:
        self._kind, self._text, self._ticking = kind, text, ticking
        self._t0 = time.monotonic()
        self._dots = 0
        self._last_render = ""
        self._render()
        if not self._visible:
            self._root.deiconify()
            self._visible = True
            self._root.attributes("-topmost", True)

    def _hide(self) -> None:
        if self._visible:
            self._root.withdraw()
            self._visible = False

    # ---- drawing -----------------------------------------------------
    def _label(self) -> str:
        text = self._text
        if self._kind == "processing":
            text = text.rstrip(".… ") + "." * self._dots
        if self._ticking:
            secs = int(time.monotonic() - self._t0)
            text += f" · {secs // 60}:{secs % 60:02d}"
        return text

    def _render(self) -> None:
        label = self._label()
        if label == self._last_render:
            return
        self._last_render = label
        c, root = self._canvas, self._root
        # reserve width for the longest dotted form so the pill does not jitter
        measure = label + ("..." if self._kind == "processing" else "")
        width = max(self._pill_min_w, self._font.measure(measure) + self._px(64))
        width = min(width, max(self._pill_min_w, root.winfo_screenwidth() - 40))
        c.delete("all")
        c.configure(width=width)
        h = self._pill_h
        r = h // 2
        c.create_oval(0, 0, h, h, fill=PILL_BG, outline=PILL_BG)
        c.create_oval(width - h, 0, width, h, fill=PILL_BG, outline=PILL_BG)
        c.create_rectangle(r, 0, width - r, h, fill=PILL_BG, outline=PILL_BG)
        dot = DOT_COLORS.get(self._kind, "#ffffff")
        dr = self._px(6)
        c.create_oval(self._px(18), r - dr, self._px(18) + 2 * dr, r + dr, fill=dot, outline=dot)
        c.create_text(self._px(42), r, text=label, anchor="w", fill=TEXT_COLORS.get(self._kind, "#ffffff"), font=self._font)
        self._place(width)

    def _place(self, width: int) -> None:
        root = self._root
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x = (sw - width) // 2
        h = self._pill_h
        y = self._edge if self.cfg.pill_position == "top" else sh - h - self._edge
        root.geometry(f"{width}x{h}+{x}+{y}")
        self._width = width

    def _tick(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            if self._visible:
                if self._kind == "processing":
                    self._dots = (self._dots + 1) % 4
                self._render()
        except Exception:
            log.exception("Overlay tick failed")
        finally:
            if self._root is not None:
                self._root.after(400 if self._kind == "processing" else 500, self._tick)
