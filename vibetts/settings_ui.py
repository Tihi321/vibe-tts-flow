"""Settings window (tkinter + sv-ttk). Lives on the overlay's Tk root / Tk thread."""
from __future__ import annotations

import copy
import ctypes
import logging
import os
import random
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from . import model_setup
from .config import OMNIVOICE_STEPS_RANGE, ROOT, Config
from .hotkeys import parse_chord
from .tts import omni_preset, voice_label

try:  # optional: UI still works (with the stock ttk theme) without it
    import sv_ttk
except ImportError:  # pragma: no cover
    sv_ttk = None

log = logging.getLogger(__name__)

TABS = ("general", "voice", "about")
TAB_TITLES = {"general": "General", "voice": "Voice", "about": "About"}
PILL_POSITIONS = ("top", "bottom")
SAMPLE_TEXT = ("Hello! This is how VibeTTS sounds. Select any text, press the hotkey, "
               "and I will read it aloud.")
HOTKEY_HELP = ("Press once to read the selected text, again (or Esc) to stop. Join keys "
               "with +, e.g. alt+win. Names: ctrl, alt, shift, win, caps_lock, space, f1..f24, "
               "a..z, 0..9. Sides: ctrl_r, lwin, rshift.")


# ---------------------------------------------------------------- pure form mapping

def _num(v: float) -> str:
    return f"{v:g}"


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def model_choice(model_path: str) -> str:
    """'int8' / 'fp32' for the known files, else the file name."""
    name = _basename(model_path)
    for key, (filename, _) in model_setup.MODELS.items():
        if name == filename:
            return key
    return name


def form_from_config(cfg: Config) -> dict[str, Any]:
    """Flat dict of everything the form edits (numbers as strings, as typed in entries)."""
    s, sel = cfg.speech, cfg.selection
    return {
        "hotkey_read": cfg.hotkey.read,
        "copy_timeout_ms": str(sel.copy_timeout_ms),
        "fallback_to_clipboard": bool(sel.fallback_to_clipboard),
        "restore_clipboard": bool(sel.restore_clipboard),
        "clean_markdown": bool(sel.clean_markdown),
        "max_chars": str(sel.max_chars),
        "show_pill": bool(cfg.ui.show_pill),
        "pill_position": cfg.ui.pill_position,
        "model": model_choice(s.model),
        "voice": s.voice,
        "speed": _num(s.speed),
        "sentence_pause": _num(s.sentence_pause),
        "volume": _num(s.volume),
        "device": s.device,
        "preload": bool(s.preload),
        "omnivoice_steps": str(s.omnivoice_steps),
        "croatian_voice": s.croatian_voice,
        "auto_language": bool(s.auto_language),
    }


def _int(form: dict, key: str, label: str, minimum: int = 0,
         maximum: int | None = None) -> int:
    raw = str(form.get(key, "")).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{label} must be a whole number (got {raw!r})") from None
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return value


def _float(form: dict, key: str, label: str, minimum: float = 0.0,
           maximum: float | None = None) -> float:
    raw = str(form.get(key, "")).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{label} must be a number (got {raw!r})") from None
    if value != value or value < minimum:
        raise ValueError(f"{label} must be at least {_num(minimum)}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {_num(maximum)}")
    return value


def _choice(form: dict, key: str, label: str, options: tuple[str, ...]) -> str:
    value = str(form.get(key, "")).strip()
    if value not in options:
        raise ValueError(f"{label} must be one of: {', '.join(options)}")
    return value


def config_from_form(base: Config, form: dict[str, Any]) -> Config:
    """Apply `form` onto a deep copy of `base`; ValueError (readable text) on invalid input."""
    cfg = copy.deepcopy(base)
    chord = str(form.get("hotkey_read", "")).strip()
    parse_chord(chord)  # raises ValueError with a readable message
    cfg.hotkey.read = chord

    sel = cfg.selection
    sel.copy_timeout_ms = _int(form, "copy_timeout_ms", "Copy timeout (ms)", 50)
    sel.fallback_to_clipboard = bool(form.get("fallback_to_clipboard"))
    sel.restore_clipboard = bool(form.get("restore_clipboard"))
    sel.clean_markdown = bool(form.get("clean_markdown"))
    sel.max_chars = _int(form, "max_chars", "Maximum characters", 0)
    cfg.ui.show_pill = bool(form.get("show_pill"))
    cfg.ui.pill_position = _choice(form, "pill_position", "Pill position", PILL_POSITIONS)

    s = cfg.speech
    choice = str(form.get("model", "")).strip()
    if choice in model_setup.MODELS:
        filename = model_setup.MODELS[choice][0]
        old = s.model
        cut = max(old.rfind("/"), old.rfind("\\"))
        s.model = (old[: cut + 1] + filename) if cut >= 0 else filename
    elif choice != _basename(base.speech.model):
        raise ValueError("Pick a voice model")
    voice = str(form.get("voice", "")).strip().split(" ", 1)[0]
    if not voice:
        raise ValueError("Pick a voice")
    s.voice = voice
    s.speed = _float(form, "speed", "Speed", 0.5, 2.0)
    s.sentence_pause = _float(form, "sentence_pause", "Pause between sentences (s)", 0.0, 3.0)
    s.volume = _float(form, "volume", "Volume", 0.0, 1.0)
    s.device = str(form.get("device", "")).strip()
    s.preload = bool(form.get("preload"))
    s.omnivoice_steps = _int(form, "omnivoice_steps", "Croatian quality steps",
                             *OMNIVOICE_STEPS_RANGE)
    croatian = str(form.get("croatian_voice", "")).strip().split(" ", 1)[0]
    if not croatian:
        raise ValueError("Pick a Croatian voice")
    s.croatian_voice = croatian
    s.auto_language = bool(form.get("auto_language"))
    return cfg


# ---------------------------------------------------------------- hooks

def _unavailable() -> Any:
    raise RuntimeError("Custom voices are not available")


@dataclass
class SettingsHooks:
    get_config: Callable[[], Config]
    apply: Callable[[Config], str | None]
    is_autostart: Callable[[], bool]
    set_autostart: Callable[[bool], None]
    set_show_on_start: Callable[[bool], None]
    models_dir: Callable[[], Path]
    voices: Callable[[], list[str]]
    output_devices: Callable[[], list[str]]
    play_test: Callable[[Config, str], None]
    stop: Callable[[], None]
    open_logs: Callable[[], None]
    open_config: Callable[[], None]
    version: str = ""
    omni_present: Callable[[], bool] = lambda: False        # Croatian model files built?
    croatian_voices: Callable[[], list[str]] = lambda: []   # omni_hr:<preset> ids
    # custom voices (design/clone block: the dialog runs them on a worker thread)
    design_clip: Callable[..., Any] = lambda instruct, seed, text=None: _unavailable()
    clone_clip: Callable[..., Any] = lambda wav_path, ref_text: _unavailable()
    save_voice: Callable[..., str] = lambda name, clip, overwrite=False: _unavailable()
    delete_voice: Callable[[str], None] = lambda name: _unavailable()
    play_clip: Callable[[Any], None] = lambda clip: _unavailable()
    encoders_present: Callable[[], bool] = lambda: False


# ---------------------------------------------------------------- win32 / theme helpers

def _windows_dark() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except Exception:
        return False


def _fmt_size(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.0f} MB"


GWL_EXSTYLE = -20
WS_EX_APPWINDOW = 0x00040000
WS_EX_TOOLWINDOW = 0x00000080
SWP_FLAGS = 0x1 | 0x2 | 0x4 | 0x10 | 0x20  # nosize nomove nozorder noactivate framechanged

OMNI_FOUND = "Croatian model: found"
OMNI_MISSING = "Croatian model: missing — run scripts/build_omnivoice.py"
FILE_ROWS = {"int8": "Model int8 (recommended)", "fp32": "Model fp32", "voices": "Voices"}


class SettingsWindow:
    def __init__(self, root: tk.Misc, hooks: SettingsHooks,
                 post: Callable[[Callable[[], None]], None]):
        self.root = root
        self.hooks = hooks
        self._post = post
        self.top: tk.Toplevel | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._snapshot: dict[str, Any] = {}
        self._icon: Any = None
        self._dark = False
        self._busy = False
        self._cancel = threading.Event()

    # -- public -----------------------------------------------------------
    def is_visible(self) -> bool:
        top = self.top
        try:
            return bool(top is not None and top.winfo_exists() and top.state() == "normal")
        except tk.TclError:
            return False

    def show(self, tab: str | None = None, banner: str | None = None,
             download: bool = False) -> None:
        """Show the window; with `download`, also start fetching the missing model files."""
        if self.top is None:
            self._build()
        top = self.top
        assert top is not None
        self._populate()
        self._set_banner(banner)
        self._fit()
        if tab in TABS:
            self._nb.select(TABS.index(tab))
        top.deiconify()
        if top.state() == "iconic":
            top.state("normal")
        top.lift()
        try:
            top.attributes("-topmost", True)
            top.after(300, lambda: self._untopmost())
            top.focus_force()
        except tk.TclError:
            pass
        if download and not self._busy:
            self._nb.select(TABS.index("voice"))
            self._download_missing()

    def refresh(self) -> None:
        """Re-read voices / model status (e.g. after the engine finished loading)."""
        if self.top is not None:
            self._safe_ui(self._refresh_files)
            self._safe_ui(self._load_voices)

    def _fit(self) -> None:
        top = self.top
        if top is None:
            return
        try:
            top.update_idletasks()
            cap = max(400, top.winfo_screenheight() - 90)
            req_h = min(top.winfo_reqheight(), cap)
            req_w = top.winfo_reqwidth()
            top.minsize(req_w, req_h)
            top.geometry(f"{max(top.winfo_width(), req_w)}x{max(top.winfo_height(), req_h)}")
        except tk.TclError:
            pass

    def hide(self) -> None:
        if self.top is not None:
            try:
                self.top.withdraw()
            except tk.TclError:
                pass

    def close_requested(self) -> None:
        """X button / Close: prompt when dirty, then hide to tray."""
        if self.top is None:
            return
        if self._is_dirty():
            answer = messagebox.askyesnocancel("VibeTTS", "Save changes?", parent=self.top)
            if answer is None:
                return
            if answer and not self.save():
                return
        self.hide()

    # -- building -----------------------------------------------------------
    def _untopmost(self) -> None:
        try:
            if self.top is not None:
                self.top.attributes("-topmost", False)
        except tk.TclError:
            pass

    def _build(self) -> None:
        self._dark = _windows_dark()
        if sv_ttk is not None:
            try:
                sv_ttk.set_theme("dark" if self._dark else "light")
                # <<ThemeChanged>> is not delivered while the root is withdrawn, so the theme's
                # palette (frame/label backgrounds) would stay unset; apply it explicitly.
                self.root.tk.eval("configure_colors")
            except Exception:
                log.exception("could not apply sv-ttk theme")
        top = tk.Toplevel(self.root)
        self.top = top
        top.withdraw()
        top.title("VibeTTS Settings")
        scale = 1.0
        try:
            scale = max(1.0, float(top.winfo_fpixels("1i")) / 96.0)
        except tk.TclError:
            pass
        top.geometry(f"{int(680 * scale)}x{min(int(700 * scale), top.winfo_screenheight() - 90)}")
        top.protocol("WM_DELETE_WINDOW", self.close_requested)
        self._set_icon()

        outer = ttk.Frame(top, padding=10)
        outer.pack(fill="both", expand=True)

        self._banner = tk.Label(outer, text="", bg="#fff3bf", fg="#5c4400", anchor="w",
                                justify="left", padx=10, pady=6, wraplength=int(600 * scale))

        bottom = ttk.Frame(outer)
        bottom.pack(side="bottom", fill="x", pady=(10, 0))
        self._show_start = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bottom, text="Show this window when VibeTTS starts", variable=self._show_start,
            command=self._on_show_start).pack(side="left")
        ttk.Button(bottom, text="Close", command=self.close_requested).pack(side="right")
        ttk.Button(bottom, text="Save", style="Accent.TButton" if sv_ttk else "TButton",
                   command=self.save).pack(side="right", padx=(0, 8))
        self._status = ttk.Label(outer, text="")
        self._status.pack(side="bottom", fill="x", pady=(6, 0))

        self._nb = ttk.Notebook(outer)
        self._nb.pack(side="top", fill="both", expand=True)
        frames = {}
        for name in TABS:
            f = ttk.Frame(self._nb, padding=12)
            self._nb.add(f, text=TAB_TITLES[name])
            frames[name] = f
        self._build_general(frames["general"])
        self._build_voice(frames["voice"])
        self._build_about(frames["about"])
        top.update_idletasks()
        self._win32_polish()

    def _set_icon(self) -> None:
        try:
            from PIL import ImageTk
            from .tray import make_icon
            img = make_icon("#e8e8e8" if self._dark else "#2b2b2b").resize((32, 32))
            self._icon = ImageTk.PhotoImage(img, master=self.top)
            self.top.iconphoto(False, self._icon)  # type: ignore[union-attr]
        except Exception:
            log.debug("window icon failed", exc_info=True)

    def _win32_polish(self) -> None:
        if sys.platform != "win32" or self.top is None:
            return
        try:
            user32 = ctypes.windll.user32
            user32.GetWindowLongW.restype = ctypes.c_long
            hwnd = user32.GetParent(self.top.winfo_id()) or self.top.winfo_id()
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_APPWINDOW) & ~WS_EX_TOOLWINDOW
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FLAGS)
            if self._dark:
                value = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            log.debug("win32 window polish failed", exc_info=True)

    # -- widget helpers ---------------------------------------------------------
    def _var(self, key: str, value: Any) -> tk.Variable:
        var: tk.Variable = tk.BooleanVar(value=value) if isinstance(value, bool) \
            else tk.StringVar(value=value)
        self._vars[key] = var
        return var

    def _entry(self, parent, row: int, label: str, key: str, width: int = 12,
               hint: str | None = None) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 10))
        e = ttk.Entry(parent, textvariable=self._var(key, ""), width=width)
        e.grid(row=row, column=1, sticky="w", pady=3)
        if hint:
            ttk.Label(parent, text=hint, foreground="gray").grid(
                row=row, column=2, sticky="w", padx=8)
        return e

    def _combo(self, parent, row: int, label: str, key: str, values, width: int = 24,
               readonly: bool = False) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 10))
        cb = ttk.Combobox(parent, textvariable=self._var(key, ""), values=list(values),
                          width=width, state="readonly" if readonly else "normal")
        cb.grid(row=row, column=1, columnspan=2, sticky="w", pady=3)
        return cb

    def _check(self, parent, row: int, label: str, key: str, span: int = 3) -> ttk.Checkbutton:
        cb = ttk.Checkbutton(parent, text=label, variable=self._var(key, False))
        cb.grid(row=row, column=0, columnspan=span, sticky="w", pady=3)
        return cb

    # -- General -----------------------------------------------------------------
    def _build_general(self, f: ttk.Frame) -> None:
        hk = ttk.LabelFrame(f, text="Hotkey", padding=10)
        hk.pack(fill="x")
        self._entry(hk, 0, "Read aloud", "hotkey_read", width=24)
        self._chord_msg = ttk.Label(hk, text="", foreground="#d33")
        self._chord_msg.grid(row=0, column=2, sticky="w", padx=8)
        ttk.Label(hk, text=HOTKEY_HELP, foreground="gray", wraplength=580,
                  justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self._vars["hotkey_read"].trace_add("write", lambda *_: self._check_chord())
        self._autostart = tk.BooleanVar(value=False)
        ttk.Checkbutton(hk, text="Launch at login", variable=self._autostart,
                        command=self._on_autostart).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        sel = ttk.LabelFrame(f, text="Selected text", padding=10)
        sel.pack(fill="x", pady=(10, 0))
        self._check(sel, 0, "Nothing selected: read the clipboard instead",
                    "fallback_to_clipboard")
        self._check(sel, 1, "Put the previous clipboard content back after copying",
                    "restore_clipboard")
        self._check(sel, 2, "Skip markdown symbols and say \"link\" for web addresses",
                    "clean_markdown")
        self._entry(sel, 3, "Copy timeout (ms)", "copy_timeout_ms", 8,
                    hint="raise for slow apps")
        self._entry(sel, 4, "Maximum characters", "max_chars", 8, hint="0 = no limit")

        ui = ttk.LabelFrame(f, text="Status pill", padding=10)
        ui.pack(fill="x", pady=(10, 0))
        self._check(ui, 0, "Show status pill while reading", "show_pill")
        self._combo(ui, 1, "Pill position", "pill_position", PILL_POSITIONS, 10, readonly=True)

    def _check_chord(self) -> None:
        try:
            parse_chord(str(self._vars["hotkey_read"].get()))
            self._chord_msg.configure(text="")
        except ValueError as e:
            self._chord_msg.configure(text=str(e).split(";")[0][:60])

    # -- Voice ----------------------------------------------------------------------
    def _build_voice(self, f: ttk.Frame) -> None:
        # shown only while the active model or the voices file is missing
        self._setup = ttk.LabelFrame(f, text="Voice model not installed", padding=10)
        ttk.Label(self._setup, text="VibeTTS needs the Kokoro voice model to read aloud. It is "
                  "downloaded once into the models folder next to the app.",
                  wraplength=580, justify="left").pack(anchor="w")
        row = ttk.Frame(self._setup)
        row.pack(fill="x", pady=(8, 0))
        self._setup_btn = ttk.Button(row, text="Download voice", command=self._download_missing,
                                     style="Accent.TButton" if sv_ttk else "TButton")
        self._setup_btn.pack(side="left")
        self._setup_progress = ttk.Progressbar(row, mode="determinate", length=220, maximum=100)
        self._setup_progress.pack(side="left", padx=(12, 6))
        self._setup_cancel = ttk.Button(row, text="Cancel", command=self._cancel.set,
                                        state="disabled")
        self._setup_cancel.pack(side="left")
        self._setup_label = ttk.Label(self._setup, text="")
        self._setup_label.pack(anchor="w", pady=(4, 0))

        vo = ttk.LabelFrame(f, text="Voice", padding=10)
        vo.pack(fill="x")
        self._voice_frame = vo
        self._voice_combo = self._combo(vo, 0, "Voice", "voice", [], 44, readonly=True)
        self._entry(vo, 1, "Speed", "speed", 6, hint="0.5 .. 2.0")
        self._entry(vo, 2, "Pause between sentences (s)", "sentence_pause", 6)
        self._entry(vo, 3, "Volume", "volume", 6, hint="0 .. 1")
        self._device_combo = self._combo(vo, 4, "Output device", "device", [], 44)
        ttk.Label(vo, text="Leave blank for the Windows default speakers.",
                  foreground="gray").grid(row=5, column=1, columnspan=2, sticky="w")

        hr = ttk.LabelFrame(f, text="Croatian (OmniVoice)", padding=10)
        hr.pack(fill="x", pady=(10, 0))
        self._omni_status = ttk.Label(hr, text="")
        self._omni_status.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self._check(hr, 1, "Switch to the Croatian voice when the selected text looks Croatian",
                    "auto_language")
        self._croatian_combo = self._combo(hr, 2, "Croatian voice", "croatian_voice", [], 44)
        self._croatian_combo.bind("<<ComboboxSelected>>", lambda e: self._update_voice_buttons())
        vrow = ttk.Frame(hr)
        vrow.grid(row=4, column=1, columnspan=2, sticky="w", pady=(0, 3))
        self._add_voice_btn = ttk.Button(vrow, text="Add voice…", command=self._add_voice)
        self._add_voice_btn.pack(side="left")
        self._delete_voice_btn = ttk.Button(vrow, text="Delete", command=self._delete_voice)
        self._delete_voice_btn.pack(side="left", padx=6)
        self._voice_msg = ttk.Label(vrow, text="", foreground="gray")
        self._voice_msg.pack(side="left", padx=6)
        ttk.Label(hr, text="Quality steps").grid(row=3, column=0, sticky="w", pady=3, padx=(0, 10))
        lo, hi = OMNIVOICE_STEPS_RANGE
        steps_var = self._var("omnivoice_steps", "6")
        steps_row = ttk.Frame(hr)
        steps_row.grid(row=3, column=1, columnspan=2, sticky="w", pady=3)
        self._steps_scale = ttk.Scale(
            steps_row, from_=lo, to=hi, orient="horizontal", length=220,
            command=lambda v: steps_var.set(str(round(float(v)))))
        self._steps_scale.pack(side="left")
        ttk.Label(steps_row, textvariable=steps_var, width=3).pack(side="left", padx=(8, 0))
        ttk.Label(steps_row, text=f"{lo} = fastest .. {hi} = best quality",
                  foreground="gray").pack(side="left")

        def sync_scale(*_: Any) -> None:
            try:
                self._steps_scale.set(int(str(steps_var.get())))
            except (ValueError, tk.TclError):
                pass
        steps_var.trace_add("write", sync_scale)

        tf = ttk.LabelFrame(f, text="Try it", padding=8)
        tf.pack(fill="x", pady=(10, 0))
        self._sample = tk.StringVar(value=SAMPLE_TEXT)
        ttk.Entry(tf, textvariable=self._sample).pack(fill="x")
        row = ttk.Frame(tf)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Play", command=self._play_test).pack(side="left")
        ttk.Button(row, text="Stop", command=self._safe_call(
            lambda: self.hooks.stop())).pack(side="left", padx=6)
        self._test_result = ttk.Label(row, text="", wraplength=440, justify="left")
        self._test_result.pack(side="left", padx=10)

        mm = ttk.LabelFrame(f, text="Model files", padding=8)
        mm.pack(fill="both", expand=True, pady=(10, 0))
        top = ttk.Frame(mm)
        top.pack(fill="x")
        self._combo(top, 0, "Model", "model", list(model_setup.MODELS), 10, readonly=True)
        self._check(top, 1, "Load the model at startup (first read starts at once)", "preload")
        cols = ("file", "size", "status")
        self._tree = ttk.Treeview(mm, columns=cols, show="headings", height=3,
                                  selectmode="browse")
        for c, w, t in (("file", 220, "File"), ("size", 80, "Size"), ("status", 160, "Status")):
            self._tree.heading(c, text=t)
            self._tree.column(c, width=w, anchor="w")
        self._tree.pack(fill="x", pady=(6, 0))
        for name, (_, size) in model_setup.files().items():
            self._tree.insert("", "end", iid=name, values=(FILE_ROWS[name], _fmt_size(size), ""))
        btns = ttk.Frame(mm)
        btns.pack(fill="x", pady=(6, 0))
        self._dl_btn = ttk.Button(btns, text="Download", command=self._download_selected)
        self._dl_btn.pack(side="left")
        self._del_btn = ttk.Button(btns, text="Delete", command=self._delete_selected)
        self._del_btn.pack(side="left", padx=6)
        self._progress = ttk.Progressbar(btns, mode="determinate", length=160, maximum=100)
        self._progress.pack(side="left", padx=(12, 6))
        self._cancel_btn = ttk.Button(btns, text="Cancel", command=self._cancel.set,
                                      state="disabled")
        self._cancel_btn.pack(side="left")
        ttk.Button(btns, text="Open folder", command=self._safe_call(self._open_models)).pack(
            side="right")
        self._progress_label = ttk.Label(mm, text="")
        self._progress_label.pack(anchor="w", pady=(4, 0))

    def _open_models(self) -> None:
        folder = self.hooks.models_dir()
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))  # type: ignore[attr-defined]

    def _missing_files(self) -> list[str]:
        """Names (model_setup.files() keys) of the active files that are not installed."""
        speech = self.hooks.get_config().speech
        mdir = self.hooks.models_dir()
        table = model_setup.files()
        wanted = [model_choice(speech.model), "voices"]
        return [n for n in wanted if n in table and
                not model_setup.is_installed(mdir / table[n][0], table[n][1])]

    def _show_setup(self, missing: list[str]) -> None:
        if missing:
            size = sum(model_setup.files()[n][1] for n in missing)
            self._setup_btn.configure(text=f"Download voice (about {_fmt_size(size)})")
            self._setup.pack(fill="x", pady=(0, 10), before=self._voice_frame)
        elif not self._busy:
            self._setup.pack_forget()

    def _load_voices(self) -> None:
        try:
            voices = list(self.hooks.voices())
        except Exception:
            log.debug("voice list failed", exc_info=True)
            voices = []
        current = str(self._vars["voice"].get()).split(" ", 1)[0]
        if current and current not in voices:
            voices.insert(0, current)
        self._voice_display = {v: f"{v}  {voice_label(v)}" for v in voices}
        self._voice_combo.configure(values=[self._voice_display[v] for v in voices])
        if current:
            self._vars["voice"].set(self._voice_display.get(current, current))
        try:
            presets = list(self.hooks.croatian_voices())
        except Exception:
            log.debug("Croatian voice list failed", exc_info=True)
            presets = []
        hr = str(self._vars["croatian_voice"].get()).split(" ", 1)[0]
        if hr and hr not in presets:
            presets.insert(0, hr)
        hr_display = {v: f"{v}  {voice_label(v)}" for v in presets}
        self._croatian_combo.configure(values=[hr_display[v] for v in presets])
        self._custom_voices = set(presets)
        if hr:
            self._vars["croatian_voice"].set(hr_display.get(hr, hr))
        self._update_voice_buttons()

    def _selected_custom_voice(self) -> str | None:
        """The selected Croatian voice id when it is a deletable custom preset, else None."""
        voice = str(self._vars["croatian_voice"].get()).split(" ", 1)[0]
        if not voice or omni_preset(voice) == "default":
            return None
        return voice if voice in getattr(self, "_custom_voices", set()) else None

    def _update_voice_buttons(self) -> None:
        try:
            found = bool(self.hooks.omni_present())
        except Exception:
            found = False
        try:
            self._add_voice_btn.configure(state="normal" if found else "disabled")
            self._delete_voice_btn.configure(
                state="normal" if self._selected_custom_voice() else "disabled")
        except tk.TclError:
            pass

    def _add_voice(self) -> None:
        dialog = getattr(self, "_add_dialog", None)
        if dialog is not None and not dialog._closed:
            try:
                dialog.top.lift()
                return
            except tk.TclError:
                pass
        if self.top is None:
            return
        self._voice_msg.configure(text="")
        self._add_dialog = AddVoiceDialog(self.top, self.hooks, self._post, self._voice_saved)

    def _voice_saved(self, voice_id: str, use: bool) -> None:
        err = None
        if use:
            try:
                cfg = self.hooks.get_config()
                cfg.speech.croatian_voice = voice_id
                err = self.hooks.apply(cfg)
            except Exception as e:  # noqa: BLE001
                log.exception("could not set the Croatian voice")
                err = str(e)
            self._vars["croatian_voice"].set(voice_id)
            self._snapshot["croatian_voice"] = voice_id
        self._load_voices()
        text = f"Added {omni_preset(voice_id)}."
        self._voice_msg.configure(text=f"{text} (but: {err})" if err else text)

    def _delete_voice(self) -> None:
        voice = self._selected_custom_voice()
        if voice is None:
            return
        name = omni_preset(voice)
        if not messagebox.askyesno("Delete voice", f"Delete the voice \"{name}\"?",
                                   parent=self.top):
            return
        try:
            self.hooks.stop()
            self.hooks.delete_voice(name)
        except Exception as e:  # noqa: BLE001
            self._voice_msg.configure(text=f"Could not delete: {e}")
            return
        fallback = "omni_hr:default"
        for key in ("croatian_voice", "voice"):
            if str(self._vars[key].get()).split(" ", 1)[0] == voice:
                self._vars[key].set(fallback)
                self._snapshot[key] = fallback
        self._load_voices()
        self._voice_msg.configure(text=f"Deleted {name}.")

    def _refresh_omni_status(self) -> None:
        try:
            found = bool(self.hooks.omni_present())
        except Exception:
            log.debug("Croatian model check failed", exc_info=True)
            found = False
        self._omni_status.configure(text=OMNI_FOUND if found else OMNI_MISSING)

    def _refresh_files(self) -> None:
        self._refresh_omni_status()
        mdir = self.hooks.models_dir()
        speech = self.hooks.get_config().speech
        active = {_basename(speech.model), _basename(speech.voices)}
        for name, (filename, size) in model_setup.files().items():
            if model_setup.is_installed(mdir / filename, size):
                status = "Installed (active)" if filename in active else "Installed"
            elif (mdir / filename).exists():
                status = "Incomplete"
            else:
                status = "Not installed"
            self._tree.set(name, "status", status)
        self._show_setup(self._missing_files())

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in (self._dl_btn, self._del_btn, self._setup_btn):
            b.configure(state=state)
        for b in (self._cancel_btn, self._setup_cancel):
            b.configure(state="normal" if busy else "disabled")
        for label in (self._progress_label, self._setup_label):
            label.configure(text=text)
        if not busy:
            for bar in (self._progress, self._setup_progress):
                bar.configure(value=0)
        self._cancel.clear()

    def _selected_file(self) -> str | None:
        sel = self._tree.selection()
        return sel[0] if sel else None

    def _progress_cb(self, label: str) -> Callable[[int, int | None], None]:
        def cb(done: int, total: int | None) -> None:
            def ui() -> None:
                try:
                    if total:
                        text = f"{label}: {done / 1e6:.0f} / {total / 1e6:.0f} MB"
                        for bar in (self._progress, self._setup_progress):
                            bar.configure(mode="determinate", value=100.0 * done / total)
                    else:
                        text = f"{label}: {done / 1e6:.0f} MB"
                    for lbl in (self._progress_label, self._setup_label):
                        lbl.configure(text=text)
                except tk.TclError:
                    pass
            self._post(ui)
        return cb

    def _download_selected(self) -> None:
        name = self._selected_file()
        if self._busy:
            return
        if not name:
            # nothing picked: fetch whatever the active setup is missing
            self._download_missing()
            return
        self._download([name])

    def _download_missing(self) -> None:
        if self._busy:
            return
        missing = self._missing_files()
        if not missing:
            self._set_status("The active model and voices are already installed.")
            return
        self._download(missing)

    def _download(self, names: list[str]) -> None:
        cancel = self._cancel
        label = "Downloading " + " + ".join(FILE_ROWS[n] for n in names)
        cb = self._progress_cb(label)
        self._set_busy(True, f"{label}...")

        def run() -> None:
            error: Exception | None = None
            try:
                for n in names:
                    model_setup.download_named(n, self.hooks.models_dir(), cb, cancel)
            except Exception as e:  # noqa: BLE001
                error = e

            def finish() -> None:
                self._set_busy(False)
                self._refresh_files()
                if isinstance(error, model_setup.DownloadCancelled):
                    self._set_status("Download cancelled.")
                elif error is not None:
                    messagebox.showerror("Download failed", str(error), parent=self.top)
                else:
                    self._set_status("Downloaded. Loading the voice model...")
                    self._set_banner(None)
                    try:
                        err = self.hooks.apply(self.hooks.get_config())  # reload the engine
                    except Exception as e:  # noqa: BLE001
                        err = str(e)
                    hotkey = self.hooks.get_config().hotkey.read
                    self._set_status(f"Downloaded, but: {err}" if err else
                                     f"Voice ready: select text anywhere and press {hotkey}.")
            self._post(lambda: self._safe_ui(finish))

        threading.Thread(target=run, name="vibetts-download", daemon=True).start()

    def _delete_selected(self) -> None:
        name = self._selected_file()
        if self._busy or not name:
            return
        filename = model_setup.files()[name][0]
        path = self.hooks.models_dir() / filename
        if not path.exists():
            self._set_status(f"{filename} is not installed.")
            return
        if not messagebox.askyesno("Delete file", f"Delete {filename}?", parent=self.top):
            return
        self.hooks.stop()
        try:
            path.unlink()
        except OSError as e:
            messagebox.showerror("Delete failed", f"{e}\n\nQuit VibeTTS first if the model is "
                                 "in use.", parent=self.top)
        self._refresh_files()

    def _play_test(self) -> None:
        try:
            cfg = config_from_form(self.hooks.get_config(), self._collect())
        except ValueError as e:
            self._test_result.configure(text=str(e), foreground="#d33")
            return
        text = self._sample.get().strip() or SAMPLE_TEXT
        self._test_result.configure(text="", foreground="")
        try:
            self.hooks.play_test(cfg, text)
        except Exception as e:  # noqa: BLE001
            self._test_result.configure(text=f"Failed: {e}", foreground="#d33")

    # -- About --------------------------------------------------------------------------
    def _build_about(self, f: ttk.Frame) -> None:
        ttk.Label(f, text="VibeTTS", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(f, text=f"Version {self.hooks.version}").pack(anchor="w", pady=(2, 8))
        ttk.Label(f, text="Local text-to-speech with Kokoro-82M (Apache 2.0); Croatian with "
                  "OmniVoice (weights CC-BY-NC, non-commercial use only).",
                  wraplength=600, justify="left").pack(anchor="w")
        ttk.Label(f, text=f"App folder: {ROOT}", wraplength=600, justify="left").pack(
            anchor="w", pady=(6, 0))
        row = ttk.Frame(f)
        row.pack(anchor="w", pady=14)
        ttk.Button(row, text="Open config.toml", command=self._safe_call(
            lambda: self.hooks.open_config())).pack(side="left")
        ttk.Button(row, text="Open log folder", command=self._safe_call(
            lambda: self.hooks.open_logs())).pack(side="left", padx=8)

    def _safe_call(self, fn: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.exception("settings action failed")
                messagebox.showerror("VibeTTS", str(e), parent=self.top)
        return run

    def _safe_ui(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except tk.TclError:
            pass

    # -- state ------------------------------------------------------------------------
    def _populate(self) -> None:
        cfg = self.hooks.get_config()
        form = form_from_config(cfg)
        for key, value in form.items():
            if key in self._vars:
                self._vars[key].set(value)
        self._load_voices()
        try:
            self._device_combo.configure(values=["", *self.hooks.output_devices()])
        except Exception:
            self._device_combo.configure(values=[""])
        self._snapshot = self._collect()
        self._show_start.set(bool(cfg.ui.show_settings_on_start))
        try:
            self._autostart.set(bool(self.hooks.is_autostart()))
        except Exception:
            self._autostart.set(False)
        self._refresh_files()
        self._check_chord()
        self._set_status("")

    def _collect(self) -> dict[str, Any]:
        form = {k: v.get() for k, v in self._vars.items()}
        for key in ("voice", "croatian_voice"):
            form[key] = str(form.get(key, "")).split(" ", 1)[0]
        return form

    def _is_dirty(self) -> bool:
        norm = lambda d: {k: (v.strip() if isinstance(v, str) else v) for k, v in d.items()}  # noqa: E731
        return norm(self._collect()) != norm(self._snapshot)

    def _set_status(self, text: str) -> None:
        try:
            self._status.configure(text=text)
        except tk.TclError:
            pass

    def _set_banner(self, text: str | None) -> None:
        if text:
            self._banner.configure(text=text)
            self._banner.pack(side="top", fill="x", pady=(0, 8), before=self._nb)
        else:
            self._banner.pack_forget()

    def _on_show_start(self) -> None:
        try:
            self.hooks.set_show_on_start(bool(self._show_start.get()))
        except Exception:
            log.exception("could not save show-on-start")

    def _on_autostart(self) -> None:
        try:
            self.hooks.set_autostart(bool(self._autostart.get()))
        except Exception:
            log.exception("could not change autostart")
        try:
            self._autostart.set(bool(self.hooks.is_autostart()))
        except Exception:
            pass

    def save(self) -> bool:
        try:
            cfg = config_from_form(self.hooks.get_config(), self._collect())
        except ValueError as e:
            messagebox.showerror("Invalid setting", str(e), parent=self.top)
            return False
        try:
            err = self.hooks.apply(cfg)
        except Exception as e:  # noqa: BLE001
            log.exception("apply failed")
            messagebox.showerror("Could not save", str(e), parent=self.top)
            return False
        self._snapshot = self._collect()
        self._refresh_files()
        if err:
            self._set_status(f"Saved, but: {err}")
        else:
            self._set_status("Saved.")
        return True


# ---------------------------------------------------------------- Add Croatian voice dialog

ANY = "(any)"
FALLBACK_INSTRUCT_OPTIONS = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "style": ["whisper"],
}
CONSENT_TEXT = "Only clone voices you have permission to use."
CLONE_HINT = ("Use 3-8 s of clear speech with no background noise. The transcript must match "
              "what is said exactly. Longer clips are refused: trim them first.")
ENCODERS_MISSING = "Run scripts/build_omnivoice.py to enable cloning"
DESIGN_BUSY = "Generating… ~5 s"
CLONE_BUSY = "Creating the voice… ~10 s"


def _omni() -> Any:
    from .app import _omni as load

    return load()


def instruct_options() -> dict[str, list[str]]:
    try:
        return {k: list(v) for k, v in _omni().INSTRUCT_OPTIONS.items()}
    except Exception:
        log.debug("INSTRUCT_OPTIONS unavailable; using the built-in list", exc_info=True)
        return {k: list(v) for k, v in FALLBACK_INSTRUCT_OPTIONS.items()}


def build_instruct(selected: dict[str, str], options: dict[str, list[str]] | None = None) -> str:
    """Description string from the dropdown choices: set values in option order, comma
    separated; "(any)" / blank entries are omitted (so all-unset gives "")."""
    options = options if options is not None else instruct_options()
    parts = []
    for key in options:
        value = str(selected.get(key, "")).strip()
        if value and value != ANY:
            parts.append(value)
    return ", ".join(parts)


def sanitize_voice_name(raw: str) -> str:
    """Engine sanitize_name (ValueError for empty / reserved names)."""
    return str(_omni().sanitize_name(raw))


def _start_thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="vibetts-voice", daemon=True).start()


class AddVoiceDialog:
    """'Add Croatian voice' window: design a voice from a description or clone one from a
    recording, preview it, then save it as a preset. Generation runs on a worker thread
    (`spawn`); results come back through `post` (the Tk thread)."""

    def __init__(self, master: tk.Misc, hooks: SettingsHooks,
                 post: Callable[[Callable[[], None]], None],
                 on_saved: Callable[[str, bool], None] = lambda voice_id, use: None,
                 spawn: Callable[[Callable[[], None]], None] = _start_thread):
        self.hooks = hooks
        self._post = post
        self._on_saved = on_saved
        self._spawn = spawn
        self._busy = False
        self._closed = False
        self._clips: dict[str, Any] = {"design": None, "clone": None}
        self.options = instruct_options()
        try:
            self.encoders = bool(hooks.encoders_present())
        except Exception:
            self.encoders = False

        self.top = tk.Toplevel(master)
        self.top.title("Add Croatian voice")
        self.top.transient(master)  # type: ignore[arg-type]
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(self.top, padding=12)
        outer.pack(fill="both", expand=True)

        namerow = ttk.Frame(outer)
        namerow.pack(fill="x")
        ttk.Label(namerow, text="Voice name").pack(side="left", padx=(0, 10))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(namerow, textvariable=self.name_var, width=26)
        self.name_entry.pack(side="left")
        ttk.Label(namerow, text="letters, digits, _ and -", foreground="gray").pack(
            side="left", padx=8)

        self.nb = ttk.Notebook(outer)
        self.nb.pack(fill="both", expand=True, pady=(10, 0))
        self._design = ttk.Frame(self.nb, padding=10)
        self._clone = ttk.Frame(self.nb, padding=10)
        self.nb.add(self._design, text="From description")
        self.nb.add(self._clone, text="From recording")
        self._buttons: dict[str, ttk.Button] = {}
        self._build_design(self._design)
        self._build_clone(self._clone)
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._update_buttons())

        self.use_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(outer, text="Use as Croatian voice", variable=self.use_var).pack(
            anchor="w", pady=(10, 0))
        self.message = ttk.Label(outer, text="", wraplength=520, justify="left")
        self.message.pack(fill="x", pady=(8, 0))
        self._update_buttons()

    # -- building
    def _button(self, parent, key: str, text: str, command: Callable[[], None]) -> ttk.Button:
        b = ttk.Button(parent, text=text, command=command)
        self._buttons[key] = b
        return b

    def _build_design(self, f: ttk.Frame) -> None:
        self.option_vars: dict[str, tk.StringVar] = {}
        self.option_combos: dict[str, ttk.Combobox] = {}
        for row, (key, values) in enumerate(self.options.items()):
            ttk.Label(f, text=key.capitalize()).grid(row=row, column=0, sticky="w", pady=3,
                                                     padx=(0, 10))
            var = tk.StringVar(value=ANY)
            cb = ttk.Combobox(f, textvariable=var, values=[ANY, *values], state="readonly",
                              width=22)
            cb.grid(row=row, column=1, sticky="w", pady=3)
            self.option_vars[key] = var
            self.option_combos[key] = cb
        row = len(self.options)
        ttk.Label(f, text="Seed").grid(row=row, column=0, sticky="w", pady=3, padx=(0, 10))
        self.seed_var = tk.StringVar(value="0")
        ttk.Spinbox(f, from_=0, to=2_147_483_647, textvariable=self.seed_var, width=12).grid(
            row=row, column=1, sticky="w", pady=3)
        ttk.Label(f, text="Leave everything on (any) for a plain random voice.",
                  foreground="gray", wraplength=480, justify="left").grid(
            row=row + 1, column=0, columnspan=3, sticky="w", pady=(2, 8))
        btns = ttk.Frame(f)
        btns.grid(row=row + 2, column=0, columnspan=3, sticky="w")
        self._button(btns, "try", "Try another", self.try_another).pack(side="left")
        self._button(btns, "generate", "Generate", self.generate).pack(side="left", padx=6)
        self._button(btns, "play_design", "Play", lambda: self.play("design")).pack(side="left")
        self._button(btns, "stop_design", "Stop", self._stop).pack(side="left", padx=6)
        self._button(btns, "save_design", "Save", lambda: self.save("design")).pack(side="left")

    def _build_clone(self, f: ttk.Frame) -> None:
        ttk.Label(f, text="WAV file").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 10))
        self.wav_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.wav_var, width=40).grid(row=0, column=1, sticky="w")
        self._button(f, "browse", "Browse…", self.browse).grid(row=0, column=2, padx=6)
        ttk.Label(f, text="Transcript").grid(row=1, column=0, sticky="nw", pady=3, padx=(0, 10))
        self.transcript = tk.Text(f, width=46, height=4, wrap="word")
        self.transcript.grid(row=1, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Label(f, text=CLONE_HINT, foreground="gray", wraplength=500, justify="left").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(2, 2))
        self.encoders_label = ttk.Label(f, text="" if self.encoders else ENCODERS_MISSING,
                                        foreground="#d33")
        self.encoders_label.grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Label(f, text=CONSENT_TEXT, font=("Segoe UI", 9, "bold")).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 8))
        btns = ttk.Frame(f)
        btns.grid(row=5, column=0, columnspan=3, sticky="w")
        self._button(btns, "create", "Create", self.create).pack(side="left")
        self._button(btns, "play_clone", "Play", lambda: self.play("clone")).pack(
            side="left", padx=6)
        self._button(btns, "stop_clone", "Stop", self._stop).pack(side="left")
        self._button(btns, "save_clone", "Save", lambda: self.save("clone")).pack(
            side="left", padx=6)

    # -- state
    def current_tab(self) -> str:
        try:
            return "design" if self.nb.index(self.nb.select()) == 0 else "clone"
        except tk.TclError:
            return "design"

    def _update_buttons(self) -> None:
        for key, b in self._buttons.items():
            if self._busy:
                state = "disabled"
            elif key == "create":
                state = "normal" if self.encoders else "disabled"
            elif key.startswith(("play_", "save_")):
                tab = key.split("_", 1)[1]
                state = "normal" if self._clips[tab] is not None else "disabled"
            else:
                state = "normal"
            try:
                b.configure(state=state)
            except tk.TclError:
                pass

    def button_state(self, key: str) -> str:
        return str(self._buttons[key].cget("state"))

    def _info(self, text: str, error: bool = False) -> None:
        if not self._closed:
            self.message.configure(text=text, foreground="#d33" if error else "")

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self._busy = busy
        if text or not busy:
            self._info(text)
        self._update_buttons()

    def _transcript_text(self) -> str:
        return self.transcript.get("1.0", "end").strip()

    # -- generation (worker thread)
    def _run(self, work: Callable[[], Any], tab: str, busy_text: str) -> None:
        if self._busy:
            return
        self._clips[tab] = None
        self._set_busy(True, busy_text)

        def thread_main() -> None:
            result: Any = None
            error: Exception | None = None
            try:
                result = work()
            except Exception as e:  # noqa: BLE001
                error = e
                if not isinstance(e, (ValueError, FileNotFoundError)):
                    log.exception("voice generation failed")

            def finish() -> None:
                if self._closed:
                    return
                try:
                    self._busy = False
                    if error is not None:
                        self._info(str(error) or error.__class__.__name__, error=True)
                    else:
                        self._clips[tab] = result
                        note = ""
                        meta = getattr(result, "meta", None)
                        if isinstance(meta, dict):
                            note = str(meta.get("warning") or "")
                        self._info((note + " " if note else "") +
                                   "Done. Press Play to listen, Save to keep it.")
                    self._update_buttons()
                except tk.TclError:
                    pass
            self._post(finish)

        self._spawn(thread_main)

    def instruct(self) -> str:
        return build_instruct({k: v.get() for k, v in self.option_vars.items()}, self.options)

    def generate(self) -> None:
        try:
            seed = int(str(self.seed_var.get()).strip())
            if seed < 0:
                raise ValueError
        except ValueError:
            self._info("Seed must be a whole number, 0 or more.", error=True)
            return
        instruct = self.instruct()
        self._run(lambda: self.hooks.design_clip(instruct, seed, None), "design", DESIGN_BUSY)

    def try_another(self) -> None:
        if self._busy:
            return
        self.seed_var.set(str(random.randint(0, 99999)))
        self.generate()

    def browse(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.top, title="Choose a recording",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if path:
            self.wav_var.set(path)

    def create(self) -> None:
        wav = str(self.wav_var.get()).strip()
        text = self._transcript_text()
        if not self.encoders:
            self._info(ENCODERS_MISSING, error=True)
            return
        if not wav:
            self._info("Choose a WAV recording first.", error=True)
            return
        if not text:
            self._info("Type the transcript of the recording (required).", error=True)
            return
        self._run(lambda: self.hooks.clone_clip(Path(wav), text), "clone", CLONE_BUSY)

    # -- preview / save
    def play(self, tab: str | None = None) -> None:
        clip = self._clips[tab or self.current_tab()]
        if clip is None:
            return
        try:
            self.hooks.play_clip(clip)
        except Exception as e:  # noqa: BLE001
            self._info(f"Could not play: {e}", error=True)

    def _stop(self) -> None:
        try:
            self.hooks.stop()
        except Exception:
            log.debug("stop failed", exc_info=True)

    def save(self, tab: str | None = None) -> bool:
        tab = tab or self.current_tab()
        clip = self._clips[tab]
        if clip is None or self._busy:
            return False
        raw = self.name_var.get().strip()
        if not raw:
            self._info("Enter a name for the voice.", error=True)
            return False
        try:
            name = sanitize_voice_name(raw)
        except ValueError as e:
            self._info(str(e), error=True)
            return False
        if name != raw:
            self.name_var.set(name)
            self._info(f"The name was cleaned up to \"{name}\". Press Save again to keep it.")
            return False
        try:
            existing = {v.split(":", 1)[-1] for v in self.hooks.croatian_voices()}
        except Exception:
            existing = set()
        if name in existing and not messagebox.askyesno(
                "Replace voice", f"A voice named \"{name}\" already exists. Replace it?",
                parent=self.top):
            return False
        try:
            voice_id = self.hooks.save_voice(name, clip, overwrite=name in existing)
        except (ValueError, FileNotFoundError, OSError) as e:
            self._info(str(e), error=True)
            return False
        except Exception as e:  # noqa: BLE001
            log.exception("saving the voice failed")
            self._info(f"Could not save: {e}", error=True)
            return False
        use = bool(self.use_var.get())
        self.close()
        try:
            self._on_saved(voice_id, use)
        except Exception:
            log.exception("after-save update failed")
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop()
        try:
            self.top.destroy()
        except tk.TclError:
            pass
