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

from . import docload, model_setup
from .config import (EXPORT_FORMATS, MCP_DEFAULT_PORT, MP3_BITRATES, OMNI_DEFAULT_VOICE,
                     OMNIVOICE_STEPS_RANGE, ROOT, Config, ENGINES, export_folder,
                     is_omni_voice, normalize_omni_voice, snap_bitrate)
from .export import ExportBusy, ExportCancelled, ExportProgress, ExportRequest
from .hotkeys import parse_chord
from .tts import omni_preset, voice_label

try:  # optional: UI still works (with the stock ttk theme) without it
    import sv_ttk
except ImportError:  # pragma: no cover
    sv_ttk = None

log = logging.getLogger(__name__)

TABS = ("general", "voice", "export", "kokoro", "omnivoice", "mcp", "about")
TAB_TITLES = {"general": "General", "voice": "Voice", "export": "Export", "kokoro": "Kokoro",
              "omnivoice": "OmniVoice", "mcp": "Agents (MCP)", "about": "About"}
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


SAME_VOICE = "Same as OmniVoice voice"  # croatian_voice == ""
KIND_LABELS = {"default": "default", "description": "description", "recording": "recording",
               "random": "random"}
LANG_NAMES = {"hr": "Croatian", "en": "English"}
CLONING_BYTES = 654_000_000
OMNI_FALLBACK_BYTES = 2_900_000_000
OMNI_NOTE = ("One-time download of about 2.9 GB, or 3.5 GB with voice cloning (from "
             "huggingface.co), converted "
             "on this computer. It then uses about 1.3 GB of disk. The OmniVoice weights are "
             "CC-BY-NC: non-commercial use only.")


EXPORT_CHARS_PER_SECOND = 15.0  # rough speech rate at speed 1.0, for the length estimate
EXPORT_LANGS = {"Auto": None, "English": "en", "Croatian": "hr"}
AUTO_VOICE = "Auto (as configured)"
MCP_NOTE = ("Listens on 127.0.0.1 only. Agents can read text aloud and write .wav/.mp3 "
            "files. Off by default.")
MCP_TOOLS = "Tools: synthesize_to_file, speak, list_voices, stop."


def mcp_url(port: Any) -> str:
    return f"http://127.0.0.1:{str(port).strip() or MCP_DEFAULT_PORT}/mcp"


def mcp_command(url: str) -> str:
    return f"claude mcp add --transport http vibetts {url}"


def mcp_snippet(url: str) -> str:
    return '{"mcpServers": {"vibetts": {"type": "http", "url": "' + url + '"}}}'


def estimate_text(text: str, speed: float = 1.0) -> str:
    n = len(text.strip())
    minutes = n / EXPORT_CHARS_PER_SECOND / max(speed, 0.1) / 60.0
    if n == 0:
        about = "0 min"
    elif minutes < 1:
        about = "under 1 min"
    else:
        about = f"{round(minutes)} min"
    return f"{n:,} characters · about {about} of audio"


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code.upper() if code else "")


def _voice_token(raw: Any) -> str:
    """Voice id from a combo entry ('omni:alma  Alma (OmniVoice)' -> 'omni:alma'); the
    'Same as OmniVoice voice' entry is ''."""
    text = str(raw or "").strip()
    if text == SAME_VOICE:
        return ""
    return text.split(" ", 1)[0]


def _start_thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="vibetts-worker", daemon=True).start()


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
        "engine": s.engine,
        "voice": s.voice,
        "omni_voice": s.omni_voice,
        "speed": _num(s.speed),
        "sentence_pause": _num(s.sentence_pause),
        "volume": _num(s.volume),
        "device": s.device,
        "preload": bool(s.preload),
        "omnivoice_steps": str(s.omnivoice_steps),
        "croatian_voice": s.croatian_voice,
        "auto_language": bool(s.auto_language),
        "export_folder": cfg.export.folder,
        "export_format": cfg.export.format,
        "mp3_bitrate": str(cfg.export.mp3_bitrate),
        "mcp_enabled": bool(cfg.mcp.enabled),
        "mcp_port": str(cfg.mcp.port),
        "mcp_max_chars": str(cfg.mcp.max_chars),
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
    s.engine = _choice(form, "engine", "Engine", ENGINES)
    voice = _voice_token(form.get("voice", ""))
    if not voice:
        raise ValueError("Pick a voice")
    s.voice = voice
    omni = _voice_token(form.get("omni_voice", "")) or OMNI_DEFAULT_VOICE
    s.omni_voice = normalize_omni_voice(omni) if is_omni_voice(omni) else OMNI_DEFAULT_VOICE
    s.speed = _float(form, "speed", "Speed", 0.5, 2.0)
    s.sentence_pause = _float(form, "sentence_pause", "Pause between sentences (s)", 0.0, 3.0)
    s.volume = _float(form, "volume", "Volume", 0.0, 1.0)
    s.device = str(form.get("device", "")).strip()
    s.preload = bool(form.get("preload"))
    s.omnivoice_steps = _int(form, "omnivoice_steps", "OmniVoice quality steps",
                             *OMNIVOICE_STEPS_RANGE)
    croatian = _voice_token(form.get("croatian_voice", ""))
    s.croatian_voice = normalize_omni_voice(croatian) if is_omni_voice(croatian) else ""
    s.auto_language = bool(form.get("auto_language"))

    cfg.export.folder = str(form.get("export_folder", "")).strip()
    cfg.export.format = _choice(form, "export_format", "Export format", EXPORT_FORMATS)
    cfg.export.mp3_bitrate = snap_bitrate(_int(form, "mp3_bitrate", "MP3 bitrate (kbps)", 1))
    cfg.mcp.enabled = bool(form.get("mcp_enabled"))
    cfg.mcp.port = _int(form, "mcp_port", "MCP port", 1024, 65535)
    cfg.mcp.max_chars = _int(form, "mcp_max_chars", "MCP maximum characters", 0)
    return cfg


# ---------------------------------------------------------------- hooks

def _unavailable() -> Any:
    raise RuntimeError("OmniVoice is not available")


def _not_supported() -> Any:
    raise RuntimeError("Not supported")


def _fallback_bytes(include_encoders: bool) -> int:
    return OMNI_FALLBACK_BYTES + (CLONING_BYTES if include_encoders else 0)


def _no_status() -> dict:
    return {"installed": False, "encoders": False, "bytes_on_disk": 0}


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
    omni_present: Callable[[], bool] = lambda: False        # OmniVoice installed?
    omni_voices: Callable[[], list[str]] = lambda: []       # omni:<preset> ids
    omni_voice_info: Callable[[], list[dict]] = lambda: []  # {name, kind, description, lang}
    # OmniVoice install: blocking, so the window runs them on a worker thread.
    # progress(phase_text, done, total); raise model_setup.DownloadCancelled / SetupError
    omni_status: Callable[[], dict] = _no_status            # installed / encoders / bytes_on_disk
    omni_download_bytes: Callable[[bool], int] = _fallback_bytes
    omni_install: Callable[..., None] = lambda include_encoders, progress, cancel: _unavailable()
    omni_uninstall: Callable[[], None] = lambda: _unavailable()
    play_preset: Callable[[str], None] = lambda name: _unavailable()
    # custom voices (design/clone block: the creator runs them on a worker thread)
    design_clip: Callable[..., Any] = lambda instruct, seed, text=None, lang=None: _unavailable()
    clone_clip: Callable[..., Any] = lambda wav_path, ref_text: _unavailable()
    save_voice: Callable[..., str] = lambda name, clip, overwrite=False: _unavailable()
    delete_voice: Callable[[str], None] = lambda name: _unavailable()
    play_clip: Callable[[Any], None] = lambda clip: _unavailable()
    encoders_present: Callable[[], bool] = lambda: False
    # export (blocking: the window runs them on a worker thread) and the MCP server
    load_document: Callable[[str], Any] = lambda path: _not_supported()  # -> docload.Document
    export_audio: Callable[..., Any] = lambda req, progress=None, cancel=None: _not_supported()
    cancel_export: Callable[[], None] = lambda: None
    mp3_available: Callable[[], bool] = lambda: False
    mcp_status: Callable[[], dict] = lambda: {"enabled": False, "running": False, "url": "",
                                              "error": ""}
    open_folder: Callable[[Path], None] = lambda path: None


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

OMNI_MISSING = "OmniVoice is not installed."
FILE_ROWS = {"int8": "Model int8 (recommended)", "fp32": "Model fp32", "voices": "Voices"}


class ScrollableFrame(ttk.Frame):
    """A frame whose `body` scrolls vertically (mouse wheel while the pointer is over it)."""

    def __init__(self, master: tk.Misc, height: int = 480):
        super().__init__(master)
        bg = None
        try:
            bg = ttk.Style().lookup("TFrame", "background") or None
        except tk.TclError:
            pass
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, height=height,
                                **({"background": bg} if bg else {}))
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.canvas, padding=12)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._window, width=e.width))
        self.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, event: tk.Event) -> None:
        try:
            first, last = self.canvas.yview()
            if first > 0.0 or last < 1.0:
                self.canvas.yview_scroll(int(-event.delta / 120), "units")
        except tk.TclError:
            pass


class SettingsWindow:
    def __init__(self, root: tk.Misc, hooks: SettingsHooks,
                 post: Callable[[Callable[[], None]], None],
                 spawn: Callable[[Callable[[], None]], None] = _start_thread):
        self.root = root
        self.hooks = hooks
        self._post = post
        self._spawn = spawn
        self.top: tk.Toplevel | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._snapshot: dict[str, Any] = {}
        self._icon: Any = None
        self._dark = False
        self._busy = False
        self._cancel = threading.Event()
        self._omni_busy = False
        self._omni_cancel = threading.Event()
        self._omni_installed = False
        self._omni_encoders = False
        self._export_busy = False
        self._export_cancel = threading.Event()
        self._export_title = ""
        self._export_markdown = True

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
            self._nb.select(TABS.index("kokoro"))
            self._download_missing()

    def refresh(self) -> None:
        """Re-read voices / model status (e.g. after the engine finished loading, an install
        or a tray change). Form fields the user has not edited follow the saved config."""
        if self.top is not None:
            self._safe_ui(self._follow_config)
            self._safe_ui(self._refresh_files)
            self._safe_ui(self._load_voices)
            self._safe_ui(self._refresh_export)
            self._safe_ui(self._refresh_mcp)

    def _follow_config(self) -> None:
        form = form_from_config(self.hooks.get_config())
        collected = self._collect()
        for key in ("engine", "voice", "omni_voice", "croatian_voice"):
            if key in self._vars and collected.get(key) == self._snapshot.get(key):
                self._vars[key].set(form[key])
                self._snapshot[key] = form[key]

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
            if name in ("omnivoice", "export"):  # the tallest tab: scrolls instead of growing the window
                holder = ScrollableFrame(self._nb, height=int(480 * scale))
                self._nb.add(holder, text=TAB_TITLES[name])
                frames[name] = holder.body
                continue
            f = ttk.Frame(self._nb, padding=12)
            self._nb.add(f, text=TAB_TITLES[name])
            frames[name] = f
        self._build_general(frames["general"])
        self._build_voice(frames["voice"])
        self._build_export(frames["export"])
        self._build_kokoro(frames["kokoro"])
        self._build_omnivoice(frames["omnivoice"])
        self._build_mcp(frames["mcp"])
        self._build_about(frames["about"])
        self._on_engine_change()
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
        eng = ttk.LabelFrame(f, text="Engine", padding=10)
        eng.pack(fill="x")
        engine_var = self._var("engine", "kokoro")
        self._engine_radios: dict[str, ttk.Radiobutton] = {}
        for value, text in (("kokoro", "Kokoro (small, fast, English and a few other languages)"),
                            ("omnivoice", "OmniVoice (any language incl. Croatian, custom voices)")):
            rb = ttk.Radiobutton(eng, text=text, value=value, variable=engine_var)
            rb.pack(anchor="w", pady=1)
            self._engine_radios[value] = rb
        self._engine_hint = ttk.Label(eng, text="", foreground="gray", wraplength=580,
                                      justify="left")
        self._engine_hint.pack(anchor="w", pady=(4, 0))
        engine_var.trace_add("write", lambda *_: self._on_engine_change())

        vo = ttk.LabelFrame(f, text="Voice", padding=10)
        vo.pack(fill="x", pady=(10, 0))
        self._voice_frame = vo
        slot = ttk.Frame(vo)  # holds the voice combo of the selected engine
        slot.pack(fill="x")
        self._kokoro_row = ttk.Frame(slot)
        self._omni_row = ttk.Frame(slot)
        self._voice_combo = self._combo(self._kokoro_row, 0, "Voice", "voice", [], 44,
                                        readonly=True)
        self._omni_voice_combo = self._combo(self._omni_row, 0, "Voice", "omni_voice", [], 44,
                                             readonly=True)
        rows = ttk.Frame(vo)
        rows.pack(fill="x")
        self._entry(rows, 1, "Speed", "speed", 6, hint="0.5 .. 2.0")
        self._entry(rows, 2, "Pause between sentences (s)", "sentence_pause", 6)
        self._entry(rows, 3, "Volume", "volume", 6, hint="0 .. 1")
        self._device_combo = self._combo(rows, 4, "Output device", "device", [], 44)
        ttk.Label(rows, text="Leave blank for the Windows default speakers.",
                  foreground="gray").grid(row=5, column=1, columnspan=2, sticky="w")

        hr = ttk.LabelFrame(f, text="Croatian text", padding=10)
        hr.pack(fill="x", pady=(10, 0))
        self._check(hr, 0, "Switch to OmniVoice (Croatian) when the selected text looks Croatian",
                    "auto_language")
        self._croatian_combo = self._combo(hr, 1, "Croatian voice", "croatian_voice", [], 44,
                                           readonly=True)
        ttk.Label(hr, text="Needs OmniVoice to be installed (OmniVoice tab). Works whichever "
                  "engine is selected above.", foreground="gray", wraplength=580,
                  justify="left").grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

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

    def _omni_ready(self) -> bool:
        try:
            return bool(self.hooks.omni_present())
        except Exception:
            log.debug("OmniVoice check failed", exc_info=True)
            return False

    def _on_engine_change(self) -> None:
        """Show the voice combo of the selected engine."""
        omni = str(self._vars["engine"].get()) == "omnivoice"
        try:
            self._kokoro_row.pack_forget()
            self._omni_row.pack_forget()
            (self._omni_row if omni else self._kokoro_row).pack(fill="x")
        except tk.TclError:
            pass

    def _update_engine_state(self) -> None:
        """OmniVoice can only be picked once it is installed."""
        ready = self._omni_ready()
        try:
            self._engine_radios["omnivoice"].configure(state="normal" if ready else "disabled")
            self._engine_hint.configure(
                text="" if ready else "OmniVoice is not installed. Install it in the "
                "OmniVoice tab.")
        except tk.TclError:
            pass
        if not ready and str(self._vars["engine"].get()) == "omnivoice":
            self._engine_hint.configure(text="OmniVoice is not installed: Kokoro reads for now. "
                                        "Install it in the OmniVoice tab.")

    # -- Export ------------------------------------------------------------------
    def _build_export(self, f: ttk.Frame) -> None:
        src = ttk.LabelFrame(f, text="Text", padding=10)
        src.pack(fill="x")
        box_row = ttk.Frame(src)
        box_row.pack(fill="x")
        self._export_text = tk.Text(box_row, height=7, wrap="word", undo=True)
        bar = ttk.Scrollbar(box_row, orient="vertical", command=self._export_text.yview)
        self._export_text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self._export_text.pack(side="left", fill="x", expand=True)
        for seq in ("<KeyRelease>", "<<Paste>>", "<<Cut>>"):
            self._export_text.bind(seq, lambda e: self._export_text_changed())
        row = ttk.Frame(src)
        row.pack(fill="x", pady=(6, 0))
        self._export_open_btn = ttk.Button(row, text="Open document…",
                                           command=self._export_open)
        self._export_open_btn.pack(side="left")
        ttk.Button(row, text="Paste", command=self._export_paste).pack(side="left", padx=6)
        ttk.Button(row, text="Clear", command=self._export_clear).pack(side="left")
        self._export_info = ttk.Label(src, text=estimate_text(""), foreground="gray")
        self._export_info.pack(anchor="w", pady=(4, 0))

        vo = ttk.LabelFrame(f, text="Voice", padding=10)
        vo.pack(fill="x", pady=(10, 0))
        self._export_voice_ids: dict[str, str | None] = {AUTO_VOICE: None}
        self._export_voice = tk.StringVar(value=AUTO_VOICE)
        ttk.Label(vo, text="Voice").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 10))
        self._export_voice_combo = ttk.Combobox(
            vo, textvariable=self._export_voice, values=[AUTO_VOICE], width=44,
            state="readonly")
        self._export_voice_combo.grid(row=0, column=1, columnspan=2, sticky="w", pady=3)
        self._export_speed = tk.StringVar(value=_num(self.hooks.get_config().speech.speed))
        ttk.Label(vo, text="Speed").grid(row=1, column=0, sticky="w", pady=3, padx=(0, 10))
        ttk.Entry(vo, textvariable=self._export_speed, width=6).grid(
            row=1, column=1, sticky="w", pady=3)
        ttk.Label(vo, text="0.5 .. 2.0", foreground="gray").grid(
            row=1, column=2, sticky="w", padx=8)
        self._export_speed.trace_add("write", lambda *_: self._safe_ui(self._export_text_changed))
        self._export_lang = tk.StringVar(value="Auto")
        ttk.Label(vo, text="Language").grid(row=2, column=0, sticky="w", pady=3, padx=(0, 10))
        ttk.Combobox(vo, textvariable=self._export_lang, values=list(EXPORT_LANGS),
                     width=12, state="readonly").grid(row=2, column=1, sticky="w", pady=3)

        out = ttk.LabelFrame(f, text="Output", padding=10)
        out.pack(fill="x", pady=(10, 0))
        fmt_var = self._var("export_format", "mp3")
        ttk.Label(out, text="Format").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 10))
        fmt_row = ttk.Frame(out)
        fmt_row.grid(row=0, column=1, columnspan=2, sticky="w", pady=3)
        self._export_radios: dict[str, ttk.Radiobutton] = {}
        for value in EXPORT_FORMATS:
            rb = ttk.Radiobutton(fmt_row, text=value.upper(), value=value, variable=fmt_var)
            rb.pack(side="left", padx=(0, 12))
            self._export_radios[value] = rb
        self._export_bitrate = self._combo(out, 1, "MP3 bitrate (kbps)", "mp3_bitrate",
                                           [str(b) for b in MP3_BITRATES], 8, readonly=True)
        self._export_mp3_note = ttk.Label(out, text="", foreground="gray")
        self._export_mp3_note.grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Label(out, text="Folder").grid(row=3, column=0, sticky="w", pady=3, padx=(0, 10))
        folder_row = ttk.Frame(out)
        folder_row.grid(row=3, column=1, columnspan=2, sticky="we", pady=3)
        folder_var = self._var("export_folder", "")
        ttk.Entry(folder_row, textvariable=folder_var, width=40).pack(side="left")
        ttk.Button(folder_row, text="Browse…", command=self._export_browse).pack(
            side="left", padx=6)
        self._export_folder_hint = ttk.Label(out, text="", foreground="gray")
        self._export_folder_hint.grid(row=4, column=1, columnspan=2, sticky="w")
        folder_var.trace_add("write", lambda *_: self._safe_ui(self._update_folder_hint))

        run = ttk.LabelFrame(f, text="Export", padding=10)
        run.pack(fill="x", pady=(10, 0))
        btns = ttk.Frame(run)
        btns.pack(fill="x")
        self._export_btn = ttk.Button(btns, text="Export…", command=self._export_start,
                                      style="Accent.TButton" if sv_ttk else "TButton")
        self._export_btn.pack(side="left")
        self._export_cancel_btn = ttk.Button(btns, text="Cancel", command=self._export_cancel_now,
                                             state="disabled")
        self._export_cancel_btn.pack(side="left", padx=6)
        self._export_progress = ttk.Progressbar(btns, mode="determinate", length=220,
                                                maximum=100)
        self._export_progress.pack(side="left", padx=(12, 0))
        self._export_folder_btn = ttk.Button(btns, text="Open folder", state="disabled",
                                             command=self._export_open_folder)
        self._export_folder_btn.pack(side="right")
        self._export_status = ttk.Label(run, text="", wraplength=580, justify="left")
        self._export_status.pack(anchor="w", pady=(4, 0))
        self._export_result: Path | None = None

    def _refresh_export(self) -> None:
        """Voice list, MP3 availability and the default-folder hint."""
        try:
            voices = list(self.hooks.voices())
        except Exception:
            log.debug("voice list failed", exc_info=True)
            voices = []
        ids: dict[str, str | None] = {AUTO_VOICE: None}
        for v in voices:
            ids[f"{v}  {voice_label(v)}"] = v
        keep = self._export_voice_ids.get(self._export_voice.get())
        self._export_voice_ids = ids
        self._export_voice_combo.configure(values=list(ids))
        self._export_voice.set(next((d for d, v in ids.items() if v == keep), AUTO_VOICE))
        mp3 = self._mp3_ok()
        try:
            self._export_radios["mp3"].configure(state="normal" if mp3 else "disabled")
            self._export_bitrate.configure(state="readonly" if mp3 else "disabled")
            self._export_mp3_note.configure(text="" if mp3 else "MP3 needs the lameenc package")
        except tk.TclError:
            pass
        self._update_folder_hint()

    def _mp3_ok(self) -> bool:
        try:
            return bool(self.hooks.mp3_available())
        except Exception:
            return False

    def _export_default_folder(self) -> Path:
        cfg = self.hooks.get_config()
        cfg.export.folder = str(self._vars["export_folder"].get()).strip()
        return export_folder(cfg)

    def _update_folder_hint(self) -> None:
        if str(self._vars["export_folder"].get()).strip():
            self._export_folder_hint.configure(text="")
        else:
            self._export_folder_hint.configure(
                text=f"Default: {self._export_default_folder()}")

    def _export_browse(self) -> None:
        chosen = filedialog.askdirectory(parent=self.top, title="Export folder")
        if chosen:
            self._vars["export_folder"].set(chosen)

    def _export_set_text(self, text: str) -> None:
        self._export_text.delete("1.0", "end")
        self._export_text.insert("1.0", text)
        self._export_text_changed()

    def _export_text_changed(self) -> None:
        text = self._export_text.get("1.0", "end-1c")
        if not text.strip():
            self._export_title, self._export_markdown = "", True
        try:
            speed = float(str(self._export_speed.get()).strip() or 1.0)
        except ValueError:
            speed = 1.0
        self._export_info.configure(text=estimate_text(text, speed))

    def _export_clear(self) -> None:
        self._export_set_text("")
        self._export_title, self._export_markdown = "", True

    def _export_paste(self) -> None:
        try:
            clip = self.root.clipboard_get()
        except tk.TclError:
            return
        self._export_text.insert("insert", clip)
        self._export_text_changed()

    def _export_open(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in docload.SUPPORTED)
        types = [("Documents", patterns)] + [
            (ext[1:].upper(), f"*{ext}") for ext in docload.SUPPORTED]
        path = filedialog.askopenfilename(parent=self.top, title="Open document",
                                          filetypes=types)
        if not path:
            return
        self._export_open_btn.configure(state="disabled")
        self._export_status.configure(text="Reading the document…", foreground="")

        def run() -> None:
            doc: Any = None
            error: Exception | None = None
            try:
                doc = self.hooks.load_document(path)
            except Exception as e:  # noqa: BLE001
                error = e
                if not isinstance(e, docload.DocumentError):
                    log.exception("could not load the document")

            def finish() -> None:
                self._export_open_btn.configure(state="normal")
                if error is not None:
                    msg = str(error) or error.__class__.__name__
                    self._export_status.configure(text=f"Could not open: {msg}",
                                                 foreground="#d33")
                    messagebox.showerror("Open document", msg, parent=self.top)
                    return
                self._export_status.configure(text="")
                self._export_set_text(doc.text)
                self._export_title = doc.title
                self._export_markdown = bool(doc.markdown)
            self._post(lambda: self._safe_ui(finish))

        self._spawn(run)

    def _set_export_busy(self, busy: bool) -> None:
        self._export_busy = busy
        self._export_btn.configure(state="disabled" if busy else "normal")
        self._export_cancel_btn.configure(state="normal" if busy else "disabled")
        if busy:
            self._export_folder_btn.configure(state="disabled")
        else:
            self._export_progress.configure(value=0)

    def _export_format(self) -> str:
        fmt = str(self._vars["export_format"].get()).strip().lower()
        if fmt not in EXPORT_FORMATS or (fmt == "mp3" and not self._mp3_ok()):
            return "wav"
        return fmt

    def _export_start(self) -> None:
        if self._export_busy:
            return
        text = self._export_text.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showinfo("Export", "Type, paste or open some text first.",
                                parent=self.top)
            return
        try:
            speed = float(str(self._export_speed.get()).strip())
            if not 0.5 <= speed <= 2.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Export", "Speed must be a number from 0.5 to 2.0.",
                                 parent=self.top)
            return
        try:
            bitrate: int | None = snap_bitrate(int(str(self._vars["mp3_bitrate"].get()).strip()))
        except ValueError:
            bitrate = None
        fmt = self._export_format()
        title = self._export_title
        chosen = filedialog.asksaveasfilename(
            parent=self.top, title="Export audio", defaultextension=f".{fmt}",
            filetypes=[("WAV audio", "*.wav"), ("MP3 audio", "*.mp3")],
            initialdir=str(self._export_default_folder()),
            initialfile=f"{title or 'speech'}.{fmt}")
        if not chosen:
            return
        out = Path(chosen)
        if out.suffix.lower() not in (".wav", ".mp3"):
            out = out.with_name(f"{out.name}.{fmt}")
        req = ExportRequest(
            text=text, out=out, voice=self._export_voice_ids.get(self._export_voice.get()),
            speed=speed, lang=EXPORT_LANGS.get(self._export_lang.get()), fmt=None,
            bitrate=bitrate, markdown=self._export_markdown, gain=1.0, overwrite=True,
            title=title)
        cancel = self._export_cancel
        cancel.clear()
        self._export_result = None
        self._set_export_busy(True)
        self._export_status.configure(text="Starting…", foreground="")

        def progress(p: ExportProgress) -> None:
            def ui() -> None:
                if p.total:
                    self._export_progress.configure(value=100.0 * p.done / p.total)
                if p.paused:
                    text = "Paused while reading aloud"
                else:
                    text = f"Chunk {p.done}/{p.total} · {_clock(p.audio_seconds)} audio"
                self._export_status.configure(text=text, foreground="")
            self._post(lambda: self._safe_ui(ui))

        def run() -> None:
            result: Any = None
            error: BaseException | None = None
            try:
                result = self.hooks.export_audio(req, progress=progress, cancel=cancel)
            except Exception as e:  # noqa: BLE001
                error = e
                if not isinstance(e, (ExportCancelled, ExportBusy)):
                    log.exception("export failed")

            def finish() -> None:
                self._set_export_busy(False)
                if isinstance(error, ExportCancelled):
                    self._export_status.configure(text="Cancelled", foreground="")
                elif isinstance(error, ExportBusy):
                    self._export_status.configure(
                        text="Another export (from an agent) is running", foreground="#d33")
                elif error is not None:
                    msg = str(error) or error.__class__.__name__
                    self._export_status.configure(text=f"Failed: {msg}", foreground="#d33")
                    messagebox.showerror("Export failed", msg, parent=self.top)
                else:
                    path = Path(result.path)
                    self._export_result = path
                    self._export_progress.configure(value=100)
                    self._export_status.configure(
                        text=f"Saved {path.name} · {_clock(result.seconds)}", foreground="")
                    self._export_folder_btn.configure(state="normal")
            self._post(lambda: self._safe_ui(finish))

        self._spawn(run)

    def _export_cancel_now(self) -> None:
        self._export_cancel.set()
        try:
            self.hooks.cancel_export()
        except Exception:
            log.debug("cancel_export failed", exc_info=True)

    def _export_open_folder(self) -> None:
        path = self._export_result
        if path is not None:
            self._safe_call(lambda: self.hooks.open_folder(path.parent))()

    # -- MCP ---------------------------------------------------------------------
    def _build_mcp(self, f: ttk.Frame) -> None:
        box = ttk.LabelFrame(f, text="Local MCP server", padding=10)
        box.pack(fill="x")
        self._check(box, 0, "Let AI agents use VibeTTS (local MCP server)", "mcp_enabled")
        self._entry(box, 1, "Port", "mcp_port", 8, hint="1024 .. 65535")
        self._entry(box, 2, "Maximum characters", "mcp_max_chars", 10,
                    hint="per tool call, 0 = no limit")
        self._mcp_status = ttk.Label(box, text="", wraplength=580, justify="left")
        self._mcp_status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(box, text=f"{MCP_NOTE} {MCP_TOOLS}", foreground="gray", wraplength=580,
                  justify="left").grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

        conn = ttk.LabelFrame(f, text="Connect an agent", padding=10)
        conn.pack(fill="x", pady=(10, 0))
        ttk.Label(conn, text="Claude Code command").pack(anchor="w")
        row = ttk.Frame(conn)
        row.pack(fill="x", pady=(2, 8))
        self._mcp_cmd = tk.StringVar()
        ttk.Entry(row, textvariable=self._mcp_cmd, state="readonly", width=64).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Copy", command=lambda: self._copy(self._mcp_cmd.get())).pack(
            side="left", padx=(6, 0))
        ttk.Label(conn, text=".mcp.json").pack(anchor="w")
        self._mcp_json = tk.Text(conn, height=3, wrap="word", width=64)
        self._mcp_json.pack(fill="x", pady=(2, 4))
        ttk.Button(conn, text="Copy", command=lambda: self._copy(
            self._mcp_json.get("1.0", "end-1c"))).pack(anchor="w")
        self._vars["mcp_port"].trace_add("write", lambda *_: self._safe_ui(self._refresh_mcp))
        self._vars["mcp_enabled"].trace_add("write", lambda *_: self._safe_ui(self._refresh_mcp))
        self._refresh_mcp()

    def _copy(self, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except tk.TclError:
            pass

    def _refresh_mcp(self) -> None:
        url = mcp_url(self._vars["mcp_port"].get())
        self._mcp_cmd.set(mcp_command(url))
        self._mcp_json.configure(state="normal")
        self._mcp_json.delete("1.0", "end")
        self._mcp_json.insert("1.0", mcp_snippet(url))
        self._mcp_json.configure(state="disabled")
        try:
            st = dict(self.hooks.mcp_status())
        except Exception:
            log.debug("MCP status failed", exc_info=True)
            st = {}
        wanted = bool(self._vars["mcp_enabled"].get())
        if st.get("running"):
            text = f"Running at {st.get('url') or url}"
        elif st.get("error"):
            text = f"Error: {st['error']}"
        elif wanted and not st.get("enabled"):
            text = "Enabled — save to start"
        elif st.get("enabled"):
            text = "Starting…"
        else:
            text = "Off"
        self._mcp_status.configure(text=text)

    # -- Kokoro -----------------------------------------------------------------
    def _build_kokoro(self, f: ttk.Frame) -> None:
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

        mm = ttk.LabelFrame(f, text="Model files", padding=8)
        mm.pack(fill="both", expand=True)
        self._model_frame = mm
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

    # -- OmniVoice ---------------------------------------------------------------
    def _build_omnivoice(self, f: ttk.Frame) -> None:
        inst = ttk.LabelFrame(f, text="Install", padding=10)
        inst.pack(fill="x")
        self._omni_status = ttk.Label(inst, text="", wraplength=580, justify="left")
        self._omni_status.pack(anchor="w")
        self._omni_cloning = tk.BooleanVar(value=True)
        self._omni_clone_check = ttk.Checkbutton(
            inst, text=f"Include voice cloning (+{_fmt_size(CLONING_BYTES)})",
            variable=self._omni_cloning, command=self._update_omni_buttons)
        self._omni_clone_check.pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(inst)
        row.pack(fill="x", pady=(8, 0))
        self._omni_install_btn = ttk.Button(
            row, text="Install", command=self._omni_install,
            style="Accent.TButton" if sv_ttk else "TButton")
        self._omni_install_btn.pack(side="left")
        self._omni_cancel_btn = ttk.Button(row, text="Cancel", command=self._omni_cancel.set,
                                           state="disabled")
        self._omni_cancel_btn.pack(side="left", padx=6)
        self._omni_uninstall_btn = ttk.Button(row, text="Uninstall", command=self._omni_uninstall)
        self._omni_uninstall_btn.pack(side="left")
        self._omni_progress = ttk.Progressbar(row, mode="determinate", length=200, maximum=100)
        self._omni_progress.pack(side="left", padx=(12, 0))
        self._omni_phase = ttk.Label(inst, text="", wraplength=580, justify="left")
        self._omni_phase.pack(anchor="w", pady=(4, 0))
        ttk.Label(inst, text=OMNI_NOTE, foreground="gray", wraplength=580,
                  justify="left").pack(anchor="w", pady=(4, 0))

        q = ttk.LabelFrame(f, text="Quality", padding=10)
        q.pack(fill="x", pady=(10, 0))
        ttk.Label(q, text="Quality steps").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 10))
        lo, hi = OMNIVOICE_STEPS_RANGE
        steps_var = self._var("omnivoice_steps", "6")
        steps_row = ttk.Frame(q)
        steps_row.grid(row=0, column=1, sticky="w", pady=3)
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

        vf = ttk.LabelFrame(f, text="Voices", padding=10)
        vf.pack(fill="x", pady=(10, 0))
        cols = ("name", "kind", "description", "lang")
        self._voices_tree = ttk.Treeview(vf, columns=cols, show="headings", height=5,
                                         selectmode="browse")
        for c, w, t in (("name", 110, "Name"), ("kind", 90, "Kind"),
                        ("description", 270, "Description / source"), ("lang", 80, "Language")):
            self._voices_tree.heading(c, text=t)
            self._voices_tree.column(c, width=w, anchor="w")
        self._voices_tree.pack(fill="x")
        self._voices_tree.bind("<<TreeviewSelect>>", lambda e: self._update_voice_buttons())
        vrow = ttk.Frame(vf)
        vrow.pack(fill="x", pady=(6, 0))
        self._voice_btns: dict[str, ttk.Button] = {}
        for key, text, cmd in (("play", "Play sample", self._play_voice),
                               ("use", "Use as OmniVoice voice", self._use_selected),
                               ("hr", "Use for Croatian", self._use_selected_croatian),
                               ("delete", "Delete", self._delete_voice)):
            b = ttk.Button(vrow, text=text, command=cmd)
            b.pack(side="left", padx=(0, 6))
            self._voice_btns[key] = b
        self._voice_msg = ttk.Label(vf, text="", foreground="gray", wraplength=580,
                                    justify="left")
        self._voice_msg.pack(anchor="w", pady=(4, 0))

        cf = ttk.LabelFrame(f, text="Create voice", padding=10)
        cf.pack(fill="x", pady=(10, 0))
        self._creator = VoiceCreator(cf, self.hooks, self._post, self._voice_saved, self._spawn)
        self._creator.pack(fill="x")

    def _omni_info(self) -> list[dict]:
        try:
            return [dict(r) for r in self.hooks.omni_voice_info()]
        except Exception:
            log.debug("voice info failed", exc_info=True)
            return []

    def _selected_voice_name(self) -> str | None:
        sel = self._voices_tree.selection()
        return str(sel[0]) if sel else None

    def _update_voice_buttons(self) -> None:
        """Voice list buttons need an installed model and a selected row."""
        name = self._selected_voice_name()
        ready = self._omni_ready() and not self._omni_busy
        state = lambda ok: "normal" if ok else "disabled"  # noqa: E731
        try:
            self._voice_btns["play"].configure(state=state(ready and name))
            self._voice_btns["use"].configure(state=state(ready and name))
            self._voice_btns["hr"].configure(state=state(ready and name))
            self._voice_btns["delete"].configure(
                state=state(ready and name and name != "default"))
        except tk.TclError:
            pass

    def _play_voice(self) -> None:
        name = self._selected_voice_name()
        if not name:
            return
        try:
            self.hooks.play_preset(name)
        except Exception as e:  # noqa: BLE001
            self._voice_msg.configure(text=f"Could not play: {e}")

    def _use_selected(self) -> None:
        name = self._selected_voice_name()
        if name:
            self._use_voice("omni_voice", f"omni:{name}")

    def _use_selected_croatian(self) -> None:
        name = self._selected_voice_name()
        if name:
            self._use_voice("croatian_voice", f"omni:{name}")

    def _use_voice(self, key: str, voice_id: str) -> str | None:
        """Make `voice_id` the OmniVoice voice (omni_voice) or the Croatian one right away
        (saved through apply(); other unsaved edits in the form are left alone)."""
        err = None
        try:
            cfg = self.hooks.get_config()
            setattr(cfg.speech, key, voice_id)
            err = self.hooks.apply(cfg)
        except Exception as e:  # noqa: BLE001
            log.exception("could not set the voice")
            err = str(e)
        self._vars[key].set(voice_id)
        self._snapshot[key] = voice_id
        self._load_voices()
        label = "OmniVoice voice" if key == "omni_voice" else "Croatian voice"
        text = f"{label}: {omni_preset(voice_id)}."
        self._voice_msg.configure(text=f"{text} (but: {err})" if err else text)
        return err

    def _voice_saved(self, voice_id: str, use: bool) -> None:
        if use:
            self._use_voice("omni_voice", voice_id)
        else:
            self._load_voices()
        text = f"Added {omni_preset(voice_id)}."
        current = self._voice_msg.cget("text")
        self._voice_msg.configure(text=text if not use else f"{text} {current}")
        try:
            self._voices_tree.selection_set(omni_preset(voice_id))
        except tk.TclError:
            pass

    def _sync_voice_vars(self) -> None:
        """Re-read the voice settings the app may have changed (delete, tray, uninstall)."""
        form = form_from_config(self.hooks.get_config())
        for key in ("engine", "voice", "omni_voice", "croatian_voice"):
            if key in self._vars:
                self._vars[key].set(form[key])
                self._snapshot[key] = form[key]

    def _delete_voice(self) -> None:
        name = self._selected_voice_name()
        if not name or name == "default":
            return
        if not messagebox.askyesno("Delete voice", f"Delete the voice \"{name}\"?",
                                   parent=self.top):
            return
        try:
            self.hooks.stop()
            self.hooks.delete_voice(name)
        except Exception as e:  # noqa: BLE001
            self._voice_msg.configure(text=f"Could not delete: {e}")
            return
        self._sync_voice_vars()
        self._load_voices()
        self._voice_msg.configure(text=f"Deleted {name}.")

    # -- OmniVoice install ---------------------------------------------------------
    def _refresh_omni(self) -> None:
        """Status line, install buttons, engine radio, voice list and creator."""
        try:
            status = dict(self.hooks.omni_status())
        except Exception:
            log.debug("OmniVoice status failed", exc_info=True)
            status = {}
        installed = self._omni_ready()
        try:
            encoders = bool(self.hooks.encoders_present()) if installed else False
        except Exception:
            encoders = False
        self._omni_installed, self._omni_encoders = installed, encoders
        size = int(status.get("bytes_on_disk") or 0)
        if not installed:
            text = OMNI_MISSING
        else:
            text = "OmniVoice is installed" + (f" ({_fmt_size(size)} on disk)" if size else "")
            text += ", voice cloning available." if encoders else \
                ". Voice cloning is not installed."
        self._omni_status.configure(text=text)
        self._update_omni_buttons()
        self._update_engine_state()
        self._update_voice_buttons()
        self._creator.set_availability(installed, encoders)

    def _install_bytes(self, include: bool) -> int:
        try:
            return int(self.hooks.omni_download_bytes(include))
        except Exception:
            return OMNI_FALLBACK_BYTES + (CLONING_BYTES if include else 0)

    def _update_omni_buttons(self) -> None:
        busy = self._omni_busy
        installed = getattr(self, "_omni_installed", False)
        encoders = getattr(self, "_omni_encoders", False)
        add_cloning = installed and not encoders
        try:
            if add_cloning:
                self._omni_cloning.set(True)
            include = bool(self._omni_cloning.get())
            if add_cloning:
                text = f"Add voice cloning ({_fmt_size(self._install_bytes(True))})"
            elif installed:
                text = "Installed"
            else:
                text = f"Install (about {_fmt_size(self._install_bytes(include))})"
            can_install = not busy and (add_cloning or not installed)
            self._omni_install_btn.configure(text=text,
                                             state="normal" if can_install else "disabled")
            self._omni_clone_check.configure(
                state="normal" if not busy and not installed else "disabled")
            self._omni_uninstall_btn.configure(
                state="normal" if not busy and installed else "disabled")
            self._omni_cancel_btn.configure(state="normal" if busy else "disabled")
        except tk.TclError:
            pass

    def _set_omni_busy(self, busy: bool, text: str = "") -> None:
        self._omni_busy = busy
        self._omni_phase.configure(text=text)
        if not busy:
            self._omni_progress.configure(value=0)
        self._update_omni_buttons()
        self._update_voice_buttons()
        self._creator.set_availability(
            getattr(self, "_omni_installed", False) and not busy,
            getattr(self, "_omni_encoders", False))

    def _omni_progress_cb(self) -> Callable[[str, int, "int | None"], None]:
        def cb(text: str, done: int, total: "int | None") -> None:
            def ui() -> None:
                try:
                    if total:
                        self._omni_progress.configure(mode="determinate",
                                                      value=100.0 * done / total)
                    self._omni_phase.configure(text=text)
                except tk.TclError:
                    pass
            self._post(ui)
        return cb

    def _omni_install(self) -> None:
        include = bool(self._omni_cloning.get())
        self._omni_run("Installing OmniVoice...", "OmniVoice installed.",
                       lambda cancel: self.hooks.omni_install(
                           include, self._omni_progress_cb(), cancel),
                       "Installation cancelled.", "OmniVoice install failed")

    def _omni_uninstall(self) -> None:
        if self._omni_busy:
            return
        if not messagebox.askyesno(
                "Uninstall OmniVoice",
                "Delete the OmniVoice model files? Your custom voices are kept, and "
                "installing again restores it.", parent=self.top):
            return
        self._omni_run("Removing OmniVoice...", "OmniVoice removed.",
                       lambda cancel: (self.hooks.stop(), self.hooks.omni_uninstall()),
                       "", "Uninstall failed")

    def _omni_run(self, busy_text: str, done_text: str, job: Callable[[threading.Event], Any],
                  cancelled_text: str, error_title: str) -> None:
        """Run `job(cancel)` on a worker thread and refresh everything when it ends."""
        if self._omni_busy:
            return
        cancel = self._omni_cancel
        cancel.clear()
        self._set_omni_busy(True, busy_text)

        def run() -> None:
            error: Exception | None = None
            try:
                job(cancel)
            except Exception as e:  # noqa: BLE001
                error = e
                if not isinstance(e, (model_setup.DownloadCancelled, model_setup.SetupError)):
                    log.exception("OmniVoice setup failed")

            def finish() -> None:
                self._set_omni_busy(False)
                self._sync_voice_vars()
                self._refresh_omni()
                self._load_voices()
                if isinstance(error, model_setup.DownloadCancelled):
                    self._omni_phase.configure(text=cancelled_text)
                    self._set_status(cancelled_text)
                elif error is not None:
                    msg = str(error) or error.__class__.__name__
                    self._omni_phase.configure(text=f"Failed: {msg}")
                    messagebox.showerror(error_title, msg, parent=self.top)
                else:
                    self._omni_phase.configure(text=done_text)
                    self._set_status(done_text)
            self._post(lambda: self._safe_ui(finish))

        self._spawn(run)

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
            self._setup.pack(fill="x", pady=(0, 10), before=self._model_frame)
        elif not self._busy:
            self._setup.pack_forget()

    def _load_voices(self) -> None:
        """Fill the Kokoro / OmniVoice / Croatian combos and the voices list."""
        try:
            voices = [v for v in self.hooks.voices() if not is_omni_voice(v)]
        except Exception:
            log.debug("voice list failed", exc_info=True)
            voices = []
        current = _voice_token(self._vars["voice"].get())
        if current and current not in voices:
            voices.insert(0, current)
        display = {v: f"{v}  {voice_label(v)}" for v in voices}
        self._voice_combo.configure(values=[display[v] for v in voices])
        if current:
            self._vars["voice"].set(display.get(current, current))

        try:
            presets = list(self.hooks.omni_voices())
        except Exception:
            log.debug("OmniVoice voice list failed", exc_info=True)
            presets = []
        shown = list(presets)
        omni = _voice_token(self._vars["omni_voice"].get())
        if omni and omni not in shown:
            shown.insert(0, omni)
        omni_display = {v: f"{v}  {voice_label(v)}" for v in shown}
        self._omni_voice_combo.configure(values=[omni_display[v] for v in shown])
        if omni:
            self._vars["omni_voice"].set(omni_display.get(omni, omni))

        hr = _voice_token(self._vars["croatian_voice"].get())
        hr_shown = list(presets)
        if hr and hr not in hr_shown:
            hr_shown.insert(0, hr)
        hr_display = {v: f"{v}  {voice_label(v)}" for v in hr_shown}
        self._croatian_combo.configure(values=[SAME_VOICE, *(hr_display[v] for v in hr_shown)])
        self._vars["croatian_voice"].set(hr_display.get(hr, hr) if hr else SAME_VOICE)

        selected = self._selected_voice_name()
        self._voices_tree.delete(*self._voices_tree.get_children())
        for row in self._omni_info():
            self._voices_tree.insert("", "end", iid=row["name"], values=(
                row["name"], KIND_LABELS.get(row.get("kind", ""), row.get("kind", "")),
                row.get("description", ""), _lang_name(row.get("lang", ""))))
        if selected and self._voices_tree.exists(selected):
            self._voices_tree.selection_set(selected)
        self._update_voice_buttons()

    def _refresh_files(self) -> None:
        self._refresh_omni()
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
        ttk.Label(f, text="Local text-to-speech with Kokoro-82M (Apache 2.0) or OmniVoice "
                  "(any language incl. Croatian; weights CC-BY-NC, non-commercial use only).",
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
        self._refresh_export()
        self._refresh_mcp()
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
        for key in ("voice", "omni_voice", "croatian_voice"):
            if key in form:
                form[key] = _voice_token(form[key])
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


# ---------------------------------------------------------------- Create voice (inline)

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
ENCODERS_MISSING = ("Voice cloning is not installed: tick \"Include voice cloning\" in the "
                    "Install section and install it")
NOT_INSTALLED = "Install OmniVoice (above) to create voices"
DESIGN_BUSY = "Generating… ~5 s"
CLONE_BUSY = "Creating the voice… ~10 s"
SAMPLE_LANGS = {"Croatian": "hr", "English": "en"}
DEFAULT_SAMPLE_LANG = "Croatian"  # measured best for both languages, see the VTF-03 notes


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


class VoiceCreator(ttk.Frame):
    """Inline 'Create voice' panel: design a voice from a description or clone one from a
    recording, preview it, then save it as a preset. Generation runs on a worker thread
    (`spawn`); results come back through `post` (the Tk thread). `on_saved(voice_id, use)`
    runs after a save; `set_availability` greys the panel out until the model is installed."""

    def __init__(self, master: tk.Misc, hooks: SettingsHooks,
                 post: Callable[[Callable[[], None]], None],
                 on_saved: Callable[[str, bool], None] = lambda voice_id, use: None,
                 spawn: Callable[[Callable[[], None]], None] = _start_thread):
        super().__init__(master)
        self.hooks = hooks
        self._post = post
        self._on_saved = on_saved
        self._spawn = spawn
        self._busy = False
        self.installed = True
        self._clips: dict[str, Any] = {"design": None, "clone": None}
        self.options = instruct_options()
        try:
            self.encoders = bool(hooks.encoders_present())
        except Exception:
            self.encoders = False

        namerow = ttk.Frame(self)
        namerow.pack(fill="x")
        ttk.Label(namerow, text="Voice name").pack(side="left", padx=(0, 10))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(namerow, textvariable=self.name_var, width=26)
        self.name_entry.pack(side="left")
        ttk.Label(namerow, text="letters, digits, _ and -", foreground="gray").pack(
            side="left", padx=8)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="x", pady=(10, 0))
        self._design = ttk.Frame(self.nb, padding=10)
        self._clone = ttk.Frame(self.nb, padding=10)
        self.nb.add(self._design, text="From description")
        self.nb.add(self._clone, text="From recording")
        self._buttons: dict[str, ttk.Button] = {}
        self._build_design(self._design)
        self._build_clone(self._clone)
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._update_buttons())

        self.use_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Use as OmniVoice voice after saving",
                        variable=self.use_var).pack(anchor="w", pady=(10, 0))
        self.message = ttk.Label(self, text="", wraplength=560, justify="left")
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
        ttk.Label(f, text="Sample language").grid(row=row, column=0, sticky="w", pady=3,
                                                  padx=(0, 10))
        self.lang_var = tk.StringVar(value=DEFAULT_SAMPLE_LANG)
        self.lang_combo = ttk.Combobox(f, textvariable=self.lang_var, values=list(SAMPLE_LANGS),
                                       state="readonly", width=22)
        self.lang_combo.grid(row=row, column=1, sticky="w", pady=3)
        ttk.Label(f, text="the language of the short clip the voice is made from",
                  foreground="gray").grid(row=row, column=2, sticky="w", padx=8)
        ttk.Label(f, text="Seed").grid(row=row + 1, column=0, sticky="w", pady=3, padx=(0, 10))
        self.seed_var = tk.StringVar(value="0")
        ttk.Spinbox(f, from_=0, to=2_147_483_647, textvariable=self.seed_var, width=12).grid(
            row=row + 1, column=1, sticky="w", pady=3)
        ttk.Label(f, text="Leave everything on (any) for a plain random voice.",
                  foreground="gray", wraplength=480, justify="left").grid(
            row=row + 2, column=0, columnspan=3, sticky="w", pady=(2, 8))
        btns = ttk.Frame(f)
        btns.grid(row=row + 3, column=0, columnspan=3, sticky="w")
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
                                        foreground="#d33", wraplength=500, justify="left")
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
    def set_availability(self, installed: bool, encoders: bool) -> None:
        """Called when the OmniVoice install state changes."""
        self.installed, self.encoders = bool(installed), bool(encoders)
        try:
            self.encoders_label.configure(text="" if self.encoders else ENCODERS_MISSING)
            if not self.installed and not self._busy:
                self._info(NOT_INSTALLED)
            elif self.installed and self.message.cget("text") == NOT_INSTALLED:
                self._info("")
        except tk.TclError:
            pass
        self._update_buttons()

    def current_tab(self) -> str:
        try:
            return "design" if self.nb.index(self.nb.select()) == 0 else "clone"
        except tk.TclError:
            return "design"

    def _update_buttons(self) -> None:
        for key, b in self._buttons.items():
            if self._busy:
                state = "disabled"
            elif not self.installed and key != "browse":
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
        try:
            self.message.configure(text=text, foreground="#d33" if error else "")
        except tk.TclError:
            pass

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self._busy = busy
        if text or not busy:
            self._info(text)
        self._update_buttons()

    def _transcript_text(self) -> str:
        return self.transcript.get("1.0", "end").strip()

    # -- generation (worker thread)
    def _run(self, work: Callable[[], Any], tab: str, busy_text: str) -> None:
        if self._busy or not self.installed:
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

    def sample_lang(self) -> str:
        return SAMPLE_LANGS.get(str(self.lang_var.get()).strip(), "hr")

    def generate(self) -> None:
        try:
            seed = int(str(self.seed_var.get()).strip())
            if seed < 0:
                raise ValueError
        except ValueError:
            self._info("Seed must be a whole number, 0 or more.", error=True)
            return
        instruct, lang = self.instruct(), self.sample_lang()
        self._run(lambda: self.hooks.design_clip(instruct, seed, None, lang), "design",
                  DESIGN_BUSY)

    def try_another(self) -> None:
        if self._busy:
            return
        self.seed_var.set(str(random.randint(0, 99999)))
        self.generate()

    def browse(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Choose a recording",
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
            existing = {v.split(":", 1)[-1] for v in self.hooks.omni_voices()}
        except Exception:
            existing = set()
        if name in existing and not messagebox.askyesno(
                "Replace voice", f"A voice named \"{name}\" already exists. Replace it?",
                parent=self):
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
        self._clips[tab] = None  # saved: ready for the next voice
        self.name_var.set("")
        self._info(f"Saved {name}.")
        self._update_buttons()
        try:
            self._on_saved(voice_id, use)
        except Exception:
            log.exception("after-save update failed")
        return True

    def destroy(self) -> None:
        self._stop()
        super().destroy()
