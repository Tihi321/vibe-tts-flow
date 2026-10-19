import copy
import tkinter as tk
from pathlib import Path

import pytest

from vibetts import settings_ui
from vibetts.config import Config
from vibetts.docload import Document, DocumentError
from vibetts.export import (ExportBusy, ExportCancelled, ExportProgress, ExportRequest,
                            ExportResult)
from vibetts.settings_ui import (TABS, SettingsHooks, SettingsWindow, config_from_form,
                                 form_from_config, model_choice)


def test_round_trip_default():
    cfg = Config()
    assert config_from_form(cfg, form_from_config(cfg)) == cfg


def test_round_trip_customised():
    cfg = Config()
    cfg.hotkey.read = "ctrl+shift+r"
    cfg.speech.model = "D:\\tts\\kokoro-v1.0.onnx"
    cfg.speech.voice = "bf_emma"
    cfg.speech.speed = 1.35
    cfg.speech.sentence_pause = 0
    cfg.speech.device = "Headphones"
    cfg.selection.max_chars = 0
    cfg.selection.restore_clipboard = False
    cfg.ui.pill_position = "top"
    assert config_from_form(cfg, form_from_config(cfg)) == cfg


def test_model_switch_keeps_folder():
    cfg = Config()
    form = form_from_config(cfg)
    assert form["model"] == "int8"
    form["model"] = "fp32"
    assert config_from_form(cfg, form).speech.model == "models/kokoro-v1.0.onnx"


def test_custom_model_kept():
    cfg = Config()
    cfg.speech.model = "other/custom.onnx"
    assert model_choice(cfg.speech.model) == "custom.onnx"
    assert config_from_form(cfg, form_from_config(cfg)).speech.model == "other/custom.onnx"


def test_voice_display_string_parsed():
    form = form_from_config(Config())
    form["voice"] = "bm_george  George (British English, male)"
    assert config_from_form(Config(), form).speech.voice == "bm_george"


@pytest.mark.parametrize("key,value", [
    ("hotkey_read", "ctrl+bogus"), ("speed", "3"), ("speed", "0.1"), ("volume", "1.5"),
    ("sentence_pause", "-1"), ("copy_timeout_ms", "10"), ("max_chars", "-1"),
    ("pill_position", "left"), ("voice", ""), ("model", "bogus"),
])
def test_invalid_values(key, value):
    form = form_from_config(Config())
    form[key] = value
    with pytest.raises(ValueError):
        config_from_form(Config(), form)


class Recorder:
    def __init__(self):
        self.calls, self.applied = [], []
        self.config = Config()
        self.mp3 = True
        self.mcp = {"enabled": False, "running": False, "url": "", "error": ""}
        self.doc = Document("Hello there.", True, "notes")
        self.exports, self.export_error = [], None

    def hooks(self):
        def apply(cfg):
            self.applied.append(cfg)
            self.config = copy.deepcopy(cfg)
            return None

        return SettingsHooks(
            get_config=lambda: copy.deepcopy(self.config), apply=apply,
            is_autostart=lambda: False, set_autostart=lambda v: None,
            set_show_on_start=lambda v: None, models_dir=lambda: Path("."),
            voices=lambda: ["af_heart", "bf_emma"],
            output_devices=lambda: [], play_test=lambda c, t: None,
            stop=lambda: self.calls.append("stop"), open_logs=lambda: None,
            open_config=lambda: None, version="t",
            load_document=self._load_document, export_audio=self._export_audio,
            cancel_export=lambda: self.calls.append("cancel_export"),
            mp3_available=lambda: self.mp3,
            mcp_status=lambda: dict(self.mcp),
            open_folder=lambda p: self.calls.append(("open_folder", p)))

    def _load_document(self, path):
        self.calls.append(("load", path))
        if isinstance(self.doc, Exception):
            raise self.doc
        return self.doc

    def _export_audio(self, req, progress=None, cancel=None):
        self.exports.append(req)
        if progress:
            progress(ExportProgress(2, 10, 192.0))
        if self.export_error:
            raise self.export_error
        return ExportResult(path=req.out, fmt="mp3", seconds=192.0, chunks=10, bytes=1,
                            voice="af_heart")


@pytest.fixture(scope="module")
def _shared_root():
    # one Tk interpreter for the module: creating/destroying many is flaky on Windows
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display for Tk: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def tk_root(_shared_root):
    yield _shared_root
    for child in _shared_root.winfo_children():
        child.destroy()


def make_window(root, rec, spawn=lambda fn: fn()):
    win = SettingsWindow(root, rec.hooks(), lambda fn: fn(), spawn)
    try:
        win.show("voice")
    except tk.TclError:
        pytest.skip("settings window cannot be built headless")
    return win


# ---------------------------------------------------------------- settings window

def state(widget):
    return str(widget.cget("state"))


def test_tabs(tk_root):
    assert TABS == ("general", "voice", "export", "kokoro", "mcp", "about")
    win = make_window(tk_root, Recorder())
    assert [win._nb.tab(i, "text") for i in range(len(TABS))] == [
        "General", "Voice", "Export", "Kokoro", "Agents (MCP)", "About"]


def test_voice_combo(tk_root):
    win = make_window(tk_root, Recorder())
    values = list(win._voice_combo.cget("values"))
    assert [v.split(" ")[0] for v in values] == ["af_heart", "bf_emma"]
    assert win._vars["voice"].get().startswith("af_heart  Heart")
    assert win._collect()["voice"] == "af_heart"
    assert not win._is_dirty()


def test_refresh_follows_tray_changes_unless_edited(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    rec.config.speech.voice = "bf_emma"  # changed from the tray
    win.refresh()
    assert win._vars["voice"].get().startswith("bf_emma") and not win._is_dirty()
    win._vars["voice"].set("af_heart")  # the user edits it
    rec.config.speech.voice = "bf_emma"
    win.refresh()
    assert win._vars["voice"].get().startswith("af_heart")  # the edit is kept


def test_save_round_trips_voice(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    win._vars["voice"].set("bf_emma  Emma (British English, female)")
    assert win.save() is True
    assert rec.applied[-1].speech.voice == "bf_emma"
    assert not win._is_dirty()


def test_window_fits_a_1080p_screen(tk_root):
    win = make_window(tk_root, Recorder())
    for tab in ("voice", "export"):
        win._nb.select(TABS.index(tab))
        win.top.update_idletasks()
        assert win.top.winfo_reqheight() <= 1080 - 90

# ---------------------------------------------------------------- export / mcp tabs

def test_export_mcp_form_round_trip():
    cfg = Config()
    form = form_from_config(cfg)
    assert form["export_format"] == "mp3" and form["mp3_bitrate"] == "64"
    cfg.export.folder, cfg.export.format, cfg.export.mp3_bitrate = "D:/audio", "wav", 128
    cfg.mcp.enabled, cfg.mcp.port, cfg.mcp.max_chars = True, 8000, 0
    assert config_from_form(cfg, form_from_config(cfg)) == cfg


def test_export_bitrate_snaps_and_bad_values_rejected():
    form = form_from_config(Config())
    form["mp3_bitrate"] = "100"
    assert config_from_form(Config(), form).export.mp3_bitrate == 96
    for key, value in (("mcp_port", "80"), ("mcp_port", "70000"), ("mcp_port", "x"),
                       ("mcp_max_chars", "-1"), ("export_format", "ogg"),
                       ("mp3_bitrate", "x")):
        bad = form_from_config(Config())
        bad[key] = value
        with pytest.raises(ValueError):
            config_from_form(Config(), bad)
    bad = form_from_config(Config())
    bad["mcp_port"] = "80"
    with pytest.raises(ValueError, match="MCP port must be at least 1024"):
        config_from_form(Config(), bad)


def test_show_export_tab(tk_root):
    win = make_window(tk_root, Recorder())
    win.show("export")
    assert win._nb.index(win._nb.select()) == TABS.index("export")
    win.show("mcp")
    assert win._nb.index(win._nb.select()) == TABS.index("mcp")


def export_win(tk_root, monkeypatch, rec=None, save_to=None):
    rec = rec or Recorder()
    win = make_window(tk_root, rec)
    if save_to is not None:
        monkeypatch.setattr(settings_ui.filedialog, "asksaveasfilename", lambda **k: str(save_to))
    return rec, win


def test_open_document_fills_text(tk_root, monkeypatch):
    rec, win = export_win(tk_root, monkeypatch)
    monkeypatch.setattr(settings_ui.filedialog, "askopenfilename", lambda **k: "C:/x/notes.md")
    win._export_open()
    assert rec.calls[-1] == ("load", "C:/x/notes.md")
    assert win._export_text.get("1.0", "end-1c") == "Hello there."
    assert win._export_title == "notes" and win._export_markdown is True
    assert "12 characters" in win._export_info.cget("text")
    assert state(win._export_open_btn) == "normal"
    win._export_clear()
    assert win._export_title == "" and win._export_text.get("1.0", "end-1c") == ""


def test_open_document_cancelled_and_error(tk_root, monkeypatch):
    rec, win = export_win(tk_root, monkeypatch)
    monkeypatch.setattr(settings_ui.filedialog, "askopenfilename", lambda **k: "")
    win._export_open()
    assert rec.calls == [] and win._export_text.get("1.0", "end-1c") == ""
    errors = []
    monkeypatch.setattr(settings_ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(settings_ui.filedialog, "askopenfilename", lambda **k: "C:/x/a.pdf")
    rec.doc = DocumentError("This PDF has no text")
    win._export_open()
    assert errors and "no text" in errors[0][1]
    assert win._export_text.get("1.0", "end-1c") == ""


def test_export_runs_hook_with_request(tk_root, monkeypatch, tmp_path):
    out = tmp_path / "x.mp3"
    rec, win = export_win(tk_root, monkeypatch, save_to=out)
    win._export_set_text("Some text to read.")
    win._export_speed.set("1.25")
    win._vars["mp3_bitrate"].set("96")
    win._export_start()
    assert len(rec.exports) == 1
    req = rec.exports[0]
    assert isinstance(req, ExportRequest)
    assert (req.text, req.out, req.voice, req.bitrate, req.speed) == (
        "Some text to read.", out, None, 96, 1.25)
    assert req.fmt is None and req.overwrite is True and req.markdown is True
    assert win._export_status.cget("text") == "Saved x.mp3 \u00b7 3:12"
    assert state(win._export_btn) == "normal" and state(win._export_cancel_btn) == "disabled"
    assert state(win._export_folder_btn) == "normal"
    win._export_folder_btn.invoke()
    assert rec.calls[-1] == ("open_folder", tmp_path)
    win._vars["mp3_bitrate"].set("64")
    assert not win._is_dirty()  # transient export controls are not settings


def test_export_voice_choice_and_missing_extension(tk_root, monkeypatch, tmp_path):
    rec, win = export_win(tk_root, monkeypatch, save_to=tmp_path / "speech")
    win._export_set_text("Hi")
    win._export_voice.set(next(d for d in win._export_voice_ids if d.startswith("bf_emma")))
    win._vars["export_format"].set("wav")
    win._export_start()
    req = rec.exports[0]
    assert req.voice == "bf_emma" and req.out == tmp_path / "speech.wav"


def spy_status(win):
    labels = []
    real = win._export_status.configure
    win._export_status.configure = lambda **kw: (labels.append(kw.get("text")), real(**kw))
    return labels


def test_export_progress_label(tk_root, monkeypatch, tmp_path):
    pending = []
    rec = Recorder()
    win = make_window(tk_root, rec, spawn=pending.append)
    monkeypatch.setattr(settings_ui.filedialog, "asksaveasfilename",
                        lambda **k: str(tmp_path / "a.wav"))
    win._export_set_text("Hi")
    labels = spy_status(win)
    win._export_start()
    assert state(win._export_btn) == "disabled" and state(win._export_cancel_btn) == "normal"
    pending.pop()()
    assert "Chunk 2/10 \u00b7 3:12 audio" in labels


def test_export_paused_label(tk_root, monkeypatch, tmp_path):
    rec = Recorder()
    orig = rec._export_audio

    def paused(req, progress=None, cancel=None):
        progress(ExportProgress(3, 10, 5.0, paused=True))
        return orig(req)
    make_hooks = rec.hooks

    def hooks():
        h = make_hooks()
        h.export_audio = paused
        return h
    rec.hooks = hooks
    win = make_window(tk_root, rec)
    monkeypatch.setattr(settings_ui.filedialog, "asksaveasfilename",
                        lambda **k: str(tmp_path / "a.wav"))
    win._export_set_text("Hi")
    labels = spy_status(win)
    win._export_start()
    assert "Paused while reading aloud" in labels


def test_export_busy_cancelled_and_failed(tk_root, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(settings_ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    rec, win = export_win(tk_root, monkeypatch, save_to=tmp_path / "a.wav")
    win._export_set_text("Hi")
    rec.export_error = ExportBusy("busy")
    win._export_start()
    assert "Another export" in win._export_status.cget("text") and errors == []
    assert state(win._export_btn) == "normal"
    rec.export_error = ExportCancelled()
    win._export_start()
    assert win._export_status.cget("text") == "Cancelled"
    rec.export_error = RuntimeError("disk full")
    win._export_start()
    assert "disk full" in win._export_status.cget("text") and errors
    assert state(win._export_btn) == "normal"


def test_export_blank_text_does_not_call_hook(tk_root, monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr(settings_ui.messagebox, "showinfo", lambda *a, **k: shown.append(a))
    rec, win = export_win(tk_root, monkeypatch, save_to=tmp_path / "a.wav")
    win._export_set_text("   ")
    win._export_start()
    assert rec.exports == [] and shown


def test_export_save_dialog_cancelled(tk_root, monkeypatch):
    rec, win = export_win(tk_root, monkeypatch, save_to="")
    win._export_set_text("Hi")
    win._export_start()
    assert rec.exports == [] and not win._export_busy


def test_export_cancel_button(tk_root, monkeypatch, tmp_path):
    pending = []
    rec = Recorder()
    win = make_window(tk_root, rec, spawn=pending.append)
    monkeypatch.setattr(settings_ui.filedialog, "asksaveasfilename",
                        lambda **k: str(tmp_path / "a.wav"))
    win._export_set_text("Hi")
    win._export_start()
    win._export_cancel_btn.invoke()
    assert win._export_cancel.is_set() and "cancel_export" in rec.calls
    pending.pop()()


def test_mp3_unavailable_disables_mp3(tk_root, monkeypatch, tmp_path):
    rec = Recorder()
    rec.mp3 = False
    win = make_window(tk_root, rec)
    assert state(win._export_radios["mp3"]) == "disabled"
    assert state(win._export_bitrate) == "disabled"
    assert "lameenc" in win._export_mp3_note.cget("text")
    monkeypatch.setattr(settings_ui.filedialog, "asksaveasfilename",
                        lambda **k: str(tmp_path / "a"))
    win._export_set_text("Hi")  # form says mp3, but only wav is possible
    win._export_start()
    assert rec.exports[0].out == tmp_path / "a.wav"


def test_export_folder_hint_when_empty(tk_root):
    win = make_window(tk_root, Recorder())
    assert "VibeTTS" in win._export_folder_hint.cget("text")
    win._vars["export_folder"].set("D:/audio")
    assert win._export_folder_hint.cget("text") == ""


@pytest.mark.parametrize("status,enabled,expected", [
    ({"enabled": True, "running": True, "url": "http://127.0.0.1:7723/mcp", "error": ""}, True,
     "Running at http://127.0.0.1:7723/mcp"),
    ({"enabled": False, "running": False, "url": "", "error": ""}, False, "Off"),
    ({"enabled": True, "running": False, "url": "", "error": "port in use"}, True,
     "Error: port in use"),
    ({"enabled": False, "running": False, "url": "", "error": ""}, True,
     "Enabled \u2014 save to start"),
])
def test_mcp_status_text(tk_root, status, enabled, expected):
    rec = Recorder()
    rec.mcp = status
    rec.config.mcp.enabled = enabled
    win = make_window(tk_root, rec)
    assert win._mcp_status.cget("text") == expected


def test_mcp_command_follows_port_and_copies(tk_root):
    win = make_window(tk_root, Recorder())
    assert "http://127.0.0.1:7723/mcp" in win._mcp_cmd.get()
    win._vars["mcp_port"].set("8123")
    assert win._mcp_cmd.get() == "claude mcp add --transport http vibetts http://127.0.0.1:8123/mcp"
    snippet = win._mcp_json.get("1.0", "end-1c")
    assert snippet == ('{"mcpServers": {"vibetts": {"type": "http", '
                       '"url": "http://127.0.0.1:8123/mcp"}}}')
    win._copy(snippet)
    assert tk_root.clipboard_get() == snippet


def test_mcp_fields_save_and_dirty(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    assert not win._is_dirty()
    win._vars["mcp_enabled"].set(True)
    win._vars["mcp_port"].set("8123")
    assert win._is_dirty()
    assert win.save() is True
    assert rec.config.mcp.enabled and rec.config.mcp.port == 8123
    assert not win._is_dirty()
