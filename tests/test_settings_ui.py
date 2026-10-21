import copy
import tkinter as tk
from pathlib import Path

import pytest

from vibetts import settings_ui
from vibetts import model_setup
from vibetts.config import (MP3_BITRATES, SUPERTONIC_LANGUAGE_NAMES, SUPERTONIC_LANGUAGES,
                            SUPERTONIC_VOICES, Config)
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
    cfg.speech.language_voices["en"] = "bf_emma"
    cfg.speech.language_voices["es"] = "ef_dora"
    cfg.speech.language_voices["de"] = ""
    cfg.speech.read_language = "es"
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
    form["language_voices"] = {**form["language_voices"],
                               "en": "bm_george  George (British English, male)"}
    assert config_from_form(Config(), form).speech.language_voices["en"] == "bm_george"


def _form_with(**voices):
    form = form_from_config(Config())
    form["language_voices"] = {**form["language_voices"], **voices}
    return form


@pytest.mark.parametrize("key,value", [
    ("hotkey_read", "ctrl+bogus"), ("speed", "3"), ("speed", "0.1"), ("volume", "1.5"),
    ("sentence_pause", "-1"), ("copy_timeout_ms", "10"), ("max_chars", "-1"),
    ("pill_position", "left"), ("model", "bogus"), ("read_language", "Always Klingon"),
    ("language_voices", "af_heart"), ("supertonic_steps", "3"), ("supertonic_steps", "17"),
    ("supertonic_steps", "x"),
])
def test_invalid_values(key, value):
    form = form_from_config(Config())
    form[key] = value
    with pytest.raises(ValueError):
        config_from_form(Config(), form)


def test_language_voices_round_trip():
    cfg = Config()
    form = form_from_config(cfg)
    assert form["language_voices"] == cfg.speech.language_voices
    assert form["language_voices"] is not cfg.speech.language_voices
    assert (form["read_language"], form["supertonic_steps"]) == ("auto", "8")
    cfg.speech.language_voices.update(en="supertonic:M3", es="ef_dora", de="", hr="supertonic:F5")
    cfg.speech.read_language = "hr"
    cfg.speech.supertonic_steps = 12
    assert config_from_form(Config(), form_from_config(cfg)) == cfg


def test_form_accepts_display_strings_and_labels():
    form = _form_with(hr="supertonic:M2  Supertonic M2 (male)", es="ef_dora  Dora")
    form["read_language"] = "Always Spanish"
    s = config_from_form(Config(), form).speech
    assert (s.language_voices["hr"], s.language_voices["es"], s.read_language) == (
        "supertonic:M2", "ef_dora", "es")


def test_invalid_voice_for_language():
    for lang, voice in (("hr", "af_heart"), ("es", "af_heart"), ("en", "ef_dora"),
                        ("en", "jf_alpha"), ("hr", "supertonic:F9"), ("hr", "bogus")):
        with pytest.raises(ValueError, match="cannot read that language"):
            config_from_form(Config(), _form_with(**{lang: voice}))


def test_all_languages_off_and_read_must_be_on():
    form = form_from_config(Config())
    form["language_voices"] = {c: "" for c in SUPERTONIC_LANGUAGES}
    with pytest.raises(ValueError, match="at least one language"):
        config_from_form(Config(), form)
    form = _form_with(de="")
    form["read_language"] = "Always German"
    with pytest.raises(ValueError, match="Read must be"):
        config_from_form(Config(), form)
    form["read_language"] = "auto"
    assert config_from_form(Config(), form).speech.language_voices["de"] == ""


def test_read_and_language_labels():
    assert settings_ui.read_label("auto") == "Auto (detect language)"
    assert settings_ui.read_label("hr") == "Always Croatian"
    assert settings_ui.read_from_label("Always Croatian") == "hr"
    assert settings_ui.read_from_label("Auto (detect language)") == "auto"
    assert settings_ui.read_from_label("hr") == "hr"
    assert settings_ui.language_label("hr", "supertonic:F1") == "Croatian"
    assert settings_ui.language_label("hr", "") == "Croatian (off)"
    assert settings_ui.language_from_label("Croatian (off)") == "hr"
    assert settings_ui.language_from_label("Nope") == ""
    assert list(settings_ui.LANGUAGE_CODES.values())[0] == "en"
    names = list(settings_ui.LANGUAGE_CODES)
    assert names[1:] == sorted(names[1:]) and len(names) == 31


def test_invalid_messages_are_readable():
    for key, value, text in (("supertonic_steps", "3", "Quality steps must be at least 4"),
                             ("supertonic_steps", "17", "Quality steps must be at most 16"),
                             ("language_voices", None, "Voices must be"),
                             ("read_language", "x", "Read must be")):
        form = form_from_config(Config())
        form[key] = value
        with pytest.raises(ValueError, match=text):
            config_from_form(Config(), form)


class Recorder:
    def __init__(self):
        self.calls, self.applied = [], []
        self.config = Config()
        self.mp3 = True
        self.mcp = {"enabled": False, "running": False, "url": "", "error": ""}
        self.reader = {"enabled": False, "running": False, "url": "", "error": ""}
        self.registered: list[str] = []
        self.register_error: Exception | None = None
        self.doc = Document("Hello there.", True, "notes")
        self.exports, self.export_error = [], None
        self.st_dir = Path("st-models")
        self.st_present = False
        self.st_changed = 0
        self.voice_list = ["af_heart", "bf_emma"]

    def hooks(self):
        def apply(cfg):
            self.applied.append(cfg)
            self.config = copy.deepcopy(cfg)
            return None

        return SettingsHooks(
            get_config=lambda: copy.deepcopy(self.config), apply=apply,
            is_autostart=lambda: False, set_autostart=lambda v: None,
            set_show_on_start=lambda v: None, models_dir=lambda: Path("."),
            voices=lambda: list(self.voice_list),
            output_devices=lambda: [], play_test=lambda c, t, lang=None: None,
            stop=lambda: self.calls.append("stop"), open_logs=lambda: None,
            open_config=lambda: None, version="t",
            load_document=self._load_document, export_audio=self._export_audio,
            cancel_export=lambda: self.calls.append("cancel_export"),
            mp3_available=lambda: self.mp3,
            mcp_status=lambda: dict(self.mcp),
            reader_status=lambda: dict(self.reader),
            open_reader=lambda: self.calls.append("open_reader"),
            extension_folder=lambda: Path("ext-dir"),
            registered_exts=lambda: list(self.registered),
            register_files=self._register_files, unregister_files=self._unregister_files,
            open_default_apps=lambda: self.calls.append("default_apps"),
            open_folder=lambda p: self.calls.append(("open_folder", p)),
            supertonic_dir=lambda: self.st_dir, supertonic_present=lambda: self.st_present,
            supertonic_changed=self._st_changed)

    def _register_files(self, exts):
        self.calls.append(("register", list(exts)))
        if self.register_error:
            raise self.register_error
        self.registered = sorted(set(self.registered) | set(exts))

    def _unregister_files(self):
        self.calls.append("unregister")
        self.registered = []

    def _st_changed(self):
        self.st_changed += 1

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
                            voice="af_heart", lang=req.lang)


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
    assert TABS == ("general", "voice", "export", "kokoro", "supertonic", "mcp", "reader", "about")
    win = make_window(tk_root, Recorder())
    assert [win._nb.tab(i, "text") for i in range(len(TABS))] == [
        "General", "Voice", "Export", "Kokoro", "Supertonic", "Agents (MCP)", "Reader", "About"]


def pick_language(win, name):
    win._sel_lang.set(name)


def test_voices_section_defaults(tk_root):
    win = make_window(tk_root, Recorder())
    langs = list(win._lang_combo.cget("values"))
    assert len(langs) == 31 and langs[0] == "English" and langs[1:] == sorted(langs[1:])
    assert win._sel_lang.get() == "English" and win._sel_engine.get() == "Kokoro"
    assert [v.split(" ")[0] for v in win._voice_combo.cget("values")] == ["af_heart", "bf_emma"]
    assert win._sel_voice.get().startswith("af_heart  Heart")
    assert win._collect()["language_voices"] == Config().speech.language_voices
    assert win._collect()["read_language"] == "auto"
    assert not win._is_dirty()


def test_engine_choices_depend_on_language(tk_root):
    rec = Recorder()
    rec.voice_list = ["af_heart", "supertonic:F1", "bf_emma", "ef_dora", "em_alex", "ff_siwis"]
    win = make_window(tk_root, rec)
    assert list(win._engine_combo.cget("values")) == ["Supertonic", "Kokoro", "Off"]
    pick_language(win, "Croatian")  # Kokoro does not read Croatian
    assert list(win._engine_combo.cget("values")) == ["Supertonic", "Off"]
    assert win._sel_engine.get() == "Supertonic"
    assert list(win._voice_combo.cget("values"))[0].startswith("supertonic:F1")
    assert len(win._voice_combo.cget("values")) == len(SUPERTONIC_VOICES)
    pick_language(win, "Spanish")
    assert list(win._engine_combo.cget("values")) == ["Supertonic", "Kokoro", "Off"]
    win._sel_engine.set("Kokoro")
    assert win._lang_voices["es"] == "ef_dora"
    # only Spanish Kokoro voices, from the loaded list
    assert [v.split(" ")[0] for v in win._voice_combo.cget("values")] == ["ef_dora", "em_alex"]
    win._sel_voice.set("em_alex  Alex")
    assert win._lang_voices["es"] == "em_alex"
    assert win._collect()["language_voices"]["es"] == "em_alex"


def test_kokoro_voices_fall_back_to_the_default_when_none_are_loaded(tk_root):
    rec = Recorder()
    rec.voice_list = []
    win = make_window(tk_root, rec)
    pick_language(win, "French")
    win._sel_engine.set("Kokoro")
    assert win._lang_voices["fr"] == "ff_siwis"
    assert [v.split(" ")[0] for v in win._voice_combo.cget("values")] == ["ff_siwis"]


def test_engine_switch_keeps_the_supertonic_voice_of_the_row(tk_root):
    win = make_window(tk_root, Recorder())
    pick_language(win, "Spanish")
    win._sel_voice.set("supertonic:M3  M3")
    assert win._lang_voices["es"] == "supertonic:M3"
    win._sel_engine.set("Kokoro")
    assert win._lang_voices["es"] == "ef_dora"
    win._sel_engine.set("Supertonic")
    assert win._lang_voices["es"] == "supertonic:M3"
    win._sel_engine.set("Off")
    win._sel_engine.set("Supertonic")
    assert win._lang_voices["es"] == "supertonic:M3"
    pick_language(win, "German")  # a row never touched: F1
    win._sel_engine.set("Off")
    win._sel_engine.set("Supertonic")
    assert win._lang_voices["de"] == "supertonic:F1"


def test_off_disables_voice_combo_and_marks_the_language(tk_root):
    win = make_window(tk_root, Recorder())
    pick_language(win, "German")
    assert state(win._voice_combo) == "readonly"
    win._sel_engine.set("Off")
    assert win._lang_voices["de"] == "" and state(win._voice_combo) == "disabled"
    assert win._sel_voice.get() == "" and list(win._voice_combo.cget("values")) == []
    assert win._sel_lang.get() == "German (off)"
    assert "German (off)" in win._lang_combo.cget("values")
    assert win._is_dirty()
    pick_language(win, "English")  # the marker stays on the list, the combo is live again
    assert state(win._voice_combo) == "readonly"
    pick_language(win, "German (off)")
    assert win._sel_engine.get() == "Off" and state(win._voice_combo) == "disabled"
    win._sel_engine.set("Supertonic")
    assert win._lang_voices["de"] == "supertonic:F1" and state(win._voice_combo) == "readonly"
    assert "German" in win._lang_combo.cget("values")


def test_last_language_cannot_be_turned_off(tk_root):
    rec = Recorder()
    rec.config.speech.language_voices = {c: "" for c in SUPERTONIC_LANGUAGES}
    rec.config.speech.language_voices["hr"] = "supertonic:F2"
    win = make_window(tk_root, rec)
    pick_language(win, "Croatian")
    assert win._sel_engine.get() == "Supertonic"
    win._sel_engine.set("Off")
    assert win._sel_engine.get() == "Supertonic" and win._lang_voices["hr"] == "supertonic:F2"
    assert win._sel_lang.get() == "Croatian" and state(win._voice_combo) == "readonly"
    assert "At least one language" in win._status.cget("text")
    assert not win._is_dirty()
    pick_language(win, "German (off)")
    win._sel_engine.set("Supertonic")  # turning another on, then Croatian off, is fine
    pick_language(win, "Croatian")
    win._sel_engine.set("Off")
    assert win._lang_voices["hr"] == ""


def test_read_combo_lists_only_enabled_languages(tk_root):
    rec = Recorder()
    rec.config.speech.language_voices = {c: "" for c in SUPERTONIC_LANGUAGES}
    rec.config.speech.language_voices.update(en="af_heart", hr="supertonic:F1")
    win = make_window(tk_root, rec)
    assert list(win._read_combo.cget("values")) == [
        "Auto (detect language)", "Always English", "Always Croatian"]
    pick_language(win, "German (off)")
    win._sel_engine.set("Supertonic")
    assert list(win._read_combo.cget("values")) == [
        "Auto (detect language)", "Always English", "Always Croatian", "Always German"]
    win._vars["read_language"].set("Always German")
    assert win._collect()["read_language"] == "de"
    assert win.save() is True
    assert rec.applied[-1].speech.read_language == "de"


def test_read_resets_to_auto_when_its_language_goes_off(tk_root):
    rec = Recorder()
    rec.config.speech.read_language = "de"
    win = make_window(tk_root, rec)
    assert win._vars["read_language"].get() == "Always German" and not win._is_dirty()
    pick_language(win, "Italian")
    win._sel_engine.set("Off")  # another language: Read stays
    assert win._vars["read_language"].get() == "Always German"
    pick_language(win, "German")
    win._sel_engine.set("Off")
    assert win._vars["read_language"].get() == "Auto (detect language)"
    assert "Always German" not in win._read_combo.cget("values")
    assert win._collect()["read_language"] == "auto"
    assert win.save() is True
    sp = rec.applied[-1].speech
    assert (sp.read_language, sp.language_voices["de"], sp.language_voices["it"]) == (
        "auto", "", "")


def test_new_keys_save_and_dirty(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    assert not win._is_dirty()
    win._sel_engine.set("Supertonic")
    assert win._is_dirty()
    win._sel_voice.set("supertonic:M4  M4")
    pick_language(win, "Croatian")
    win._sel_voice.set("supertonic:F3  F3")
    win._vars["supertonic_steps"].set("12")
    assert win.save() is True
    sp = rec.config.speech
    assert (sp.language_voices["en"], sp.language_voices["hr"], sp.supertonic_steps) == (
        "supertonic:M4", "supertonic:F3", 12)
    assert not win._is_dirty()


def test_supertonic_hint_follows_presence(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    assert "Supertonic tab" in win._supertonic_hint.cget("text")
    # the Supertonic choices stay selectable
    assert state(win._engine_combo) == "readonly" and state(win._voice_combo) == "readonly"
    rec.st_present = True
    win.refresh()
    assert win._supertonic_hint.cget("text") == ""
    # nothing enabled uses Supertonic: no hint even when it is missing
    rec.st_present = False
    rec.config.speech.language_voices = {c: "" for c in SUPERTONIC_LANGUAGES}
    rec.config.speech.language_voices["en"] = "af_heart"
    win.refresh()
    assert win._supertonic_hint.cget("text") == ""


def test_refresh_follows_config_unless_edited(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    rec.config.speech.language_voices["en"] = "bf_emma"  # tray changes
    rec.config.speech.language_voices["fr"] = ""
    rec.config.speech.read_language = "es"
    win.refresh()
    assert win._lang_voices["en"] == "bf_emma" and win._lang_voices["fr"] == ""
    assert win._vars["read_language"].get() == "Always Spanish"
    assert win._sel_voice.get().startswith("bf_emma")
    assert not win._is_dirty()
    pick_language(win, "Spanish")  # the user edits a row
    win._sel_engine.set("Kokoro")
    rec.config.speech.language_voices["es"] = "supertonic:F5"
    win.refresh()
    assert win._lang_voices["es"] == "ef_dora" and win._is_dirty()
    win._vars["read_language"].set("Auto (detect language)")  # and Read
    rec.config.speech.read_language = "de"
    win.refresh()
    assert win._vars["read_language"].get() == "Auto (detect language)"


def test_save_round_trips_voice(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    win._sel_voice.set("bf_emma  Emma (British English, female)")
    assert win.save() is True
    assert rec.applied[-1].speech.language_voices["en"] == "bf_emma"
    assert not win._is_dirty()


def test_play_previews_the_selected_language(tk_root):
    rec = Recorder()
    calls = []
    hooks = rec.hooks()
    hooks.play_test = lambda cfg, text, lang: calls.append((cfg.speech.language_voices[lang],
                                                              text, lang))
    win = SettingsWindow(tk_root, hooks, lambda fn: fn(), lambda fn: fn())
    try:
        win.show("voice")
    except tk.TclError:
        pytest.skip("settings window cannot be built headless")
    pick_language(win, "Spanish")
    win._sel_engine.set("Kokoro")
    win._sample.set("Hola")
    win._play_test()
    assert calls == [("ef_dora", "Hola", "es")]
    win._sel_engine.set("Off")
    win._play_test()
    assert len(calls) == 1 and "is off" in win._test_result.cget("text")


def test_window_fits_a_1080p_screen(tk_root):
    win = make_window(tk_root, Recorder())
    for tab in ("voice", "export", "supertonic"):
        win._nb.select(TABS.index(tab))
        win.top.update_idletasks()
        assert win.top.winfo_reqheight() <= 1080 - 90

# ---------------------------------------------------------------- supertonic tab

class FakeSupertonic:
    """Stands in for the model_setup Supertonic functions (slice A may not have them yet)."""

    def __init__(self, monkeypatch, installed=False):
        self.installed = installed
        self.downloads, self.deletes = [], []
        self.error = None
        self.progress = [(50_000_000, 400_000_000), (400_000_000, 400_000_000)]
        monkeypatch.setattr(model_setup, "supertonic_missing",
                            lambda d: [] if self.installed else ["onnx/vocoder.onnx"],
                            raising=False)
        monkeypatch.setattr(model_setup, "download_supertonic", self.download, raising=False)
        monkeypatch.setattr(model_setup, "delete_supertonic", self.delete, raising=False)

    def download(self, dest, progress=None, cancel=None):
        self.downloads.append(dest)
        for done, total in self.progress:
            if progress:
                progress(done, total)
        if self.error:
            raise self.error
        self.installed = True

    def delete(self, dest):
        self.deletes.append(dest)
        self.installed = False


def st_win(tk_root, monkeypatch, installed=False, spawn=lambda fn: fn(), tmp_path=None):
    fake = FakeSupertonic(monkeypatch, installed)
    rec = Recorder()
    if tmp_path is not None:
        rec.st_dir = tmp_path
    win = SettingsWindow(tk_root, rec.hooks(), lambda fn: fn(), spawn)
    try:
        win.show("supertonic")
    except tk.TclError:
        pytest.skip("settings window cannot be built headless")
    return fake, rec, win


def test_supertonic_tab_status_not_installed_and_installed(tk_root, monkeypatch, tmp_path):
    fake, rec, win = st_win(tk_root, monkeypatch, tmp_path=tmp_path)
    assert win._nb.index(win._nb.select()) == TABS.index("supertonic")
    assert win._st_status.cget("text") == "Not installed"
    assert state(win._st_dl_btn) == "normal" and state(win._st_del_btn) == "disabled"
    assert "OpenRAIL-M" in settings_ui.SUPERTONIC_NOTE
    (tmp_path / "a.bin").write_bytes(b"x" * 2_000_000)
    fake.installed = True
    win.refresh()
    assert win._st_status.cget("text") == "Installed (2 MB on disk)"
    assert state(win._st_dl_btn) == "disabled" and state(win._st_del_btn) == "normal"


def test_supertonic_download_success(tk_root, monkeypatch, tmp_path):
    fake, rec, win = st_win(tk_root, monkeypatch, tmp_path=tmp_path)
    labels = []
    real = win._st_progress_label.configure
    win._st_progress_label.configure = lambda **kw: (labels.append(kw.get("text")), real(**kw))
    win._st_dl_btn.invoke()
    assert fake.downloads == [tmp_path]
    assert "Downloading Supertonic: 50 / 400 MB" in labels
    assert rec.st_changed == 1
    assert not win._st_busy and state(win._st_cancel_btn) == "disabled"
    assert win._st_status.cget("text").startswith("Installed")


def test_show_with_download_on_supertonic_tab_fetches_supertonic(tk_root, monkeypatch, tmp_path):
    fake, rec, win = st_win(tk_root, monkeypatch, tmp_path=tmp_path)
    kokoro = []
    monkeypatch.setattr(win, "_download_missing", lambda: kokoro.append(1))
    win.show("supertonic", download=True)
    assert fake.downloads == [tmp_path] and kokoro == []
    assert TABS[win._nb.index(win._nb.select())] == "supertonic"


def test_supertonic_download_cancelled_and_failed(tk_root, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(settings_ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    fake, rec, win = st_win(tk_root, monkeypatch, tmp_path=tmp_path)
    fake.error = model_setup.DownloadCancelled()
    win._st_dl_btn.invoke()
    assert rec.st_changed == 0 and errors == [] and not win._st_busy
    assert win._status.cget("text") == "Download cancelled."
    fake.error = model_setup.SetupError("no network")
    win._st_dl_btn.invoke()
    assert rec.st_changed == 0 and errors and "no network" in errors[0][1]
    assert state(win._st_dl_btn) == "normal"


def test_supertonic_cancel_button_sets_event(tk_root, monkeypatch, tmp_path):
    pending = []
    fake, rec, win = st_win(tk_root, monkeypatch, spawn=pending.append, tmp_path=tmp_path)
    win._st_dl_btn.invoke()
    assert win._st_busy and state(win._st_cancel_btn) == "normal"
    assert state(win._st_dl_btn) == "disabled" and state(win._st_del_btn) == "disabled"
    win._st_dl_btn.invoke()  # ignored while busy
    assert len(pending) == 1
    win._st_cancel_btn.invoke()
    assert win._st_cancel.is_set()
    pending.pop()()
    assert not win._st_busy


def test_supertonic_delete_confirms(tk_root, monkeypatch, tmp_path):
    fake, rec, win = st_win(tk_root, monkeypatch, installed=True, tmp_path=tmp_path)
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: False)
    win._st_del_btn.invoke()
    assert fake.deletes == [] and rec.st_changed == 0
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: True)
    win._st_del_btn.invoke()
    assert fake.deletes == [tmp_path] and rec.st_changed == 1 and "stop" in rec.calls
    assert win._st_status.cget("text") == "Not installed"
    assert "Supertonic tab" in win._supertonic_hint.cget("text")


def test_supertonic_steps_slider_and_entry(tk_root, monkeypatch, tmp_path):
    fake, rec, win = st_win(tk_root, monkeypatch, tmp_path=tmp_path)
    assert win._vars["supertonic_steps"].get() == "8"
    win._st_scale.set(12)
    assert win._vars["supertonic_steps"].get() == "12"
    win._vars["supertonic_steps"].set("5")
    assert round(float(win._st_scale.get())) == 5
    assert win._is_dirty()
    assert win.save() is True and rec.config.speech.supertonic_steps == 5
    monkeypatch.setattr(settings_ui.messagebox, "showerror", lambda *a, **k: None)
    win._vars["supertonic_steps"].set("99")
    assert win.save() is False


def test_supertonic_hooks_defaults():
    hooks = Recorder().hooks()
    h = SettingsHooks(get_config=hooks.get_config, apply=hooks.apply, is_autostart=lambda: False,
                      set_autostart=lambda v: None, set_show_on_start=lambda v: None,
                      models_dir=lambda: Path("."), voices=lambda: [], output_devices=lambda: [],
                      play_test=lambda c, t, lang=None: None, stop=lambda: None, open_logs=lambda: None,
                      open_config=lambda: None)
    assert h.supertonic_present() is False and h.supertonic_changed() is None
    assert h.supertonic_dir().parts[-2:] == ("models", "supertonic-3")


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
    assert req.lang is None
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


def test_export_language_combo_and_mapping(tk_root, monkeypatch, tmp_path):
    rec, win = export_win(tk_root, monkeypatch, save_to=tmp_path / "a.wav")
    values = list(win._export_lang_combo.cget("values"))
    assert values[:2] == ["Auto", "English"] and len(values) == 32
    assert values[2:] == sorted(values[2:])
    assert win._export_lang.get() == "Auto"
    win._export_set_text("Hi")
    for label, lang in (("Auto", None), ("English", "en"), ("Croatian", "hr"),
                        ("Korean", "ko")):
        win._export_lang.set(label)
        win._export_start()
        assert rec.exports[-1].lang == lang
    assert {win._export_lang_codes[v] for v in values[2:]} == set(SUPERTONIC_LANGUAGES) - {"en"}
    assert not win._is_dirty()  # transient export control


def test_export_voice_lists_supertonic_voices(tk_root, monkeypatch, tmp_path):
    rec = Recorder()
    rec.voice_list = ["af_heart", "supertonic:F1", "supertonic:M5"]
    win = make_window(tk_root, rec)
    ids = win._export_voice_ids
    assert ids[settings_ui.AUTO_VOICE] is None
    assert [v for v in ids.values() if v] == ["af_heart", "supertonic:F1", "supertonic:M5"]


def test_export_bitrates_come_from_config(tk_root):
    win = make_window(tk_root, Recorder())
    assert list(win._export_bitrate.cget("values")) == [str(b) for b in MP3_BITRATES]


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


# ---------------------------------------------------------------- reader tab (VTF-07)

READER_ID = "injaiiefkhoimpkeandbjfbnnjddaceh"


def test_reader_form_round_trip():
    cfg = Config()
    form = form_from_config(cfg)
    assert form["reader_enabled"] is True and form["reader_port"] == "7724"
    assert form["reader_library_folder"] == "library" and form["reader_extension_ids"] == READER_ID
    assert config_from_form(cfg, form) == cfg
    cfg.reader.enabled, cfg.reader.port, cfg.reader.max_chars = False, 8124, 0
    cfg.reader.library_folder = "D:/books"
    cfg.reader.extension_ids = [READER_ID, "a" * 32]
    assert config_from_form(Config(), form_from_config(cfg)) == cfg
    cfg.reader.extension_ids = []
    assert config_from_form(Config(), form_from_config(cfg)) == cfg


def test_reader_form_extension_ids_parsing_and_errors():
    form = form_from_config(Config())
    form["reader_extension_ids"] = f" {READER_ID.upper()};{'b' * 32}  {READER_ID}"
    assert config_from_form(Config(), form).reader.extension_ids == [READER_ID, "b" * 32]
    for key, value in (("reader_port", "80"), ("reader_port", "x"), ("reader_max_chars", "-1"),
                       ("reader_library_folder", "  "), ("reader_extension_ids", "short")):
        bad = form_from_config(Config())
        bad[key] = value
        with pytest.raises(ValueError):
            config_from_form(Config(), bad)
    bad = form_from_config(Config())
    bad["reader_extension_ids"] = "nope"
    with pytest.raises(ValueError, match="Extension id 'nope' is not valid"):
        config_from_form(Config(), bad)


def test_reader_tab_builds_and_only_config_keys_are_vars(tk_root):
    win = make_window(tk_root, Recorder())
    win.show("reader")
    assert win._nb.index(win._nb.select()) == TABS.index("reader")
    assert {k for k in win._vars if k.startswith("reader_")} == {
        "reader_enabled", "reader_port", "reader_library_folder", "reader_max_chars",
        "reader_extension_ids"}
    assert set(win._assoc_vars) == {".epub", ".txt"} and not set(win._assoc_vars) & set(win._vars)
    assert not win._is_dirty()
    win._assoc_vars[".txt"].set(True)  # UI state only: does not make the form dirty
    assert not win._is_dirty()


@pytest.mark.parametrize("status,enabled,expected,button", [
    ({"enabled": True, "running": True, "url": "http://127.0.0.1:7724/", "error": ""}, True,
     "Running at http://127.0.0.1:7724/", "normal"),
    ({"enabled": False, "running": False, "url": "", "error": ""}, False, "Off", "disabled"),
    ({"enabled": True, "running": False, "url": "", "error": "Port 7724 is busy"}, True,
     "Error: Port 7724 is busy", "disabled"),
    ({"enabled": False, "running": False, "url": "", "error": ""}, True,
     "Enabled \u2014 save to start", "disabled"),
])
def test_reader_status_text_and_open_button(tk_root, status, enabled, expected, button):
    rec = Recorder()
    rec.reader = status
    rec.config.reader.enabled = enabled
    win = make_window(tk_root, rec)
    assert win._reader_status.cget("text") == expected
    assert state(win._reader_open_btn) == button


def test_reader_open_reader_and_folder_hooks(tk_root):
    rec = Recorder()
    rec.reader = {"enabled": True, "running": True, "url": "http://127.0.0.1:7724/", "error": ""}
    win = make_window(tk_root, rec)
    win._reader_open_btn.invoke()
    assert rec.calls[-1] == "open_reader"
    win._reader_open_library()
    assert rec.calls[-1] == ("open_folder", settings_ui.resolve_path("library"))


def test_reader_register_and_remove_fire_hooks(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    assert win._assoc_status.cget("text") == "Not registered"
    assert win._assoc_vars[".epub"].get() and not win._assoc_vars[".txt"].get()
    win._assoc_register()
    assert rec.calls[-1] == ("register", [".epub"])
    assert win._assoc_status.cget("text") == "Registered for .epub"
    win._assoc_vars[".txt"].set(True)
    win._assoc_register()
    assert rec.calls[-1] == ("register", [".epub", ".txt"])
    assert win._assoc_status.cget("text") == "Registered for .epub, .txt"
    win._assoc_remove()
    assert rec.calls[-1] == "unregister"
    assert win._assoc_status.cget("text") == "Not registered"


def test_reader_register_needs_a_tick_and_reports_errors(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    win._assoc_vars[".epub"].set(False)
    win._assoc_register()
    assert not any(isinstance(c, tuple) and c[0] == "register" for c in rec.calls)
    assert "Tick" in win._assoc_status.cget("text")
    win._assoc_vars[".epub"].set(True)
    rec.register_error = OSError("access denied")
    win._assoc_register()
    assert win._assoc_status.cget("text") == "Failed: access denied"


def test_reader_ticks_follow_the_registration_on_show(tk_root):
    rec = Recorder()
    rec.registered = [".txt"]
    win = make_window(tk_root, rec)
    assert not win._assoc_vars[".epub"].get() and win._assoc_vars[".txt"].get()
    assert win._assoc_status.cget("text") == "Registered for .txt"


def test_reader_default_apps_and_extension_folder_buttons(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    win._safe_call(lambda: win.hooks.open_default_apps())()
    assert rec.calls[-1] == "default_apps"
    win._safe_call(lambda: win.hooks.open_folder(win.hooks.extension_folder()))()
    assert rec.calls[-1] == ("open_folder", Path("ext-dir"))


def test_reader_fields_save_and_dirty(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    win._vars["reader_port"].set("8124")
    win._vars["reader_extension_ids"].set("")
    assert win._is_dirty()
    assert win.save() is True
    assert rec.config.reader.port == 8124 and rec.config.reader.extension_ids == []
    assert not win._is_dirty()
