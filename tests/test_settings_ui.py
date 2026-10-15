import copy
import re
import tkinter as tk
from pathlib import Path

import pytest

from vibetts import model_setup, settings_ui
from vibetts.config import Config
from vibetts.settings_ui import (ANY, ENCODERS_MISSING, SAME_VOICE, TABS, SettingsHooks,
                                 SettingsWindow, VoiceCreator, build_instruct,
                                 config_from_form, form_from_config, model_choice)


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
    ("pill_position", "left"), ("voice", ""), ("model", "bogus"), ("engine", "bogus"),
    ("omnivoice_steps", "3"), ("omnivoice_steps", "17"), ("omnivoice_steps", "x"),
])
def test_invalid_values(key, value):
    form = form_from_config(Config())
    form[key] = value
    with pytest.raises(ValueError):
        config_from_form(Config(), form)


def test_omnivoice_fields_round_trip():
    cfg = Config()
    cfg.speech.engine = "omnivoice"
    cfg.speech.omnivoice_steps = 12
    cfg.speech.omni_voice = "omni:alma"
    cfg.speech.croatian_voice = "omni:zora"
    cfg.speech.auto_language = False
    assert config_from_form(cfg, form_from_config(cfg)) == cfg


def test_voice_display_strings_parsed():
    form = form_from_config(Config())
    form["omni_voice"] = "omni:alma  Alma (OmniVoice)"
    form["croatian_voice"] = "omni_hr:zora  Zora (OmniVoice)"  # legacy id is normalised
    form["omnivoice_steps"] = "16"
    cfg = config_from_form(Config(), form)
    assert cfg.speech.omni_voice == "omni:alma"
    assert cfg.speech.croatian_voice == "omni:zora"
    assert cfg.speech.omnivoice_steps == 16


def test_croatian_same_as_omni_voice_is_empty():
    form = form_from_config(Config())
    form["croatian_voice"] = SAME_VOICE
    assert config_from_form(Config(), form).speech.croatian_voice == ""
    form["croatian_voice"] = ""
    assert config_from_form(Config(), form).speech.croatian_voice == ""
    form["omni_voice"] = ""  # nothing picked: the default preset
    assert config_from_form(Config(), form).speech.omni_voice == "omni:default"


# ---------------------------------------------------------------- voice creator

OPTIONS = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "style": ["whisper"],
}


class FakeOmni:
    INSTRUCT_OPTIONS = OPTIONS

    @staticmethod
    def sanitize_name(s):
        name = re.sub(r"[^a-z0-9_-]+", "_", s.strip().lower())[:32].strip("_")
        if not name:
            raise ValueError("Name needs letters or digits")
        if name == "default":
            raise ValueError("the name default is reserved")
        return name


INFO = {"default": {"name": "default", "voice": "omni:default", "kind": "default",
                    "description": "", "lang": "hr"},
        "alma": {"name": "alma", "voice": "omni:alma", "kind": "description",
                 "description": "female, young adult", "lang": "en"}}


class FakeInstaller:
    """Stand-in for the omnivoice_setup behind the install hooks."""

    def __init__(self, rec):
        self.rec = rec
        self.error = None
        self.cancel_after = None
        self.progress_calls = []

    def install(self, include_encoders, progress, cancel):
        self.rec.calls.append(("install", include_encoders))
        progress("Downloading model.safetensors 1.2 / 2.4 GB", 1_200_000_000, 2_400_000_000)
        progress("Converting 120 / 197", 120, 197)
        if self.cancel_after:
            cancel.set()
            raise model_setup.DownloadCancelled()
        if self.error:
            raise self.error
        self.rec.installed = True
        self.rec.encoders = bool(include_encoders)

    def uninstall(self):
        self.rec.calls.append("uninstall")
        self.rec.installed = False
        self.rec.encoders = False
        if self.rec.config.speech.engine == "omnivoice":
            self.rec.config.speech.engine = "kokoro"


class Recorder:
    def __init__(self):
        self.calls, self.saved, self.deleted, self.played, self.applied = [], [], [], [], []
        self.presets = ["default", "alma"]
        self.installed = True
        self.encoders = True
        self.error = None
        self.config = Config()
        self.installer = FakeInstaller(self)

    def hooks(self):
        def design(instruct, seed, text=None, lang=None):
            if self.error:
                raise self.error
            self.calls.append(("design", instruct, seed, text, lang))
            return {"clip": "d", "instruct": instruct}

        def clone(wav, text):
            self.calls.append(("clone", wav, text))
            return {"clip": "c"}

        def save(name, clip, overwrite=False):
            self.saved.append((name, clip))
            self.presets.append(name)
            return f"omni:{name}"

        def apply(cfg):
            self.applied.append((cfg.speech.omni_voice, cfg.speech.croatian_voice))
            self.config = copy.deepcopy(cfg)
            return None

        def delete(name):
            self.deleted.append(name)
            self.presets.remove(name)
            sp = self.config.speech
            if sp.omni_voice == f"omni:{name}":
                sp.omni_voice = "omni:default"
            if sp.croatian_voice == f"omni:{name}":
                sp.croatian_voice = ""

        return SettingsHooks(
            get_config=lambda: copy.deepcopy(self.config), apply=apply,
            is_autostart=lambda: False, set_autostart=lambda v: None,
            set_show_on_start=lambda v: None, models_dir=lambda: Path("."),
            voices=lambda: ["af_heart", "bf_emma"] + [f"omni:{n}" for n in self.presets],
            output_devices=lambda: [], play_test=lambda c, t: None,
            stop=lambda: self.calls.append("stop"), open_logs=lambda: None,
            open_config=lambda: None, version="t",
            omni_present=lambda: self.installed,
            omni_voices=lambda: [f"omni:{n}" for n in self.presets] if self.installed else [],
            omni_voice_info=lambda: [INFO.get(n, {"name": n, "kind": "random", "lang": "",
                                                  "description": ""}) for n in self.presets],
            omni_status=lambda: {"installed": self.installed, "encoders": self.encoders,
                                 "bytes_on_disk": 1_300_000_000 if self.installed else 0},
            omni_download_bytes=lambda inc: 2_900_000_000 + (654_000_000 if inc else 0),
            omni_install=self.installer.install, omni_uninstall=self.installer.uninstall,
            play_preset=lambda name: self.played.append(("preset", name)),
            design_clip=design, clone_clip=clone, save_voice=save, delete_voice=delete,
            play_clip=lambda clip: self.played.append(clip),
            encoders_present=lambda: self.encoders)


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
def tk_root(_shared_root, monkeypatch):
    monkeypatch.setattr(settings_ui, "_omni", lambda: FakeOmni)
    yield _shared_root
    for child in _shared_root.winfo_children():
        child.destroy()


def make_creator(root, rec, on_saved=None, spawn=lambda fn: fn()):
    c = VoiceCreator(root, rec.hooks(), lambda fn: fn(), on_saved or (lambda v, u: None), spawn)
    c.pack()
    return c


def make_window(root, rec, spawn=lambda fn: fn()):
    win = SettingsWindow(root, rec.hooks(), lambda fn: fn(), spawn)
    try:
        win.show("voice")
    except tk.TclError:
        pytest.skip("settings window cannot be built headless")
    return win


def test_build_instruct():
    assert build_instruct({}, OPTIONS) == ""
    assert build_instruct({k: ANY for k in OPTIONS}, OPTIONS) == ""
    sel = {"pitch": "low pitch", "gender": "female", "age": "young adult", "style": ANY}
    assert build_instruct(sel, OPTIONS) == "female, young adult, low pitch"  # option order
    assert build_instruct({"style": "whisper"}, OPTIONS) == "whisper"


def test_instruct_options_fallback(monkeypatch):
    monkeypatch.setattr(settings_ui, "_omni", lambda: object())  # no INSTRUCT_OPTIONS yet
    assert list(settings_ui.instruct_options()) == ["gender", "age", "pitch", "style"]


def test_creator_assembles_instruct_and_language(tk_root):
    rec = Recorder()
    d = make_creator(tk_root, rec)
    assert all(v.get() == ANY for v in d.option_vars.values())
    assert list(d.option_combos["gender"].cget("values"))[0] == ANY
    assert d.lang_var.get() == "Croatian"  # default sample language
    d.generate()  # all (any): plain random voice
    assert rec.calls[-1] == ("design", "", 0, None, "hr")
    d.option_vars["gender"].set("female")
    d.option_vars["age"].set("young adult")
    d.option_vars["pitch"].set("low pitch")
    d.seed_var.set("42")
    d.lang_var.set("English")
    d.generate()
    assert rec.calls[-1] == ("design", "female, young adult, low pitch", 42, None, "en")
    assert d.button_state("play_design") == "normal" and d.button_state("save_design") == "normal"


def test_try_another_picks_new_seed_and_generates(tk_root, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(settings_ui.random, "randint", lambda a, b: 777)
    d = make_creator(tk_root, rec)
    d.try_another()
    assert d.seed_var.get() == "777" and rec.calls[-1][2] == 777


def test_creator_busy_state_and_inline_errors(tk_root):
    rec = Recorder()
    pending = []
    d = make_creator(tk_root, rec, spawn=pending.append)
    d.generate()
    assert "Generating" in d.message.cget("text")
    assert d.button_state("generate") == "disabled" and d.button_state("try") == "disabled"
    pending.pop()()  # worker finishes
    assert d.button_state("generate") == "normal" and d.button_state("save_design") == "normal"
    rec.error = ValueError("unknown item 'femle'; did you mean 'female'?")
    d.generate()
    pending.pop()()
    assert "did you mean" in d.message.cget("text")
    assert d.button_state("save_design") == "disabled"  # failed generation clears the clip


def test_save_validates_sanitizes_and_resets(tk_root):
    rec = Recorder()
    saved = []
    d = make_creator(tk_root, rec, on_saved=lambda v, u: saved.append((v, u)))
    d.generate()
    assert d.save() is False and "Enter a name" in d.message.cget("text")
    d.name_var.set("default")
    assert d.save() is False and "reserved" in d.message.cget("text")
    d.name_var.set("Ana Maria!")
    assert d.save() is False  # shows the cleaned name first
    assert d.name_var.get() == "ana_maria" and "ana_maria" in d.message.cget("text")
    assert rec.saved == []
    assert d.save() is True
    assert rec.saved == [("ana_maria", {"clip": "d", "instruct": ""})]
    assert saved == [("omni:ana_maria", True)]  # "Use as OmniVoice voice" defaults on
    assert d.name_var.get() == "" and d.button_state("save_design") == "disabled"  # reusable


def test_save_existing_name_needs_replace_confirmation(tk_root, monkeypatch):
    rec = Recorder()
    answers = iter([False, True])
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: next(answers))
    d = make_creator(tk_root, rec)
    d.generate()
    d.name_var.set("alma")
    d.use_var.set(False)
    assert d.save() is False and rec.saved == []
    assert d.save() is True and rec.saved[0][0] == "alma"


def test_clone_tab_requires_wav_and_transcript(tk_root):
    rec = Recorder()
    d = make_creator(tk_root, rec)
    assert d.button_state("create") == "normal"
    d.create()
    assert "WAV" in d.message.cget("text")
    d.wav_var.set("C:/x/me.wav")
    d.create()
    assert "transcript" in d.message.cget("text").lower() and rec.calls == []
    d.transcript.insert("1.0", "Dobar dan svima.")
    d.create()
    assert rec.calls[-1] == ("clone", Path("C:/x/me.wav"), "Dobar dan svima.")
    assert d.button_state("save_clone") == "normal"


def test_encoders_missing_disables_cloning(tk_root):
    rec = Recorder()
    rec.encoders = False
    d = make_creator(tk_root, rec)
    assert d.encoders_label.cget("text") == ENCODERS_MISSING
    assert d.button_state("create") == "disabled"
    d.wav_var.set("x.wav")
    d.transcript.insert("1.0", "tekst")
    d.create()
    assert rec.calls == []
    d.set_availability(True, True)  # cloning got installed
    assert d.button_state("create") == "normal" and d.encoders_label.cget("text") == ""


def test_creator_disabled_until_installed(tk_root):
    rec = Recorder()
    d = make_creator(tk_root, rec)
    d.set_availability(False, False)
    assert d.button_state("generate") == "disabled" and d.button_state("try") == "disabled"
    d.generate()
    assert rec.calls == []
    d.set_availability(True, True)
    assert d.button_state("generate") == "normal"


def test_play_goes_through_hook(tk_root):
    rec = Recorder()
    d = make_creator(tk_root, rec)
    d.play("design")
    assert rec.played == []  # nothing generated yet
    d.generate()
    d.play("design")
    assert rec.played == [{"clip": "d", "instruct": ""}]


# ---------------------------------------------------------------- settings window

def state(widget):
    return str(widget.cget("state"))


def test_tabs_and_engine_radio(tk_root):
    assert TABS == ("general", "voice", "kokoro", "omnivoice", "about")
    rec = Recorder()
    win = make_window(tk_root, rec)
    assert [win._nb.tab(i, "text") for i in range(len(TABS))] == [
        "General", "Voice", "Kokoro", "OmniVoice", "About"]
    assert win._vars["engine"].get() == "kokoro"
    assert state(win._engine_radios["omnivoice"]) == "normal"
    assert win._engine_hint.cget("text") == ""
    # the voice combo follows the engine
    assert win._kokoro_row.winfo_manager() == "pack"
    assert win._omni_row.winfo_manager() == ""
    win._vars["engine"].set("omnivoice")
    assert win._omni_row.winfo_manager() == "pack" and win._kokoro_row.winfo_manager() == ""
    assert win._collect()["engine"] == "omnivoice"


def test_engine_radio_disabled_until_installed(tk_root):
    rec = Recorder()
    rec.installed, rec.encoders = False, False
    rec.presets = []
    win = make_window(tk_root, rec)
    assert state(win._engine_radios["omnivoice"]) == "disabled"
    assert "OmniVoice tab" in win._engine_hint.cget("text")
    assert state(win._engine_radios["kokoro"]) == "normal"
    assert "not installed" in win._omni_status.cget("text").lower()
    assert state(win._omni_install_btn) == "normal"
    assert state(win._omni_uninstall_btn) == "disabled"
    assert all(state(b) == "disabled" for b in win._voice_btns.values())
    assert win._creator.button_state("generate") == "disabled"


def test_kokoro_and_omni_voice_combos(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    kokoro = list(win._voice_combo.cget("values"))
    omni = list(win._omni_voice_combo.cget("values"))
    assert all(v.startswith(("af_", "bf_")) for v in kokoro)
    assert [v.split(" ")[0] for v in omni] == ["omni:default", "omni:alma"]
    croatian = list(win._croatian_combo.cget("values"))
    assert croatian[0] == SAME_VOICE and len(croatian) == 3
    assert win._vars["croatian_voice"].get() == SAME_VOICE
    assert win._collect()["croatian_voice"] == ""
    assert win._collect()["omni_voice"] == "omni:default"
    assert not win._is_dirty()


def test_voices_tree_rows_and_buttons(tk_root, monkeypatch):
    rec = Recorder()
    win = make_window(tk_root, rec)
    tree = win._voices_tree
    assert list(tree.get_children()) == ["default", "alma"]
    assert tree.item("alma", "values") == ("alma", "description", "female, young adult",
                                            "English")
    assert tree.item("default", "values")[1] == "default"
    assert all(state(b) == "disabled" for b in win._voice_btns.values())  # no selection
    tree.selection_set("default")
    win._update_voice_buttons()
    assert state(win._voice_btns["play"]) == "normal"
    assert state(win._voice_btns["delete"]) == "disabled"  # 'default' stays
    tree.selection_set("alma")
    win._update_voice_buttons()
    assert state(win._voice_btns["delete"]) == "normal"
    win._play_voice()
    assert rec.played == [("preset", "alma")]
    win._use_selected()  # persisted straight away
    assert rec.applied[-1] == ("omni:alma", "")
    assert win._vars["omni_voice"].get().startswith("omni:alma")
    win._use_selected_croatian()
    assert rec.applied[-1] == ("omni:alma", "omni:alma")
    assert win._vars["croatian_voice"].get().startswith("omni:alma")
    assert not win._is_dirty()
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: True)
    win._delete_voice()
    assert rec.deleted == ["alma"]
    assert "alma" not in tree.get_children()
    assert win._vars["omni_voice"].get().startswith("omni:default")  # fell back
    assert win._vars["croatian_voice"].get() == SAME_VOICE


def test_creator_save_uses_voice_and_refreshes_list(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    c = win._creator
    c.generate()
    c.name_var.set("zora")
    assert c.save() is True
    assert rec.applied[-1] == ("omni:zora", "")  # "use" persists through apply()
    assert "zora" in win._voices_tree.get_children()
    assert win._vars["omni_voice"].get().startswith("omni:zora")


def test_install_flow_with_fake_installer(tk_root):
    rec = Recorder()
    rec.installed, rec.encoders, rec.presets = False, False, []
    pending = []
    win = make_window(tk_root, rec, spawn=pending.append)
    assert "3.6 GB" in win._omni_install_btn.cget("text")
    win._omni_cloning.set(False)
    win._update_omni_buttons()
    assert "2.9 GB" in win._omni_install_btn.cget("text")
    win._omni_install()
    assert state(win._omni_install_btn) == "disabled" and state(win._omni_cancel_btn) == "normal"
    assert state(win._omni_uninstall_btn) == "disabled"
    pending.pop()()  # the worker runs the fake installer
    assert rec.calls[-1] == ("install", False)
    assert state(win._omni_cancel_btn) == "disabled"
    assert win._omni_status.cget("text").startswith("OmniVoice is installed")
    assert state(win._engine_radios["omnivoice"]) == "normal"
    assert state(win._omni_uninstall_btn) == "normal"
    assert win._omni_install_btn.cget("text").startswith("Add voice cloning")  # encoders missing
    assert win._creator.button_state("create") == "disabled"
    rec.presets = ["default"]
    win.refresh()
    assert list(win._voices_tree.get_children()) == ["default"]


def test_install_reports_progress_phases(tk_root):
    rec = Recorder()
    rec.installed, rec.encoders, rec.presets = False, False, []
    phases, bars = [], []
    win = make_window(tk_root, rec, spawn=lambda fn: fn())
    real = win._omni_phase.configure

    def spy(**kw):
        if "text" in kw:
            phases.append(kw["text"])
        real(**kw)
    win._omni_phase.configure = spy
    real_bar = win._omni_progress.configure
    win._omni_progress.configure = lambda **kw: (bars.append(kw.get("value")), real_bar(**kw))
    win._omni_install()
    assert "Downloading model.safetensors 1.2 / 2.4 GB" in phases
    assert "Converting 120 / 197" in phases
    assert any(b and abs(b - 50.0) < 0.1 for b in bars)  # 1.2 / 2.4 GB


def test_install_cancel(tk_root, monkeypatch):
    rec = Recorder()
    rec.installed, rec.encoders, rec.presets = False, False, []
    rec.installer.cancel_after = True
    errors = []
    monkeypatch.setattr(settings_ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    win = make_window(tk_root, rec)
    win._omni_install()
    assert errors == []  # a cancel is not an error
    assert "cancelled" in win._omni_phase.cget("text").lower()
    assert state(win._omni_install_btn) == "normal" and not win._omni_busy
    assert not rec.installed


def test_install_error_is_shown_and_ui_recovers(tk_root, monkeypatch):
    rec = Recorder()
    rec.installed, rec.encoders, rec.presets = False, False, []
    rec.installer.error = model_setup.SetupError("Could not connect to huggingface.co")
    errors = []
    monkeypatch.setattr(settings_ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    win = make_window(tk_root, rec)
    win._omni_install()
    assert errors and "huggingface.co" in errors[0][1]
    assert "Failed" in win._omni_phase.cget("text")
    assert state(win._omni_install_btn) == "normal" and not win._omni_busy


def test_cancel_button_sets_event(tk_root):
    rec = Recorder()
    rec.installed, rec.encoders, rec.presets = False, False, []
    pending = []
    win = make_window(tk_root, rec, spawn=pending.append)
    win._omni_install()
    assert not win._omni_cancel.is_set()
    win._omni_cancel_btn.invoke()
    assert win._omni_cancel.is_set()
    rec.installer.cancel_after = True
    pending.pop()()
    assert not win._omni_busy


def test_uninstall_flow_switches_engine_back(tk_root, monkeypatch):
    rec = Recorder()
    rec.config.speech.engine = "omnivoice"
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: True)
    win = make_window(tk_root, rec)
    assert win._vars["engine"].get() == "omnivoice"
    win._omni_uninstall()
    assert rec.calls[-2:] == ["stop", "uninstall"]
    assert win._vars["engine"].get() == "kokoro"
    assert state(win._engine_radios["omnivoice"]) == "disabled"
    assert state(win._omni_install_btn) == "normal"
    assert all(state(b) == "disabled" for b in win._voice_btns.values())
    assert not win._is_dirty()


def test_uninstall_declined_does_nothing(tk_root, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: False)
    win = make_window(tk_root, rec)
    win._omni_uninstall()
    assert "uninstall" not in rec.calls and rec.installed


def test_refresh_follows_tray_changes_unless_edited(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    rec.config.speech.engine = "omnivoice"  # changed from the tray
    win.refresh()
    assert win._vars["engine"].get() == "omnivoice" and not win._is_dirty()
    win._vars["engine"].set("kokoro")  # the user edits it
    rec.config.speech.engine = "kokoro"
    rec.config.speech.voice = "bf_emma"
    win.refresh()
    assert win._vars["voice"].get().startswith("bf_emma")  # untouched field follows


def test_save_round_trips_engine(tk_root):
    rec = Recorder()
    applied = []
    hooks = rec.hooks()
    hooks.apply = lambda cfg: applied.append(cfg) or None
    win = SettingsWindow(tk_root, hooks, lambda fn: fn(), lambda fn: fn())
    try:
        win.show("voice")
    except tk.TclError:
        pytest.skip("settings window cannot be built headless")
    win._vars["engine"].set("omnivoice")
    win._vars["omni_voice"].set("omni:alma  Alma (OmniVoice)")
    win._vars["croatian_voice"].set("omni:alma  Alma (OmniVoice)")
    assert win.save() is True
    sp = applied[-1].speech
    assert (sp.engine, sp.omni_voice, sp.croatian_voice) == ("omnivoice", "omni:alma",
                                                             "omni:alma")


def test_window_fits_a_1080p_screen(tk_root):
    rec = Recorder()
    win = make_window(tk_root, rec)
    win.top.update_idletasks()
    assert win.top.winfo_reqheight() <= 1080 - 90
    win._nb.select(TABS.index("omnivoice"))
    win.top.update_idletasks()
    assert win.top.winfo_reqheight() <= 1080 - 90
