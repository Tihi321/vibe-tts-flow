import pytest

from vibetts.config import Config
from vibetts.settings_ui import config_from_form, form_from_config, model_choice


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
    ("omnivoice_steps", "3"), ("omnivoice_steps", "17"), ("omnivoice_steps", "x"),
    ("croatian_voice", ""),
])
def test_invalid_values(key, value):
    form = form_from_config(Config())
    form[key] = value
    with pytest.raises(ValueError):
        config_from_form(Config(), form)


def test_croatian_fields_round_trip():
    cfg = Config()
    cfg.speech.omnivoice_steps = 12
    cfg.speech.croatian_voice = "omni_hr:alma"
    cfg.speech.auto_language = False
    assert config_from_form(cfg, form_from_config(cfg)) == cfg


def test_croatian_voice_display_string_parsed():
    form = form_from_config(Config())
    form["croatian_voice"] = "omni_hr:alma  Alma (Croatian)"
    form["omnivoice_steps"] = "16"
    cfg = config_from_form(Config(), form)
    assert cfg.speech.croatian_voice == "omni_hr:alma"
    assert cfg.speech.omnivoice_steps == 16


# ---------------------------------------------------------------- custom voice dialog

import copy  # noqa: E402
import re  # noqa: E402
import tkinter as tk  # noqa: E402
from pathlib import Path  # noqa: E402

from vibetts import settings_ui  # noqa: E402
from vibetts.settings_ui import (ANY, ENCODERS_MISSING, AddVoiceDialog,  # noqa: E402
                                 SettingsHooks, SettingsWindow, build_instruct)

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


class Recorder:
    def __init__(self):
        self.calls, self.saved, self.deleted, self.played, self.applied = [], [], [], [], []
        self.voices = ["omni_hr:default", "omni_hr:alma"]
        self.encoders = True
        self.error = None
        self.config = Config()

    def hooks(self):
        def design(instruct, seed, text=None):
            if self.error:
                raise self.error
            self.calls.append(("design", instruct, seed, text))
            return {"clip": "d", "instruct": instruct}

        def clone(wav, text):
            self.calls.append(("clone", wav, text))
            return {"clip": "c"}

        def save(name, clip, overwrite=False):
            self.saved.append((name, clip))
            return f"omni_hr:{name}"

        def apply(cfg):
            self.applied.append(cfg.speech.croatian_voice)
            return None

        return SettingsHooks(
            get_config=lambda: copy.deepcopy(self.config), apply=apply,
            is_autostart=lambda: False, set_autostart=lambda v: None,
            set_show_on_start=lambda v: None, models_dir=lambda: Path("."),
            voices=lambda: ["af_heart"], output_devices=lambda: [], play_test=lambda c, t: None,
            stop=lambda: self.calls.append("stop"), open_logs=lambda: None,
            open_config=lambda: None, version="t",
            omni_present=lambda: True, croatian_voices=lambda: list(self.voices),
            design_clip=design, clone_clip=clone, save_voice=save,
            delete_voice=lambda name: self.deleted.append(name),
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


def make_dialog(root, rec, on_saved=None, spawn=lambda fn: fn()):
    return AddVoiceDialog(root, rec.hooks(), lambda fn: fn(), on_saved or (lambda v, u: None),
                          spawn)


def test_build_instruct():
    assert build_instruct({}, OPTIONS) == ""
    assert build_instruct({k: ANY for k in OPTIONS}, OPTIONS) == ""
    sel = {"pitch": "low pitch", "gender": "female", "age": "young adult", "style": ANY}
    assert build_instruct(sel, OPTIONS) == "female, young adult, low pitch"  # option order
    assert build_instruct({"style": "whisper"}, OPTIONS) == "whisper"


def test_instruct_options_fallback(monkeypatch):
    monkeypatch.setattr(settings_ui, "_omni", lambda: object())  # no INSTRUCT_OPTIONS yet
    assert list(settings_ui.instruct_options()) == ["gender", "age", "pitch", "style"]


def test_dialog_assembles_instruct_from_dropdowns(tk_root):
    rec = Recorder()
    d = make_dialog(tk_root, rec)
    assert all(v.get() == ANY for v in d.option_vars.values())
    assert list(d.option_combos["gender"].cget("values"))[0] == ANY
    d.generate()  # all (any): plain random voice
    assert rec.calls[-1] == ("design", "", 0, None)
    d.option_vars["gender"].set("female")
    d.option_vars["age"].set("young adult")
    d.option_vars["pitch"].set("low pitch")
    d.seed_var.set("42")
    d.generate()
    assert rec.calls[-1] == ("design", "female, young adult, low pitch", 42, None)
    assert d.button_state("play_design") == "normal" and d.button_state("save_design") == "normal"
    d.close()


def test_try_another_picks_new_seed_and_generates(tk_root, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(settings_ui.random, "randint", lambda a, b: 777)
    d = make_dialog(tk_root, rec)
    d.try_another()
    assert d.seed_var.get() == "777" and rec.calls[-1][2] == 777
    d.close()


def test_dialog_busy_state_and_inline_errors(tk_root):
    rec = Recorder()
    pending = []
    d = make_dialog(tk_root, rec, spawn=pending.append)
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
    d.close()


def test_save_validates_and_sanitizes_name(tk_root):
    rec = Recorder()
    saved = []
    d = make_dialog(tk_root, rec, on_saved=lambda v, u: saved.append((v, u)))
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
    assert saved == [("omni_hr:ana_maria", True)]  # "Use as Croatian voice" defaults on
    assert d._closed


def test_save_existing_name_needs_replace_confirmation(tk_root, monkeypatch):
    rec = Recorder()
    answers = iter([False, True])
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: next(answers))
    d = make_dialog(tk_root, rec)
    d.generate()
    d.name_var.set("alma")
    d.use_var.set(False)
    assert d.save() is False and rec.saved == [] and not d._closed
    assert d.save() is True and rec.saved[0][0] == "alma"


def test_clone_tab_requires_wav_and_transcript(tk_root):
    rec = Recorder()
    d = make_dialog(tk_root, rec)
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
    d.close()


def test_encoders_missing_disables_cloning(tk_root):
    rec = Recorder()
    rec.encoders = False
    d = make_dialog(tk_root, rec)
    assert d.encoders_label.cget("text") == ENCODERS_MISSING
    assert d.button_state("create") == "disabled"
    d.wav_var.set("x.wav")
    d.transcript.insert("1.0", "tekst")
    d.create()
    assert rec.calls == []
    d.close()


def test_play_goes_through_hook(tk_root):
    rec = Recorder()
    d = make_dialog(tk_root, rec)
    d.play("design")
    assert rec.played == []  # nothing generated yet
    d.generate()
    d.play("design")
    assert rec.played == [{"clip": "d", "instruct": ""}]
    d.close()


def test_settings_window_voice_buttons(tk_root, monkeypatch):
    rec = Recorder()
    win = SettingsWindow(tk_root, rec.hooks(), lambda fn: fn())
    try:
        win.show("voice")
    except tk.TclError:
        pytest.skip("settings window cannot be built headless")
    state = lambda b: str(b.cget("state"))  # noqa: E731
    assert state(win._add_voice_btn) == "normal"
    assert state(win._delete_voice_btn) == "disabled"  # default selected
    win._vars["croatian_voice"].set("omni_hr:alma  Alma (Croatian)")
    win._update_voice_buttons()
    assert state(win._delete_voice_btn) == "normal"
    win._vars["croatian_voice"].set("omni_hr:ghost")  # not a stored preset
    win._update_voice_buttons()
    assert state(win._delete_voice_btn) == "disabled"
    win._vars["croatian_voice"].set("omni_hr:alma")
    win._vars["voice"].set("omni_hr:alma")
    win._update_voice_buttons()
    monkeypatch.setattr(settings_ui.messagebox, "askyesno", lambda *a, **k: True)
    rec.voices = ["omni_hr:default"]  # what the app lists after the delete
    win._delete_voice()
    assert rec.deleted == ["alma"]
    assert win._vars["croatian_voice"].get().startswith("omni_hr:default")
    assert win._vars["voice"].get().startswith("omni_hr:default")
    # save with "use as Croatian voice" persists through apply() and updates the combo
    rec.voices = ["omni_hr:default", "omni_hr:zora"]
    win._voice_saved("omni_hr:zora", True)
    assert rec.applied == ["omni_hr:zora"]
    assert win._vars["croatian_voice"].get().startswith("omni_hr:zora")
