import dataclasses
import time
from pathlib import Path
from pathlib import Path

import numpy as np
import pytest

from vibetts import app
from vibetts.config import Config, SpeechConfig


def test_engine_holder_missing_files(tmp_path):
    s = SpeechConfig(model=str(tmp_path / "m.onnx"), voices=str(tmp_path / "v.bin"))
    assert not app.models_present(s)
    with pytest.raises(RuntimeError, match="Download the voice model"):
        app.EngineHolder().get(s)


def test_voices_in_file(tmp_path):
    path = tmp_path / "voices.bin"
    with open(path, "wb") as fh:
        np.savez(fh, af_heart=np.zeros(2), jf_alpha=np.zeros(2), bm_george=np.zeros(2))
    assert app.voices_in_file(path) == ["af_heart", "bm_george"]


def test_short():
    assert app._short("a  b\nc") == "a b c"
    assert len(app._short("x" * 100, 10)) == 10


# ---------------------------------------------------------------- Croatian / routing

class FakeOmniEngine:
    def __init__(self, model_dir, steps=6):
        self.model_dir, self.steps, self.calls = model_dir, steps, []

    def voices(self):
        return ["omni_hr:default", "omni_hr:alma"]

    def synth(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        return np.zeros(10, np.float32)


class FakeKokoro:
    def __init__(self, *paths):
        self.paths, self.calls = paths, []

    def voices(self):
        return ["af_heart"]

    def synth(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        return np.ones(10, np.float32)


class FakeOmniModule:
    def __init__(self, present=True):
        self.present = present
        self.OmniVoiceEngine = FakeOmniEngine

    def models_present(self, model_dir):
        return self.present

    def list_presets(self, model_dir):
        return ["omni_hr:default", "omni_hr:alma"]


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    omni = FakeOmniModule()
    monkeypatch.setattr(app, "_omni", lambda: omni)
    monkeypatch.setattr("vibetts.tts.KokoroEngine", FakeKokoro)
    model, voices = tmp_path / "m.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    s = SpeechConfig(model=str(model), voices=str(voices), omnivoice_dir=str(tmp_path / "omni"))
    return s, omni


def test_holder_routes_by_voice(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    assert holder.loaded is None
    engine = holder.get(s)  # Kokoro family only
    assert engine.omni is None
    s_hr = dataclasses.replace(s, voice="omni_hr:alma", omnivoice_steps=9)
    engine = holder.get(s_hr)
    assert isinstance(engine.omni, FakeOmniEngine) and engine.omni.steps == 9
    assert engine.kokoro is holder.get(s).kokoro  # cached, not reloaded
    engine.synth("a", "af_heart", 1.0)
    engine.synth("b", "omni_hr:alma", 1.2)
    assert engine.kokoro.calls[-1] == ("a", "af_heart", 1.0)
    assert engine.omni.calls[-1] == ("b", "omni_hr:alma", 1.2)
    assert engine.voices() == ["af_heart", "omni_hr:default", "omni_hr:alma"]


def test_holder_omni_only_does_not_load_kokoro(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    engine = holder.get(dataclasses.replace(s, voice="omni_hr:default"))
    assert engine.kokoro is None
    with pytest.raises(RuntimeError, match="Download the voice model"):
        engine.synth("x", "af_heart", 1.0)


def test_holder_reloads_omni_when_steps_change(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    a = holder.get(dataclasses.replace(s, voice="omni_hr:default", omnivoice_steps=6)).omni
    b = holder.get(dataclasses.replace(s, voice="omni_hr:default", omnivoice_steps=8)).omni
    assert a is not b and b.steps == 8


def test_holder_omni_missing(fakes):
    s, omni = fakes
    omni.present = False
    with pytest.raises(RuntimeError, match="build_omnivoice.py"):
        app.EngineHolder().get(dataclasses.replace(s, voice="omni_hr:default"))


def test_preload_loads_only_voice_family(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    holder.preload(dataclasses.replace(s, voice="omni_hr:default"))
    for _ in range(100):
        if holder.loaded is not None:
            break
        time.sleep(0.02)
    assert holder.loaded.omni is not None and holder.loaded.kokoro is None


def test_voice_ready_per_family(fakes):
    s, omni = fakes
    assert app.voice_ready(s, "af_heart") and app.voice_ready(s, "omni_hr:default")
    omni.present = False
    assert app.voice_ready(s, "af_heart") and not app.voice_ready(s, "omni_hr:default")
    assert app.missing_message("omni_hr:default") == app.MISSING_OMNI
    assert app.missing_message("af_heart") == app.MISSING_MODEL


HR_TEXT = "Ovo je test čitanja na hrvatskom jeziku."


def test_choose_voice():
    s = SpeechConfig()
    assert app.choose_voice(s, HR_TEXT, True) == ("omni_hr:default", False)
    assert app.choose_voice(s, "This is plain English text.", True) == ("af_heart", False)
    assert app.choose_voice(s, HR_TEXT, False) == ("af_heart", True)  # models missing: fall back
    off = dataclasses.replace(s, auto_language=False)
    assert app.choose_voice(off, HR_TEXT, True) == ("af_heart", False)
    omni = dataclasses.replace(s, voice="omni_hr:alma")  # already Croatian: keep the choice
    assert app.choose_voice(omni, "English text here", True) == ("omni_hr:alma", False)
    assert app.choose_voice(omni, HR_TEXT, True) == ("omni_hr:alma", False)


def test_voice_language():
    assert app.voice_language("omni_hr:default") == "hr"
    assert app.voice_language("af_heart") == "en"


def test_say_routes_croatian(fakes, monkeypatch, tmp_path, capsys):
    s, _ = fakes
    cfg = Config(speech=s)
    out = tmp_path / "o.wav"
    assert app.say(cfg, HR_TEXT + " Vidi https://x.org/a", str(out)) == 0
    assert out.exists()


def test_say_missing_croatian_model_falls_back(fakes, tmp_path, capsys):
    s, omni = fakes
    omni.present = False
    out = tmp_path / "o.wav"
    assert app.say(Config(speech=s), HR_TEXT, str(out)) == 0  # read with Kokoro instead
    assert "Croatian model is missing" in capsys.readouterr().err
    assert app.say(Config(speech=dataclasses.replace(s, voice="omni_hr:default")), HR_TEXT,
                   str(out)) == 1
    assert "build_omnivoice.py" in capsys.readouterr().err


# ---------------------------------------------------------------- custom voices (hooks)

class VoiceEngine(FakeOmniEngine):
    def design_clip(self, instruct, seed, text=None):
        return ("design", instruct, seed, text)

    def clone_clip(self, wav_path, ref_text):
        return ("clone", wav_path, ref_text)


class VoiceModule(FakeOmniModule):
    def __init__(self):
        super().__init__()
        self.OmniVoiceEngine = VoiceEngine
        self.saved, self.deleted, self.encoders = [], [], True

    def save_preset(self, model_dir, name, clip, overwrite=False):
        self.saved.append((name, clip))
        return f"omni_hr:{name}"

    def delete_preset(self, model_dir, name):
        self.deleted.append(name)

    def encoders_present(self, model_dir):
        return self.encoders


class FakeTray:
    def __init__(self):
        self.updates = 0

    def update_menu(self):
        self.updates += 1


class FakeOverlay:
    root = None

    def call_soon(self, fn):
        fn()


class FakeSettingsWindow:
    refreshed = 0

    def __init__(self, root, hooks, post):
        self.hooks = hooks

    def refresh(self):
        type(self).refreshed += 1


@pytest.fixture
def vapp(fakes, monkeypatch):
    s, _ = fakes
    omni = VoiceModule()
    monkeypatch.setattr(app, "_omni", lambda: omni)
    a = object.__new__(app.TTSApp)
    a.cfg = Config(speech=s)
    a.engine = app.EngineHolder()
    a.tray, a.overlay, a.settings = FakeTray(), FakeOverlay(), None
    a._clip_engine = None
    a._current = None
    saved = []
    a._save = lambda: saved.append(dataclasses.replace(a.cfg.speech))
    a.saved_configs = saved
    return a, omni


def test_get_omni_loads_shared_engine_for_kokoro_voice(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    holder.get(s)  # Kokoro voice configured
    engine = holder.get_omni(s)
    assert isinstance(engine, FakeOmniEngine)
    assert holder.get_omni(s) is engine is holder.get(
        dataclasses.replace(s, voice="omni_hr:alma")).omni


def test_get_omni_missing_files(fakes):
    s, omni = fakes
    omni.present = False
    with pytest.raises(RuntimeError, match="build_omnivoice.py"):
        app.EngineHolder().get_omni(s)


def test_settings_hooks_wiring(vapp, monkeypatch):
    a, omni = vapp
    a._autostart = type("A", (), {"is_enabled": staticmethod(lambda: False)})
    monkeypatch.setattr("vibetts.settings_ui.SettingsWindow", FakeSettingsWindow)
    hooks = a._settings_window().hooks
    assert hooks.design_clip("female", 3) == ("design", "female", 3, None)
    assert hooks.design_clip("male", 4, "Zdravo") == ("design", "male", 4, "Zdravo")
    assert hooks.clone_clip("a.wav", "tekst") == ("clone", Path("a.wav"), "tekst")
    assert hooks.encoders_present() is True
    omni.encoders = False
    assert hooks.encoders_present() is False
    assert hooks.save_voice("ana", "clip") == "omni_hr:ana"
    assert omni.saved == [("ana", "clip")]
    assert a.tray.updates == 1  # tray menu refreshed after save


def test_delete_voice_falls_back_to_default(vapp):
    a, omni = vapp
    a.cfg.speech.croatian_voice = "omni_hr:alma"
    a.cfg.speech.voice = "omni_hr:alma"
    a._delete_voice("alma")
    assert omni.deleted == ["alma"]
    assert a.cfg.speech.croatian_voice == "omni_hr:default"
    assert a.cfg.speech.voice == "omni_hr:default"
    assert len(a.saved_configs) == 1 and a.tray.updates == 1


def test_delete_other_voice_keeps_config(vapp):
    a, omni = vapp
    a.cfg.speech.croatian_voice = "omni_hr:alma"
    a._delete_voice("omni_hr:zora")  # an id is accepted too
    assert omni.deleted == ["zora"]
    assert a.cfg.speech.croatian_voice == "omni_hr:alma"
    assert a.saved_configs == [] and a.tray.updates == 1


def test_play_clip_uses_output_and_stop(vapp):
    from vibetts.tts import Speaker

    a, _ = vapp
    written = []

    class Out:
        def __init__(self, device=""):
            self.device = device

        def write(self, block):
            written.append(block.copy())

        def drain(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    a.cfg.speech.device = "Speakers"
    a.speaker = Speaker(a._engine_for_speaker, Out)
    a._play_clip(type("Clip", (), {"audio": np.full(5, 0.5, np.float32)})())
    for _ in range(100):
        if not a.speaker.busy:
            break
        time.sleep(0.02)
    assert len(written) == 1 and written[0].tolist() == [0.5] * 5
    a.speaker.stop(wait=1.0)
