import dataclasses
import time
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


# ---------------------------------------------------------------- engines / routing

class FakeOmniEngine:
    def __init__(self, model_dir, steps=6):
        self.model_dir, self.steps, self.calls = model_dir, steps, []

    def voices(self):
        return ["omni:default", "omni:alma"]

    def synth(self, text, voice, speed, lang=None):
        self.calls.append((text, voice, speed, lang))
        return np.zeros(10, np.float32)


class FakeKokoro:
    def __init__(self, *paths):
        self.paths, self.calls = paths, []

    def voices(self):
        return ["af_heart"]

    def synth(self, text, voice, speed, lang=None):
        self.calls.append((text, voice, speed, lang))
        return np.ones(10, np.float32)


class FakeOmniModule:
    def __init__(self, present=True):
        self.present = present
        self.OmniVoiceEngine = FakeOmniEngine

    def models_present(self, model_dir):
        return self.present

    def list_presets(self, model_dir):
        return ["omni:default", "omni:alma"]

    def list_preset_info(self, model_dir):
        return [{"name": "default", "voice": "omni:default", "kind": "default",
                 "description": "", "lang": "hr"}]


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
    engine = holder.get(s)  # engine kokoro: Kokoro family only
    assert engine.omni is None
    s_hr = dataclasses.replace(s, engine="omnivoice", omni_voice="omni:alma", omnivoice_steps=9)
    engine = holder.get(s_hr)
    assert isinstance(engine.omni, FakeOmniEngine) and engine.omni.steps == 9
    assert engine.kokoro is holder.get(s).kokoro  # cached, not reloaded
    engine.synth("a", "af_heart", 1.0, "en")
    engine.synth("b", "omni:alma", 1.2, "hr")
    assert engine.kokoro.calls[-1] == ("a", "af_heart", 1.0, "en")
    assert engine.omni.calls[-1] == ("b", "omni:alma", 1.2, "hr")
    assert engine.voices() == ["af_heart", "omni:default", "omni:alma"]


def test_holder_get_with_explicit_voice(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    engine = holder.get(s, voice="omni:alma")  # Croatian text under the Kokoro engine
    assert engine.omni is not None and engine.kokoro is None
    assert holder.get(s).kokoro is not None  # the configured engine loads on demand


def test_holder_omni_only_does_not_load_kokoro(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    engine = holder.get(dataclasses.replace(s, engine="omnivoice"))
    assert engine.kokoro is None
    with pytest.raises(RuntimeError, match="Download the voice model"):
        engine.synth("x", "af_heart", 1.0)


def test_holder_reloads_omni_when_steps_change(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    cfg = dataclasses.replace(s, engine="omnivoice")
    a = holder.get(dataclasses.replace(cfg, omnivoice_steps=6)).omni
    b = holder.get(dataclasses.replace(cfg, omnivoice_steps=8)).omni
    assert a is not b and b.steps == 8


def test_holder_omni_missing(fakes):
    s, omni = fakes
    omni.present = False
    with pytest.raises(RuntimeError, match="Settings > OmniVoice"):
        app.EngineHolder().get(s, voice="omni:default")


def test_holder_unload_omni(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    first = holder.get_omni(s)
    holder.unload_omni()
    assert holder.loaded is None
    assert holder.get_omni(s) is not first


def test_preload_loads_only_the_engine_in_use(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    holder.preload(dataclasses.replace(s, engine="omnivoice"))
    for _ in range(100):
        if holder.loaded is not None:
            break
        time.sleep(0.02)
    assert holder.loaded.omni is not None and holder.loaded.kokoro is None


def test_voice_ready_per_engine(fakes):
    s, omni = fakes
    assert app.voice_ready(s, "af_heart") and app.voice_ready(s, "omni:default")
    assert app.voice_ready(s, "omni_hr:default")  # legacy ids still count as OmniVoice
    omni.present = False
    assert app.voice_ready(s, "af_heart") and not app.voice_ready(s, "omni:default")
    assert app.missing_message("omni:default") == app.MISSING_OMNI
    assert app.missing_message("af_heart") == app.MISSING_MODEL


HR_TEXT = "Ovo je test čitanja na hrvatskom jeziku."
EN_TEXT = "This is plain English text."


def sp(**kw):
    return dataclasses.replace(SpeechConfig(), **kw)


@pytest.mark.parametrize("engine,text,installed,auto,expected", [
    # engine kokoro: only Croatian text goes to OmniVoice
    ("kokoro", EN_TEXT, True, True, ("af_heart", "en", False)),
    ("kokoro", HR_TEXT, True, True, ("omni:alma", "hr", False)),      # croatian_voice
    ("kokoro", HR_TEXT, False, True, ("af_heart", "en", True)),       # not installed: fall back
    ("kokoro", HR_TEXT, True, False, ("af_heart", "en", False)),      # auto language off
    # engine omnivoice: everything goes to OmniVoice, language per request
    ("omnivoice", EN_TEXT, True, True, ("omni:zora", "en", False)),
    ("omnivoice", HR_TEXT, True, True, ("omni:alma", "hr", False)),
    ("omnivoice", HR_TEXT, True, False, ("omni:zora", "en", False)),
    ("omnivoice", EN_TEXT, False, True, ("af_heart", "en", True)),    # not installed: Kokoro
    ("omnivoice", HR_TEXT, False, True, ("af_heart", "en", True)),
])
def test_choose_voice_table(engine, text, installed, auto, expected):
    s = sp(engine=engine, omni_voice="omni:zora", croatian_voice="omni:alma",
           auto_language=auto)
    assert tuple(app.choose_voice(s, text, installed)) == expected


def test_choose_voice_croatian_voice_defaults_to_omni_voice():
    s = sp(omni_voice="omni:zora", croatian_voice="")
    assert app.choose_voice(s, HR_TEXT, True).voice == "omni:zora"
    assert app.choose_voice(sp(), HR_TEXT, True).voice == "omni:default"


def test_choose_voice_accepts_legacy_ids():
    s = sp(engine="omnivoice", omni_voice="omni_hr:alma", croatian_voice="omni_hr:zora")
    assert app.choose_voice(s, EN_TEXT, True).voice == "omni:alma"
    assert app.choose_voice(s, HR_TEXT, True).voice == "omni:zora"
    assert app.choose_voice(sp(voice="omni_hr:alma"), EN_TEXT, False).voice == "af_heart"


def test_route_flags_and_active_voice(fakes):
    s, omni = fakes
    assert app.choose_voice(sp(engine="omnivoice"), EN_TEXT, True).omni
    assert not app.choose_voice(sp(), EN_TEXT, True).omni
    omni_cfg = dataclasses.replace(s, engine="omnivoice", omni_voice="omni:alma")
    assert app.active_voice(omni_cfg) == "omni:alma"
    assert app.active_voice(s) == "af_heart"
    omni.present = False
    assert app.active_voice(dataclasses.replace(s, engine="omnivoice")) == "af_heart"


class SpyOmniEngine(FakeOmniEngine):
    seen: list = []

    def synth(self, text, voice, speed, lang=None):
        SpyOmniEngine.seen.append((text, voice, lang))
        return super().synth(text, voice, speed, lang)


def test_say_routes_croatian_with_lang(fakes, monkeypatch, tmp_path):
    s, omni = fakes
    SpyOmniEngine.seen = []
    omni.OmniVoiceEngine = SpyOmniEngine
    out = tmp_path / "o.wav"
    assert app.say(Config(speech=s), HR_TEXT + " Vidi https://x.org/a", str(out)) == 0
    assert out.exists()
    chunks = [x for x in SpyOmniEngine.seen if x[0] != "Ready."]
    assert chunks and all(v == "omni:default" and lang == "hr" for _, v, lang in chunks)
    assert "poveznica" in chunks[-1][0]  # the URL word follows the language


def test_say_english_with_omnivoice_engine(fakes, tmp_path):
    s, omni = fakes
    SpyOmniEngine.seen = []
    omni.OmniVoiceEngine = SpyOmniEngine
    cfg = Config(speech=dataclasses.replace(s, engine="omnivoice", omni_voice="omni:alma"))
    assert app.say(cfg, "Read https://x.org/a now.", str(tmp_path / "o.wav")) == 0
    assert SpyOmniEngine.seen[-1] == ("Read link now.", "omni:alma", "en")


def test_say_missing_omnivoice_falls_back(fakes, tmp_path, capsys):
    s, omni = fakes
    omni.present = False
    out = tmp_path / "o.wav"
    assert app.say(Config(speech=s), HR_TEXT, str(out)) == 0  # read with Kokoro instead
    assert "OmniVoice is not installed" in capsys.readouterr().err
    cfg = Config(speech=dataclasses.replace(s, engine="omnivoice"))
    assert app.say(cfg, EN_TEXT, str(out)) == 0  # engine omnivoice but not installed: Kokoro


# ---------------------------------------------------------------- long chunks / lang plumbing

def test_prepare_long_chunks_for_any_omnivoice_request():
    from vibetts.textprep import prepare

    text = " ".join(f"Sentence number {i} is short." for i in range(12))
    short, long = prepare(text), prepare(text, long_chunks=True)
    assert len(long) < len(short)
    assert prepare(text, lang="hr") == short  # the language alone does not change chunking


def test_request_route_logs_fallback_once(fakes, caplog):
    s, omni = fakes
    omni.present = False
    a = object.__new__(app.TTSApp)
    a._warned_omni_missing = False
    cfg = dataclasses.replace(s, engine="omnivoice")
    with caplog.at_level("WARNING"):
        r1 = a._request_route(cfg, EN_TEXT)
        a._request_route(cfg, EN_TEXT)
    assert r1.fell_back and r1.voice == "af_heart"
    assert sum("not installed" in rec.message for rec in caplog.records) == 1


class FakeOverlayShow:
    def show(self, *a, **k):
        pass


def test_speak_passes_lang_and_voice_to_the_engine(fakes):
    from vibetts.textprep import Chunk

    s, _ = fakes
    a = object.__new__(app.TTSApp)
    a._clip_engine = None
    a._current = None
    a.cfg = Config(speech=s)
    a.engine = app.EngineHolder()
    a.overlay = FakeOverlayShow()
    reqs = []
    a.speaker = type("Sp", (), {"speak": lambda self, req: reqs.append(req)})()
    a.speak([Chunk("Bok", 0.0)], s, "omni:alma", "hr")
    assert reqs[0].voice == "omni:alma" and reqs[0].lang == "hr"
    engine = a._engine_for_speaker()  # loads OmniVoice for the request's voice only
    assert engine.omni is not None and engine.kokoro is None


# ---------------------------------------------------------------- custom voices (hooks)

class VoiceEngine(FakeOmniEngine):
    def design_clip(self, instruct, seed, text=None, lang=None):
        return ("design", instruct, seed, text, lang)

    def clone_clip(self, wav_path, ref_text, lang=None):
        return ("clone", wav_path, ref_text, lang)


class VoiceModule(FakeOmniModule):
    def __init__(self):
        super().__init__()
        self.OmniVoiceEngine = VoiceEngine
        self.saved, self.deleted, self.encoders = [], [], True

    def save_preset(self, model_dir, name, clip, overwrite=False):
        self.saved.append((name, clip))
        return f"omni:{name}"

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


def make_app(s):
    a = object.__new__(app.TTSApp)
    a.cfg = Config(speech=s)
    a.engine = app.EngineHolder()
    a.tray, a.overlay, a.settings = FakeTray(), FakeOverlay(), None
    a._clip_engine = None
    a._current = None
    saved = []
    a._save = lambda: saved.append(dataclasses.replace(a.cfg.speech))
    a.saved_configs = saved
    return a


@pytest.fixture
def vapp(fakes, monkeypatch):
    s, _ = fakes
    omni = VoiceModule()
    monkeypatch.setattr(app, "_omni", lambda: omni)
    return make_app(s), omni


def test_get_omni_loads_shared_engine_for_kokoro_engine(fakes):
    s, _ = fakes
    holder = app.EngineHolder()
    holder.get(s)  # Kokoro engine configured
    engine = holder.get_omni(s)
    assert isinstance(engine, FakeOmniEngine)
    assert holder.get_omni(s) is engine is holder.get(s, voice="omni:alma").omni


def test_get_omni_missing_files(fakes):
    s, omni = fakes
    omni.present = False
    with pytest.raises(RuntimeError, match="Settings > OmniVoice"):
        app.EngineHolder().get_omni(s)


def test_settings_hooks_wiring(vapp, monkeypatch):
    a, omni = vapp
    a._autostart = type("A", (), {"is_enabled": staticmethod(lambda: False)})
    monkeypatch.setattr("vibetts.settings_ui.SettingsWindow", FakeSettingsWindow)
    hooks = a._settings_window().hooks
    assert hooks.design_clip("female", 3) == ("design", "female", 3, None, None)
    assert hooks.design_clip("male", 4, None, "en") == ("design", "male", 4, None, "en")
    assert hooks.clone_clip("a.wav", "Ovo je moj glas, hvala.") == (
        "clone", Path("a.wav"), "Ovo je moj glas, hvala.", "hr")  # language guessed from the text
    assert hooks.clone_clip("a.wav", "Hello there my friend.")[3] == "en"
    assert hooks.encoders_present() is True
    omni.encoders = False
    assert hooks.encoders_present() is False
    assert hooks.save_voice("ana", "clip") == "omni:ana"
    assert omni.saved == [("ana", "clip")]
    assert a.tray.updates == 1  # tray menu refreshed after save
    assert hooks.omni_voices() == ["omni:default", "omni:alma"]
    assert hooks.omni_voice_info()[0]["name"] == "default"


def test_install_hooks_use_the_installer_contract(vapp, monkeypatch):
    import types

    a, omni = vapp
    calls = []
    fake = types.ModuleType("vibetts.omnivoice_setup")
    fake.DOWNLOAD_BYTES = lambda include: 2_900_000_000 + (654_000_000 if include else 0)
    fake.status = lambda d: {"installed": True, "encoders": False, "bytes_on_disk": 5}
    fake.install = lambda d, include_encoders=True, progress=None, cancel=None, keep_cache=False: (
        calls.append(("install", Path(d).name, include_encoders, progress, cancel)))
    fake.uninstall = lambda d, keep_voices=True: calls.append(("uninstall", keep_voices))
    monkeypatch.setattr(app.TTSApp, "_setup", staticmethod(lambda: fake))
    monkeypatch.setattr("vibetts.settings_ui.SettingsWindow", FakeSettingsWindow)
    a._autostart = type("A", (), {"is_enabled": staticmethod(lambda: False)})
    a.stop = lambda: calls.append("stop")
    a.cfg.speech.preload = False
    hooks = a._settings_window().hooks
    assert hooks.omni_download_bytes(True) == 3_554_000_000
    assert hooks.omni_status() == {"installed": True, "encoders": False, "bytes_on_disk": 5}
    marker = object()
    hooks.omni_install(True, print, marker)
    assert calls[-1] == ("install", "omni", True, print, marker)
    assert a.tray.updates == 1  # menus refreshed after the install
    a.cfg.speech.engine = "omnivoice"
    hooks.omni_uninstall()
    assert calls[-2:] == ["stop", ("uninstall", True)]
    assert a.cfg.speech.engine == "kokoro" and len(a.saved_configs) == 1


def test_install_failure_still_refreshes(vapp, monkeypatch):
    import types

    from vibetts import model_setup

    a, _ = vapp
    fake = types.ModuleType("vibetts.omnivoice_setup")

    def boom(*args, **kw):
        raise model_setup.SetupError("no network")
    fake.install = boom
    monkeypatch.setattr(app.TTSApp, "_setup", staticmethod(lambda: fake))
    with pytest.raises(model_setup.SetupError):
        a._omni_install(True, None, None)
    assert a.tray.updates == 1


def test_delete_voice_falls_back_to_default(vapp):
    a, omni = vapp
    a.cfg.speech.croatian_voice = "omni:alma"
    a.cfg.speech.omni_voice = "omni:alma"
    a._delete_voice("alma")
    assert omni.deleted == ["alma"]
    assert a.cfg.speech.croatian_voice == ""  # "same as the OmniVoice voice"
    assert a.cfg.speech.omni_voice == "omni:default"
    assert len(a.saved_configs) == 1 and a.tray.updates == 1


def test_delete_other_voice_keeps_config(vapp):
    a, omni = vapp
    a.cfg.speech.croatian_voice = "omni:alma"
    a._delete_voice("omni:zora")  # an id is accepted too
    assert omni.deleted == ["zora"]
    assert a.cfg.speech.croatian_voice == "omni:alma"
    assert a.saved_configs == [] and a.tray.updates == 1


def test_play_preset_plays_the_stored_wav(vapp):
    a, omni = vapp
    played = []
    omni.read_wav = lambda path: (np.full(3, 0.25, np.float32), 24000)
    a._play_clip = lambda clip: played.append(clip.audio.tolist())
    a._play_preset("alma")
    assert played == [[0.25, 0.25, 0.25]]


def test_tray_set_voice_switches_engine(fakes):
    s, _ = fakes
    a = make_app(s)
    a._set_voice("omni_hr:alma")  # a legacy id is normalised
    sp_ = a.cfg.speech
    assert (sp_.engine, sp_.omni_voice, sp_.voice) == ("omnivoice", "omni:alma", "af_heart")
    assert a._current_voice() == "omni:alma"
    a._set_voice("bf_emma")
    assert (sp_.engine, sp_.voice, sp_.omni_voice) == ("kokoro", "bf_emma", "omni:alma")
    assert a._current_voice() == "bf_emma"
    assert len(a.saved_configs) == 2 and a.tray.updates == 2


def test_tray_set_engine(fakes):
    s, omni = fakes
    a = make_app(dataclasses.replace(s, preload=False))
    a._set_engine("omnivoice")
    assert a.cfg.speech.engine == "omnivoice"
    a._set_engine("bogus")
    assert a.cfg.speech.engine == "omnivoice"
    a._set_engine("kokoro")
    omni.present = False
    a._set_engine("omnivoice")  # not installed: refused
    assert a.cfg.speech.engine == "kokoro"


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
    a._speak_voice = None
    a._speak_cfg = a.cfg.speech
    a.speaker = Speaker(a._engine_for_speaker, Out)
    a._play_clip(type("Clip", (), {"audio": np.full(5, 0.5, np.float32)})())
    for _ in range(100):
        if not a.speaker.busy:
            break
        time.sleep(0.02)
    assert len(written) == 1 and written[0].tolist() == [0.5] * 5
    a.speaker.stop(wait=1.0)


# ---------------------------------------------------------------- tray menus

def make_tray(engine="kokoro", omni=True, voices=None, current="af_heart"):
    from vibetts.tray import Tray, TrayHooks

    calls = []
    noop = lambda *a: None  # noqa: E731
    hooks = TrayHooks(
        is_enabled=lambda: True, set_enabled=noop, is_speaking=lambda: False, stop=noop,
        model_missing=lambda: False, download_model=noop,
        voices=lambda: voices if voices is not None else
        ["af_heart", "am_adam", "bf_emma", "omni:default", "omni:ana_marija"],
        current_voice=lambda: current, set_voice=lambda v: calls.append(("voice", v)),
        current_speed=lambda: 1.0, set_speed=noop, is_autostart=lambda: False,
        set_autostart=noop, open_settings=noop, open_config=noop, open_logs=noop,
        reload_config=noop, quit=noop, current_engine=lambda: engine,
        set_engine=lambda e: calls.append(("engine", e)), omni_available=lambda: omni)
    return Tray(hooks), calls


def test_tray_engine_submenu():
    tray, calls = make_tray(engine="omnivoice", omni=False)
    kokoro, omni = tray._engine_menu().items
    assert (kokoro.text, omni.text) == ("Kokoro", "OmniVoice")
    assert kokoro.radio and omni.radio
    assert kokoro.enabled and not omni.enabled  # greyed out until installed
    assert not kokoro.checked and omni.checked
    tray, calls = make_tray(omni=True)
    assert tray._engine_menu().items[1].enabled
    tray._engine_menu().items[1](tray._icon)
    assert calls == [("engine", "omnivoice")]


def test_tray_voices_grouped_by_language_and_omnivoice():
    tray, calls = make_tray(current="omni:ana_marija")
    groups = {g.text: [i.text for i in g.submenu.items] for g in tray._voice_menu().items}
    assert groups["American English"] == ["Heart (female)", "Adam (male)"]
    assert groups["British English"] == ["Emma (female)"]
    assert groups["OmniVoice"] == ["Default", "Ana Marija"]
    assert "Croatian" not in groups
    omni_group = [g for g in tray._voice_menu().items if g.text == "OmniVoice"][0]
    assert [i.checked for i in omni_group.submenu.items] == [False, True]
    omni_group.submenu.items[0](tray._icon)  # picking a voice goes through set_voice
    assert calls == [("voice", "omni:default")]


def test_tray_menu_has_engine_and_voice_entries():
    tray, _ = make_tray()
    texts = [getattr(i, "text", "") for i in tray._menu_items()]
    assert "Engine" in texts and "Voice" in texts
    tray, _ = make_tray(voices=[])
    assert "Voice" not in [getattr(i, "text", "") for i in tray._menu_items()]
