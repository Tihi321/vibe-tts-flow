import dataclasses
import threading
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
                 "description": "", "lang": "hr"},
                {"name": "alma", "voice": "omni:alma", "kind": "design",
                 "description": "female, young adult", "lang": "en"}]

    INSTRUCT_OPTIONS = {"gender": ["male", "female"]}

    def sanitize_name(self, s):
        name = "".join(c if c.isalnum() or c in "-_" else "-" for c in s.lower()).strip("-_")
        if not name or name == "default":
            raise ValueError("bad voice name")
        return name


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

    def show(self, *a, **k):
        pass


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
    a._export_lock = threading.Lock()
    a._export_cancel = None
    a._mcp, a._mcp_key, a._mcp_error = None, None, None
    a.enabled = True
    a.flashes = []
    a._flash = lambda kind, text, seconds=1.5: a.flashes.append((kind, text))
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


# ---------------------------------------------------------------- plan_speech / run_export

def _wav_frames(path):
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes()


def test_plan_speech_auto_route(fakes):
    s, _ = fakes
    chunks, voice, lang, omni = app.plan_speech(s, EN_TEXT)
    assert (voice, lang, omni) == ("af_heart", "en", False) and chunks
    chunks, voice, lang, omni = app.plan_speech(sp(croatian_voice="omni:alma",
                                                   omnivoice_dir=s.omnivoice_dir), HR_TEXT)
    assert (voice, lang, omni) == ("omni:alma", "hr", True)


def test_plan_speech_lang_override(fakes):
    s, omni = fakes
    assert app.plan_speech(s, EN_TEXT, lang="hr")[1:3] == ("omni:default", "hr")
    assert app.plan_speech(s, HR_TEXT, lang="en")[1:3] == ("af_heart", "en")
    omni.present = False
    assert app.plan_speech(s, EN_TEXT, lang="hr")[1:3] == ("af_heart", "en")


def test_plan_speech_explicit_voices(fakes):
    s, _ = fakes
    _, voice, lang, omni = app.plan_speech(s, "Ovo je test.", voice="bf_emma", lang="en")
    assert (voice, lang, omni) == ("bf_emma", "en", False)
    _, voice, lang, omni = app.plan_speech(s, HR_TEXT, voice="bf_emma")
    assert (voice, lang, omni) == ("bf_emma", "hr", False)  # language only steers text prep
    _, voice, lang, omni = app.plan_speech(s, EN_TEXT, voice="omni_hr:alma")
    assert (voice, lang, omni) == ("omni:alma", "en", True)  # legacy id normalized
    _, voice, lang, _ = app.plan_speech(s, EN_TEXT, voice="omni:alma", lang="hr")
    assert (voice, lang) == ("omni:alma", "hr")


def test_plan_speech_max_chars_and_markdown(fakes):
    s, _ = fakes
    text = "First sentence here. Second sentence here. Third one is here too."
    assert len(app.plan_speech(s, text, voice="af_heart", max_chars=25)[0]) \
        < len(app.plan_speech(s, text, voice="af_heart")[0])
    chunks = app.plan_speech(s, "**bold** word", voice="af_heart", markdown=True)[0]
    assert "*" not in chunks[0].text


def test_run_export_writes_wav(fakes, tmp_path):
    from vibetts.export import ExportProgress, ExportRequest

    s, _ = fakes
    s = dataclasses.replace(s, sentence_pause=0.0)
    kokoro = FakeKokoro()
    progress = []
    req = ExportRequest("One sentence. Another sentence.", Path("a.wav"), speed=1.3)
    res = app.run_export(kokoro, s, req, tmp_path / "out", progress.append)
    assert res.path == tmp_path / "out" / "a.wav" and res.fmt == "wav"
    assert res.chunks == len(kokoro.calls) and _wav_frames(res.path) == 10 * res.chunks
    assert kokoro.calls[0][1:3] == ("af_heart", 1.3) and res.voice == "af_heart"
    assert isinstance(progress[-1], ExportProgress) and progress[-1].done == res.chunks


def test_run_export_default_format_and_errors(fakes, tmp_path):
    from vibetts.export import ExportRequest

    s, omni = fakes
    res = app.run_export(FakeKokoro(), s, ExportRequest("Hello there.", title="My doc"),
                         tmp_path, fmt_default="wav")
    assert res.path == tmp_path / "My doc.wav"
    with pytest.raises(ValueError, match="does not match"):
        app.run_export(FakeKokoro(), s, ExportRequest("Hi.", Path("x.wav"), fmt="mp3"), tmp_path)
    with pytest.raises(ValueError, match="Only .wav and .mp3"):
        app.run_export(FakeKokoro(), s, ExportRequest("Hi.", Path("x.ogg")), tmp_path)
    with pytest.raises(FileExistsError):
        app.run_export(FakeKokoro(), s, ExportRequest("Hi.", Path("My doc.wav")), tmp_path,
                       fmt_default="wav")
    omni.present = False
    with pytest.raises(RuntimeError, match="Settings > OmniVoice"):
        app.run_export(FakeKokoro(), s, ExportRequest("Hi.", voice="omni:alma"), tmp_path)
    with pytest.raises(ValueError, match="Nothing to read"):
        app.run_export(FakeKokoro(), s, ExportRequest("  "), tmp_path)


def test_run_export_mp3(fakes, tmp_path):
    pytest.importorskip("lameenc")
    from vibetts.export import ExportRequest

    s, _ = fakes
    res = app.run_export(FakeKokoro(), s, ExportRequest("Hello there.", fmt="mp3"), tmp_path,
                         bitrate_default=32)
    assert res.fmt == "mp3" and res.path.suffix == ".mp3" and res.bytes > 0
    res = app.run_export(FakeKokoro(), s, ExportRequest("Hello there.", Path("b.mp3")), tmp_path)
    assert res.fmt == "mp3"


# ---------------------------------------------------------------- TTSApp export

class FakeSpeaker:
    def __init__(self, busy=False):
        self.busy = busy
        self.stopped = 0

    def stop(self, wait=0.0):
        self.stopped += 1


@pytest.fixture
def eapp(fakes, tmp_path):
    s, _ = fakes
    a = make_app(dataclasses.replace(s, sentence_pause=0.0))
    a.cfg.export.folder = str(tmp_path / "exports")
    a.cfg.export.format = "wav"
    a.speaker = FakeSpeaker()
    return a


def test_export_audio_uses_config_snapshot_and_folder(eapp, tmp_path):
    from vibetts.export import ExportRequest

    res = eapp.export_audio(ExportRequest("Hello there.", title="doc"))
    assert res.path == tmp_path / "exports" / "doc.wav" and res.path.exists()
    assert eapp._export_cancel is None and not eapp._export_lock.locked()


def test_export_audio_busy_when_lock_held(eapp):
    from vibetts.export import ExportBusy, ExportRequest

    eapp._export_lock.acquire()
    with pytest.raises(ExportBusy):
        eapp.export_audio(ExportRequest("Hello."))
    assert eapp._export_lock.locked()  # still held by the "other" export


def test_export_audio_releases_lock_on_error(eapp):
    from vibetts.export import ExportRequest

    with pytest.raises(ValueError):
        eapp.export_audio(ExportRequest("Hello.", Path("x.ogg")))
    assert not eapp._export_lock.locked() and eapp._export_cancel is None


def test_export_yield_pauses_while_speaking(eapp):
    from vibetts.export import ExportProgress

    eapp.speaker = FakeSpeaker(busy=True)
    seen = []
    threading.Timer(0.25, lambda: setattr(eapp.speaker, "busy", False)).start()
    t0 = time.monotonic()
    eapp._export_yield(ExportProgress(2, 9, 1.5), seen.append, threading.Event())
    assert time.monotonic() - t0 >= 0.2
    assert [p.paused for p in seen] == [True, False]  # reported once, then resumed
    assert seen[0].done == 2 and seen[0].total == 9


def test_export_yield_honors_cancel(eapp):
    from vibetts.export import ExportCancelled, ExportProgress

    eapp.speaker = FakeSpeaker(busy=True)
    cancel = threading.Event()
    threading.Timer(0.15, cancel.set).start()
    with pytest.raises(ExportCancelled):
        eapp._export_yield(ExportProgress(0, 1, 0.0), None, cancel)


def test_export_yield_passes_straight_through_when_idle(eapp):
    from vibetts.export import ExportProgress

    seen = []
    eapp._export_yield(ExportProgress(0, 1, 0.0), seen.append, threading.Event())
    assert seen == []


def test_cancel_export_cancels_a_running_export(eapp, tmp_path):
    from vibetts.export import ExportCancelled, ExportRequest

    eapp.cancel_export()  # nothing running: harmless
    text = " ".join(f"Sentence number {i} is here." for i in range(30))

    def progress(p):
        if p.done == 2:
            eapp.cancel_export()

    with pytest.raises(ExportCancelled):
        eapp.export_audio(ExportRequest(text, Path("c.wav")), progress)
    assert not list((tmp_path / "exports").glob("c.wav*"))  # no file and no .part
    assert not eapp._export_lock.locked()


def test_export_audio_pauses_for_the_speaker(eapp):
    from vibetts.export import ExportRequest

    eapp.speaker = FakeSpeaker(busy=True)
    threading.Timer(0.2, lambda: setattr(eapp.speaker, "busy", False)).start()
    seen = []
    eapp.export_audio(ExportRequest("Hello there.", Path("p.wav")), seen.append)
    assert any(p.paused for p in seen)


# ---------------------------------------------------------------- AgentBridge

@pytest.fixture
def bridge(eapp):
    eapp._voices = lambda: ["af_heart", "bf_emma", "omni:default", "omni:alma"]
    return app.AgentBridge(eapp), eapp


def test_bridge_list_voices(bridge):
    b, a = bridge
    out = b.list_voices()
    assert out["engine"] == "kokoro" and out["default_voice"] == "af_heart"
    assert out["omnivoice_installed"] is True
    assert {"id": "af_heart", "label": "Heart (American English, female)",
            "engine": "kokoro"} in out["voices"]
    assert [v["engine"] for v in out["voices"]][-2:] == ["omnivoice", "omnivoice"]
    alma = out["voices"][-1]
    assert alma == {"id": "omni:alma", "label": alma["label"], "engine": "omnivoice",
                    "kind": "design", "description": "female, young adult", "lang": "en"}


def test_bridge_speak_waits_for_the_real_speaker(bridge):
    from vibetts.tts import Speaker

    b, a = bridge

    class Out:
        def __init__(self, device=""):
            pass

        def write(self, block):
            pass

        def drain(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    a._speak_cfg, a._speak_voice = a.cfg.speech, None
    a.speaker = Speaker(a._engine_for_speaker, Out)
    out = b.speak("Hello there.", None, 1.4, None, True, 5.0)
    assert out == {"started": True, "completed": True, "interrupted": False, "error": None,
                   "voice": "af_heart", "lang": "en"}
    assert a._speak_cfg.speed == 1.4 and a.cfg.speech.speed != 1.4  # a copy, config untouched
    out = b.speak("Hello there.", "bf_emma", None, None, False, 5.0)  # fire and forget
    assert out["started"] and not out["completed"] and out["voice"] == "bf_emma"
    a.speaker.stop(wait=2.0)


def test_bridge_speak_reports_interruption_and_error(bridge, monkeypatch):
    import types

    b, a = bridge
    for interrupted, error in [(True, None), (False, "boom")]:
        req = types.SimpleNamespace(done=threading.Event(), interrupted=interrupted, error=error)
        req.done.set()
        monkeypatch.setattr(a, "speak", lambda *args, req=req: req)
        out = b.speak("Hello there.", None, None, None, True, 1.0)
        assert (out["completed"], out["interrupted"], out["error"]) == (False, interrupted, error)


def test_bridge_speak_wait_times_out(bridge, monkeypatch):
    import types

    b, a = bridge
    req = types.SimpleNamespace(done=threading.Event(), interrupted=False, error=None)
    monkeypatch.setattr(a, "speak", lambda *args: req)
    out = b.speak("Hello there.", None, None, None, True, 0.05)
    assert out["started"] and not out["completed"]


def test_bridge_speak_refusals(bridge, fakes):
    from vibetts.mcp_server import ToolError

    b, a = bridge
    a.enabled = False
    with pytest.raises(ToolError, match="turned off"):
        b.speak("Hello.", None, None, None, False, 1.0)
    a.enabled = True
    with pytest.raises(ToolError, match="Unknown voice 'zz_nobody'.*list_voices"):
        b.speak("Hello.", "zz_nobody", None, None, False, 1.0)
    with pytest.raises(ToolError, match="Nothing to read"):
        b.speak("   ", None, None, None, False, 1.0)
    _, omni = fakes
    omni.present = False
    with pytest.raises(ToolError, match="not installed"):
        b.speak("Hello.", "omni:alma", None, None, False, 1.0)


@pytest.fixture
def designer(bridge, monkeypatch):
    import types

    b, a = bridge
    st = types.SimpleNamespace(designed=[], saved=[], error=None)

    def design(instruct, seed, text=None, lang=None):
        st.designed.append((instruct, seed, text, lang))
        if st.error:
            raise st.error
        return types.SimpleNamespace(meta={"instruct": instruct.strip().lower()})

    def save(name, clip, overwrite=False):
        st.saved.append((name, overwrite))
        return "omni:" + name

    monkeypatch.setattr(a, "_design_clip", design)
    monkeypatch.setattr(a, "_save_voice", save)
    return b, a, st


def test_bridge_create_voice_success(designer):
    b, a, st = designer
    before = (a.cfg.speech.omni_voice, a.cfg.speech.croatian_voice)
    out = b.create_voice({"name": "Nova", "description": " Female ", "seed": 7, "lang": "hr"})
    assert out == {"voice": "omni:nova", "name": "nova", "description": "female",
                   "seed": 7, "lang": "hr"}
    assert st.designed == [(" Female ", 7, None, "hr")] and st.saved == [("nova", False)]
    assert (a.cfg.speech.omni_voice, a.cfg.speech.croatian_voice) == before


def test_bridge_create_voice_random_seed_and_defaults(designer):
    b, a, st = designer
    out = b.create_voice({"name": "rnd"})
    seed = out["seed"]
    assert isinstance(seed, int) and 0 <= seed < 2**31
    assert st.designed == [("", seed, None, "en")] and out["lang"] == "en"


def test_bridge_create_voice_refusals(designer, fakes):
    from vibetts.mcp_server import ToolError

    b, a, st = designer
    with pytest.raises(ToolError, match="already exists.*overwrite=true"):
        b.create_voice({"name": "Alma"})
    assert st.designed == [] and st.saved == []
    b.create_voice({"name": "alma", "overwrite": True})
    assert st.saved == [("alma", True)]
    with pytest.raises(ToolError, match="bad voice name"):
        b.create_voice({"name": "default"})
    st.error = ValueError("'robot' (unsupported)")
    with pytest.raises(ToolError, match="robot"):
        b.create_voice({"name": "fresh", "description": "robot"})
    assert len(st.saved) == 1
    _, omni = fakes
    omni.present = False
    with pytest.raises(ToolError, match="not installed"):
        b.create_voice({"name": "fresh"})


def test_bridge_stop(bridge):
    b, a = bridge
    assert b.stop(False) == {"stopped": True, "export_cancelled": False}
    a._export_cancel = threading.Event()
    assert b.stop(False)["export_cancelled"] is False and not a._export_cancel.is_set()
    assert b.stop(True)["export_cancelled"] is True and a._export_cancel.is_set()


def test_bridge_synthesize_text_and_file(bridge, tmp_path):
    b, a = bridge
    out = b.synthesize_to_file({"text": "Hello there.", "output": "talk.wav"})
    assert out["path"] == str(tmp_path / "exports" / "talk.wav")
    assert out["format"] == "wav" and out["voice"] == "af_heart" and out["lang"] == "en"
    assert out["chunks"] == 1 and out["bytes"] > 0 and out["duration_seconds"] >= 0
    doc = tmp_path / "My Notes.txt"
    doc.write_text("A short note. Nothing more.", encoding="utf-8")
    out = b.synthesize_to_file({"path": str(doc), "voice": "bf_emma", "speed": 1.2})
    assert out["path"] == str(tmp_path / "exports" / "My Notes.wav")  # titled after the file
    assert out["voice"] == "bf_emma"
    with pytest.raises(FileExistsError):
        b.synthesize_to_file({"path": str(doc)})
    b.synthesize_to_file({"path": str(doc), "overwrite": True})


def test_bridge_synthesize_errors(bridge, tmp_path):
    from vibetts.mcp_server import ToolError

    b, a = bridge
    with pytest.raises(ToolError, match="Unknown voice"):
        b.synthesize_to_file({"text": "Hi.", "voice": "zz_nobody"})
    with pytest.raises(ToolError, match="does not match"):
        b.synthesize_to_file({"text": "Hi.", "output": "x.wav", "format": "mp3"})
    with pytest.raises(ToolError):
        b.synthesize_to_file({"path": str(tmp_path / "missing.txt")})
    a.cfg.mcp.max_chars = 5
    big = tmp_path / "big.txt"
    big.write_text("Much longer than five characters.", encoding="utf-8")
    with pytest.raises(ToolError, match="limit"):
        b.synthesize_to_file({"path": str(big)})


def test_bridge_synthesize_busy(bridge):
    from vibetts.export import ExportBusy

    b, a = bridge
    a._export_lock.acquire()
    with pytest.raises(ExportBusy):
        b.synthesize_to_file({"text": "Hi."})


# ---------------------------------------------------------------- MCP lifecycle

class FakeMcpServer:
    instances: list = []
    fail: Exception | None = None

    def __init__(self, tools, port, **kw):
        self.tools, self.port, self.kw = tools, port, kw
        self.started = self.stopped = False
        self.running = False
        FakeMcpServer.instances.append(self)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self):
        if FakeMcpServer.fail:
            raise FakeMcpServer.fail
        self.started = self.running = True

    def stop(self):
        self.stopped, self.running = True, False


@pytest.fixture
def mapp(eapp, monkeypatch):
    FakeMcpServer.instances, FakeMcpServer.fail = [], None
    monkeypatch.setattr("vibetts.mcp_server.McpServer", FakeMcpServer)
    return eapp


def test_sync_mcp_start_noop_restart_stop(mapp):
    a = mapp
    a._sync_mcp()
    assert FakeMcpServer.instances == [] and a.mcp_status()["running"] is False
    a.cfg.mcp.enabled, a.cfg.mcp.port = True, 7800
    a._sync_mcp()
    (first,) = FakeMcpServer.instances
    assert first.started and first.port == 7800 and first.kw["version"]
    assert {t.name for t in first.tools} == {"synthesize_to_file", "speak", "list_voices", "create_voice", "delete_voice",
                                           "synthesize_batch", "stop"}
    assert a.mcp_status() == {"enabled": True, "running": True,
                              "url": "http://127.0.0.1:7800/mcp", "error": ""}
    a._sync_mcp()  # unchanged: nothing happens
    assert len(FakeMcpServer.instances) == 1 and not first.stopped
    a.cfg.mcp.port = 7801
    a._sync_mcp()
    assert first.stopped and len(FakeMcpServer.instances) == 2
    assert FakeMcpServer.instances[1].started and FakeMcpServer.instances[1].port == 7801
    a.cfg.mcp.max_chars = 10
    a._sync_mcp()  # max_chars is baked into the tools: restart
    assert len(FakeMcpServer.instances) == 3
    a.cfg.mcp.enabled = False
    a._sync_mcp()
    assert FakeMcpServer.instances[2].stopped
    assert a.mcp_status() == {"enabled": False, "running": False,
                              "url": "http://127.0.0.1:7801/mcp", "error": ""}


def test_sync_mcp_port_busy(mapp):
    a = mapp
    a.cfg.mcp.enabled = True
    FakeMcpServer.fail = OSError("address in use")
    a._sync_mcp()
    st = a.mcp_status()
    assert st["running"] is False and "7723 is busy" in st["error"]
    assert a.flashes and a.flashes[-1][0] == "error"
    FakeMcpServer.fail = None
    a._sync_mcp()  # retried on the next apply: the error clears
    assert a.mcp_status()["running"] is True and a.mcp_status()["error"] == ""
    a.cfg.mcp.enabled = False
    a._sync_mcp()
    assert a.mcp_status()["error"] == ""


def test_apply_config_syncs_mcp_even_when_the_voice_is_missing(mapp, tmp_path):
    a = mapp
    a.listener = type("L", (), {"set_bindings": lambda self, b: None})()
    new = Config(speech=dataclasses.replace(a.cfg.speech, model=str(tmp_path / "nope.onnx")))
    new.mcp.enabled = True
    msg = a.apply_config(new)
    assert msg == app.MISSING_MODEL
    assert len(FakeMcpServer.instances) == 1 and FakeMcpServer.instances[0].started


def test_quit_stops_mcp_and_cancels_export(mapp):
    a = mapp
    a.cfg.mcp.enabled = True
    a._sync_mcp()
    server = FakeMcpServer.instances[0]
    cancel = threading.Event()
    a._export_cancel = cancel
    a._quit_lock, a._quit_done, a._instance = threading.Lock(), False, None
    noop = type("N", (), {"stop": lambda self, *x: None, "quit": lambda self: None})()
    a.speaker, a.listener, a.tray, a.overlay = noop, noop, noop, noop
    a.quit()
    assert server.stopped and cancel.is_set()


def test_omni_uninstall_cancels_export(vapp, monkeypatch):
    import types

    a, _ = vapp
    a._export_lock = threading.Lock()
    a._export_cancel = threading.Event()
    a.speaker = FakeSpeaker()
    fake = types.ModuleType("vibetts.omnivoice_setup")
    fake.uninstall = lambda d, keep_voices=True: None
    monkeypatch.setattr(app.TTSApp, "_setup", staticmethod(lambda: fake))
    a._omni_uninstall()
    assert a._export_cancel.is_set()


# ---------------------------------------------------------------- hooks / tray

def test_settings_hooks_include_export_and_mcp(mapp, monkeypatch, tmp_path):
    a = mapp
    a._autostart = type("A", (), {"is_enabled": staticmethod(lambda: False)})
    monkeypatch.setattr("vibetts.settings_ui.SettingsWindow", FakeSettingsWindow)
    hooks = a._settings_window().hooks
    assert hooks.export_audio == a.export_audio and hooks.cancel_export == a.cancel_export
    assert hooks.mcp_status == a.mcp_status and hooks.open_folder == a._open_folder
    assert hooks.mp3_available() in (True, False)
    doc = tmp_path / "d.md"
    doc.write_text("# Title\n\nSome text.", encoding="utf-8")
    loaded = hooks.load_document(str(doc))
    assert loaded.title == "d" and loaded.markdown is True and "Some text" in loaded.text
    from vibetts.docload import DocumentError

    with pytest.raises(DocumentError):
        hooks.load_document(str(tmp_path / "missing.txt"))


def test_open_folder_creates_and_opens(eapp, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(app.os, "startfile", opened.append, raising=False)
    eapp._open_folder(tmp_path / "new" / "dir")
    assert (tmp_path / "new" / "dir").is_dir() and opened == [str(tmp_path / "new" / "dir")]


def test_tray_menu_has_export_item_after_settings():
    from vibetts.tray import Tray, TrayHooks

    calls = []
    noop = lambda *a: None  # noqa: E731
    hooks = TrayHooks(
        is_enabled=lambda: True, set_enabled=noop, is_speaking=lambda: False, stop=noop,
        model_missing=lambda: False, download_model=noop, voices=lambda: [],
        current_voice=lambda: "af_heart", set_voice=noop, current_speed=lambda: 1.0,
        set_speed=noop, is_autostart=lambda: False, set_autostart=noop, open_settings=noop,
        open_config=noop, open_logs=noop, reload_config=noop, quit=noop,
        open_export=lambda: calls.append("export"))
    tray = Tray(hooks)
    items = tray._menu_items()
    texts = [getattr(i, "text", "") for i in items]
    assert texts.index("Export audio…") == texts.index("Settings…") + 1
    items[texts.index("Export audio…")](tray._icon)
    assert calls == ["export"]
    assert "Export audio…" in [getattr(i, "text", "") for i in make_tray()[0]._menu_items()]


# ---------------------------------------------------------------- CLI

@pytest.fixture
def cli(fakes, tmp_path, monkeypatch):
    s, _ = fakes
    cfg = Config(speech=dataclasses.replace(s, sentence_pause=0.0))
    monkeypatch.setattr(app, "setup_logging", lambda console=False: tmp_path / "log.txt")
    monkeypatch.setattr(app, "load_config", lambda: cfg)
    return cfg


@pytest.mark.parametrize("flag", ["--out", "--wav"])
def test_cli_say_out_and_wav_alias(cli, tmp_path, flag, capsys):
    out = tmp_path / "say.wav"
    assert app.main(["--say", "Hello there.", flag, str(out)]) == 0
    assert out.exists() and _wav_frames(out) > 0
    printed = capsys.readouterr().out
    assert "model loaded in" in printed and f"wrote {out}" in printed and "real time" in printed


def test_cli_out_mp3_by_extension(cli, tmp_path):
    pytest.importorskip("lameenc")
    out = tmp_path / "say.mp3"
    assert app.main(["--say", "Hello there.", "--out", str(out)]) == 0
    assert out.read_bytes()[:4] != b"RIFF" and out.stat().st_size > 0


def test_cli_out_without_extension_uses_configured_format(cli, tmp_path):
    cli.export.format = "wav"
    assert app.main(["--say", "Hello there.", "--out", str(tmp_path / "plain")]) == 0
    assert (tmp_path / "plain.wav").exists()


def test_cli_out_overwrites_like_before(cli, tmp_path):
    out = tmp_path / "again.wav"
    for _ in range(2):
        assert app.main(["--say", "Hello there.", "--out", str(out)]) == 0


def test_cli_doc_to_file(cli, tmp_path, capsys):
    doc = tmp_path / "story.txt"
    doc.write_text("Once upon a time. The end.", encoding="utf-8")
    out = tmp_path / "story.wav"
    assert app.main(["--doc", str(doc), "--out", str(out)]) == 0
    assert out.exists() and "wrote" in capsys.readouterr().out


def test_cli_doc_errors(cli, tmp_path, capsys):
    assert app.main(["--doc", str(tmp_path / "missing.txt"), "--out", str(tmp_path / "o.wav")]) == 1
    assert "error:" in capsys.readouterr().err
    bad = tmp_path / "x.xyz"
    bad.write_text("hi", encoding="utf-8")
    assert app.main(["--doc", str(bad)]) == 1


def test_cli_argument_conflicts(cli, tmp_path):
    with pytest.raises(SystemExit):
        app.main(["--say", "a", "--doc", "b"])
    with pytest.raises(SystemExit):
        app.main(["--out", str(tmp_path / "o.wav")])


def test_say_never_prints_the_text(cli, tmp_path, capsys):
    secret = "Confidential sentence number one."
    app.say(cli, secret, str(tmp_path / "s.wav"))
    captured = capsys.readouterr()
    assert "Confidential" not in captured.out + captured.err


# ---------------------------------------------------------------- batch, delete_voice, headless

def test_bridge_synthesize_batch(bridge, tmp_path):
    b, a = bridge
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    items = [
        {"text": "Hello there.", "output": str(out_dir / "a.wav")},
        {"text": "Second line.", "output": str(out_dir / "b.wav"), "voice": "bf_emma", "speed": 1.2},
        {"text": "", "output": str(out_dir / "c.wav")},                      # empty text
        {"text": "Rel.", "output": "rel.wav"},                               # not absolute
        {"text": "Bad.", "output": str(out_dir / "d.ogg")},                  # bad extension
        {"text": "Hi.", "output": str(out_dir / "e.wav"), "lang": "de"},    # bad lang
        {"text": "Hi.", "output": str(out_dir / "f.wav"), "voice": "zz_nobody"},
        {"text": "Hi.", "output": str(tmp_path / "nodir" / "g.wav")},        # folder missing
    ]
    out = b.synthesize_batch({"items": items})
    r = out["results"]
    assert [x["ok"] for x in r] == [True, True, False, False, False, False, False, False]
    assert r[0]["output"] == str(out_dir / "a.wav") and r[0]["duration_s"] >= 0
    assert (out_dir / "a.wav").exists() and (out_dir / "b.wav").exists()
    assert "non-empty" in r[2]["error"] and "absolute" in r[3]["error"]
    assert ".wav and .mp3" in r[4]["error"] and "'lang'" in r[5]["error"]
    assert "Unknown voice" in r[6]["error"] and "does not exist" in r[7]["error"]
    assert out["ok_count"] == 2 and out["failed_count"] == 6
    assert not a._export_lock.locked() and a._export_cancel is None
    # existing files fail unless overwrite is set
    again = b.synthesize_batch({"items": items[:1]})
    assert again["results"][0]["ok"] is False and "already exists" in again["results"][0]["error"]
    assert b.synthesize_batch({"items": items[:1], "overwrite": True})["ok_count"] == 1


def test_bridge_synthesize_batch_text_limit_and_busy(bridge, tmp_path):
    from vibetts.mcp_server import ToolError
    from vibetts.export import ExportBusy

    b, a = bridge
    a.cfg.mcp.max_chars = 10
    items = [{"text": "x" * 11, "output": str(tmp_path / "long.wav")},
             {"text": "short", "output": str(tmp_path / "ok.wav")}]
    r = b.synthesize_batch({"items": items})["results"]
    assert "too long" in r[0]["error"] and r[1]["ok"]
    a._export_lock.acquire()
    with pytest.raises(ExportBusy):
        b.synthesize_batch({"items": [{"text": "hi", "output": str(tmp_path / "z.wav")}]})
    assert a._export_lock.locked()


def test_export_batch_cancel_marks_rest(eapp, tmp_path):
    from vibetts.export import ExportRequest

    cancel = threading.Event()
    reqs = [ExportRequest("One.", tmp_path / "1.wav", overwrite=True),
            ExportRequest("Two.", tmp_path / "2.wav", overwrite=True)]
    seen = []

    def progress(done, total):
        seen.append((done, total))
        cancel.set()

    rows = eapp.export_batch(reqs, progress, cancel)
    assert rows[0]["ok"] and rows[1] == {"output": str(tmp_path / "2.wav"), "duration_s": 0.0,
                                         "ok": False, "error": "cancelled"}
    assert seen == [(1, 2), (2, 2)] and not eapp._export_lock.locked()


def test_bridge_delete_voice(bridge, fakes):
    from vibetts.mcp_server import ToolError

    b, a = bridge
    _, omni = fakes
    deleted = []

    def delete_preset(model_dir, name):
        if name == "default":
            raise ValueError("The 'default' voice cannot be deleted.")
        if name == "ghost":
            raise FileNotFoundError("No voice named 'ghost'.")
        deleted.append(name)

    omni.delete_preset = delete_preset
    assert b.delete_voice({"name": "alma"}) == {"deleted": "omni:alma"}
    assert b.delete_voice({"name": "omni:alma"}) == {"deleted": "omni:alma"}
    assert deleted == ["alma", "alma"]
    with pytest.raises(ToolError, match="cannot be deleted"):
        b.delete_voice({"name": "default"})
    with pytest.raises(ToolError, match="No voice named"):
        b.delete_voice({"name": "ghost"})
    omni.present = False
    with pytest.raises(ToolError, match="not installed"):
        b.delete_voice({"name": "alma"})


def test_headless_app_serves_exports_and_refuses_speech(fakes, tmp_path):
    s, _ = fakes
    cfg = Config(speech=dataclasses.replace(s, sentence_pause=0.0))
    cfg.export.folder = str(tmp_path / "exports")
    cfg.export.format = "wav"
    h = app.HeadlessApp(cfg)
    h._voices = lambda: ["af_heart", "omni:default"]
    b = app.AgentBridge(h)
    assert b.list_voices()["voices"][0]["id"] == "af_heart"
    out = b.synthesize_batch({"items": [{"text": "Hello there.", "output": str(tmp_path / "h.wav")}]})
    assert out["ok_count"] == 1
    with pytest.raises(RuntimeError, match="headless"):
        b.speak("Hello.", None, None, None, False, 1.0)
    assert b.stop(False) == {"stopped": True, "export_cancelled": False}


def test_cli_mcp_only_forces_mcp_on_and_uses_port(cli, monkeypatch):
    seen = {}

    def fake_run(cfg, port=None):
        seen["cfg"], seen["port"] = cfg, port
        return 0

    monkeypatch.setattr(app, "run_mcp_only", fake_run)
    assert app.main(["--mcp-only", "--port", "7799"]) == 0
    assert seen["port"] == 7799 and seen["cfg"] is cli


def test_cli_mcp_only_argument_conflicts(cli):
    with pytest.raises(SystemExit):
        app.main(["--port", "7799"])
    with pytest.raises(SystemExit):
        app.main(["--mcp-only", "--say", "x"])


def test_run_mcp_only_serves_until_interrupted(cli, monkeypatch, capsys):
    import vibetts.mcp_server as ms

    started = {}

    class FakeServer:
        port, url, running = 7701, "http://127.0.0.1:7701/mcp", True

        def __init__(self, tools, port, **kw):
            started["names"] = [t.name for t in tools]
            started["port"] = port

        def start(self):
            started["started"] = True

        def stop(self):
            started["stopped"] = True

    monkeypatch.setattr(ms, "McpServer", FakeServer)

    class Boom(threading.Event):
        def wait(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(app.threading, "Event", Boom)
    cli.mcp.enabled = False
    assert app.run_mcp_only(cli, 7701) == 0
    assert cli.mcp.enabled is True and started["port"] == 7701
    assert "synthesize_batch" in started["names"] and started["stopped"]
    assert "7701" in capsys.readouterr().out
    assert app.run_mcp_only(cli, 80) == 2
