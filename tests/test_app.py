import dataclasses
import re
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from vibetts import app
from vibetts.config import ALL_LANGUAGES, Config, SpeechConfig


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


# ---------------------------------------------------------------- engine

class FakeKokoro:
    def __init__(self, *paths):
        self.paths, self.calls, self.langs = paths, [], []

    def voices(self):
        return ["af_heart"]

    def synth(self, text, voice, speed, lang=None):
        self.calls.append((text, voice, speed))
        self.langs.append(lang)
        return np.ones(10, np.float32)


class FakeSupertonic:
    instances: list = []

    def __init__(self, model_dir, steps=8):
        self.model_dir, self.steps, self.calls, self.langs = model_dir, steps, [], []
        FakeSupertonic.instances.append(self)

    def voices(self):
        return app.supertonic_voice_ids()

    def synth(self, text, voice, speed, lang=None):
        self.calls.append((text, voice, speed))
        self.langs.append(lang)
        return np.ones(20, np.float32)


class FakeHiggs:
    instances: list = []

    def __init__(self, model_dir, voices_dir, temperature=0.8, threads=0):
        self.model_dir, self.voices_dir, self.temperature = model_dir, voices_dir, temperature
        self.calls, self.langs = [], []
        FakeHiggs.instances.append(self)

    def voices(self):
        return fake_higgs_voice_ids(self.voices_dir)

    def synth(self, text, voice, speed, lang=None):
        self.calls.append((text, voice, speed))
        self.langs.append(lang)
        return np.ones(30, np.float32)


def fake_higgs_voice_ids(voices_dir):
    custom = sorted(p.name for p in Path(voices_dir).glob("*") if p.is_dir())
    return [f"higgs:{n}" for n in ("F1", "F2", "M1", "M2", *custom)]


def install_higgs(s):
    """Make the fake Higgs 'downloaded' for the SpeechConfig `s`."""
    d = Path(s.higgs_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ok").write_bytes(b"x")


def install_supertonic(s):
    """Make the fake Supertonic 'downloaded' for the SpeechConfig `s`."""
    d = Path(s.supertonic_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ok").write_bytes(b"x")


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """A SpeechConfig whose Kokoro files exist; Supertonic is faked and not installed yet
    (install_supertonic(s) installs it). Nothing here touches the real engines or models."""
    monkeypatch.setattr("vibetts.tts.KokoroEngine", FakeKokoro)
    FakeSupertonic.instances = []
    fake = types.SimpleNamespace(models_present=lambda d: (Path(d) / "ok").exists(),
                                 SupertonicEngine=FakeSupertonic)
    monkeypatch.setattr(app, "_st", lambda: fake)
    FakeHiggs.instances = []
    fake_hg = types.SimpleNamespace(models_present=lambda d: (Path(d) / "ok").exists(),
                                    voice_ids=fake_higgs_voice_ids, HiggsEngine=FakeHiggs)
    monkeypatch.setattr(app, "_hg", lambda: fake_hg)
    monkeypatch.setattr(app, "_fallback_warned", False)
    monkeypatch.setattr(app, "_fallback_warned_higgs", False)
    model, voices = tmp_path / "m.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return SpeechConfig(model=str(model), voices=str(voices),
                        supertonic_dir=str(tmp_path / "st"), higgs_dir=str(tmp_path / "hg"),
                        higgs_voices_dir=str(tmp_path / "hgv"))


def test_holder_loads_once_and_reloads_on_new_files(fakes, tmp_path):
    s = fakes
    holder = app.EngineHolder()
    assert holder.loaded is None
    engine = holder.get(s).kokoro
    assert isinstance(engine, FakeKokoro) and holder.loaded.kokoro is engine
    assert engine.calls == [("Ready.", "af_heart", 1.0)]  # warm-up
    assert holder.get(s).kokoro is engine  # cached, not reloaded
    other = tmp_path / "m2.onnx"
    other.write_bytes(b"x")
    assert holder.get(dataclasses.replace(s, model=str(other))).kokoro is not engine


def test_preload_loads_in_the_background(fakes):
    holder = app.EngineHolder()
    holder.preload(fakes)
    for _ in range(100):
        if holder.loaded is not None:
            break
        time.sleep(0.02)
    assert isinstance(holder.loaded.kokoro, FakeKokoro) and holder.loaded.supertonic is None


EN_TEXT = "This is plain English text."


class SpyKokoro(FakeKokoro):
    seen: list = []

    def synth(self, text, voice, speed, lang=None):
        SpyKokoro.seen.append((text, voice))
        return super().synth(text, voice, speed, lang)


def test_say_reads_with_the_configured_voice(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr("vibetts.tts.KokoroEngine", SpyKokoro)
    SpyKokoro.seen = []
    cfg = Config(speech=dataclasses.replace(
        fakes, language_voices={**fakes.language_voices, "en": "bf_emma"}))
    assert app.say(cfg, "Read https://x.org/a now.", str(tmp_path / "o.wav")) == 0
    assert SpyKokoro.seen[-1] == ("Read link now.", "bf_emma")


def test_say_missing_model(fakes, tmp_path, capsys):
    s = dataclasses.replace(fakes, model=str(tmp_path / "nope.onnx"))
    assert app.say(Config(speech=s), EN_TEXT, str(tmp_path / "o.wav")) == 1
    assert "model files missing" in capsys.readouterr().err


class FakeOverlayShow:
    def show(self, *a, **k):
        pass


def test_speak_passes_the_voice_to_the_engine(fakes):
    from vibetts.textprep import Chunk

    s = fakes
    a = object.__new__(app.TTSApp)
    a._current = None
    a.cfg = Config(speech=s)
    a.engine = app.EngineHolder()
    a.overlay = FakeOverlayShow()
    reqs = []
    a.speaker = type("Sp", (), {"speak": lambda self, req: reqs.append(req)})()
    a.speak([Chunk("Hi", 0.0)], s, "bf_emma")
    a.speak([Chunk("Hi", 0.0)], s)
    a.speak([Chunk("Hi", 0.0)], s, "supertonic:F2", "hr")
    assert [r.voice for r in reqs] == ["bf_emma", "af_heart", "supertonic:F2"]  # default: the Kokoro English voice
    assert [r.lang for r in reqs] == [None, None, "hr"]
    install_supertonic(s)
    engine = a._engine_for_speaker()  # loads the engine of the request being spoken
    assert isinstance(engine.supertonic, FakeSupertonic) and engine.kokoro is None
    a.speak([Chunk("Hi", 0.0)], s)
    assert isinstance(a._engine_for_speaker().kokoro, FakeKokoro)


# ---------------------------------------------------------------- app helpers

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
    a._current = None
    a._export_lock = threading.Lock()
    a._export_cancel = None
    a._mcp, a._mcp_key, a._mcp_error = None, None, None
    a._reader, a._reader_bridge, a._reader_key, a._reader_error = None, None, None, None
    a.cfg.reader.enabled = False  # tests that need the reader turn it on (with a fake server)
    a.enabled = True
    a.flashes = []
    a._flash = lambda kind, text, seconds=1.5: a.flashes.append((kind, text))
    saved = []
    a._save = lambda: saved.append(dataclasses.replace(a.cfg.speech))
    a.saved_configs = saved
    return a


def test_tray_set_voice(fakes):
    a = make_app(fakes)
    install_supertonic(a.cfg.speech)
    a.cfg.speech.preload = False
    a._set_voice("en", "bf_emma")
    assert a.cfg.speech.language_voices["en"] == "bf_emma"
    assert len(a.saved_configs) == 1 and a.tray.updates == 1
    a._set_voice("en", "supertonic:M2")
    assert a.cfg.speech.language_voices["en"] == "supertonic:M2"
    a._set_voice("es", "ef_dora")  # a Kokoro voice of that language
    assert a.cfg.speech.language_voices["es"] == "ef_dora"
    a._set_voice("hr", "supertonic:M3")
    assert a.cfg.speech.language_voices["hr"] == "supertonic:M3"
    before = dict(a.cfg.speech.language_voices)
    a._set_voice("hr", "af_heart")  # a Kokoro voice of another language: ignored
    a._set_voice("es", "bf_emma")
    a._set_voice("xx", "supertonic:F1")  # unknown language
    a._set_voice("hr", "bogus")
    assert a.cfg.speech.language_voices == before and len(a.saved_configs) == 4


def test_tray_set_voice_off_and_read_language(fakes):
    a = make_app(fakes)
    a.cfg.speech.preload = False
    a._set_read_language("de")
    assert a.cfg.speech.read_language == "de"
    a._set_read_language("bogus")
    assert a.cfg.speech.read_language == "de"
    a._set_voice("de", "")  # Off while read_language is "de": back to auto
    assert a.cfg.speech.language_voices["de"] == "" and a.cfg.speech.read_language == "auto"
    a._set_read_language("de")  # a language that is Off cannot be picked
    assert a.cfg.speech.read_language == "auto"
    a._set_read_language("hr")
    a._set_voice("es", "")  # another language Off: read_language stays
    assert a.cfg.speech.read_language == "hr"
    a._set_read_language("auto")
    assert a.cfg.speech.read_language == "auto"
    assert len(a.saved_configs) == 5 and a.tray.updates == 5


def test_tray_cannot_turn_off_the_last_language(fakes):
    a = make_app(fakes)
    a.cfg.speech.preload = False
    for code in app.SUPERTONIC_LANGUAGES:
        if code != "en":
            a._set_voice(code, "")
    assert app.enabled_languages(a.cfg.speech) == ["en"]
    a._set_voice("en", "")
    assert a.cfg.speech.language_voices["en"] == "af_heart"


def test_tray_hooks_are_wired_to_the_config(fakes, monkeypatch):
    seen = {}

    class T:
        def __init__(self, hooks):
            seen["hooks"] = hooks

    monkeypatch.setattr("vibetts.tray.Tray", T)
    monkeypatch.setattr("vibetts.overlay.Overlay", lambda ui: FakeOverlay())
    monkeypatch.setattr("vibetts.reading.ReadingSession", lambda a: None)
    monkeypatch.setattr("vibetts.hotkeys.HotkeyListener", lambda *a, **k: None)
    monkeypatch.setattr("vibetts.tts.Speaker", lambda *a, **k: None)
    a = app.TTSApp(Config(speech=dataclasses.replace(
        fakes, language_voices={**fakes.language_voices, "de": "", "es": "ef_dora"})))
    h = seen["hooks"]
    assert "de" not in h.languages() and h.languages()[:2] == ["en", "ko"]
    assert h.current_voice("es") == "ef_dora" and h.current_voice("en") == "af_heart"
    assert h.current_voice("hr") == "supertonic:F1"
    assert h.read_language() == "auto"
    a._save = lambda: None
    a.tray = FakeTray()
    a.cfg.speech.preload = False
    h.set_read_language("es")
    assert h.read_language() == "es"


# ---------------------------------------------------------------- tray menus

def make_tray(voices=None, current=None, languages=("en", "es", "hr"), read="auto"):
    from vibetts.tray import Tray, TrayHooks

    calls = []
    noop = lambda *a: None  # noqa: E731
    current = current if current is not None else {"en": "af_heart", "es": "ef_dora",
                                                   "hr": "supertonic:M1"}
    hooks = TrayHooks(
        is_enabled=lambda: True, set_enabled=noop, is_speaking=lambda: False, stop=noop,
        model_missing=lambda: False, download_model=noop,
        voices=lambda: voices if voices is not None else [
            "af_heart", "am_adam", "bf_emma", "ef_dora", "em_alex", "pf_dora"],
        languages=lambda: list(languages),
        current_voice=lambda lang: current.get(lang, ""),
        set_voice=lambda lang, v: calls.append(("voice", lang, v)),
        read_language=lambda: read,
        set_read_language=lambda c: calls.append(("read", c)),
        current_speed=lambda: 1.0, set_speed=noop, is_autostart=lambda: False,
        set_autostart=noop, open_settings=noop, open_config=noop, open_logs=noop,
        reload_config=noop, quit=noop)
    return Tray(hooks), calls


def _submenus(tray):
    return {i.text: i for i in tray._menu_items() if getattr(i, "submenu", None)}


def test_tray_voices_menu_has_a_submenu_per_enabled_language():
    tray, calls = make_tray()
    langs = {g.text: [i.text for i in g.submenu.items] for g in tray._voice_menu().items}
    assert list(langs) == ["English", "Spanish", "Croatian"]
    assert langs["English"] == ["Off", "Heart (female)", "Adam (male)", "Emma (female)"]
    assert langs["Spanish"] == ["Off", "Dora (female)", "Alex (male)"]
    assert langs["Croatian"] == ["Off"]  # Supertonic not installed: only Off
    es = tray._voice_menu().items[1]
    assert [i.checked for i in es.submenu.items] == [False, True, False]
    es.submenu.items[2](tray._icon)  # picking a voice goes through set_voice(lang, voice)
    es.submenu.items[0](tray._icon)  # Off is the empty voice
    assert calls == [("voice", "es", "em_alex"), ("voice", "es", "")]


def test_tray_voices_menu_lists_supertonic_for_every_language():
    tray, calls = make_tray(voices=["af_heart", "ef_dora", *app.supertonic_voice_ids()])
    langs = {g.text: [i.text for i in g.submenu.items] for g in tray._voice_menu().items}
    assert langs["Croatian"] == ["Off", *[f"{v[11:]} ({'female' if v[11] == 'F' else 'male'})"
                                         for v in app.supertonic_voice_ids()]]
    assert langs["English"][:2] == ["Off", "Heart (female)"] and len(langs["English"]) == 12
    hr = tray._voice_menu().items[2]
    assert [i.checked for i in hr.submenu.items][1 + 5] is True  # M1
    hr.submenu.items[1](tray._icon)
    assert calls == [("voice", "hr", "supertonic:F1")]


def test_tray_menu_has_voice_entry():
    tray, _ = make_tray()
    texts = [getattr(i, "text", "") for i in tray._menu_items()]
    assert "Voices" in texts and "Language" in texts and "Speed" in texts
    assert "English voice" not in texts and "Croatian voice" not in texts
    tray, _ = make_tray(languages=())
    texts = [getattr(i, "text", "") for i in tray._menu_items()]
    assert "Voices" not in texts and "Language" in texts


def test_tray_language_menu():
    tray, calls = make_tray(read="es")
    lang = _submenus(tray)["Language"].submenu.items
    assert [i.text for i in lang] == ["Auto (detect language)", "Always English",
                                      "Always Spanish", "Always Croatian"]
    assert [i.checked for i in lang] == [False, False, True, False]
    lang[3](tray._icon)
    lang[0](tray._icon)
    assert calls == [("read", "hr"), ("read", "auto")]
    tray, _ = make_tray()
    assert [i.checked for i in _submenus(tray)["Language"].submenu.items][0] is True


# ---------------------------------------------------------------- plan_speech / run_export

def _wav_frames(path):
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes()


def test_plan_speech_voice(fakes):
    s = fakes
    chunks, route = app.plan_speech(s, EN_TEXT)
    assert route == app.Route("kokoro", "af_heart", "en") and chunks
    assert app.plan_speech(s, EN_TEXT, voice="bf_emma")[1].voice == "bf_emma"


def test_plan_speech_max_chars_and_markdown(fakes):
    s = fakes
    text = "First sentence here. Second sentence here. Third one is here too."
    assert len(app.plan_speech(s, text, voice="af_heart", max_chars=25)[0]) \
        < len(app.plan_speech(s, text, voice="af_heart")[0])
    chunks = app.plan_speech(s, "**bold** word", voice="af_heart", markdown=True)[0]
    assert "*" not in chunks[0].text


def test_run_export_writes_wav(fakes, tmp_path):
    from vibetts.export import ExportProgress, ExportRequest

    s = dataclasses.replace(fakes, sentence_pause=0.0)
    kokoro = FakeKokoro()
    progress = []
    req = ExportRequest("One sentence. Another sentence.", Path("a.wav"), speed=1.3)
    res = app.run_export(kokoro, s, req, tmp_path / "out", progress.append)
    assert res.path == tmp_path / "out" / "a.wav" and res.fmt == "wav"
    assert res.chunks == len(kokoro.calls) and _wav_frames(res.path) == 10 * res.chunks
    assert kokoro.calls[0][1:3] == ("af_heart", 1.3) and res.voice == "af_heart"
    assert res.lang == "en" and kokoro.langs[0] == "en"
    assert isinstance(progress[-1], ExportProgress) and progress[-1].done == res.chunks


def test_run_export_default_format_and_errors(fakes, tmp_path):
    from vibetts.export import ExportRequest

    s = fakes
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
    missing = dataclasses.replace(s, model=str(tmp_path / "nope.onnx"))
    with pytest.raises(RuntimeError, match="Download the voice model"):
        app.run_export(FakeKokoro(), missing, ExportRequest("Hi."), tmp_path)
    with pytest.raises(ValueError, match="Nothing to read"):
        app.run_export(FakeKokoro(), s, ExportRequest("  "), tmp_path)


def test_run_export_mp3(fakes, tmp_path):
    pytest.importorskip("lameenc")
    from vibetts.export import ExportRequest

    s = fakes
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
    a = make_app(dataclasses.replace(fakes, sentence_pause=0.0))
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
    eapp._voices = lambda: ["af_heart", "bf_emma"]
    return app.AgentBridge(eapp), eapp


def test_bridge_list_voices(bridge):
    b, a = bridge
    out = b.list_voices()
    assert out["voices"] == [
        {"id": "af_heart", "label": "Heart (American English, female)", "engine": "kokoro",
         "languages": ["en"]},
        {"id": "bf_emma", "label": "Emma (British English, female)", "engine": "kokoro",
         "languages": ["en"]}]
    assert [l["code"] for l in out["languages"]][:3] == ["en", "ko", "ja"]
    assert [l["code"] for l in out["languages"]] == list(ALL_LANGUAGES) and len(out["languages"]) == 101
    assert {"code": "hr", "name": "Croatian"} in out["languages"]
    assert {"code": "bs", "name": "Bosnian"} in out["languages"]
    assert out["higgs_installed"] is False
    assert out["language_voices"] == Config().speech.language_voices
    assert out["language_voices"]["en"] == "af_heart" and out["language_voices"]["hr"] == "supertonic:F1"
    assert out["read_language"] == "auto" and out["supertonic_installed"] is False
    assert not {"english", "other", "language_mode"} & set(out)


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

    a._speak_cfg = a.cfg.speech
    a.speaker = Speaker(a._engine_for_speaker, Out)
    out = b.speak("Hello there.", None, 1.4, True, 5.0)
    assert out == {"started": True, "completed": True, "interrupted": False, "error": None,
                   "voice": "af_heart", "lang": "en"}
    assert a._speak_cfg.speed == 1.4 and a.cfg.speech.speed != 1.4  # a copy, config untouched
    out = b.speak("Hello there.", "bf_emma", None, False, 5.0)  # fire and forget
    assert out["started"] and not out["completed"] and out["voice"] == "bf_emma"
    a.speaker.stop(wait=2.0)


def test_bridge_speak_reports_interruption_and_error(bridge, monkeypatch):
    import types

    b, a = bridge
    for interrupted, error in [(True, None), (False, "boom")]:
        req = types.SimpleNamespace(done=threading.Event(), interrupted=interrupted, error=error)
        req.done.set()
        monkeypatch.setattr(a, "speak", lambda *args, req=req: req)
        out = b.speak("Hello there.", None, None, True, 1.0)
        assert (out["completed"], out["interrupted"], out["error"]) == (False, interrupted, error)


def test_bridge_speak_wait_times_out(bridge, monkeypatch):
    import types

    b, a = bridge
    req = types.SimpleNamespace(done=threading.Event(), interrupted=False, error=None)
    monkeypatch.setattr(a, "speak", lambda *args: req)
    out = b.speak("Hello there.", None, None, True, 0.05)
    assert out["started"] and not out["completed"]


def test_bridge_speak_refusals(bridge, tmp_path):
    from vibetts.mcp_server import ToolError

    b, a = bridge
    a.enabled = False
    with pytest.raises(ToolError, match="turned off"):
        b.speak("Hello.", None, None, False, 1.0)
    a.enabled = True
    with pytest.raises(ToolError, match="Unknown voice 'zz_nobody'.*list_voices"):
        b.speak("Hello.", "zz_nobody", None, False, 1.0)
    with pytest.raises(ToolError, match="Nothing to read"):
        b.speak("   ", None, None, False, 1.0)
    a.cfg.speech.model = str(tmp_path / "nope.onnx")
    with pytest.raises(ToolError, match="Download the voice model"):
        b.speak("Hello.", "af_heart", None, False, 1.0)


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
    assert {t.name for t in first.tools} == {"synthesize_to_file", "speak", "list_voices",
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
        current_voice=lambda lang: "af_heart", set_voice=noop, current_speed=lambda: 1.0,
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
    cfg = Config(speech=dataclasses.replace(fakes, sentence_pause=0.0))
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


# ---------------------------------------------------------------- batch, headless

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
        {"text": "Hi.", "output": str(out_dir / "f.wav"), "voice": "zz_nobody"},
        {"text": "Hi.", "output": str(tmp_path / "nodir" / "g.wav")},        # folder missing
    ]
    out = b.synthesize_batch({"items": items})
    r = out["results"]
    assert [x["ok"] for x in r] == [True, True, False, False, False, False, False]
    assert r[0]["output"] == str(out_dir / "a.wav") and r[0]["duration_s"] >= 0
    assert (out_dir / "a.wav").exists() and (out_dir / "b.wav").exists()
    assert "non-empty" in r[2]["error"] and "absolute" in r[3]["error"]
    assert ".wav and .mp3" in r[4]["error"]
    assert "Unknown voice" in r[5]["error"] and "does not exist" in r[6]["error"]
    assert out["ok_count"] == 2 and out["failed_count"] == 5
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


def test_headless_app_serves_exports_and_refuses_speech(fakes, tmp_path):
    cfg = Config(speech=dataclasses.replace(fakes, sentence_pause=0.0))
    cfg.export.folder = str(tmp_path / "exports")
    cfg.export.format = "wav"
    h = app.HeadlessApp(cfg)
    h._voices = lambda: ["af_heart"]
    b = app.AgentBridge(h)
    assert b.list_voices()["voices"][0]["id"] == "af_heart"
    out = b.synthesize_batch({"items": [{"text": "Hello there.", "output": str(tmp_path / "h.wav")}]})
    assert out["ok_count"] == 1
    with pytest.raises(RuntimeError, match="headless"):
        b.speak("Hello.", None, None, False, 1.0)
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


# ---------------------------------------------------------------- language routing

HR_TEXT = "Ovo je jednostavan test sustava koji čita tekst na hrvatskom jeziku, i to dobro."
EN_LONG = "This is a simple test of the system that reads the text out loud in English."


DE_TEXT = "Das ist ein einfacher Test des Systems, der den deutschen Text vorliest, weil es schön ist."
ES_TEXT = "Este es un sencillo test del sistema que lee el texto en español para todos los usuarios."


def _speech(s, read_language="auto", **voices):
    """`s` with some language voices replaced (a fresh dict: replace() would share it)."""
    return dataclasses.replace(s, language_voices={**s.language_voices, **voices},
                               read_language=read_language)


def test_route_per_language(fakes):
    s = _speech(fakes, es="ef_dora", de="supertonic:M1", hr="supertonic:F2")
    install_supertonic(s)
    assert app.route_for(s, EN_LONG) == app.Route("kokoro", "af_heart", "en")
    assert app.route_for(s, ES_TEXT) == app.Route("kokoro", "ef_dora", "es")
    assert app.route_for(s, DE_TEXT) == app.Route("supertonic", "supertonic:M1", "de")
    assert app.route_for(s, HR_TEXT) == app.Route("supertonic", "supertonic:F2", "hr")
    # Supertonic English
    s2 = _speech(s, en="supertonic:M3")
    assert app.route_for(s2, EN_LONG) == app.Route("supertonic", "supertonic:M3", "en")


def test_route_explicit_lang_skips_detection(fakes):
    s = _speech(fakes, es="ef_dora", de="supertonic:M4")
    install_supertonic(s)
    assert app.route_for(s, HR_TEXT, "en") == app.Route("kokoro", "af_heart", "en")
    assert app.route_for(s, EN_LONG, "es") == app.Route("kokoro", "ef_dora", "es")
    assert app.route_for(s, EN_LONG, "de") == app.Route("supertonic", "supertonic:M4", "de")
    assert app.route_for(s, EN_LONG, "fr") == app.Route("supertonic", "supertonic:F1", "fr")


def test_route_languages_that_are_off_are_never_detected(fakes):
    s = _speech(fakes, de="", sl="")
    install_supertonic(s)
    assert app.enabled_languages(s) == [c for c in app.SUPERTONIC_LANGUAGES if c not in ("de", "sl")]
    for text in (DE_TEXT, ES_TEXT, HR_TEXT, EN_LONG):
        assert app.route_for(s, text).lang not in ("de", "sl")
    # only two languages on: everything is one of them
    only = _speech(fakes, **{c: "" for c in app.SUPERTONIC_LANGUAGES if c not in ("en", "hr")})
    assert app.route_for(only, DE_TEXT).lang in ("en", "hr")
    assert app.route_for(only, HR_TEXT).lang == "hr" and app.route_for(only, EN_LONG).lang == "en"


def test_route_fallback_language_when_english_is_off(fakes):
    s = _speech(fakes, en="", **{c: "" for c in app.SUPERTONIC_LANGUAGES if c not in ("en", "hr", "de")})
    install_supertonic(s)
    assert app.fallback_lang(s) == "de"  # English is off: the first enabled language
    assert app.route_for(s, "ok").lang == "de"  # too short to detect


def test_route_read_language_forces_the_language(fakes):
    s = _speech(fakes, read_language="hr", es="ef_dora")
    install_supertonic(s)
    for text in (EN_LONG, DE_TEXT, HR_TEXT):
        assert app.route_for(s, text) == app.Route("supertonic", "supertonic:F1", "hr")
    s = _speech(fakes, read_language="es", es="ef_dora")
    assert app.route_for(s, EN_LONG) == app.Route("kokoro", "ef_dora", "es")
    s = _speech(fakes, read_language="en")
    assert app.route_for(s, HR_TEXT) == app.Route("kokoro", "af_heart", "en")
    # a forced language that is Off is ignored (auto)
    s = _speech(fakes, read_language="de", de="")
    assert app.profile_lang(s, EN_LONG) == "en"


def test_route_falls_back_to_kokoro_when_supertonic_is_missing(fakes, caplog):
    s = fakes  # not installed
    route = app.route_for(s, HR_TEXT)  # Kokoro has no Croatian: Kokoro English reads it
    assert route == app.Route("kokoro", "af_heart", "en", True)
    # Kokoro reads Spanish: the Kokoro Spanish voice, not English
    es = _speech(s, es="supertonic:F3")
    assert app.route_for(es, ES_TEXT) == app.Route("kokoro", "ef_dora", "es", True)
    assert app.route_for(es, EN_LONG) == app.Route("kokoro", "af_heart", "en")
    s2 = _speech(s, en="supertonic:F1")
    assert app.route_for(s2, EN_LONG) == app.Route("kokoro", "af_heart", "en", True)
    with caplog.at_level("WARNING", logger="vibetts.app"):
        assert app.note_fallback(route) is True and app.note_fallback(route) is False  # once
    assert caplog.text.count("Supertonic is not installed") == 1
    assert app.note_fallback(app.route_for(s, EN_LONG)) is False  # no fallback, no warning
    install_supertonic(s)
    assert app.route_for(s, HR_TEXT).fell_back is False


def test_route_request_explicit_voice_and_lang(fakes):
    s = fakes
    install_supertonic(s)
    r = app.route_request(s, HR_TEXT, "supertonic:F3")
    assert r == app.Route("supertonic", "supertonic:F3", "hr")  # language: as routed
    assert app.route_request(s, EN_LONG, "supertonic:F3").lang == "en"
    assert app.route_request(s, EN_LONG, "supertonic:F3", "de").lang == "de"
    assert app.route_request(s, HR_TEXT, "bf_emma") == app.Route("kokoro", "bf_emma", "en")
    assert app.route_request(s, HR_TEXT, "bf_emma", "en").lang == "en"
    with pytest.raises(ValueError, match="English only"):
        app.route_request(s, HR_TEXT, "bf_emma", "hr")
    with pytest.raises(ValueError, match="Unknown language"):
        app.route_request(s, EN_LONG, None, "xx")


def test_route_request_kokoro_voice_for_its_own_language(fakes):
    s = fakes
    assert app.route_request(s, HR_TEXT, "ef_dora") == app.Route("kokoro", "ef_dora", "es")
    assert app.route_request(s, EN_LONG, "ef_dora", "es") == app.Route("kokoro", "ef_dora", "es")
    assert app.route_request(s, EN_LONG, "pf_dora", "pt").lang == "pt"
    assert app.route_request(s, EN_LONG, "bm_george", "en").voice == "bm_george"
    with pytest.raises(ValueError, match="reads Spanish only"):
        app.route_request(s, EN_LONG, "ef_dora", "en")
    with pytest.raises(ValueError, match="reads English only"):
        app.route_request(s, EN_LONG, "af_heart", "hr")
    with pytest.raises(ValueError, match="Unknown voice"):
        app.route_request(s, EN_LONG, "jf_alpha")


def test_needed_and_primary_voices(fakes):
    s = fakes  # auto: one Kokoro voice plus the Supertonic family
    assert app.needed_voices(s) == ["af_heart", "supertonic:F1"]
    assert app.primary_voice(s) == "af_heart"
    assert app.missing_voices(s) == ["supertonic:F1"]
    install_supertonic(s)
    assert app.missing_voices(s) == []
    mixed = _speech(s, es="ef_dora", de="supertonic:M2")
    assert app.needed_voices(mixed) == ["af_heart", "supertonic:F1", "supertonic:M2", "ef_dora"]
    # a fixed language needs only its voice
    en = _speech(s, read_language="en")
    assert app.needed_voices(en) == ["af_heart"] and app.primary_voice(en) == "af_heart"
    hr = _speech(s, read_language="hr", hr="supertonic:F5")
    assert app.needed_voices(hr) == ["supertonic:F5"] and app.primary_voice(hr) == "supertonic:F5"
    # a language that is Off is not needed
    off = _speech(s, **{c: "" for c in app.SUPERTONIC_LANGUAGES if c != "en"})
    assert app.needed_voices(off) == ["af_heart"]
    # English off: the first enabled language is the primary one
    no_en = _speech(s, en="", **{c: "" for c in app.SUPERTONIC_LANGUAGES if c not in ("en", "de")})
    assert app.needed_voices(no_en) == ["supertonic:F1"] and app.primary_voice(no_en) == "supertonic:F1"


# ---------------------------------------------------------------- engines

def test_holder_loads_each_family_lazily(fakes):
    s = fakes
    install_supertonic(s)
    holder = app.EngineHolder()
    holder.get(s, voice="af_heart")
    assert holder.loaded.supertonic is None and FakeSupertonic.instances == []
    engine = holder.get(s, voice="supertonic:F1")
    st = engine.supertonic
    assert isinstance(st, FakeSupertonic) and isinstance(engine.kokoro, FakeKokoro)
    assert st.calls == [("Ready.", "supertonic:F1", 1.0)] and st.langs == ["en"]  # warm-up
    assert holder.get(s, voice="supertonic:M1").supertonic is st  # cached
    assert st.model_dir == Path(s.supertonic_dir) and st.steps == 8
    # the key is (dir, steps): other steps reload only Supertonic
    kokoro = engine.kokoro
    again = holder.get(dataclasses.replace(s, supertonic_steps=12), voice="supertonic:F1")
    assert again.supertonic is not st and again.supertonic.steps == 12 and again.kokoro is kokoro
    holder.unload_supertonic()
    assert holder.loaded.supertonic is None and holder.loaded.kokoro is kokoro


def test_holder_default_voice_follows_the_profile(fakes):
    s = _speech(fakes, read_language="hr")
    install_supertonic(s)
    engine = app.EngineHolder().get(s)  # no voice: the primary one (Croatian, Supertonic)
    assert engine.kokoro is None and isinstance(engine.supertonic, FakeSupertonic)


def test_holder_supertonic_missing(fakes):
    with pytest.raises(RuntimeError, match="Supertonic is not installed"):
        app.EngineHolder().get(fakes, voice="supertonic:F1")


def test_routing_engine_dispatches_by_voice(fakes):
    k, st = FakeKokoro(), FakeSupertonic("d")
    r = app.RoutingEngine(k, st)
    r.synth("a", "af_heart", 1.0, "en")
    r.synth("b", "supertonic:F1", 1.2, "hr")
    assert k.calls == [("a", "af_heart", 1.0)] and st.calls == [("b", "supertonic:F1", 1.2)]
    assert st.langs == ["hr"]
    assert r.voices() == ["af_heart", *app.supertonic_voice_ids()]
    with pytest.raises(RuntimeError, match="Supertonic is not installed"):
        app.RoutingEngine(k, None).synth("c", "supertonic:F1", 1.0)
    with pytest.raises(RuntimeError, match="Download the voice model"):
        app.RoutingEngine(None, st).synth("c", "af_heart", 1.0)


def _wait_loaded(holder, kokoro=False, supertonic=False):
    for _ in range(100):
        loaded = holder.loaded
        if loaded is not None and (not kokoro or loaded.kokoro) \
                and (not supertonic or loaded.supertonic):
            return loaded
        time.sleep(0.02)
    raise AssertionError("preload did not finish")


def test_preload_loads_the_engines_the_profiles_need(fakes):
    s = fakes
    install_supertonic(s)
    holder = app.EngineHolder()
    holder.preload(s)  # auto: English (Kokoro) and the other languages (Supertonic)
    loaded = _wait_loaded(holder, kokoro=True, supertonic=True)
    assert isinstance(loaded.kokoro, FakeKokoro) and isinstance(loaded.supertonic, FakeSupertonic)
    only_en = app.EngineHolder()
    only_en.preload(_speech(s, read_language="en"))
    _wait_loaded(only_en, kokoro=True)
    assert only_en.loaded.supertonic is None
    only_other = app.EngineHolder()
    only_other.preload(_speech(s, read_language="hr"))
    _wait_loaded(only_other, supertonic=True)
    assert only_other.loaded.kokoro is None


def test_preload_without_supertonic_loads_kokoro_only(fakes):
    holder = app.EngineHolder()
    holder.preload(fakes)  # the Supertonic languages fall back to Kokoro
    _wait_loaded(holder, kokoro=True)
    assert holder.loaded.supertonic is None and FakeSupertonic.instances == []


# ---------------------------------------------------------------- export / bridge with lang

def test_run_export_routes_by_lang(fakes, tmp_path):
    from vibetts.export import ExportRequest

    s = dataclasses.replace(fakes, sentence_pause=0.0)
    install_supertonic(s)
    k, st = FakeKokoro(), FakeSupertonic("d")
    engine = app.RoutingEngine(k, st)
    res = app.run_export(engine, s, ExportRequest("Zdravo svijete.", Path("a.wav"), lang="hr"),
                         tmp_path)
    assert (res.voice, res.lang) == ("supertonic:F1", "hr") and st.langs == ["hr"] and not k.calls
    res = app.run_export(engine, s, ExportRequest(HR_TEXT, Path("b.wav")), tmp_path)
    assert (res.voice, res.lang) == ("supertonic:F1", "hr")  # detected
    res = app.run_export(engine, s, ExportRequest(EN_LONG, Path("c.wav")), tmp_path)
    assert (res.voice, res.lang) == ("af_heart", "en") and k.langs == ["en"]
    res = app.run_export(engine, s, ExportRequest(HR_TEXT, Path("d.wav"), voice="supertonic:M2",
                                                  lang="sl"), tmp_path)
    assert (res.voice, res.lang) == ("supertonic:M2", "sl")
    with pytest.raises(ValueError, match="English only"):
        app.run_export(engine, s, ExportRequest("Hi.", Path("e.wav"), voice="bf_emma", lang="hr"),
                       tmp_path)
    with pytest.raises(ValueError, match="Unknown language"):
        app.run_export(engine, s, ExportRequest("Hi.", Path("f.wav"), lang="xx"), tmp_path)


def test_run_export_44k_wav_header(fakes, tmp_path):
    import wave

    from vibetts.export import ExportRequest

    s = dataclasses.replace(fakes, sentence_pause=0.0)
    res = app.run_export(FakeKokoro(), s, ExportRequest("Hello there.", Path("r.wav")), tmp_path)
    with wave.open(str(res.path)) as w:
        assert w.getframerate() == 44100 and w.getnchannels() == 1


def test_run_export_supertonic_missing_for_explicit_voice(fakes, tmp_path):
    from vibetts.export import ExportRequest

    with pytest.raises(RuntimeError, match="Supertonic is not installed"):
        app.run_export(FakeKokoro(), fakes, ExportRequest("Hi.", voice="supertonic:F1"), tmp_path)


def test_run_export_falls_back_when_supertonic_is_missing(fakes, tmp_path):
    from vibetts.export import ExportRequest

    s = dataclasses.replace(fakes, sentence_pause=0.0)
    k = FakeKokoro()
    res = app.run_export(k, s, ExportRequest(HR_TEXT, Path("fb.wav")), tmp_path)
    assert (res.voice, res.lang) == ("af_heart", "en") and k.calls


def test_export_audio_loads_the_engine_of_the_route(eapp):
    from vibetts.export import ExportRequest

    install_supertonic(eapp.cfg.speech)
    res = eapp.export_audio(ExportRequest("Zdravo.", Path("hr.wav"), lang="hr"))
    assert res.voice == "supertonic:F1" and res.lang == "hr"
    assert eapp.engine.loaded.kokoro is None  # English was never needed
    assert eapp.engine.loaded.supertonic.langs[-1] == "hr"
    res = eapp.export_audio(ExportRequest("Hello there.", Path("en.wav")))
    assert res.voice == "af_heart" and eapp.engine.loaded.kokoro is not None


def test_bridge_checks_voice_and_lang(bridge):
    from vibetts.mcp_server import ToolError

    b, a = bridge
    s = a.cfg.speech
    with pytest.raises(ToolError, match="Supertonic is not installed"):
        b.synthesize_to_file({"text": "Hi.", "voice": "supertonic:F1"})
    with pytest.raises(ToolError, match="Unknown language 'xx'"):
        b.synthesize_to_file({"text": "Hi.", "lang": "xx"})
    with pytest.raises(ToolError, match="English only"):
        b.synthesize_to_file({"text": "Hi.", "voice": "bf_emma", "lang": "hr"})
    with pytest.raises(ToolError, match="Unknown voice"):
        b.speak("Hello.", "supertonic:Z9", None, False, 1.0)
    install_supertonic(s)
    out = b.synthesize_to_file({"text": "Zdravo svijete.", "lang": "hr", "output": "z.wav"})
    assert (out["voice"], out["lang"]) == ("supertonic:F1", "hr")
    out = b.synthesize_to_file({"text": "Hello.", "voice": "supertonic:M1", "output": "m.wav"})
    assert (out["voice"], out["lang"]) == ("supertonic:M1", "en")
    out = b.synthesize_to_file({"text": "Hello.", "lang": "en", "output": "e.wav"})
    assert (out["voice"], out["lang"]) == ("af_heart", "en")


def test_bridge_list_voices_with_supertonic(bridge):
    b, a = bridge
    install_supertonic(a.cfg.speech)
    a._voices = lambda: ["af_heart", *app.supertonic_voice_ids()]
    a.cfg.speech.language_voices["de"] = ""
    a.cfg.speech.read_language = "hr"
    out = b.list_voices()
    st = out["voices"][1]
    assert st == {"id": "supertonic:F1", "label": "F1 (Supertonic, female)", "engine": "supertonic",
                  "languages": list(app.SUPERTONIC_LANGUAGES)}
    assert out["supertonic_installed"] is True and out["read_language"] == "hr"
    assert out["language_voices"]["de"] == "" and out["language_voices"]["hr"] == "supertonic:F1"


def test_bridge_list_voices_kokoro_voices_list_their_own_language(bridge):
    b, a = bridge
    a._voices = lambda: ["af_heart", "ef_dora", "pf_dora", "supertonic:F1"]
    by_id = {v["id"]: v["languages"] for v in b.list_voices()["voices"]}
    assert by_id["af_heart"] == ["en"] and by_id["ef_dora"] == ["es"] and by_id["pf_dora"] == ["pt"]
    assert by_id["supertonic:F1"] == list(app.SUPERTONIC_LANGUAGES)


def test_bridge_accepts_a_kokoro_voice_for_its_own_language(bridge, monkeypatch):
    from vibetts.mcp_server import ToolError

    b, a = bridge
    a._voices = lambda: ["af_heart", "ef_dora"]
    b._check_voice(a.cfg.speech, "ef_dora", "es")
    b._check_voice(a.cfg.speech, "ef_dora", None)
    with pytest.raises(ToolError, match="reads Spanish only"):
        b._check_voice(a.cfg.speech, "ef_dora", "en")
    with pytest.raises(ToolError, match="reads English only"):
        b._check_voice(a.cfg.speech, "af_heart", "hr")


def test_bridge_speak_routes_by_lang(bridge, monkeypatch):
    b, a = bridge
    install_supertonic(a.cfg.speech)
    seen = []
    req = types.SimpleNamespace(done=threading.Event(), interrupted=False, error=None)
    req.done.set()
    monkeypatch.setattr(a, "speak", lambda chunks, s, voice=None, lang=None: seen.append(
        (voice, lang)) or req)
    assert b.speak("Zdravo svijete.", None, None, True, 1.0, "hr")["lang"] == "hr"
    assert b.speak(HR_TEXT, None, None, True, 1.0)["voice"] == "supertonic:F1"
    assert b.speak(EN_LONG, None, None, True, 1.0)["voice"] == "af_heart"
    assert seen == [("supertonic:F1", "hr"), ("supertonic:F1", "hr"), ("af_heart", "en")]


def test_bridge_batch_with_lang(bridge, tmp_path):
    b, a = bridge
    install_supertonic(a.cfg.speech)
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    items = [
        {"text": "Dobar dan.", "output": str(out_dir / "a.wav"), "lang": "hr"},
        {"text": "Hello.", "output": str(out_dir / "b.wav"), "voice": "bf_emma", "lang": "hr"},
        {"text": "Hello.", "output": str(out_dir / "c.wav"), "lang": "xx"},
    ]
    r = b.synthesize_batch({"items": items})["results"]
    assert [x["ok"] for x in r] == [True, False, False]
    assert "English only" in r[1]["error"] and "Unknown language" in r[2]["error"]
    assert (out_dir / "a.wav").exists()


# ---------------------------------------------------------------- app wiring

def test_voices_lists_kokoro_then_supertonic_when_installed(fakes, tmp_path):
    a = make_app(fakes)
    with open(a.cfg.speech.voices, "wb") as fh:
        np.savez(fh, af_heart=np.zeros(2), ff_siwis=np.zeros(2), bm_george=np.zeros(2),
                 jf_alpha=np.zeros(2), zf_xiaobei=np.zeros(2))
    # French is supported now; Japanese and Chinese are not
    assert a._voices() == ["af_heart", "bm_george", "ff_siwis"]
    install_supertonic(a.cfg.speech)
    assert a._voices() == ["af_heart", "bm_george", "ff_siwis", *app.supertonic_voice_ids()]


def test_supertonic_changed_reloads_and_refreshes(fakes):
    a = make_app(fakes)
    install_supertonic(a.cfg.speech)
    a.engine.get(a.cfg.speech, voice="supertonic:F1")
    assert a.engine.loaded.supertonic is not None
    app._fallback_warned = True
    import shutil

    shutil.rmtree(a.cfg.speech.supertonic_dir)
    a.cfg.speech.preload = False
    a._supertonic_changed()
    assert a.engine.loaded is None and app._fallback_warned is False and a.tray.updates == 1


def test_settings_hooks_get_supertonic_wiring(mapp, monkeypatch):
    a = mapp
    a._autostart = type("A", (), {"is_enabled": staticmethod(lambda: False)})
    monkeypatch.setattr("vibetts.settings_ui.SettingsWindow", FakeSettingsWindow)
    hooks = a._settings_window().hooks
    assert hooks.supertonic_dir() == Path(a.cfg.speech.supertonic_dir)
    assert hooks.supertonic_present() is False
    install_supertonic(a.cfg.speech)
    assert hooks.supertonic_present() is True
    hooks.supertonic_changed()


def test_apply_config_reports_the_missing_engine(mapp):
    a = mapp
    a.listener = type("L", (), {"set_bindings": lambda self, b: None})()
    base = a.cfg.speech
    new = Config(speech=_speech(base, "en", en="supertonic:F1"))
    assert a.apply_config(new) == app.MISSING_SUPERTONIC  # the fallback language (English) needs it
    ok = Config(speech=dataclasses.replace(base, preload=False))
    assert a.apply_config(ok) is None  # Kokoro is there; Supertonic is only a soft need


def test_read_selection_uses_the_route(fakes, monkeypatch):
    """The hotkey path: routing decided once from the grabbed text, then speak(route)."""
    from vibetts import hotkeys, selection

    a = make_app(fakes)
    install_supertonic(a.cfg.speech)
    a.cfg.selection.max_chars = 0
    a.listener = type("L", (), {"chord": lambda self, n: None})()
    spoken = []
    a.speak = lambda chunks, s, voice=None, lang=None: spoken.append((voice, lang))
    a.open_settings = lambda *args, **kw: spoken.append(("settings", args))
    a._grabbing, a._grab_lock, a._hint = True, threading.Lock(), None
    monkeypatch.setattr(hotkeys, "wait_for_chord_release", lambda chord, t: True)
    for text in (HR_TEXT, EN_LONG):
        monkeypatch.setattr(selection, "grab_selection", lambda *a, text=text: types.SimpleNamespace(
            text=text, from_clipboard=False))
        a._read_selection()
    assert spoken == [("supertonic:F1", "hr"), ("af_heart", "en")]
    # Supertonic missing: Kokoro reads it and a hint is queued (once)
    import shutil

    shutil.rmtree(a.cfg.speech.supertonic_dir)
    for _ in range(2):
        monkeypatch.setattr(selection, "grab_selection", lambda *a: types.SimpleNamespace(
            text=HR_TEXT, from_clipboard=False))
        a._read_selection()
    assert spoken[2:] == [("af_heart", "en"), ("af_heart", "en")]
    assert a._hint == app.SUPERTONIC_HINT


def test_speak_done_shows_the_hint(fakes):
    a = make_app(fakes)
    a.tray = type("T", (), {"set_speaking": lambda self, v: None, "update_menu": lambda self: None})()
    a.overlay = type("O", (), {"hide": lambda self: None, "show": lambda self, *x, **k: None})()
    a._hint = app.SUPERTONIC_HINT
    a._current = req = object()
    a._on_speak_done(req, None)
    assert a.flashes == [("ok", app.SUPERTONIC_HINT)] and a._hint is None


def test_tray_download_opens_the_engine_that_is_missing(fakes):
    a = make_app(fakes)
    opened = []
    a.open_settings = lambda tab=None, banner=None, download=False: opened.append((tab, download))
    a._download_missing()
    assert opened == [("supertonic", True)]  # Kokoro is there, Supertonic (other languages) is not
    a.cfg.speech.model = str(Path(a.cfg.speech.model).with_name("nope.onnx"))
    a._download_missing()
    assert opened[-1] == ("kokoro", True)


# ---------------------------------------------------------------- CLI routing

def test_say_routes_like_the_hotkey(fakes, tmp_path, monkeypatch, capsys):
    s = dataclasses.replace(fakes, sentence_pause=0.0)
    install_supertonic(s)
    cfg = Config(speech=s)
    cfg.export.format = "wav"
    assert app.say(cfg, HR_TEXT, str(tmp_path / "hr.wav")) == 0
    assert "voice supertonic:F1, language hr" in capsys.readouterr().out
    assert FakeSupertonic.instances[-1].langs[-1] == "hr"
    assert app.say(cfg, EN_LONG, str(tmp_path / "en.wav")) == 0
    assert "voice af_heart, language en" in capsys.readouterr().out
    # forced English: Croatian text goes to the English voice
    cfg.speech.read_language = "en"
    assert app.say(cfg, HR_TEXT, str(tmp_path / "forced.wav")) == 0
    assert "voice af_heart" in capsys.readouterr().out


def test_say_without_supertonic_falls_back_and_says_so(fakes, tmp_path, capsys):
    cfg = Config(speech=dataclasses.replace(fakes, sentence_pause=0.0))
    assert app.say(cfg, HR_TEXT, str(tmp_path / "fb.wav")) == 0
    captured = capsys.readouterr()
    assert "Supertonic is not installed" in captured.err and "voice af_heart" in captured.out


def test_say_supertonic_only_missing_message(fakes, tmp_path, capsys):
    cfg = Config(speech=_speech(fakes, "en", en="supertonic:F1"))
    cfg.speech.model = str(tmp_path / "nope.onnx")  # even Kokoro is gone: the fallback fails too
    assert app.say(cfg, EN_LONG, str(tmp_path / "x.wav")) == 1
    assert "model files missing" in capsys.readouterr().err


# ---------------------------------------------------------------- speak speed / on_chunk, pause

def test_speak_speed_and_on_chunk(fakes):
    from vibetts.textprep import Chunk

    s = dataclasses.replace(fakes, speed=1.3)
    a = make_app(s)
    reqs = []
    a.speaker = type("Sp", (), {"speak": lambda self, req: reqs.append(req)})()
    seen = []
    a.speak([Chunk("Hi", 0.0)], s)
    a.speak([Chunk("Hi", 0.0)], s, speed=0.8, on_chunk=seen.append)
    assert [r.speed for r in reqs] == [1.3, 0.8]
    assert reqs[0].on_chunk is None
    reqs[1].on_chunk("chunk")
    assert seen == ["chunk"]


class FakePausable:
    def __init__(self, can=True):
        self.can = can
        self.paused = False
        self.busy = True

    def pause(self):
        if not self.can or self.paused:
            return False
        self.paused = True
        return True

    def resume(self):
        if not self.paused:
            return False
        self.paused = False
        return True


class FakeReading:
    def __init__(self):
        self.calls = []

    def notify_paused(self, paused):
        self.calls.append(paused)


class RecordingOverlay(FakeOverlay):
    def __init__(self):
        self.shown = []

    def show(self, kind, text, ticking=False):
        self.shown.append((kind, text))


def test_app_pause_resume_toggle(fakes):
    a = make_app(fakes)
    a.speaker, a.reading, a.overlay = FakePausable(), FakeReading(), RecordingOverlay()
    assert a.pause() is True and a.speaker.paused
    assert a.overlay.shown[-1] == ("speaking", "Paused · Esc to stop") and a.reading.calls == [True]
    assert a.pause() is False and a.reading.calls == [True]  # already paused
    assert a.resume() is True and not a.speaker.paused
    assert a.overlay.shown[-1][1].startswith("Reading") and a.reading.calls == [True, False]
    assert a.resume() is False and a.reading.calls == [True, False]
    a.toggle_pause()
    assert a.speaker.paused and a.reading.calls[-1] is True
    a.toggle_pause()
    assert not a.speaker.paused and a.reading.calls[-1] is False


def test_app_pause_when_nothing_is_playing(fakes):
    a = make_app(fakes)
    a.speaker, a.reading, a.overlay = FakePausable(can=False), FakeReading(), RecordingOverlay()
    assert a.pause() is False
    a.toggle_pause()
    assert a.reading.calls == [] and a.overlay.shown == []


# ---------------------------------------------------------------- reader lifecycle (VTF-07)

class FakeReaderServer:
    instances: list = []
    fail: Exception | None = None

    def __init__(self, api, port, extension_ids=(), max_chars=500_000, web_root=None):
        self.api, self.port, self.extension_ids, self.max_chars = api, port, extension_ids, max_chars
        self.started = self.stopped = False
        FakeReaderServer.instances.append(self)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/"

    def start(self):
        if FakeReaderServer.fail:
            raise FakeReaderServer.fail
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.fixture
def rapp(eapp, monkeypatch):
    FakeReaderServer.instances, FakeReaderServer.fail = [], None
    monkeypatch.setattr("vibetts.reader_server.ReaderServer", FakeReaderServer)
    eapp.cfg.reader.enabled = True
    eapp.cfg.reader.library_folder = str(Path(eapp.cfg.export.folder).parent / "books")
    return eapp


def test_sync_reader_start_noop_restart_stop(rapp):
    a = rapp
    a.cfg.reader.enabled = False
    a._sync_reader()
    assert FakeReaderServer.instances == [] and a.reader_status()["running"] is False
    a.cfg.reader.enabled, a.cfg.reader.port = True, 7800
    a._sync_reader()
    (first,) = FakeReaderServer.instances
    assert first.started and first.port == 7800 and first.max_chars == 500000
    assert first.extension_ids == ("injaiiefkhoimpkeandbjfbnnjddaceh",)
    assert isinstance(first.api, app.ReaderBridge) and first.api.version
    assert a.reader_status() == {"enabled": True, "running": True,
                                 "url": "http://127.0.0.1:7800/", "error": ""}
    a._sync_reader()  # unchanged: nothing happens
    assert len(FakeReaderServer.instances) == 1 and not first.stopped
    a.cfg.reader.port = 7801
    a._sync_reader()
    assert first.stopped and FakeReaderServer.instances[1].started
    a.cfg.reader.extension_ids = ["a" * 32]
    a._sync_reader()
    assert FakeReaderServer.instances[2].extension_ids == ("a" * 32,)
    a.cfg.reader.max_chars = 10
    a._sync_reader()
    assert len(FakeReaderServer.instances) == 4 and FakeReaderServer.instances[3].max_chars == 10
    a.cfg.reader.library_folder = "elsewhere"
    a._sync_reader()
    assert len(FakeReaderServer.instances) == 5
    a.cfg.reader.enabled = False
    a._sync_reader()
    assert FakeReaderServer.instances[4].stopped
    assert a.reader_status()["running"] is False and a.reader_status()["error"] == ""


def test_sync_reader_port_busy(rapp):
    a = rapp
    FakeReaderServer.fail = OSError("address in use")
    a._sync_reader()
    st = a.reader_status()
    assert st["running"] is False and "7724 is busy" in st["error"]
    assert a.flashes and a.flashes[-1][0] == "error"
    FakeReaderServer.fail = None
    a._sync_reader()  # retried on the next apply: the error clears
    assert a.reader_status()["running"] is True and a.reader_status()["error"] == ""
    a.cfg.reader.enabled = False
    a._sync_reader()
    assert a.reader_status()["error"] == ""


def test_apply_config_syncs_reader(rapp):
    a = rapp
    a.cfg.reader.enabled = False
    a.listener = type("L", (), {"set_bindings": lambda self, b: None})()
    new = Config(speech=dataclasses.replace(a.cfg.speech, model="nope.onnx"))
    a.apply_config(new)
    assert len(FakeReaderServer.instances) == 1 and FakeReaderServer.instances[0].started


def test_quit_stops_reader(rapp):
    a = rapp
    a._sync_reader()
    server = FakeReaderServer.instances[0]
    a._quit_lock, a._quit_done, a._instance = threading.Lock(), False, None
    noop = type("N", (), {"stop": lambda self, *x: None, "quit": lambda self: None})()
    a.speaker, a.listener, a.tray, a.overlay = noop, noop, noop, noop
    a.quit()
    assert server.stopped and a.reader_status()["running"] is False


def test_open_reader_opens_the_browser_or_explains(rapp, monkeypatch):
    opened = []
    monkeypatch.setattr(app.webbrowser, "open", lambda url: opened.append(url))
    with pytest.raises(RuntimeError, match="turned off"):
        rapp.open_reader()
    rapp._sync_reader()
    rapp.open_reader()
    assert opened == ["http://127.0.0.1:7724/"]


def test_reader_bridge_library_follows_the_folder(rapp, tmp_path):
    b = app.ReaderBridge(rapp)
    rapp.cfg.reader.library_folder = str(tmp_path / "one")
    first = b.library
    assert b.library is first and first.folder == tmp_path / "one"
    rapp.cfg.reader.library_folder = str(tmp_path / "two")
    assert b.library is not first and b.library.folder == tmp_path / "two"


def test_reader_bridge_list_voices_and_session(rapp):
    rapp._voices = lambda: ["af_heart"]
    rapp.reading = object()
    b = app.ReaderBridge(rapp)
    assert b.session is rapp.reading
    out = b.list_voices()
    assert [v["id"] for v in out["voices"]] == ["af_heart"]
    assert out["read_language"] == "auto" and out["language_voices"]["en"] == "af_heart"


def test_reader_bridge_open_book_returns_url_and_opens_browser(rapp, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(app.webbrowser, "open", lambda url: opened.append(url))
    book = tmp_path / "story.txt"
    book.write_text("Chapter 1\n\nOnce upon a time there was a book.\n", encoding="utf-8")
    rapp._sync_reader()
    out = rapp._reader_bridge.open_book(str(book))
    assert re.fullmatch(r"[0-9a-f]{16}", out["id"]) and out["title"]
    assert out["url"] == f"http://127.0.0.1:7724/#/book/{out['id']}"
    assert opened == [out["url"]]
    assert rapp._reader_bridge.library.list()[0]["id"] == out["id"]


def test_reader_bridge_open_book_errors(rapp, tmp_path, monkeypatch):
    from vibetts.library import LibraryError

    monkeypatch.setattr(app.webbrowser, "open", lambda url: pytest.fail("no browser"))
    b = app.ReaderBridge(rapp)
    with pytest.raises(RuntimeError, match="not running"):
        b.open_book(str(tmp_path / "x.txt"))
    rapp._sync_reader()
    with pytest.raises(LibraryError):
        rapp._reader_bridge.open_book(str(tmp_path / "missing.epub"))


class FakeNotifyTray:
    def __init__(self):
        self.notes = []

    def update_menu(self):
        pass

    def notify(self, title, message):
        self.notes.append((title, message))


def test_open_book_on_start_ok_and_errors(rapp, tmp_path, monkeypatch):
    monkeypatch.setattr(app.webbrowser, "open", lambda url: None)
    a = rapp
    a.tray = FakeNotifyTray()
    book = tmp_path / "story.txt"
    book.write_text("Hello there, this is a short book.\n", encoding="utf-8")
    a._open_book(str(book))  # reader not running yet
    assert a.flashes[-1][0] == "error" and "turned off" in a.tray.notes[-1][1]
    a.flashes.clear()
    a._sync_reader()
    a._open_book(str(book))
    assert a.flashes == [] and len(a.tray.notes) == 1
    a._open_book(str(tmp_path / "missing.epub"))
    assert a.flashes[-1][0] == "error" and "missing.epub" in a.tray.notes[-1][1]


def test_settings_hooks_include_reader(rapp, monkeypatch):
    a = rapp
    a._autostart = type("A", (), {"is_enabled": staticmethod(lambda: False)})
    monkeypatch.setattr("vibetts.settings_ui.SettingsWindow", FakeSettingsWindow)
    hooks = a._settings_window().hooks
    assert hooks.reader_status == a.reader_status and hooks.open_reader == a.open_reader
    assert hooks.extension_folder() == app.ROOT / "extension"
    calls = []
    monkeypatch.setattr("vibetts.fileassoc.register", lambda exts: calls.append(("register", exts)))
    monkeypatch.setattr("vibetts.fileassoc.unregister", lambda: calls.append("unregister"))
    monkeypatch.setattr("vibetts.fileassoc.registered_exts", lambda: [".epub"])
    monkeypatch.setattr("vibetts.fileassoc.open_default_apps", lambda: calls.append("defaults"))
    hooks.register_files([".epub", ".txt"])
    hooks.unregister_files()
    hooks.open_default_apps()
    assert hooks.registered_exts() == [".epub"]
    assert calls == [("register", [".epub", ".txt"]), "unregister", "defaults"]


def test_tray_menu_has_reader_and_pause_items():
    tray, calls = make_tray()
    names = [getattr(i, "text", "") for i in tray._menu_items()]
    assert "Open reader" in names and "Pause" in names
    assert names.index("Open reader") == names.index("Export audio…") + 1


def test_tray_reader_item_visibility_and_pause_label():
    from vibetts.tray import Tray, TrayHooks

    state = {"running": False, "paused": False, "speaking": False}
    calls = []
    noop = lambda *a: None  # noqa: E731
    hooks = TrayHooks(
        is_enabled=lambda: True, set_enabled=noop, is_speaking=lambda: state["speaking"],
        stop=noop, model_missing=lambda: False, download_model=noop, voices=lambda: [],
        current_voice=lambda lang: "af_heart", set_voice=noop, current_speed=lambda: 1.0,
        set_speed=noop, is_autostart=lambda: False, set_autostart=noop, open_settings=noop,
        open_config=noop, open_logs=noop, reload_config=noop, quit=noop,
        reader_running=lambda: state["running"], open_reader=lambda: calls.append("reader"),
        is_paused=lambda: state["paused"], toggle_pause=lambda: calls.append("pause"))
    tray = Tray(hooks)
    items = {getattr(i, "text", ""): i for i in tray._menu_items()}
    reader = items["Open reader"]
    assert reader.visible is False
    state["running"] = True
    assert reader.visible is True
    reader(tray._icon)
    pause = items["Pause"]
    assert pause.enabled is False
    state["speaking"] = True
    assert pause.enabled is True
    pause(tray._icon)
    state["paused"] = True
    assert pause.text == "Resume"
    assert calls == ["reader", "pause"]


# ---------------------------------------------------------------- forward_open / --open

@pytest.fixture
def open_server():
    """A tiny local HTTP server standing in for the running instance's /api/open."""
    import http.server
    import json as _json

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        status = 200

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.update(path=self.path, body=_json.loads(body), headers=dict(self.headers))
            data = b"{}"
            self.send_response(type(self).status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, Handler, seen
    server.shutdown()
    server.server_close()


def test_forward_open_posts_the_absolute_path_without_origin(open_server, tmp_path):
    server, handler, seen = open_server
    book = tmp_path / "b.epub"
    assert app.forward_open(server.server_address[1], str(book)) is True
    assert seen["path"] == "/api/open" and seen["body"] == {"path": str(book.resolve())}
    assert "Origin" not in seen["headers"]
    assert seen["headers"]["Content-Type"] == "application/json"


def test_forward_open_failures(open_server, tmp_path, caplog):
    server, handler, seen = open_server
    handler.status = 400
    assert app.forward_open(server.server_address[1], str(tmp_path / "b.epub")) is False
    assert "refused" in caplog.text
    handler.status = 200
    port = server.server_address[1]
    server.shutdown()
    server.server_close()
    assert app.forward_open(port, str(tmp_path / "b.epub")) is False  # nothing listens


class FakeInstance:
    def __init__(self, first):
        self.first = first
        self.shown = self.closed = 0

    def acquire(self):
        return self.first

    def signal_show(self):
        self.shown += 1

    def close(self):
        self.closed += 1


@pytest.fixture
def second_launch(cli, monkeypatch):
    inst = FakeInstance(first=False)
    monkeypatch.setattr("vibetts.single_instance.SingleInstance", lambda: inst)
    monkeypatch.setattr(app, "_set_dpi_aware", lambda: None)
    return inst


def test_main_open_forwards_to_the_running_instance(second_launch, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app, "forward_open", lambda port, path: calls.append((port, path)) or True)
    assert app.main(["--open", str(tmp_path / "b.epub")]) == 0
    assert calls == [(7724, str(tmp_path / "b.epub"))]
    assert second_launch.shown == 0 and second_launch.closed == 1


@pytest.mark.parametrize("forwarded,enabled", [(False, True), (True, False)])
def test_main_open_falls_back_to_showing_the_window(second_launch, cli, monkeypatch, tmp_path,
                                                    forwarded, enabled):
    cli.reader.enabled = enabled
    monkeypatch.setattr(app, "forward_open", lambda port, path: forwarded)
    assert app.main(["--open", str(tmp_path / "b.epub")]) == 0
    assert second_launch.shown == 1


def test_main_second_launch_without_open_just_signals(second_launch, monkeypatch):
    monkeypatch.setattr(app, "forward_open", lambda *a: pytest.fail("no --open given"))
    assert app.main([]) == 0
    assert second_launch.shown == 1


def test_main_first_instance_passes_open_path_to_the_app(cli, monkeypatch):
    inst = FakeInstance(first=True)
    monkeypatch.setattr("vibetts.single_instance.SingleInstance", lambda: inst)
    monkeypatch.setattr(app, "_set_dpi_aware", lambda: None)
    made = {}

    class FakeTTSApp:
        def __init__(self, cfg, start_hidden=False, instance=None, open_path=None):
            made.update(open_path=open_path, instance=instance)

        def run(self):
            made["ran"] = True

    monkeypatch.setattr(app, "TTSApp", FakeTTSApp)
    assert app.main(["--open", "relative.epub"]) == 0
    assert made["ran"] and made["open_path"] == str(Path("relative.epub").resolve())
    assert made["instance"] is inst


def test_main_open_argument_conflicts(cli):
    for extra in (["--say", "x"], ["--doc", "d"], ["--mcp-only"]):
        with pytest.raises(SystemExit):
            app.main(["--open", "b.epub", *extra])


def test_start_skips_the_settings_window_when_opening_a_book(fakes):
    a = make_app(fakes)
    opened = []
    a.cfg.ui.show_settings_on_start = True
    a._start_hidden, a._open_path, a._instance = False, "b.epub", None
    a._startup_errors = []
    noop = type("N", (), {"start": lambda self: None, "notify": lambda self, *x: None})()
    a.listener, a.tray = noop, noop
    a._autostart = type("A", (), {"sync": staticmethod(lambda: None)})
    a.open_settings = lambda *args, **kw: opened.append(args)
    a._preload = lambda: None
    a._sync_mcp = lambda: None
    a._sync_reader = lambda: None
    started = []
    a._open_book = lambda path: started.append(path)
    a._start()
    assert opened == []  # the voice is installed (fakes) and a book is opening
    deadline = time.time() + 2
    while not started and time.time() < deadline:
        time.sleep(0.01)
    assert started == ["b.epub"] and a._open_path is None
    a._start()
    assert opened == [()]


# ---------------------------------------------------------------- Higgs (VTF-09)

def test_higgs_helpers_and_setup(fakes):
    s = fakes
    assert not app.higgs_present(s) and app.voice_ready(s, "higgs:F1") is False
    assert app.missing_message("higgs:F1") == app.MISSING_HIGGS
    assert app.setup_tab("higgs:F1") == "higgs" and app.setup_tab("supertonic:F1") == "supertonic"
    install_higgs(s)
    assert app.higgs_present(s) and app.voice_ready(s, "higgs:F1") is True
    assert app.higgs_voice_ids(s)[:4] == ["higgs:F1", "higgs:F2", "higgs:M1", "higgs:M2"]
    assert app.voice_ready(s, "higgs:Gone") is False  # a deleted custom voice


def test_route_higgs_languages(fakes):
    s = _speech(fakes, hr="higgs:F1", bs="higgs:M1", de="supertonic:M1")
    install_higgs(s)
    assert app.route_for(s, HR_TEXT, "hr") == app.Route("higgs", "higgs:F1", "hr")
    assert app.route_for(s, "x", "bs") == app.Route("higgs", "higgs:M1", "bs")
    # a Higgs-only language that is Off still has a default for an explicit request
    assert app.route_for(s, "x", "sq") == app.Route("higgs", "higgs:F1", "sq")


def test_route_higgs_missing_falls_back(fakes, caplog):
    s = _speech(fakes, hr="higgs:F1", es="higgs:F2", bs="higgs:F1")
    # nothing installed: Kokoro Spanish for es, Kokoro English otherwise
    assert app.route_for(s, "x", "es") == app.Route("kokoro", "ef_dora", "es", True, "higgs")
    assert app.route_for(s, "x", "hr") == app.Route("kokoro", "af_heart", "en", True, "higgs")
    assert app.route_for(s, "x", "bs") == app.Route("kokoro", "af_heart", "en", True, "higgs")
    install_supertonic(s)  # Supertonic reads hr and es, not bs
    assert app.route_for(s, "x", "hr") == app.Route("supertonic", "supertonic:F1", "hr", True, "higgs")
    assert app.route_for(s, "x", "es") == app.Route("supertonic", "supertonic:F1", "es", True, "higgs")
    assert app.route_for(s, "x", "bs") == app.Route("kokoro", "af_heart", "en", True, "higgs")
    route = app.route_for(s, "x", "hr")
    assert app.fallback_hint(route) == app.HIGGS_HINT
    with caplog.at_level("WARNING", logger="vibetts.app"):
        assert app.note_fallback(route) is True and app.note_fallback(route) is False
    assert "Higgs is not installed" in caplog.text and "Supertonic is not installed" not in caplog.text
    # the missing-Supertonic note is a separate once-per-session message
    st = app.Route("kokoro", "af_heart", "en", True)
    assert app.fallback_hint(st) == app.SUPERTONIC_HINT and app.note_fallback(st) is True


def test_route_with_a_deleted_custom_higgs_voice_falls_back(fakes):
    s = _speech(fakes, hr="higgs:Gone")
    install_higgs(s)
    install_supertonic(s)
    assert app.voice_for_lang(s, "hr") == "higgs:Gone"  # no crash
    assert app.voice_ready(s, "higgs:Gone") is False
    assert app.route_for(s, "x", "hr") == app.Route("supertonic", "supertonic:F1", "hr", True, "higgs")
    Path(s.higgs_voices_dir, "Gone").mkdir(parents=True)  # the clip folder is back
    assert app.route_for(s, "x", "hr") == app.Route("higgs", "higgs:Gone", "hr")


def test_route_request_higgs(fakes):
    s = fakes
    assert app.route_request(s, "x", "higgs:F1", "bs") == app.Route("higgs", "higgs:F1", "bs")
    assert app.route_request(s, "x", "higgs:Anna", "hr") == app.Route("higgs", "higgs:Anna", "hr")
    with pytest.raises(ValueError, match="Higgs does not read Japanese"):
        app.route_request(s, "x", "higgs:F1", "ja")
    with pytest.raises(ValueError, match="Supertonic does not read Bosnian"):
        app.route_request(s, "x", "supertonic:F1", "bs")
    with pytest.raises(ValueError, match="higgs: voice"):
        app.route_request(s, "x", "af_heart", "bs")
    with pytest.raises(ValueError, match="Unknown language"):
        app.route_request(s, "x", None, "xx")
    assert app.route_request(s, "x", None, "bs").lang == "en"  # Higgs missing: Kokoro English
    install_higgs(s)
    assert app.route_request(s, "x", None, "bs") == app.Route("higgs", "higgs:F1", "bs")


def test_routing_engine_sends_higgs_voices_to_higgs():
    k, st, hg = FakeKokoro(), FakeSupertonic("d"), FakeHiggs("d", "v")
    eng = app.RoutingEngine(k, st, hg)
    eng.synth("a", "higgs:F1", 1.0, "hr")
    eng.synth("b", "supertonic:F1", 1.0, "hr")
    eng.synth("c", "af_heart", 1.0)
    assert [c[0] for c in hg.calls] == ["a"] and [c[0] for c in st.calls] == ["b"]
    assert [c[0] for c in k.calls] == ["c"]
    assert "higgs:F1" in eng.voices()
    with pytest.raises(RuntimeError, match="Settings > Higgs"):
        app.RoutingEngine(k, st).synth("a", "higgs:F1", 1.0)


def test_holder_loads_higgs_with_warm_up_and_keys_on_temperature(fakes):
    s = fakes
    holder = app.EngineHolder()
    with pytest.raises(RuntimeError, match="Higgs is not installed"):
        holder.get(s, voice="higgs:F1")
    install_higgs(s)
    eng = holder.get(s, voice="higgs:F2").higgs
    assert isinstance(eng, FakeHiggs) and eng.calls == [("Ready.", "higgs:F2", 1.0)]
    assert eng.model_dir == Path(s.higgs_dir) and eng.temperature == 0.8
    assert holder.get(s, voice="higgs:F2").higgs is eng and holder.loaded.higgs is eng
    assert holder.get(dataclasses.replace(s, higgs_temperature=0.5), voice="higgs:F1").higgs is not eng
    holder.unload_higgs()
    assert holder.loaded is None


def test_higgs_warm_up_failure_does_not_block_loading(fakes, monkeypatch):
    install_higgs(fakes)

    class Boom(FakeHiggs):
        def synth(self, *a, **k):
            raise RuntimeError("no kernels")

    monkeypatch.setattr(app, "_hg", lambda: types.SimpleNamespace(
        models_present=lambda d: True, voice_ids=fake_higgs_voice_ids, HiggsEngine=Boom))
    assert isinstance(app.EngineHolder().get(fakes, voice="higgs:F1").higgs, Boom)


def test_preload_covers_higgs(fakes):
    s = _speech(fakes, read_language="hr", hr="higgs:F1")
    install_higgs(s)
    holder = app.EngineHolder()
    holder.preload(s)
    for _ in range(100):
        if holder.loaded is not None and holder.loaded.higgs is not None:
            break
        time.sleep(0.02)
    assert isinstance(holder.loaded.higgs, FakeHiggs)


def test_voices_include_higgs_when_installed(fakes):
    a = make_app(fakes)
    install_higgs(a.cfg.speech)
    assert a._voices()[-4:] == ["higgs:F1", "higgs:F2", "higgs:M1", "higgs:M2"]


def test_higgs_unloaded_when_no_language_uses_it(fakes):
    s = _speech(fakes, hr="higgs:F1")
    install_higgs(s)
    a = make_app(s)
    a.cfg.speech.preload = False
    a.engine.get(s, voice="higgs:F1")
    assert a.engine.loaded.higgs is not None
    a._set_voice("de", "supertonic:M1")  # Higgs still used by hr
    assert a.engine.loaded is not None
    a._set_voice("hr", "supertonic:F1")
    assert a.engine.loaded is None
    a._set_voice("bs", "higgs:M1")  # a Higgs-only language accepts Higgs
    assert a.cfg.speech.language_voices["bs"] == "higgs:M1"
    a._set_voice("bs", "supertonic:F1")  # but not Supertonic
    assert a.cfg.speech.language_voices["bs"] == "higgs:M1"
    a._set_voice("bs", "")
    new = Config(speech=dataclasses.replace(s, language_voices={**s.language_voices, "hr": "higgs:F1"}))
    a.engine.get(new.speech, voice="higgs:F1")
    assert a.engine.loaded.higgs is not None
    new.speech.language_voices["hr"] = "supertonic:F1"
    a.cfg = new
    a.listener = type("L", (), {"set_bindings": lambda self, b: None})()
    a.apply_config(new)  # Settings applied: no language uses Higgs any more
    assert a.engine.loaded is None


def test_higgs_changed_reloads_in_the_background(fakes):
    s = _speech(fakes, hr="higgs:F1")
    install_higgs(s)
    a = make_app(s)
    a.cfg.speech.preload = False
    a.engine.get(s, voice="higgs:F1")
    app._fallback_warned_higgs = True
    import shutil

    shutil.rmtree(s.higgs_dir)
    a._higgs_changed()
    for t in threading.enumerate():
        if t.name == "vibetts-higgs-reload":
            t.join(5)
    assert a.engine.loaded is None and app._fallback_warned_higgs is False and a.tray.updates == 1


def test_settings_hooks_get_higgs_wiring(mapp, monkeypatch):
    a = mapp
    a._autostart = type("A", (), {"is_enabled": staticmethod(lambda: False)})
    monkeypatch.setattr("vibetts.settings_ui.SettingsWindow", FakeSettingsWindow)
    hooks = a._settings_window().hooks
    assert hooks.higgs_dir() == Path(a.cfg.speech.higgs_dir)
    assert hooks.higgs_voices_dir() == Path(a.cfg.speech.higgs_voices_dir)
    assert hooks.higgs_present() is False and hooks.higgs_changed == a._higgs_changed


def test_bridge_list_voices_and_check_voice_with_higgs(bridge):
    from vibetts.mcp_server import ToolError

    b, a = bridge
    s = a.cfg.speech
    a._voices = lambda: ["af_heart", *app.higgs_voice_ids(s)] if app.higgs_present(s) else ["af_heart"]
    with pytest.raises(ToolError, match="Settings > Higgs"):
        b._check_voice(s, "higgs:F1", "hr")
    install_higgs(s)
    out = b.list_voices()
    hg = [v for v in out["voices"] if v["engine"] == "higgs"]
    assert [v["id"] for v in hg] == ["higgs:F1", "higgs:F2", "higgs:M1", "higgs:M2"]
    assert hg[0]["label"] == "F1 (Higgs, female)" and hg[0]["languages"] == list(app.HIGGS_LANGUAGES)
    assert out["higgs_installed"] is True
    b._check_voice(s, "higgs:F1", "bs")
    with pytest.raises(ToolError, match="Unknown voice"):
        b._check_voice(s, "higgs:Nobody", "hr")
    with pytest.raises(ToolError, match="Higgs does not read"):
        b._check_voice(s, "higgs:F1", "ja")
    with pytest.raises(ToolError, match="Unknown language"):
        b._check_voice(s, None, "xx")
    with pytest.raises(ToolError, match="higgs: voice"):
        b._check_voice(s, "af_heart", "bs")


def test_tray_voices_menu_lists_higgs_voices_that_fit():
    ids = ["af_heart", *app.supertonic_voice_ids(), "higgs:F1", "higgs:M2", "higgs:Anna"]
    tray, calls = make_tray(voices=ids, languages=("en", "hr", "bs"),
                            current={"en": "af_heart", "hr": "higgs:F1", "bs": ""})
    langs = {g.text: [i.text for i in g.submenu.items] for g in tray._voice_menu().items}
    assert langs["Croatian"][-3:] == ["F1 (female)", "M2 (male)", "Anna (custom)"]
    assert len(langs["Croatian"]) == 1 + 10 + 3
    assert langs["Bosnian"] == ["Off", "F1 (female)", "M2 (male)", "Anna (custom)"]  # no Supertonic
    assert langs["English"][-3:] == ["F1 (female)", "M2 (male)", "Anna (custom)"]
    bs = tray._voice_menu().items[2]
    bs.submenu.items[1](tray._icon)
    assert calls == [("voice", "bs", "higgs:F1")]


def test_say_with_a_higgs_voice_and_lang(fakes, tmp_path, capsys):
    cfg = Config(speech=fakes)
    out = str(tmp_path / "o.wav")
    assert app.say(cfg, HR_TEXT, out, voice="higgs:F1", lang="hr") == 1  # not installed
    assert "model_setup --higgs" in capsys.readouterr().err
    install_higgs(fakes)
    assert app.say(cfg, HR_TEXT, out, voice="higgs:F1", lang="hr") == 0
    eng = FakeHiggs.instances[-1]
    assert eng.calls[-1][1] == "higgs:F1" and eng.langs[-1] == "hr"
    assert app.say(cfg, HR_TEXT, out, voice="higgs:F1", lang="ja") == 1
    assert "Higgs does not read Japanese" in capsys.readouterr().err


def test_cli_voice_and_lang_need_say_or_doc():
    with pytest.raises(SystemExit):
        app.main(["--voice", "higgs:F1"])


def test_run_export_with_higgs_route(fakes, tmp_path):
    from vibetts.export import ExportRequest

    install_higgs(fakes)
    engine = app.EngineHolder().get(fakes, voice="higgs:F1")
    req = ExportRequest(HR_TEXT, tmp_path / "x.wav", "higgs:F1", None, "hr", "wav")
    res = app.run_export(engine, fakes, req, tmp_path)
    assert (res.voice, res.lang) == ("higgs:F1", "hr") and res.path.exists()
    req2 = ExportRequest(HR_TEXT, tmp_path / "y.wav", "higgs:Gone", None, "hr", "wav")
    with pytest.raises(RuntimeError, match="Settings > Higgs"):
        app.run_export(engine, fakes, req2, tmp_path)


def test_routing_engine_forwards_cancel_only_to_higgs():
    class Cancellable(FakeHiggs):
        accepts_cancel = True

        def synth(self, text, voice, speed, lang=None, cancel=None):
            self.cancel_seen = cancel
            return super().synth(text, voice, speed, lang)

    hg, k, st = Cancellable("d", "v"), FakeKokoro(), FakeSupertonic("d")
    eng = app.RoutingEngine(k, st, hg)
    assert eng.accepts_cancel is True
    token = threading.Event()
    eng.synth("a", "higgs:F1", 1.0, "hr", cancel=token)
    assert hg.cancel_seen is token
    eng.synth("b", "af_heart", 1.0, None, cancel=token)  # Kokoro's signature is unchanged
    eng.synth("c", "supertonic:F1", 1.0, "hr", cancel=token)
    assert k.calls and st.calls
    # a Higgs engine without cancel support is still called the old way
    plain = app.RoutingEngine(higgs=FakeHiggs("d", "v"))
    plain.synth("a", "higgs:F1", 1.0, "hr", cancel=token)


def test_export_cancel_stops_higgs_mid_chunk(fakes, tmp_path):
    from vibetts.export import ExportCancelled, ExportRequest
    from vibetts.tts import SynthCancelled

    class Slow(FakeHiggs):
        accepts_cancel = True

        def synth(self, text, voice, speed, lang=None, cancel=None):
            while not cancel.is_set():
                time.sleep(0.01)
            raise SynthCancelled()

    install_higgs(fakes)
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    req = ExportRequest(HR_TEXT, tmp_path / "x.wav", "higgs:F1", None, "hr", "wav")
    t0 = time.monotonic()
    with pytest.raises(ExportCancelled):
        app.run_export(app.RoutingEngine(higgs=Slow("d", "v")), fakes, req, tmp_path, cancel=cancel)
    assert time.monotonic() - t0 < 3 and not (tmp_path / "x.wav").exists()
