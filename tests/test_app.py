import dataclasses
import threading
import time
import types
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
    monkeypatch.setattr(app, "_fallback_warned", False)
    model, voices = tmp_path / "m.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return SpeechConfig(model=str(model), voices=str(voices),
                        supertonic_dir=str(tmp_path / "st"))


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
    cfg = Config(speech=dataclasses.replace(fakes, voice="bf_emma"))
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
    assert [r.voice for r in reqs] == ["bf_emma", "af_heart", "supertonic:F2"]  # default: speech.voice
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
    a.enabled = True
    a.flashes = []
    a._flash = lambda kind, text, seconds=1.5: a.flashes.append((kind, text))
    saved = []
    a._save = lambda: saved.append(dataclasses.replace(a.cfg.speech))
    a.saved_configs = saved
    return a


def test_tray_set_voice(fakes):
    a = make_app(fakes)
    a._set_voice("bf_emma")
    assert a.cfg.speech.voice == "bf_emma" and a.cfg.speech.english_engine == "kokoro"
    assert len(a.saved_configs) == 1 and a.tray.updates == 1
    a._set_voice("supertonic:M2")  # a Supertonic voice also switches the English engine
    assert (a.cfg.speech.english_engine, a.cfg.speech.english_supertonic_voice) == ("supertonic", "M2")
    assert a.cfg.speech.voice == "bf_emma"
    a._set_voice("af_heart")
    assert a.cfg.speech.english_engine == "kokoro" and a.cfg.speech.english_supertonic_voice == "M2"


def test_tray_set_other_voice_and_language_mode(fakes):
    a = make_app(fakes)
    a._set_other_voice("supertonic:M3")
    assert a.cfg.speech.other_voice == "M3"
    a._set_other_voice("af_heart")  # not a Supertonic voice: ignored
    assert a.cfg.speech.other_voice == "M3"
    a._set_language_mode("other")
    assert a.cfg.speech.language_mode == "other"
    a._set_language_mode("bogus")
    assert a.cfg.speech.language_mode == "other"
    assert len(a.saved_configs) == 2 and a.tray.updates == 2


# ---------------------------------------------------------------- tray menus

def make_tray(voices=None, current="af_heart"):
    from vibetts.tray import Tray, TrayHooks

    calls = []
    noop = lambda *a: None  # noqa: E731
    hooks = TrayHooks(
        is_enabled=lambda: True, set_enabled=noop, is_speaking=lambda: False, stop=noop,
        model_missing=lambda: False, download_model=noop,
        voices=lambda: voices if voices is not None else ["af_heart", "am_adam", "bf_emma"],
        current_voice=lambda: current, set_voice=lambda v: calls.append(("voice", v)),
        current_speed=lambda: 1.0, set_speed=noop, is_autostart=lambda: False,
        set_autostart=noop, open_settings=noop, open_config=noop, open_logs=noop,
        reload_config=noop, quit=noop)
    return Tray(hooks), calls


def test_tray_voices_grouped_by_language():
    tray, calls = make_tray(current="am_adam", voices=["af_heart", "am_adam", "bf_emma",
                                                       "supertonic:F1", "supertonic:M1"])
    groups = {g.text: [i.text for i in g.submenu.items] for g in tray._voice_menu().items}
    assert groups == {"American English": ["Heart (female)", "Adam (male)"],
                      "British English": ["Emma (female)"],
                      "Supertonic": ["F1 (female)", "M1 (male)"]}
    us = tray._voice_menu().items[0]
    assert [i.checked for i in us.submenu.items] == [False, True]
    us.submenu.items[0](tray._icon)  # picking a voice goes through set_voice
    assert calls == [("voice", "af_heart")]


def test_tray_menu_has_voice_entry():
    tray, _ = make_tray()
    texts = [getattr(i, "text", "") for i in tray._menu_items()]
    assert "English voice" in texts and "Language" in texts and "Speed" in texts
    assert "Croatian voice" not in texts  # Supertonic not installed: no other-language voices
    tray, _ = make_tray(voices=[])
    texts = [getattr(i, "text", "") for i in tray._menu_items()]
    assert "English voice" not in texts and "Language" in texts


def make_profile_tray(mode="auto", other="supertonic:F1", english="af_heart"):
    from vibetts.tray import Tray, TrayHooks

    calls = []
    noop = lambda *a: None  # noqa: E731
    hooks = TrayHooks(
        is_enabled=lambda: True, set_enabled=noop, is_speaking=lambda: False, stop=noop,
        model_missing=lambda: False, download_model=noop,
        voices=lambda: ["af_heart", "bf_emma"] + app.supertonic_voice_ids(),
        current_voice=lambda: english, set_voice=lambda v: calls.append(("voice", v)),
        current_speed=lambda: 1.0, set_speed=noop, is_autostart=lambda: False,
        set_autostart=noop, open_settings=noop, open_config=noop, open_logs=noop,
        reload_config=noop, quit=noop,
        current_language_mode=lambda: mode, set_language_mode=lambda m: calls.append(("mode", m)),
        other_language_name=lambda: "Croatian", supertonic_available=lambda: True,
        current_other_voice=lambda: other, set_other_voice=lambda v: calls.append(("other", v)))
    return Tray(hooks), calls


def test_tray_language_menu():
    tray, calls = make_profile_tray(mode="other")
    items = {i.text: i for i in tray._menu_items() if hasattr(i, "submenu") and i.submenu}
    lang = items["Language"].submenu.items
    assert [i.text for i in lang] == ["Auto (detect English)", "English", "Croatian"]
    assert [i.checked for i in lang] == [False, False, True]
    lang[1](tray._icon)
    assert calls == [("mode", "english")]


def test_tray_other_voice_menu():
    tray, calls = make_profile_tray(other="supertonic:M2")
    items = {i.text: i for i in tray._menu_items() if hasattr(i, "submenu") and i.submenu}
    other = items["Croatian voice"].submenu.items
    assert [i.text for i in other][:2] == ["F1 (female)", "F2 (female)"] and len(other) == 10
    assert [i.checked for i in other] == [v.endswith("M2") for v in app.supertonic_voice_ids()]
    other[5](tray._icon)
    assert calls == [("other", "supertonic:M1")]
    eng = items["English voice"].submenu.items
    assert [g.text for g in eng] == ["American English", "British English", "Supertonic"]


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
    assert [l["code"] for l in out["languages"]][:3] == ["en", "ko", "ja"] and len(out["languages"]) == 31
    assert {"code": "hr", "name": "Croatian"} in out["languages"]
    assert out["english"] == {"engine": "kokoro", "voice": "af_heart"}
    assert out["other"] == {"language": "hr", "voice": "supertonic:F1"}
    assert out["language_mode"] == "auto" and out["supertonic_installed"] is False


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


@pytest.mark.parametrize("mode,text,engine,expected", [
    ("auto", EN_LONG, "kokoro", ("kokoro", "af_heart", "en")),
    ("auto", HR_TEXT, "kokoro", ("supertonic", "supertonic:F2", "hr")),
    ("auto", EN_LONG, "supertonic", ("supertonic", "supertonic:M1", "en")),
    ("auto", HR_TEXT, "supertonic", ("supertonic", "supertonic:F2", "hr")),
    ("english", HR_TEXT, "kokoro", ("kokoro", "af_heart", "en")),
    ("english", HR_TEXT, "supertonic", ("supertonic", "supertonic:M1", "en")),
    ("other", EN_LONG, "kokoro", ("supertonic", "supertonic:F2", "hr")),
    ("other", EN_LONG, "supertonic", ("supertonic", "supertonic:F2", "hr")),
])
def test_route_matrix(fakes, mode, text, engine, expected):
    s = dataclasses.replace(fakes, language_mode=mode, english_engine=engine,
                            english_supertonic_voice="M1", other_voice="F2")
    install_supertonic(s)
    route = app.route_for(s, text)
    assert tuple(route[:3]) == expected and route.fell_back is False


def test_route_other_language_and_explicit_lang(fakes):
    s = dataclasses.replace(fakes, other_language="de", other_voice="M4")
    install_supertonic(s)
    assert app.route_for(s, "Das ist ein einfacher Test des Systems, der den Text vorliest, schön.")[::2] \
        == ("supertonic", "de")
    # an explicit lang: "en" is the English profile, anything else the other-language voice
    assert app.route_for(s, HR_TEXT, "en") == app.Route("kokoro", "af_heart", "en")
    assert app.route_for(s, EN_LONG, "fr") == app.Route("supertonic", "supertonic:M4", "fr")


def test_route_falls_back_to_kokoro_when_supertonic_is_missing(fakes, caplog):
    s = fakes  # not installed
    route = app.route_for(s, HR_TEXT)
    assert route == app.Route("kokoro", "af_heart", "en", True)
    s2 = dataclasses.replace(s, english_engine="supertonic")
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


def test_active_voices_per_mode(fakes):
    s = dataclasses.replace(fakes, english_engine="supertonic", english_supertonic_voice="F5")
    assert app.needed_voices(s) == ["supertonic:F5", "supertonic:F1"]
    assert app.needed_voices(dataclasses.replace(s, language_mode="english")) == ["supertonic:F5"]
    assert app.needed_voices(dataclasses.replace(s, language_mode="other")) == ["supertonic:F1"]
    assert app.primary_voice(dataclasses.replace(s, language_mode="other")) == "supertonic:F1"
    assert app.primary_voice(fakes) == "af_heart"
    assert app.missing_voices(fakes) == ["supertonic:F1"]
    install_supertonic(fakes)
    assert app.missing_voices(fakes) == []


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
    s = dataclasses.replace(fakes, language_mode="other")
    install_supertonic(s)
    engine = app.EngineHolder().get(s)  # no voice: the primary one (other-language profile)
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
    holder.preload(s)  # auto: English (Kokoro) and other language (Supertonic)
    loaded = _wait_loaded(holder, kokoro=True, supertonic=True)
    assert isinstance(loaded.kokoro, FakeKokoro) and isinstance(loaded.supertonic, FakeSupertonic)
    only_en = app.EngineHolder()
    only_en.preload(dataclasses.replace(s, language_mode="english"))
    _wait_loaded(only_en, kokoro=True)
    assert only_en.loaded.supertonic is None
    only_other = app.EngineHolder()
    only_other.preload(dataclasses.replace(s, language_mode="other"))
    _wait_loaded(only_other, supertonic=True)
    assert only_other.loaded.kokoro is None


def test_preload_without_supertonic_loads_kokoro_only(fakes):
    holder = app.EngineHolder()
    holder.preload(fakes)  # the other profile falls back to Kokoro English
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
    a.cfg.speech.other_language, a.cfg.speech.language_mode = "de", "other"
    out = b.list_voices()
    st = out["voices"][1]
    assert st == {"id": "supertonic:F1", "label": "F1 (Supertonic, female)", "engine": "supertonic",
                  "languages": list(app.SUPERTONIC_LANGUAGES)}
    assert out["supertonic_installed"] is True and out["language_mode"] == "other"
    assert out["other"] == {"language": "de", "voice": "supertonic:F1"}


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

def test_voices_lists_english_kokoro_then_supertonic_when_installed(fakes, tmp_path):
    a = make_app(fakes)
    with open(a.cfg.speech.voices, "wb") as fh:
        np.savez(fh, af_heart=np.zeros(2), ff_siwis=np.zeros(2), bm_george=np.zeros(2))
    assert a._voices() == ["af_heart", "bm_george"]
    install_supertonic(a.cfg.speech)
    assert a._voices() == ["af_heart", "bm_george", *app.supertonic_voice_ids()]


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
    new = Config(speech=dataclasses.replace(base, english_engine="supertonic"))
    assert a.apply_config(new) == app.MISSING_SUPERTONIC  # the English profile needs it
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
    assert opened == [("supertonic", True)]  # Kokoro is there, Supertonic (other profile) is not
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
    cfg.speech.language_mode = "english"
    assert app.say(cfg, HR_TEXT, str(tmp_path / "forced.wav")) == 0
    assert "voice af_heart" in capsys.readouterr().out


def test_say_without_supertonic_falls_back_and_says_so(fakes, tmp_path, capsys):
    cfg = Config(speech=dataclasses.replace(fakes, sentence_pause=0.0))
    assert app.say(cfg, HR_TEXT, str(tmp_path / "fb.wav")) == 0
    captured = capsys.readouterr()
    assert "Supertonic is not installed" in captured.err and "voice af_heart" in captured.out


def test_say_supertonic_only_missing_message(fakes, tmp_path, capsys):
    cfg = Config(speech=dataclasses.replace(fakes, english_engine="supertonic", language_mode="english"))
    cfg.speech.model = str(tmp_path / "nope.onnx")  # even Kokoro is gone: the fallback fails too
    assert app.say(cfg, EN_LONG, str(tmp_path / "x.wav")) == 1
    assert "model files missing" in capsys.readouterr().err
