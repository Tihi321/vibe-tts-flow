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


# ---------------------------------------------------------------- engine

class FakeKokoro:
    def __init__(self, *paths):
        self.paths, self.calls = paths, []

    def voices(self):
        return ["af_heart"]

    def synth(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        return np.ones(10, np.float32)


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr("vibetts.tts.KokoroEngine", FakeKokoro)
    model, voices = tmp_path / "m.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return SpeechConfig(model=str(model), voices=str(voices))


def test_holder_loads_once_and_reloads_on_new_files(fakes, tmp_path):
    s = fakes
    holder = app.EngineHolder()
    assert holder.loaded is None
    engine = holder.get(s)
    assert isinstance(engine, FakeKokoro) and holder.loaded is engine
    assert engine.calls == [("Ready.", "af_heart", 1.0)]  # warm-up
    assert holder.get(s) is engine  # cached, not reloaded
    other = tmp_path / "m2.onnx"
    other.write_bytes(b"x")
    assert holder.get(dataclasses.replace(s, model=str(other))) is not engine


def test_preload_loads_in_the_background(fakes):
    holder = app.EngineHolder()
    holder.preload(fakes)
    for _ in range(100):
        if holder.loaded is not None:
            break
        time.sleep(0.02)
    assert isinstance(holder.loaded, FakeKokoro)


EN_TEXT = "This is plain English text."


class SpyKokoro(FakeKokoro):
    seen: list = []

    def synth(self, text, voice, speed):
        SpyKokoro.seen.append((text, voice))
        return super().synth(text, voice, speed)


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
    assert [r.voice for r in reqs] == ["bf_emma", "af_heart"]  # default: speech.voice
    assert isinstance(a._engine_for_speaker(), FakeKokoro)


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
    assert a.cfg.speech.voice == "bf_emma"
    assert len(a.saved_configs) == 1 and a.tray.updates == 1


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
    tray, calls = make_tray(current="am_adam")
    groups = {g.text: [i.text for i in g.submenu.items] for g in tray._voice_menu().items}
    assert groups == {"American English": ["Heart (female)", "Adam (male)"],
                      "British English": ["Emma (female)"]}
    us = tray._voice_menu().items[0]
    assert [i.checked for i in us.submenu.items] == [False, True]
    us.submenu.items[0](tray._icon)  # picking a voice goes through set_voice
    assert calls == [("voice", "af_heart")]


def test_tray_menu_has_voice_entry():
    tray, _ = make_tray()
    texts = [getattr(i, "text", "") for i in tray._menu_items()]
    assert "Voice" in texts and "Engine" not in texts
    tray, _ = make_tray(voices=[])
    assert "Voice" not in [getattr(i, "text", "") for i in tray._menu_items()]


# ---------------------------------------------------------------- plan_speech / run_export

def _wav_frames(path):
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes()


def test_plan_speech_voice(fakes):
    s = fakes
    chunks, voice = app.plan_speech(s, EN_TEXT)
    assert voice == "af_heart" and chunks
    assert app.plan_speech(s, EN_TEXT, voice="bf_emma")[1] == "bf_emma"


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
    assert b.list_voices() == {
        "voices": [{"id": "af_heart", "label": "Heart (American English, female)"},
                   {"id": "bf_emma", "label": "Emma (British English, female)"}],
        "default_voice": "af_heart"}


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
                   "voice": "af_heart"}
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
    assert out["format"] == "wav" and out["voice"] == "af_heart"
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
