import dataclasses
import threading
import time
import types

import pytest

from vibetts.app import MISSING_MODEL
from vibetts.config import SpeechConfig
from vibetts.reading import ReadingError, ReadingSession


class FakeReq:
    def __init__(self, chunks, on_chunk):
        self.chunks = chunks
        self.on_chunk = on_chunk
        self.done = threading.Event()
        self.error = None
        self.interrupted = False

    def end(self, error=None, interrupted=False):
        self.error, self.interrupted = error, interrupted
        self.done.set()


class FakeApp:
    """Stands in for TTSApp: one playback slot, requests are ended by the test."""

    def __init__(self, s):
        self.cfg = types.SimpleNamespace(speech=s)
        self.session = None  # set by the test: pause/resume tell it, like TTSApp does
        self.calls = []
        self.current = None
        self.paused = False
        self.stops = 0

    def speak(self, chunks, s, voice=None, lang=None, speed=None, on_chunk=None):
        if self.current is not None and not self.current.done.is_set():
            self.current.end(interrupted=True)  # replaced, like Speaker.speak
        self.calls.append(types.SimpleNamespace(chunks=chunks, voice=voice, lang=lang,
                                                speed=speed))
        self.paused = False
        self.current = FakeReq(chunks, on_chunk)
        return self.current

    def play(self, k):
        """The player starts the k-th chunk of the current request."""
        self.current.on_chunk(self.current.chunks[k])

    def pause(self):
        self.paused = True
        self.session.notify_paused(True)
        return True

    def resume(self):
        self.paused = False
        self.session.notify_paused(False)
        return True

    def stop(self):
        self.stops += 1
        if self.current is not None:
            self.current.end(interrupted=True)


@pytest.fixture
def speech(tmp_path):
    model, voices = tmp_path / "m.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return SpeechConfig(model=str(model), voices=str(voices),
                        supertonic_dir=str(tmp_path / "st"), speed=1.2)


@pytest.fixture
def app(speech):
    return FakeApp(speech)


@pytest.fixture
def rs(app):
    session = ReadingSession(app)
    app.session = session
    return session


ITEMS = [{"text": "First sentence here."}, {"text": "Second sentence here.", "pause": "paragraph"},
         {"text": "Third sentence here."}]


def wait_state(rs, state, timeout=3.0):
    deadline = time.monotonic() + timeout
    while rs.status()["state"] != state and time.monotonic() < deadline:
        time.sleep(0.005)
    return rs.status()


def test_initial_status(rs):
    st = rs.status()
    assert st == {"version": 0, "session": None, "source": None, "meta": {}, "state": "idle",
                  "item": 0, "count": 0, "error": None, "voice": None, "lang": None,
                  "speed": 1.0}


def test_start_plays_and_reports_progress(rs, app):
    res = rs.start(ITEMS, source="extension", meta={"url": "u"})
    assert res["count"] == 3 and len(res["session"]) == 32
    st = rs.status()
    assert (st["state"], st["item"], st["count"]) == ("loading", 0, 3)
    assert st["session"] == res["session"] and st["source"] == "extension"
    assert st["meta"] == {"url": "u"} and st["voice"] == "af_heart" and st["lang"] == "en"
    assert st["speed"] == 1.2  # the configured one
    call = app.calls[0]
    assert [c.tag for c in call.chunks] == [0, 1, 2] and call.voice == "af_heart"
    assert call.speed is None  # TTSApp.speak falls back to the configured speed
    versions = [st["version"]]
    for k in range(3):
        app.play(k)
        st = rs.status()
        assert (st["state"], st["item"]) == ("playing", k)
        versions.append(st["version"])
    assert versions == sorted(set(versions))  # every change bumps the version
    app.current.end()
    st = wait_state(rs, "finished")
    assert st["state"] == "finished" and st["item"] == 2 and st["error"] is None


def test_start_from_an_item_speaks_only_the_rest(rs, app):
    rs.start(ITEMS, start=1, speed=1.5, voice="bm_george")
    assert [c.tag for c in app.calls[0].chunks] == [1, 2]
    assert app.calls[0].speed == 1.5 and app.calls[0].voice == "bm_george"
    st = rs.status()
    assert st["item"] == 1 and st["speed"] == 1.5 and st["voice"] == "bm_george"


def test_new_session_id_per_start(rs, app):
    a = rs.start(ITEMS)["session"]
    b = rs.start(ITEMS)["session"]
    assert a != b and rs.status()["session"] == b


def test_seek_keeps_the_session(rs, app):
    sid = rs.start(ITEMS, speed=1.5)["session"]
    app.play(0)
    first = app.current
    st = rs.seek(2)
    assert st["session"] == sid and st["state"] == "loading" and st["item"] == 2
    assert [c.tag for c in app.calls[1].chunks] == [2] and app.calls[1].speed == 1.5
    assert first.done.is_set()  # replaced by the seek
    time.sleep(0.05)
    assert rs.status()["state"] == "loading"  # the old request's end is not ours
    app.play(0)
    assert rs.status()["item"] == 2 and rs.status()["state"] == "playing"


def test_late_chunk_of_the_old_request_is_ignored(rs, app):
    rs.start(ITEMS)
    old = app.current
    rs.seek(1)
    old.on_chunk(old.chunks[0])  # a straggler from the player thread of the old request
    assert rs.status()["item"] == 1 and rs.status()["state"] == "loading"


def test_seek_after_finished_restarts(rs, app):
    sid = rs.start(ITEMS)["session"]
    app.current.end()
    wait_state(rs, "finished")
    st = rs.seek(0)
    assert st["session"] == sid and st["state"] == "loading"


def test_seek_validation(rs, app):
    with pytest.raises(ReadingError, match="session"):
        rs.seek(0)
    rs.start(ITEMS)
    for bad in (-1, 3, "1", True, None):
        with pytest.raises(ReadingError):
            rs.seek(bad)


def test_pause_and_resume(rs, app):
    rs.start(ITEMS)
    app.play(0)
    v = rs.status()["version"]
    assert rs.pause()["state"] == "paused" and app.paused
    assert rs.status()["version"] > v and rs.status()["item"] == 0
    assert rs.resume()["state"] == "playing" and not app.paused
    # a pause before the first audio goes back to loading
    rs.start(ITEMS)
    assert rs.pause()["state"] == "paused"
    assert rs.resume()["state"] == "loading"


def test_pause_resume_from_the_tray_or_hotkey(rs, app):
    rs.start(ITEMS)
    app.play(1)
    rs.notify_paused(True)  # TTSApp.pause() called by something else
    assert rs.status()["state"] == "paused"
    rs.notify_paused(False)
    assert rs.status()["state"] == "playing" and rs.status()["item"] == 1


def test_pause_and_resume_do_nothing_without_a_live_session(rs, app):
    assert rs.pause()["state"] == "idle" and not app.paused
    assert rs.resume()["state"] == "idle"
    rs.notify_paused(True)  # e.g. a hotkey read was paused: not ours
    assert rs.status()["state"] == "idle" and rs.status()["version"] == 0
    rs.start(ITEMS)
    app.current.end()
    wait_state(rs, "finished")
    assert rs.pause()["state"] == "finished"


def test_stop(rs, app):
    rs.start(ITEMS)
    app.play(0)
    st = rs.stop()
    assert st["state"] == "stopped" and app.stops == 1 and st["session"] is not None
    time.sleep(0.05)
    assert rs.status()["state"] == "stopped"
    assert rs.stop()["state"] == "stopped" and app.stops == 1  # nothing left to stop


def test_stop_while_paused(rs, app):
    rs.start(ITEMS)
    app.play(0)
    rs.pause()
    assert rs.stop()["state"] == "stopped" and app.stops == 1


def test_stop_does_not_touch_foreign_playback(rs, app):
    rs.start(ITEMS)
    app.current.end()
    wait_state(rs, "finished")
    rs.stop()
    assert app.stops == 0 and rs.status()["state"] == "finished"


def test_replaced_by_another_speak_ends_stopped(rs, app):
    rs.start(ITEMS)
    app.play(0)
    app.speak([], app.cfg.speech)  # hotkey read / MCP speak takes the slot
    st = wait_state(rs, "stopped")
    assert st["state"] == "stopped" and st["session"] is not None and st["error"] is None


def test_synthesis_error(rs, app):
    rs.start(ITEMS)
    app.current.end(error="boom")
    st = wait_state(rs, "error")
    assert st["state"] == "error" and st["error"] == "boom"


def test_speak_raising_gives_error_state(rs, app):
    def broken(*a, **k):
        raise RuntimeError("no device")

    app.speak = broken
    rs.start(ITEMS)
    st = rs.status()
    assert st["state"] == "error" and st["error"] == "no device"


def test_status_meta_is_a_copy(rs):
    rs.start(ITEMS, meta={"a": 1})
    rs.status()["meta"]["a"] = 2
    assert rs.status()["meta"] == {"a": 1}


# ---------------------------------------------------------------- long-poll

def test_wait_returns_at_once_when_stale_or_without_since(rs):
    rs.start(ITEMS)
    t0 = time.monotonic()
    assert rs.wait(None, 5)["version"] == rs.status()["version"]
    assert rs.wait(0, 5)["version"] == rs.status()["version"]
    assert time.monotonic() - t0 < 1


def test_wait_times_out_with_the_unchanged_status(rs):
    rs.start(ITEMS)
    since = rs.status()["version"]
    t0 = time.monotonic()
    st = rs.wait(since, 0.15)
    assert st["version"] == since and 0.1 <= time.monotonic() - t0 < 2


def test_wait_wakes_on_a_change(rs, app):
    rs.start(ITEMS)
    since = rs.status()["version"]
    threading.Timer(0.1, lambda: app.play(1)).start()
    t0 = time.monotonic()
    st = rs.wait(since, 5)
    assert st["version"] > since and st["item"] == 1 and time.monotonic() - t0 < 2


def test_wait_wakes_on_end(rs, app):
    rs.start(ITEMS)
    app.play(0)
    since = rs.status()["version"]
    threading.Timer(0.1, app.current.end).start()
    assert rs.wait(since, 5)["state"] == "finished"


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("kwargs", [
    {"items": []},
    {"items": "text"},
    {"items": [{"text": 5}]},
    {"items": ["text"]},
    {"items": [{"pause": "sentence"}]},
    {"items": [{"text": "Hello there.", "pause": "long"}]},
    {"items": [{"text": "Hello there."}], "start": 1},
    {"items": [{"text": "Hello there."}], "start": -1},
    {"items": [{"text": "Hello there."}], "start": "0"},
    {"items": [{"text": "Hello there."}], "speed": 0.4},
    {"items": [{"text": "Hello there."}], "speed": 2.1},
    {"items": [{"text": "Hello there."}], "speed": "fast"},
    {"items": [{"text": "Hello there."}], "source": "toaster"},
    {"items": [{"text": "Hello there."}], "meta": []},
    {"items": [{"text": "Hello there."}], "lang": "xx"},
    {"items": [{"text": "Hello there."}], "voice": 3},
    {"items": [{"text": "Hello there."}], "voice": "af_heart", "lang": "hr"},
    {"items": [{"text": "Hello there."}], "max_chars": 5},
    {"items": [{"text": "---"}, {"text": "..."}]},  # nothing speakable
    {"items": [{"text": "Hello there."}, {"text": "..."}], "start": 1},  # nothing from there
])
def test_bad_requests(rs, app, kwargs):
    with pytest.raises(ReadingError):
        rs.start(**kwargs)
    assert app.calls == [] and rs.status()["version"] == 0  # nothing started, nothing changed


def test_too_many_items(rs):
    with pytest.raises(ReadingError, match="Too many"):
        rs.start([{"text": "a"}] * 20_001)


def test_speed_limits_are_inclusive(rs, app):
    rs.start(ITEMS, speed=0.5)
    rs.start(ITEMS, speed=2)
    assert [c.speed for c in app.calls] == [0.5, 2.0]


def test_reading_error_is_a_value_error():
    assert issubclass(ReadingError, ValueError)


def test_missing_voice_model(speech, tmp_path):
    app = FakeApp(dataclasses.replace(speech, model=str(tmp_path / "nope.onnx")))
    rs = ReadingSession(app)
    with pytest.raises(RuntimeError, match=MISSING_MODEL):
        rs.start(ITEMS)
    assert app.calls == [] and rs.status()["state"] == "idle"


def test_explicit_supertonic_voice_without_its_model(rs, app):
    # an explicit Supertonic voice whose model is not installed is a missing voice (503)
    with pytest.raises(RuntimeError):
        rs.start(ITEMS, voice="supertonic:F1")


def test_texts_are_not_logged(rs, caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        rs.start([{"text": "Secret sentence number one."}])
    assert "Secret" not in caplog.text
