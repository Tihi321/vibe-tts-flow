import time

import numpy as np

from vibetts.textprep import Chunk
from vibetts.tts import (SAMPLE_RATE, SpeakRequest, Speaker, voice_label, voice_lang,
                         voice_supported)


class FakeEngine:
    def __init__(self, delay=0.0, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on
        self.calls = []

    def voices(self):
        return ["af_heart"]

    def synth(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        if text == self.fail_on:
            raise RuntimeError("boom")
        time.sleep(self.delay)
        return np.full(int(0.05 * SAMPLE_RATE), 0.5, dtype=np.float32)


class FakeOutput:
    instances: list = []

    def __init__(self, device=""):
        self.device = device
        self.frames = []
        self.aborted = self.drained = self.closed = False
        FakeOutput.instances.append(self)

    def write(self, block):
        self.frames.append(block.copy())
        time.sleep(0.001)

    def drain(self):
        self.drained = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


def make(engine, events):
    FakeOutput.instances = []
    return Speaker(lambda: engine, FakeOutput,
                   on_start=lambda req: events.append(("start", req)),
                   on_done=lambda req, err: events.append(("done", req, err)))


def req(texts, pause=0.1, volume=1.0, device="spk"):
    chunks = [Chunk(t, 1.0) for t in texts]
    chunks[-1].pause_after = 0.0
    return SpeakRequest(chunks, "af_heart", 1.1, pause, volume, device)


def test_plays_all_chunks_with_gaps_and_volume():
    events = []
    engine = FakeEngine()
    sp = make(engine, events)
    r = req(["one", "two"], pause=0.1, volume=0.5)
    sp.speak_and_wait(r, timeout=5)
    assert [c[0] for c in engine.calls] == ["one", "two"]
    assert engine.calls[0][1:] == ("af_heart", 1.1)
    out = FakeOutput.instances[0]
    audio = np.concatenate(out.frames)
    # chunk (0.05 s) + gap (0.1 s) + chunk (0.05 s)
    assert len(audio) == int(0.05 * SAMPLE_RATE) * 2 + int(0.1 * SAMPLE_RATE)
    assert np.isclose(audio.max(), 0.25)
    assert out.device == "spk" and out.drained and out.closed and not out.aborted
    assert [e[0] for e in events] == ["start", "done"]
    assert events[1] == ("done", r, None)
    assert not sp.busy


def test_stop_interrupts():
    events = []
    sp = make(FakeEngine(delay=0.2), events)
    sp.speak(req([f"s{i}" for i in range(20)]))
    deadline = time.monotonic() + 3
    while not events and time.monotonic() < deadline:
        time.sleep(0.01)
    sp.stop(wait=3)
    assert not sp.busy
    assert events[-1][0] == "done" and events[-1][2] is None
    assert FakeOutput.instances[0].aborted


def test_synth_error_reported():
    events = []
    sp = make(FakeEngine(fail_on="bad"), events)
    sp.speak_and_wait(req(["good", "bad", "never"]), timeout=5)
    assert events[-1][0] == "done"
    assert "boom" in events[-1][2]


def test_engine_load_error_reported():
    events = []

    def broken():
        raise RuntimeError("no model")

    sp = Speaker(broken, FakeOutput, on_done=lambda req, err: events.append(err))
    sp.speak_and_wait(req(["x"]), timeout=5)
    assert events == ["no model"]


def test_new_request_replaces_old():
    events = []
    sp = make(FakeEngine(delay=0.1), events)
    first = req([f"a{i}" for i in range(20)])
    second = req(["b"])
    sp.speak(first)
    time.sleep(0.2)
    sp.speak_and_wait(second, timeout=5)
    dones = [e for e in events if e[0] == "done"]
    assert dones[0][1] is first and dones[-1][1] is second


def test_voice_helpers():
    assert voice_lang("af_heart") == "en-us"
    assert voice_lang("bm_george") == "en-gb"
    assert voice_supported("af_heart") and voice_supported("ff_siwis")
    assert not voice_supported("jf_alpha") and not voice_supported("zm_yunxi")
    assert voice_label("af_heart") == "Heart (American English, female)"
    assert voice_label("bm_george") == "George (British English, male)"


def test_unsupported_voice_ids():
    assert not voice_supported("ox_foo") and not voice_supported("omni:default")
    assert voice_label("omni:default") == "omni:default"  # unknown ids are shown as is


def test_kokoro_uses_the_voice_language(monkeypatch):
    import sys
    import types

    from vibetts.tts import KokoroEngine

    seen = {}

    class K:
        def __init__(self, *a):
            pass

        def create(self, text, voice, speed, lang):
            seen.update(voice=voice, lang=lang)
            return np.zeros(4, np.float32), 24000

    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=K))
    KokoroEngine("m", "v").synth("Hello", "bf_emma", 1.0)
    assert seen == {"voice": "bf_emma", "lang": "en-gb"}


# ---------------------------------------------------------------- request completion state

def test_request_done_after_normal_play():
    sp = make(FakeEngine(), [])
    r = req(["one", "two"])
    assert not r.done.is_set()
    sp.speak_and_wait(r, timeout=5)
    assert r.done.is_set() and r.error is None and not r.interrupted


def test_request_interrupted_by_stop():
    events = []
    sp = make(FakeEngine(delay=0.2), events)
    r = req([f"s{i}" for i in range(20)])
    sp.speak(r)
    deadline = time.monotonic() + 3
    while not events and time.monotonic() < deadline:
        time.sleep(0.01)
    sp.stop(wait=3)
    assert r.done.wait(3) and r.interrupted and r.error is None


def test_request_error_is_recorded():
    sp = make(FakeEngine(fail_on="bad"), [])
    r = req(["good", "bad"])
    sp.speak_and_wait(r, timeout=5)
    assert r.done.is_set() and "boom" in r.error and not r.interrupted

    def broken():
        raise RuntimeError("no model")

    r2 = req(["x"])
    Speaker(broken, FakeOutput).speak_and_wait(r2, timeout=5)
    assert r2.done.is_set() and r2.error == "no model"


def test_replaced_request_is_done_and_interrupted():
    sp = make(FakeEngine(delay=0.1), [])
    first, second = req([f"a{i}" for i in range(20)]), req(["b"])
    sp.speak(first)
    time.sleep(0.2)
    sp.speak_and_wait(second, timeout=5)
    assert first.done.is_set() and first.interrupted
    assert second.done.is_set() and not second.interrupted


def test_write_wav_roundtrip(tmp_path):
    import wave

    from vibetts.tts import write_wav

    path = tmp_path / "a.wav"
    write_wav(path, np.array([0.0, 0.5, -2.0], np.float32))
    with wave.open(str(path)) as w:
        assert w.getnframes() == 3 and w.getframerate() == SAMPLE_RATE
        assert np.frombuffer(w.readframes(3), "<i2").tolist() == [0, 16383, -32767]
