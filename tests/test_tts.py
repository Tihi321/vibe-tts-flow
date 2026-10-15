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


def test_omni_voice_helpers():
    assert voice_supported("omni_hr:default") and voice_supported("omni_hr")
    assert not voice_supported("omni_")
    assert voice_label("omni_hr:default") == "Default (Croatian)"
    assert voice_label("omni_hr:ana_marija") == "Ana Marija (Croatian)"
    assert voice_label("omni_hr") == "Default (Croatian)"
    assert voice_label("af_heart") == "Heart (American English, female)"  # unchanged
    assert not voice_supported("ox_foo")
