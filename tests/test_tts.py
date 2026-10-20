import time

import numpy as np

from vibetts.textprep import Chunk
from vibetts.tts import (BLOCK_FRAMES, KOKORO_RATE, SAMPLE_RATE, SpeakRequest, Speaker,
                         voice_label, voice_lang, voice_supported)


class FakeEngine:
    def __init__(self, delay=0.0, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on
        self.calls = []
        self.langs = []

    def voices(self):
        return ["af_heart"]

    def synth(self, text, voice, speed, lang=None):
        self.calls.append((text, voice, speed))
        self.langs.append(lang)
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


def req(texts, pause=0.1, volume=1.0, device="spk", lang=None):
    chunks = [Chunk(t, 1.0) for t in texts]
    chunks[-1].pause_after = 0.0
    return SpeakRequest(chunks, "af_heart", 1.1, pause, volume, device, lang)


def test_pipeline_is_44k():
    assert SAMPLE_RATE == 44100 and KOKORO_RATE == 24000 and BLOCK_FRAMES == 4410
    assert BLOCK_FRAMES * 10 == SAMPLE_RATE  # 100 ms blocks


def test_lang_reaches_the_engine():
    engine = FakeEngine()
    make(engine, []).speak_and_wait(req(["one", "two"], lang="hr"), timeout=5)
    assert engine.langs == ["hr", "hr"]
    engine = FakeEngine()
    make(engine, []).speak_and_wait(req(["one"]), timeout=5)
    assert engine.langs == [None] and SpeakRequest([], "af_heart", 1.0, 0.0).lang is None


def test_output_blocks_are_100ms():
    events = []
    sp = make(FakeEngine(), events)
    sp.speak_and_wait(req(["one"], pause=0.0), timeout=5)
    assert {len(b) for b in FakeOutput.instances[0].frames} == {int(0.05 * SAMPLE_RATE)}


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
    assert voice_supported("af_heart") and voice_supported("bm_george")
    # Kokoro reads English only: the other languages go to Supertonic
    assert not voice_supported("ff_siwis") and not voice_supported("ef_dora")
    assert not voice_supported("jf_alpha") and not voice_supported("zm_yunxi")
    assert voice_label("af_heart") == "Heart (American English, female)"
    assert voice_label("bm_george") == "George (British English, male)"


def test_supertonic_voice_ids_and_labels():
    from vibetts.tts import LANG_LABELS, VOICE_LANGS

    assert voice_supported("supertonic:F1") and voice_supported("supertonic:M5")
    assert not voice_supported("supertonic:F6") and not voice_supported("supertonic:")
    assert voice_label("supertonic:F1") == "F1 (Supertonic, female)"
    assert voice_label("supertonic:M3") == "M3 (Supertonic, male)"
    assert sorted(VOICE_LANGS) == sorted(LANG_LABELS) == ["a", "b"]


def test_unsupported_voice_ids():
    assert not voice_supported("ox_foo") and not voice_supported("omni:default")
    assert voice_label("omni:default") == "omni:default"  # unknown ids are shown as is
    assert voice_label("supertonic:Z9") == "supertonic:Z9"


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
    KokoroEngine("m", "v").synth("Hello", "af_heart", 1.0, "hr")  # lang is ignored
    assert seen == {"voice": "af_heart", "lang": "en-us"}


def _fake_kokoro(monkeypatch, samples):
    import sys
    import types

    class K:
        def __init__(self, *a):
            pass

        def create(self, text, voice, speed, lang):
            return samples, 24000

        def get_voices(self):
            return ["af_heart", "ff_siwis", "bf_emma"]

    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=K))


def test_kokoro_resamples_to_44k(monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("soxr")
    from vibetts.tts import KokoroEngine

    tone = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)  # 1 s
    _fake_kokoro(monkeypatch, tone)
    out = KokoroEngine("m", "v").synth("Hello", "af_heart", 1.0)
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - SAMPLE_RATE) <= 2  # 24000 -> 44100 samples per second
    assert 0.9 < np.abs(out[2000:-2000]).max() < 1.1  # the level is kept
    # the pitch is kept: the dominant frequency is still 440 Hz
    spectrum = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spectrum) * SAMPLE_RATE / len(out) - 440) < 2


def test_kokoro_voices_are_english_only(monkeypatch):
    from vibetts.tts import KokoroEngine

    _fake_kokoro(monkeypatch, np.zeros(4, np.float32))
    assert KokoroEngine("m", "v").voices() == ["af_heart", "bf_emma"]


def test_resample_edge_cases():
    from vibetts.tts import resample

    same = np.ones(5, np.float32)
    assert resample(same, SAMPLE_RATE).tolist() == same.tolist()
    assert len(resample(np.zeros(0, np.float32), KOKORO_RATE)) == 0


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
        assert w.getnframes() == 3 and w.getframerate() == SAMPLE_RATE == 44100
        assert np.frombuffer(w.readframes(3), "<i2").tolist() == [0, 16383, -32767]


# ---------------------------------------------------------------- on_chunk, pause, resume

def wait_for(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    return cond()


def tagged_req(texts, pause=0.5, on_chunk=None):
    """One chunk per text (tag = index): 0.05 s of speech + `pause` s of silence, except the last."""
    chunks = [Chunk(t, 1.0, i) for i, t in enumerate(texts)]
    chunks[-1].pause_after = 0.0
    return SpeakRequest(chunks, "af_heart", 1.0, pause, 1.0, "spk", None, on_chunk)


def test_on_chunk_in_order_when_each_chunk_starts():
    seen = []
    events = []
    sp = make(FakeEngine(), events)
    r = tagged_req(["one", "two", "three"], pause=0.0,
                   on_chunk=lambda c: seen.append(
                       (c.text, c.tag, len(FakeOutput.instances[0].frames))))
    sp.speak_and_wait(r, timeout=5)
    assert [s[:2] for s in seen] == [("one", 0), ("two", 1), ("three", 2)]
    assert [s[2] for s in seen] == [0, 1, 2]  # called before the chunk's first write
    assert [e[0] for e in events] == ["start", "done"]


def test_on_chunk_error_does_not_break_playback():
    def boom(chunk):
        raise RuntimeError("callback")

    r = tagged_req(["one", "two"], pause=0.0, on_chunk=boom)
    make(FakeEngine(), []).speak_and_wait(r, timeout=5)
    assert r.done.is_set() and r.error is None and not r.interrupted


def test_pause_stops_writes_and_resume_continues_without_losing_samples():
    events = []
    sp = make(FakeEngine(), events)
    r = tagged_req(["one", "two"])
    total = 2 * int(0.05 * SAMPLE_RATE) + int(0.5 * SAMPLE_RATE)  # the last chunk has no gap
    assert not sp.paused and not sp.pause() and not sp.resume()  # nothing playing
    sp.speak(r)
    assert wait_for(lambda: FakeOutput.instances and FakeOutput.instances[0].frames)
    assert sp.pause() and sp.paused and not sp.pause()
    assert wait_for(lambda: FakeOutput.instances[0].closed)
    first = FakeOutput.instances[0]
    assert first.aborted and not first.drained  # the device is released
    written = sum(len(b) for b in first.frames)
    time.sleep(0.3)
    assert sum(len(b) for b in first.frames) == written and len(FakeOutput.instances) == 1
    assert not r.done.is_set()
    assert sp.resume() and not sp.paused and not sp.resume()
    assert r.done.wait(5) and not r.interrupted and r.error is None
    assert len(FakeOutput.instances) == 2  # reopened for the rest
    assert sum(len(b) for o in FakeOutput.instances for b in o.frames) == total
    assert FakeOutput.instances[1].drained and FakeOutput.instances[1].device == "spk"
    assert [e[0] for e in events] == ["start", "done"]  # on_start once


def test_pause_after_the_request_ended_does_nothing():
    sp = make(FakeEngine(), [])
    r = tagged_req(["one"], pause=0.0)
    sp.speak_and_wait(r, timeout=5)
    assert sp.pause() is False and not sp.paused and len(FakeOutput.instances) == 1


def test_on_chunk_not_repeated_after_resume():
    seen = []
    sp = make(FakeEngine(), [])
    r = tagged_req(["one", "two"], on_chunk=lambda c: seen.append(c.tag))
    sp.speak(r)
    assert wait_for(lambda: FakeOutput.instances and FakeOutput.instances[0].frames)
    assert sp.pause()
    assert wait_for(lambda: FakeOutput.instances[0].closed)
    assert sp.resume()
    assert r.done.wait(5)
    assert seen == [0, 1]


def test_stop_while_paused_releases_and_ends():
    events = []
    sp = make(FakeEngine(), events)
    r = tagged_req(["one", "two"])
    sp.speak(r)
    assert wait_for(lambda: FakeOutput.instances and FakeOutput.instances[0].frames)
    assert sp.pause()
    assert wait_for(lambda: FakeOutput.instances[0].closed)
    sp.stop(wait=3)
    assert not sp.busy and not sp.paused
    assert r.done.is_set() and r.interrupted and r.error is None
    assert events[-1][0] == "done" and events[-1][2] is None
    assert len(FakeOutput.instances) == 1  # not reopened just to stop


def test_new_speak_while_paused_plays_normally():
    sp = make(FakeEngine(), [])
    first = tagged_req(["a", "b"])
    sp.speak(first)
    assert wait_for(lambda: FakeOutput.instances and FakeOutput.instances[0].frames)
    assert sp.pause()
    second = tagged_req(["c"], pause=0.0)
    sp.speak_and_wait(second, timeout=5)
    assert first.done.is_set() and first.interrupted
    assert second.done.is_set() and not second.interrupted and second.error is None
    assert not sp.paused and not sp.resume()


def test_pause_before_the_first_audio_holds_until_resume():
    sp = make(FakeEngine(delay=0.3), [])
    r = tagged_req(["one"], pause=0.0)
    sp.speak(r)
    assert sp.pause()
    time.sleep(0.6)  # the chunk is ready by now
    assert not FakeOutput.instances and not r.done.is_set()
    assert sp.resume()
    assert r.done.wait(5) and not r.interrupted
    assert sum(len(b) for b in FakeOutput.instances[0].frames) == int(0.05 * SAMPLE_RATE)
