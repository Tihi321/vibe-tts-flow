import os
import re
import threading
import wave

import numpy as np
import pytest

from vibetts import export
from vibetts.config import MP3_BITRATES
from vibetts.export import (ExportCancelled, ExportProgress, ExportRequest, ExportResult,
                            format_for, mp3_available, pcm16, render, resolve_output)
from vibetts.textprep import Chunk
from vibetts.tts import SAMPLE_RATE


def fake_synth(n=1000, value=0.5):
    def synth(text):
        return np.full(n, value, dtype=np.float32)
    return synth


def chunks3():
    return [Chunk("a", 1.0), Chunk("b", 0.0), Chunk("c", 2.0)]


def test_pcm16_clips_and_scales():
    data = np.frombuffer(pcm16(np.array([0.0, 0.5, 2.0, -3.0], dtype=np.float32)), dtype="<i2")
    assert list(data) == [0, int(0.5 * 32767), 32767, -32767]


def test_wav_roundtrip_with_gain(tmp_path):
    out = tmp_path / "x.wav"
    seconds, size = render(chunks3(), fake_synth(1000), out, "wav", pause=0.01, gain=0.5)
    gaps = sum(int(0.01 * c.pause_after * SAMPLE_RATE) for c in chunks3())
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE == 44100
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 3000 + gaps
        first = np.frombuffer(w.readframes(1000), dtype="<i2")
    assert (first == int(0.25 * 32767)).all()
    assert seconds == pytest.approx((3000 + gaps) / SAMPLE_RATE)
    assert size == out.stat().st_size
    assert not out.with_name("x.wav.part").exists()


def test_progress_and_before_chunk(tmp_path):
    events, calls = [], []
    render(chunks3(), fake_synth(), tmp_path / "x.wav", "wav", pause=0.01,
           progress=events.append, before_chunk=lambda: calls.append(1))
    assert [e.done for e in events] == [0, 1, 2, 3]
    assert all(isinstance(e, ExportProgress) and e.total == 3 for e in events)
    secs = [e.audio_seconds for e in events]
    assert secs[0] == 0.0 and secs == sorted(secs) and secs[-1] > secs[1]
    assert len(calls) == 3


def test_cancel_cleans_up(tmp_path):
    out = tmp_path / "x.wav"
    cancel = threading.Event()
    calls = []

    def before():
        calls.append(1)
        if len(calls) == 2:
            cancel.set()

    with pytest.raises(ExportCancelled):
        render(chunks3(), fake_synth(), out, "wav", pause=0.0, cancel=cancel, before_chunk=before)
    assert not out.exists()
    assert not out.with_name("x.wav.part").exists()


def test_synth_error_cleans_up_and_keeps_existing(tmp_path):
    out = tmp_path / "x.wav"
    out.write_bytes(b"old")

    def synth(text):
        if text == "b":
            raise RuntimeError("boom")
        return np.zeros(100, dtype=np.float32)

    with pytest.raises(RuntimeError, match="boom"):
        render(chunks3(), synth, out, "wav", pause=0.0)
    assert out.read_bytes() == b"old"
    assert not out.with_name("x.wav.part").exists()


def test_existing_replaced_only_at_end(tmp_path, monkeypatch):
    out = tmp_path / "x.wav"
    out.write_bytes(b"old")
    seen = []

    def synth(text):
        seen.append(out.read_bytes())
        return np.zeros(100, dtype=np.float32)

    render(chunks3(), synth, out, "wav", pause=0.0)
    assert seen == [b"old"] * 3
    assert out.read_bytes() != b"old"


def test_replace_failure_cleans_up(tmp_path, monkeypatch):
    out = tmp_path / "x.wav"

    def boom(a, b):
        raise OSError("locked")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        render(chunks3(), fake_synth(), out, "wav", pause=0.0)
    assert not out.exists() and not out.with_name("x.wav.part").exists()


def test_empty_chunks(tmp_path):
    with pytest.raises(ValueError, match="nothing to export"):
        render([], fake_synth(), tmp_path / "x.wav", "wav", pause=0.0)


def test_format_for(tmp_path):
    assert format_for("a.WAV") == "wav"
    assert format_for(tmp_path / "a.mp3") == "mp3"
    assert format_for("a", "mp3") == "mp3"
    with pytest.raises(ValueError):
        format_for("a")
    with pytest.raises(ValueError, match="Only .wav and .mp3"):
        format_for("a.ogg", "wav")


def test_resolve_title_and_timestamp(tmp_path):
    folder = tmp_path / "exports"
    p = resolve_output(None, folder, 'My: "Doc"/1?.', "mp3")
    assert p == folder / "My Doc1.mp3"
    assert folder.is_dir()
    p = resolve_output("", folder, "", "wav")
    assert re.fullmatch(r"vibetts-\d{8}-\d{6}\.wav", p.name)
    assert p.parent == folder


def test_resolve_names_and_paths(tmp_path):
    folder = tmp_path / "exports"
    assert resolve_output("note", folder, "", "mp3") == folder / "note.mp3"
    assert resolve_output("sub/dir/note.wav", folder, "", "wav") == folder / "sub" / "dir" / "note.wav"
    assert (folder / "sub" / "dir").is_dir()
    other = tmp_path / "other"
    other.mkdir()
    assert resolve_output(other / "n.mp3", folder, "", "wav") == other / "n.mp3"
    assert resolve_output(other / "n", folder, "", "mp3") == other / "n.mp3"


def test_resolve_errors(tmp_path):
    folder = tmp_path / "exports"
    with pytest.raises(ValueError):
        resolve_output("a.ogg", folder, "", "wav")
    existing = tmp_path / "e.wav"
    existing.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        resolve_output(existing, folder, "", "wav")
    assert resolve_output(existing, folder, "", "wav", overwrite=True) == existing
    with pytest.raises(FileNotFoundError):
        resolve_output(tmp_path / "missing" / "a.wav", folder, "", "wav")


def _first_mp3_frame(data: bytes) -> tuple[int, int, int]:
    """(MPEG version bits, layer bits, sample-rate index) of the first frame header."""
    if data[:3] == b"ID3":  # skip an ID3v2 tag
        data = data[10 + (data[6] << 21 | data[7] << 14 | data[8] << 7 | data[9]):]
    i = next(i for i in range(len(data) - 3) if data[i] == 0xFF and data[i + 1] & 0xE0 == 0xE0)
    return (data[i + 1] >> 3) & 3, (data[i + 1] >> 1) & 3, (data[i + 2] >> 2) & 3


def test_mp3_render(tmp_path):
    pytest.importorskip("lameenc")
    assert mp3_available() is True
    out = tmp_path / "x.mp3"
    seconds, size = render(chunks3(), fake_synth(SAMPLE_RATE, 0.3), out, "mp3", pause=0.1,
                           bitrate=64)
    b = out.read_bytes()
    assert len(b) > 0 and size == len(b)
    assert b[:3] == b"ID3" or (b[0] == 0xFF and b[1] & 0xE0 == 0xE0)
    assert seconds > 3.0
    assert not out.with_name("x.mp3.part").exists()
    # MPEG-1 Layer III at 44.1 kHz
    assert _first_mp3_frame(b) == (3, 1, 0)


@pytest.mark.parametrize("kbps", MP3_BITRATES)
def test_mp3_every_bitrate_is_mpeg1_44k(tmp_path, kbps):
    pytest.importorskip("lameenc")
    out = tmp_path / f"b{kbps}.mp3"
    render([Chunk("a", 0.0)], fake_synth(SAMPLE_RATE, 0.3), out, "mp3", pause=0.0, bitrate=kbps)
    data = out.read_bytes()
    assert _first_mp3_frame(data) == (3, 1, 0)
    bitrate_index = {32: 1, 48: 3, 64: 5, 96: 7, 128: 9, 160: 10, 192: 11, 256: 13,
                     320: 14}[kbps]
    if data[:3] == b"ID3":
        data = data[10 + (data[6] << 21 | data[7] << 14 | data[8] << 7 | data[9]):]
    i = next(i for i in range(len(data) - 3) if data[i] == 0xFF and data[i + 1] & 0xE0 == 0xE0)
    assert data[i + 2] >> 4 == bitrate_index  # MPEG-1 Layer III bitrate table


def test_bitrate_choices():
    assert MP3_BITRATES == (32, 48, 64, 96, 128, 160, 192, 256, 320)


def test_request_and_result_carry_lang(tmp_path):
    req = ExportRequest("hi")
    assert req.lang is None
    assert ExportRequest("hi", None, "supertonic:F1", 1.0, "hr", "wav").lang == "hr"
    res = ExportResult(tmp_path / "a.wav", "wav", 1.0, 1, 10, "supertonic:F1", "hr")
    assert (res.voice, res.lang) == ("supertonic:F1", "hr")


def test_open_sink_bad_format(tmp_path):
    with pytest.raises(ValueError):
        export.open_sink(tmp_path / "x.ogg", "ogg")
