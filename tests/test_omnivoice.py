import json
import threading

import numpy as np
import pytest

from vibetts import omnivoice as ov
from vibetts.omnivoice import (CODEBOOKS, MASK_ID, OmniVoiceEngine, cfg_combine, estimate_frames,
                               is_omni, list_presets, load_preset, log_softmax, models_present,
                               parse_voice, post_process, unmask_schedule)

VOCAB = 1025
DIM = 4


class FakeTokenizer:
    """One id per character; special markers like <|lang_start|> count as one."""

    def encode(self, s, add_special_tokens=False):
        import re

        class Enc:
            ids = [ord(p[0]) % 900 + 1 for p in re.findall(r"<\|[a-z_]+\|>|.", s)]

        return Enc()


class FakeSession:
    def __init__(self, fn, dtype="tensor(float)"):
        self.fn = fn
        self.calls = []
        self._type = dtype

    def get_inputs(self):
        class In:
            type = self._type

        return [In()]

    def run(self, names, feed):
        self.calls.append({k: v.copy() for k, v in feed.items()})  # the engine reuses its arrays
        return [self.fn(feed)]


def fake_sessions(seq_log=None):
    """Embeddings -> hidden (B, S, DIM) from the ids, heads -> logits favouring token = position."""
    def embed(feed):
        ids = feed["input_ids"]
        return np.repeat(ids[:, 0, :, None].astype(np.float32), DIM, axis=2)

    def llm(feed):
        if seq_log is not None:
            seq_log.append(feed["inputs_embeds"].shape[1])
        return feed["inputs_embeds"]

    def heads(feed):
        h = feed["hidden_states"]  # (B, T, DIM)
        t = h.shape[1]
        logits = np.zeros((1, CODEBOOKS, t, VOCAB), np.float32)
        for i in range(t):
            logits[0, :, i, i % 1000] = 5.0  # frame i wants token i
        return logits

    def decode(feed):
        codes = feed["codes"]  # (8, 1, T)
        return np.repeat(np.ones(codes.shape[2], np.float32), 960).reshape(1, 1, -1)

    return {"embed": FakeSession(embed), "llm": FakeSession(llm), "heads": FakeSession(heads),
            "decoder": FakeSession(decode)}


def fake_engine(tmp_path, steps=3, **kw):
    return OmniVoiceEngine(tmp_path, steps, sessions=fake_sessions(**kw), tokenizer=FakeTokenizer())


def write_preset(model_dir, name, text="Dobar dan.", frames=10):
    d = model_dir / "voices"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", np.arange(CODEBOOKS * frames).reshape(CODEBOOKS, frames) % 1000)
    (d / f"{name}.txt").write_text(text + "\n", encoding="utf-8")


def test_is_omni_and_parse_voice():
    assert is_omni("omni:default") and is_omni("omni") and is_omni("omni_hr:default")
    assert is_omni("omni_hr")  # legacy ids are still accepted
    assert not is_omni("af_heart") and not is_omni("") and not is_omni("omnivore")
    assert parse_voice("omni:default") == (None, "default")  # the language comes per request
    assert parse_voice("omni") == (None, "")
    assert parse_voice("omni_hr:default") == ("hr", "default")
    assert parse_voice("omni_hr") == ("hr", "")
    assert parse_voice("omni_sr:ana") == ("sr", "ana")


def test_models_present(tmp_path):
    assert not models_present(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"a.onnx": 1, "b.json": 2}}))
    assert not models_present(tmp_path)
    (tmp_path / "a.onnx").write_bytes(b"x")
    assert not models_present(tmp_path)
    (tmp_path / "b.json").write_bytes(b"xx")
    assert models_present(tmp_path)
    (tmp_path / "manifest.json").write_text("not json")
    assert not models_present(tmp_path)


def test_list_presets(tmp_path):
    assert list_presets(tmp_path) == []
    write_preset(tmp_path, "default")
    write_preset(tmp_path, "ana")
    np.save(tmp_path / "voices" / "orphan.npy", np.zeros((8, 3)))  # no text file
    (tmp_path / "voices" / "ana.wav").write_bytes(b"")
    assert list_presets(tmp_path) == ["omni:ana", "omni:default"]
    assert fake_engine(tmp_path).voices() == ["omni:ana", "omni:default"]


def test_load_preset_validates_shape(tmp_path):
    write_preset(tmp_path, "ok", "Tekst", frames=5)
    codes, text = load_preset(tmp_path, "ok")
    assert codes.shape == (CODEBOOKS, 5) and codes.dtype == np.int64 and text == "Tekst"
    np.save(tmp_path / "voices" / "bad.npy", np.zeros((4, 5)))
    (tmp_path / "voices" / "bad.txt").write_text("x")
    with pytest.raises(ValueError):
        load_preset(tmp_path, "bad")


def test_duration_estimate():
    # reference "Nice to meet you." (17 chars: 13 letters, 3 spaces, 1 dot) took 25 frames
    ref = "Nice to meet you."
    assert ov.text_weight(ref) == pytest.approx(13 + 3 * 0.2 + 0.5)
    long_text = "a" * 200
    assert estimate_frames(long_text, ref, 25) == pytest.approx(200 / 14.1 * 25)
    assert estimate_frames("a" * 50, ref, 25) == pytest.approx(50 * 25 / 14.1)  # no boost
    short = estimate_frames("Da.", ref, 25)  # tiny estimate is boosted towards the 50 frame floor
    assert 25 * 1.5 / 14.1 < short < 50
    assert estimate_frames("x", "", 25) == 0.0 and estimate_frames("x", ref, 0) == 0.0
    # diacritics are free, digits cost 3.5, other scripts have their own weight
    assert ov.text_weight("č") == 1.0 and ov.text_weight("č") == 1.0
    assert ov.text_weight("12") == 7.0
    assert ov.text_weight("你好") == 6.0


def test_unmask_schedule_covers_every_cell():
    for frames, steps in [(1, 4), (75, 6), (230, 16), (3, 8)]:
        plan = unmask_schedule(frames, steps)
        assert len(plan) == steps and sum(plan) == frames * CODEBOOKS
        assert all(k >= 0 for k in plan)
    plan = unmask_schedule(100, 6)
    assert plan[0] < plan[-2]  # t_shift warps towards unmasking more cells later


def test_cfg_combine():
    cond = log_softmax(np.log(np.array([[0.7, 0.2, 0.1]])))
    uncond = log_softmax(np.log(np.array([[0.4, 0.4, 0.2]])))
    assert np.allclose(cfg_combine(cond, uncond, 0.0), cond)
    out = cfg_combine(cond, uncond, 2.0)
    assert np.isclose(np.exp(out).sum(), 1.0)
    assert out.argmax() == 0 and np.exp(out)[0, 0] > 0.7  # sharper than the conditional


def test_post_process_trims_normalises_and_pads():
    sr = ov.SAMPLE_RATE
    w = np.zeros(sr * 2, np.float32)
    w[sr // 2: sr] = 0.2
    out = post_process(w)
    assert out.dtype == np.float32
    assert np.isclose(np.abs(out).max(), 0.5)
    assert len(out) < len(w)  # silence at both ends trimmed to 100 ms
    assert np.all(out[:sr // 10] == 0) and np.all(out[-sr // 10:] == 0)
    assert len(post_process(np.zeros(100, np.float32))) == 2 * (sr // 10) + 100


def test_build_prompt_layout_without_and_with_preset(tmp_path):
    eng = fake_engine(tmp_path)
    ids, audio, start = eng._build("ab", 5, "hr", None, "")
    style = len("<|lang_start|>hr<|lang_end|><|instruct_start|>None<|instruct_end|>")
    n_style = len(eng._ids("<|lang_start|>hr<|lang_end|><|instruct_start|>None<|instruct_end|>"))
    n_text = len(eng._ids("<|text_start|>ab<|text_end|>"))
    assert style > n_style  # markers are single tokens
    assert ids.shape == (1, CODEBOOKS, n_style + n_text + 5)
    assert start == n_style + n_text
    assert (ids[0, :, start:] == MASK_ID).all()
    assert (ids[0, 0, :start] == ids[0, 7, :start]).all()  # text rows copied to every codebook
    assert not audio[0, :start].any() and audio[0, start:].all()

    codes = np.arange(CODEBOOKS * 4).reshape(CODEBOOKS, 4)
    ids, audio, start = eng._build("ab", 5, "hr", codes, "Hej.")
    # <|denoise|> is added, the reference text is spoken before the target text
    assert eng._ids("<|denoise|>")[0] == ids[0, 0, 0]
    n_pre = len(eng._ids("<|denoise|><|lang_start|>hr<|lang_end|><|instruct_start|>None"
                         "<|instruct_end|>")) + len(eng._ids("<|text_start|>Hej. ab<|text_end|>"))
    assert start == n_pre + 4
    assert (ids[0, :, n_pre:start] == codes).all()
    assert (ids[0, :, start:] == MASK_ID).all()
    assert audio[0, n_pre:].all() and not audio[0, :n_pre].any()


def test_synth_fills_every_cell_and_is_deterministic(tmp_path):
    eng = fake_engine(tmp_path, steps=3)
    a = eng.synth("Dobar dan", "omni_hr", 1.0)
    b = eng.synth("Dobar dan", "omni_hr", 1.0)
    assert a.dtype == np.float32 and a.ndim == 1
    assert np.array_equal(a, b)
    assert np.abs(a).max() <= 0.5 + 1e-6
    tokens = eng.generate_codes("Dobar dan")
    assert tokens.shape[0] == CODEBOOKS and (tokens != MASK_ID).all()
    # the fake heads want token = frame index (mod 1000) in every codebook
    assert (tokens == np.arange(tokens.shape[1]) % 1000).all()


def test_speed_scales_the_duration(tmp_path):
    eng = fake_engine(tmp_path, steps=2)
    normal = len(eng.synth("Ovo je jedna malo duza recenica za test.", "omni_hr", 1.0))
    fast = len(eng.synth("Ovo je jedna malo duza recenica za test.", "omni_hr", 2.0))
    assert fast < normal * 0.7


def test_preset_prefix_is_used_for_every_chunk(tmp_path):
    write_preset(tmp_path, "default", "Dobar dan.", frames=10)
    seq = []
    eng = fake_engine(tmp_path, steps=2, seq_log=seq)
    eng.synth("Prva recenica.", "omni_hr:default", 1.0)
    eng.synth("Druga recenica.", "omni_hr:default", 1.0)
    embed_calls = eng._embed.calls
    ref = load_preset(tmp_path, "default")[0]
    # each step is a conditional + an unconditional pass (2 steps per chunk = 4 calls); in the
    # first call of a chunk the reference codes sit right before the fully masked target
    for call in (embed_calls[0], embed_calls[4]):
        row = call["input_ids"][0]
        start = row.shape[1] - int((row[0] == MASK_ID).sum())
        assert (row[:, start - 10:start] == ref).all()
    # a longer sequence than the same text without a preset
    seq.clear()
    eng.synth("Prva recenica.", "omni_hr", 1.0)
    no_preset = max(seq)
    seq.clear()
    eng.synth("Prva recenica.", "omni_hr:default", 1.0)
    assert max(seq) > no_preset


def test_unknown_preset_raises(tmp_path):
    eng = fake_engine(tmp_path)
    with pytest.raises(FileNotFoundError):
        eng.synth("Tekst.", "omni_hr:nope", 1.0)


def test_synth_calls_are_serialized(tmp_path):
    eng = fake_engine(tmp_path, steps=2)
    inside, overlap = [0], []
    real = eng._decode

    def slow_decode(tokens):
        inside[0] += 1
        overlap.append(inside[0])
        threading.Event().wait(0.02)
        out = real(tokens)
        inside[0] -= 1
        return out

    eng._decode = slow_decode
    ts = [threading.Thread(target=eng.synth, args=("Tekst.", "omni_hr", 1.0)) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert max(overlap) == 1


def test_make_voice_saves_preset_files(tmp_path):
    eng = fake_engine(tmp_path, steps=2)
    wav = ov.make_voice(eng, "ana", seed=3, text="Dobar dan.")
    assert wav == tmp_path / "voices" / "ana.wav" and wav.is_file()
    meta = json.loads((tmp_path / "voices" / "ana.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 3 and meta["lang"] == "hr" and meta["kind"] == "seed"
    assert list_presets(tmp_path) == ["omni:ana"]
    codes, text = load_preset(tmp_path, "ana")
    assert codes.shape[0] == CODEBOOKS and text == "Dobar dan."
    assert eng.seed == 3


# ---------------------------------------------------------------- custom voices

def test_validate_instruct_normalises():
    assert ov.validate_instruct("Female,  Young Adult ,low pitch") == "female, young adult, low pitch"
    assert ov.validate_instruct("male，middle-aged") == "male, middle-aged"  # full-width comma
    assert ov.validate_instruct("male, ,") == "male"
    assert ov.validate_instruct("male, male") == "male"  # repeating an item is fine
    assert ov.validate_instruct("") == "" and ov.validate_instruct("  ") == ""


def test_validate_instruct_rejects_unknown_with_hint():
    with pytest.raises(ValueError, match=r"did you mean 'female'"):
        ov.validate_instruct("femal, low pitch")
    with pytest.raises(ValueError, match="unsupported"):
        ov.validate_instruct("zzzzzz")
    with pytest.raises(ValueError):
        ov.validate_instruct("american accent")  # English accents are not offered


def test_validate_instruct_rejects_same_category_conflict():
    with pytest.raises(ValueError, match="gender"):
        ov.validate_instruct("male, female")
    with pytest.raises(ValueError, match="pitch"):
        ov.validate_instruct("female, low pitch, high pitch")
    assert ov.validate_instruct("female, whisper, child, very low pitch")  # one per category


def test_instruct_options_shape():
    assert set(ov.INSTRUCT_OPTIONS) == {"gender", "age", "pitch", "style"}
    assert ov.INSTRUCT_OPTIONS["gender"] == ["male", "female"]
    for items in ov.INSTRUCT_OPTIONS.values():
        for item in items:
            assert ov.validate_instruct(item) == item


def test_sanitize_name():
    assert ov.sanitize_name("Ana") == "ana"
    assert ov.sanitize_name("  Moj glas 2 ") == "moj-glas-2"
    assert ov.sanitize_name("Šime Đurđević") == "sime-djurdjevic"
    assert ov.sanitize_name("a_b-c") == "a_b-c"
    assert ov.sanitize_name("../../etc") == "etc"
    assert len(ov.sanitize_name("x" * 100)) == 32
    for bad in ("", "   ", "???", "default", "DEFAULT", "-_-"):
        with pytest.raises(ValueError):
            ov.sanitize_name(bad)


def wav_bytes(samples, rate=24000, channels=1, bits=16, fmt=1, extensible=False):
    """RIFF/WAVE bytes; `samples` are already-encoded raw bytes."""
    import struct

    width = bits // 8
    if extensible:
        guid = struct.pack("<H", fmt) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt_body = struct.pack("<HHIIHHHHI", 0xFFFE, channels, rate, rate * channels * width,
                               channels * width, bits, 22, bits, 3) + guid
    else:
        fmt_body = struct.pack("<HHIIHH", fmt, channels, rate, rate * channels * width,
                               channels * width, bits)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt_body) + 8 + len(samples)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
            + b"data" + struct.pack("<I", len(samples)) + samples)


def test_read_wav_pcm16_float_24bit_and_stereo(tmp_path):
    t = np.linspace(0, 1, 2400, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "a.wav"
    p.write_bytes(wav_bytes((sine * 32767).astype("<i2").tobytes()))
    x, rate = ov.read_wav(p)
    assert rate == 24000 and x.dtype == np.float32 and np.allclose(x, sine, atol=1e-3)

    p.write_bytes(wav_bytes(sine.astype("<f4").tobytes(), bits=32, fmt=3))
    assert np.allclose(ov.read_wav(p)[0], sine)
    p.write_bytes(wav_bytes(sine.astype("<f4").tobytes(), bits=32, fmt=3, extensible=True))
    assert np.allclose(ov.read_wav(p)[0], sine)  # WAVE_FORMAT_EXTENSIBLE float

    v = (sine * 8388607).astype(np.int32)
    raw = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], axis=1).astype(np.uint8).tobytes()
    p.write_bytes(wav_bytes(raw, bits=24))
    assert np.allclose(ov.read_wav(p)[0], sine, atol=1e-4)

    p.write_bytes(wav_bytes((sine * 2 ** 31 * 0.999).astype("<i4").tobytes(), bits=32))
    assert np.allclose(ov.read_wav(p)[0], sine, atol=2e-3)

    left, right = sine, -sine / 2
    stereo = np.stack([left, right], axis=1)
    p.write_bytes(wav_bytes((stereo * 32767).astype("<i2").tobytes(), channels=2, rate=48000))
    x, rate = ov.read_wav(p)
    assert rate == 48000 and len(x) == 2400 and np.allclose(x, sine / 4, atol=1e-3)  # mean of L/R


def test_read_wav_rejects_bad_files(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not a wav at all")
    with pytest.raises(ValueError, match="Not a WAV"):
        ov.read_wav(p)
    p.write_bytes(wav_bytes(b"\x00" * 100, fmt=2, bits=16))  # ADPCM tag
    with pytest.raises(ValueError, match="Unsupported"):
        ov.read_wav(p)
    with pytest.raises(ValueError):
        ov.read_wav(tmp_path / "missing.wav")


def test_trim_silence():
    sr = 24000
    x = np.zeros(sr * 4, np.float32)
    x[sr: sr * 3] = 0.3 * np.sin(np.linspace(0, 2000, sr * 2)).astype(np.float32)
    t = ov.trim_silence(x, sr)
    assert 2.0 * sr <= len(t) <= 2.3 * sr  # 2 s of speech plus a 0.1 s margin each side
    assert len(ov.trim_silence(np.zeros(sr, np.float32), sr)) == 0
    assert len(ov.trim_silence(np.zeros(5, np.float32), sr)) == 0


def fake_encoders():
    """Acoustic/semantic/quantizer stand-ins; the quantizer returns codes (8, 1, T)."""
    seen = {}

    def acoustic(feed):
        w = feed["waveform_24k"]
        seen["a"] = w.shape
        return np.zeros((1, 4, w.shape[2] // 960), np.float32)

    def semantic(feed):
        w = feed["waveform_16k"]
        seen["s"] = w.shape
        return np.zeros((1, 4, w.shape[1] // 640 + 1), np.float32)  # one frame longer

    def quant(feed):
        t = feed["acoustic_features"].shape[2]
        assert feed["semantic_features"].shape[2] == t  # aligned to the shorter of the two
        seen["q"] = t
        return np.arange(8 * t).reshape(8, 1, t) % 1024

    sess = {"acoustic_encoder": FakeSession(acoustic), "semantic_encoder": FakeSession(semantic),
            "quantizer_encoder": FakeSession(quant)}
    return sess, seen


def clone_engine(tmp_path):
    enc, seen = fake_encoders()
    calls = []

    def load():
        calls.append(1)
        return enc

    eng = OmniVoiceEngine(tmp_path, 2, sessions=fake_sessions(), tokenizer=FakeTokenizer(),
                          encoder_loader=load)
    return eng, enc, seen, calls


def speech_wav(path, seconds, rate=24000, lead=0.3):
    n = int(seconds * rate)
    x = np.zeros(n + int(2 * lead * rate), np.float32)
    x[int(lead * rate): int(lead * rate) + n] = 0.02 * np.sin(np.linspace(0, 3000 * seconds, n))
    path.write_bytes(wav_bytes((x * 32767).astype("<i2").tobytes(), rate=rate))
    return path


def test_clone_clip_pipeline(tmp_path):
    eng, enc, seen, calls = clone_engine(tmp_path)
    wav = speech_wav(tmp_path / "ref.wav", 3.0, rate=16000)  # resampled up to 24 kHz
    clip = eng.clone_clip(wav, "  Dobar dan svima.  ")
    assert calls == [1]  # encoders loaded lazily, once
    assert clip.text == "Dobar dan svima."
    assert clip.codes.dtype == np.int16 and clip.codes.shape[0] == CODEBOOKS
    assert clip.codes.shape[1] == seen["q"]
    n24 = seen["a"][2]
    assert n24 % 960 == 0 and 3.0 * 24000 <= n24 <= 3.3 * 24000  # trimmed, 24 kHz, whole frames
    assert abs(seen["s"][1] - n24 * 16000 / 24000) < 10
    assert clip.audio.dtype == np.float32 and clip.audio.ndim == 1
    assert clip.meta["kind"] == "clone" and clip.meta["source"] == "ref.wav"
    assert clip.warning == "" and "warning" not in clip.meta
    sent = enc["acoustic_encoder"].calls[0]["waveform_24k"]
    assert sent.dtype == np.float32
    assert 0.09 < float(np.sqrt((sent ** 2).mean())) < 0.11  # a quiet recording is levelled to 0.1


def test_clone_clip_rejects_short_and_empty(tmp_path):
    eng, _, _, calls = clone_engine(tmp_path)
    with pytest.raises(ValueError, match="too short"):
        eng.clone_clip(speech_wav(tmp_path / "s.wav", 1.5), "Kratko.")
    with pytest.raises(ValueError, match="transcript"):
        eng.clone_clip(speech_wav(tmp_path / "ok.wav", 3.0), "   ")
    with pytest.raises(ValueError):
        eng.clone_clip(tmp_path / "nope.wav", "Tekst.")
    assert calls == []  # rejected before the encoders were loaded


def test_clone_clip_rejects_long_recordings(tmp_path):
    eng, _, _, calls = clone_engine(tmp_path)
    with pytest.raises(ValueError, match="too long"):
        eng.clone_clip(speech_wav(tmp_path / "long.wav", 12.0), "Dugacak tekst.")
    assert calls == []  # rejected before the encoders were loaded


def test_clone_clip_without_encoders_raises(tmp_path):
    eng = OmniVoiceEngine(tmp_path, 2, sessions=fake_sessions(), tokenizer=FakeTokenizer())
    assert not ov.encoders_present(tmp_path)
    with pytest.raises(FileNotFoundError, match="encoders"):
        eng.clone_clip(speech_wav(tmp_path / "r.wav", 3.0), "Tekst.")
    assert not eng._lock.locked()  # the lock is released after the error


def test_encoders_present(tmp_path):
    assert not ov.encoders_present(tmp_path)
    d = tmp_path / "encoders"
    d.mkdir()
    for n in ("acoustic_encoder", "semantic_encoder"):
        (d / f"{n}.onnx").write_bytes(b"x")
    assert not ov.encoders_present(tmp_path)
    (d / "quantizer_encoder.onnx").write_bytes(b"x")
    assert ov.encoders_present(tmp_path)


def test_clone_failure_releases_lock_and_encoders(tmp_path):
    eng, enc, _, _ = clone_engine(tmp_path)
    released = []
    eng._release_encoders = lambda: released.append(1)

    def boom(feed):
        raise RuntimeError("boom")

    enc["quantizer_encoder"].fn = boom
    with pytest.raises(RuntimeError):
        eng.clone_clip(speech_wav(tmp_path / "r.wav", 3.0), "Tekst.")
    assert released == [1] and not eng._lock.locked()


def test_design_clip_puts_instruct_in_the_prompt(tmp_path):
    eng = fake_engine(tmp_path, steps=2)
    clip = eng.design_clip("Female,  low pitch", 5)
    assert clip.meta == {"kind": "design", "instruct": "female, low pitch", "seed": 5,
                         "lang": ov.DESIGN_LANG}
    assert clip.text == ov.DESIGN_TEXTS[ov.DESIGN_LANG]
    assert clip.codes.dtype == np.int16 and clip.codes.shape[0] == CODEBOOKS
    assert clip.audio.dtype == np.float32
    first = eng._embed.calls[0]["input_ids"][0, 0]
    expected = eng._ids("<|lang_start|>hr<|lang_end|><|instruct_start|>female, low pitch"
                        "<|instruct_end|>")
    assert list(first[:len(expected)]) == expected  # and no <|denoise|>: there is no reference
    eng.design_clip("", 1, "Neki tekst.")  # no description: the 'None' prompt
    plain = eng._ids("<|lang_start|>hr<|lang_end|><|instruct_start|>None<|instruct_end|>")
    assert list(eng._embed.calls[-2]["input_ids"][0, 0][:len(plain)]) == plain
    with pytest.raises(ValueError):
        eng.design_clip("male, female", 0)
    with pytest.raises(ValueError):
        eng.design_clip("male", 0, "  ")


def test_design_clip_uses_16_steps_and_keeps_engine_settings(tmp_path):
    eng = fake_engine(tmp_path, steps=2)
    eng.seed = 9
    eng.design_clip("male", 3)
    # 16 steps, each a conditional + an unconditional pass
    assert len(eng._embed.calls) == 2 * ov.DESIGN_STEPS
    assert eng.steps == 2 and eng.seed == 9


def test_design_and_clone_hold_the_synth_lock(tmp_path):
    eng, _, _, _ = clone_engine(tmp_path)
    held = []
    real = eng._decode

    def spy(tokens):
        held.append(eng._lock.locked())
        return real(tokens)

    eng._decode = spy
    eng.design_clip("male", 0)
    eng.clone_clip(speech_wav(tmp_path / "r.wav", 3.0), "Tekst.")
    assert held == [True, True]


def a_clip(frames=12, text="Dobar dan."):
    codes = np.arange(CODEBOOKS * frames).reshape(CODEBOOKS, frames).astype(np.int16) % 1000
    return ov.VoiceClip(codes, text, np.zeros(4800, np.float32), {"kind": "design", "seed": 1})


def test_save_preset_writes_files_and_meta(tmp_path):
    vid = ov.save_preset(tmp_path, "Moj Glas", a_clip())
    assert vid == "omni:moj-glas"
    d = tmp_path / "voices"
    assert sorted(p.name for p in d.iterdir()) == ["moj-glas.json", "moj-glas.npy", "moj-glas.txt",
                                                   "moj-glas.wav"]
    meta = json.loads((d / "moj-glas.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "design" and meta["seed"] == 1 and "created" in meta
    codes, text = load_preset(tmp_path, "moj-glas")
    assert codes.shape == (CODEBOOKS, 12) and text == "Dobar dan."
    assert list_presets(tmp_path) == ["omni:moj-glas"]  # the .json/.wav are ignored


def test_save_preset_refuses_default_bad_names_and_overwrite(tmp_path):
    with pytest.raises(ValueError):
        ov.save_preset(tmp_path, "default", a_clip())
    with pytest.raises(ValueError):
        ov.save_preset(tmp_path, "???", a_clip())
    assert not (tmp_path / "voices").exists()
    ov.save_preset(tmp_path, "ana", a_clip())
    with pytest.raises(FileExistsError):
        ov.save_preset(tmp_path, "ana", a_clip(text="Drugo."))
    assert load_preset(tmp_path, "ana")[1] == "Dobar dan."
    ov.save_preset(tmp_path, "ana", a_clip(text="Drugo."), overwrite=True)
    assert load_preset(tmp_path, "ana")[1] == "Drugo."


def test_saved_preset_is_usable_by_synth(tmp_path):
    eng = fake_engine(tmp_path, steps=2)
    vid = ov.save_preset(tmp_path, "ana", eng.design_clip("female", 0))
    assert eng.voices() == [vid]
    assert len(eng.synth("Tekst.", vid, 1.0)) > 0


def test_delete_preset(tmp_path):
    write_preset(tmp_path, "default")
    ov.save_preset(tmp_path, "ana", a_clip())
    ov.delete_preset(tmp_path, "ana")
    assert list_presets(tmp_path) == ["omni:default"]
    assert sorted(p.name for p in (tmp_path / "voices").iterdir()) == ["default.npy", "default.txt"]
    with pytest.raises(ValueError, match="cannot be deleted"):
        ov.delete_preset(tmp_path, "default")
    assert list_presets(tmp_path) == ["omni:default"]
    with pytest.raises(FileNotFoundError):
        ov.delete_preset(tmp_path, "ana")
    with pytest.raises(ValueError):
        ov.delete_preset(tmp_path, "../default")


def test_cli_design_without_model_files(tmp_path, capsys):
    assert ov.main(["design-voice", "x", "--instruct", "male", "--model-dir", str(tmp_path)]) == 1
    assert "missing" in capsys.readouterr().err


# ---------------------------------------------------------------- language per request

def spy_lang(eng):
    seen = []
    real = eng._build

    def build(text, frames, lang, *a, **k):
        seen.append(lang)
        return real(text, frames, lang, *a, **k)
    eng._build = build
    return seen


def test_synth_uses_the_request_language(tmp_path):
    write_preset(tmp_path, "default")
    eng = fake_engine(tmp_path, steps=1)
    seen = spy_lang(eng)
    eng.synth("Hello there.", "omni:default", 1.0)  # no language given: English
    eng.synth("Dobar dan.", "omni:default", 1.0, "hr")
    eng.synth("Hello there.", "omni:default", 1.0, None)
    assert seen[0] == "en" and seen[1] == "hr" and seen[2] == "en"


def test_legacy_voice_id_keeps_its_language(tmp_path):
    write_preset(tmp_path, "default")
    eng = fake_engine(tmp_path, steps=1)
    seen = spy_lang(eng)
    eng.synth("Dobar dan.", "omni_hr:default", 1.0)
    eng.synth("Hello.", "omni_hr:default", 1.0, "en")  # an explicit language wins
    assert seen[0] == "hr" and seen[1] == "en"


def test_design_clip_language_and_meta(tmp_path):
    eng = fake_engine(tmp_path, steps=1)
    hr = eng.design_clip("female", 0)  # default: Croatian sample
    assert hr.meta["lang"] == "hr" and hr.text == ov.DESIGN_TEXTS["hr"]
    en = eng.design_clip("female", 0, None, "en")
    assert en.meta["lang"] == "en" and en.text == ov.DESIGN_TEXTS["en"]
    with pytest.raises(ValueError, match="sample text"):
        eng.design_clip("female", 0, None, "xx")
    custom = eng.design_clip("male", 1, "Hello world.", "en")
    assert custom.text == "Hello world." and custom.meta["lang"] == "en"


def test_preset_json_records_lang(tmp_path):
    clip = a_clip()
    clip.meta["lang"] = "en"
    ov.save_preset(tmp_path, "eng", clip)
    meta = json.loads((tmp_path / "voices" / "eng.json").read_text(encoding="utf-8"))
    assert meta["lang"] == "en"


def test_preset_info_kinds(tmp_path):
    write_preset(tmp_path, "default")  # no json: the built-in Croatian default
    d = a_clip()
    d.meta.update(instruct="female, young adult", lang="en")
    ov.save_preset(tmp_path, "alma", d)
    c = ov.VoiceClip(a_clip().codes, "Tekst.", np.zeros(10, np.float32),
                     {"kind": "clone", "source": "me.wav", "lang": "hr"})
    ov.save_preset(tmp_path, "ja", c)
    rows = {r["name"]: r for r in ov.list_preset_info(tmp_path)}
    assert [r["name"] for r in ov.list_preset_info(tmp_path)][0] == "default"
    assert rows["default"]["kind"] == "default" and rows["default"]["lang"] == "hr"
    assert rows["alma"] == {"name": "alma", "voice": "omni:alma", "kind": "description",
                            "description": "female, young adult", "lang": "en"}
    assert rows["ja"]["kind"] == "recording" and rows["ja"]["description"] == "me.wav"
    write_preset(tmp_path, "plain")
    assert ov.preset_info(tmp_path, "plain")["kind"] == "random"
