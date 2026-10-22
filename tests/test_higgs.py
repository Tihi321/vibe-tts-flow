import json
import logging
import os
import sys
import types
import wave

import numpy as np
import pytest

from vibetts import higgs
from vibetts.config import HIGGS_LANGUAGES, HIGGS_PRESETS

EOC = higgs.EOC_ID
BOC = higgs.BOC_ID
HIDDEN = 4
TEXT_VOCAB = ["hello", "world", "dobar", "dan", "ref", "words", "about", "this", "is", "a",
              "long", "sentence"]


# ---------------------------------------------------------------- pure helpers

def test_sanitize_strips_control_fragments():
    assert higgs.sanitize_text("a <|emotion:happy|> b") == "a emotion:happy b"
    assert higgs.sanitize_text("x<|y|>z |> <| w") == "xyz w"
    assert higgs.sanitize_text("  plain   text \n") == "plain text"
    assert "<|" not in higgs.sanitize_text("<|<|tts|>|>")


@pytest.mark.parametrize("speed,tag", [
    (1.0, ""), (1.14, ""), (0.86, ""), (1.15, "<|prosody:speed_fast|>"),
    (1.3, "<|prosody:speed_fast|>"), (1.35, "<|prosody:speed_very_fast|>"),
    (2.0, "<|prosody:speed_very_fast|>"), (0.85, "<|prosody:speed_slow|>"),
    (0.76, "<|prosody:speed_slow|>"), (0.75, "<|prosody:speed_very_slow|>"),
    (0.5, "<|prosody:speed_very_slow|>")])
def test_speed_tag(speed, tag):
    assert higgs.speed_tag(speed) == tag


def test_delay_pattern_layout_and_roundtrip():
    codes = np.arange(1, 5 * 8 + 1, dtype=np.int64).reshape(5, 8)
    delayed = higgs.apply_delay_pattern(codes)
    assert delayed.shape == (5 + 7, 8)
    assert delayed[0, 0] == codes[0, 0] and delayed[0, 1] == BOC and delayed[3, 3] == codes[0, 3]
    assert delayed[-1, 0] == EOC and delayed[-1, 7] == codes[-1, 7]
    assert (delayed[:7, 7] == BOC).all()
    assert np.array_equal(higgs.reverse_delay_pattern(delayed), codes)
    assert higgs.reverse_delay_pattern(delayed[:7]).shape == (0, 8)


def test_stable_seed_is_deterministic_and_distinct():
    assert higgs.stable_seed("higgs:F1", "hr") == higgs.stable_seed("higgs:F1", "hr")
    assert higgs.stable_seed("higgs:F1", "hr") != higgs.stable_seed("higgs:F1", "en")
    assert higgs.stable_seed("higgs:F1", "hr") != higgs.stable_seed("higgs:F2", "hr")
    assert higgs.stable_seed("higgs:F1", "hr") == int.from_bytes(
        __import__("hashlib").sha256(b"higgs:F1\0hr").digest()[:4], "little")


def test_frame_cap_scales_and_is_bounded():
    assert higgs.max_frames_for("Hi") < higgs.max_frames_for("Hi " * 50)
    assert higgs.max_frames_for("x" * 10_000) == higgs.MAX_FRAMES
    assert higgs.max_frames_for("日本語" * 20) > higgs.max_frames_for("abc" * 20)
    assert higgs.max_frames_for("Dobar dan.") > higgs.expected_frames("Dobar dan.")


def test_needs_retry_thresholds():
    text = "x" * 100                              # expected 200 frames, threshold 100
    assert higgs.expected_frames(text) == 200
    assert higgs.needs_retry(0, "repeat", text) and higgs.needs_retry(0, "eoc", text)
    assert higgs.needs_retry(higgs.MIN_FRAMES - 1, "eoc", text)
    assert not higgs.needs_retry(higgs.MIN_FRAMES, "eoc", text)      # EOC is trusted
    assert higgs.needs_retry(99, "repeat", text)
    assert not higgs.needs_retry(100, "repeat", text)
    assert not higgs.needs_retry(150, "cap", text)


def test_model_files_table():
    paths = [p for p, _ in higgs.MODEL_FILES]
    assert len(set(paths)) == len(paths)
    assert "LICENSE" in paths and "cpu_int4/llm_decoder.onnx.data" in paths
    for need in ("llm_decoder", "text_embed", "audio_embed", "audio_heads", "audio_tokenizer",
                 "audio_encoder"):
        assert f"cpu_int4/{need}.onnx" in paths
    assert "cpu_int4/tokenizer.json" in paths and "cpu_int4/manifest.json" in paths
    assert higgs.DOWNLOAD_BYTES == sum(size for _, size in higgs.MODEL_FILES) == 3_849_550_721
    assert higgs.REVISION == "ecc25d6cc6e48909401bfa7ce5ac95e84f41f331"


def test_models_present(tmp_path):
    assert not higgs.models_present(tmp_path)
    for rel, size in higgs.MODEL_FILES:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            fh.truncate(size)
    assert higgs.models_present(tmp_path)
    (tmp_path / "cpu_int4" / "manifest.json").write_bytes(b"x")
    assert not higgs.models_present(tmp_path)
    (tmp_path / "cpu_int4" / "manifest.json").unlink()
    assert not higgs.models_present(tmp_path)


def test_presets_dir_follows_resources_root(tmp_path, monkeypatch):
    monkeypatch.setattr(higgs, "RESOURCES", tmp_path)
    assert higgs.presets_dir() == tmp_path / "vibetts" / "voices" / "higgs"


def test_clip_candidates_order():
    assert higgs.clip_candidates("hr") == ["hr", "en"]
    assert higgs.clip_candidates("bs") == ["bs", "hr", "en"]
    assert higgs.clip_candidates("mk") == ["mk", "bg", "en"]
    assert higgs.clip_candidates("en") == ["en"]
    assert higgs.clip_candidates("fi") == ["fi", "en"]


# ---------------------------------------------------------------- bundled presets

def test_bundled_presets_are_complete():
    base = higgs.presets_dir()
    for voice in HIGGS_PRESETS:
        langs = higgs.voice_languages(voice, base.parent)
        assert "en" in langs and "hr" in langs, voice
        for lang in langs:
            codes = np.load(base / voice / f"{lang}.codes.npy")
            assert codes.dtype == np.int16 and codes.shape[0] == 8 and 50 < codes.shape[1] < 500
            assert 0 <= codes.min() and codes.max() < 1024
            assert (base / voice / f"{lang}.txt").read_text(encoding="utf-8").strip()
            assert lang in HIGGS_LANGUAGES


# ---------------------------------------------------------------- fake runtime

class Heads:
    """Scripts the codebook-0 token per generated frame and per attempt."""
    plan = staticmethod(lambda attempt, step: 1 + step % 2)
    nan = False


class FakeSession:
    instances: list = []
    attempt = -1
    step = 0

    def __init__(self, path, sess_options=None, providers=None):
        self.name = os.path.basename(path)
        self.options = sess_options
        self.calls: list = []
        FakeSession.instances.append(self)

    def get_outputs(self):
        names = ["hidden_states", "present.0.key", "present.0.value"]
        return [types.SimpleNamespace(name=n) for n in names]

    def run(self, _outputs, feed):
        self.calls.append(feed)
        if self.name == "text_embed.onnx":
            ids = feed["input_ids"]
            out = np.zeros((1, ids.shape[1], HIDDEN), dtype=np.float32)
            out[0, :, 0] = ids[0]
            return [out]
        if self.name == "audio_embed.onnx":
            return [np.full((1, feed["codes"].shape[1], HIDDEN), 0.5, dtype=np.float32)]
        if self.name == "llm_decoder.onnx":
            if feed["past_key_values.0.key"].shape[2] == 0:     # prefill: a new attempt starts
                FakeSession.attempt += 1
                FakeSession.step = 0
            n = feed["inputs_embeds"].shape[1]
            past = np.zeros((1, 1, 1, higgs.HEAD_DIM), dtype=np.float32)
            return [np.ones((1, n, HIDDEN), dtype=np.float32), past, past]
        if self.name == "audio_heads.onnx":
            logits = np.zeros((1, 1, 8, 1026), dtype=np.float32)
            cb0 = Heads.plan(FakeSession.attempt, FakeSession.step)
            FakeSession.step += 1
            logits[0, 0, :, 5] = 100.0
            logits[0, 0, 0, 5] = 0.0
            logits[0, 0, 0, cb0] = 100.0
            if Heads.nan is True:
                logits[:] = np.nan
            return [logits]
        if self.name == "audio_tokenizer.onnx":
            n = feed["audio_codes"].shape[2]
            val = np.nan if Heads.nan == "codec" else 0.1
            return [np.full((1, 1, n * higgs.FRAME_SAMPLES), val, dtype=np.float32)]
        if self.name == "audio_encoder.onnx":
            n = feed["input_values"].shape[2] // higgs.FRAME_SAMPLES
            return [np.full((1, 8, n), 7, dtype=np.int64)]
        raise AssertionError(self.name)


def sessions(name):
    return [s for s in FakeSession.instances if s.name == name]


def write_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(TEXT_VOCAB)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.add_special_tokens(["<|tts|>", "<|text|>", "<|audio|>", "<|ref_text|>", "<|ref_audio|>",
                           "<|prosody:speed_slow|>", "<|prosody:speed_very_slow|>",
                           "<|prosody:speed_fast|>", "<|prosody:speed_very_fast|>"])
    tk.save(str(path))
    return {t.content: i for i, t in tk.get_added_tokens_decoder().items()}


def write_wav(path, seconds=4.0, rate=24000, channels=1, width=2):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    mono = (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2")
    data = np.repeat(mono[:, None], channels, axis=1) if channels > 1 else mono
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(data.astype("<i2").tobytes() if width == 2 else bytes(n * channels))


@pytest.fixture(autouse=True)
def reset_heads():
    Heads.plan = staticmethod(lambda attempt, step: 1 + step % 2)
    Heads.nan = False
    FakeSession.instances = []
    FakeSession.attempt = -1
    FakeSession.step = 0
    yield


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(),
                                 InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    model = tmp_path / "model"
    (model / "cpu_int4").mkdir(parents=True)
    (model / "cpu_int4" / "manifest.json").write_text(json.dumps(
        {"sub_models": {"llm_decoder": {"num_layers": 1, "num_kv_heads": 1}}}))
    ids = write_tokenizer(model / "cpu_int4" / "tokenizer.json")
    presets = tmp_path / "presets" / "vibetts" / "voices" / "higgs"
    monkeypatch.setattr(higgs, "RESOURCES", tmp_path / "presets")
    for voice in ("F1",):
        for lang, words in (("hr", "dobar dan"), ("en", "hello world")):
            d = presets / voice / lang
            d.parent.mkdir(parents=True, exist_ok=True)
            np.save(presets / voice / f"{lang}.codes.npy",
                    np.full((8, 10 if lang == "hr" else 12), 3, dtype=np.int16))
            (presets / voice / f"{lang}.txt").write_text(words, encoding="utf-8")
    voices = tmp_path / "voices"
    voices.mkdir()
    engine = higgs.HiggsEngine(model, voices, threads=3)
    return types.SimpleNamespace(engine=engine, ids=ids, voices=voices, presets=presets,
                                 model=model, tmp=tmp_path)


LONG_TEXT = "hello world this is a long sentence about this " * 2      # ~96 characters


def eoc_plan(frames_before_eoc=30):
    return lambda attempt, step: EOC if step == frames_before_eoc else 1 + step % 2


# ---------------------------------------------------------------- engine

def test_voices_lists_presets_and_custom(env):
    assert env.engine.voices() == ["higgs:F1", "higgs:F2", "higgs:M1", "higgs:M2"]


def test_threads_default_and_explicit(tmp_path, monkeypatch, env):
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert higgs.HiggsEngine(tmp_path, tmp_path)._threads == 8
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert higgs.HiggsEngine(tmp_path, tmp_path)._threads == 2
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert higgs.HiggsEngine(tmp_path, tmp_path)._threads == 1
    env.engine.synth("hello", "higgs:F1", 1.0, "en")
    assert all(s.options.intra_op_num_threads == 3 for s in FakeSession.instances)


def test_lazy_loading(env):
    assert FakeSession.instances == []
    env.engine.synth("hello", "higgs:F1", 1.0, "en")
    assert {s.name for s in FakeSession.instances} == {
        "llm_decoder.onnx", "text_embed.onnx", "audio_embed.onnx", "audio_heads.onnx",
        "audio_tokenizer.onnx"}                       # no audio_encoder for a preset
    count = len(FakeSession.instances)
    env.engine.synth("hello", "higgs:F1", 1.0, "en")
    assert len(FakeSession.instances) == count


def test_eoc_stop_output_and_codec_input(env):
    Heads.plan = staticmethod(eoc_plan(30))
    wav = env.engine.synth("hello", "higgs:F1", 1.0, "en")
    assert wav.dtype == np.float32 and wav.ndim == 1
    assert len(sessions("audio_heads.onnx")[0].calls) == 37        # EOC + 6 frames, then stop
    codec = sessions("audio_tokenizer.onnx")[0].calls[0]["audio_codes"]
    assert codec.shape == (1, 8, 30) and codec.dtype == np.int64
    assert codec.min() >= 0 and codec.max() <= 1023
    assert len(wav) == pytest.approx(30 * 960 * 44100 / 24000, abs=4)
    assert np.allclose(wav[2000:-2000], 0.1, atol=0.01)
    assert len(sessions("llm_decoder.onnx")[0].calls) == 1 + 36    # prefill + one step per frame


def test_prompt_layout(env):
    Heads.plan = staticmethod(eoc_plan(20))
    env.engine.synth("hello world", "higgs:F1", 1.0, "en")
    ids = env.ids
    calls = [c["input_ids"][0].tolist() for c in sessions("text_embed.onnx")[0].calls]
    vocab = {w: i + 1 for i, w in enumerate(TEXT_VOCAB)}
    assert calls == [
        [ids["<|tts|>"]],
        [ids["<|ref_text|>"], vocab["hello"], vocab["world"]],
        [ids["<|ref_audio|>"]],
        [ids["<|text|>"], vocab["hello"], vocab["world"]],
        [ids["<|audio|>"]]]
    ref_codes = sessions("audio_embed.onnx")[0].calls[0]["codes"]
    assert ref_codes.shape == (1, 12 + 7, 8)                       # en clip: 12 frames + delay
    assert ref_codes[0, 0, 0] == 3 and ref_codes[0, 0, 1] == BOC
    prefill = sessions("llm_decoder.onnx")[0].calls[0]
    assert prefill["inputs_embeds"].shape == (1, 1 + 3 + 1 + 19 + 3 + 1, HIDDEN)
    assert prefill["attention_mask"].shape == (1, 28)
    embeds = prefill["inputs_embeds"][0, :, 0]
    assert embeds[0] == ids["<|tts|>"] and embeds[1] == ids["<|ref_text|>"]
    assert embeds[4] == ids["<|ref_audio|>"] and embeds[5:24].tolist() == [0.5] * 19
    assert embeds[24] == ids["<|text|>"] and embeds[-1] == ids["<|audio|>"]


def text_body_ids(env):
    calls = [c["input_ids"][0].tolist() for c in sessions("text_embed.onnx")[0].calls]
    return next(c for c in calls if c[0] == env.ids["<|text|>"])


def test_sanitised_text_cannot_inject_tags(env):
    Heads.plan = staticmethod(eoc_plan(20))
    env.engine.synth("hello <|tts|> world <|audio|>", "higgs:F1", 1.0, "en")
    body = text_body_ids(env)
    assert body[0] == env.ids["<|text|>"]
    assert not set(body[1:]) & set(env.ids.values())               # no control token in the text


@pytest.mark.parametrize("speed,tag", [(1.0, None), (1.1, None), (1.2, "<|prosody:speed_fast|>"),
                                       (0.8, "<|prosody:speed_slow|>"),
                                       (0.7, "<|prosody:speed_very_slow|>"),
                                       (1.5, "<|prosody:speed_very_fast|>")])
def test_speed_tag_in_prompt(env, speed, tag):
    Heads.plan = staticmethod(eoc_plan(20))
    env.engine.synth("hello world", "higgs:F1", speed, "en")
    body = text_body_ids(env)
    if tag is None:
        assert len(body) == 3
    else:
        assert body[1] == env.ids[tag] and len(body) == 4


def test_empty_text_returns_empty_without_loading(env):
    assert env.engine.synth(" <||> ", "higgs:F1", 1.0, "en").size == 0
    assert FakeSession.instances == []


def test_repeat_guard_empty_result_retries_with_new_seed(env, monkeypatch):
    seeds = []
    real = np.random.default_rng
    monkeypatch.setattr(higgs.np.random, "default_rng", lambda s: seeds.append(s) or real(s))
    # attempts 0 and 1: stuck on one token from the start (repeat guard empties the result)
    Heads.plan = staticmethod(lambda a, s: 9 if a < 2 else (EOC if s == 30 else 1 + s % 2))
    wav = env.engine.synth(LONG_TEXT, "higgs:F1", 1.0, "en")
    assert len(wav) > 0
    base = higgs.stable_seed("higgs:F1", "en")
    assert seeds == [base, base + 1, base + 2]
    assert len(sessions("audio_tokenizer.onnx")[0].calls) == 1     # only the good take is decoded


def test_repeat_guard_far_below_expected_retries(env):
    # attempt 0: 70 varied frames then a stuck loop -> 63 usable frames, expected ~190
    def plan(attempt, step):
        if attempt == 0:
            return 1 + step % 2 if step < 70 else 9
        return EOC if step == 120 else 1 + step % 2
    Heads.plan = staticmethod(plan)
    env.engine.synth(LONG_TEXT, "higgs:F1", 1.0, "en")
    codec = sessions("audio_tokenizer.onnx")[0].calls
    assert len(codec) == 1 and codec[0]["audio_codes"].shape[2] == 120 + 1 + 6 - 7


def test_all_attempts_short_returns_best(env):
    def plan(attempt, step):
        limit = {0: 40, 1: 70, 2: 50}[attempt]
        return 1 + step % 2 if step < limit else 9
    Heads.plan = staticmethod(plan)
    wav = env.engine.synth(LONG_TEXT, "higgs:F1", 1.0, "en")
    assert len(wav) > 0
    codec = sessions("audio_tokenizer.onnx")[0].calls
    assert len(codec) == 1 and codec[0]["audio_codes"].shape[2] == 70 - 7    # the 70-step take
    assert FakeSession.attempt == 2


def test_all_attempts_empty_raises(env):
    Heads.plan = staticmethod(lambda a, s: 9)
    with pytest.raises(RuntimeError, match="no speech"):
        env.engine.synth(LONG_TEXT, "higgs:F1", 1.0, "en")
    assert FakeSession.attempt == higgs.ATTEMPTS - 1
    # an empty result never reaches the codec (it raises on a 0-frame input)
    assert all(not s.calls for s in sessions("audio_tokenizer.onnx"))


def test_frame_cap_stops_a_runaway(env):
    Heads.plan = staticmethod(lambda a, s: 1 + s % 2)       # never ends
    wav = env.engine.synth("hello", "higgs:F1", 1.0, "en")
    cap = higgs.max_frames_for("hello")
    assert len(sessions("audio_heads.onnx")[0].calls) == cap
    assert sessions("audio_tokenizer.onnx")[0].calls[0]["audio_codes"].shape[2] == cap - 7
    assert len(wav) > 0


def test_nan_logits_raise_clear_error(env):
    Heads.nan = True
    with pytest.raises(RuntimeError, match="NaN"):
        env.engine.synth("hello", "higgs:F1", 1.0, "en")


def test_nan_audio_raises_clear_error(env):
    Heads.plan = staticmethod(eoc_plan(30))
    Heads.nan = "codec"
    with pytest.raises(RuntimeError, match="NaN"):
        env.engine.synth("hello", "higgs:F1", 1.0, "en")


def test_same_voice_and_language_use_the_same_seed(env, monkeypatch):
    seeds = []
    real = np.random.default_rng
    monkeypatch.setattr(higgs.np.random, "default_rng", lambda s: seeds.append(s) or real(s))
    Heads.plan = staticmethod(eoc_plan(20))
    env.engine.synth("hello", "higgs:F1", 1.0, "en")
    env.engine.synth("world", "higgs:F1", 1.0, "en")
    env.engine.synth("hello", "higgs:F1", 1.0, "hr")
    assert seeds[0] == seeds[1] != seeds[2]


@pytest.mark.parametrize("voice", ["F1", "supertonic:F1", "higgs:", "higgs:a b", ""])
def test_invalid_voice(env, voice):
    with pytest.raises(ValueError, match="voice"):
        env.engine.synth("hello", voice, 1.0, "en")


@pytest.mark.parametrize("lang", ["xx", "ja", "EN"])
def test_invalid_language(env, lang):
    with pytest.raises(ValueError, match="language"):
        env.engine.synth("hello", "higgs:F1", 1.0, lang)


def test_unknown_voice_without_clips(env):
    with pytest.raises(ValueError, match="voice"):
        env.engine.synth("hello", "higgs:Nobody", 1.0, "en")


# ---------------------------------------------------------------- clip lookup / fallback

def ref_frames(env):
    return sessions("audio_embed.onnx")[0].calls[0]["codes"].shape[1] - 7


def test_exact_clip_then_fallbacks(env, caplog):
    Heads.plan = staticmethod(eoc_plan(20))
    env.engine.synth("hello", "higgs:F1", 1.0, "hr")
    assert ref_frames(env) == 10                                    # own hr clip
    FakeSession.instances = []
    env.engine._loaded = False
    with caplog.at_level(logging.WARNING, logger="vibetts.higgs"):
        env.engine.synth("hello", "higgs:F1", 1.0, "bs")            # bs -> hr clip
        assert ref_frames(env) == 10
        FakeSession.instances = []
        env.engine._loaded = False
        env.engine.synth("hello", "higgs:F1", 1.0, "fi")            # no fi, no fallback -> en
        assert ref_frames(env) == 12
    assert any("no bs clip" in r.message and "hr" in r.message for r in caplog.records)


def test_fallback_warns_once_per_voice_and_language(env, caplog):
    Heads.plan = staticmethod(eoc_plan(20))
    with caplog.at_level(logging.WARNING, logger="vibetts.higgs"):
        for _ in range(3):
            env.engine.synth("hello", "higgs:F1", 1.0, "fi")
        env.engine.synth("hello", "higgs:F1", 1.0, "et")
    warned = [r for r in caplog.records if "no fi clip" in r.message]
    assert len(warned) == 1 and len([r for r in caplog.records if "no et clip" in r.message]) == 1


def test_no_english_clip_uses_any_clip(env):
    Heads.plan = staticmethod(eoc_plan(20))
    (env.presets / "F1" / "en.codes.npy").unlink()
    env.engine.synth("hello", "higgs:F1", 1.0, "fi")
    assert ref_frames(env) == 10


# ---------------------------------------------------------------- custom voices

def test_add_validate_list_delete_custom_voice(env, tmp_path):
    wav = tmp_path / "me.wav"
    write_wav(wav, 5.0, rate=44100, channels=2)
    assert higgs.validate_clip(wav) == pytest.approx(5.0, abs=0.01)
    higgs.add_custom_voice(env.voices, "Anna", wav, " Dobar   dan ", "hr")
    higgs.add_custom_voice(env.voices, "Anna", wav, "Hello", "en")
    higgs.add_custom_voice(env.voices, "bob_2", wav, "Hello", "en")
    assert (env.voices / "Anna" / "hr.wav").is_file()
    assert (env.voices / "Anna" / "hr.txt").read_text(encoding="utf-8") == "Dobar dan"
    assert higgs.custom_voices(env.voices) == ["Anna", "bob_2"]
    assert higgs.voice_ids(env.voices) == [f"higgs:{p}" for p in HIGGS_PRESETS] + [
        "higgs:Anna", "higgs:bob_2"]
    assert higgs.voice_languages("higgs:Anna", env.voices) == ["en", "hr"]
    assert higgs.voice_languages("bob_2", env.voices) == ["en"]
    assert higgs.voice_languages("higgs:F1", env.voices) == ["en", "hr"]
    assert higgs.voice_languages("higgs:Nobody", env.voices) == []
    higgs.delete_custom_voice(env.voices, "Anna")
    assert higgs.custom_voices(env.voices) == ["bob_2"]
    higgs.delete_custom_voice(env.voices, "Anna")                   # already gone: no error


def test_custom_voices_ignores_junk(env):
    (env.voices / "empty").mkdir()
    (env.voices / "bad name!").mkdir()
    (env.voices / "bad name!" / "en.txt").write_text("x")
    (env.voices / "F1").mkdir()                                     # preset name never custom
    (env.voices / "F1" / "en.txt").write_text("x")
    (env.voices / "F1" / "en.wav").write_bytes(b"")
    (env.voices / "stray.txt").write_text("x")
    (env.voices / "notrans").mkdir()
    (env.voices / "notrans" / "en.wav").write_bytes(b"")
    assert higgs.custom_voices(env.voices) == []
    assert higgs.custom_voices(env.tmp / "missing") == []


def test_add_custom_voice_rejects_bad_input(env, tmp_path):
    good = tmp_path / "ok.wav"
    write_wav(good, 4.0)
    with pytest.raises(ValueError, match="name"):
        higgs.add_custom_voice(env.voices, "bad name", good, "x", "en")
    with pytest.raises(ValueError, match="name"):
        higgs.add_custom_voice(env.voices, "", good, "x", "en")
    with pytest.raises(ValueError, match="built-in"):
        higgs.add_custom_voice(env.voices, "F1", good, "x", "en")
    with pytest.raises(ValueError, match="built-in"):
        higgs.add_custom_voice(env.voices, "m2", good, "x", "en")
    with pytest.raises(ValueError, match="transcript"):
        higgs.add_custom_voice(env.voices, "Ann", good, "  ", "en")
    with pytest.raises(ValueError, match="anguage"):
        higgs.add_custom_voice(env.voices, "Ann", good, "x", "ja")
    short = tmp_path / "short.wav"
    write_wav(short, 1.0)
    with pytest.raises(ValueError, match="too short"):
        higgs.add_custom_voice(env.voices, "Ann", short, "x", "en")
    assert higgs.custom_voices(env.voices) == []
    assert not (env.voices / "Ann").exists()


def test_validate_clip_errors(tmp_path):
    long = tmp_path / "long.wav"
    write_wav(long, 31.0, rate=8000)
    with pytest.raises(ValueError, match="too long"):
        higgs.validate_clip(long)
    edge = tmp_path / "edge.wav"
    write_wav(edge, 3.0, rate=8000)
    assert higgs.validate_clip(edge) == pytest.approx(3.0)
    eight = tmp_path / "eight.wav"
    with wave.open(str(eight), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(bytes(8000 * 5))
    with pytest.raises(ValueError, match="16-bit"):
        higgs.validate_clip(eight)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not a wav file at all")
    with pytest.raises(ValueError, match="WAV"):
        higgs.validate_clip(junk)
    with pytest.raises(ValueError):
        higgs.validate_clip(tmp_path / "missing.wav")


def test_delete_preset_refused(env):
    for name in ("F1", "m1"):
        with pytest.raises(ValueError, match="built-in"):
            higgs.delete_custom_voice(env.voices, name)
    with pytest.raises(ValueError):
        higgs.delete_custom_voice(env.voices, "../x")
    assert (env.presets / "F1" / "en.txt").exists()


def test_custom_voice_codes_cached_and_encoder_released(env, tmp_path):
    Heads.plan = staticmethod(eoc_plan(20))
    wav = tmp_path / "me.wav"
    write_wav(wav, 4.0, rate=24000)
    higgs.add_custom_voice(env.voices, "Anna", wav, "hello world", "en")
    cache = env.voices / "Anna" / "en.codes.npy"
    assert not cache.exists()
    env.engine.synth("hello", "higgs:Anna", 1.0, "en")
    assert len(sessions("audio_encoder.onnx")) == 1
    enc_in = sessions("audio_encoder.onnx")[0].calls[0]["input_values"]
    assert enc_in.shape == (1, 1, 4 * 24000) and enc_in.dtype == np.float32
    codes = np.load(cache)
    assert codes.dtype == np.int16 and codes.shape == (8, 100) and (codes == 7).all()
    env.engine.synth("world", "higgs:Anna", 1.0, "en")              # cached: no encoder
    assert len(sessions("audio_encoder.onnx")) == 1
    assert env.engine._sess.keys().isdisjoint({"audio_encoder"})   # never kept resident
    # replacing the clip drops the stale cache and re-encodes
    higgs.add_custom_voice(env.voices, "Anna", wav, "hello world", "en")
    assert not cache.exists()
    env.engine.synth("hello", "higgs:Anna", 1.0, "en")
    assert len(sessions("audio_encoder.onnx")) == 2


def test_stale_codes_are_re_encoded_when_wav_is_newer(env, tmp_path):
    Heads.plan = staticmethod(eoc_plan(20))
    wav = tmp_path / "me.wav"
    write_wav(wav, 4.0)
    higgs.add_custom_voice(env.voices, "Anna", wav, "hello world", "en")
    env.engine.synth("hello", "higgs:Anna", 1.0, "en")
    cache = env.voices / "Anna" / "en.codes.npy"
    t = cache.stat().st_mtime
    os.utime(env.voices / "Anna" / "en.wav", (t + 10, t + 10))
    env.engine.synth("hello", "higgs:Anna", 1.0, "en")
    assert len(sessions("audio_encoder.onnx")) == 2


def test_encode_clip_pads_to_whole_frames(env, tmp_path):
    wav = tmp_path / "odd.wav"
    write_wav(wav, 3.5003, rate=22050)
    codes = env.engine.encode_clip(wav)
    n = sessions("audio_encoder.onnx")[0].calls[0]["input_values"].shape[2]
    assert n % higgs.FRAME_SAMPLES == 0 and n >= 84000
    assert codes.shape == (8, n // higgs.FRAME_SAMPLES) and codes.dtype == np.int16


# ---------------------------------------------------------------- cancel (Stop / export Cancel)

def test_synth_stops_within_one_frame_when_cancel_is_set(env):
    import threading

    from vibetts.tts import SynthCancelled

    Heads.plan = staticmethod(eoc_plan(500))
    cancel = threading.Event()
    heads = lambda: sessions("audio_heads.onnx")  # noqa: E731

    class CancelAfter:
        """A callable cancel: fires once the AR loop has run 5 frames."""
        def __call__(self):
            return bool(heads()) and len(heads()[0].calls) >= 5

    assert higgs.SynthCancelled is SynthCancelled
    with pytest.raises(SynthCancelled):
        env.engine.synth("hello", "higgs:F1", 1.0, "en", cancel=CancelAfter())
    assert len(heads()[0].calls) == 5  # checked every frame: stopped at once
    assert env.engine._lock.acquire(False)  # the lock was released
    env.engine._lock.release()
    # an Event that is already set stops before any frame
    cancel.set()
    with pytest.raises(SynthCancelled):
        env.engine.synth("hello", "higgs:F1", 1.0, "en", cancel=cancel)
    assert len(heads()[0].calls) == 5
    # and the engine still works afterwards
    Heads.plan = staticmethod(eoc_plan(30))
    assert len(env.engine.synth("hello", "higgs:F1", 1.0, "en", cancel=threading.Event())) > 0


def test_engine_accepts_cancel_flag(env):
    assert env.engine.accepts_cancel is True
