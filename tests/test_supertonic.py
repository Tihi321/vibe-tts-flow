import json
import sys
import types

import numpy as np
import pytest

from vibetts import supertonic
from vibetts.config import SUPERTONIC_LANGUAGES, SUPERTONIC_VOICES


def test_preprocess_language_token_and_period():
    assert supertonic.preprocess("Dobar dan", "hr") == "<hr>Dobar dan.</hr>"
    assert supertonic.preprocess("Hello there!", "en") == "<en>Hello there!</en>"


def test_preprocess_normalises():
    out = supertonic.preprocess("Hi “there”  — a_b [x] , ok \U0001f600 @ home", "en")
    assert out == '<en>Hi "there" - a b x , ok at home.</en>'
    assert supertonic.preprocess("e.g., this", "en") == "<en>for example, this.</en>"
    assert supertonic.preprocess('say ""hi""', "en") == '<en>say "hi"</en>'


def test_preprocess_nfkd_splits_diacritics():
    out = supertonic.preprocess("č", "hr")  # c-caron -> c + combining caron
    assert out == "<hr>č.</hr>"


def test_model_speed_mapping_and_clamping():
    assert supertonic.model_speed(1.0) == pytest.approx(1.05)
    assert supertonic.model_speed(1.2) == pytest.approx(1.26)
    assert supertonic.model_speed(0.1) == supertonic.MIN_SPEED
    assert supertonic.model_speed(10) == supertonic.MAX_SPEED


def test_model_files_table():
    paths = [p for p, _ in supertonic.MODEL_FILES]
    assert len(paths) == 17 and len(set(paths)) == 17
    assert "LICENSE" in paths and "onnx/vocoder.onnx" in paths
    for v in SUPERTONIC_VOICES:
        assert f"voice_styles/{v}.json" in paths
    assert all(size > 0 for _, size in supertonic.MODEL_FILES)


def test_models_present(tmp_path):
    assert not supertonic.models_present(tmp_path)
    for rel, size in supertonic.MODEL_FILES:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
        with open(f, "r+b") as fh:
            fh.truncate(size)
    assert supertonic.models_present(tmp_path)
    (tmp_path / "onnx" / "tts.json").write_bytes(b"short")
    assert not supertonic.models_present(tmp_path)
    (tmp_path / "onnx" / "tts.json").unlink()
    assert not supertonic.models_present(tmp_path)


# ---------------------------------------------------------------- engine with fake sessions

CHUNK = 512 * 6
LATENT = 24 * 6


class FakeSession:
    instances: list = []

    def __init__(self, path, sess_options=None, providers=None):
        self.name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        self.calls: list = []
        FakeSession.instances.append(self)

    def run(self, _outputs, feed):
        self.calls.append(feed)
        if self.name == "duration_predictor.onnx":
            return [np.array([2.0], dtype=np.float32)]
        if self.name == "text_encoder.onnx":
            return [np.zeros((1, 4, feed["text_ids"].shape[1]), dtype=np.float32)]
        if self.name == "vector_estimator.onnx":
            return [feed["noisy_latent"] * 0.5]
        if self.name == "vocoder.onnx":
            n = feed["latent"].shape[2] * CHUNK
            return [np.full((1, n), 0.25, dtype=np.float64)]
        raise AssertionError(self.name)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    FakeSession.instances = []
    fake = types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99),
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=1),
        InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    (tmp_path / "onnx").mkdir()
    (tmp_path / "voice_styles").mkdir()
    (tmp_path / "onnx" / "tts.json").write_text(json.dumps({
        "ae": {"sample_rate": 44100, "base_chunk_size": 512},
        "ttl": {"chunk_compress_factor": 6, "latent_dim": 24}}))
    indexer = [-1] * 0x300
    for i, ch in enumerate("<>/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?-"):
        indexer[ord(ch)] = i
    (tmp_path / "onnx" / "unicode_indexer.json").write_text(json.dumps(indexer))
    style = {"style_ttl": {"dims": [1, 2, 3], "data": [[[0.5] * 3] * 2]},
             "style_dp": {"dims": [1, 2], "data": [[1.0, 2.0]]}}
    (tmp_path / "voice_styles" / "F1.json").write_text(json.dumps(style))
    (tmp_path / "voice_styles" / "M2.json").write_text(json.dumps(style))
    return supertonic.SupertonicEngine(tmp_path, steps=5)


def session(name):
    return next(s for s in FakeSession.instances if s.name == name)


def test_voices(engine):
    voices = engine.voices()
    assert voices[0] == "supertonic:F1" and voices[-1] == "supertonic:M5" and len(voices) == 10


def test_synth_output_and_pipeline(engine):
    wav = engine.synth("Dobar dan", "supertonic:F1", 1.0, "hr")
    assert wav.dtype == np.float32 and wav.ndim == 1
    # duration 2.0 s / 1.05 -> latent length ceil(dur * 44100 / CHUNK)
    dur = np.float32(2.0) / np.float32(1.05)
    latent_len = int((float(dur) * 44100 + CHUNK - 1) // CHUNK)
    assert wav.size == latent_len * CHUNK
    assert np.allclose(wav, 0.25)

    dp = session("duration_predictor.onnx").calls[0]
    ids = dp["text_ids"]
    assert ids.dtype == np.int64 and ids.shape == (1, len("<hr>Dobar dan.</hr>"))
    assert dp["text_mask"].shape == (1, 1, ids.shape[1])
    assert dp["style_dp"].shape == (1, 2)

    est = session("vector_estimator.onnx").calls
    assert len(est) == 5
    assert [float(c["current_step"][0]) for c in est] == [0, 1, 2, 3, 4]
    assert all(float(c["total_step"][0]) == 5 for c in est)
    assert est[0]["noisy_latent"].shape == (1, LATENT, latent_len)
    assert est[0]["latent_mask"].shape == (1, 1, latent_len)
    assert est[0]["noisy_latent"].dtype == np.float32
    assert est[0]["style_ttl"].shape == (1, 2, 3)
    assert session("vocoder.onnx").calls[0]["latent"].shape == (1, LATENT, latent_len)


def test_synth_speed_changes_length(engine):
    normal = engine.synth("hello", "supertonic:F1", 1.0)
    fast = engine.synth("hello", "supertonic:F1", 1.5)
    assert fast.size < normal.size


def test_synth_default_language_is_english(engine):
    engine.synth("hello", "supertonic:M2", 1.0)
    ids = session("duration_predictor.onnx").calls[0]["text_ids"]
    assert ids.shape[1] == len("<en>hello.</en>")


def test_synth_drops_unknown_characters(engine):
    engine.synth("a中b", "supertonic:F1", 1.0)  # CJK char is -1 in the indexer
    ids = session("duration_predictor.onnx").calls[0]["text_ids"]
    assert ids.shape[1] == len("<en>ab.</en>")
    assert (ids >= 0).all()


@pytest.mark.parametrize("voice", ["af_heart", "supertonic:Z9", "supertonic:", "F1", ""])
def test_synth_invalid_voice(engine, voice):
    with pytest.raises(ValueError, match="voice"):
        engine.synth("hello", voice, 1.0)


@pytest.mark.parametrize("lang", ["xx", "na", "EN", ""])
def test_synth_invalid_language(engine, lang):
    if lang == "":
        engine.synth("hello", "supertonic:F1", 1.0, lang)  # empty means default
        return
    with pytest.raises(ValueError, match="language"):
        engine.synth("hello", "supertonic:F1", 1.0, lang)


def test_all_languages_accepted(engine):
    for lang in SUPERTONIC_LANGUAGES:
        assert engine.synth("hi", "supertonic:F1", 1.0, lang).size > 0


def test_styles_load_lazily_once(engine, tmp_path):
    assert engine._styles == {}
    engine.synth("hi", "supertonic:F1", 1.0)
    engine.synth("hi", "supertonic:F1", 1.0)
    assert list(engine._styles) == ["F1"]


def test_steps_clamped(tmp_path, engine):
    eng = supertonic.SupertonicEngine(engine._dir, steps=0)
    assert eng._steps == supertonic.STEPS_RANGE[0]
