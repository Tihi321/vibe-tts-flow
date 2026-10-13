import pytest

from vibetts.config import Config
from vibetts.settings_ui import config_from_form, form_from_config, model_choice


def test_round_trip_default():
    cfg = Config()
    assert config_from_form(cfg, form_from_config(cfg)) == cfg


def test_round_trip_customised():
    cfg = Config()
    cfg.hotkey.read = "ctrl+shift+r"
    cfg.speech.model = "D:\\tts\\kokoro-v1.0.onnx"
    cfg.speech.voice = "bf_emma"
    cfg.speech.speed = 1.35
    cfg.speech.sentence_pause = 0
    cfg.speech.device = "Headphones"
    cfg.selection.max_chars = 0
    cfg.selection.restore_clipboard = False
    cfg.ui.pill_position = "top"
    assert config_from_form(cfg, form_from_config(cfg)) == cfg


def test_model_switch_keeps_folder():
    cfg = Config()
    form = form_from_config(cfg)
    assert form["model"] == "int8"
    form["model"] = "fp32"
    assert config_from_form(cfg, form).speech.model == "models/kokoro-v1.0.onnx"


def test_custom_model_kept():
    cfg = Config()
    cfg.speech.model = "other/custom.onnx"
    assert model_choice(cfg.speech.model) == "custom.onnx"
    assert config_from_form(cfg, form_from_config(cfg)).speech.model == "other/custom.onnx"


def test_voice_display_string_parsed():
    form = form_from_config(Config())
    form["voice"] = "bm_george  George (British English, male)"
    assert config_from_form(Config(), form).speech.voice == "bm_george"


@pytest.mark.parametrize("key,value", [
    ("hotkey_read", "ctrl+bogus"), ("speed", "3"), ("speed", "0.1"), ("volume", "1.5"),
    ("sentence_pause", "-1"), ("copy_timeout_ms", "10"), ("max_chars", "-1"),
    ("pill_position", "left"), ("voice", ""), ("model", "bogus"),
])
def test_invalid_values(key, value):
    form = form_from_config(Config())
    form[key] = value
    with pytest.raises(ValueError):
        config_from_form(Config(), form)
