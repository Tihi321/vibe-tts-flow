import tomllib
from pathlib import Path

from vibetts.config import Config, config_from_dict, load_config, save_config

ROOT = Path(__file__).resolve().parent.parent


def test_example_matches_defaults():
    with open(ROOT / "config.example.toml", "rb") as fh:
        assert config_from_dict(tomllib.load(fh)) == Config()


def test_invalid_values_fall_back():
    cfg = config_from_dict({"speech": {"speed": "fast", "volume": 0.5, "preload": 1},
                            "selection": {"copy_timeout_ms": 2.5}, "ui": "nope"})
    assert cfg.speech.speed == 1.0
    assert cfg.speech.volume == 0.5
    assert cfg.speech.preload is True
    assert cfg.selection.copy_timeout_ms == 400
    assert cfg.ui == Config().ui


def test_save_load_round_trip(tmp_path):
    cfg = Config()
    cfg.hotkey.read = "ctrl+shift+r"
    cfg.speech.voice = "bm_george"
    cfg.speech.speed = 1.25
    cfg.selection.max_chars = 0
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_load_creates_missing(tmp_path):
    path = tmp_path / "sub" / "config.toml"
    assert load_config(path) == Config()
    assert path.exists()


def test_broken_toml_uses_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[speech\nvoice=", encoding="utf-8")
    assert load_config(path) == Config()


def test_croatian_defaults():
    s = Config().speech
    assert (s.omnivoice_dir, s.omnivoice_steps) == ("models/omnivoice", 6)
    assert s.croatian_voice == "omni_hr:default" and s.auto_language is True


def test_omnivoice_steps_clamped():
    assert config_from_dict({"speech": {"omnivoice_steps": 2}}).speech.omnivoice_steps == 4
    assert config_from_dict({"speech": {"omnivoice_steps": 99}}).speech.omnivoice_steps == 16
    assert config_from_dict({"speech": {"omnivoice_steps": 10}}).speech.omnivoice_steps == 10
