import tomllib
from pathlib import Path

from vibetts.config import (Config, config_from_dict, export_folder, load_config, resolve_path,
                            save_config)

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


def test_unknown_voice_falls_back():
    assert config_from_dict({"speech": {"voice": "bm_george"}}).speech.voice == "bm_george"
    for bad in ("", "omni:alma", "heart"):
        assert config_from_dict({"speech": {"voice": bad}}).speech.voice == "af_heart"


def test_old_keys_are_dropped_on_save(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[speech]\nvoice = "bf_emma"\nengine = "omnivoice"\nauto_language = true\n',
                    encoding="utf-8")
    cfg = load_config(path)
    assert cfg.speech.voice == "bf_emma"
    save_config(cfg, path)
    text = path.read_text(encoding="utf-8")
    assert "engine" not in text and "auto_language" not in text


def test_export_mcp_defaults():
    cfg = Config()
    assert (cfg.export.folder, cfg.export.format, cfg.export.mp3_bitrate) == ("", "mp3", 64)
    assert (cfg.mcp.enabled, cfg.mcp.port, cfg.mcp.max_chars) == (False, 7723, 200000)


def test_export_mcp_round_trip(tmp_path):
    cfg = Config()
    cfg.export.folder = "out/audio"
    cfg.export.format = "wav"
    cfg.export.mp3_bitrate = 128
    cfg.mcp.enabled = True
    cfg.mcp.port = 8123
    cfg.mcp.max_chars = 0
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_export_mcp_invalid_values_fall_back():
    cfg = config_from_dict({"export": {"format": "ogg", "mp3_bitrate": 100},
                            "mcp": {"port": 80, "max_chars": -5}})
    assert cfg.export.format == "wav"
    assert cfg.export.mp3_bitrate == 96
    assert cfg.mcp.port == 7723
    assert cfg.mcp.max_chars == 0
    assert config_from_dict({"mcp": {"port": 70000}}).mcp.port == 7723


def test_export_folder_default_and_relative(tmp_path):
    cfg = Config()
    assert export_folder(cfg) == Path.home() / "Documents" / "VibeTTS"
    cfg.export.folder = "audio_out"
    assert export_folder(cfg) == resolve_path("audio_out")
    assert export_folder(cfg).is_absolute()
    cfg.export.folder = str(tmp_path)
    assert export_folder(cfg) == tmp_path
