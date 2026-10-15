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


def test_croatian_defaults():
    s = Config().speech
    assert (s.omnivoice_dir, s.omnivoice_steps) == ("models/omnivoice", 6)
    assert s.engine == "kokoro" and s.voice == "af_heart"
    assert s.omni_voice == "omni:default" and s.croatian_voice == "" and s.auto_language is True


def test_omnivoice_steps_clamped():
    assert config_from_dict({"speech": {"omnivoice_steps": 2}}).speech.omnivoice_steps == 4
    assert config_from_dict({"speech": {"omnivoice_steps": 99}}).speech.omnivoice_steps == 16
    assert config_from_dict({"speech": {"omnivoice_steps": 10}}).speech.omnivoice_steps == 10


def test_migrates_old_omni_voice_in_voice():
    cfg = config_from_dict({"speech": {"voice": "omni_hr:alma", "croatian_voice": "omni_hr:default"}})
    s = cfg.speech
    assert (s.engine, s.omni_voice, s.voice) == ("omnivoice", "omni:alma", "af_heart")
    assert s.croatian_voice == ""  # the old code ignored it when `voice` was an omni voice
    bare = config_from_dict({"speech": {"voice": "omni_hr"}}).speech
    assert (bare.engine, bare.omni_voice) == ("omnivoice", "omni:default")


def test_migrates_old_croatian_voice_and_keeps_kokoro():
    s = config_from_dict({"speech": {"voice": "bf_emma", "croatian_voice": "omni_hr:zora"}}).speech
    assert (s.engine, s.voice, s.croatian_voice, s.omni_voice) == (
        "kokoro", "bf_emma", "omni:zora", "omni:default")
    old_default = config_from_dict({"speech": {"croatian_voice": "omni_hr:default"}}).speech
    assert old_default.croatian_voice == "omni:default" and old_default.engine == "kokoro"


def test_new_config_is_not_touched_and_bad_values_fall_back():
    s = config_from_dict({"speech": {"engine": "omnivoice", "voice": "bm_george",
                                     "omni_voice": "omni:alma", "croatian_voice": "omni:zora"}}).speech
    assert (s.engine, s.voice, s.omni_voice, s.croatian_voice) == (
        "omnivoice", "bm_george", "omni:alma", "omni:zora")
    bad = config_from_dict({"speech": {"engine": "magic", "omni_voice": "af_heart",
                                       "croatian_voice": "af_heart"}}).speech
    assert (bad.engine, bad.omni_voice, bad.croatian_voice) == ("kokoro", "omni:default", "")


def test_migrated_config_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[speech]\nvoice = "omni_hr:alma"\n', encoding="utf-8")
    cfg = load_config(path)
    save_config(cfg, path)
    again = load_config(path)
    assert again == cfg and again.speech.engine == "omnivoice"
    assert "omni_hr" not in path.read_text(encoding="utf-8")


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
