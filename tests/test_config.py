import tomllib
from pathlib import Path

import pytest

from vibetts.config import (MP3_BITRATES, SUPERTONIC_LANGUAGES, Config, config_from_dict,
                            export_folder, load_config, resolve_path, save_config, snap_bitrate)

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
    # only English Kokoro voices (a* / b*) are valid for `voice`
    for bad in ("", "omni:alma", "heart", "ff_siwis", "ef_dora", "jf_alpha", "supertonic:F1"):
        assert config_from_dict({"speech": {"voice": bad}}).speech.voice == "af_heart"


def test_profile_defaults():
    s = Config().speech
    assert (s.english_engine, s.voice, s.english_supertonic_voice) == ("kokoro", "af_heart", "F1")
    assert (s.other_language, s.other_voice, s.language_mode) == ("hr", "F1", "auto")
    assert (s.supertonic_dir, s.supertonic_steps) == ("models/supertonic-3", 8)


def test_profile_values_round_trip(tmp_path):
    cfg = Config()
    s = cfg.speech
    s.english_engine, s.english_supertonic_voice = "supertonic", "M3"
    s.other_language, s.other_voice, s.language_mode = "de", "F5", "other"
    s.supertonic_dir, s.supertonic_steps = "D:/models/st", 12
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    assert load_config(path) == cfg


@pytest.mark.parametrize("key,bad", [
    ("english_engine", "omnivoice"), ("english_engine", ""),
    ("language_mode", "croatian"), ("language_mode", "AUTO"),
    ("other_language", "xx"), ("other_language", "en"), ("other_language", ""),
    ("english_supertonic_voice", "F6"), ("english_supertonic_voice", "supertonic:F1"),
    ("other_voice", "X1"), ("other_voice", "f1"),
])
def test_invalid_profile_values_fall_back(key, bad):
    default = getattr(Config().speech, key)
    assert getattr(config_from_dict({"speech": {key: bad}}).speech, key) == default


def test_valid_profile_values_are_kept():
    s = config_from_dict({"speech": {"english_engine": "supertonic", "language_mode": "english",
                                     "other_language": "sl", "english_supertonic_voice": "M5",
                                     "other_voice": "M1"}}).speech
    assert (s.english_engine, s.language_mode, s.other_language) == ("supertonic", "english", "sl")
    assert (s.english_supertonic_voice, s.other_voice) == ("M5", "M1")
    for code in SUPERTONIC_LANGUAGES:
        if code != "en":
            assert config_from_dict({"speech": {"other_language": code}}).speech.other_language == code


def test_supertonic_steps_clamped():
    for given, want in [(1, 4), (4, 4), (8, 8), (16, 16), (40, 16), (-3, 4)]:
        assert config_from_dict({"speech": {"supertonic_steps": given}}).speech.supertonic_steps == want
    assert config_from_dict({"speech": {"supertonic_steps": "many"}}).speech.supertonic_steps == 8


def test_mp3_bitrates_are_mpeg1():
    assert MP3_BITRATES == (32, 48, 64, 96, 128, 160, 192, 256, 320)
    assert config_from_dict({"export": {"mp3_bitrate": 320}}).export.mp3_bitrate == 320
    assert config_from_dict({"export": {"mp3_bitrate": 192}}).export.mp3_bitrate == 192
    assert config_from_dict({"export": {"mp3_bitrate": 1000}}).export.mp3_bitrate == 320
    assert config_from_dict({"export": {"mp3_bitrate": 230}}).export.mp3_bitrate == 256
    assert snap_bitrate(64) == 64 and snap_bitrate(1) == 32


def test_old_keys_are_dropped_on_save(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[speech]\nvoice = "bf_emma"\nengine = "omnivoice"\nauto_language = true\n',
                    encoding="utf-8")
    cfg = load_config(path)
    assert cfg.speech.voice == "bf_emma"
    save_config(cfg, path)
    text = path.read_text(encoding="utf-8")
    assert "[speech]\nengine =" not in text and "\nengine =" not in text
    assert "auto_language" not in text


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


PINNED_ID = "injaiiefkhoimpkeandbjfbnnjddaceh"


def test_reader_defaults():
    r = Config().reader
    assert (r.enabled, r.port, r.library_folder, r.max_chars) == (True, 7724, "library", 500000)
    assert r.extension_ids == [PINNED_ID]
    assert Config().reader.extension_ids is not Config().reader.extension_ids  # not shared


def test_reader_round_trip(tmp_path):
    cfg = Config()
    cfg.reader.enabled = False
    cfg.reader.port = 8124
    cfg.reader.library_folder = "D:/books"
    cfg.reader.max_chars = 0
    cfg.reader.extension_ids = [PINNED_ID, "a" * 32]
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    assert load_config(path) == cfg
    cfg.reader.extension_ids = []
    save_config(cfg, path)
    assert load_config(path).reader.extension_ids == []


def test_reader_invalid_values_fall_back(caplog):
    cfg = config_from_dict({"reader": {"port": 80, "max_chars": -1}})
    assert cfg.reader.port == 7724
    assert cfg.reader.max_chars == 0
    assert config_from_dict({"reader": {"port": 70000}}).reader.port == 7724
    bad = config_from_dict({"reader": {"extension_ids": "nope"}})
    assert bad.reader.extension_ids == [PINNED_ID]  # wrong type: default kept
    some = config_from_dict({"reader": {"extension_ids": [PINNED_ID, "short", "B" * 32]}})
    assert some.reader.extension_ids == [PINNED_ID, "b" * 32]  # invalid dropped, case folded
    dupes = config_from_dict({"reader": {"extension_ids": ["b" * 32, "b" * 32]}})
    assert dupes.reader.extension_ids == ["b" * 32]
    mixed = config_from_dict({"reader": {"extension_ids": [PINNED_ID, 5]}})
    assert mixed.reader.extension_ids == [PINNED_ID]  # not all strings: default kept
    assert "not a Chrome extension id" in caplog.text


def test_export_folder_default_and_relative(tmp_path):
    cfg = Config()
    assert export_folder(cfg) == Path.home() / "Documents" / "VibeTTS"
    cfg.export.folder = "audio_out"
    assert export_folder(cfg) == resolve_path("audio_out")
    assert export_folder(cfg).is_absolute()
    cfg.export.folder = str(tmp_path)
    assert export_folder(cfg) == tmp_path
