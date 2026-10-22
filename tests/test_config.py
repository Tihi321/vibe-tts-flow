import tomllib
from pathlib import Path

import pytest

from vibetts.config import (ALL_LANGUAGES, DEFAULT_LANGUAGE_VOICES, KOKORO_LANGUAGES, MP3_BITRATES,
                            SUPERTONIC_LANGUAGES, Config, config_from_dict, enabled_languages,
                            export_folder, kokoro_voice_lang, load_config, resolve_path,
                            save_config, snap_bitrate, voice_fits_language)

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
    cfg.speech.language_voices["en"] = "bm_george"
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


def test_kokoro_voice_lang():
    assert KOKORO_LANGUAGES == {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it",
                                "p": "pt"}
    assert kokoro_voice_lang("af_heart") == "en" and kokoro_voice_lang("bm_george") == "en"
    assert kokoro_voice_lang("ef_dora") == "es" and kokoro_voice_lang("pf_dora") == "pt"
    for bad in ("", "heart", "jf_alpha", "zf_xiaobei", "supertonic:F1", "af"):
        assert kokoro_voice_lang(bad) is None
    assert voice_fits_language("ef_dora", "es") and not voice_fits_language("ef_dora", "en")
    assert voice_fits_language("supertonic:M2", "hr") and not voice_fits_language("supertonic:X", "hr")


def test_language_voice_defaults():
    s = Config().speech
    assert tuple(s.language_voices) == ALL_LANGUAGES and len(s.language_voices) == 101
    assert s.language_voices["en"] == "af_heart"
    assert all(v == "supertonic:F1" for c, v in s.language_voices.items()
               if c != "en" and c in SUPERTONIC_LANGUAGES)
    assert all(v == "" for c, v in s.language_voices.items() if c not in SUPERTONIC_LANGUAGES)
    assert s.language_voices == DEFAULT_LANGUAGE_VOICES and s.read_language == "auto"
    assert enabled_languages(s) == list(SUPERTONIC_LANGUAGES)  # Higgs-only languages start Off
    assert (s.supertonic_dir, s.supertonic_steps) == ("models/supertonic-3", 8)
    for gone in ("voice", "english_engine", "english_supertonic_voice", "other_language",
                 "other_voice", "language_mode"):
        assert not hasattr(s, gone)


def test_missing_languages_are_filled_with_defaults():
    s = config_from_dict({"speech": {"language_voices": {"es": "ef_dora", "xx": "supertonic:F1"}}}).speech
    assert s.language_voices["es"] == "ef_dora"
    assert s.language_voices["en"] == "af_heart" and s.language_voices["hr"] == "supertonic:F1"
    assert "xx" not in s.language_voices and len(s.language_voices) == 101


def test_language_voices_round_trip(tmp_path):
    cfg = Config()
    s = cfg.speech
    s.language_voices.update({"en": "supertonic:M3", "es": "ef_dora", "de": "", "fr": "ff_siwis"})
    s.read_language = "es"
    s.supertonic_dir, s.supertonic_steps = "D:/models/st", 12
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    text = path.read_text(encoding="utf-8")
    assert "[speech.language_voices]" in text and 'es = "ef_dora"' in text and 'de = ""' in text
    assert load_config(path) == cfg
    assert enabled_languages(cfg.speech)[:3] == ["en", "ko", "ja"] and "de" not in enabled_languages(cfg.speech)


@pytest.mark.parametrize("lang,bad", [
    ("hr", "af_heart"), ("en", "ef_dora"), ("es", "af_heart"), ("hr", "ef_dora"),
    ("en", "jf_alpha"), ("de", "heart"), ("fr", "supertonic:F6"), ("it", "omni:alma"),
    ("en", "supertonic"),
])
def test_invalid_language_voice_falls_back(lang, bad):
    s = config_from_dict({"speech": {"language_voices": {lang: bad}}}).speech
    assert s.language_voices[lang] == DEFAULT_LANGUAGE_VOICES[lang]


def test_valid_language_voices_are_kept():
    voices = {"en": "bf_emma", "es": "em_alex", "fr": "ff_siwis", "hi": "hm_omega",
              "it": "if_sara", "pt": "pm_santa", "hr": "supertonic:M5", "en2": "x"}
    s = config_from_dict({"speech": {"language_voices": voices}}).speech
    for code, voice in voices.items():
        if code != "en2":
            assert s.language_voices[code] == voice


def test_non_table_language_voices_use_defaults():
    assert config_from_dict({"speech": {"language_voices": "af_heart"}}).speech.language_voices \
        == DEFAULT_LANGUAGE_VOICES
    assert config_from_dict({"speech": {"language_voices": {"en": 5}}}).speech.language_voices \
        == DEFAULT_LANGUAGE_VOICES


def test_all_languages_off_turns_english_back_on():
    s = config_from_dict({"speech": {"language_voices": {c: "" for c in SUPERTONIC_LANGUAGES},
                                     "read_language": "de"}}).speech
    assert enabled_languages(s) == ["en"] and s.language_voices["en"] == "af_heart"
    assert s.read_language == "auto"


def test_read_language_must_be_auto_or_enabled():
    assert config_from_dict({"speech": {"read_language": "hr"}}).speech.read_language == "hr"
    assert config_from_dict({"speech": {"read_language": "auto"}}).speech.read_language == "auto"
    for bad in ("xx", "", "AUTO", "croatian"):
        assert config_from_dict({"speech": {"read_language": bad}}).speech.read_language == "auto"
    off = {"language_voices": {"hr": ""}, "read_language": "hr"}
    assert config_from_dict({"speech": off}).speech.read_language == "auto"


def test_migration_english_kokoro_and_other_language():
    s = config_from_dict({"speech": {"voice": "bm_george", "english_engine": "kokoro",
                                     "other_language": "de", "other_voice": "M2",
                                     "language_mode": "auto", "speed": 1.2}}).speech
    assert s.language_voices["en"] == "bm_george" and s.language_voices["de"] == "supertonic:M2"
    assert s.language_voices["hr"] == "supertonic:F1" and s.read_language == "auto"
    assert s.speed == 1.2


def test_migration_supertonic_english():
    s = config_from_dict({"speech": {"voice": "af_heart", "english_engine": "supertonic",
                                     "english_supertonic_voice": "M3"}}).speech
    assert s.language_voices["en"] == "supertonic:M3"
    assert s.language_voices["hr"] == "supertonic:F1" and s.read_language == "auto"


def test_migration_language_modes():
    old = {"voice": "af_heart", "other_language": "sl", "other_voice": "F4"}
    english = config_from_dict({"speech": {**old, "language_mode": "english"}}).speech
    assert english.read_language == "en" and english.language_voices["sl"] == "supertonic:F4"
    other = config_from_dict({"speech": {**old, "language_mode": "other"}}).speech
    assert other.read_language == "sl" and other.language_voices["sl"] == "supertonic:F4"
    auto = config_from_dict({"speech": {**old, "language_mode": "auto"}}).speech
    assert auto.read_language == "auto"


def test_migration_old_defaults_and_bad_old_values():
    s = config_from_dict({"speech": {"voice": "ef_dora", "other_language": "en"}}).speech
    assert s.language_voices == DEFAULT_LANGUAGE_VOICES  # not an English voice: default
    s = config_from_dict({"speech": {"language_mode": "other"}}).speech
    assert s.read_language == "hr"  # the old default other language


def test_new_shape_wins_over_old_keys():
    s = config_from_dict({"speech": {"voice": "bm_george", "language_voices": {"en": "af_nova"}}}).speech
    assert s.language_voices["en"] == "af_nova"


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
    path.write_text('[speech]\nvoice = "bf_emma"\nengine = "omnivoice"\nauto_language = true\n'
                    'other_language = "de"\nother_voice = "M1"\nlanguage_mode = "other"\n',
                    encoding="utf-8")
    cfg = load_config(path)
    assert cfg.speech.language_voices["en"] == "bf_emma"
    assert cfg.speech.language_voices["de"] == "supertonic:M1" and cfg.speech.read_language == "de"
    save_config(cfg, path)
    text = path.read_text(encoding="utf-8")
    for gone in ("\nengine =", "auto_language", "other_language", "other_voice", "language_mode",
                 "english_engine", "\nvoice ="):
        assert gone not in text
    assert "[speech.language_voices]" in text and 'en = "bf_emma"' in text
    assert load_config(path) == cfg


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


# ---------------------------------------------------------------- Higgs (VTF-09)

def test_higgs_language_tables():
    from vibetts.config import (HIGGS_LANGUAGE_NAMES, HIGGS_LANGUAGES, LANGUAGE_NAMES,
                                SUPERTONIC_LANGUAGE_NAMES)

    assert HIGGS_LANGUAGES == tuple(HIGGS_LANGUAGE_NAMES) and "ja" not in HIGGS_LANGUAGES
    assert HIGGS_LANGUAGE_NAMES["hr"] == "Croatian" and HIGGS_LANGUAGE_NAMES["bs"] == "Bosnian"
    assert set(LANGUAGE_NAMES) == set(SUPERTONIC_LANGUAGE_NAMES) | set(HIGGS_LANGUAGES)
    assert ALL_LANGUAGES[:31] == SUPERTONIC_LANGUAGES and len(set(ALL_LANGUAGES)) == len(ALL_LANGUAGES)
    only = ALL_LANGUAGES[31:]
    assert not set(only) & set(SUPERTONIC_LANGUAGES) and set(only) <= set(HIGGS_LANGUAGES)
    assert [LANGUAGE_NAMES[c] for c in only] == sorted(LANGUAGE_NAMES[c] for c in only)


def test_higgs_voice_ids():
    from vibetts.config import higgs_voice_name, is_higgs_voice

    assert is_higgs_voice("higgs:F1") and is_higgs_voice("higgs:my-voice_2")
    assert not is_higgs_voice("higgs:") and not is_higgs_voice("higgs:a b")
    assert not is_higgs_voice("higgs:" + "x" * 33) and not is_higgs_voice("F1")
    assert higgs_voice_name("higgs:F1") == "F1"
    assert voice_fits_language("higgs:F1", "bs") and voice_fits_language("higgs:F1", "hr")
    assert not voice_fits_language("higgs:F1", "ja")
    assert not voice_fits_language("supertonic:F1", "bs")


def test_higgs_only_language_rows_are_validated():
    s = config_from_dict({"speech": {"language_voices": {
        "bs": "higgs:M1", "sr": "supertonic:F1", "mk": "af_heart", "ca": "higgs:Anna",
        "hr": "higgs:F2", "ja": "higgs:F1"}}}).speech
    assert s.language_voices["bs"] == "higgs:M1" and s.language_voices["ca"] == "higgs:Anna"
    assert s.language_voices["sr"] == "" and s.language_voices["mk"] == ""  # do not fit: Off
    assert s.language_voices["hr"] == "higgs:F2"
    assert s.language_voices["ja"] == "supertonic:F1"  # Higgs does not read Japanese: default
    assert "bs" in enabled_languages(s) and "sr" not in enabled_languages(s)


def test_old_config_loads_unchanged():
    old = {c: ("af_heart" if c == "en" else "supertonic:F1") for c in SUPERTONIC_LANGUAGES}
    old["es"] = "ef_dora"
    s = config_from_dict({"speech": {"language_voices": old}}).speech
    assert all(s.language_voices[c] == v for c, v in old.items())
    assert all(s.language_voices[c] == "" for c in ALL_LANGUAGES[31:])


def test_higgs_settings_defaults_clamp_and_round_trip(tmp_path):
    s = Config().speech
    assert (s.higgs_dir, s.higgs_voices_dir, s.higgs_temperature) == (
        "models/higgs-audio-v3", "models/higgs-voices", 0.8)
    assert config_from_dict({"speech": {"higgs_temperature": 5}}).speech.higgs_temperature == 1.2
    assert config_from_dict({"speech": {"higgs_temperature": 0.0}}).speech.higgs_temperature == 0.3
    assert config_from_dict({"speech": {"higgs_temperature": 0.5}}).speech.higgs_temperature == 0.5
    cfg = Config()
    cfg.speech.language_voices["bs"] = "higgs:F1"
    cfg.speech.higgs_temperature = 0.6
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_example_config_has_the_higgs_keys():
    text = (Path(__file__).resolve().parent.parent / "config.example.toml").read_text(encoding="utf-8")
    for key in ("higgs_dir", "higgs_voices_dir", "higgs_temperature"):
        assert key in text
    s = config_from_dict(tomllib.loads(text)).speech
    assert s.higgs_temperature == 0.8
