# VTF-09 contract: interfaces between work items A, B and C

Items A, B and C build against these names. A name that changes during implementation is reported to the orchestrator, not renamed silently.

## `vibetts/config.py` (owner: B)

```python
HIGGS_PREFIX = "higgs:"
HIGGS_PRESETS = ("F1", "F2", "M1", "M2")              # bundled preset voice names
HIGGS_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
HIGGS_TEMPERATURE_RANGE = (0.3, 1.2)
HIGGS_LANGUAGE_NAMES: dict[str, str]   # the 94 Higgs metadata codes -> English names (no "ja")
HIGGS_LANGUAGES: tuple[str, ...]       # tuple(HIGGS_LANGUAGE_NAMES)
LANGUAGE_NAMES: dict[str, str]         # SUPERTONIC_LANGUAGE_NAMES | HIGGS_LANGUAGE_NAMES
ALL_LANGUAGES: tuple[str, ...]         # "en", the other Supertonic codes in their order, then Higgs-only codes sorted by name

def is_higgs_voice(voice: str) -> bool             # prefix + HIGGS_VOICE_NAME_RE (existence is not checked)
def higgs_voice_name(voice: str) -> str            # "higgs:F1" -> "F1"
def voice_fits_language(voice, lang) -> bool       # higgs -> lang in HIGGS_LANGUAGES
def enabled_languages(s) -> list[str]              # over ALL_LANGUAGES

# DEFAULT_LANGUAGE_VOICES covers ALL_LANGUAGES: the 31 unchanged, Higgs-only codes "" (Off)

# SpeechConfig new fields
higgs_dir: str = "models/higgs-audio-v3"
higgs_voices_dir: str = "models/higgs-voices"
higgs_temperature: float = 0.8
```

## `vibetts/higgs.py` (owner: A)

```python
SAMPLE_RATE = 24000
REPO = "onnx-community/higgs-audio-v3-tts-4b"
REVISION = "ecc25d6cc6e48909401bfa7ce5ac95e84f41f331"
SUBDIR = "cpu_int4"
MODEL_FILES: tuple[tuple[str, int], ...]   # paths relative to the model dir, e.g. "cpu_int4/llm_decoder.onnx", plus "LICENSE"
DOWNLOAD_BYTES: int                        # sum of MODEL_FILES sizes
CLIP_FALLBACK: dict[str, str]              # language -> language whose clip to use when the voice has none
MIN_CLIP_SECONDS, MAX_CLIP_SECONDS = 3.0, 30.0

def models_present(model_dir: Path) -> bool
def presets_dir() -> Path                                   # vibetts/voices/higgs (frozen-app aware)
def custom_voices(voices_dir: Path) -> list[str]            # names, sorted
def voice_ids(voices_dir: Path) -> list[str]                # ["higgs:F1", ..., "higgs:<custom>", ...]
def voice_languages(voice: str, voices_dir: Path) -> list[str]  # languages with an own clip
def validate_clip(wav_path: Path) -> float                  # seconds; ValueError with a user-facing message
def add_custom_voice(voices_dir: Path, name: str, wav_path: Path, transcript: str,
                     lang: str) -> None                     # ValueError: bad name / clip / empty transcript / preset name
def delete_custom_voice(voices_dir: Path, name: str) -> None  # ValueError for a preset

class HiggsEngine:
    def __init__(self, model_dir: Path, voices_dir: Path, temperature: float = 0.8,
                 threads: int = 0): ...                     # threads 0 = min(8, cpu_count // 2)
    def voices(self) -> list[str]: ...                      # voice_ids(voices_dir)
    def synth(self, text: str, voice: str, speed: float,
              lang: str | None = None) -> np.ndarray: ...   # float32 mono, 44.1 kHz
```

## `vibetts/model_setup.py` (owner: A)

```python
def higgs_files() -> tuple[tuple[str, int], ...]
def higgs_missing(dest_dir: Path) -> list[str]
def download_higgs(dest_dir: Path, progress=None, cancel=None) -> None   # same callback types as download_supertonic; OSError when the disk is short of space
def delete_higgs(dest_dir: Path) -> None
# CLI: python -m vibetts.model_setup --higgs
```

## `vibetts/app.py` (owner: B)

```python
MISSING_HIGGS = "Higgs is not installed: download it in Settings > Higgs"
HIGGS_HINT = "Download Higgs in Settings"
def higgs_dir(s) -> Path; def higgs_voices_dir(s) -> Path; def higgs_present(s) -> bool
def higgs_voice_ids(s) -> list[str]
# setup_tab(voice) -> "higgs" for higgs voices; voice_ready / missing_message handle them
# Route.engine can be "higgs"
# EngineHolder.unload_higgs()
# AgentBridge.list_voices(): voices[] entries for Higgs {"id","label","engine":"higgs","languages": list(HIGGS_LANGUAGES)},
#   "languages": [{code, name}] over ALL_LANGUAGES, "higgs_installed": bool
```

## `SettingsHooks` additions (`settings_ui.py`, owner: C; the wiring in `TTSApp._settings_window`, owner: B)

```python
higgs_dir: Callable[[], Path] = lambda: ROOT / "models" / "higgs-audio-v3"
higgs_present: Callable[[], bool] = lambda: False
higgs_changed: Callable[[], None] = lambda: None        # after download / delete / voice add / delete: reload or unload the engine, refresh the tray
higgs_voices_dir: Callable[[], Path] = lambda: ROOT / "models" / "higgs-voices"
```

C calls `higgs.add_custom_voice` / `delete_custom_voice` / `custom_voices` / `validate_clip` directly with `hooks.higgs_voices_dir()`, and `model_setup.download_higgs` / `delete_higgs` / `higgs_missing` with `hooks.higgs_dir()`, the same way the Supertonic tab does. `hooks.voices()` returns the Higgs ids when Higgs is installed. The form key `higgs_temperature` maps to `SpeechConfig.higgs_temperature`. `TABS` gains `"higgs"` (label "Higgs"), placed after `"supertonic"`.

## `TrayHooks` (`tray.py`, owner: B)

`higgs_available: Callable[[], bool] = lambda: False`. The Voices submenus list the Higgs ids that fit the language.

## File ownership

- **A:** `vibetts/higgs.py`, `vibetts/model_setup.py`, `scripts/make_higgs_presets.py`, `vibetts/voices/higgs/**`, `tests/test_higgs.py`, `tests/test_model_setup.py`, `requirements.txt`, `pyproject.toml`, `packaging/VibeTTS.spec`, and `presets.md` in the ticket folder.
- **B:** `vibetts/config.py`, `vibetts/langdetect.py`, `vibetts/tts.py`, `vibetts/app.py`, `vibetts/tray.py`, `vibetts/mcp_server.py`, `vibetts/export.py`, `config.example.toml`, `README.md`, `mcp-changes.md` in the ticket folder, and every other `tests/*` file except those owned by A and C.
- **C:** `vibetts/settings_ui.py`, `tests/test_settings_ui.py`, `vibetts/web/*`, `extension/*`.
- **D:** only `.claude/temp/vtf09dml/` and a new "DirectML" section in `spike-results.md`.
- `plan.md` and `changelog.md` belong to the orchestrator only.
