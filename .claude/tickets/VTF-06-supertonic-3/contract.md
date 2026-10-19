# VTF-06 interface contract (shared by the parallel implementer slices)

Plan: `plan.md` (same folder). Branch `VTF-06_supertonic-3`. Repo: `C:\projects\Personal\vibe-tts-flow`.
Code against these names exactly. If a slice needs a contract change, report it instead of editing another slice's files.

## Already in `vibetts/config.py` (orchestrator, done)

`ENGINES = ("kokoro", "supertonic")`, `LANGUAGE_MODES = ("auto", "english", "other")`,
`SUPERTONIC_PREFIX = "supertonic:"`, `SUPERTONIC_VOICES = ("F1".."F5", "M1".."M5")`,
`SUPERTONIC_STEPS_RANGE = (4, 16)`, `SUPERTONIC_LANGUAGE_NAMES` (code -> English name, 31 entries, "en" first),
`SUPERTONIC_LANGUAGES` (the 31 codes), `is_supertonic_voice(voice)`.
`SpeechConfig` fields: `voice` (Kokoro, English), `english_engine`, `english_supertonic_voice` ("F1".."M5", bare name),
`other_language` (code, not "en"), `other_voice` (bare name), `language_mode`, `supertonic_dir`, `supertonic_steps`.
Config *validation* of these keys is slice B's job.

## Voice ids everywhere (tray, Settings combos, MCP, ExportRequest.voice, SpeakRequest.voice)

- Kokoro: `af_heart`, `bm_george`, ... Only English Kokoro voices (prefix `a` / `b`) are offered from now on.
- Supertonic: `supertonic:F1` .. `supertonic:M5`. Config keys `english_supertonic_voice` / `other_voice` store the bare name (`F1`).

## Slice A: `vibetts/supertonic.py` (new), `vibetts/model_setup.py`

`vibetts/supertonic.py`: top-level imports stdlib + numpy + `.config` only (onnxruntime imported inside `SupertonicEngine.__init__`).

```python
SAMPLE_RATE = 44100
REPO = "Supertone/supertonic-3"
REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
MODEL_FILES: tuple[tuple[str, int], ...]   # (posix path relative to the model dir, size in bytes)
def models_present(model_dir: Path) -> bool  # every MODEL_FILES entry exists with the right size
class SupertonicEngine:
    def __init__(self, model_dir: Path, steps: int = 8): ...
    def voices(self) -> list[str]            # ["supertonic:F1", ..., "supertonic:M5"]
    def synth(self, text: str, voice: str, speed: float, lang: str | None = None) -> np.ndarray
        # mono float32 at 44100 Hz; voice must be a supertonic id (ValueError otherwise);
        # lang None -> "en"; lang not in SUPERTONIC_LANGUAGES -> ValueError;
        # `speed` is the app speed (1.0 = normal): model speed = 1.05 * speed, clamped to the model range.
        # No internal chunking or added silence. Thread-safe (one lock).
```

`vibetts/model_setup.py` additions (Kokoro API unchanged):

```python
SUPERTONIC_BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
def supertonic_files() -> tuple[tuple[str, int], ...]           # == supertonic.MODEL_FILES
def supertonic_missing(dest_dir: Path) -> list[str]              # relative paths not installed
def download_supertonic(dest_dir: Path, progress: ProgressFn | None = None,
                        cancel: threading.Event | None = None) -> None
    # downloads only missing files (subfolders created), progress(done_bytes, total_bytes) over all of them,
    # raises DownloadCancelled / SetupError like the Kokoro download; no partial file left behind
def delete_supertonic(dest_dir: Path) -> None                   # removes MODEL_FILES (and empty subfolders)
# CLI: python -m vibetts.model_setup --supertonic [--dest DIR]
```

## Slice B: everything else in the engine/app layer

- `vibetts/tts.py`: `SAMPLE_RATE = 44100`, `KOKORO_RATE = 24000`, `BLOCK_FRAMES = 4410`.
  `KokoroEngine.synth(text, voice, speed, lang=None)` resamples to 44.1 kHz with `soxr`.
  `Engine.synth(text, voice, speed, lang=None)`. `SpeakRequest(chunks, voice, speed, pause, volume=1.0, device="", lang=None)`.
  `voice_supported` / `voice_label` handle both kinds; `voice_label("supertonic:F1") == "F1 (Supertonic, female)"`.
- `vibetts/export.py`: `ExportRequest(text, out=None, voice=None, speed=None, lang=None, fmt=None, bitrate=None,
  markdown=True, gain=1.0, overwrite=False, title="")`; `ExportResult(path, fmt, seconds, chunks, bytes, voice, lang)`.
  `lang`: None = routed as configured, else one of `SUPERTONIC_LANGUAGES`.
- `vibetts/textprep.py`: `looks_english(text) -> bool`.
- `vibetts/app.py` `SettingsHooks(...)` wiring adds (slice C consumes them):
  - `supertonic_dir: Callable[[], Path]`  -> resolved `speech.supertonic_dir`
  - `supertonic_present: Callable[[], bool]`
  - `supertonic_changed: Callable[[], None]` -> call after a Supertonic download or delete finishes (app unloads the
    engine, preloads if needed, refreshes tray and window)
  - existing `voices()` returns English Kokoro ids + Supertonic ids (the latter only when installed)
  - existing `export_audio(req, progress, cancel)` accepts `req.lang`

## Slice C: `vibetts/settings_ui.py`, `tests/test_settings_ui.py`

Consumes the config fields, constants, voice ids, `ExportRequest.lang`, the SettingsHooks fields above and the
`model_setup` Supertonic functions. Adds the three new `SettingsHooks` fields with harmless defaults
(`supertonic_dir=lambda: ROOT / "models" / "supertonic-3"`, `supertonic_present=lambda: False`,
`supertonic_changed=lambda: None`).
