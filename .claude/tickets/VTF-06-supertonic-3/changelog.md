# VTF-06 changelog

## 2026-10-19 — Supertonic 3 next to Kokoro, all audio at 44.1 kHz

Outcome: VibeTTS has two engines. Kokoro reads English; Supertonic 3 (31 languages, default Croatian) reads everything else. Each read picks the English or other-language profile by a cheap "looks English" guess, which the tray can override. All output is 44.1 kHz.

Repository `vibe-tts-flow`, branch `VTF-06_supertonic-3` (from local `master` at `c6d75f3 NO-TASK: Remove omnivoice`, which the user committed before this work).

### Files

- `vibetts/supertonic.py` (new): port of the MIT `supertonic` SDK 1.3.1 inference path (text normalisation with `<lang>` tokens, duration predictor, text encoder, vector-estimator loop, vocoder). Only numpy + onnxruntime. `SupertonicEngine(model_dir, steps)`, voices `supertonic:F1..M5`, `synth(text, voice, speed, lang)` → float32 44.1 kHz. App speed 1.0 = model speed 1.05. Voice styles load lazily. `MODEL_FILES` (17 files with sizes), pinned `REVISION`, `models_present`.
- `vibetts/model_setup.py`: Supertonic download from Hugging Face at the pinned revision (only missing files, resumable per file, cancel leaves no partial file), `supertonic_missing`, `delete_supertonic`, CLI `--supertonic [--dest]`. Kokoro download unchanged.
- `vibetts/config.py`: engine/profile constants, 31 language codes and names, new `[speech]` keys (`english_engine`, `english_supertonic_voice`, `other_language`, `other_voice`, `language_mode`, `supertonic_dir`, `supertonic_steps`) with validation and fallbacks; Kokoro `voice` must be an English voice; MP3 bitrates 32..320 (MPEG-1).
- `config.example.toml`: the new keys and bitrates.
- `vibetts/tts.py`: 44.1 kHz pipeline (`BLOCK_FRAMES` 4410), Kokoro resampled 24 → 44.1 kHz with soxr, Kokoro limited to American/British English voices, `lang` back on `Engine.synth` / `SpeakRequest`, labels for Supertonic voices.
- `vibetts/textprep.py`: `looks_english(text)` (non-Latin script or non-English diacritics → not English; otherwise share of English function words; short/ambiguous → English).
- `vibetts/export.py`: `lang` back on `ExportRequest` / `ExportResult`; WAV and MP3 at 44.1 kHz.
- `vibetts/app.py`: `Route` decided once per read from `language_mode` + `looks_english`; explicit voice/lang rules for export and MCP (Kokoro voice + non-English lang is an error); `EngineHolder` loads Kokoro and Supertonic lazily (`RoutingEngine` dispatches by voice id); fallback to Kokoro English when Supertonic is missing (logged once, overlay hint); Settings hooks `supertonic_dir/present/changed`; tray hooks for language mode and both voices; `say` CLI routes like the hotkey.
- `vibetts/tray.py`: `Language` (Auto / English / other), `English voice` (Kokoro en-US / en-GB, Supertonic), `<language> voice` menus.
- `vibetts/settings_ui.py`: Voice tab with English, Other language and Language frames (scrolls); new Supertonic tab (status, download/cancel/delete, open folder, quality steps 4..16, licence note); Export tab language combo; form mapping for the new keys; About text.
- `vibetts/mcp_server.py`: `lang` (31 codes) on `speak`, `synthesize_to_file` and batch items; descriptions say 44.1 kHz.
- `requirements.txt`, `pyproject.toml`, `packaging/VibeTTS.spec`: `soxr` back; hidden imports `vibetts.supertonic`, `soxr`.
- `README.md`: engines and profiles, language detection, Supertonic download, 44.1 kHz, bitrates, MCP params, troubleshooting, layout, licences.
- Tests: `tests/test_supertonic.py` (new), `tests/test_model_setup.py`, `test_config.py`, `test_tts.py`, `test_textprep.py`, `test_export.py`, `test_app.py`, `test_mcp_server.py`, `test_settings_ui.py`.

### Orchestrator integration fixes

- `app.py`: the Settings hooks were wired conditionally (a workaround while slices ran in parallel); now plain keyword arguments, and the matching test no longer skips.
- `settings_ui.py`: `show(tab="supertonic", download=True)` (tray "Download voice model…" when only Supertonic is missing) started the Kokoro download; it now starts the Supertonic download. Regression test added.

### Deviations from the plan

- Step 0 was done by the user (`c6d75f3` on local `master`); the branch was cut from local `master`, not `origin/master`.
- `Route` has a fourth field `fell_back`; `plan_speech` returns `(chunks, Route)`.
- Language codes/names live in `config.py` (shared with validation and UI), not in `supertonic.py`.
- The port drops characters the model's unicode indexer does not know (the SDK passes -1 to the model).
- A missing Supertonic only for the other-language profile is a soft need: no Settings auto-open, but the tray shows "Download voice model…".

### Verification

- `.venv/Scripts/python -m pytest -q` → `381 passed`.
- pyflakes on `vibetts` and `tests`: only pre-existing findings (`export.py` lameenc probe import, an unused import in `test_app.py`).
- Port vs SDK, same text/voice/steps: identical durations (7.04 s hr, 5.57 s en), RMS within ~1.5%, RTF ~0.11.
- Real CLI with the real models: `--say` English → Kokoro `af_heart`, 44.1 kHz WAV, 8.1x real time; Croatian → `supertonic:F1`, MPEG-1 MP3, 9.1x real time.
- Headless MCP over HTTP (`--mcp-only`, port 7799): `list_voices` (38 voices, 31 languages, supertonic_installed true); `synthesize_to_file` German with `supertonic:M2`; `af_heart` + `lang: hr` rejected with a clear error; `synthesize_batch` 2/2 ok.

Not run: manual GUI QA (hotkey on English and Croatian selections, tray Language switch and voice menus, Settings > Supertonic download from an empty folder, Settings layout at 1080p by eye), the PyInstaller build.

### Follow-ups / known limitations

- `looks_english` misreads English jargon with no function words ("deploy kubernetes cluster now") as not English; Tray > Language > English fixes it.
- VN Studio used OmniVoice voice design (`create_voice`) and must move to the 10 Supertonic voices + `lang`.
- The Supertone GitHub repo is archived; the model revision is pinned.
- The installed exe at `D:\Programs\VibeTTS` needs a rebuild.

### Commit status

Committed on `VTF-06_supertonic-3` (on top of the OmniVoice removal `c6d75f3`), pushed, PR opened against `master`.
