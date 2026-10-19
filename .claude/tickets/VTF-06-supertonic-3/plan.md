# VTF-06: Supertonic 3 next to Kokoro, everything at 44.1 kHz

Branch `VTF-06_supertonic-3`, cut from `master`. Spike results: `spike-results.md` (same folder).

## Goal

VibeTTS gets a second engine, **Supertonic 3** (31 languages incl. Croatian, CPU, ONNX, ~11x real time), next to **Kokoro**. The user sets up two **profiles**:

- **English profile:** engine Kokoro or Supertonic, plus a voice of that engine.
- **Other-language profile:** a Supertonic language (one of the 30 non-English codes, default `hr`) plus a Supertonic voice.

Each read decides once, up front: text that **looks English** uses the English profile, anything else uses the other-language profile. A tray / Settings switch can force one profile instead (`auto` / `english` / `other`).

All audio is **44.1 kHz**: Supertonic natively, Kokoro resampled up from 24 kHz with `soxr`. One output stream, one export format.

## Decisions (from the user, 2026-10-19)

- Keep both engines (OmniVoice is gone, Kokoro stays).
- English default and other-language default are configured separately.
- No Croatian-specific detection; instead English vs. not English.
- Supertonic offers all 31 languages via a picker.
- 44.1 kHz output for both engines.

## Step 0: land the OmniVoice removal

- [x] The user committed the removal on local `master` as `c6d75f3 NO-TASK: Remove omnivoice` (not pushed).
- [x] Branch `VTF-06_supertonic-3` created from local `master` at `c6d75f3`, not from `origin/master`, so the removal is included. Baseline: 215 tests pass.

## Implementation status (2026-10-19)

Done on `VTF-06_supertonic-3` by three parallel implementers against `contract.md`: A (supertonic.py, model_setup), B (config/tts/textprep/export/app/mcp_server/tray/README/packaging), C (settings_ui), plus orchestrator integration fixes. Details: `changelog.md`.

- [x] 1. Config · [x] 2. Supertonic engine · [x] 3. Engines, routing, 44.1 kHz · [x] 4. Model download · [x] 5. UI (Settings, tray) · [x] 6. MCP · [x] 7. Packaging, docs · [x] 8. Tests
- [x] Verification: 381 tests pass; real CLI runs (en → Kokoro, hr → Supertonic, 44.1 kHz WAV and MPEG-1 MP3); headless MCP over HTTP (list_voices, de export, Kokoro+hr rejected, batch).
- [ ] Manual GUI QA (hotkey on English/Croatian selections, tray Language switch, Settings > Supertonic download from scratch) — not run.

Deviations: `Route` carries `fell_back`; `plan_speech` returns `(chunks, Route)`; language codes live in `config.py` (not `supertonic.py`); `supertonic.py` drops characters the model's indexer does not know (the SDK would pass -1).

## 1. Config (`config.py`, `config.example.toml`)

New `[speech]` keys (flat, like the existing ones):

| Key | Default | Meaning |
|---|---|---|
| `english_engine` | `"kokoro"` | `kokoro` \| `supertonic`: who reads English |
| `voice` | `"af_heart"` | Kokoro voice for English (existing key; only `a*`/`b*` English voices are offered) |
| `english_supertonic_voice` | `"F1"` | Supertonic voice for English when `english_engine = "supertonic"` |
| `other_language` | `"hr"` | one of Supertonic's 30 non-English codes |
| `other_voice` | `"F1"` | Supertonic voice for the other language |
| `language_mode` | `"auto"` | `auto` (detect) \| `english` \| `other` |
| `supertonic_dir` | `"models/supertonic-3"` | model files, downloaded from Settings |
| `supertonic_steps` | `8` | 4..16, clamped (fewer = faster) |

- Validation: unknown engine / mode / language / voice fall back to the defaults with a log warning (same pattern as `_validate_export`). `_validate_speech` keeps the Kokoro-voice check.
- `MP3_BITRATES` becomes the MPEG-1 set valid at 44.1 kHz: `(32, 48, 64, 96, 128, 160, 192, 256, 320)`; default stays 64.

## 2. Supertonic engine (`vibetts/supertonic.py`, new)

- Port the inference path of the `supertonic` SDK 1.3.1 (MIT, credit in the module docstring): `UnicodeProcessor` text normalisation + `<lang>` tokens, duration predictor, text encoder, vector-estimator loop (`total_steps`), vocoder, voice-style JSON loading. Only numpy + onnxruntime: no `huggingface_hub`, no `soundfile`, no SDK dependency.
- No internal chunking or inserted silence: VibeTTS already chunks (`textprep`) and adds its own pauses.
- `SupertonicEngine(model_dir, steps)` with `voices()` → `["supertonic:F1", …, "supertonic:M5"]` and `synth(text, voice, speed, lang)` → float32 44.1 kHz. Thread-safe like `KokoroEngine` (one lock).
- `LANGUAGES` (31 codes) and `LANGUAGE_NAMES` (`hr` → "Croatian", …) for the UI and MCP.
- `models_present(model_dir)`.
- Speed: app speed 1.0 maps to the SDK default 1.05 (it sounds slow at 1.0); clamp to the SDK range.

## 3. Engines, routing, audio (`tts.py`, `app.py`, `textprep.py`)

- `tts.SAMPLE_RATE = 44100`. `KokoroEngine.synth` resamples 24 kHz → 44.1 kHz with `soxr` (`HQ`). `BLOCK_FRAMES` = 4410 (100 ms).
- Voice ids: Kokoro `af_heart` (unchanged), Supertonic `supertonic:F1`. `voice_supported` / `voice_label` handle both ("F1 (Supertonic, female)").
- `Engine.synth(text, voice, speed, lang)` gets `lang` back (Kokoro ignores it). `SpeakRequest` and `ExportRequest` / `ExportResult` get `lang` back.
- `textprep.looks_english(text)`: cheap guess. Not English when it has letters outside the Latin script or non-English diacritics (č ć đ š ž ä ö ü é à ñ …), or few common English function words. Short or ambiguous text counts as English. Unit-tested on English, Croatian, Slovene, German, Spanish, Russian samples.
- `Route(engine, voice, lang)` decided once per read: `language_mode` + `looks_english` → English profile (`en`, Kokoro voice or Supertonic voice) or other profile (`other_language`, `other_voice`).
- `EngineHolder` loads Kokoro and/or Supertonic lazily (one engine each), `RoutingEngine` dispatches by voice id (pattern from VTF-03, git `d9415e5`). Preload loads the engine(s) of the configured profile(s) when their files are present.
- Missing model: a read that needs Supertonic when it is not downloaded falls back to Kokoro English with a one-time log warning and an overlay hint "Download Supertonic in Settings".

## 4. Model download (`model_setup.py`, Settings)

- Supertonic files from `huggingface.co/Supertone/supertonic-3` at the pinned revision `724fb5abbf5502583fb520898d45929e62f02c0b`: `onnx/{duration_predictor,text_encoder,vector_estimator,vocoder}.onnx`, `onnx/tts.json`, `onnx/unicode_indexer.json`, `voice_styles/{F1..F5,M1..M5}.json`, `LICENSE` (~400 MB) into `models/supertonic-3/`. Same downloader as Kokoro (resume, progress, cancel); `python -m vibetts.model_setup --supertonic` for the CLI.

## 5. UI

- **Settings > Voice:** "English" frame (engine radio Kokoro / Supertonic, voice combo of that engine); "Other language" frame (language combo, 30 entries by name; voice combo F1..M5); "Language" combo `Auto (detect English)` / `Always English` / `Always <other language>`; speed, pause, volume, device as today.
- **Settings > Supertonic** (new tab, like the Kokoro tab): download / delete / status, quality steps slider (4..16), licence note (OpenRAIL-M).
- **Settings > Export:** language combo `Auto` / `English` / 30 languages, plus the voice options; Auto uses the same routing as the hotkey.
- **Tray:** `Language` submenu (Auto / English / <other>), `English voice` (Kokoro voices grouped en-US / en-GB, then Supertonic), `<Other> voice` (F1..M5), `Speed`. "Download voice model…" also covers Supertonic when the current profiles need it.

## 6. MCP (`mcp_server.py`, `AgentBridge`)

- `lang` comes back on `speak`, `synthesize_to_file` and batch items: any of the 31 codes. Omitted = the configured routing (auto detect).
- `voice` accepts Kokoro ids and `supertonic:F1`.. ids; a Kokoro voice with a non-English `lang` is an error.
- `list_voices` → `{voices: [{id, label, engine, languages}], languages: [{code, name}], english: {engine, voice}, other: {language, voice}, language_mode, supertonic_installed}`.
- Descriptions say 44.1 kHz instead of 24 kHz.

## 7. Packaging, docs

- `requirements.txt` / `pyproject.toml`: add `soxr` back. Spec: hidden import `vibetts.supertonic`, `soxr`, `soxr.soxr_ext`; no model files bundled.
- README: features, two engines and profiles, language detection, Supertonic download, 44.1 kHz, MP3 bitrates, MCP params; licence line (Supertonic 3: OpenRAIL-M).

## 8. Tests

- `test_supertonic.py`: text processing (`<hr>` tokens, normalisation), synth with tiny fake ONNX sessions (shapes/loop/steps), voices, missing files.
- `test_textprep.py`: `looks_english` cases.
- `test_tts.py`: Kokoro resampling to 44.1 kHz (length ratio), voice ids/labels for both engines.
- `test_config.py`: new keys, defaults match `config.example.toml`, invalid values fall back, steps clamped, bitrates.
- `test_app.py`: routing matrix (mode × english/non-English text × english_engine), fallback when Supertonic is missing, EngineHolder lazy loading per family, export/MCP with `lang`, tray menus.
- `test_settings_ui.py`, `test_mcp_server.py`, `test_export.py` (44.1 kHz WAV header, MPEG-1 MP3).

## Verification

- Full pytest green.
- Real run (`--say` / `--doc --out`): English text → Kokoro `af_heart`, Croatian text → Supertonic `hr`, both 44.1 kHz WAV and MP3; time to first audio and real-time factor logged for both.
- Manual: hotkey on English and Croatian selections, tray language switch, Settings download of Supertonic, MCP `list_voices` / `synthesize_to_file` with `lang: "hr"`.

## Out of scope

- Voice cloning / custom Supertonic voice styles (Voice Builder is a paid service).
- More than one "other" language profile at a time (the language picker switches it).
- VN Studio changes (its own repo).
