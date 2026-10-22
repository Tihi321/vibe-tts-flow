# VTF-09 changelog

## 2026-10-22: VTF-09, Higgs Audio v3 added as a third engine (live, export, MCP)

Higgs Audio v3 TTS (Boson AI, 4B, community ONNX `cpu_int4`) reads the same way as Kokoro and Supertonic: in the per-language voice picker, hotkey live reads, export, MCP, the tray, and the web reader and extension. On CPU it runs slower than real time (RTF about 1.7 to 2.4 through the app), so live reads pause between chunks. DirectML on the Radeon 8060S was tested and does not work (wrong output, 3 to 5 times slower).

### Repository `vibe-tts-flow`

**State:** committed directly on **`master`** at the user's request, not on `VTF-09_higgs-audio-v3`. Mid-session the user committed the plan to `VTF-09_higgs-audio-v3` (74cda5f), merged it as PR #8 (103f2de), and switched to `master` and pulled. The implementation went on in that working tree.

**New files**

| File | What it does |
|---|---|
| `vibetts/higgs.py` | The Higgs engine, see below. |
| `vibetts/voices/higgs/{F1,F2,M1,M2}/<lang>.codes.npy` + `.txt` | 119 preset clips (about 0.9 MB) covering the Supertonic languages except ja; F2/et was dropped by the ASR check. |
| `scripts/make_higgs_presets.py` | Dev tool: builds the presets with Supertonic, encodes them with Higgs, and checks them with ASR. |
| `tests/test_higgs.py` | 62 tests with fake ONNX sessions, plus cancel tests. |

`vibetts/higgs.py` in detail:
- **Port:** a port of the ONNX repo's `inference.py` that needs only numpy, onnxruntime and `tokenizers`.
- **Pinned files:** `MODEL_FILES` at revision `ecc25d6c…`, 3,849,550,721 bytes.
- **CPU and memory:** 8 intra-op threads. The 650 MB `audio_encoder` is loaded only to encode a clip, then released.
- **Text and speed:** control tags are stripped from the read text. App speed maps to the weak speed tags only when |speed − 1| ≥ 0.15.
- **Seeds and retries:** the seed is fixed per voice and language. A take that is empty or stopped early by the repeat guard is retried with the next seed, up to 3 attempts.
- **Guards:** NaN or all-zero output raises a clear error, and the codec never gets 0 frames.
- **Clips:** a clip is looked up for the exact language, then `CLIP_FALLBACK` (bs/sr/cnr/sl→hr, mk→bg, …), then English.
- **Custom voices:** add, delete and validate (16-bit PCM WAV, 3 to 30 s). Their codes are cached next to the WAV.
- **Cancel:** `synth(cancel=)` is checked on every frame.

**Changed files**

| File | What changed |
|---|---|
| `vibetts/config.py` | `HIGGS_*` constants, 100 Higgs language codes, `LANGUAGE_NAMES`, and `ALL_LANGUAGES` (101 codes). `language_voices` now covers all of them: Higgs-only languages default to Off, and their row accepts only Off or a Higgs voice. New keys `higgs_dir`, `higgs_voices_dir` and `higgs_temperature` (clamped 0.3..1.2). Old configs load unchanged. |
| `vibetts/langdetect.py` | `detectable_languages()`. Candidates are limited to the codes py3langid knows; 11 enabled Higgs codes can never be detected. |
| `vibetts/tts.py` | Higgs voice ids and labels ("F1 (Higgs, female)", "<name> (Higgs, custom)"). Adds `SynthCancelled`. The Speaker passes its stop event into engines that accept `cancel`, so Stop ends a Higgs chunk within one frame. |
| `vibetts/app.py` | Routing with the `higgs` engine and its fallback chain (Higgs missing or voice deleted → Supertonic → Kokoro → Kokoro English); `Route.missing` names the missing engine. `RoutingEngine` and `EngineHolder` get a Higgs slot, preload, and unload when no language uses Higgs. `route_request` and the MCP voice check accept Higgs ids. `list_voices` reports `engine: "higgs"` and `higgs_installed`. Settings and tray hooks; `_higgs_changed` does its work on a background thread. The new CLI flags `--voice` and `--lang` feed `--say` and `--doc`. Export cancel is forwarded into Higgs. |
| `vibetts/tray.py` | The Voices submenus list the Higgs voices that fit each language. |
| `vibetts/mcp_server.py` | Descriptions: `higgs:` voices, the `ALL_LANGUAGES` enum, and "slower than real time". |
| `vibetts/model_setup.py` | `higgs_files`, `higgs_missing`, `download_higgs` (checks free disk space first) and `delete_higgs`, plus the CLI `--higgs`. |
| `vibetts/settings_ui.py` | A new **Higgs** tab with download, delete and status, info and licence text, a temperature slider, and a Voices list (presets and own clips) with Add…, Delete and Play test. In the Voice tab the language list covers `ALL_LANGUAGES` and the Engine combo offers Higgs, with red hints for a missing download and for undetectable languages. The Export tab lists all languages and the Higgs voices. |
| `requirements.txt`, `pyproject.toml` | Add `tokenizers`. |
| `packaging/VibeTTS.spec` | Hidden imports `vibetts.higgs` and `tokenizers`; bundles `vibetts/voices/higgs`. |
| `config.example.toml`, `README.md` | Document the third engine, the speed on CPU, the 3.85 GB download, about 5 GB of RAM, custom voices, the CLI flags, undetectable languages, and the licence (personal non-commercial use; no creator grant). |
| `tests/test_app.py`, `test_config.py`, `test_langdetect.py`, `test_mcp_server.py`, `test_model_setup.py`, `test_settings_ui.py`, `test_tts.py` | Extended to cover the changes above. |
| Ticket folder | Adds `contract.md`, `mcp-changes.md` (the `list_voices` shape) and `presets.md` (the ASR table). `spike-results.md` gets a DirectML section. |

### Deviations from the plan

- **Live instead of export-only.** The spike gate measured export-only (RTF 1.37 to 1.78 at best). The user decided Higgs should work "same as other just this model", so it is a full live engine.
- **100 Higgs language codes**, not 94 (101 in `ALL_LANGUAGES`).
- **Cancellation was added.** Found in review: without it, Stop and a new read waited up to about 40 s for the chunk in progress.
- **CLI `--voice` and `--lang` were added.** They did not exist, and the verification steps needed them.
- **Contract additions:**
  - `Route.missing` and `route_for(higgs_ok=)`;
  - `HiggsEngine.encode_clip` and `synth(seed_offset=, cancel=)`;
  - `accepts_cancel`;
  - the `model_setup.HIGGS_MODEL_FILES` alias.
  - `MODEL_FILES` leaves out `chat_template.jinja` and `genai_config.json`, which the loop does not use.
- **Speed tags are approximate.** The spike measured −1% / −7% instead of ×0.85 / ×1.2.

### Verification

Passed:
- `.venv\Scripts\python.exe -m pytest -q tests`: **744 passed, 1 skipped**.
- Model files: `higgs.models_present(models/higgs-audio-v3)` is True and `higgs_missing` is empty. The files were copied from the spike download; `download_higgs` was tested with mocks plus a real fetch of 2 small files.
- CLI: `python -m vibetts --doc hr.txt --out hr_higgs.wav --voice higgs:F1 --lang hr` wrote 19.95 s of audio in 42.97 s (RTF 2.15). On the MP3 run it was 47.7 s (RTF 2.4). Whisper medium transcript: "Stari svijetionik stajao je na rubu litice i više od stotinu godina promatrao sivom more. Svake večeri čuvar se uskim stepenicama penjao do svijetla. Zapisao bi nekoliko tihih reda kao dnevnik prije nego što bi se iznad luke pojavile prve zvijezde."
- Speaker with the real engine (null audio output):
  - first audio after 18.4 s, including about 9 s of model load;
  - `stop()` returned in 0.46 s with no error;
  - the next short read finished in 8.3 s.
- PyInstaller build (`scripts\build.ps1 -SkipTests`) succeeded and produced the 3 zips.
  - The exe bundles the 119 preset clips and `tokenizers`.
  - `dist\VibeTTS\VibeTTS.exe --doc … --voice higgs:M1 --lang hr` wrote 21.9 s of audio in 40.4 s. One take was retried automatically.
  - Whisper transcript of that file: near verbatim.
- HKCU Run `VibeTTS` is unchanged (`"D:\Programs\VibeTTS\VibeTTS.exe" --hidden`).

Not run:
- **Listening test (user):** the spike `listen.html`, the presets, and cross-chunk voice consistency.
- **Live hotkey read** in the running tray app, including audible pauses.
- **The Settings Higgs tab** in the running app: a real 3.85 GB download, Add voice with a real WAV, Play test.
- **MCP calls** against the running app.
- **The web reader and extension** in a browser.
- **The tray Voices menu** in the running app.

### Follow-ups and known limitations

- **CPU only, slower than real time.** Live reads pause between chunks, and each chunk repeats the reference prefill of about 2 s. Caching the reference-prefix KV per voice and language is a possible speed-up.
- **DirectML does not work** with this export: the GQA past-KV is wrong and ConvTranspose fails. See `spike-results.md`.
- **Croatian numbers** ("1.250,00 eura") are misread. Number normalisation is not done.
- **Higgs-only languages have no own preset clips.** They use the `CLIP_FALLBACK` clip or English. 11 codes cannot be auto-detected, and Bosnian Latin text detects as hr.
- **Play test always reads the English sample sentence,** whatever the voice's language.
- **Licence:** the Boson Higgs Audio v3 Research and Non-Commercial License allows personal use only. Distributing the app with Higgs, or any commercial use, needs a licence from Boson AI.

**Commit status:** committed directly on `master` and pushed (the user works on master without a branch for this ticket).
