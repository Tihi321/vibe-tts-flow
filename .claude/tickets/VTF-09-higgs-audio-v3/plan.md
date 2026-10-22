# VTF-09: Higgs Audio v3 as a third engine (next to Kokoro and Supertonic)

Ticket folder: `.claude/tickets/VTF-09-higgs-audio-v3/`. Branch: `VTF-09_higgs-audio-v3`, cut from `origin/master` (61b5636). The interfaces between the parallel work items are in `contract.md`, in this folder.

Revision 2 (2026-10-22, after the spike): Higgs is a **full engine, the same as the others**: live hotkey reads, export, MCP, tray and the per-language picker. The user decided this after seeing the spike numbers ("should be same as other just this model"). The spike measured it on this CPU at RTF 1.4–1.8, so live reads pause between chunks. A DirectML test on the Radeon 8060S (item D) checks whether the GPU reaches real time.

## Context

VibeTTS has two engines:
- **Kokoro:** English plus 5 more languages.
- **Supertonic 3:** 31 languages, including Croatian.

Since VTF-08 there is one voice per language (`speech.language_voices`), with py3langid detection. The user wants **Higgs Audio v3 TTS** (Boson AI, a 4B model, 24 kHz output, about 94 languages, Croatian in its top tier) added on the same terms.

Spike facts (`spike-results.md`):
- **Model.** Community ONNX `onnx-community/higgs-audio-v3-tts-4b` @ `ecc25d6cc6e48909401bfa7ce5ac95e84f41f331`, folder `cpu_int4/`, 3,849,529,018 bytes. The file sizes are in spike-results.
- **Speed.** It loads in 5 s. On CPU the best is 8 ORT threads: RTF 1.37–1.55 zero-shot and 1.65–1.78 with a clone.
- **Memory.** 4.4–4.8 GB of RAM, and up to 7.5 GB with `audio_encoder` loaded.
- **Voices need a same-language reference clip.**
  - Zero-shot Croatian is a seed lottery (ASR 0.00–1.00).
  - A Croatian reference gives ASR 0.92–1.00.
  - An English reference reading Croatian is poor.
- **Reference code.** It does not run as shipped (it imports transformers and librosa). Patched copy: `.claude/temp/vtf09/scripts/higgs_patched.py`. The repeat guard can empty the output, so the port retries with another seed.
- **Speed tags.** They are weak: −1% / −7% instead of the documented ×0.85 / ×1.2.
- **Numbers.** "1.250,00 eura" is misread. Normalising numbers is a follow-up.
- **Licence.** The Boson Higgs Audio v3 Research and Non-Commercial License allows personal and hobbyist use. An end-user app or product, or any revenue, needs a commercial licence. There is **no** Creator Use Grant in this licence file.

User decisions:
1. Voices: bundled presets, plus the user's own clips (WAV and transcript) added in Settings.
2. Higgs works the same as the other engines (live, export, MCP). This revision replaces the export-only choice.
3. Expand languages: languages only Higgs covers (bs, sr, mk, sq, ca, …) join the picker.

## Step 0: spike

- [x] Done (2026-10-22). The gate measured export-only; the user overrode it to live (see Revision 2). Details: [spike-results.md](spike-results.md). Listening test (user): pending.

## Work items (parallel, see `contract.md` for the interfaces)

### A. Engine, model download, presets (`vibetts/higgs.py`, `vibetts/model_setup.py`, `scripts/make_higgs_presets.py`, `vibetts/voices/higgs/`)

- [x] **Port.** `vibetts/higgs.py` ports `higgs_patched.py`. Add numpy, onnxruntime and `tokenizers` only.
  - Credit the module, with the ONNX repo and the Higgs licence note.
  - `MODEL_FILES` at the pinned revision is the `cpu_int4/` runtime subset plus the root `LICENSE`.
- [x] **`HiggsEngine`.** Mirror `SupertonicEngine`.
  - One lock.
  - `intra_op_num_threads = min(8, os.cpu_count() // 2)`. The spike found 8 best and more threads slower.
  - `audio_encoder` loads lazily, only to encode a clip that has no cached codes. Unload it after encoding.
  - `synth`:
    1. Strip `<|` and `|>` from the text.
    2. Add the speed tag when |speed−1| ≥ 0.15 (approximate, as documented).
    3. Build the clone prompt with the voice's clip for `lang`.
    4. Run the AR loop with a fixed seed per voice and language.
    5. Retry with seed+1 (up to 3 tries) when the output is empty or stopped by the repeat guard far below the expected length.
    6. Raise a clear error on NaN.
    7. Resample to 44.1 kHz with `tts.resample`.
- [x] **Voice clips.**
  - A voice is a set of reference clips, one per language: `<voice>/<lang>.codes.npy` (int16 `[8, T]`) + `<lang>.txt`.
  - Clip lookup for `lang`: the exact language, then `CLIP_FALLBACK` (bs/sr/cnr/sl→hr, mk→bg, ca/gl→es, …), then `en`. The fallback logs a warning once.
- [x] **Presets.** `F1`, `F2`, `M1`, `M2`, shipped as codes and transcripts in `vibetts/voices/higgs/` (a few KB each).
  - `scripts/make_higgs_presets.py` (a dev tool) generates them. It synthesises a neutral sentence about 10 s long in each of the 31 Supertonic languages with Supertonic (same voice name), encodes it with Higgs's `audio_encoder`, and ASR-checks a Higgs test read where Whisper is available.
  - Drop clips that fail, and record the table in `presets.md` in the ticket folder.
- [x] **User clips.** Stored in `models/higgs-voices/<name>/<lang>.wav` + `.txt`. Codes are cached on first use. Functions: `add_custom_voice`, `delete_custom_voice`, `custom_voices`, and `validate_clip` (3–30 s).
- [x] **Download.** `model_setup`: `higgs_files/higgs_missing/download_higgs/delete_higgs` plus the CLI `--higgs`, mirroring Supertonic. Check free disk space (4 GB) first.
- [x] **Dependencies and packaging.**
  - Add `tokenizers` to `requirements.txt` and `pyproject.toml`.
  - `packaging/VibeTTS.spec`: hidden import `vibetts.higgs`, `tokenizers` data, and `vibetts/voices/higgs` as data.
- [x] **Tests.** `tests/test_higgs.py`, with fake ONNX sessions as in `test_supertonic.py`:
  - sanitising, speed tags, prompt layout, delay pattern and its reversal, EOC stop, repeat guard and retry, NaN guard;
  - clip fallback, the codes cache, and custom-voice add/delete/validate.
  - Also extend `tests/test_model_setup.py`.

### B. Config, detection, routing, app, tray, MCP, docs (`config.py`, `langdetect.py`, `tts.py`, `app.py`, `tray.py`, `mcp_server.py`, `config.example.toml`, `README.md`)

- [x] **Config.** Higgs ids, `HIGGS_LANGUAGES`, `ALL_LANGUAGES` and `LANGUAGE_NAMES` (see contract). `language_voices` covers `ALL_LANGUAGES`:
  - the 31 keep their defaults;
  - Higgs-only languages default to Off;
  - validation checks that the voice fits the language;
  - old configs load unchanged.
  - New keys: `higgs_dir`, `higgs_voices_dir`, `higgs_temperature` (clamped).
- [x] **Detection.** Candidates are only enabled codes that py3langid knows. Undetectable ones are reachable only through Read "Always X" or an explicit `lang`.
- [x] **Routing.**
  - `RoutingEngine` gets a Higgs slot, and `EngineHolder._ensure_higgs`/`unload_higgs` warm up with one short read.
  - `voice_ready`, `missing_message` and `setup_tab("higgs")` handle Higgs voices.
  - If Higgs is missing, fall back to the language's Supertonic voice, then Kokoro, then Kokoro English, as today.
  - Preload when a readable language uses Higgs.
  - Unload Higgs when no language uses it any more, and when its files are deleted.
- [x] **App.**
  - `route_request` accepts `higgs:` voices for any `HIGGS_LANGUAGES` code.
  - The language check in `route_request` uses `ALL_LANGUAGES`.
  - `TTSApp._voices()` includes the Higgs ids when installed.
  - New hooks: `higgs_changed` and the custom-voice hooks.
- [x] **Tray.** Voices submenus show the Higgs voices that fit each language.
- [x] **MCP.** `list_voices`: Higgs entries `engine: "higgs"`, `languages: HIGGS_LANGUAGES`, `higgs_installed`, `languages` = `LANGUAGE_NAMES`.
  - Update the `mcp_server.py` descriptions.
  - Record the shape change in `mcp-changes.md`.
- [x] **`tts.voice_label`.** "F1 (Higgs, female)", "<name> (Higgs, custom)".
- [x] **README.** The third engine, live pauses on CPU (RTF numbers), the 3.85 GB download, 4–5 GB of RAM, custom voices, and the licence (personal non-commercial; a commercial licence is needed for products).
- [x] **Tests.** `test_config.py`, `test_langdetect.py`, `test_app.py`, `test_tts.py`, `test_mcp_server.py`, `test_export.py`, `test_reading.py` as needed.

### C. Settings UI, web reader, extension (`settings_ui.py`, `vibetts/web/app.js`, `extension/popup.js`, `tests/test_settings_ui.py`)

- [x] **Higgs tab** (`TABS` gains `"higgs"`, like Supertonic):
  - download, delete and status (3.85 GB, about 5 GB RAM, slower than real time on CPU);
  - the licence note and a temperature slider;
  - a **Voices** list (presets read-only, own clips) with **Add…** (WAV, transcript, name, language; it runs on a worker), **Delete** and **Play test**.
- [x] **Voice tab.**
  - The Language combo lists `ALL_LANGUAGES`.
  - Engine gets `Higgs` when the language is in `HIGGS_LANGUAGES`; Higgs-only languages offer Higgs/Off.
  - Add a red hint when Higgs is chosen but not installed, and a hint that undetectable languages need "Always X".
- [x] **Export tab.** The language list is Auto + `ALL_LANGUAGES`, and the voice list includes Higgs voices.
- [x] **Web reader and extension.** Make sure the `engine: "higgs"` voices and the extra languages show up. Check `engineOf` and any hard-coded engine lists.
- [x] **Tests.** `tests/test_settings_ui.py`: the Higgs tab, Add voice, the engine combo per language, and the export lists.

### D. DirectML spike (temp only, no tracked files)

- [x] In a separate venv, try `onnxruntime-directml` with the `cpu_int4` graphs on the Radeon 8060S. Measure RTF and first audio, and check the output is valid.
- [x] Add a "DirectML" section to `spike-results.md`. If it reaches RTF < 1, a follow-up adds a DML provider option (the app's onnxruntime would have to change to onnxruntime-directml).

## Verification

1. `python -m pytest -q tests`: all green.
2. `python -m vibetts.model_setup --higgs` downloads into `models/higgs-audio-v3`, and the sizes match.
3. CLI: `--say "<hr text>" --voice higgs:F1 --lang hr` plays. `--doc <hr text> --out x.mp3 --voice higgs:F1 --lang hr` gives an intelligible 44.1 kHz MP3 (ASR check) with the RTF logged.
4. App: set Croatian to Higgs F1 and use the hotkey on Croatian text. It reads with Higgs, with the expected pauses between chunks. English stays on Kokoro.
5. Settings > Higgs: add a custom WAV and transcript. It shows up in the Voice tab, Export, the tray and MCP `list_voices`. Read with it.
6. Turn on Bosnian with Higgs: Croatian samples are still detected as hr (check the log).
7. PyInstaller build (`scripts/build.ps1`): the exe loads the tokenizer and the presets and reads with Higgs. Afterwards restore HKCU Run (memory: dev-run autostart).

## Out of scope

- The GPU server path (SGLang or vLLM). DirectML in the app is a follow-up after D.
- The emotion and SFX tags in the UI (their syntax is stripped from read text).
- Normalising numbers for Croatian (follow-up).
- Japanese through Higgs (it is not in Higgs's list).
