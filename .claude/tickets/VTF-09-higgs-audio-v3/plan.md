# VTF-09: Higgs Audio v3 as a third engine (next to Kokoro and Supertonic)

Ticket folder: `.claude/tickets/VTF-09-higgs-audio-v3/`. Copy this plan there as `plan.md` on the first implementation step. Branch: `VTF-09_higgs-audio-v3`, cut from `master` (61b5636).

## Context

VibeTTS has two engines: Kokoro (English plus 5 more languages) and Supertonic 3 (31 languages including Croatian). Since VTF-08 there is one voice per language (`speech.language_voices`), and the app detects the language of each read with py3langid. The user wants **Higgs Audio v3 TTS** (Boson AI, June 2026) added the same way Supertonic was. Its Croatian is rated in the "WER < 5, production quality" tier, and it covers about 94 languages.

Facts that shape the design (from research, 2026-10-22):
- **Size.** It is a 4B autoregressive decoder (Qwen3-4B body) that emits 8 codebooks at 25 frames/s, decoded by the Higgs audio tokenizer to **24 kHz** audio. The official weights are 9.3 GB of BF16 safetensors and expect a GPU (SGLang-Omni or vLLM-Omni).
- **ONNX path.** A community export, `onnx-community/higgs-audio-v3-tts-4b`, has a `cpu_int4/` build of about 3.9 GB. It has six graphs (`text_embed`, `llm_decoder` with a KV cache, `audio_heads`, `audio_embed`, `audio_tokenizer`, `audio_encoder`) and a reference `inference.py`, which runs without torch. Its `STATUS.md` says intelligibility of the full loop is **not verified**. Greedy decoding buzzes, so sampling is required (temperature 0.8, top-k 50).
- **No preset voices.** Without a reference, each read gets a random voice. Voice cloning needs a reference WAV plus its transcript.
- **Speed control.** There is no continuous speed. Inline prosody tags exist instead: `<|prosody:speed_slow|>` (≈0.85×), `speed_fast` (≈1.2×), and others.
- **Licence.** The weights are under the Boson Research and Non-Commercial licence. A Creator Use Grant allows monetised content when Boson AI's Higgs Audio is credited. Personal use is fine. The README must say this.
- **This machine.** Ryzen AI Max+ 395 (16 cores), 32 GB RAM, Radeon 8060S, no CUDA. Being slower than real time on CPU is a real risk. OmniVoice was dropped earlier at 1.4× real time.

User decisions (2026-10-22):
1. Voices: bundled **presets** plus **your own clips** (WAV and transcript) added in Settings.
2. If the spike shows slower than real time, Higgs is for **export and MCP only**. Live hotkey reads stay on Kokoro and Supertonic.
3. **Expand languages**: languages only Higgs covers (bs, sr, mk, sq, ca, …) join the picker.

## Step 0: spike (go/no-go, sets the live-vs-export gate)

- [x] **Step 0 done (2026-10-22): gate = Export-only.** Best RTF 1.37 to 1.55 zero-shot / 1.65 to 1.78 with a clone (8 threads; default 1.9 to 2.3), first short chunk 2.6 s zero-shot / 6 to 9 s with a clone, load 5 s, RAM 4.8 GB (up to 7.5 GB with the encoder); Croatian is intelligible only with a same-language reference clip (ASR 0.92 to 1.00). Details: [spike-results.md](spike-results.md). Listening test (user): pending.

Work in `.claude/temp/vtf09/` (venv with onnxruntime, numpy, tokenizers, soundfile). Record the results in `.claude/tickets/VTF-09-higgs-audio-v3/spike-results.md`, following the format of `VTF-06-supertonic-3/spike-results.md`.

- Download `cpu_int4/` and pin the repo commit sha. Run the repo's `inference.py` on English and Croatian text (a short sentence, a paragraph, numbers and dates), both zero-shot and with a reference clip.
- Measure load time and RAM, time to the first audio of a short sentence, and RTF across ORT thread counts. Optional: also try `onnxruntime-directml` on the 8060S, in the throwaway venv only.
- Check intelligibility by ASR: reuse `.claude/tickets/VTF-02-…/spike-scripts/asr.py`. The user does a listening test (`listen.html`, as in VTF-06).
- Check that cloning keeps one voice across chunks: the same reference and seed for every chunk. Also check cross-lingual cloning: an English reference reading Croatian text.
- Check that the speed tags work.
- Make the presets. Generate about 10 s reference clips with Supertonic (for example F1, F2, M1, M2), one neutral sentence each, and listen to the Higgs clones. Keep the ones that sound good. Any user-supplied clip can replace them.
- **Gates:**
  - **Stop** if the Croatian output is unintelligible or the int4 kernels give NaN or silence.
  - **Live** if RTF ≤ 0.8 and the first short chunk takes ≤ 3 s. Higgs is then a full per-language engine (live, export and MCP).
  - **Export-only** otherwise.

## 1. Engine: `vibetts/higgs.py` (new)

Model it on `vibetts/supertonic.py`:
- Credit the module: a port of the `onnx-community` `inference.py` (state its licence in the docstring), plus a Higgs licence note.
- `REPO`, `REVISION` (pinned sha), and `MODEL_FILES` as `(rel, size)` for the `cpu_int4` subset: the 6 graphs plus `.onnx.data`, `tokenizer.json`, `tokenizer_config.json`, `manifest.json`, `genai_config.json` and `LICENSE`. Also `models_present(dir)`.
- `HiggsEngine(model_dir, voices_dir)`:
  - One `threading.Lock`, CPU sessions created lazily as in Supertonic, and the Qwen tokenizer via `tokenizers.Tokenizer.from_file`.
  - `audio_encoder` (654 MB) loads only when a reference clip has no cached codes.
  - `voices()` → `["higgs:<name>", …]`: the presets plus the user's clips.
  - `synth(text, voice, speed, lang)`:
    1. Sanitise the text by removing `<|` and `|>`, so the read text cannot inject control tags.
    2. Prepend the speed tag closest to `speed` (none at 1.0).
    3. Build the clone prompt (`<|tts|><|ref_text|>…<|ref_audio|>[ref embeds]<|text|>…<|audio|>`).
    4. Run the AR loop (delay pattern, BOC/EOC, the repeat guard, a frame cap scaled to the text length) with a fixed seed per voice, so chunks sound alike.
    5. Decode with `audio_tokenizer` and return float32 44.1 kHz through the existing `tts.resample()` (`vibetts/tts.py:91`).
  - Raise a clear error on NaN or Inf logits (the int4 guard from the reference code).
- Reference clips: `<name>.wav` + `<name>.txt` + a cached `<name>.codes.npy` (the encoder output). Presets ship pre-encoded in `vibetts/voices/higgs/`. The user's clips live in `models/higgs-voices/`, outside the model dir, so deleting the model keeps them.

## 2. Config and languages (`vibetts/config.py`, `config.example.toml`)

- `HIGGS_PREFIX = "higgs:"`, plus `is_higgs_voice(voice)`. Because the user's clip names are dynamic, the check is on the prefix plus a safe-name regex (`[A-Za-z0-9_-]{1,32}`). Whether the clip exists is checked at route time, not in config validation.
- `HIGGS_LANGUAGE_NAMES`: the Higgs metadata codes (94; no `ja`). Add `LANGUAGE_NAMES = SUPERTONIC ∪ HIGGS` and `ALL_LANGUAGES` (English first, then the Supertonic order, then the Higgs-only codes by name).
- Replace `SUPERTONIC_LANGUAGES` with `ALL_LANGUAGES` where it means "all codes": `enabled_languages`, `_validate_speech`, and `DEFAULT_LANGUAGE_VOICES`. Keep `SUPERTONIC_LANGUAGES` where it means "what Supertonic reads".
- `voice_fits_language`: a Higgs voice fits `HIGGS_LANGUAGES`.
- Defaults: the existing 31 are unchanged. **Higgs-only languages default to Off (`""`)**, so adding them does not make Croatian text detect as Bosnian or Serbian. The user turns them on deliberately. Old configs then load unchanged, with the missing codes filled with `""`.
- New `[speech]` keys: `higgs_dir = "models/higgs-audio-v3"`, `higgs_voices_dir = "models/higgs-voices"`, `higgs_temperature = 0.8` (clamped 0.3..1.2).

## 3. Detection (`vibetts/langdetect.py`)

- The candidates are the enabled codes **that py3langid knows** (it has 97 classes, which include bs, sr, mk, sq, ca, …). Add a `DETECTABLE` set computed from the model's classes.
- Enabled but undetectable languages (for example `ceb`, `kab`) can only be used through "Always <language>" or an explicit `lang`. The Settings hint says so.

## 4. Routing and engines (`vibetts/app.py:63-400`)

- `RoutingEngine` (287) gets a `higgs` slot. `EngineHolder` gets `_ensure_higgs` and `unload_higgs`, mirroring `_ensure_supertonic` (359) and `unload_supertonic` (376). Preload Higgs only on the live gate, and only when a read-able language uses it.
- `voice_ready`, `missing_message` and `setup_tab` handle `higgs:` voices (the Higgs tab).
- `route_for` (149): if the voice is Higgs and the model or clip is missing, fall back to that language's Supertonic or Kokoro voice. If neither exists (a Higgs-only language), the read fails with "Download Higgs in Settings".
- **Export-only gate.** `language_voices` holds live voices and never holds `higgs:` values. Higgs voices appear only in the Export tab voice list, in MCP `voice`, and in `synthesize_to_file` / batch. Higgs-only languages appear in the Export language list and the MCP `lang` list, but not in the live Voices picker or the tray. `route_request` (165) rejects `higgs:` voices for `speak` with a clear message.
- **Live gate.** No such restriction: Higgs is a third engine choice per language everywhere.
- The ~3–4 GB in RAM should not live for ever. Unload Higgs after an export finishes on the export-only gate.

## 5. Model download (`vibetts/model_setup.py`)

- Add `higgs_files()`, `higgs_missing()`, `download_higgs()` and `delete_higgs()`, mirroring the Supertonic functions (130-182). They use the same `_stream_to` (resume, progress, cancel) against `huggingface.co/onnx-community/higgs-audio-v3-tts-4b/resolve/<sha>/cpu_int4/…`.
- CLI: `python -m vibetts.model_setup --higgs`.
- Check free disk space (about 4 GB) before starting.

## 6. UI

- **Settings > Higgs tab** (new, like the Supertonic tab): download, delete and status; the size and RAM note; the licence and credit note; a temperature slider. It also has a **Voices** list: presets (read-only) and the user's clips, with **Add…** (WAV file, transcript box, name; the clip is copied and encoded in the background, 3–30 s mono or stereo, resampled to 24 kHz), **Delete** and **Play test**.
- **Settings > Voice** (`settings_ui.py`): on the live gate, the Engine combo gets `Higgs`, shown only when the language is in `HIGGS_LANGUAGES`. A Higgs-only language offers only Higgs or Off. The language combo lists `ALL_LANGUAGES` (live gate) or the existing 31 (export-only).
- **Export tab:** the voice list adds `higgs:*`, and the language list adds the Higgs-only languages.
- **Tray:** on the live gate, the Voices submenus include Higgs voices. On export-only, there is no change.

## 7. MCP, reader, web, extension

- `AgentBridge.list_voices` (`app.py:1213`): adds Higgs voices with `engine: "higgs"`, `languages: HIGGS_LANGUAGES` and `live: bool`, plus `higgs_installed`. Document this in `.claude/tickets/VTF-09-…/mcp-changes.md`.
- `mcp_server.py` descriptions: voice ids `higgs:<name>`, the language list, and 24 kHz audio resampled to 44.1 kHz.
- The web reader and extension already filter languages by per-voice `languages` (VTF-08). They also have to hide voices with `live: false` from the Speak voice select.

## 8. Packaging and docs

- Add `tokenizers` to `requirements.txt` and `pyproject.toml`.
- `packaging/VibeTTS.spec`: hidden import `vibetts.higgs`, `collect_data_files("tokenizers")` if needed, and the `vibetts/voices/higgs/*` presets as data. No model files are bundled.
- README: the third engine, the gate result, the download size and RAM, custom voices, the licence and the Creator Use Grant credit line.

## 9. Tests

- `tests/test_higgs.py` (new):
  - Text sanitising and speed tags.
  - Prompt token layout.
  - Delay pattern and its reversal, and stopping on EOC.
  - The repeat guard.
  - A synth with tiny fake ONNX sessions, in the same style as `tests/test_supertonic.py`.
  - The reference-clip cache.
  - Missing files.
- `test_config.py`: Higgs voice ids, Higgs-only languages default Off, an old config loads unchanged, and temperature is clamped.
- `test_langdetect.py`: undetectable codes are excluded from the candidates, and hr is still detected as hr with bs and sr enabled (or document the confusion).
- `test_app.py`: routing with Higgs on each gate, the fallbacks, `speak` rejecting Higgs on export-only, EngineHolder lazy loading and unloading, and the `list_voices` shape.
- `test_model_setup.py`, `test_settings_ui.py` (Higgs tab, Add voice, the engine combo), `test_mcp_server.py`, `test_export.py`.

## Verification

1. `python -m pytest -q tests`: all green.
2. Run `python -m vibetts.model_setup --higgs`, then `--doc <hr text> --out x.mp3 --voice higgs:F1 --lang hr`. This gives a 44.1 kHz MP3 that is intelligible Croatian with the same voice across chunks. Log the RTF.
3. In Settings > Higgs, add a custom WAV and transcript. It appears in the Export voices and in MCP `list_voices`. Export with it.
4. Export-only gate: MCP `speak` with `higgs:F1` gives a clear error, and `synthesize_to_file` works. Live gate: the hotkey on Croatian text with `hr = higgs:F1` reads with Higgs.
5. Turn on Bosnian: Croatian samples are still detected as hr (checked against the log).
6. PyInstaller build (`scripts/build.ps1`): the exe loads the tokenizer and the presets. Afterwards, restore HKCU Run (the dev-run autostart note in memory).

## Out of scope

- The GPU or SGLang server path, and DirectML in the app (only measured in the spike).
- The emotion and SFX control tags in the UI. Their syntax is stripped from read text.
- Japanese through Higgs (it is not in Higgs's list).
