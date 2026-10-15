# Changelog: VTF-02 Croatian speech (OmniVoice, CPU-only)

## 2026-10-15 (later): custom Croatian voices, cloned or from a description

**Outcome:** you can add Croatian voices from Settings or the CLI. A voice is either designed from a description (gender, age, pitch, whisper) or cloned from a 2-8 s WAV with its transcript. Every custom voice is saved as a preset, so it reads exactly as fast as `default`. All changes are uncommitted on branch `VTF-02_croatian-omnivoice-tts`.

### Files
- `vibetts/omnivoice.py`:
  - **Validation:** `INSTRUCT_OPTIONS` and `validate_instruct()` follow upstream's English vocabulary, with "did you mean" hints and a same-category conflict check. `sanitize_name()` produces `[a-z0-9_-]`, at most 32 characters, strips diacritics (đ becomes dj), and reserves `default`.
  - **`VoiceClip`** carries `codes`, `text`, `audio`, `meta` and `warning`.
  - **Design:** `OmniVoiceEngine.design_clip()` generates from Croatian text at 16 steps and takes about 4.8 s.
  - **Cloning:** `OmniVoiceEngine.clone_clip()` takes 1.2-1.6 s.
    - It reads the WAV with the standard-library `wave` module and numpy, mixes to mono and trims silence.
    - It rejects clips under 2 s or over 8 s of speech, resamples with `soxr` and levels quiet audio.
    - It encodes with the fp32 Higgs encoders, which are loaded for that call only.
  - **Presets:** `save_preset(overwrite=)` writes `.npy` last so a half-written preset is never listed, and stores a `.json` meta. `delete_preset()` refuses `default`. `encoders_present()` checks `models/omnivoice/encoders/`.
  - **CLI:** new `design-voice` and `clone-voice` commands.
- `scripts/build_omnivoice.py`: installs the three pinned fp32 encoders by default (654 MB, size-checked) and records them in an `encoders` section of the manifest. `--no-encoders` skips them.
- `vibetts/app.py`:
  - `EngineHolder.get_omni()` returns the shared engine, so voice creation never loads a second copy.
  - New settings hooks: `design_clip`, `clone_clip`, `save_voice(overwrite=)`, `delete_voice`, `play_clip` and `encoders_present`.
  - `play_clip` plays a `ClipEngine` through the existing `Speaker`, so it uses the configured device and volume and Stop works.
  - Deleting the active voice falls back to `omni_hr:default` and saves the config. The tray and settings refresh after a change.
- `vibetts/settings_ui.py`: Croatian **Add voice…** and **Delete** buttons, and the `AddVoiceDialog`.
  - **Description tab:** dropdowns, seed, Try another, Generate, Play, Stop, Save.
  - **Recording tab:** WAV picker, transcript, consent line, Create, Play, Stop, Save. Create is disabled when the encoders are missing.
  - Generation runs on a worker thread and shows a busy state. Errors appear inline in the dialog.
  - **Save:** the name is sanitised, replacing an existing voice asks first, and "Use as Croatian voice" is on by default.
- `requirements.txt`, `pyproject.toml`, `packaging/VibeTTS.spec`: `soxr` is back, with its hidden imports.
- `README.md`: a "Custom Croatian voices" section.
- Tests: `tests/test_omnivoice.py` (20 new), `tests/test_app.py` (+6), `tests/test_settings_ui.py` (+13). The UI tests share one Tk root and skip if Tk can't start.

### Deviations
- Recordings over 8 s are rejected rather than cut. A cut clip no longer matches its transcript, and the test clone made that way scored 0.46 word match at RTF 1.09.
- Additions beyond the contract:
  - `design_clip(lang=)` and the `VoiceClip.warning` field.
  - A Stop button in the dialog.
  - The dialog's delete confirmation uses a native `askyesno`, as the rest of the settings window already does.
- Review fix: the dialog's "Replace?" confirmation now passes `overwrite=True`. Before, replacing an existing voice failed with `FileExistsError`.

### Verification
- `.venv\Scripts\python -m pytest -q`: 142 passed.
- Design experiment (6-step reading, 10 spike sentences):
  - Croatian design text averaged ~0.76 word match, against ~0.62 for English design text.
  - Female voices came out at 247-253 Hz and male voices at 144-147 Hz.
  - RTF ~0.69, the same as `default`.
- Clone experiment on clips of 4.4 s or less: 0.64-0.68 word match at RTF 0.70-0.79. These were clones of generated audio, not real recordings.
- CLI end to end, on a copy of the model folder: `design-voice` took 4.8 s and `clone-voice` took 1.5 s. A 53 s clip was refused with a clear message. `say` read with the new voice.
- **Not run:**
  - the dialog by hand;
  - cloning from a real microphone recording;
  - listening to the results (audition files are in `.claude/temp/vtf02/listen/custom/`);
  - the PyInstaller build.

### Limitations
- Voice description works only through the fixed descriptor vocabulary, and it was trained on zh/en only. Results vary by seed, so use Try another.
- Creating a voice holds the engine lock for about 1.5-5 s, which pauses any reading in progress.

**Commit status:** uncommitted.

## 2026-10-15: Croatian read-aloud via a local OmniVoice ONNX engine

**Outcome:** VibeTTS reads Croatian on CPU (no CUDA) with OmniVoice. Kokoro stays the engine for English and the other languages, and Croatian text is detected and routed automatically. Everything is uncommitted on branch `VTF-02_croatian-omnivoice-tts`, which was cut from `origin/master`.

### Files
- `vibetts/omnivoice.py` (new): the `OmniVoiceEngine` is pure numpy + onnxruntime + `tokenizers` on the CPU execution provider.
  - It ports upstream's prompt with the `hr` language token, upstream's duration estimator, the masked-unmask schedule with Gumbel position noise, and CFG 2.0. The default is 6 steps.
  - A voice preset (codes of a short generated clip) is prefixed to every chunk so the speaker stays the same. Seeded synthesis is deterministic.
  - It has a CLI: `make-voice` and `say`.
- `scripts/build_omnivoice.py`, `scripts/omnivoice-build-requirements.txt` (new): a developer-only build run in a torch venv.
  - It downloads the pinned community ONNX parts and the upstream weights.
  - It exports the **bidirectional** LLM, checks it against torch and quantizes it to int4.
  - It writes `models/omnivoice/` with `manifest.json` and creates the `default` preset. Re-running it skips finished steps.
- `vibetts/app.py`:
  - `RoutingEngine` sends `omni_*` voices to OmniVoice and everything else to Kokoro. `EngineHolder` loads each family lazily and keeps a separate cache key for each.
  - The voice is chosen once per request (`choose_voice` / `_request_voice`). If the text looks Croatian but the model is missing, the normal voice is used and a warning is logged once.
  - The model-missing check covers the family a request actually needs. `say()` uses the same routing.
- `vibetts/textprep.py`:
  - `looks_croatian()` is a heuristic based on diacritics and function words.
  - The word spoken for a URL depends on the language ("poveznica" in Croatian).
  - Croatian text is merged into longer chunks after the first one (`HR_MERGE_*`).
- `vibetts/tts.py`: the voice helpers understand `omni_hr:<preset>` and label it "Default (Croatian)".
- `vibetts/config.py`, `config.example.toml`: new `omnivoice_dir`, `omnivoice_steps` (clamped to 4-16), `croatian_voice` and `auto_language`.
- `vibetts/settings_ui.py`: the Voice tab gains a Croatian group with a model status line, auto-language checkbox, Croatian voice combo and steps slider.
- `vibetts/tray.py`: Croatian presets appear in a "Croatian" group.
- `requirements.txt`, `pyproject.toml`, `packaging/VibeTTS.spec`: `tokenizers` is added, and the spec gets hidden imports for `vibetts.omnivoice` and `tokenizers`.
- `README.md`: a Croatian section covering the licence, the build, presets, speed and auto-language.
- Tests: `tests/test_omnivoice.py` (new) plus additions to `test_app.py`, `test_tts.py`, `test_config.py`, `test_settings_ui.py` and `test_textprep.py`.

### Deviations from the plan
- The community `llm_decoder.onnx` is causal and produces noise, so the build exports its own bidirectional LLM (spike finding).
- The OmniVoice **weights are CC-BY-NC**, not Apache-2.0 as the original research table said. The user chose to build them locally with no in-app download or hosting, so `model_setup.py` and the Settings download row are unchanged.
- Voice presets were added because auto mode changes the speaker on every sentence. Encoder-based cloning is deferred and `soxr` was dropped.
- CFG is required. The default is 6 steps rather than 16.
- Croatian chunk merging was added after review, because the preset prefix makes short chunks slower than real time.

### Verification
- `.venv\Scripts\python -m pytest -q`: 103 passed.
- `python -m vibetts --say "<7 Croatian sentences>" --wav ...`: the model loaded in 5.03 s, and 11.16 s of audio was synthesized in 7.66 s (1.5x real time) on a Ryzen AI Max+ 395.
- Engine measurements with the default preset at 6 steps over the 10 spike sentences: aggregate RTF 0.69, median-F0 spread 24 Hz, Whisper-small word match 0.74.
- **Not run:** the live hotkey/tray flow with audible playback, the Settings UI by hand, and the PyInstaller exe build.

### Follow-ups and limitations
- Quality is understandable but rough; words with đ/ć are the weak spots. Raising the steps improves quality and costs speed.
- On slower CPUs, playback may pause between chunks.
- Engine RAM is about 1.6 GB while the Croatian engine is loaded.
- Cloning from user recordings is deferred (it needs the fp32 Higgs encoders, about 650 MB more).

**Commit status:** uncommitted.
