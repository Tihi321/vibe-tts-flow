# VTF-02: Croatian speech with a CPU-only engine (OmniVoice ONNX)

Ticket folder at implementation time: `.claude/tickets/VTF-02-croatian-tts/` (copy this plan there as `plan.md`; implement via the `implement` skill).

Branch: `VTF-02_croatian-omnivoice-tts` (from `origin/master`).

## Context

VibeTTS reads selected text aloud with Kokoro-82M (`kokoro-onnx`, CPU, ~8x real time). Kokoro has no Croatian. The user asked whether a newer model (ComfyUI's TTS Audio Suite was only an example of what exists) can add Croatian, with two hard constraints: **no CUDA** on this machine, and **fast**.

### Research summary (Oct 2026)

| Model | Croatian? | CPU-viable? | License | Verdict |
|---|---|---|---|---|
| Kokoro (current) | no | yes, very fast | Apache-2.0 | keep for English etc. |
| **OmniVoice (k2-fsa)** | **yes, `hr`, ~2,795 h training data** (also sr/bs/sl) | **yes via community ONNX export** (`Emerald7664/OmniVoice-Onnx`): onnxruntime + numpy only, reported CPU RTF ≈ 0.48 (fp16 audio parts) / 0.59 (int4) at 32 steps | Apache-2.0 | **recommended** |
| Fish Audio S2 Pro | yes (low tier) | no (5B, GPU) | non-commercial | no |
| Higgs Audio v3 | maybe (102 langs) | no (4B) | non-commercial | no |
| Chatterbox ML, Qwen3-TTS, IndexTTS-2, VibeVoice, MOSS-TTS, Step Audio | no Croatian | — | — | no |
| Piper | no `hr` voice (has `sr`, `sl`) | yes, very fast | MIT | fallback only (Serbian voice reading Croatian) |

OmniVoice is the only Croatian-trained, permissively licensed model that runs without CUDA. It is ~2x faster than real time on CPU (vs Kokoro's ~8x), which the existing one-chunk-ahead streaming in `Speaker` hides after the first sentence. Halving the unmasking steps (16; the paper's benchmark setting) should roughly halve the cost.

Known gaps in the ONNX port's `inference.py` that we must handle ourselves: no classifier-free guidance, no explicit language token, fixed `--num_audio_tokens` (no duration estimate), uses `transformers` + `torchaudio` (we avoid both).

## Approach

### Phase 0: spike (go/no-go, scripts in `.claude/temp/vtf02/`)
- [x] 1. Download the CPU build of `Emerald7664/OmniVoice-Onnx` (backbone: `audio_embeddings_encoder`, `llm_decoder` int4, `audio_heads_decoder`, `tokenizer.json`; plus `audio_tokenizer/` Higgs encoders/decoder). Expect ~1 GB.
   - Result: done, repo commit `f312525229ea4c4d9101c2b3e317ac77d995ec59`, file list/sizes pinned in `spike-results.md`. KEY FINDING: community `llm_decoder.onnx` is causal (GQA) but OmniVoice is bidirectional, so output is noise; we must export our own bidirectional LLM (done with torch, int4 235 MB works).
- [x] 2. Read upstream `k2-fsa/OmniVoice` generation code to port what the ONNX script omits: prompt layout incl. language id (`hr`), duration estimation, and CFG (only if quality needs it, since it doubles backbone passes).
   - Result: done; ported prompt/lang token/duration estimator/schedule/scoring/CFG in `spike-scripts/spike.py`. CFG 2.0 is REQUIRED for intelligible few-step output (costs ~1.7x).
- [x] 3. Synthesize ~10 Croatian sentences (with č ć đ š ž, numbers, long compound sentence) in auto-voice and with a reference clip. Measure RTF at 32 / 16 / 8 steps and first-sentence latency.
   - Result: done; Croatian intelligible (Whisper-small word match 0.68-0.78, cloning 0.85). bidir fp32, CFG 2: 16 steps RTF 1.46, 8 steps 0.76, 6 steps 0.57, 5 steps 0.47; no-CFG 8 steps 0.48 but poor (0.48 match); 32 steps no CFG RTF 1.7. First sentence 1.5-1.9 s at 5-6 steps. Cloning ref inflates sequence (RTF 1.4-2.3). Numbers on a 16-core Ryzen AI Max+ 395 (fast CPU).
- [x] 4. **Go** if Croatian is clearly intelligible at RTF ≤ 0.5 with some step count. **No-go** → stop and report; fallback is a Piper `sr_RS` voice with espeak `hr` phonemes (fast, Serbian-accented).
   - Result: CONDITIONAL GO. RTF <= 0.5 reached only at 5 steps + CFG 2 (0.47) on a fast CPU; recommended default 6 steps (0.55-0.57). Requires own bidirectional LLM ONNX export + hosting, CFG, and exact upstream scheduling port. See `spike-results.md`; decision on whether to proceed is the user's.

### Decision gate (2026-10-15) — GO (user: "ok go with omni voice")

**Revised scope for Phases 1-2 (supersedes conflicting bullets below):**
- Model files are built locally by `scripts/build_omnivoice.py` (dev-only; run in its own torch venv, requirements in `scripts/omnivoice-build-requirements.txt`). It downloads the pinned community parts (embeddings fp16, heads, Higgs decoder, tokenizer @ `f312525229ea4c4d9101c2b3e317ac77d995ec59`), exports + int4-quantizes the bidirectional LLM from `k2-fsa/OmniVoice` @ `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` (recipe: `spike-scripts/torch_ref.py`, `quant.py`), writes everything to `models/omnivoice/` with a `manifest.json`, and creates the default voice preset. No in-app download, no hosting, no `model_setup.py` change.
- **Speaker consistency (new, found after the spike):** auto mode picks a new speaker per generation (median F0 of the 11 sample sentences ranges 112-247 Hz). Fix without the Higgs encoders: a *voice preset* = the generated codes (`.npy`, 8×T) + text of a short (~2-3 s) Croatian clip produced by the model itself, used as the reference prefix for every chunk. Presets live in `models/omnivoice/voices/<name>.{npy,txt,wav}`; `python -m vibetts.omnivoice make-voice <name> --seed N [--text ...]` creates one so the user can audition and pick. Measure the RTF cost of the prefix and keep the preset short enough that the default stays near RTF 0.6-0.7 on this machine.
- Voice ids: `omni_hr:<preset>` (default `omni_hr:default`). Encoder-based cloning from user WAVs stays deferred.
- Engine defaults: 6 steps (setting range 4-16), CFG 2.0, upstream duration estimator, schedule/scoring exactly as `spike-scripts/spike.py`; int4 LLM.
- Settings UI: instead of a download row, a status line ("Croatian model: found / missing — run scripts/build_omnivoice.py"), steps slider, Croatian voice combo, auto-language checkbox.

**Engine interface contract (shared by both implementers):** module `vibetts/omnivoice.py` exposes
`OMNI_PREFIX = "omni_"`, `def is_omni(voice) -> bool`, `def models_present(model_dir: Path) -> bool`, `def list_presets(model_dir: Path) -> list[str]` (voice ids, no model load), `class OmniVoiceEngine(model_dir: Path, steps: int = 6)` with `voices() -> list[str]` and `synth(text, voice, speed) -> np.ndarray` (float32 mono, 24 kHz). onnxruntime/tokenizers/soxr imported lazily inside the class.

### Decision gate history
- Orchestrator check: upstream `k2-fsa/OmniVoice` model card says the **weights are CC-BY-NC** (training data, e.g. Emilia); Apache-2.0 covers the code only. The research table above is wrong on that point.
- User chose: listen to samples first, then decide. Samples: `.claude/temp/vtf02/listen/{1_default_int4_6steps,2_fastest_5steps,3_best_16steps}.wav` (10 Croatian + 1 English sentence each).
- User chose model files **local only**: if go, a repo script rebuilds the bidirectional int4 LLM into `models/omnivoice`; no in-app download, no hosting. This replaces the `model_setup.py` / Settings download row items in Phase 2.
- If go, Phase 1 must port `spike-scripts/spike.py` (prompt, `hr` language token, upstream duration estimator, schedule/scoring, CFG 2.0), default 6 steps (range 4-16); cloning deferred.

### Phase 1: OmniVoice engine
- **New `vibetts/omnivoice.py`**: `OmniVoiceEngine` implementing the `Engine` protocol from `vibetts/tts.py:52`.
  - Three `onnxruntime.InferenceSession`s (CPU EP) for the backbone plus the Higgs decoder; encoders only loaded when a reference clip is used.
  - Tokenize with the lightweight `tokenizers` package reading `tokenizer.json` (no `transformers`).
  - Iterative unmask loop (numpy port of `iterative_unmask`, confidence weights `[8,8,6,6,4,4,2,2]`), `num_steps` from config.
  - Duration: frames = 25 fps × estimated seconds; seconds from characters/sec (default tuned in spike, or measured from the reference clip's text/duration) divided by `speed`.
  - Output float32 @ 24 kHz — matches `SAMPLE_RATE` in `vibetts/tts.py:23`, so `Speaker`, `SoundDeviceOutput`, `write_wav` are unchanged.
  - Voice cloning: reference `voices/<name>.wav` + `<name>.txt`; encode once with the Higgs encoders and cache prefix codes as `<name>.codes.npy`. Resample with `soxr` (small wheel) instead of torchaudio/scipy.
  - Lock around synth, like `KokoroEngine`.
- **Voice ids**: `omni_hr` (auto voice) and `omni_hr:<clip>` (cloned). Extend `voice_supported` / `voice_label` / `LANG_LABELS` in `vibetts/tts.py` so these show as e.g. "Auto voice (Croatian)" / "Ana (Croatian, cloned)".

### Phase 2: routing, config, setup, UI
- **`vibetts/app.py` `EngineHolder`** (lines 55-96): instead of one `KokoroEngine`, hold engines per family, loaded lazily; `get()` returns a small `RoutingEngine` whose `synth` dispatches on the voice prefix (`omni_` → OmniVoice, else Kokoro) and whose `voices()` merges both lists. Same change in the `--say` path (~lines 451-470). Preload only the family of the configured voice.
- **Auto language switch**: new `SpeechConfig.croatian_voice` (default `omni_hr`) and `auto_language: bool = True`. When the selection looks Croatian (contains č ć đ š ž, or a small stop-word check: "je", "i", "da", "se", "u", "na", "što", "nije"), the request uses `croatian_voice`; otherwise the normal voice. Decision made once per request in `app.py` where `SpeakRequest` is built (~line 267).
- **`vibetts/config.py` `SpeechConfig`**: add `omnivoice_dir = "models/omnivoice"`, `omnivoice_steps = 16`, `croatian_voice`, `auto_language`. Update `config.example.toml`.
- **`vibetts/model_setup.py`**: add an optional "Croatian voice model" download set (Hugging Face `resolve/main/...` URLs, pinned to a commit hash), reusing the existing `requests` download/progress code used for the Kokoro files.
- **`vibetts/settings_ui.py`**: second download row for the OmniVoice files; steps slider (8–32); Croatian voice combo; auto-language checkbox.
- **`vibetts/textprep.py`**: make the URL replacement word language-aware ("poveznica" for Croatian).
- **Deps/packaging**: add `tokenizers`, `soxr` to `requirements.txt` / `pyproject.toml`; `onnxruntime` already ships via `kokoro-onnx`. Update `packaging/VibeTTS.spec` hidden imports. Models are downloaded at runtime, not bundled.
- **README**: Croatian section, model size, speed expectations, how to add a cloned voice.

## Verification
- `pytest` for existing suites plus new tests in `tests/test_tts.py`: routing by voice prefix, voice labels, Croatian detection, duration estimate, unmask loop on fake sessions (tiny numpy stand-ins, no model files).
- `python -m vibetts --say "Dobar dan, ovo je test čitanja na hrvatskom jeziku."` with `voice = "omni_hr"`: audio plays, log shows synth time; confirm RTF in the `synth ... in ...s` debug line.
- Manual: select mixed English and Croatian paragraphs in a browser, press Alt+Win, and confirm auto-switching, stop (Esc) responsiveness, and no gap between sentences after the first.
- Build the exe with `scripts/build.ps1`, run it, download the Croatian model from Settings, and repeat the manual test.

## Implementation status (2026-10-15)
- [x] Phase 1 engine: `vibetts/omnivoice.py` (contract + `make-voice` / `say` CLI), `scripts/build_omnivoice.py` + `scripts/omnivoice-build-requirements.txt`; `models/omnivoice/` built on this machine (634 MB + `voices/default`, seed 0). 17 unit tests.
- [x] Speaker consistency: preset prefix on every chunk; median-F0 spread 24 Hz (was 112-247 Hz without a preset).
- [x] Phase 2 integration: RoutingEngine/EngineHolder per family, auto-language (`looks_croatian`), config fields + steps clamp 4-16, Settings Croatian group, tray "Croatian" group, `say()` routing, language-aware URL word, README, deps (`tokenizers`), spec hidden imports.
- [x] Orchestrator fix: Croatian text is merged into chunks of up to 260 characters after the first chunk (`HR_MERGE_*` in `textprep.py`), because the preset prefix makes short chunks slower than real time. `soxr` was dropped (cloning is deferred).
- [x] Verification: `pytest -q` passes 103 tests; `python -m vibetts --say <Croatian> --wav` takes 5.0 s to load and produces 11.16 s of audio in 7.66 s (1.5x real time).
- [ ] Not run: the live hotkey/tray flow with audible playback, the Settings UI by hand, and the PyInstaller build (`scripts/build.ps1`).
- Deferred: encoder-based cloning from user WAVs, and Croatian ASR quality beyond Whisper-small word match (~0.74).

## Phase 3: custom voices, created in the app (requested 2026-10-15)

The user asked: "add clone feature and add that we can create voice with description, so we can add custom voices on the fly".

**Key idea:** a custom voice is just another preset (`voices/<name>.npy` codes + `.txt` + `.wav`, plus a new `.json` meta). Reading speed therefore stays the same as `default`, and the encoders are only needed when a voice is created.

### Engine (`vibetts/omnivoice.py`), contract for the UI side
- `INSTRUCT_OPTIONS: dict[str, list[str]]`, ported from upstream `_resolve_instruct` (English items only): `gender` [male, female], `age` [child, teenager, young adult, middle-aged, elderly], `pitch` [very low pitch … very high pitch], `style` [whisper]. Accents are left out because they are English accents and the voices speak Croatian.
- `validate_instruct(s: str) -> str`: normalises the description, and raises `ValueError` (with upstream-style "did you mean" hints) for unknown items or two items from the same category.
- `sanitize_name(s) -> str`: allows `[a-z0-9_-]`, max 32 characters; `default` is reserved.
- `OmniVoiceEngine.design_clip(instruct: str, seed: int, text: str | None = None) -> VoiceClip`: generates a ~3 s clip with `<|instruct_start|>…<|instruct_end|>` and no reference, at 16 steps.
- `OmniVoiceEngine.clone_clip(wav_path: Path, ref_text: str) -> VoiceClip`:
  - Reads the WAV with stdlib `wave` + numpy (PCM 16/24/32 and float). Mono mixdown, silence trimmed. Under 2 s is rejected, over 8 s is cut at a low-energy point. When the clip is cut, the user gets a warning that the transcript must still match.
  - Resamples to 24 k / 16 k with `soxr`, which goes back into the dependencies.
  - Encodes with the **fp32** Higgs encoders: acoustic, semantic and quantizer. The encoders are loaded lazily and released after use.
- `VoiceClip`: dataclass with `codes` (8×T int16), `text`, `audio` (float32 24 kHz; for a clone this is the decoded codes, so the preview is what the model will imitate) and `meta` (dict: kind, instruct/seed or source file).
- `save_preset(model_dir, name, clip) -> str` (voice id), `delete_preset(model_dir, name)` (refuses `default`), `encoders_present(model_dir) -> bool`. `models_present` does NOT require the encoders.
- CLI: `design-voice <name> --instruct "female, young adult, low pitch" [--seed N] [--text ...]` and `clone-voice <name> --wav f.wav --text "..."`. The existing `make-voice` stays.
- Design is trained only on zh/en. Generate the design clip from Croatian text first; if that is unintelligible or unstable, try an English design sentence as the preset. Measure which one gives better Croatian reading afterwards (ASR word match over the 10 spike sentences plus median-F0 spread) and use the better one by default.

### Build script
- The fp32 Higgs encoders are downloaded by default from the pinned community commit; `--no-encoders` skips them. The manifest gets a separate `encoders` section. Run it on this machine.

### App / UI
- Settings › Voice › Croatian gets **Add voice…** and **Delete** buttons (Delete is disabled for `default`).
- The Add voice dialog has two tabs:
  - **From recording:** WAV picker, transcript field, a consent line ("Only clone voices you have permission to use"), Create → Preview → Save.
  - **From description:** dropdowns from `INSTRUCT_OPTIONS` (each with a "(any)" choice), seed spinbox, **Try another** (new seed), Preview, Save.
  - Both tabs have a name field.
- Generation runs on a worker thread with a busy indicator. Preview plays through the configured output device. Errors are shown inline.
- The dialog uses the app's shared OmniVoice engine through `EngineHolder` (never a second 1.6 GB copy), via new `SettingsHooks`: `design_clip`, `clone_clip`, `save_voice`, `delete_voice`, `play_clip`, `encoders_present`.
- After Save, the new voice appears at once in the Croatian combo and the tray menu, and Save can make it the Croatian voice.
- README: a "Custom Croatian voices" section.

### Verification
- `pytest`, including the new tests:
  - instruct validation;
  - name sanitising;
  - WAV reading/resampling;
  - clone pipeline with fake encoder sessions;
  - preset save/delete/meta;
  - settings dialog hooks (headless).
- CLI: clone a voice from a generated clip and create 2 design voices (female/male). For each, read the 10 spike sentences and record ASR word match, F0 spread and RTF; RTF should match `default` (~0.69) for presets of 4 s or less.
- Not automatable: a human listen and a real-microphone clone.

### Phase 3 status (2026-10-15)
- [x] Engine:
  - `INSTRUCT_OPTIONS`, `validate_instruct`, `sanitize_name`, `VoiceClip`, `design_clip`, `clone_clip`, `save_preset(overwrite=)`, `delete_preset`, `encoders_present`, `.json` meta, and the `design-voice` / `clone-voice` CLI.
  - The fp32 encoders (654 MB) are installed by the build script and are present on this machine.
- [x] UI:
  - The Croatian group has Add voice… and Delete buttons.
  - The `AddVoiceDialog` has the description and recording tabs.
  - Clips are shared through `EngineHolder.get_omni`, previews play via `play_clip`, and the tray and combo refresh after a save or delete.
  - Deleting the active voice falls back to `omni_hr:default`.
- [x] Orchestrator fixes:
  - Replacing a voice now passes `overwrite=True` to the save call; it used to fail with `FileExistsError`.
  - Recordings over 8 s are **rejected** instead of cut. The cut path produced a transcript mismatch and scored 0.46 word match at RTF 1.09.
- [x] Design text: Croatian design text scores better than English (word match ~0.76 vs ~0.62), so `DESIGN_LANG = "hr"`. Designed voices read at RTF ~0.69, the same as `default`.
- [x] Verification:
  - `pytest -q` passes 142 tests.
  - CLI `design-voice` with a female description took 4.8 s, `clone-voice` from a 4.6 s clip took 1.5 s, the 53 s clip was refused, and `say` read with the new voice.
  - The model folder holds only `default` afterwards.
- [ ] Not run:
  - opening the dialog by hand;
  - cloning from a real microphone recording;
  - listening to the voices (audition files are in `.claude/temp/vtf02/listen/custom/`);
  - the PyInstaller build.
