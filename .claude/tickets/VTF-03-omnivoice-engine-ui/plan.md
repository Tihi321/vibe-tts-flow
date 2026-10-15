# VTF-03: OmniVoice as a first-class engine (in-app install, English + Croatian, Voices tab)

Ticket folder: `.claude/tickets/VTF-03-omnivoice-engine-ui/`. Branch `VTF-03_omnivoice-engine-ui`, cut from `VTF-02_croatian-omnivoice-tts` (commit `1fc3115`, not merged yet, and this work builds on it).

## Context

VTF-02 added OmniVoice for Croatian, but testing the built exe exposed three gaps:
1. **No in-app install.** On another copy of the app, Settings shows "Croatian model: missing — run scripts/build_omnivoice.py", and that script needs a PyTorch venv. Users need an **Install** button, like Kokoro's Download.
2. **OmniVoice is Croatian-only.** The user wants OmniVoice as a normal engine that can read English too, with Kokoro as just another option.
3. **Voice creation is hidden.** The Add voice dialog sits behind a button that stays disabled until the model is present. The user wants a dedicated tab where they can create a voice from a description and clone from a sound file.

User decisions (2026-10-15):
- **Install from official sources and convert in the app**, with no hosting. That means downloading the k2-fsa weights and community ONNX parts, then converting to int4 with numpy.
- **Kokoro stays the default engine** on a fresh install; OmniVoice is optional.

## Approach

### 1. In-app OmniVoice install (no torch, no hosting)
- **Dev-time artifact, generated once by `scripts/build_omnivoice.py --export-graph` and committed:**
  - `vibetts/data/omnivoice/llm_bidir_int4.onnx`: the int4 bidirectional LLM **graph only**, about 1.2 MB, with no weights. Its external-data references point at `llm_bidir_int4.onnx.data`.
  - `vibetts/data/omnivoice/llm_weight_map.json`: for each initializer, its safetensors source tensor, transpose flag, whether it is quantized (MatMulNBits B/scales) or plain fp32, plus offset and length in the `.data` file.
  - The map is derived by matching values between the export and the safetensors.
  - Both ship in the exe: add a `datas` entry in `packaging/VibeTTS.spec`.
- **New `vibetts/omnivoice_setup.py`:**
  - **Install plan:**
    - community parts (embeddings, heads, Higgs decoder, tokenizer) at pinned commit `f312525…`, about 0.4 GB;
    - upstream `config.json` and `model.safetensors` from `k2-fsa/OmniVoice` at `c5fdb5cc…` (2,450,344,112 bytes);
    - optionally the fp32 encoders for cloning (654 MB).
  - All URLs are `https://huggingface.co/<repo>/resolve/<commit>/<path>` with size checks.
  - **Download:** reuse `vibetts/model_setup.py`: generalise `download()` / `_stream_to()` to take a full URL (Kokoro keeps `file_url()`), including the existing progress, cancel and `.part` handling.
  - **Convert:**
    - A numpy safetensors reader: parse the JSON header, then memory-map the file. It handles F32, F16 and BF16.
    - For each map entry, write the bytes at the map's offset. Quantized tensors are int4 block-128 symmetric in the exact MatMulNBits layout (B uint8 `[N, k_blocks, 64]`, scales `[N*k_blocks]`).
    - Progress is reported per tensor.
    - The install then writes `manifest.json`, deletes the 2.45 GB safetensors unless "keep download cache" is set, and creates the `default` preset with seed 0.
  - **Acceptance:** the generated `.data` matches today's torch-path `models/omnivoice/llm_bidir_int4.onnx.data` byte for byte. If ORT's rounding can't be matched exactly, the fallback is LLM hidden-state max diff ≤ 1e-3 against it, with ASR word match and RTF unchanged.
  - `uninstall(model_dir, keep_voices=True)`.
- `scripts/build_omnivoice.py` becomes a thin wrapper around `omnivoice_setup` for normal installs; `--export-graph` stays the torch-only developer path.

### 2. OmniVoice as an engine for any language
- **Language-free voice ids:** `omni:<name>`. Legacy `omni_hr:<name>` / `omni_hr` are still accepted by `parse_voice`, and config load migrates them.
  - `list_presets` returns `omni:<name>`. Preset `.json` gains `lang`, the language of its sample.
- **Per-request language:** `SpeakRequest` gets `lang`. The `Engine.synth` protocol becomes `synth(text, voice, speed, lang=None)`; Kokoro ignores it (`vibetts/tts.py`). The OmniVoice engine uses `lang` for the `<|lang_start|>` token, defaulting to `en`.
- **Choosing the language:** `app.py` decides it once per request: `hr` if `textprep.looks_croatian(text)`, else `en`. The same value also picks the URL word.
- **Config (`vibetts/config.py`):**
  - new `engine = "kokoro" | "omnivoice"`;
  - `voice` stays the Kokoro voice;
  - new `omni_voice = "omni:default"`;
  - `croatian_voice` becomes `"" (= same as the OmniVoice voice) | omni:<name>`;
  - `auto_language` is kept.
  - **Migration:** if the old `voice` is an omni id, set `engine = "omnivoice"` and `omni_voice` to that voice, and reset `voice` to `af_heart`.
- **Routing (`app.py` `choose_voice`):**

  | engine | Croatian text (auto on, OmniVoice installed) | other text |
  |---|---|---|
  | kokoro | `croatian_voice` or `omni_voice`, lang hr | Kokoro `voice` |
  | omnivoice | `croatian_voice` or `omni_voice`, lang hr | `omni_voice`, lang en |

  If OmniVoice is not installed, the request falls back to Kokoro (already implemented, logged once).
- **Chunking:** Croatian-only merging becomes engine-based. `textprep.prepare(..., long_chunks=is_omni)` uses the longer chunks after the first for any OmniVoice request, because the prefix cost is language-independent.
- **Design voices:** the design form gains a "Sample language: Croatian / English" choice, so a voice can be made for English. Default is Croatian, matching the measured result.

### 3. Settings restructure (`vibetts/settings_ui.py`)
- **Tabs:** General | Voice | Kokoro | OmniVoice | About (update `TABS` / `TAB_TITLES`).
- **Voice tab:**
  - "Engine: (•) Kokoro ( ) OmniVoice" radio. OmniVoice is disabled, with "Install it in the OmniVoice tab", until installed.
  - The Voice combo shows the selected engine's voices: Kokoro voices or OmniVoice presets.
  - Speed, pause, volume and device stay.
  - "Croatian text" group: auto checkbox plus a Croatian voice combo ("Same as OmniVoice voice" + presets).
  - "Try it" stays.
- **Kokoro tab:** the existing setup panel and "Model files" section move here unchanged.
- **OmniVoice tab** (always visible; parts are disabled with a hint until installed):
  - **Install:**
    - status line (installed / not installed / cloning available);
    - "Include voice cloning (+654 MB)" checkbox;
    - Install / Cancel / Uninstall buttons and a progress bar with a phase label ("Downloading model.safetensors 1.2 / 2.4 GB", "Converting 120 / 197");
    - a note that the download is about 2.9 GB once, uses ~1.3 GB of disk, and the weights are CC-BY-NC for non-commercial use.
  - **Quality steps** slider (4-16).
  - **Voices:**
    - a Treeview of presets: name, kind (default / description / recording), description or source, language;
    - buttons Play sample, Use as OmniVoice voice, Use for Croatian, Delete (Delete is disabled for `default`).
  - **Create voice:** an inline ttk sub-notebook with "From description" and "From recording". This is the existing `AddVoiceDialog` logic refactored into a reusable `VoiceCreator(ttk.Frame)`. The modal dialog and the Add voice… button are removed. Saving refreshes the Voices list, the combos and the tray.
- `SettingsHooks` gains `omni_install(include_encoders, progress, cancel)`, `omni_uninstall()` and `omni_status()`. The `omni_present` / `encoders_present` hooks are reused.
- **Tray (`vibetts/tray.py`):** an "Engine" radio submenu (Kokoro / OmniVoice) and voices grouped as Kokoro languages plus "OmniVoice". Picking a voice switches the engine.
- **README:** install OmniVoice from Settings, English and Croatian behaviour, and the Voices tab. The dev-only build script moves to a developer section.

### Reused pieces
- `model_setup._stream_to` / `download` (progress, cancel, `.part`).
- `EngineHolder.get_omni` / `RoutingEngine` (`app.py`).
- `omnivoice.design_clip` / `clone_clip` / `save_preset` / `delete_preset` / `list_presets` / `encoders_present`.
- `AddVoiceDialog` internals, which become the inline `VoiceCreator`.
- The Kokoro download panel pattern in `settings_ui.py` (worker thread plus `call_soon`).

## Verification
- `pytest`: everything existing passes, plus new tests for:
  - safetensors reader (F32/BF16);
  - int4 block quantization against ORT on a small matrix (`MatMulNBitsQuantizer` in a test-only check, or known vectors);
  - converter on a tiny fake graph and map;
  - install plan and URLs;
  - config migration of old `omni_hr` voices;
  - `choose_voice` routing table;
  - `lang` passed to the engine;
  - tab construction and engine-radio behaviour (headless Tk).
- **Real install:** run the in-app installer into a fresh folder (`.claude/temp/vtf03/install/`). Compare `.data` with the current torch-built file (byte-equal or hidden-state diff ≤ 1e-3). Time the download and convert, then check `python -m vibetts --say` works for Croatian.
- **English quality:** ASR word match on 10 English sentences with `default` (Croatian sample) and with an English-designed preset. Record both and set the design default accordingly.
- **Build:** `scripts/build.ps1`, then on a **fresh copy of `dist\VibeTTS` without models/omnivoice**:
  1. install OmniVoice from the Settings › OmniVoice tab;
  2. switch the engine to OmniVoice and read English text;
  3. read Croatian text with Kokoro selected (auto switch);
  4. design a voice and clone one from a WAV in the OmniVoice tab, and confirm they appear in the Voice combo and the tray.
- **Not automatable:** listening, and a real-microphone clone.

## Execution
- Branch: `VTF-03_omnivoice-engine-ui` (from `VTF-02_croatian-omnivoice-tts` @ `1fc3115`).
- Installer contract (shared by both implementers), in `vibetts/omnivoice_setup.py`:
  - `DOWNLOAD_BYTES(include_encoders: bool) -> int`
  - `status(model_dir: Path) -> dict` with keys `installed: bool`, `encoders: bool`, `bytes_on_disk: int`
  - `install(model_dir: Path, include_encoders: bool = True, progress: Callable[[str, int, int | None], None] | None = None, cancel: threading.Event | None = None, keep_cache: bool = False) -> None`
    - `progress(phase_text, done, total)`
    - raises `model_setup.DownloadCancelled` or `model_setup.SetupError`
    - creates the `default` preset at the end
  - `uninstall(model_dir: Path, keep_voices: bool = True) -> None`

## Status (2026-10-15)
- [x] 1. In-app install:
  - `vibetts/omnivoice_setup.py` with `install`, `status`, `uninstall` and `DOWNLOAD_BYTES`, plus a CLI. The graph and weight map ship in `vibetts/data/omnivoice/`.
  - The numpy conversion output is **byte-identical** to the torch-built `.data`. Conversion takes ~5 s, with a private RAM peak of about 40 MB.
  - A local-files install takes 18 s. Download size: 2.85 GB, or 3.50 GB with cloning.
  - Running `install` again to add the encoders is idempotent and covered by a test.
- [x] 2. Any-language engine: `omni:<name>` ids with legacy migration, `lang` per request, config `engine`/`omni_voice`/`croatian_voice`, the routing table, and `long_chunks` for every OmniVoice request.
- [x] English quality (word match): `omni:default` 0.96, English-designed 0.98, Croatian-designed 0.97. Croatian design text keeps the better Croatian quality (0.80 vs 0.65), so Croatian stays the default sample language.
- [x] 3. Settings tabs General | Voice | Kokoro | OmniVoice | About:
  - Voice tab: engine radio.
  - OmniVoice tab: install, steps, the Voices list and the inline `VoiceCreator`. The modal dialog was removed.
  - Tray: Engine submenu and an OmniVoice voice group.
- [x] Orchestrator fix: the download-size text now says "2.9 GB, or 3.5 GB with voice cloning" in the UI and README.
- [x] Verification:
  - `pytest -q` passes (218 passed, 1 skipped; the skip needs `onnx`).
  - `scripts/build.ps1` builds, and the graph and map are bundled in `_internal/vibetts/data/omnivoice`.
  - On a copy of the exe with models and `engine="omnivoice"`, `--say` worked for English (1.6x real time) and Croatian (1.3x).
- [ ] Not run:
  - pressing Install in the built exe, which needs a real 2.9-3.5 GB download (`dist\VibeTTS` is left fresh for this);
  - a manual walk through the tabs and tray;
  - listening.
