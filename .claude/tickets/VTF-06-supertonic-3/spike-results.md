# VTF-06 spike: Supertonic 3 (multilingual, CPU)

Date: 2026-10-19. Replaces OmniVoice (removed: poor Croatian, slow, 1.3 GB, non-commercial weights).

## Setup

- SDK `supertonic` 1.3.1 (MIT) in a throwaway venv: `.claude/temp/vtf06/venv`. Deps: onnxruntime, numpy, soundfile, huggingface-hub.
- Model `Supertone/supertonic-3` (OpenRAIL-M), auto-downloaded to `~/.cache/supertonic3`: 385 MB (fp32 ONNX + 10 voice style JSONs).
- 31 languages incl. `hr` (+ `na` fallback for unknown text); language passed per call (`lang="hr"`), wrapped as `<hr>…</hr>` tokens.
- Voices: F1–F5, M1–M5 (preset styles; no cloning in the open package).
- Output 44.1 kHz mono float32 (VibeTTS works at 24 kHz: needs resampling).
- Script: `.claude/temp/vtf06/spike.py`; raw numbers `.claude/temp/vtf06/results.json`; samples + `listen.html` in the same folder.

## Speed (this machine, 32 logical CPUs, default ORT threads)

| Measure | Value |
|---|---|
| Model load, cached | 1.1 s (23 s first run incl. download) |
| Short first sentence ("Dobar dan, kako ste?") | 0.36 s |
| RTF, 8 steps (default), all voices, hr and en | 0.087 – 0.104 (~10-11x real time) |
| RTF by steps (hr story, F1) | 4: 0.048 · 6: 0.072 · 8: 0.087 · 12: 0.128 · 16: 0.170 |

Kokoro on the same English text and machine (`kokoro_bench.py`, app venv): RTF 0.117 – 0.126 for int8 and fp32 alike (~8x real time), short first sentence 0.20 s, load 1.0 s. So Supertonic at 8 steps is ~25% faster on longer text and ~0.15 s slower on a short first sentence (6 steps would roughly even that out). OmniVoice was ~1.4x with 2-3 s to first audio.

Decision (user, 2026-10-19): if it holds up, Supertonic 3 replaces Kokoro entirely, and output stays at its native 44.1 kHz.

## Quality

- Listening test (user): pending — compare voices, Croatian accents, numbers/dates (`hr_numbers`).
- User's earlier impression from the HF Space: "seems really good".

## Open points for the plan

- Engine design: second engine next to Kokoro, or replace Kokoro? (Kokoro stays smaller: 89 MB int8 vs 385 MB.)
- Language choice per request: config default language + per-call override; Croatian auto-detection could return (cheap `looks_croatian` from git history) now that hr actually works.
- Use the SDK (adds soundfile, huggingface-hub) or call the 4 ONNX graphs directly like Kokoro (~200 lines, onnxruntime is already a dependency via kokoro-onnx).
- SDK does its own chunking + 0.3 s silence; VibeTTS already chunks, so call per chunk with `max_chunk_length` large enough.
- Download from Settings like the Kokoro model (385 MB), not on first use.
- Repo `supertone-inc/supertonic` is archived: pin the model revision.
