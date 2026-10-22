# VTF-09 spike: Higgs Audio v3 TTS 4B, community ONNX `cpu_int4` (CPU)

Date: 2026-10-22. Step 0 of `plan.md`. Throwaway measurement only: no tracked file was changed.
Machine: Ryzen AI Max+ 395 (16 cores / 32 threads), 32 GB RAM, Windows 11, no CUDA.

## Gate recommendation

**Export-only** (neither Stop nor Live).

- Stop gate not hit: no NaN/Inf, no silence, no crash from the int4 kernels. With a Croatian reference clip the Croatian output is intelligible (ASR word-match 0.92 on the 49-word paragraph, 1.00 on the short sentence, 0.80 on numbers/dates). Caveat: zero-shot Croatian (random voice) is unreliable (see ASR table).
- Live gate failed: best RTF is **1.37 to 1.55** zero-shot and **1.65 to 1.78** with a voice-clone reference (8 ORT threads); at default threads it is 1.9 to 2.3. The limit is ≤ 0.8. The AR loop only reaches about 15 frames/s at default threads and 19 to 25 frames/s at 8 threads, against the 25 frames/s needed for real time (RTF 1.0). Reaching 0.8 would need roughly 31+ frames/s, i.e. about 1.5x to 2x faster than the best measured here.
- First short chunk: zero-shot HR short 2.6 s (8 threads) / 3.7 s (default) after the model is loaded and warm. With a clone reference it is 6 to 9 s (the reference adds a ~300 to 420 token prefill of 1.8 to 2.4 s, plus the AR loop; the first call of a session also loads and runs `audio_encoder`). The gate is ≤ 3 s, so this also fails for any usable voice.
- Not measured: DirectML / GPU (see Not run).

Suggested consequence for the plan: ship Higgs as an export / MCP engine (about 1.7 to 2.3 s of CPU per second of audio, so a one-minute read takes about 2 minutes), voices only via reference clips (see Presets), and keep live hotkey reads on Kokoro and Supertonic.

## Setup

- Repo `onnx-community/higgs-audio-v3-tts-4b`, pinned revision **`ecc25d6cc6e48909401bfa7ce5ac95e84f41f331`** (main on 2026-10-22).
- Download base: `https://huggingface.co/onnx-community/higgs-audio-v3-tts-4b/resolve/ecc25d6cc6e48909401bfa7ce5ac95e84f41f331/<path>`. All files downloaded; sizes match the HF API (`?blobs=true`).
- Venv `.claude/temp/vtf09/venv` (Python 3.13.12): onnxruntime 1.31.0 (CPU), numpy 2.5.4, tokenizers 0.23.3, soundfile 0.14.0, soxr 1.1.0, psutil 7.2.2, huggingface_hub 2.2.0 (not needed in the end; curl was used). ASR in `.claude/temp/vtf02/asrvenv` (faster-whisper, model `medium`, int8, beam 5, language forced per sample).
- Model dir: `.claude/temp/vtf09/model/` (not tracked). Scripts: `.claude/temp/vtf09/scripts/` (`higgs_patched.py` = patched inference.py, `run.py` bench, `asr9.py`, `mkpresets.py`, `mklisten.py`, `texts.py`). Raw numbers: `.claude/temp/vtf09/out/results_*.json` (+ `.asr.json`).

### Files and sizes (bytes), `cpu_int4/`

| File | Bytes |
|---|---|
| audio_embed.onnx | 13461914 |
| audio_encoder.onnx | 654423379 |
| audio_heads.onnx | 13462729 |
| audio_tokenizer.onnx | 86516167 |
| chat_template.jinja | 2481 |
| genai_config.json | 1501 |
| llm_decoder.onnx | 384495 |
| llm_decoder.onnx.data | 2291924992 |
| manifest.json | 2771 |
| text_embed.onnx | 777912621 |
| tokenizer.json | 11433924 |
| tokenizer_config.json | 2044 |
| **Total cpu_int4/** | **3849529018** (3.85 GB; 3.58 GiB) |

Root files fetched: `inference.py` 25174, `stream_inference.py` 7723, `STATUS.md` 16724, `Onnx.md` 1541, `LICENSE` 25685, `README.md` 42844, `ref.wav` 4266924 (48 kHz mono, 44.4 s), plus `PROMPTING.md` 2812 and `CAPABILITIES.md` 4795. No `NOTICE` file exists in the repo.

For `MODEL_FILES` later: the runtime needs the 6 graphs + `.onnx.data` + `tokenizer.json` + `tokenizer_config.json` + `manifest.json` (read by the reference code) + `genai_config.json` (not read by the loop; optional) + `LICENSE` (root, 25685). `chat_template.jinja` is unused by the TTS loop (it uses the raw `<|tts|>` prompt). `audio_encoder.onnx` (654 MB) is only needed to encode a reference clip; presets with pre-encoded codes would not need it at read time.

## Code changes needed to `inference.py`

The repo's `inference.py` imports `safetensors`, `transformers` (AutoTokenizer) and `librosa`, which are not available in a lean runtime. It does not run as shipped in the spike venv. The patched copy (`scripts/higgs_patched.py`) changes only:

1. Tokenizer: `tokenizers.Tokenizer.from_file(tokenizer.json)` instead of `transformers.AutoTokenizer`. The special-token ids come from `get_added_tokens_decoder()` (`<|tts|>`, `<|text|>`, `<|audio|>`, `<|ref_text|>`, `<|ref_audio|>` resolved fine). No safetensors / standalone dir.
2. `librosa.resample` replaced by `soxr`.
3. `intra_op_num_threads` option; `audio_encoder` session created lazily; timing counters (prefill, AR, codec).

The AR loop, sampling (temperature 0.8, top-k 50, top-p 1.0), delay pattern, repeat guard (32) and the NaN guard are unchanged. The reference-audio path was exercised with its default `HOP=960` padding.

Issues found in the reference code that the port must handle:

- **Repeat guard can empty the output.** With seed 0, zero-shot `"Hello, how are you today?"` stopped by the repeat guard within the first 32 frames; the guard deletes the last 32 frames, leaving zero frames, and the codec then raises `Invalid input shape: {0}` (ONNXRuntimeError in `acoustic_decoder/conv1`). Seeds 1 to 5 were fine. The port needs a guard for an empty/too-short result (retry with another seed, or fail clearly).
- **Zero-shot Croatian is a seed lottery** (random voice and random content drift, see ASR table). Cloning was stable.
- `stop=repeat` ended both zero-shot paragraphs (EOC never fired); the audio was still complete up to the dropped tail (ASR matched the text), but the ending can be noisy. With a clone reference all full-length reads ended via EOC except where noted.
- Reference clips: the original `ref.wav` is 44 s, which gives a 1160+ token prefill (7 to 8 s prefill) and also gave worse output; a trimmed ~10 s clip is what to use (see Presets). The encoder needs lengths padded to 960 samples (done in the reference code).

## Speed

Short-run timing (zero-shot, seed 0, load excluded; one warm-up generation first). Numbers are wall-clock seconds on an otherwise idle machine; ORT intra-op threads as listed.

| Threads | HR short (1.92 s audio) synth s / RTF | EN paragraph (14.2 s) synth s / RTF | HR paragraph (13.9 s) synth s / RTF | AR frames/s (HR para) |
|---|---|---|---|---|
| default (32) | 3.7 / 1.92 | 27.7 / 1.95 | 27.7 / 1.99 | 14.6 |
| 4 | 3.2 / 1.67 | 25.4 / 1.79 | 25.9 / 1.86 | 15.9 |
| 6 | 3.0 / 1.55 | 23.5 / 1.66 | 23.8 / 1.71 | 17.1 |
| **8** | **2.6 / 1.37** | **21.2 / 1.49** | **21.6 / 1.55** | **18.9** |
| 16 | 3.6 / 1.89 | 28.7 / 2.03 | 29.0 / 2.09 | 13.8 |

(Run-to-run variance: the default-thread rows in the full-suite run were 5 to 10% slower than the first timing run, e.g. HR paragraph 30.7 s.) Short sequences run at 20 to 25 frames/s at 8 threads (zero-shot short phrases RTF 1.23 to 1.45); the rate drops as the KV cache grows and with a longer clone prefill. The codec decode is small (0.1 s short, 0.6 s paragraph); prefill is 0.05 s for zero-shot short, 0.5 s for the HR paragraph, 1.8 to 2.9 s with a clone reference (300 to 525 tokens).

Voice clone speed (HR paragraph, F1_hr reference, 22.7 s audio): default threads 52.1 s (RTF 2.29), 8 threads 40.5 s (RTF 1.78, 15.7 frames/s); EN paragraph with F1_en: default 34.1 s (RTF 2.11), 8 threads 26.7 s (RTF 1.65). Clone HR short, 8 threads: 8.8 s wall for 2.8 s of audio, including the first-time encoder load and encode (about 1 s); the prefill was 2.4 s. A cached reference would take about 6 s.

Load and memory:

| Measure | Value |
|---|---|
| Model load (5 sessions: llm_decoder 3.6 s, text_embed 0.8 s, codec 0.3 s, rest <0.1 s) | 5.1 to 5.3 s (files in OS cache; cold disk not measured) |
| `audio_encoder` load (lazy, first clone only) | 1.1 s |
| RSS after load | 4.4 GB |
| Peak RSS, zero-shot generation | 4.8 GB (4.9 GB at 8 threads) |
| Peak RSS, with clone (encoder loaded) | 6.2 to 7.5 GB (8.6 GB with the original 44 s reference) |

Not an issue for 32 GB, but a lot for a resident tray app: unload after use, as the plan says.

## Intelligibility (faster-whisper medium; word-match = difflib ratio on lowercase, punctuation-stripped words)

No NaN/Inf in any output; RMS 0.02 to 0.13, peak ≤ 1.09 (valid runs). Croatian ASR with Whisper is itself noisy (diacritics, ije/je variants), so ratios of 0.9+ are near ceiling.

Zero-shot (random voice, seed 0):

| Sample | Ratio | ASR transcript |
|---|---|---|
| EN short | n/a (crash, see issues) | seed 1 to 5: 1.00 each, "Hello, how are you today?" |
| EN paragraph | 0.83 | "...watching a gray flea for more than a hundred years... before the first to start." |
| HR short | 0.50 | "Vim radan, kako ste." (seeds 1 to 5: 0.33, 0.00, 0.67, 1.00, 1.00) |
| HR paragraph | 0.64 | "Stari svjetionik stajavao je na rubu litice i š' od stotinu godina promatrao sivo more. Svake veća je čuvarelo zuske stepenice, ..." |
| HR numbers/dates | 0.17 | "Sastavak je 14. ožurjka, 23. 26. od 9. r. pravstava ro..." |

Voice clone (F1 reference, Supertonic clip, same language as the text):

| Sample | Ratio | ASR transcript |
|---|---|---|
| EN short | 1.00 | "Hello, how are you today?" |
| EN paragraph | 0.92 | "...a few quiet in his log book before the first stars appeared above the harbor." (dropped "lines") |
| HR short | 0.57 (default run) / 1.00 (8-thread run) | "Dobardan, kako ste?" / "Dobar dan, kako ste?" |
| HR paragraph | 0.92 | "Stari svijetionik stajao je na rubu litice i više od stotinu godina promatrao sivo more. Svake je večeri čuvar uz uske stepenice došao do svijetla, ... pri nego što su se iznad luke pojavile prve zvijezde, a šume je šumjela šaptom." |
| HR numbers/dates | 0.80 | "Sastanak je 14. ožujka 2026 u 9.30, a cijena je 1. 20.000 euro." (date and time right; "1.250,00 eura" wrong; the word "eura" read as "euro") |

Summary: ASR word-match EN 0.83 zero-shot / 0.92 to 1.00 clone; HR 0.50 to 0.64 zero-shot (0.00 to 1.00 across seeds on the short phrase) / 0.92 to 1.00 clone (HR numbers 0.80 clone, 0.17 zero-shot). The clone prompt (reference clip plus transcript) is what makes it reliable. "Dobar dan, kako ste?" without a reference is wrong on 3 of 6 seeds. Number reading is the weak spot ("1.250,00" mis-read), so Croatian numbers likely need normalisation in front of Higgs (a later step, not decided here).

## Voice consistency, cross-lingual, speed tags

Consistency (HR paragraph split into 3 sentences, reference F1_hr and M1_hr, seed 0 for every chunk; the same reference gives the same voice by construction, listening needed):

| Reference | Sentence ratios (ASR) | Durations | Notes |
|---|---|---|---|
| F1_hr | 0.93 / 0.74 / 1.00 | 7.0 / 9.7 / 9.1 s | sentence 2 ended "o temnjemnik" instead of "u dnevnik" |
| M1_hr | 0.93 / 0.95 / 1.00 | 7.0 / 10.5 / 7.3 s | |

Durations are in a sane range for 14 to 19 words (about 2 words/s: the Higgs clones talk slowly compared with the Supertonic references). All ended via EOC; RMS is stable across sentences (0.043 to 0.052). Whether the timbre truly stays identical across chunks is for the user's listening test.

Cross-lingual (English reference F1_en reading Croatian text):

| Text | Ratio | Result |
|---|---|---|
| HR short | 0.75 | "Dopar dan, kako ste?" |
| HR paragraph | 0.67 | partly garbled ("Star, sviječnak ...", "u ucu") |
| HR numbers | 0.00 | empty ASR, ended via repeat guard (3.5 s) |
| 3 separate sentences | 0.80 / 0.34 / 0.12 | sentences 2 and 3 degraded (babble, "vomoooo") and ended via the repeat guard |

Cross-lingual cloning works poorly with these English references (and varies by reference: M1_en gave 0.84, F2_en 0.57, F1_en 0.67, M2_en 0.08 at seed 0 but 0.91 at seed 1). Same-language references are clearly better. Recommendation: ship per-language reference clips (at least Croatian) rather than relying on an English clip for other languages.

Speed tags:

| Case | none | `<|prosody:speed_slow|>` | `<|prosody:speed_fast|>` |
|---|---|---|---|
| HR short zero-shot, seed 0 | 1.9 s (ASR 0.50) | 1.4 s (shorter, ASR 0.67 "Dobar dan!") | 2.0 s (ASR 1.00; louder, peak 1.09) |
| HR paragraph, F1_hr clone | 22.7 s (0.92) | 22.4 s (0.92) | 21.0 s (0.79) |

The tags change the output (they are not ignored) but not by the advertised 0.85x / 1.2x: on the paragraph slow is -1% and fast is -7% in duration, and on the short zero-shot sample the "slow" tag gave a shorter clip (the zero-shot voice and content also change with the extra token, so one sample does not isolate speed). Effect is weak and inconsistent; mapping the app's speed slider onto these tags would be approximate at best. Listening needed.

## Presets

Reference clips made with Supertonic 3 (app engine, 24 kHz mono 16-bit WAV + `.txt` transcript) in `.claude/temp/vtf09/presets/`: `F1_en`, `F2_en`, `M1_en`, `M2_en` (one neutral English sentence, 9.9 to 10.5 s) and `F1_hr`, `M1_hr` (Croatian sentence, 13.4 s). Also `repo.wav` + `repo.txt`: the first 10.4 s of the repo's `ref.wav`; the repo ships no transcript, so I produced it with Whisper medium (two sentences: "Now let's discuss the process of photosynthesis in detail. Plants absorb light energy through chlorophyll, which is located in the chloroplasts.").

Each used as the reference for the HR paragraph (seed 0; ASR ratio):

| Reference | Ratio | Duration | Comment |
|---|---|---|---|
| F1_hr | 0.92 | 22.7 s | best |
| M1_hr | 0.69 | 20.2 s | prefixed spurious words at the start, ended via repeat guard |
| M1_en | 0.84 | 17.4 s | |
| F1_en | 0.67 | 20.6 s | |
| F2_en | 0.57 | 19.9 s | ended via repeat guard |
| M2_en | 0.08 (seed 0), 0.91 (seed 1) | 17.2 s / 20.9 s | seed 0 produced babble; seed 1 fine |
| repo (trimmed) | 0.77 | 17.1 s | English-reference cross-lingual |

Reliability is seed dependent, so a read should check for early repeat-guard stops and retry with a different seed. A same-language reference (F1_hr, and M1_hr for short sentences: sentences 0.93 to 1.00) is the dependable path. For the later presets step: make Croatian clips for each preset voice (Supertonic F1/F2/M1/M2 reading Croatian), and verify each with ASR; the English-only clips are not good enough for Croatian. The user should listen and keep whichever sound good. Note that using Supertonic as the source means the Higgs presets inherit its voices, and Supertonic output is OpenRAIL-M licensed (check use as cloning reference before bundling).

The first set of `pre_repo_*` runs (`out/extra8/`) used the original 44 s `ref.wav` against the 10 s transcript (a bug in my bench script, since fixed); those results were discarded and the `repo` row above is from the corrected re-run (`out/extra2_8/`). `listen.html` excludes the discarded files.

## Not run / limits

- DirectML (`onnxruntime-directml` on the Radeon 8060S): **not tried**. The decoder is a ModelBuilder int4 graph using `MatMulNBits` and `GroupQueryAttention` contrib ops, and the plan marks DML as optional; I skipped it rather than fiddle with a second venv. This stays unmeasured.
- Thread counts 2, 12 and 24 not measured. 8 was best of {default, 4, 6, 8, 16}; behaviour looks memory-bandwidth bound (more threads hurt).
- `stream_inference.py` (streaming) not run; first-audio times above are for the complete non-streaming result (time until the whole short clip is ready), not a streamed first chunk.
- One speaker clip per reference, seed 0 in most runs: the numbers above are single samples, not distributions (except the 5-seed zero-shot short runs).
- Cold-disk load time not measured.
- Licence: `LICENSE` is the "Boson Higgs Audio v3 Research and Non-Commercial License". Its definition of Commercial Purpose includes use made available to end users as an application or plug-in, so a distributed app that downloads and runs the weights needs the README's licence note (non-commercial personal use; the plan's Creator Use Grant mention was not verified in this spike; a read of `LICENSE` beyond the first sections was not done).

## Quality

- Listening test (user): pending. Open `.claude/temp/vtf09/listen.html` (groups: references, zero-shot, clone, consistency, cross-lingual, speed tags, presets, seeds; each with text, ASR, ratio, RTF).

## DirectML (Radeon 8060S)

Date: 2026-10-22 (work item D). Throwaway measurement only; nothing tracked was changed. Scripts, logs and raw JSON are in `.claude/temp/vtf09dml/` (`higgs_patched.py` = the CPU port plus a per-graph DML switch, `run.py` bench, `t1.py` to `t7.py` diagnostics, `out/results_dml_*.json`).

**Conclusion: no. DirectML does not work for this model on this machine.** The int4 `llm_decoder` runs on DML but computes wrong results once the KV cache is used, it is about 3x slower per step than CPU, and the codec graph does not run on DML at all. RTF is 5.6 to 8.6 (CPU: 1.37 to 1.78) and the output is not speech. Neither RTF < 1 nor <= 0.8 is reachable with onnxruntime-directml 1.24.4.

### Setup

- Venv `.claude/temp/vtf09dml/venv` (Python 3.13.12): onnxruntime-directml **1.24.4** (providers `DmlExecutionProvider`, `CPUExecutionProvider`), numpy 2.5.4, tokenizers 0.23.3, soxr 1.1.0, soundfile 0.14.0, psutil 7.2.2. The CPU spike used onnxruntime 1.31.0; 1.24.4 was the directml wheel pip resolved, so a newer DML build might behave differently (none available).
- Model: the same `cpu_int4` graphs, unchanged. DML sessions use `enable_mem_pattern=False`, `ORT_SEQUENTIAL`, providers `["DmlExecutionProvider", "CPUExecutionProvider"]`. Graph optimisation level ALL / BASIC / DISABLED made no difference. 8 threads for CPU-side sessions. Single samples, seed 0.
- Because the DML decoder output goes non-finite, `FORCE=1` in the bench replaces non-finite logits with finite values so the AR loop completes and can be timed. Those runs are timing only; their audio is invalid. Without `FORCE` the stock NaN guard raises during the first generation.

### What ran where (ORT verbose log, `llm_decoder.onnx` on DML)

617 nodes on `DmlExecutionProvider`, 2 on CPU (one `Gather`, one `Cast`). This includes all 180 `MatMulNBits` and all 36 `GroupQueryAttention` (plus `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `QuickGelu`): the DML EP in 1.24.4 registers these contrib ops, so nothing falls back to CPU in the decoder. The problem is correctness and speed, not missing kernels.

Other graphs on DML vs CPU (random inputs, one call):

| Graph | Result on DML |
|---|---|
| audio_embed, audio_heads, text_embed | run and match CPU (max diff 1e-7 to 1.5e-5; text_embed exact); sub-millisecond, speed irrelevant |
| audio_tokenizer (codec decode) | **fails** for 13, 50 and 100 frames: `ConvTranspose` node `/acoustic_decoder/block.1/conv_t1/ConvTranspose`, DML error `80070057 The parameter is incorrect` |
| audio_encoder | runs (0.21 s vs 0.59 s CPU for 10 s of noise); code correctness not verified (noise input) |
| llm_decoder | runs, but wrong and slow (below) |

So "(1) all graphs on DML" cannot complete: the codec raises on the first generation. The runnable variants are (1b) everything except the codec on DML, and (2) `llm_decoder` only on DML.

### llm_decoder correctness on DML

Compared with the CPU graph on identical inputs (HR short prompt, 10 tokens):

- Prefill: the last hidden state is close (about 0.01 to 0.05 off, magnitudes up to 90), but the returned `present.*.key/value` tensors already differ from CPU (layer 0 key max diff 12.7 of 200; layer 35 value 34 of 60).
- Decode step with the CPU's own KV cache as input: hidden state max diff 4.9 (about 5% of range).
- Second decode step (past = DML's own `present`): hidden state differs by about 88 to 100, i.e. different values entirely. The old positions of `present.*.key` come back wrong (the difference equals the full tensor magnitude), so DML's `GroupQueryAttention` does not carry the past KV into `present` correctly for this graph's inputs (separate past/present tensors plus `attention_mask`). Non-finite values appear in `audio_heads` a few frames later.
- Net effect: never intelligible. `FORCE` zero-shot runs do not end naturally (900-frame cap, 36 s of babble); ASR (faster-whisper medium) returned empty text for all four clips (word ratio 0.00). Not silent (RMS 0.06 to 0.08) and no NaN in the waveform (logits were sanitised), but not speech.

### Speed

Decode step, llm_decoder, 100-token cache, 30 steps, median: CPU 8 threads **55 ms**; DML **157 ms**; DML with IOBinding keeping the KV cache on the device (`present.*` bound to `dml`, next past passed as `OrtValue`) **178 ms**. IO binding / a device-resident KV cache does not help (slightly slower), so the cost is not the host/device copy. A profiled run shows time spread over many small per-layer kernels (per step, inflated by profiling: MatMulNBits about 95 ms, SkipSimplifiedLayerNorm about 43 ms, GroupQueryAttention about 37 ms, SimplifiedLayerNorm about 29 ms), the pattern of per-op dispatch overhead at batch 1 with 36 layers and about 600 nodes, not of a compute-bound graph. Prefill of 100 tokens: CPU 0.56 s, DML 0.73 s.

Full generations (HR texts from `texts.py`, seed 0, 8 threads, load excluded, `FORCE=1`, audio invalid, timing only):

| Config | Load s | Peak RSS | Clip | Synth s | RTF | Frames/s | Valid |
|---|---|---|---|---|---|---|---|
| CPU reference (spike) HR short | 5.1 | 4.9 GB | 1.9 s | 2.6 | 1.37 | about 20 | yes |
| CPU reference (spike) HR paragraph | 5.1 | 4.9 GB | 13.9 s | 21.6 | 1.55 | 18.9 | yes |
| (1b) all but codec on DML, zero-shot HR short | 3.7 | 3.5 GB | 35.7 s (900-frame cap) | 200.9 | 5.63 | 4.5 | no, ASR empty |
| (1b) zero-shot HR paragraph | | | 35.7 s (cap) | 254.1 | 7.11 | 3.6 | no, ASR empty |
| (1b) clone F1_hr, HR short (prefill 418 tokens, 1.25 s) | | | 21.2 s (EOC) | 172.3 | 8.15 | 3.2 | no, ASR empty |
| (1b) clone F1_hr, HR paragraph (prefill 525 tokens, 1.35 s) | | | 35.7 s (cap) | 306.0 | 8.57 | 3.0 | no, ASR empty |
| (2) llm_decoder only on DML, zero-shot HR short | 5.1 | 2.8 GB | 35.7 s (cap) | 236.7 | 6.63 | 3.8 | no (NaN guard fires without FORCE) |
| (1) everything on DML incl. codec | 3.6 | n/a | n/a | n/a | n/a | n/a | fails in codec (ConvTranspose) |

Because the DML runs ramble up to the 900-frame cap, the RTF column divides by a clip that is far too long (the real speech would be 1.9 s / 13.9 s), so frames/s is the reliable number: 3.0 to 4.5 on DML vs 15 to 25 on CPU. Real time needs 25 frames/s (40 ms per frame in total); DML is at about 220 to 330 ms per frame. The first short chunk would be several seconds even without the wrong output, so the first-chunk target also fails. Load time on DML is a little shorter (3.7 s vs 5.1 s) and RAM is lower (2.8 to 3.5 GB peak), neither matters here.

### Verdict and consequences for the plan

- DML does not reach RTF < 1 or <= 0.8. It is 3 to 5 times slower per frame than CPU, and the decoder output is wrong. It is not a speed-up path for this model with onnxruntime-directml 1.24.4.
- If someone wants to revisit: (a) a DML `GroupQueryAttention` bug or unsupported input pattern (separate past/present, no shared buffer); (b) the codec's `ConvTranspose`; (c) a newer ORT DML build than 1.24.4; (d) an fp16 or DML-targeted export of the graphs (not checked; the repo ships only what is in the pinned revision). None was pursued.
- Dependency note, for completeness (moot since DML is dropped): `onnxruntime-directml` and `onnxruntime` both provide the `onnxruntime` module and cannot be installed side by side, and `kokoro-onnx` requires `onnxruntime`, so pip sees a conflict. Using DML would need a separate environment or process for Higgs, or a swap to `onnxruntime-directml` for every ONNX engine in the app (Kokoro and Supertonic would then also go through the DML EP unless forced to CPU).
- Recommendation unchanged: Higgs ships as Export-only; live reads stay on Kokoro and Supertonic.
