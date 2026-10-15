# VTF-02 Phase 0 spike results

Date: 2026-10-15. Scripts: `.claude/temp/vtf02/` (copies of the durable ones in `spike-scripts/` next to this file). WAVs: `.claude/temp/vtf02/out/`.

## TL;DR

- **The community ONNX LLM (`Emerald7664/OmniVoice-Onnx` `llm_decoder.onnx`, fp16 and int4 builds) is unusable as shipped.** It is built with onnxruntime-genai `GroupQueryAttention`, which is strictly causal, but OmniVoice is a bidirectional (non-causal) model (upstream passes an all-True `(B,1,S,S)` mask). Verified: changing audio positions does not change hidden states of earlier positions (`causal_check.py`: max dh before audio region = 0.0). Output with it is noise for Croatian and English alike (ASR: "Zimcimcimc", "Woo-hoo!"; word match ~0.00-0.04 at 8/16/32 steps).
- **Fix works:** exporting our own bidirectional Qwen3 LLM from the upstream `k2-fsa/OmniVoice` safetensors with torch (one-time, dev-side; `torch_ref.py`) and keeping the community embeddings / heads / Higgs decoder ONNX gives clearly Croatian speech. onnx-vs-torch max abs diff 6e-5.
- **Speed:** with the correct model, CFG (guidance 2.0, upstream default) is needed for good quality and it costs ~1.7x. On this (fast, 16-core) CPU the best trade-off is **5-6 steps + CFG 2.0: RTF 0.47-0.57, first sentence ~1.5-1.9 s**. The plan's "32 steps RTF ~0.48" claim does not reproduce (it was for the broken causal graph and/or a different CPU): 32 steps = RTF 1.7 here.
- **Recommendation: CONDITIONAL GO** (see end). Hard new requirement for Phase 1: we must produce and host our own bidirectional LLM ONNX (int4 ~235 MB works as well as fp32 1.76 GB).

## Machine

- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 16 cores / 32 threads, 31.6 GB RAM, Windows 11, no CUDA. This is a fast CPU; a typical laptop will be slower (expect RTF roughly 1.5-3x worse).
- onnxruntime 1.31.0 (CPUExecutionProvider), numpy 2.5.4, Python 3.13.12 (`.venv`).
- Installed into project `.venv` for the spike: `tokenizers 0.23.3`, `soxr 1.1.0`, `soundfile 0.14.0` (these are what Phase 1 would add to requirements; torch/transformers NOT installed there).
- Scratch venvs (under `.claude/temp/vtf02/`): `asrvenv` (faster-whisper small int8, onnx, soxr), `torchvenv` (torch 2.14.1+cpu, transformers 5.x, onnx, onnxscript, onnxruntime; used only for the one-time LLM export and quantization).

## Downloaded files (Phase 2 pins these)

Repo `Emerald7664/OmniVoice-Onnx` (Apache-2.0), commit **`f312525229ea4c4d9101c2b3e317ac77d995ec59`**. Base URL: `https://huggingface.co/Emerald7664/OmniVoice-Onnx/resolve/f312525229ea4c4d9101c2b3e317ac77d995ec59/<path>`

Used by the spike (root = "cpu_fp16" build, fp16 audio parts):

| file | bytes |
|---|---|
| audio_embeddings_encoder.onnx | 2172 |
| audio_embeddings_encoder.onnx.data | 327426048 (sha256 e2c605b0...3316660, matches LFS oid) |
| audio_heads_decoder.onnx | 16795584 |
| tokenizer.json | 11423986 |
| tokenizer_config.json | 533 |
| config.json / model_config.json / omnivoice_manifest.json | 1416 / 341 / 3517 |
| audio_tokenizer/fp16/higgs_decoder.onnx | 308494 |
| audio_tokenizer/fp16/higgs_decoder.onnx.data | 43100160 |
| audio_tokenizer/fp16/acoustic_encoder.onnx(+.data) | 295846 + 102615040 |
| audio_tokenizer/fp16/quantizer_encoder.onnx | 6091791 |
| audio_tokenizer/fp16/semantic_encoder.onnx(+.data) | 334295 + 218235904 (**fails to load on ORT 1.31 CPU**: LayerNormalization float/float16 type error) |
| audio_tokenizer/acoustic_encoder.onnx (fp32, for cloning) | 205546480 (sha256 starts 431a5e58, matches LFS oid) |
| audio_tokenizer/semantic_encoder.onnx (fp32, for cloning; works) | 436736856 |
| audio_tokenizer/quantizer_encoder.onnx (fp32 dir) | 12131293 |
| llm_decoder.onnx + .data (int4, **causal, unusable**) | 298798 + 296484864 |
| int4/ (all-int4 build: embeddings 87160832, heads 4462676, llm 282365952 + 1284416) | downloaded, embeddings int4 not needed; its I/O differs (audio_mask bool, fp32 out) |

Also fetched for the export (dev-side only, NOT a runtime dependency): `k2-fsa/OmniVoice` commit `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` (`model.safetensors` 2450344112 B, config.json, tokenizer files; no license field in the HF API metadata, upstream GitHub repo is Apache-2.0 per community card - confirm before redistributing the derived ONNX).

Our own artifacts (must be hosted somewhere for Phase 2): `models/bidir/llm_bidir_fp32.onnx` 1762990464 B; `llm_bidir_int4.onnx` 1211063 B + `.onnx.data` 234196992 B (MatMulNBits block 128 symmetric, via `quant.py`); `llm_bidir_int8.onnx` 444203803 B (slower, 165 ms/step vs 100 ms, not recommended).

Runtime sets: fp16 audio parts (embeddings 327 MB + heads 17 MB + Higgs decoder 43 MB) + bidir int4 LLM (235 MB) + tokenizer (11 MB) = ~635 MB. Cloning adds fp32 encoders ~655 MB (or fix/skip fp16 semantic encoder).

## What upstream does that the ONNX `inference.py` omits (and what I ported in `spike.py`)

| upstream (k2-fsa/OmniVoice `models/omnivoice.py`) | ONNX port `inference.py` | ported in spike |
|---|---|---|
| LLM attention is bidirectional (all-True 4D mask) | genai GQA = causal (breaks the model) | **own bidirectional export** (torch_ref.py) |
| Prompt = `[<|denoise|>]<|lang_start|>hr<|lang_end|><|instruct_start|>None<|instruct_end|>` + `<|text_start|>text<|text_end|>` (+ ref codes) + masks; text ids copied to all 8 rows | raw `tokenizer.encode`, no style/lang/text markers | yes (special tokens are in tokenizer.json; `tokenizers` package encodes them as added tokens) |
| `audio_mask` True for ref codes AND target; False only for style+text | ref prefix marked non-audio (bug for cloning) | yes |
| Language id (`hr`) via `<|lang_start|>` | none | yes (`None` string when unknown) |
| Duration: `RuleDurationEstimator` (per-script char weights, digits 3.5, punct 0.5, space 0.2, marks 0), reference = "Nice to meet you." / 25 frames when no ref; `<50`-frame boost curve; /speed | fixed `--num_audio_tokens` | yes: verbatim copy of upstream `duration.py` (pure python, no deps, ~9.2 s est for the 150-char sentence) |
| Scheduler: `t_shift=0.1` timestep warp, tokens-per-step schedule over all `T*8` cells | ceil(remaining/steps) frames per step, confidence = prob-weighted per frame | yes |
| Per-(codebook,frame) scoring = max log-prob minus `layer_penalty_factor(5) * codebook_idx`, Gumbel position noise (temp 5), unmask top-k cells | uses `[8,8,6,6,4,4,2,2]` weights as confidence weights (not what upstream does; those weights are only the training loss weights) | yes |
| CFG: guidance 2.0, uncond = target-only sequence (no text/ref); `log_softmax(c + s*(c-u))` | none | yes (separate batch-1 pass for the shorter uncond sequence) |
| mask-token logit set to -inf before argmax | only first 1024 sliced | yes |
| Post: silence trim (-50 dB, keep 100 ms), mid-silence removal, peak-normalise to 0.5 (no ref) or ref-RMS scale, 0.1 s fade + 0.1 s pad | none | simplified (edge trim, peak 0.5, fade, pad; no mid-silence removal) |
| Text normalisation (`normalize_text`, opt-in, num2words for numbers) | none | not ported (numbers like "15." / "2026." were spoken fine by the model: "15 listopada 2026 godina"; the model reads digits itself) |
| Chunking of >30 s targets, cross-fade | none | not needed (app already splits per sentence) |

Frame rate 25/s (960 samples @ 24 kHz), mask id 1024, 8 codebooks x 1025 vocab confirmed.

## Results (this CPU)

Setup: 10 Croatian sentences + 1 English, auto voice, one process, sequential, bidir fp32 LLM unless noted. "agg RTF" = total synth s / total audio s over all 11 sentences (audio = trimmed speech without the 0.2 s pad). "match" = difflib word ratio of faster-whisper-small (int8, `language="hr"`) transcript vs input over the 10 Croatian sentences (a lower bound: Whisper-small is itself imperfect on Croatian and the ratio punishes diacritics/numerals "800.000" vs "osamsto tisuća"). Sentence #3 is the 150-char compound sentence (~9 s audio).

| LLM | steps | CFG | agg RTF | RTF long sentence | 1st sentence (3 s audio) | mean/sentence | match |
|---|---|---|---|---|---|---|---|
| community causal int4 (broken) | 32 | 0 | 1.72* | 1.11 | 4.55 s | - | 0.00 |
| community causal int4 (broken) | 16 | 0 | 0.98* | 0.57 | 2.18 s | - | 0.02 |
| community causal int4 (broken) | 8 | 0 | 0.43* | 0.31 | 1.20 s | - | 0.04 |
| bidir fp32 | 32 | 0 | 1.71 | 1.52 | 4.76 s | 6.13 s | 0.67 |
| bidir fp32 | 16 | 0 | 0.86 | 0.93 | 2.35 s | 3.14 s | 0.60 |
| bidir fp32 | 8 | 0 | 0.48 | 0.45 | 1.22 s | 1.69 s | 0.48 |
| bidir fp32 | 16 | 2 | 1.46 | 1.33 | 4.68 s | 5.90 s | 0.78 |
| bidir fp32 | 8 | 2 | 0.76 | 0.68 | 2.54 s | 3.13 s | 0.74 |
| bidir fp32 | 8 | 1 | 0.74 | 0.73 | 2.29 s | 2.88 s | 0.71 |
| bidir fp32 | 8 | 2 first 50% steps only | 0.58 | 0.56 | 1.82 s | 2.30 s | 0.60 |
| bidir fp32 | 12 | 2 first 50% steps only | 0.88 | 0.80 | 2.87 s | 3.49 s | 0.64 |
| bidir fp32 | 6 | 2 | 0.57 | 0.50 | 1.90 s | 2.33 s | 0.75 |
| bidir fp32 | **5** | **2** | **0.47** | **0.43** | **1.46 s** | **1.93 s** | **0.68** |
| bidir fp32 | 4 | 2 | 0.38 | 0.33 | 1.24 s | 1.53 s | 0.45 |
| bidir int4 | 16 | 0 | 0.91 | 0.86 | 2.55 s | 3.49 s | 0.55 |
| bidir int4 | 8 | 0 | 0.49 | 0.42 | 1.36 s | 1.86 s | 0.48 |
| bidir int4 | 6 | 2 | 0.55 | 0.51 | 1.91 s | - | 0.70 |

(*) measured on garbage audio (trimmed durations are not meaningful, treat as indicative only). Match values carry noise of roughly +-0.07 (single run, Gumbel sampling, seed fixed per process). English "Hello, this is an English sentence..." was transcribed perfectly with CFG at all step counts (ratio 1.00).

Takeaways:
- CFG matters more than steps: 8 steps + CFG ~0.74 vs 32 steps no CFG 0.67. 16 steps no CFG is poor (0.60). Below ~5 steps quality collapses (4 steps 0.45).
- Step cost is ~0.1 s for S~105 tokens (embeddings ~0 ms, LLM ~85-100 ms, heads 2-9 ms, numpy 3-8 ms); it scales with sequence length (text prefix ~30 tokens + 25/s of audio), so short sentences have worse RTF (prefix overhead) and the first sentence latency is 1.5-2 s at 5-6 steps.
- ORT thread count: 4 -> 223 ms, 8 -> 131, 12 -> 104, 16 (default) -> 100, 24 -> 85, 32 -> 107 ms for the LLM. int8 dynamic quantization is slower (165 ms); int4 MatMulNBits equals fp32 speed (110 ms) but with 1/7 the size and half the RAM. The CPU is already at ~0.9 TFLOPS effective; no cheap speed-up left in ORT settings.
- RAM: peak working set 3.0-3.1 GB (bidir fp32), 1.6 GB (bidir int4), 1.8 GB (community fp16+int4 build), up to 4.0 GB with cloning encoders loaded.
- Load time: 1.4 s (community build), 3.7-3.9 s (bidir fp32), 2.9 s (bidir int4); encoders +0.7 s.

### Transcripts (bidir fp32, 5 steps, CFG 2.0; `h_s5_c2`)

```
IN : Dobar dan, ovo je test čitanja na hrvatskom jeziku.
ASR: Dobar dan, ovo je testić titanja na hrvatskom jeziku.
IN : Šećer, čaj i đumbir kupila sam u dućanu.
ASR: Šećer čaj dumbir kupila sam od učanu.
IN : Danas je 15. listopada 2026. godine.
ASR: Danas je 15 listopada 2026 godina.
IN : Zagreb je glavni grad Republike Hrvatske, a nalazi se na sjeveru zemlje, između Medvednice i rijeke Save, gdje živi gotovo osamsto tisuća stanovnika.
ASR: Zagreba je glavni grad repukaj Hrvatske hrvatske analazi se na sjeveru zemlje između medvednice i rijeke Savje gdje živi gotovo 800.000 če ostanovi KK.
IN : Što radiš večeras, hoćeš li doći na kavu ili ćemo se naći sutra poslijepodne?
ASR: Što radiš većera, schoćeš li doće na kavu ili ćemo se naći sutra poslije pod?
IN : Hrvatska ima više od tisuću otoka, a najpoznatiji su Hvar, Brač, Korčula i Krk.
ASR: Hrvatska ima više od tisuću otuka, a najpoznatiji su Hvar Brač Korčula i Krk.
IN : Ljeto je bilo toplo, ali jesen je donijela kišu i hladne jutarnje vjetrove.
ASR: Bilo toplo, ali jesim je donjela hkišu i hlati otrne vjetrove.
IN : Molim vas, možete li mi reći gdje je najbliža ljekarna?
ASR: Možda vas možda smo žili mi reći da je ne najbiža ljekarno.
IN : Učenici su napisali zadaću, a nastavnica ih je pohvalila za trud i urednost.
ASR: Učenci su napisali zadaću, a nazvjastavnice ih je pohvalila za trud i urednost.
IN : Tehnologija se razvija brzo, pa je važno stalno učiti nove stvari.
ASR: Technogija se se razvija s brzo pa je važno stalno učit nove stvari.
IN : Hello, this is an English sentence for comparison.
ASR: Hello, this is an English sentence for comparison.
```

Same set at 16 steps + CFG 2 (`bd_fp32_cfg2_s16`, best quality, RTF 1.46): near-perfect on sentences 0, 2, 3, 4, 6, 8, 9 (e.g. "Dobar dan, ovo je test čitanja na hrvatskom jeziku." exact; "Zagreb ... gotovo 800.000 stanovnika." with one word error), weak spots: "Šećer, čaj ... đumbir ... dućanu" (đ/ć words), "Molim vas" (-> "Moni vas"). Full transcripts for every config: `.claude/temp/vtf02/out/*.asr.txt` (and `*.asr.json`).

Verdict on intelligibility: a Croatian listener should understand every sentence at 5-8 steps + CFG; some words are slightly mispronounced (d/dj, č/ć, "Molim" at low steps). Whisper-small agreement 0.68-0.78. **A human listen is still recommended** before committing; sample WAVs below.

### Cloning (reference = an auto-voice clip reused as reference)

No Croatian human reference clip was available; I used `bd_fp32_cfg2_s16_cfg2_03.wav` (8.4 s of generated speech, text = sentence #3) via the fp32 Higgs encoders (acoustic + semantic + quantizer; 211 frames encoded in 0.64 s, encoder load 0.66 s). The fp16 semantic encoder does not load on ORT 1.31 CPU, so cloning requires the 650 MB fp32 encoder set (or a re-export).
- Quality: word match 0.85 at 8 steps CFG 2 (better than auto-voice 0.74).
- Cost: the reference codes enlarge the sequence (8.4 s = 211 frames), so RTF at 8 steps CFG 2 was 1.4-2.3 (4.6-5.8 s per sentence) vs 0.76 without a reference. Use short references (3-5 s) if cloning is offered, and it will not meet RTF 0.5 on this CPU. Recommend cloning stays optional / out of Phase 1 default.

## WAV samples (human listen)

Folder `C:\projects\Personal\vibe-tts-flow\.claude\temp\vtf02\out\` (file pattern `<tag>_s<steps>_cfg<scale>_<idx>.wav`, idx 00-09 Croatian in the order above, 10 = English):
- Recommended default (5 steps CFG 2): `h_s5_c2_s5_cfg2_00.wav` ... `_10.wav`
- 6 steps CFG 2: `h_s6_c2_s6_cfg2_NN.wav`; int4 LLM: `i4_s6_c2_s6_cfg2_NN.wav`
- Best quality (16 steps CFG 2): `bd_fp32_cfg2_s16_cfg2_NN.wav`; 8 steps: `bd_fp32_cfg2_s8_cfg2_NN.wav`
- No CFG for contrast: `bd_fp32_cfg0_s{32,16,8}_cfg0_NN.wav`
- Broken causal community build (noise): `fp16_s{32,16,8}_cfg0_NN.wav`
- Cloned: `clone_s8_cfg2_{00,01,02,04,05}.wav` (reference `bd_fp32_cfg2_s16_cfg2_03.wav`)

## Recommendation against the plan's threshold (intelligible Croatian at RTF <= 0.5)

**CONDITIONAL GO.** Borderline-pass on this CPU, not as the plan assumed:

1. Intelligible Croatian is achievable (`hr`, auto voice) with ASR agreement 0.68-0.78 and RTF 0.47-0.57 at 5-6 steps + CFG 2 on a 16-core Ryzen AI Max+ 395; 5 steps + CFG 2 passes the 0.5 threshold with margin of 0.03 only. On a typical 4-8 core laptop expect RTF ~1-2, i.e. slower than real time per sentence; the existing one-chunk-ahead streaming would then stall between sentences. First-sentence latency 1.5-2 s here (3-6 s on a weaker CPU).
2. **Plan changes required (blocking for Phase 1):** (a) the community `llm_decoder.onnx` cannot be used; Phase 1/2 must ship our own bidirectional LLM ONNX (export recipe in `spike-scripts/torch_ref.py` + `quant.py`; torch only needed once at build/export time, runtime stays numpy + onnxruntime + tokenizers + soxr). Needs a hosting decision (HF repo of our own, or GitHub release asset; license check: upstream model Apache-2.0 per README). (b) CFG must be implemented (plan said "only if quality needs it": it does). (c) language token, style tokens, duration estimator, and the upstream scoring/schedule must be ported exactly as in `spike.py` (the repo's `inference.py` loop is not equivalent). (d) fp16 semantic encoder does not load on CPU ORT 1.31, so cloning needs the fp32 encoders (+650 MB) and is slow (reference tokens inflate sequence); make cloning optional/later.
3. If the user will not accept an own-export + hosting burden or RTF > 0.5 on weaker hardware, fall back to the plan's Piper `sr_RS`/espeak-hr option.

## Recommended defaults for Phase 1

- LLM: our bidir **int4** export (235 MB, same speed as fp32, ~1.6 GB RAM); audio parts fp16 from the pinned community commit; Higgs decoder fp16.
- `omnivoice_steps` default **6** (RTF ~0.5-0.57, match 0.70-0.75); allow 4-16 in settings (5 = fastest acceptable, 16 = best quality at RTF ~1.5). Do not offer 32.
- CFG **on**, guidance_scale 2.0 (upstream default; the only setting that makes few-step decoding intelligible). Optional "fast" mode: CFG only on first half of steps (8 steps: RTF 0.58, match 0.60) - not better than 6 steps full CFG, skip.
- Other upstream params: t_shift 0.1, layer_penalty_factor 5, position_temperature 5, class_temperature 0 (greedy), all unchanged.
- Duration: use the upstream `RuleDurationEstimator` (weights table) with the "Nice to meet you."/25-frame default reference (no ref clip) rather than a flat chars/sec; for reference sanity: it gives ~0.04-0.05 s per Latin char incl. spaces/punct, e.g. 51 chars -> 3.1 s, 150 chars -> 9.2 s (~16 chars/s); divide by `speed`. Actual trimmed speech was 0.7-1.0x of the estimate (silence is trimmed afterwards).
- Language: pass `hr` for Croatian (also the model accepts sr, bs, sl); English text via `en` was perfect, so the auto-language router can send English to either engine.
- Threads: leave ORT default (16 physical cores gave 100 ms; 24 gave 85 ms); consider `intra_op_num_threads = physical cores`.
- Sentence-level synth only (<= ~10 s of audio per call); cost per step grows with sequence length.
- Post-process: edge silence trim, peak normalise to 0.5, 0.1 s fade/pad, as in `spike.py` `Engine.post`.
