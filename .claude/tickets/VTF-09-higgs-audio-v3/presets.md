# VTF-09 Higgs preset voices: ASR check

Generated 2026-10-22 with `scripts/make_higgs_presets.py` (make, check, asr, prune, table).

- Clips: a neutral ~9-11 s sentence per language ("The morning train arrived a little earlier than expected ...") read by the app's SupertonicEngine with the same voice name (F1, F2, M1, M2), resampled to 24 kHz, encoded by Higgs `audio_encoder`, stored as int16 `[8, T]` codes + transcript in `vibetts/voices/higgs/<VOICE>/<lang>.*` (about 0.9 MB for all).
- Check: a Higgs read (cpu_int4, 8 threads, temperature 0.8, speed 1.0) of a different short sentence ("The library opens at nine, and the quiet reading room is on the second floor.") in the same language with the clip as the voice, transcribed with faster-whisper `medium` (int8, language forced); ratio = difflib word-match on lowercase, punctuation-stripped words.
- Rule: a clip below 0.6 gets one more take (seed + 1); if it is still below 0.6 the clip is dropped. "(2 takes)" = the second take was made; the ratio shown is the best of the takes.
- Japanese is not in the table: Higgs does not read it (no clip made). 30 languages x 4 voices = 120 clips, all checked, 1 dropped.
- Whisper itself is noisy for some languages (ko, hi, et, lt: no spaces/diacritics/variants), so 0.6 to 0.8 there is mostly ASR noise, not a bad clip. Whisper medium word ratio is not a listening test: listen before trusting the quieter voices.

| ratio | state |
|---|---|
| >= 0.6 | kept |
| < 0.6 after a second take | dropped (voice falls back through CLIP_FALLBACK to the English clip) |

| lang | F1 | F2 | M1 | M2 |
|---|---|---|---|---|
| hr | 0.93 kept | 0.86 kept | 0.86 kept | 0.79 kept (2 takes) |
| en | 1.00 kept | 1.00 kept | 0.93 kept | 0.93 kept |
| de | 0.93 kept | 0.87 kept | 0.93 kept | 0.93 kept |
| sl | 0.75 kept | 0.83 kept | 0.83 kept | 0.72 kept |
| sk | 0.72 kept | 0.78 kept | 0.70 kept | 0.78 kept |
| cs | 1.00 kept | 0.85 kept | 1.00 kept | 0.92 kept |
| pl | 1.00 kept | 0.92 kept | 0.92 kept | 1.00 kept |
| es | 0.94 kept | 0.94 kept | 0.94 kept | 0.94 kept |
| it | 0.94 kept | 0.94 kept | 0.94 kept | 0.94 kept |
| fr | 0.85 kept | 0.91 kept | 0.85 kept | 0.88 kept |
| ko | 0.67 kept | 0.67 kept (2 takes) | 0.67 kept | 0.67 kept |
| ar | 0.91 kept (2 takes) | 0.82 kept | 1.00 kept | 0.91 kept |
| bg | 0.83 kept | 0.72 kept | 0.92 kept | 0.83 kept |
| da | 0.88 kept | 0.92 kept | 0.83 kept | 0.92 kept |
| el | 0.93 kept | 0.86 kept | 1.00 kept | 0.93 kept |
| et | 0.76 kept | 0.38 dropped (2 takes) | 0.60 kept | 0.67 kept |
| fi | 1.00 kept | 1.00 kept | 0.89 kept | 0.67 kept |
| hi | 0.77 kept | 0.62 kept | 0.62 kept | 0.77 kept |
| hu | 0.92 kept | 0.78 kept | 0.75 kept | 0.92 kept |
| id | 0.85 kept | 0.85 kept | 0.85 kept | 0.85 kept |
| lt | 0.80 kept | 0.80 kept | 0.70 kept | 0.70 kept |
| lv | 0.89 kept | 0.67 kept | 1.00 kept (2 takes) | 0.89 kept |
| nl | 0.94 kept | 0.91 kept | 0.94 kept | 1.00 kept |
| pt | 0.94 kept | 0.94 kept | 0.94 kept | 0.94 kept |
| ro | 0.88 kept (2 takes) | 1.00 kept | 1.00 kept | 0.81 kept |
| ru | 0.92 kept | 0.85 kept | 0.92 kept | 0.89 kept |
| sv | 0.92 kept | 0.92 kept | 0.92 kept | 1.00 kept |
| tr | 0.90 kept | 0.70 kept | 1.00 kept | 1.00 kept |
| uk | 0.96 kept | 0.83 kept | 0.86 kept | 0.78 kept |
| vi | 0.88 kept | 0.88 kept | 0.88 kept | 0.88 kept |


Dropped: F2/et (0.32, then 0.38). Et for F2 therefore reads with the F2 en clip (CLIP_FALLBACK has no et entry).
