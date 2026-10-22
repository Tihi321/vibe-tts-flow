# VibeTTS MCP changes in VTF-09 (for MCP clients such as VN Studio)

Branch `VTF-09_higgs-audio-v3`. Tools, parameters, transport, port, headless mode and the export queue are unchanged. `list_voices` gains Higgs voices, more languages and `higgs_installed`; the `lang` enum and the voice/language rules grow.

## What changed

- A third engine, Higgs Audio v3, reads about 100 languages (`HIGGS_LANGUAGES`, no `ja`). Its voices are `higgs:F1`, `higgs:F2`, `higgs:M1`, `higgs:M2` plus the user's own clips (`higgs:<name>`). It needs its own download (Settings > Higgs, 3.85 GB).
- On a CPU Higgs is slower than real time (measured RTF 1.4 to 1.8 on one machine): `speak` pauses between chunks and `synthesize_to_file` takes longer than the audio lasts. Allow for that in timeouts.
- `lang` is now one of 101 codes (`ALL_LANGUAGES`: the 31 Supertonic codes plus 70 that only Higgs reads, e.g. `bs`, `sr`, `mk`, `sq`, `ca`). The `lang` enum in the tool schemas of `synthesize_to_file`, `speak` and `synthesize_batch` lists all 101.
- Languages only Higgs reads are Off by default in `language_voices`; the user turns them on in Settings > Voice. Language detection only picks among enabled languages that py3langid knows; the others (ast, ceb, ckb, kam, kea, luo, mhr, mi, ny, sd, umb) are reached only with an explicit `lang`.

## `list_voices`

Before (VTF-08):

```json
{"voices": [{"id": "supertonic:F1", "label": "...", "engine": "supertonic", "languages": ["en", "...", "vi"]}],
 "languages": [{"code": "en", "name": "English"}, "... (31)"],
 "language_voices": {"en": "af_heart", "hr": "supertonic:F1"},
 "read_language": "auto", "supertonic_installed": true}
```

After:

```json
{
  "voices": [
    {"id": "af_heart", "label": "...", "engine": "kokoro", "languages": ["en"]},
    {"id": "supertonic:F1", "label": "...", "engine": "supertonic", "languages": ["en", "...", "vi"]},
    {"id": "higgs:F1", "label": "F1 (Higgs, female)", "engine": "higgs", "languages": ["af", "ar", "...", "zu"]},
    {"id": "higgs:Anna", "label": "Anna (Higgs, custom)", "engine": "higgs", "languages": ["af", "...", "zu"]}
  ],
  "languages": [{"code": "en", "name": "English"}, "... (101: the 31, then Higgs-only codes sorted by name)"],
  "language_voices": {"en": "af_heart", "hr": "supertonic:F1", "bs": ""},
  "read_language": "auto",
  "supertonic_installed": true,
  "higgs_installed": false
}
```

- New: `higgs_installed`; Higgs entries in `voices` only when Higgs is installed (like Supertonic); `engine` can be `"higgs"`.
- Changed: `languages` has 101 entries (was 31) and `language_voices` has 101 keys (Higgs-only codes `""` unless the user turned them on).
- A Higgs voice's `languages` lists every Higgs code, but a custom voice reads best in languages it has its own clip for (not exposed).

## Voice and `lang` rules (`synthesize_to_file`, `speak`, `synthesize_batch`)

- `higgs:*` voice: reads any code except `ja`; without `lang` the language is detected. Other `lang` (`ja`) is an error: `Higgs does not read Japanese ('ja')`.
- `supertonic:*` voice with a Higgs-only `lang` (e.g. `bs`) is an error: `Supertonic does not read Bosnian ('bs'); use a higgs: voice for it`.
- Kokoro voice with another `lang`: the error now ends `use a supertonic: voice for language 'hr'` for the 31 codes and `use a higgs: voice for language 'bs'` for the rest.
- Unknown `higgs:<name>` (not a preset, no clip folder): `Unknown voice 'higgs:Nobody'. Call list_voices to get the valid ids.`
- Higgs not installed and a `higgs:` voice requested: `Higgs is not installed: download it in Settings > Higgs`.
- No `voice`, `lang` given: that language's configured voice; for a Higgs-only language that is Off, `higgs:F1`.
- A configured Higgs voice that is not available (Higgs missing, or the custom voice was deleted): the language falls back to its Supertonic voice if Supertonic is installed and reads it, else the Kokoro voice of that language, else Kokoro English. Pass `voice` and `lang` for predictable output.

## Migration

- No client action is needed unless a client hard-codes 31 languages or `engine in ("kokoro", "supertonic")`: accept `"higgs"` and the extra codes.
- Clients that must not use the slower engine can filter `voices` by `engine != "higgs"`.
