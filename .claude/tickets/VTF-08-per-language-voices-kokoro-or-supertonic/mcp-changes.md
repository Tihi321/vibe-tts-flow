# VibeTTS MCP changes in VTF-08 (for MCP clients such as VN Studio)

Branch `VTF-08_per-language-voices`. Tools, parameters, transport, port, headless mode and the export queue are unchanged. Only `list_voices` and the voice/language rules change.

## What changed

- The English / Other-language profiles are gone. Each language has its own voice (`[speech.language_voices]`, code -> voice id; empty = Off).
- Kokoro now reads six languages: `en` (`a*`/`b*` voices), `es` (`e*`), `fr` (`f*`), `hi` (`h*`), `it` (`i*`), `pt` (`p*`, Brazilian). Supertonic still reads all 31.
- Language detection (py3langid) runs on the whole text among the enabled languages; text under 4 words falls back to English if enabled.

## `list_voices`

Before (VTF-06):

```json
{"voices": [...], "languages": [...],
 "english": {"engine": "kokoro", "voice": "af_heart"},
 "other": {"language": "hr", "voice": "supertonic:F1"},
 "language_mode": "auto", "supertonic_installed": true}
```

After:

```json
{
  "voices": [
    {"id": "af_heart", "label": "...", "engine": "kokoro", "languages": ["en"]},
    {"id": "ef_dora", "label": "...", "engine": "kokoro", "languages": ["es"]},
    {"id": "supertonic:F1", "label": "...", "engine": "supertonic", "languages": ["en", "...", "vi"]}
  ],
  "languages": [{"code": "en", "name": "English"}, "... (31, unchanged)"],
  "language_voices": {"en": "af_heart", "hr": "supertonic:F1", "de": ""},
  "read_language": "auto",
  "supertonic_installed": true
}
```

- Dropped: `english`, `other`, `language_mode`.
- New: `language_voices` (voice id per language; `""` = that language is Off), `read_language` (`"auto"` or a code).
- Per-voice `languages` is now meaningful for Kokoro (one code each); use it to filter a voice/language picker.

## Voice and `lang` rules (`synthesize_to_file`, `speak`, `synthesize_batch`)

- No `voice`, no `lang`: language detected from the text, read with that language's configured voice (or `read_language` when fixed).
- `lang` only: that language's configured voice (if Off or invalid, its default: `en` = `af_heart`, others `supertonic:F1`).
- `supertonic:*` voice: reads any language; without `lang` the language is detected.
- Kokoro voice without `lang`: reads its own language.
- Kokoro voice with a different `lang` is an error (batch: that item fails, the rest run), e.g. `The Kokoro voice 'af_heart' reads English only; use a supertonic: voice for language 'hr'`.
- Supertonic not installed: a language falls back to its Kokoro default voice, or Kokoro English. Always pass `voice` and `lang` for predictable output.

## Migration

- Old config keys `voice`, `english_engine`, `english_supertonic_voice`, `other_language`, `other_voice`, `language_mode` are converted on load and no longer written; no client action needed.
- Clients reading `english` / `other` / `language_mode` from `list_voices` must switch to `language_voices` / `read_language`.
- Clients that sent non-English `lang` only with Supertonic voices keep working; Kokoro voices for es/fr/hi/it/pt are new options.
