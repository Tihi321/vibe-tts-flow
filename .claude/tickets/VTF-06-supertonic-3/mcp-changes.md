# VibeTTS MCP changes in VTF-06 (for MCP clients such as VN Studio)

Branch `VTF-06_supertonic-3` (commits `c6d75f3` Remove OmniVoice, `64bc427` Supertonic 3). Server, transport, port, headless mode (`python -m vibetts --mcp-only [--port N]`), `stop` and the one-at-a-time export queue are unchanged.

## Summary

| Area | Before (VTF-05, OmniVoice) | Now (VTF-06, Kokoro + Supertonic 3) |
|---|---|---|
| Engines | Kokoro (English) + OmniVoice (optional, Croatian and more) | Kokoro (English only) + Supertonic 3 (optional download, 31 languages) |
| Voice ids | `af_heart` ..., `omni:<preset>` (e.g. `omni:petra`, `omni:default`) | `af_heart` ... (English Kokoro voices only), `supertonic:F1` .. `supertonic:F5`, `supertonic:M1` .. `supertonic:M5` |
| Custom voices | `create_voice` designs a voice from a description, `delete_voice` removes it | **Removed.** 10 fixed Supertonic voices; there is no voice design or cloning |
| `lang` | `en` / `hr` | any of the 31 codes: `en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi` |
| Audio | 24 kHz mono | **44.1 kHz** mono (WAV 16-bit; MP3 is MPEG-1, bitrate from `[export] mp3_bitrate`, 32..320) |

## Tools

### Removed

- `create_voice` (name, description, seed, lang, overwrite). Calling it now returns a JSON-RPC "unknown tool" error.
- `delete_voice` (name).

### `list_voices` (new response shape)

```json
{
  "voices": [
    {"id": "af_heart", "label": "Heart (American English, female)", "engine": "kokoro", "languages": ["en"]},
    {"id": "supertonic:F1", "label": "F1 (Supertonic, female)", "engine": "supertonic", "languages": ["en", "ko", "...", "vi"]}
  ],
  "languages": [{"code": "en", "name": "English"}, {"code": "hr", "name": "Croatian"}, "..."],
  "english": {"engine": "kokoro", "voice": "af_heart"},
  "other": {"language": "hr", "voice": "supertonic:F1"},
  "language_mode": "auto",
  "supertonic_installed": true
}
```

- Gone: top-level `engine`, `default_voice`, `omnivoice_installed`; per-voice `kind`, `description`, `lang`.
- New: per-voice `engine` + `languages`; top-level `languages`, `english`, `other`, `language_mode`, `supertonic_installed`.
- Supertonic voices are listed only when `supertonic_installed` is true (the user downloads it in VibeTTS Settings > Supertonic, about 400 MB). `F*` voices are female, `M*` male.

### `synthesize_to_file`, `speak`, `synthesize_batch` items

Parameters are the same as before (`text`/`path`, `output`, `format`, `voice`, `lang`, `speed`, `overwrite`; `wait`, `timeout_seconds` on `speak`; `items` + `overwrite` on the batch), with these rule changes:

- `voice`: a Kokoro id or a `supertonic:` id. `omni:*` ids are unknown now: `Unknown voice '<id>'. Call list_voices to get the valid ids.`
- `lang`: one of the 31 codes (anything else is a tool error).
- A Kokoro voice with a non-English `lang` is an error: `The Kokoro voice 'af_heart' reads English only; use a supertonic: voice for language 'hr'.` (batch: that item fails, the rest run).
- A `supertonic:` voice before Supertonic is downloaded: `Supertonic is not installed: download it in Settings > Supertonic`.
- Both omitted: routed like the hotkey (English-looking text → the user's English profile, anything else → the other-language profile).
- `lang` without `voice`: `en` → the English profile voice; any other code → the user's other-language Supertonic voice, in that language.
- `voice` (Supertonic) without `lang`: the language is guessed per text (English or the user's other language). **Always pass `lang` for predictable output.**
- Results: `synthesize_to_file` still returns `path, format, duration_seconds, chunks, bytes, voice, lang`; batch rows still `{output, duration_s, ok, error}`. Durations are in seconds as before, now of 44.1 kHz audio.

## VN Studio migration (`C:\projects\Personal\visual-novels-orchestrator`)

What it uses today and what has to change:

| Where | Today | Change |
|---|---|---|
| `src/engine/voice/vibetts.ts` | `createVoice`, `deleteVoice`; `VibeVoice.kind/description/lang`; `VibeVoiceList.engine/default_voice/omnivoice_installed`; `CreatedVoice` | Drop `createVoice` / `deleteVoice` and `CreatedVoice`. `VibeVoice` → `{id, label, engine, languages}`; `VibeVoiceList` → `{voices, languages, english, other, language_mode, supertonic_installed}`. Update the timeout comments (no more create_voice). |
| `src/engine/voice/jobs.ts` (plan voices ~L65-85, `designVoice` ~L250-263) | creates one `omni:<name>` voice per character from a description + seed via `create_voice`; strips/checks the `omni:` prefix | Assign each character one of the 10 Supertonic voices instead (e.g. by gender: `F1..F5` / `M1..M5`, spread so speakers differ; narrator gets its own). No creation step, nothing to delete. If `supertonic_installed` is false, fail the step with a hint to download Supertonic in VibeTTS. |
| `generateVibeClips` / batch items (~L150-166) | `{text, output, voice, lang, speed}` | Unchanged shape. Keep sending `lang` (`en` / `hr`). Use the **same `supertonic:` voice for both editions** so a character sounds the same in English and Croatian (a Kokoro voice cannot read `hr`). Clip hashes include the voice id, so existing clips regenerate once (new ids, 44.1 kHz). |
| `previewVoice` (~L237-243) | `synthesizeToFile` with the character's voice | Unchanged; works with `supertonic:` ids. |
| `src/shared/voice-vocab.ts`, `src/engine/voice/vocab.ts`, `tests/voice-vocab.test.ts` | OmniVoice description vocabulary (gender / age / pitch / style) | Not needed for VibeTTS any more. Keep only what still drives the voice choice (gender), or remove. |
| `seed/prompts/voice-director/plan_voices.md`, `src/engine/vn/agents/voice-director.ts` | asks the model for a voice description + seed per character | Ask for gender (and optionally a preferred voice `F1..M5`) per character instead. |
| `src/renderer/lib/voice.ts` (~L74), `src/renderer/ui/book/VoiceTab.tsx` (~L69) | filters voices by `id.startsWith('omni:')` | Filter by `engine === 'supertonic'` (or `id.startsWith('supertonic:')`); show the `label`. |
| `src/shared/vn.ts` (~L195) | doc comment: voice id "such as `omni:petra`" | "such as `supertonic:F2`". Existing projects store `omni:*` ids: map them to a Supertonic voice on load or ask the user to re-plan voices. |
| `seed/config/voice.md`, docs, tests (`voice-vibetts-*`, `ui-voice-vibetts`, `vn-*` fixtures) | OmniVoice wording, `omni:` ids, `create_voice` fakes | Update to the new ids and response shapes; drop create/delete fakes. |
| Audio pipeline (Ren'Py emit, any resampling) | clips were 24 kHz | Clips are now 44.1 kHz WAV/MP3; nothing to do unless something assumed 24 kHz. |

Requirement for users: VibeTTS needs **Supertonic downloaded** (Settings > Supertonic) for Croatian and for the per-character voices; Kokoro alone only covers English.
