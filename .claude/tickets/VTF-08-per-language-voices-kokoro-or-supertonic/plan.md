# VTF-08: Per-language voices (Kokoro or Supertonic) with full language detection

Ticket folder: `.claude/tickets/VTF-08-per-language-voices-kokoro-or-supertonic/`. Branch: `VTF-08_per-language-voices`, cut from `origin/master` (6beb26b).

## Implementation checklist

- [x] A. Core: config model + migration, `langdetect.py` + py3langid dep + PyInstaller spec, routing in `app.py`, Kokoro languages in `tts.py`, bridges/MCP text, tray hooks + menus, `config.example.toml`, and their tests (config, langdetect, app, tray, mcp, reading, reader_server, export).
  - Done. `pytest tests --ignore=tests/test_settings_ui.py`: 519 passed, 1 skipped, 3 failed. The 3 failures are `test_app.py::test_settings_hooks_*`, which import `settings_ui` (still imports the removed `ENGINES`); item B fixes them. `test_settings_ui.py` fails at import for the same reason.
  - Deviations: `tests/test_tts.py` updated too (Kokoro voice widening). `detect_language` also treats unspaced text with 15 or more letters (Japanese) as long enough to detect. Added `DEFAULT_KOKORO_VOICES` (Kokoro voice used when Supertonic is missing), `word_count` in textprep, and `TrayHooks.languages`. `config.example.toml` mirrors the defaults, with other examples commented out.
- [x] B. Settings UI (`settings_ui.py`) + `tests/test_settings_ui.py`.
  - Done. One Voices section (Language/Engine/Voice/Read combos, last-language-Off guard, Read resets to Auto). `SettingsHooks.play_test` is now `(cfg, text, lang)` and `TTSApp._play_test(cfg, text, lang=None)` routes with it. `pytest -q tests`: 607 passed, 1 skipped.
- [x] C. Web reader `vibetts/web/app.js`, `extension/popup.js`, README.
  - Done. Language selects list only the chosen voice's languages (per-voice `languages`); stored lang resets to Auto when unsupported. README rewritten; `mcp-changes.md` added. `node --check` passes on both JS files; not exercised in a browser.
- [~] D. Full test suite green (607 passed, 1 skipped). Detection spot check: en/hr/sl/sk/cs/de/es/pt/it all correct. Settings Voice tab rendered and screenshotted (Kokoro for Spanish, German Off). Not run: live app hotkey reads, old-config migration on a real `config.toml`, MCP calls against the running app, browser/extension, PyInstaller build.

## Context

Today the voice settings are split in two:
- An **English** profile, with Kokoro or Supertonic.
- One **Other language** profile, always Supertonic, with one fixed language such as Croatian.

Detection only answers "English or not" (`looks_english`, `vibetts/textprep.py:203`). Every non-English text is read as the single configured `other_language`, so German text is read with the Croatian setting.

The goal is one voice setting per language. The app should detect which of the 31 Supertonic languages a text is in and read it with that language's voice. Each language can use Kokoro or Supertonic, where Kokoro supports that language. A language can be set to **Off** so it is never detected or offered.

Decisions from the user:
- Detection runs once per read, on the whole selection, as today. There is no switching per paragraph.
- Every language is detected by default. Each language has an "Off" voice choice that removes it from detection, so languages can be turned on and off.

## Data model (`vibetts/config.py`)

### Kokoro languages

Kokoro languages come from the voice prefixes in `voices-v1.0.bin`:

| Prefix | Language | espeak code |
|---|---|---|
| `a` | English | `en-us` |
| `b` | English | `en-gb` |
| `e` | Spanish | `es` |
| `f` | French | `fr-fr` |
| `h` | Hindi | `hi` |
| `i` | Italian | `it` |
| `p` | Portuguese | `pt-br` |

- Japanese (`j`) is left out because espeak cannot read kanji. Chinese (`z`) is left out because Supertonic does not have it.
- Add `KOKORO_LANGUAGES = {"a":"en","b":"en","e":"es","f":"fr","h":"hi","i":"it","p":"pt"}`.
- Add `kokoro_voice_lang(voice) -> str | None` next to `is_supertonic_voice`.

### Replaced settings

Six fields are replaced by two new ones:
- **Replaced:** `voice`, `english_engine`, `english_supertonic_voice`, `other_language`, `other_voice`, `language_mode`.
- **New field `language_voices: dict[str, str]`:**
  - Maps a language code to a voice id: a Kokoro id such as `af_heart` or `ef_dora`, `supertonic:F1`, or `""` for Off.
  - Defaults: `en = "af_heart"`; the other 30 codes `"supertonic:F1"`.
  - Missing codes are filled with these defaults.
  - In TOML it is the table `[speech.language_voices]`.
- **New field `read_language: str = "auto"`:** either `"auto"` or one language code (meaning "Always <language>").

### Loading and validating

- Extend `_coerce` so it can read a `dict[str, str]`. Today it only handles str, int, float, bool and list.
- `_validate_speech`:
  - A voice for a language must be one of: `""`, a Supertonic voice, or a Kokoro voice whose language matches that row. Anything else falls back to the default and logs a warning.
  - `read_language` must be `auto` or an enabled code.
  - If every language is Off, turn English back on.

### Migrating old configs

In `config_from_dict`, before `_build`, map the old keys when `language_voices` is missing:
- `en`: the old English voice, which is `voice`, or `supertonic:`+`english_supertonic_voice` when `english_engine` was Supertonic.
- `other_language`: `supertonic:`+`other_voice`.
- `language_mode`:
  - `english` becomes `read_language = "en"`.
  - `other` becomes `read_language = <other_language>`.
  - `auto` stays `auto`.

`save_config` then writes only the new shape.

Also update `config.example.toml` (lines 11-26) to the new keys.

## Detection (`vibetts/langdetect.py`, new)

- Add the dependency `py3langid` to `pyproject.toml`. It is small, needs only numpy (already a dependency), and is deterministic.
- Add `detect_language(text, candidates: Sequence[str], fallback: str) -> str`:
  - Call `langid.set_languages(candidates)` on a cached `LanguageIdentifier`.
  - Return `fallback` when the text has fewer than `EN_SHORT_WORDS` words or there is no candidate.
  - The candidates are the enabled codes (voice not `""`).
  - The fallback is `en` if enabled, otherwise the first enabled code.
- Keep `looks_english` as a cheap pre-check: if `en` is enabled and the text has fewer than 4 words, return `en`. This keeps today's behaviour for short selections.
- PyInstaller: add the py3langid model file with `collect_data_files("py3langid")` in `packaging/VibeTTS.spec`.

## Routing (`vibetts/app.py:102-200`)

**Rewrite the route helpers.** Keep the `Route` NamedTuple.
- `voice_for_lang(s, lang) -> str` returns `s.language_voices[lang]`, or the default if it is empty or invalid.
- `profile_lang(s, text)` becomes `s.read_language` when that is not `auto`, and `detect_language(text, enabled_langs(s), fallback)` otherwise.
- `route_for(s, text, lang=None, supertonic_ok=None)`:
  1. Resolve `lang`, then the voice.
  2. If the voice is Supertonic and Supertonic is missing:
     - Use the first Kokoro voice for that language, if Kokoro supports it.
     - Otherwise fall back to Kokoro English as today (`fell_back=True`).
  3. Return `Route(engine, voice, lang)`.

**Explicit requests.** `route_request` (157) now accepts a Kokoro voice with any language Kokoro supports. It raises `ValueError` only when the voice and the language do not match, for example `af_heart` with `lang="hr"`.

**Preloading.** `needed_voices`, `primary_voice` and `profile_routes`:
- With `auto`: the distinct voices of all enabled languages. In practice that is one Kokoro voice plus the Supertonic family, because all Supertonic voices load as one family.
- With a fixed language: only that language's voice.

`missing_voices` stays the same.

**Kokoro synthesis.** In `KokoroEngine.synth` (`vibetts/tts.py:81`), take the espeak language from the voice prefix:
- Extend `VOICE_LANGS` with `e:"es"`, `f:"fr-fr"`, `h:"hi"`, `i:"it"`, `p:"pt-br"`.
- Extend `LANG_LABELS` to match.
- Widen `kokoro_voice_supported` and `voices()` to these prefixes.

## Settings UI (`vibetts/settings_ui.py:712-799`)

**Voice section.** Replace the "English", "Other language" and "Language" sections with one **Voices** section:
- **Language** combo:
  - Lists all 31 languages by display name, sorted with English first.
  - Each label ends with " (off)" when that language is disabled.
- **Engine** combo:
  - Choices are `Supertonic`, `Kokoro` and `Off`.
  - `Kokoro` appears only when `KOKORO_LANGUAGES` covers the chosen language.
- **Voice** combo:
  - Kokoro: the Kokoro voices whose prefix maps to that language.
  - Supertonic: F1..M5.
  - Off: the combo is disabled.
- **Read** combo:
  - Choices are `Auto (detect language)` and `Always <name>` for each enabled language.
  - Hint: "Auto detects the language of each read and uses that language's voice. Set a language to Off to stop detecting it."
- The red Supertonic hint (`_refresh_voice_hint`) stays.

**Form state.** The form keeps `language_voices` as a dict in the window state, `self._lang_voices`:
- Changing Engine or Voice writes to the selected language.
- `form_from_config`, `config_from_form`, `_collect`, `_set_form_var` and `_follow_config` handle the dict and `read_language`.
- `FOLLOW_KEYS` swaps the old keys for `language_voices` and `read_language`.
- Remove `LANG_AUTO`, `LANG_ENGLISH`, `OTHER_LANGUAGES` and `language_mode_label` / `_from_label`. Replace them with `read_label` / `read_from_label`.

**Other parts of the window.**
- `_load_voices` (1549): keep all supported Kokoro voices, not just English ones.
- Export tab (824-916): its language list is Auto plus all 31 languages. Its voice list includes the new Kokoro voices. `ExportRequest` is unchanged.
- `_play_test` reads the test sentence with the selected language's voice and `lang`, so each row can be previewed.

## Tray (`vibetts/tray.py:107-203`, hooks at `app.py:438-469`, setters at `app.py:1122-1149`)

- **Language** menu: Auto, plus "Always <name>" for each enabled language. It sets `read_language`.
- **Voices** menu: one submenu for each enabled language. Each submenu lists Off, the Supertonic voices, and the Kokoro voices for that language. Selecting one sets `language_voices[lang]`.
- These replace the separate English voice menu and the other-language voice menu.
- Hooks: `current_voice(lang)`, `set_voice(lang, voice)` and `set_read_language`. They replace `current_voice` / `current_other_voice` / `set_other_voice` / `set_language_mode` / `other_language_name`.

## MCP, reader API, web reader, extension

- **`AgentBridge.list_voices`** (`app.py:1201`) returns:
  - `language_voices` (code → voice, `""` = off).
  - `read_language`.
  - `languages`.
  - Per-voice `languages`: a Kokoro voice lists its language code, a Supertonic voice lists all 31.
  - It drops `english`, `other` and `language_mode`.
- **`_check_voice`, `route_request` and the MCP schemas** (`mcp_server.py:20-28, 415-416`): update the text to say Kokoro voices read en/es/fr/hi/it/pt.
- **Reader** (`ReaderBridge.list_voices`, `app.py:1372`, the same data): the behaviour of `POST /api/speak` is unchanged. Auto means detect.
- **Web reader** (`vibetts/web/app.js:742-776`): `engineOf` / `langRelevant` must stop assuming Kokoro reads English only. Use the per-voice `languages` from `/api/voices` to filter the language select.
- **Extension** (`extension/popup.js`): same filtering if it applies the English-only rule. Check `popup.js:62-84`.

## Docs

- `README.md`: rewrite "## Engines, profiles and language detection" (184-209), the tray section (146-147) and the MCP `lang`/`voice` notes (79-87).
- `.claude/tickets/VTF-08-per-language-voices/`: add `plan.md` and `changelog.md`, plus `mcp-changes.md` for the `list_voices` shape change.

## Tests (`python -m pytest -q tests`)

- **`tests/test_config.py`:**
  - Defaults.
  - Round trip of `[speech.language_voices]`.
  - Migration from the old keys: all three modes, and Supertonic English.
  - Invalid voices per language, for example `af_heart` set for `hr`.
  - All languages Off.
- **`tests/test_langdetect.py`** (new):
  - The en/hr/de/fr/ru/sl/cs samples are detected correctly.
  - A restricted candidate set is honoured.
  - Short text uses the fallback.
  - Reuse the `ENGLISH` / `NOT_ENGLISH` fixtures from `tests/test_textprep.py`.
- **`tests/test_app.py`:**
  - Replace `test_route_matrix` / `test_route_other_language_*` (991-1059) with routing tests:
    - Per language: Kokoro es, Supertonic de, Off languages never chosen.
    - `read_language` forced.
    - Fallback when Supertonic is missing (Kokoro for es, Kokoro English for hr).
    - `route_request` with matching and mismatched Kokoro voice and language.
  - Update the tray setter and menu tests (215-330), bridge `list_voices` (505, 1211-1278) and the read/say routing tests (1323-1405).
- **`tests/test_settings_ui.py`:**
  - Replace the voice combo tests and the language label tests (224-272).
  - Engine choices depend on the language.
  - Off disables the voice combo.
  - The Read combo lists only enabled languages.
  - Update follow-config (300-336) and export language (594-609).
- **Smaller updates:** `tests/test_mcp_server.py` (273-290), `tests/test_reading.py`, `tests/test_reader_server.py` and `tests/test_export.py` where they build old config keys.

## Verification

1. Run `python -m pytest -q tests`. Everything should pass.
2. Start the app with an old `config.toml` that has `other_language = "hr"` and `english_engine = "kokoro"`:
   - Settings shows English set to Kokoro Heart, Croatian set to Supertonic F1, and the rest set to Supertonic F1.
   - The saved file has `[speech.language_voices]` and no old keys.
3. Select English, Croatian, German and Spanish text and press the hotkey. The log shows the detected language and voice for each.
4. Set Spanish to Kokoro `ef_dora` and German to Off:
   - Spanish text is read by Kokoro.
   - German no longer appears under Read or in the tray Voices menu.
5. Set Read to "Always Croatian". English text is read with the Croatian voice.
6. MCP `list_voices` returns the new shape. `speak` with `voice="ef_dora", lang="es"` works, and `voice="af_heart", lang="hr"` gives a clear error.
7. Open the web reader and the extension popup. The Voice/Language selects offer Kokoro voices only for their own languages.
8. Do a PyInstaller build (`scripts/build.ps1`). The exe detects languages, so the py3langid model is bundled. Watch out for the dev-run autostart rewrite noted in memory: restore HKCU Run afterwards.
