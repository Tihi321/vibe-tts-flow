# Changelog: VTF-08

## 2026-10-21: Per-language voices (Kokoro or Supertonic) with full language detection

Every one of the 31 Supertonic languages has its own voice, set to Kokoro (where Kokoro reads it), Supertonic, or Off. Each read detects which of the enabled languages the text is in. This replaces the English profile plus one "other language" profile.

### Repo `vibe-tts-flow`, branch `VTF-08_per-language-voices` (from `origin/master` 6beb26b)

**Config**
- `vibetts/config.py`:
  - Six `SpeechConfig` fields are replaced:
    - Removed: `voice`, `english_engine`, `english_supertonic_voice`, `other_language`, `other_voice`, `language_mode`.
    - Added `language_voices` (code to voice, `""` = Off; all 31 keys).
    - Added `read_language` (`auto` or a code).
  - Old configs are migrated on load: the English voice goes to `en`, the other voice to `other_language`, and `language_mode` to `read_language`.
  - Validation:
    - A voice must fit its language.
    - If every language is Off, English is turned back on.
    - A `read_language` that is not enabled resets to `auto`.
  - `_coerce` now reads string dicts.
  - New helpers: `KOKORO_LANGUAGES`, `DEFAULT_KOKORO_VOICES`, `DEFAULT_LANGUAGE_VOICES`, `READ_AUTO`, `kokoro_voice_lang`, `voice_fits_language`, `enabled_languages`.
- `config.example.toml`: documents `read_language` and the `[speech.language_voices]` table.

**Detection**
- `vibetts/langdetect.py` (new):
  - `detect_language(text, candidates, fallback)` uses py3langid, loaded lazily and behind a lock.
  - Text under 4 words (and under 15 letters) goes to English if English is enabled, otherwise to the fallback.
- `vibetts/textprep.py`: adds `word_count`.
- `pyproject.toml` and `requirements.txt`: add the `py3langid` dependency.
- `packaging/VibeTTS.spec`: bundles the py3langid model with `collect_data_files`.

**Engines and routing**
- `vibetts/tts.py`: Kokoro now reads Spanish, French, Hindi, Italian and Brazilian Portuguese voices as well as English. The espeak language is taken from the voice prefix.
- `vibetts/app.py`:
  - Routing changes:
    - The voice comes from the detected or fixed language's row.
    - If Supertonic is missing, a language uses its Kokoro default voice where Kokoro reads it, otherwise Kokoro English.
    - `route_request` accepts a Kokoro voice for its own language and raises a clear error on a mismatch.
    - Preloading covers the distinct voices of the enabled languages.
  - Each read logs its language and voice.
  - New `TTSApp._set_voice(lang, voice)` and `_set_read_language`. Turning off the last enabled language is refused, and turning off the Read language resets Read to Auto.
  - `_play_test` takes a `lang`.
  - `list_voices` (MCP and reader API) returns per-voice `languages`, `language_voices` and `read_language`. It no longer returns `english`, `other` or `language_mode`.

**Tray, MCP and Settings**
- `vibetts/tray.py`:
  - The **Language** menu offers Auto and "Always <name>" for each enabled language.
  - The **Voices** menu has a submenu per enabled language with Off, its Kokoro voices and the Supertonic voices.
  - The hooks changed to match.
- `vibetts/mcp_server.py`: the instructions and schema text describe which languages Kokoro and Supertonic read.
- `vibetts/settings_ui.py`:
  - The Voice tab has one **Voices** section with Language, Engine, Voice and Read selectors.
  - Engine shows Kokoro only for languages Kokoro reads.
  - Off turns off the Voice selector and adds " (off)" to the language name.
  - Turning off the last enabled language is refused.
  - Read lists only enabled languages.
  - The Supertonic-missing hint now depends on the enabled rows.
  - Export lists Auto plus all 31 languages and all supported voices.
  - Play previews the selected language.

**Web reader, extension and docs**
- `vibetts/web/app.js` and `extension/popup.js`: the Language select lists only the languages the chosen voice reads, and a language the new voice doesn't read resets to Auto.
- `README.md`: rewrites the engine, language and voice section, and updates the tray, MCP, export and troubleshooting text and the module list.

**Tests**
- `tests/test_config.py`, `tests/test_app.py`, `tests/test_settings_ui.py`, `tests/test_mcp_server.py`, `tests/test_tts.py` updated.
- `tests/test_langdetect.py` added.

**Ticket**
- `.claude/tickets/VTF-08-per-language-voices-kokoro-or-supertonic/mcp-changes.md` describes the `list_voices` change for MCP client authors.

### Deviations from the plan
- **Settings Play button:** the hook changed to `play_test(cfg, text, lang)` so Play can preview one language row.
- **Detection of unspaced scripts:** text with 15 or more letters counts as long enough to detect, so Japanese is not forced to English.
- **New helpers not in the plan:** `DEFAULT_KOKORO_VOICES` (the Kokoro fallback when Supertonic is missing), `TrayHooks.languages` and `textprep.word_count`.
- **Extra test file:** `tests/test_tts.py` was updated, which the plan did not list.

### Verification
- **Tests:** `.venv/Scripts/python.exe -m pytest -q tests` gives 607 passed, 1 skipped.
- **Detection spot check:** whole-sentence samples were all detected correctly in en, hr, sl, sk, cs, de, es, pt and it. Serbian in Latin script comes out as Slovenian. A 3-word Croatian phrase goes to English, by design.
- **Settings window:** I rendered the Voice tab with the test hooks and took screenshots. Spanish offers Kokoro with Dora, German set to Off shows "(off)" with the Voice selector turned off, and Read lists Auto plus the enabled languages.
- **JS syntax:** `node --check` passes on `vibetts/web/app.js` and `extension/popup.js`.

### Not run
- Real hotkey reads in the running app.
- Migrating a real old `config.toml`.
- MCP `speak` / `list_voices` against the running app.
- The web reader and extension in a browser.
- Running the built exe. `scripts/build.ps1` (tests included) succeeded, and `dist/VibeTTS/_internal/py3langid/data/model.npz.xz` is bundled.

### Follow-ups and known limits
- Detection runs once per read, so a mixed-language selection is read in one voice (the user chose this).
- Serbian, Bosnian and other unsupported languages are mapped to the closest enabled language.
- `langdetect.py` reads py3langid's private `_full_model` to filter candidates to the languages it knows.

### Commit status
Uncommitted, on branch `VTF-08_per-language-voices`.
