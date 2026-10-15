# VTF-04 changelog

## 2026-10-15: export text and documents to audio, plus a local MCP server

**Outcome:**
- **Export tab.** A new **Settings > Export** tab turns typed or pasted text, or a loaded document (.txt, .md, .html, .docx, .epub, .pdf), into a WAV or MP3 file. It shows progress and has a Cancel button.
- **MCP server.** An opt-in localhost MCP server (`http://127.0.0.1:7723/mcp`) lets AI agents call `synthesize_to_file`, `speak`, `list_voices` and `stop`.
- **CLI.** New flags `--out` and `--doc`.

Repo: `vibe-tts-flow`. Branch: `VTF-04_export-audio-mcp`, created from `origin/master` at 5aedbad.

### Files

**New modules**
- **`vibetts/docload.py`** (new)
  - `extract_text(path) -> Document(text, markdown, title)` for txt, md, html, docx, epub and pdf.
  - It uses the standard library for every format except PDF, which goes through pypdf, imported lazily.
  - Text is decoded as utf-8, falling back to cp1250 and then latin-1.
  - `reflow()` joins hard-wrapped lines, de-hyphenates words split across lines and drops page numbers. Without it, textprep would treat every PDF line as a sentence.
  - Files over 100 MB are refused, and so is any zip entry over 200 MB (zip-bomb guard).
  - Errors are `DocumentError` with a message meant for the user.
- **`vibetts/export.py`** (new)
  - `render()` streams chunks one at a time into a `.part` file, then renames it over the target with `os.replace`. On cancel or error the partial file is deleted.
  - `WavSink` uses the stdlib `wave` module; `Mp3Sink` uses lameenc at 24 kHz mono, 32-160 kbps.
  - `resolve_output()` applies the path rules: bare names go into the export folder, only `.wav`/`.mp3` are allowed, and no overwrite without the flag.
  - Also `format_for`, `pcm16`, `mp3_available`, and the `ExportRequest`/`ExportProgress`/`ExportResult` dataclasses with `ExportBusy` and `ExportCancelled`.
- **`vibetts/mcp_server.py`** (new)
  - A stdlib `ThreadingHTTPServer` speaking MCP Streamable HTTP. It replies with plain JSON (no SSE) and supports protocol versions 2024-11-05 through 2025-11-25.
  - **Request checks:**
    - binds 127.0.0.1 only;
    - `Host` and `Origin` must be localhost (403 otherwise);
    - body limit 4 MiB (413);
    - GET and DELETE get 405.
  - **Tools.** `build_tools()` defines the four tools with strict JSON schemas and input validation.
  - **Logging.** Never prints; logs tool names and text lengths only.

**Changed modules**
- **`vibetts/config.py`**
  - New `[export]` section (`folder`, `format`, `mp3_bitrate`) and `[mcp]` section (`enabled`, `port`, `max_chars`).
  - Values are validated on load, with a warning for each correction.
  - New helpers `export_folder()` (default `~/Documents/VibeTTS`) and `snap_bitrate()`.
- **`vibetts/app.py`**
  - **Shared export path.** New `plan_speech()` and `run_export()`, used by the CLI, the Export tab and MCP.
  - **`TTSApp.export_audio()`.** One export runs at a time; a second one gets `ExportBusy`. It works on a copy of the config and pauses while the app is reading aloud. `cancel_export()` stops it.
  - **`AgentBridge`.** This is the app side of the MCP tools.
  - **Server lifecycle.** `_sync_mcp()` starts, stops or restarts the server on startup and whenever settings are applied. A busy port is reported in the Settings tab and on the pill. `mcp_status()` reports the state.
  - **Quit and uninstall.** On quit the app stops the MCP server and cancels any export. Uninstalling OmniVoice also cancels any export first.
  - **`speak()`** now returns its `SpeakRequest`.
  - **New hooks.** The six new `SettingsHooks` are wired, plus the tray hook `open_export`.
  - **CLI.**
    - `--out PATH` replaces `--wav`, which still works as an alias. The format is picked by extension.
    - `--doc PATH` reads a document aloud, or exports it when combined with `--out`.
- **`vibetts/tts.py`**
  - `SpeakRequest` gains `done` (an Event), `error` and `interrupted`, set on every exit path. MCP `speak(wait=true)` relies on them.
  - `write_wav` reuses `export.pcm16`.
- **`vibetts/tray.py`:** new "Export audio…" menu item after "Settings…".
- **`vibetts/settings_ui.py`**
  - The tab list is now General, Voice, Export, Kokoro, OmniVoice, Agents (MCP), About.
  - **Export tab:**
    - a text box with Open document…, Paste and Clear;
    - character count and an estimate of the audio length;
    - voice, speed and language choices that are not saved to the config;
    - format, bitrate and folder settings that are saved;
    - a Save dialog, progress (including "Paused while reading aloud"), Cancel, and Open folder.
  - **Agents (MCP) tab:** enable switch, port, max characters, status line, and a copyable `claude mcp add` command and `.mcp.json` snippet.

**Config, packaging and docs**
- **`config.example.toml`:** the `[export]` and `[mcp]` sections.
- **`requirements.txt`, `pyproject.toml`:** `pypdf>=4`, `lameenc>=1.7`.
- **`packaging/VibeTTS.spec`:** hidden imports for the three new modules plus lameenc and pypdf.
- **`README.md`:**
  - feature bullets;
  - "Export to an audio file" and "Agents (MCP)" sections, covering tools and security;
  - CLI examples;
  - privacy note;
  - project layout.

**Tests**
- New: `tests/test_docload.py`, `tests/test_export.py`, `tests/test_mcp_server.py`.
- Extended: `tests/test_config.py`, `tests/test_tts.py`, `tests/test_app.py`, `tests/test_settings_ui.py`.

### Added mid-implementation (user request): MCP `create_voice`
- `vibetts/mcp_server.py`: new `create_voice` tool, placed after `list_voices`.
  - Inputs: `name` (required), `description`, `seed`, `lang` (en/hr), `overwrite`.
  - The schema lists the valid description vocabulary, which comes from `omnivoice.INSTRUCT_OPTIONS` via `build_tools(voice_options=...)`.
  - `AgentApi.create_voice` was added.
- `vibetts/app.py`: new `AgentBridge.create_voice`.
  - It needs OmniVoice installed.
  - It checks the name and refuses an existing preset unless `overwrite` is set. Both checks run before the seconds-long design step.
  - It picks a random seed when none is given and returns it.
  - It designs the voice with `_design_clip`, then saves it with `_save_voice`.
  - The user's default voices are left unchanged.
  - `list_voices` OmniVoice entries now also include `kind`, `description` and `lang`.
- `README.md`: added a `create_voice` row to the tools table.
- Tests: new `create_voice` cases in `tests/test_mcp_server.py` and `tests/test_app.py`.

### Deviations from the plan
- **Contract stubs.** The orchestrator wrote stubs for `export.py` and `docload.py` and did the config validation in Wave 0. This let the UI slice code against the final dataclasses while the other slices were still being written.
- **MCP slice.**
  - Added after review: `mcp_server` maps a cancelled export to a tool error ("The export was cancelled; no file was written") rather than an internal error, and its instructions text describes OmniVoice as multilingual.
  - Content-Type other than JSON gets 415, and an empty batch gets 400.
- **`plan_speech` language override.**
  - `lang="hr"` with no voice uses the Croatian voice when OmniVoice is installed, else Kokoro with `en`.
  - `lang="en"` routes with auto-language off.
- **MCP checks beyond the plan.** Documents loaded through a `path` also count against `[mcp] max_chars`. A format that doesn't match the output extension is refused.
- **CLI behaviour changes.**
  - `--out` without `--say`/`--doc` is now an error. Before, `--wav` alone was silently ignored.
  - An `--out` with no extension gets `.mp3` (the `[export] format` default) instead of writing a WAV without an extension.
- **Export text box** shortened to 7 lines, so the Export button is visible without scrolling. Checked with a screenshot.

### Verification
- **Unit tests.** `.venv\Scripts\python -m pytest -q`: **352 passed, 1 skipped** after `create_voice`; it was 346 passed before it. The skip is the existing `test_omnivoice_setup.py` onnx skip. Before this ticket the count was 218 passed, 1 skipped.
- **CLI with the real Kokoro engine:**
  - `--say … --out hello.mp3` wrote a 3.5 s MP3 at 6.9x real time.
  - `--doc doc.md --out doc.wav` wrote 19.7 s of audio at 8.5x real time.
- **App from source** (`run.pyw --hidden`) with MCP on port 7799, called over raw HTTP:
  - `initialize`, `tools/list` and `list_voices` (42 voices) worked.
  - `synthesize_to_file` turned text into an MP3 in the export folder, and a `.md` path into a WAV with voice `bm_george`.
  - Overwriting an existing file without `overwrite` was refused with a tool error, and so was an unknown voice in `speak`.
- **Built exe.** `scripts\build.ps1 -SkipTests` succeeded. `dist\VibeTTS\VibeTTS.exe --hidden` with MCP on: `list_voices` worked, and `synthesize_to_file` on a generated PDF produced a 4.1 s MP3, so pypdf and lameenc work in the frozen build. The log shows only text lengths.
- **Screenshots** of the Export and Agents (MCP) tabs were checked.
- **`create_voice` against the real OmniVoice engine**, with the app running from source:
  - A bad description item was refused with the list of valid items.
  - A preset was created from "female, young adult, low pitch" with seed 7.
  - Creating it again without `overwrite` was refused.
  - `list_voices` showed it with its description.
  - A Croatian `synthesize_to_file` with the new voice wrote an MP3.
  - The test preset was deleted afterwards.
  - Not run in the built exe.

**Not run:**
- the Export tab clicked through by hand with real .docx, .epub and .pdf files;
- pausing an export while the hotkey reads aloud, and Cancel in the UI, with real audio (covered by unit tests only);
- the MCP `speak` tool producing audible output;
- a real `claude mcp add` client and the MCP Inspector;
- quitting during a long export.

### Follow-ups and limitations
- **Long exports over MCP.** A long `synthesize_to_file` call blocks until it finishes, and some clients may time out. A background job mode with an `export_status` tool would fix this.
- **No MCP token.** Any local process can use the server while it is on, as the README says. A bearer token can be added later.
- **Scanned PDFs** have no text and are refused with a message; there is no OCR.
- **Build output.** `dist\` was rebuilt by the verification build.

### Commit status
Not committed. All changes are in the working tree on `VTF-04_export-audio-mcp`.
