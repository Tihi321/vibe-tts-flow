# VTF-04: Export text/documents to audio + in-app MCP server

Ticket folder: `.claude/tickets/VTF-04-export-text-documents-to-audio-in/` (this file is `plan.md`; `changelog.md` goes in the same folder). Branch: `VTF-04_export-audio-mcp`, created from `origin/master` (5aedbad) on 2026-10-15. Implement through the global `implement` skill.

## Context

VibeTTS can only read selected text aloud today. The closest thing to writing a file is the hidden CLI `python -m vibetts --say TEXT --wav out.wav`. The user wants two additions:

1. **An Export tab in the app.** It loads a document, or takes typed or pasted text, and exports the speech to an audio file.
2. **An MCP server**, so AI agents can ask VibeTTS to speak, or to synthesize text into a file.

User decisions (2026-10-15):
- **MCP host:** inside the tray app, as a localhost streamable-HTTP endpoint (`http://127.0.0.1:7723/mcp`). It is off by default and switched on in Settings.
- **Document formats:** .txt, .md, .docx, .pdf, .html, .epub.
- **Audio formats:** WAV (stdlib) and MP3 (`lameenc`, no ffmpeg). The format follows the file extension.
- **MCP tools:** `synthesize_to_file`, `speak`, `list_voices`, `stop`.

Defaults I picked; change them when approving if you want something else:
- **Export volume.** Export ignores the playback volume (`gain=1.0`). Only the `--say` CLI keeps `s.volume`.
- **Default export folder.** `~/Documents/VibeTTS`, configurable.
- **Where MCP can write.** Absolute paths are allowed anywhere. A relative path or bare name goes into the export folder. Only `.wav`/`.mp3`, and no overwrite unless `overwrite=true`.
- **Authentication.** No bearer token. Security comes from binding 127.0.0.1 only, plus Host and Origin checks.
- **Long exports over MCP.** A long `synthesize_to_file` call blocks until it is done. There is no background job mode in v1.

## Key codebase facts

- **Engines.** Both engines output float32 mono at 24 kHz through `Engine.synth(text, voice, speed, lang)` (`vibetts/tts.py`). Their per-engine locks are not fair.
- **App plumbing** (`vibetts/app.py`):
  - `EngineHolder.get(s, announce, voice)`, `route_for`, `voice_ready`, `missing_message`, `TTSApp.speak/stop/_voices`.
  - The `say()` CLI (~l.850) already synthesizes chunks with pause gaps and calls `write_wav`.
- **Text preparation.** `textprep.prepare(text, markdown, max_chars, lang, long_chunks)` treats every `\n` as a sentence end. PDF and hard-wrapped text must therefore be reflowed first.
- **Settings window** (`vibetts/settings_ui.py`):
  - Tabs come from `TABS`/`TAB_TITLES` and a `_build_<tab>(frame)` method for each.
  - Hooks come from the `SettingsHooks` dataclass, wired in `TTSApp._settings_window()` (app.py ~l.551).
  - `_var()` keys are config and count towards the "unsaved changes" check, so transient export inputs must not use `_var`.
  - The worker pattern to copy is `_omni_run()` (~l.1003): `_spawn`, `_post`, a cancel `Event` and a progress bar.
- **No stdout or stderr in the built app.** The windowed PyInstaller exe has `sys.stdout`/`sys.stderr` set to None. Any HTTP server must not log to stderr.
- **Tests.** Plain pytest with fakes: `FakeKokoro`, `FakeOmniEngine`, `make_app`, and the `make_window(root, rec, spawn=lambda fn: fn())` UI fixture.

## Approach

### 1. `vibetts/docload.py` (new; stdlib, with pypdf imported lazily)
```python
SUPPORTED = (".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".epub", ".pdf")
MAX_FILE_BYTES = 100 * 1024 * 1024
class DocumentError(ValueError): ...
@dataclass
class Document: text: str; markdown: bool; title: str   # title = file stem
def extract_text(path) -> Document
def reflow(text: str) -> str          # join wrapped lines, de-hyphenate, drop page-number lines
def decode_bytes(data: bytes) -> str  # utf-8-sig -> cp1250 -> latin-1
```
- **txt / md:** decode with `decode_bytes`. `.md` sets `markdown=True`. Hard-wrapped .txt is reflowed.
- **html:** an `HTMLParser` that skips script, style, head and svg, and turns block tags into paragraph breaks.
- **docx:** `zipfile` plus `iterparse` of `word/document.xml`. `w:p` ends a paragraph, `w:tab` is a space and `w:br` a newline. Guard against zip bombs by checking `file_size`.
- **epub:** `container.xml` → OPF → spine order → the html extractor. If there is no spine, fall back to the sorted xhtml files.
- **pdf:** `pypdf.PdfReader`. Try `decrypt("")`, and raise `DocumentError` for an encrypted file. Join pages, then `reflow`.
- Raise `DocumentError` for an unsupported suffix, a missing file, a file that is too large, or no text found.

### 2. `vibetts/export.py` (new; numpy, with lameenc imported lazily)
```python
FORMATS = ("wav", "mp3"); MP3_BITRATES = (32, 48, 64, 96, 128, 160)  # MPEG-2 @24 kHz caps at 160
class ExportCancelled(Exception); class ExportBusy(RuntimeError)
def mp3_available() -> bool
def format_for(path, default=None) -> str
def resolve_output(out, folder: Path, title: str, fmt: str, overwrite=False) -> Path
@dataclass ExportRequest(text, out=None, voice=None, speed=None, lang=None, fmt=None,
                         bitrate=None, markdown=True, gain=1.0, overwrite=False, title="")
@dataclass ExportProgress(done, total, audio_seconds, paused=False)
@dataclass ExportResult(path, fmt, seconds, chunks, bytes, voice, lang)
def open_sink(path, fmt, bitrate=64) -> AudioSink          # WavSink / Mp3Sink: write/close/abort
def render(chunks, synth, out, fmt, *, pause, gain=1.0, bitrate=64,
           progress=None, cancel=None, before_chunk=None) -> tuple[float, int]
def pcm16(samples) -> bytes                                 # shared; tts.write_wav reuses it
```
- **Streaming.** `render` writes to `<out>.part` one chunk at a time, so the whole file never sits in memory. For each chunk it calls `before_chunk`, checks cancel, synthesizes, applies gain, adds the gap, writes, and reports progress.
- **Finishing.** On success it renames the `.part` file onto the final path with `os.replace`. On any error or cancel it aborts and deletes the `.part` file.
- **MP3.** Uses `lameenc.Encoder` with input rate 24000, mono, quality 2, default 64 kbps.

### 3. `vibetts/mcp_server.py` (new; hand-rolled with the stdlib, no SDK)
Why not the official `mcp` SDK:
- It pulls in about 15 extra packages: pydantic, starlette, uvicorn and others.
- It needs PyInstaller hooks.
- uvicorn has logging trouble in the windowed exe, where there is no stdout.
- A blocking tool call blocks its event loop.

`ThreadingHTTPServer` fits the app's thread model: every request gets its own thread, so `stop` still works during a long export. The tool layer is kept separate from the transport, so moving to FastMCP later only touches one file.
```python
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
class ToolError(Exception)
@dataclass Tool(name, description, input_schema, handler, annotations={})
class McpServer:
    __init__(tools, port, host="127.0.0.1", path="/mcp", name="vibetts", version="",
             instructions="", max_body=4 MiB)
    start() / stop(); port; url; running
    handle_message(msg) -> dict | None      # pure JSON-RPC dispatch (unit-testable)
class AgentApi(Protocol): list_voices(); speak(text, voice, speed, lang, wait, timeout);
                          stop(cancel_export); synthesize_to_file(args)
def build_tools(api: AgentApi, max_chars: int) -> list[Tool]
```
**Protocol:**
- Supports `initialize` (version negotiation), `notifications/*` (HTTP 202), `ping`, `tools/list` and `tools/call`.
- `tools/call` returns text content plus `structuredContent`; a `ToolError` sets `isError: true`.
- Standard JSON-RPC error codes. Batches are handled.
- POST answers with JSON. GET and DELETE return 405.

**Server details:**
- `daemon_threads = True` and `block_on_close = False`, so quitting never hangs on a running request.
- `log_message` goes to `logging`. Logs record tool names and text lengths, never the text itself.

**Security:**
- Bind 127.0.0.1 only.
- The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`, and an `Origin` header, if present, must be a localhost origin. Anything else gets 403 (DNS-rebinding defence).
- A body that is too large gets 413.

**Tools:**
- **`synthesize_to_file`**
  - Inputs: `text` or `path` (exactly one), plus optional `output`, `format`, `voice`, `speed` (0.5-2), `lang` (en/hr), `overwrite`.
  - Returns `{path, format, duration_seconds, chunks, voice, lang, bytes}`.
  - Text longer than `max_chars`, or a second export while one is running, gives a ToolError.
- **`speak`**
  - Inputs: `text` plus optional `voice`, `speed`, `lang`, `wait` (default false), `timeout_seconds` (≤ 600).
  - Returns `{started, completed, interrupted, error}`.
  - Refused while the app is disabled from the tray.
- **`list_voices`**
  - Returns `{voices: [{id, label, engine}], engine, default_voice, omnivoice_installed}`.
- **`stop`**
  - Input: optional `cancel_export`.

### 4. Changes to existing files
- **`vibetts/config.py`:**
  - Add `ExportConfig(folder="", format="mp3", mp3_bitrate=64)`. An empty `folder` means `~/Documents/VibeTTS`; add the helper `export_folder(cfg) -> Path`.
  - Add `McpConfig(enabled=False, port=7723, max_chars=200000)`.
  - Add both as fields on `Config`.
  - Validate with a warning for each correction: an unknown format becomes "wav", the bitrate snaps to an allowed value, and an out-of-range port becomes 7723.
  - Mirror both sections in `config.example.toml`.
- **`vibetts/tts.py`:**
  - `SpeakRequest` gets `done: threading.Event`, `error` and `interrupted`. `Speaker._play`'s `finally` sets them; MCP `speak(wait=true)` needs this.
  - `write_wav` reuses `export.pcm16`.
- **`vibetts/app.py`:**
  - **`plan_speech(s, text, voice=None, lang=None, markdown=True) -> (chunks, voice, lang, omni)`**
    - With no voice, it uses `route_for`.
    - With an explicit voice, it normalizes omni ids and guesses the language with `looks_croatian`.
  - **`run_export(engine, s, req, folder, progress, cancel, before_chunk) -> ExportResult`:** shared by the CLI, the Export tab and MCP.
  - **`TTSApp.export_audio(req, progress=None, cancel=None)` and `cancel_export()`**
    - One export at a time: `_export_lock` is acquired without blocking, otherwise `ExportBusy`.
    - Each call snapshots the config with `deepcopy(self.cfg)`.
    - `before_chunk = _export_yield`, which pauses the export while `speaker.busy`, so reading aloud gets priority.
  - **`speak()`** returns its `SpeakRequest`.
  - **`AgentBridge`** implements `AgentApi` on top of `TTSApp`.
  - **`_sync_mcp()`**
    - Starts, stops or restarts the `McpServer` when `(enabled, port)` changes. It is called from `_start()` and `apply_config()`.
    - If the port is busy, it records the error and flashes the pill.
    - Add `mcp_status() -> {enabled, running, url, error}`.
  - **`quit()`** stops MCP and cancels the export. `_omni_uninstall` also cancels the export first.
  - **CLI**
    - `say(cfg, text, out)` delegates to `run_export`, keeping `gain=s.volume`. The format comes from the file extension.
    - New flags `--out PATH` (`--wav` stays as an alias) and `--doc PATH`, which goes through `docload`.
  - **Wiring:** the new SettingsHooks, and the tray hook `open_export`.
- **`vibetts/tray.py`:** Add `TrayHooks.open_export` and an "Export audio…" menu item after "Settings…".
- **`vibetts/settings_ui.py`:**
  - **Tabs:** `TABS = ("general", "voice", "export", "kokoro", "omnivoice", "mcp", "about")`, with titles "Export" and "Agents (MCP)".
  - **Form keys:** `export_folder`, `export_format`, `mp3_bitrate`, `mcp_enabled`, `mcp_port`, `mcp_max_chars`, mapped in `form_from_config` and `config_from_form`.
  - **New `SettingsHooks`:** `load_document`, `export_audio`, `cancel_export`, `mp3_available`, `mcp_status`, `open_folder`. Each defaults to a generic `_not_supported()`.
  - **`_build_export`**
    - **Source:** a `tk.Text` with a scrollbar, plus "Open document…" (`askopenfilename` filtered to `SUPPORTED`, loaded on a worker thread) and "Clear". Show the character count and an estimated length.
    - **Voice:** a combo with "Auto (as configured)" plus all voices, a speed entry and a language combo (Auto/English/Croatian). These are transient tk variables, not `_var`.
    - **Output:** format, bitrate and folder controls (these are `_var`), and an "Export…" button that opens `asksaveasfilename` with `initialdir=folder` and `initialfile=<title>.<fmt>`.
    - **Progress:** a progress bar, a label ("Chunk 12/340 · 3:12 audio" or "Paused while reading aloud"), Cancel, and "Open folder" when finished. This copies the `_omni_run` pattern.
    - MP3 is disabled when `mp3_available()` is false. `ExportBusy` shows "An agent export is running".
  - **`_build_mcp`**
    - An enable checkbox, port and max-characters entries, and a status label from `mcp_status()`.
    - A copyable `claude mcp add --transport http vibetts <url>` line and a `.mcp.json` snippet.
    - A short note on the security model.

### 5. Packaging, dependencies, docs
- **Dependencies:** add `pypdf>=4` and `lameenc>=1.7` to `requirements.txt` and `pyproject.toml`. First check that lameenc has a cp313 win_amd64 wheel. If it doesn't, MP3 is disabled in the UI and the app still runs.
- **Packaging:** add `vibetts.docload`, `vibetts.export`, `vibetts.mcp_server`, `lameenc` and `pypdf` to the `hiddenimports` in `packaging/VibeTTS.spec`.
- **`README.md`:**
  - New sections "Export to audio file" and "Agents (MCP)": the add command, the tools and the security model.
  - CLI: `--out`/`--doc`.
  - Privacy: export writes only the audio file you choose; text is never stored; MCP is off by default and listens on localhost only.
  - Project layout: the 3 new modules.

### Reused pieces
- `textprep.prepare`, `looks_croatian`
- `route_for`, `voice_ready`, `missing_message`, `EngineHolder.get`
- `tts.voice_label`, `omni_preset_info`, `voices_in_file`
- `Speaker`
- The `_omni_run` worker/progress pattern
- `config._build/_coerce`, which handles new sections automatically

## Execution (implement skill)
- **Wave 0, orchestrator:** stub the contracts above (signatures and docstrings) and add the config dataclasses.
- **Wave 1, in parallel, each in its own files:**
  - **A:** docload with tests.
  - **B:** export with tests.
  - **C:** mcp_server with tests, against `FakeApi`.
  - **D:** config validation, `config.example.toml` and tests.
  - **F:** settings_ui Export and MCP tabs with tests, using fake hooks.
- **Wave 2:**
  - **E:** app.py, tts.py, tray.py wiring, CLI, and tests in `test_app` and `test_tts`.
- **Wave 3:**
  - **G:** spec, dependencies, README, changelog, then the build and manual checks.

## Verification
- **Automated:** `.venv\Scripts\python -m pytest -q`. The existing 218 tests stay green, plus these new tests:
  - **`test_docload`:** txt in utf-8-sig and cp1250, html, docx and epub built in `tmp_path`, pdf through a fake `PdfReader` (including encrypted and empty), `reflow` cases, and errors.
  - **`test_export`:**
    - WAV frame count equals audio plus gaps, and gain is applied.
    - The progress sequence is right, and `before_chunk` is called once per chunk.
    - Cancel or an error leaves neither a file nor a `.part`.
    - `resolve_output` rules.
    - MP3 frame sync, skipped when lameenc is missing.
  - **`test_mcp_server`:**
    - `handle_message`: negotiation, list, call, ToolError, -32601, notification.
    - HTTP on `port=0`: 200, 202, 405, 403 for a bad Host or Origin, 413, and the port is freed after `stop`.
    - `build_tools` validation, and ExportBusy maps to a ToolError.
  - **`test_config`:** round-trip of the new sections, and validation.
  - **`test_tts`:** `req.done`, `error` and `interrupted`.
  - **`test_app`:**
    - `plan_speech`, ExportBusy, the export pausing while the speaker is busy, and `AgentBridge.speak(wait=True)`.
    - `_sync_mcp` start, stop and restart, and the port-busy path.
    - The tray item and the hooks wiring.
    - `say` with `--out` and `--doc`.
  - **`test_settings_ui`:**
    - The new form keys round-trip.
    - Export tab with `filedialog` monkeypatched: a loaded document fills the box, and Export calls the hook with the right request.
    - MCP tab: status text and the command string.
- **Manual, from source (`python run.pyw --console`):**
  1. In the Export tab, load a .pdf, .docx and .epub, export each to MP3 and WAV, and play the results.
  2. Start an export, then press the read hotkey: reading aloud starts right away and the export shows "Paused". Cancel the export: no `.part` file is left.
  3. Enable MCP and run `claude mcp add --transport http vibetts http://127.0.0.1:7723/mcp`. From Claude Code, call `list_voices`, `speak` (with `wait=true`), `stop` and `synthesize_to_file` (with text, and with a document path).
  4. Run `npx @modelcontextprotocol/inspector` against the URL.
  5. Change the port, or set a port that is already in use: the status label updates or shows the error.
  6. Quit during an export: the app does not hang.
- **Build:** run `scripts/build.ps1`, then repeat steps 1 and 3 with `dist\VibeTTS\VibeTTS.exe`. This checks that the exe with no console runs the HTTP server, pypdf and lameenc work.
- **Cannot be automated:** audio quality, real MCP clients, and the frozen exe.

## Status (2026-10-15)

- [x] Wave 0 — config dataclasses + validation (`ExportConfig`, `McpConfig`, `export_folder`, `snap_bitrate`), contract stubs for `export.py` / `docload.py`
- [x] A — `vibetts/docload.py` + `tests/test_docload.py` (30 tests)
- [x] B — `vibetts/export.py` + `tests/test_export.py` (14 tests)
- [x] C — `vibetts/mcp_server.py` + `tests/test_mcp_server.py` (16 tests, incl. ExportCancelled → ToolError added in review)
- [x] D — `config.example.toml` `[export]`/`[mcp]`, `tests/test_config.py`
- [x] E — `app.py` (plan_speech, run_export, export_audio, AgentBridge, _sync_mcp, CLI `--out`/`--doc`), `tts.py` (SpeakRequest done/error/interrupted), `tray.py` ("Export audio…")
- [x] F — `settings_ui.py` Export + Agents (MCP) tabs
- [x] G — requirements/pyproject (pypdf, lameenc 1.8.4 has a cp313 wheel), spec hiddenimports, README
- [x] pytest: 346 passed, 1 skipped (pre-existing onnx skip)
- [x] Real engine: CLI `--say … --out x.mp3` (6.9x real time) and `--doc doc.md --out doc.wav` (8.5x)
- [x] Real app from source with MCP on: initialize, tools/list, list_voices, synthesize_to_file (text → mp3, .md path → wav), overwrite refusal, unknown voice
- [x] Built exe (`scripts/build.ps1 -SkipTests`): MCP server runs windowed; PDF → MP3 via pypdf + lameenc; log has lengths only
- [ ] Manual, not run: Export tab with real documents end to end, pause-while-reading, Cancel in UI, `speak` audible output, a real `claude mcp add` client, MCP Inspector

## Addendum (2026-10-15, user request mid-implementation): MCP `create_voice`

User: "add to mcp option to create voices by sending description of voice as well".

- New MCP tool **`create_voice`** designs and saves an OmniVoice voice preset from a description. It reuses `TTSApp._design_clip(instruct, seed, text, lang)` and `TTSApp._save_voice(name, clip, overwrite)`, the same path as Settings > OmniVoice > Create voice.
  - **Inputs:**
    - `name` (required). Sanitised by `omnivoice.sanitize_name`: `[a-z0-9_-]`, at most 32 characters, `default` reserved.
    - `description` (optional): comma-separated items from `omnivoice.INSTRUCT_OPTIONS`, at most one per category (gender, age, pitch, style). Empty means a random voice.
    - `seed` (optional integer; random when absent, and returned).
    - `lang` (`en`/`hr`): the language of the sample clip.
    - `overwrite` (default false).
  - **Returns** `{voice: "omni:<name>", name, description, seed, lang}`.
  - **Errors** are tool errors: OmniVoice not installed (`MISSING_OMNI`), a bad description (`validate_instruct`'s message with the valid items), a bad or reserved name, or an existing preset without `overwrite`.
  - It does not change the user's default voices.
- `list_voices` entries for OmniVoice presets gain `kind`, `description` and `lang` (from `omni_preset_info`), so agents can see what a voice is.
- The tool schema lists the valid description items, so agents don't need to guess them.

### Addendum status (2026-10-15)
- [x] `create_voice` tool + `AgentBridge.create_voice` + `list_voices` preset details; pytest 352 passed, 1 skipped
- [x] Real OmniVoice via MCP (app from source): bad description refused with the valid items; `vtf04-test` created (female, young adult, low pitch, seed 7); duplicate refused without overwrite; listed with description; Croatian export with it → MP3. Test preset deleted afterwards.
