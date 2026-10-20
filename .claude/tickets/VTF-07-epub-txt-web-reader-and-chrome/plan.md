# VTF-07: EPUB/TXT web reader and Chrome "read this page" extension

> Ticket folder: `.claude/tickets/VTF-07-epub-txt-web-reader-and-chrome/` (this plan, `contract.md`, `changelog.md`). Branch: `VTF-07_book-reader-extension` (from `origin/master` 4bfeba7). Implemented through the global `implement` skill.

## Context

VibeTTS is a Windows tray app (Python, tkinter, Kokoro + Supertonic 3, 44.1 kHz). It reads the selected text or exports documents to audio files. You want to read books with it and have web pages read aloud. Today:

- `docload.extract_text` can already parse EPUB and TXT. It flattens them to one string with no chapters, though.
- Playback can only start and stop: no pause, no per-sentence progress, no saved position.
- The only HTTP server is MCP (`mcp_server.py`, port 7723, off by default). It returns 403 to any non-localhost `Origin`, so a browser extension can't call it.

**Decisions you made:**
- The reader is a **local web page** served by the app.
- The extension sends text to the app, and **the app plays it on the PC speakers** through the existing `Speaker`.
- The extension can:
  - read the selected text;
  - read the whole article (or start from a point in it);
  - highlight the sentence being read;
  - show popup controls.

**Result:** a new localhost "Reader API" server (default port **7724**, on by default) serves both the web reader and the extension. Both split text into sentences on the client and send a list of items. The app reports which item it is playing, so each client highlights the sentence in its own view. Clients never need to map source offsets back to the text.

## Architecture

```
web reader (http://127.0.0.1:7724/)  ─┐
Chrome extension (service worker)    ─┼─► reader_server.py ─► reading.ReadingSession ─► TTSApp.speak ─► tts.Speaker
                                       │        │                                         (pause/resume, on_chunk)
                                       │        └─► library.Library ─► docload.extract_book
                                       └─ long-poll GET /api/status ◄── session state / current item
```

The app has one playback slot. A new session, a hotkey read or an MCP `speak` replaces whatever is playing. A client sees that as a change of `session` in the status.

## API contract (`contract.md`)

All routes are under `http://127.0.0.1:<port>`. Bodies are JSON unless noted otherwise.

| Method & path | Body / query | Response |
|---|---|---|
| `GET /` and static files | (none) | Web reader files from `vibetts/web/` |
| `GET /api/info` | (none) | `{app:"vibetts", version, api:1}` |
| `GET /api/voices` | (none) | Same shape as MCP `list_voices`; reuse `AgentBridge.list_voices` |
| `POST /api/speak` | `{items:[{text, pause?:"sentence"\|"paragraph"}], start?:int=0, voice?, lang?, speed?, source:"reader"\|"extension", meta?:{}}` | `{session, count}` |
| `POST /api/pause` · `/api/resume` · `/api/stop` | `{}` | status |
| `POST /api/seek` | `{item:int}` | status (restarts playback at that item, same session) |
| `GET /api/status` | `?since=<version>&timeout=<s≤25>` (long-poll: returns at once if `version != since`) | `{version, session, source, meta, state:"idle"\|"loading"\|"playing"\|"paused"\|"finished"\|"stopped"\|"error", item, count, error, voice, lang, speed}` |
| `GET /api/books` | (none) | `{books:[{id,title,author,format,chapters:int,position:{chapter,item},added,opened}]}` |
| `POST /api/books` | Raw bytes, `Content-Type: application/octet-stream`, `X-Filename: <name>` | The new or existing book (de-duplicated by sha256) |
| `GET /api/books/{id}` | (none) | `{...book, toc:[{index,title,chars}]}` |
| `GET /api/books/{id}/chapters/{n}` | (none) | `{index,title,paragraphs:[str]}` |
| `PUT /api/books/{id}/position` | `{chapter,item}` | `{ok:true}` |
| `DELETE /api/books/{id}` | (none) | `{ok:true}` |

**Security**, modelled on the MCP server checks:
- `Host` must be `127.0.0.1:<port>` or `localhost:<port>`; otherwise 403. This protects against DNS rebinding.
- A request with an `Origin` header must come from the reader's own origin or from `chrome-extension://<id>` with `<id>` in `reader.extension_ids`; otherwise 403.
- Requests that change state (POST, PUT, DELETE) must have an allowed `Origin`, and their Content-Type must be JSON or octet-stream.
- There are no CORS headers. The extension calls the API from its service worker using `host_permissions`, which doesn't need CORS.
- Limits:
  - upload ≤ 100 MB (`docload.MAX_FILE_BYTES`);
  - JSON body ≤ 4 MiB;
  - ≤ 20 000 items;
  - total text ≤ `reader.max_chars` (default 500 000).
- Logs record lengths and counts, never the text (same as MCP).

## Changes

### 1. Playback core
- **`vibetts/textprep.py`:** add `tag: Any = None` to `Chunk` (:72). Add `prepare_items(items, markdown=False) -> list[Chunk]`:
  - Run `clean_text` and `split_chunks` on each item.
  - Tag each resulting chunk with its item index.
  - Use the item's `pause` for the pause after its last chunk: 1.0 at a sentence end, 2.0 at a paragraph end.
  - Apply the short-first-chunk rule only to item 0.
- **`vibetts/tts.py`:**
  - `SpeakRequest` (:164) gets an `on_chunk: Callable[[Chunk], None] | None`. `_produce` puts `(chunk, samples)` on the queue, and `_play` calls `on_chunk` when it starts writing that chunk.
  - Add `Speaker.pause()` and `Speaker.resume()` plus a `paused` property. Pausing:
    - gates the 100 ms write loop in `_play` (:287) with a resume Event;
    - aborts and closes the output on pause and reopens it on resume, so the device is released;
    - keeps the remaining samples of the current chunk.
  - `stop()` also releases a pause.
- **`vibetts/app.py`:** `TTSApp.speak` (:569) takes optional `speed` and `on_chunk`. Add `TTSApp.pause()`, `resume()` and `toggle_pause()`; the overlay shows "Paused". `stop()` stays as it is.

### 2. Reading session: new `vibetts/reading.py`
`ReadingSession` is owned by `TTSApp` and thread-safe. Its state is the status dict above, plus a `version` counter and a `threading.Condition` for the long-poll.
- **`start(items, start, voice, lang, speed, source, meta)`:**
  - Route with `route_request` (`app.py:153`) on the joined text.
  - Check `voice_ready` and `note_fallback`.
  - Call `prepare_items` and start `TTSApp.speak` with the chunks from `start` onward and `on_chunk` set to update `item`.
  - The request's `done` event moves the state to `finished`, `stopped` or `error`. A replaced request gives `stopped`.
- **`seek(i)`:** speak again from item `i` with the same session id.
- **`pause()`, `resume()`, `stop()`:** delegate to `TTSApp`.
- **`wait(since, timeout)`:** the long-poll.
- When something else replaces playback (hotkey, MCP `speak`), the session ends in `stopped`. Hook this through `TTSApp._on_speak_done` comparing `req is self._current`.

### 3. Books
- **`vibetts/docload.py`:** add `Chapter(title, text)`, `Book(title, author, chapters)` and `extract_book(path) -> Book`. Keep `extract_text` unchanged; it can join the chapters.
  - **EPUB:** reuse the spine walk in `_epub_text` (:178) and `_html_to_text`.
    - Take the title and author from OPF `dc:title` and `dc:creator`.
    - Take chapter titles from EPUB3 `nav[epub:type=toc]` or EPUB2 `toc.ncx`, matching hrefs without the `#fragment`.
    - With a TOC, a spine item that has no entry is appended to the previous chapter (the first ones become "Front matter"). Without a TOC, each spine item is a chapter titled by its first heading, or "Section N".
    - Drop empty chapters.
  - **TXT:** `decode_bytes` plus `reflow`, as in `extract_text`. Chapter headings are short standalone paragraphs (≤ 80 chars) matching `^(chapter|part|book|prologue|epilogue|[ivxlc]+\.?$|\d+\.?$)`, case-insensitive. With fewer than two headings, split into ~20 000-char "Part N" chunks at paragraph boundaries.
  - **Other `SUPPORTED` types** (md, html, docx, pdf) become one-chapter books via `extract_text`.
- **New `vibetts/library.py`:** `Library(folder)`.
  - `add(data, filename)` stores `library/<id><ext>`, where `id` is the first 16 hex chars of the sha256. It caches the parsed book as `library/<id>.json`.
  - Also provides `list()`, `get(id)`, `chapter(id, n)` (paragraphs split on `\n\n`), `set_position()` and `delete()`.
  - The index lives in `library/index.json`, written atomically (tmp file, then `os.replace`), behind a lock.

### 4. Reader server: new `vibetts/reader_server.py`
- A stdlib `ThreadingHTTPServer` in the same style as `McpServer`: `start()`, `stop()` and `port=0` for tests. It takes a small `ReaderApi` protocol (session + library + voices), so tests can pass a fake.
- It serves static files from `Path(__file__).parent / "web"`, rejecting any path that escapes that folder.
- **Config** (`config.py`): new `ReaderConfig` dataclass, section `[reader]`:
  - `enabled=true`, `port=7724`, `library_folder="library"`, `max_chars=500000`;
  - `extension_ids=["<pinned id>"]`;
  - validation in a `_validate_reader` like the existing `_validate_*` functions;
  - add the section to `config.example.toml`;
  - add `library/` to `.gitignore`.
- **`app.py`:** `_sync_reader()`, a copy of the `_sync_mcp` (:736) pattern keyed on `(port, extension_ids, folder)`. Call it at startup and after settings are saved; stop it at quit.
- **Tray** (`tray.py`): an "Open reader" item that runs `webbrowser.open`, plus "Pause/Resume" while speaking.
- **Settings** (`settings_ui.py`): a new `"reader"` tab in `TABS` (:32), built by `_build_reader`:
  - enable, port, library folder plus "Open folder";
  - "Open reader" button and server status;
  - extension IDs entry;
  - "Open extension folder" button and a 3-step load-unpacked hint.
  - New `SettingsHooks` fields go through `_settings_window` (`app.py:792`).

### 5. Web reader: new `vibetts/web/`
Vanilla HTML, CSS and ES modules with no build step and no CDN: `index.html`, `app.js`, `api.js`, `segment.js`, `style.css`.
- **Library view:** upload via drag-and-drop or the file picker (epub, txt and the other supported types). It shows a book list with progress and a delete button.
- **Book view:**
  - TOC sidebar.
  - The chapter rendered as paragraphs, each split into sentence `<span data-i>` items by `segment.js` (`Intl.Segmenter`, falling back to a regex).
  - Click a sentence to start or seek there.
  - The current sentence is highlighted and scrolled into view, driven by long-polling `/api/status`.
  - At `finished`, it moves to the next chapter and continues.
  - The position (`chapter`, `item`) is saved with a debounce, and the book reopens there.
- **Controls:**
  - play/pause, previous/next sentence, previous/next chapter;
  - voice (from `/api/voices`, default "Auto"), speed, font size and light/dark theme, stored in `localStorage`;
  - keyboard: Space toggles playback, ←/→ moves by sentence.
- If `status.session` changes to a session the page didn't start, it shows "Playing elsewhere" and goes idle.

### 6. Chrome extension: new `extension/` (Manifest V3, no build step)
- **`manifest.json`:**
  - permissions `contextMenus`, `activeTab`, `scripting`, `storage`;
  - `host_permissions` for `http://127.0.0.1:7724/*` and `http://localhost:7724/*`;
  - a `key` that pins a stable extension ID; generate it once with openssl, and put the ID in the `reader.extension_ids` default;
  - commands: Alt+Shift+R reads the selection or article, Alt+Shift+P pauses or resumes, Alt+Shift+S stops;
  - icons rendered with Pillow from the tray icon drawing.
- **`background.js`** (service worker):
  - Context menus: "Read selection", "Read article from here" and "Read whole article".
  - All `fetch` calls to the app go through here; the port is in `chrome.storage`.
  - After a speak it long-polls `/api/status` while its session is live and forwards `{item, state}` to the tab's content script.
- **`content.js`**, injected on demand with `chrome.scripting.executeScript`, so it needs no `<all_urls>` permission:
  - Find the article root: the selection's ancestor block, else `article`, `main` or `[role=main]`, else the element with the most `<p>` text.
  - Walk the visible text nodes, skipping `script`, `style`, `nav`, `aside`, `header`, `footer`, forms and hidden elements, and group them by block element.
  - Split each block into sentences with `segment.js` and build a DOM `Range` per sentence.
  - "From here" starts at the item that contains the selection's start (a plain click leaves a collapsed "Caret" selection).
  - "Read selection" segments only the selected range.
  - The highlight uses the CSS Custom Highlight API (`CSS.highlights` plus `::highlight(vibetts-current)`), which doesn't change the page DOM. It scrolls the sentence into view when it is off screen and clears it on stop or finish.
- **`segment.js`** is a byte-for-byte copy of `vibetts/web/segment.js`; a pytest checks that they match.
- **`popup.html`, `popup.js`, `popup.css`:**
  - connection status (from `/api/info`, with a "start VibeTTS" hint when offline);
  - play/pause, stop and previous/next sentence (seek);
  - "Read this page";
  - voice and speed (`chrome.storage.sync`, sent with every speak);
  - port setting.

### 7. Packaging and docs
- **`packaging/VibeTTS.spec`:** add `vibetts/web` to `datas`, and add `reading`, `library` and `reader_server` to `hiddenimports`.
- **`scripts/build.ps1`:** also zip `extension/` to `dist/VibeTTS-chrome-extension-<version>.zip`.
- **`README.md`:**
  - a "Book reader" section;
  - a "Chrome extension" section covering install (load unpacked) and usage;
  - update "Project layout";
  - an `[reader]` config reference.
- **`.claude/tickets/VTF-07-.../changelog.md`:** maintained as in earlier tickets.

### 8. Open local books from Windows (added mid-implementation at the user's request)
The user asked to open local `.epub` files from Explorer with VibeTTS as the "Open with" / default app.
- **CLI:** `vibetts --open PATH` (also `run.pyw --open PATH`, `VibeTTS.exe --open PATH`).
  - If no instance is running, the app starts normally. Once the reader server is up, it imports PATH into the library (`Library.add_path`) and opens `http://127.0.0.1:<port>/#/book/<id>` in the default browser.
  - If an instance is running (the `SingleInstance.acquire()` mutex is taken), the new process POSTs `{"path": PATH}` to `/api/open` on the configured reader port, then exits. If the reader is disabled or unreachable, it falls back to `signal_show()` and logs a warning.
- **`POST /api/open {path}`:** imports a local file and opens it in the browser on the app side. It returns the summary plus a `url`.
  - This is the only state-changing route that accepts a request with **no** `Origin` header (a local process). A browser always sends `Origin` on a cross-site POST, so a website can't use it. An allowed `Origin` is accepted too.
  - Bad path or type: 400.
- **New `vibetts/fileassoc.py`:** per-user registration under `HKCU\Software\Classes` with stdlib `winreg`, so no admin rights are needed.
  - ProgID `VibeTTS.Book`, with a friendly name and icon (the exe icon when frozen), and command `"<exe>" --open "%1"`. From source, the command is `"<pythonw.exe>" "<run.pyw>" --open "%1"`.
  - Adds `VibeTTS.Book` to `.epub\OpenWithProgids`, plus `.txt` only when the user ticks that box. It never overwrites the `.txt` default.
  - Adds `Applications\<exe name>\SupportedTypes`.
  - Calls `SHChangeNotify(SHCNE_ASSOCCHANGED)`.
  - Provides `register(exts)`, `unregister()` and `registered_exts()`. The registry writes go through an injectable writer, so tests never touch the real registry.
  - Windows 10/11 doesn't allow setting the default app silently. The UI explains this and opens `ms-settings:defaultapps`. In Explorer, choose "Open with → VibeTTS → Always".
- **Settings > Reader:**
  - "Open books with VibeTTS" with `.epub` and `.txt` check boxes and Register / Remove buttons;
  - "Choose default app…", which opens `ms-settings:defaultapps`;
  - the current registration state.
- **Tests:** `test_fileassoc.py` (commands frozen and from source; register and unregister through a fake writer). Also test `--open` argument parsing and the forwarding path with the HTTP call mocked, and `/api/open` in `test_reader_server.py` (no-Origin accepted, website Origin rejected).
- **Verification:** register from Settings, then right-click an `.epub` in Explorer → Open with → VibeTTS. The book opens in the reader, both when the app is running and when it isn't.

## Implementation slices (for the `implement` skill)

1. **Core** (sections 1–3 and tests). Do this first; the other slices build on it.
2. **Server and app wiring** (sections 4, 7 and 8: config, settings, tray, packaging, file association and tests). Can start once the contract is fixed and runs against a fake session.
3. **Web reader** (section 5), against the contract.
4. **Extension** (section 6), against the contract.

Slices 3 and 4 run in parallel once slice 2 exposes the API.

## Tests (pytest, `.venv\Scripts\python -m pytest -q`)

- **`test_textprep.py`:** `prepare_items` tags, pauses and the first-chunk rule.
- **`test_tts.py`:**
  - `on_chunk` order;
  - pause stops writes and resume continues with no samples lost;
  - stop while paused.
  - Use the existing `FakeEngine`/`FakeOutput`.
- **`test_docload.py`:** `extract_book` on EPUBs built in the test with `zipfile`:
  - nav TOC, NCX TOC and no TOC;
  - an untitled spine item merges into the previous chapter;
  - title and author.
  - TXT: chapter headings, and the "Part N" fallback.
- **New `test_library.py`:**
  - add, de-duplicate, list, chapter and position;
  - positions survive a reload;
  - atomic index write;
  - delete.
- **New `test_reading.py`:**
  - the state machine with `FakeSpeaker` from `test_app.py`;
  - item progress, seek, pause and resume;
  - replaced by a hotkey read, which ends in `stopped`;
  - long-poll wakes on a change and times out otherwise.
- **New `test_reader_server.py`** (live HTTP on port 0, like `test_mcp_server.py:168`):
  - Host and Origin 403 cases;
  - the extension id allowed;
  - website origin rejected on POST;
  - upload limit;
  - static path traversal rejected;
  - every route against a fake API.
- **`test_config.py`:** `[reader]` defaults and validation.
- **`test_settings_ui.py`:** the reader tab builds, and its hooks fire.
- **New identity test:** `extension/segment.js` equals `vibetts/web/segment.js`.

## Verification (end to end)

1. Run `.venv\Scripts\python -m pytest -q`. All tests should pass: the 381 existing ones plus the new ones.
2. Run `pythonw run.pyw --console`. The log shows the reader server on 7724, and the tray "Open reader" item opens the browser.
3. **Web reader:**
   1. Upload an EPUB with a TOC and a TXT novel. The chapters list correctly.
   2. Play: the sentence highlight follows the audio.
   3. Pause, then resume: playback continues mid-sentence.
   4. Click a later sentence: playback seeks there.
   5. Let a chapter end: the next chapter starts.
   6. Restart the app and reopen the book: it resumes at the saved position.
   7. Press the hotkey during playback: the reader shows "Playing elsewhere".
4. **Extension:** load `extension/` unpacked (you need to do this once in `chrome://extensions`; then I can drive Chrome with the claude-in-chrome tools to check it). On a Wikipedia article:
   1. Select text and choose Read selection.
   2. Click mid-article and choose Read article from here.
   3. Use Read whole article.
   4. Check that the highlight follows and scrolls.
   5. Check the popup's pause, resume, stop, seek, voice and speed controls.
   6. Close the app: the popup shows offline.
5. **Security check:** from a normal web page's DevTools console, run `fetch("http://127.0.0.1:7724/api/stop",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"})`. It must fail; the server answers 403 or the CORS preflight is rejected.
6. Run `scripts/build.ps1`. The exe serves the reader (the web files are bundled), and the extension zip is in `dist/`.

## Out of scope
- Audio played in the browser and per-tab audio.
- Reading PDFs page by page.
- Running the reader in `--mcp-only` headless mode.
- Publishing to the Chrome Web Store.
- Changes to MCP tools.

## Status (2026-10-20)

- [x] Slice 1a: playback core and reading session (`textprep.prepare_items`, `Speaker.pause/resume`, `on_chunk`, `TTSApp.pause/resume/toggle_pause`, `reading.ReadingSession`)
- [x] Slice 1b: books (`docload.extract_book`, `library.Library`)
- [x] Slice 2a: `reader_server.ReaderServer`
- [x] Slice 2b: config `[reader]`, app wiring, tray, Settings > Reader, `fileassoc`, `--open`, packaging, README
- [x] Slice 3: web reader (`vibetts/web/`), with a Stop button added after review
- [x] Slice 4: Chrome extension (`extension/`)
- [x] Section 8: Open with / `--open` (added mid-implementation at the user's request)

Verification: see `changelog.md`. The pytest suite passes. Real playback through the API and the `--open` forwarding were checked against a running dev app, and the PyInstaller build bundles the web files and the extension zip. **Not verified in a browser:** neither connected Chrome could reach 127.0.0.1:7724, so the web reader UI and the extension still need manual QA. Explorer registration (`fileassoc` against the real registry) also still needs manual QA.
