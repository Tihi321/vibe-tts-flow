# VTF-07 changelog

## 2026-10-20: EPUB/TXT web reader, Chrome "read this page" extension, Windows "Open with"

VibeTTS now serves a local book reader at `http://127.0.0.1:7724/`. The reader has chapters, a highlight that follows the sentence being read, pause/resume and a saved position. A Chrome extension sends web page text to the app to be read aloud and highlights each sentence on the page. Book files can be opened from Explorer with `--open`, and Settings can register VibeTTS under "Open with". The PC speakers play all audio through the existing `Speaker`.

Repo `vibe-tts-flow`, branch `VTF-07_book-reader-extension` (from `origin/master` 4bfeba7).

### Files

**Playback core**
- `vibetts/textprep.py`:
  - `Chunk.tag`.
  - `prepare_items()`: one list of chunks from (text, pause) items. Each chunk is tagged with its item index. A sentence or paragraph pause follows each item's last chunk.
  - `split_chunks(short_first=...)`.
- `vibetts/tts.py`:
  - `SpeakRequest.on_chunk`: called when a chunk's audio starts.
  - `Speaker.pause()/resume()/paused`. Pausing releases the audio device and keeps the remaining samples; resuming reopens the device. Each request gets its own pause gate, and stopping or starting a new speak releases a pause.
- `vibetts/app.py`, playback part:
  - `TTSApp.speak(speed=, on_chunk=)`.
  - `pause()`, `resume()`, `toggle_pause()`. The pill shows "Paused".
  - `TTSApp.reading`.
- `vibetts/reading.py` (new): `ReadingSession`.
  - One reading session at a time, with states idle, loading, playing, paused, finished, stopped and error.
  - Reports the item currently playing.
  - Seeking keeps the same session.
  - A version counter backs the long-poll `wait()`.
  - If a hotkey read or MCP `speak` takes over playback, the session ends as `stopped`.
  - Validates items, ranges, speed, language and the length limit.

**Books**
- `vibetts/docload.py`: `Chapter`, `Book`, `extract_book()`.
  - EPUB:
    - title and author from the OPF;
    - chapter titles from the EPUB3 nav or EPUB2 NCX;
    - untitled spine items merge into the previous chapter;
    - "Front matter" and "Section N" as fallbacks.
  - TXT:
    - chapter/part/numeral headings;
    - an "Opening" chapter for text before the first heading;
    - a "Part N" split when there are too few headings.
  - Other types become one chapter.
  - `extract_text` behaves as before; it now shares the spine walk and the file checks.
- `vibetts/library.py` (new): `Library`.
  - Books are stored by sha256 id, and the same file uploaded twice is stored once.
  - A parsed-book cache, plus `index.json` with the position and the added and opened dates. Writes are atomic and behind a lock.
  - Book ids are validated before any file access.
  - `add_path` reads a local file.

**Server and wiring**
- `vibetts/reader_server.py` (new): stdlib HTTP server on 127.0.0.1.
  - Serves the static web reader.
  - JSON API: info, voices, speak, pause, resume, stop, seek, long-poll status, books CRUD, position, open.
  - Checks:
    - `Host` must be 127.0.0.1 or localhost on the reader port.
    - The `Origin` allowlist is the reader's own origin plus the pinned extension ids.
    - POST, PUT and DELETE need an allowed `Origin`. `/api/open` also accepts a request with no `Origin` (a local process).
    - Body size limits, and safe static paths.
  - No CORS headers. It never logs text.
- `vibetts/config.py`:
  - `ReaderConfig` (`[reader]`: enabled, port 7724, library_folder, max_chars, extension_ids) with validation.
  - List-of-string values now round-trip through config load and save.
- `config.example.toml`: new `[reader]` section. `.gitignore`: `library/`.
- `vibetts/app.py`, wiring part:
  - `ReaderBridge`.
  - `_sync_reader()` on start and on settings save; the server stops when the app quits.
  - `--open PATH`:
    - On first launch it imports the book and opens the reader at it, without popping the Settings window.
    - When another copy is running it uses `forward_open()` to POST `/api/open` to that copy, falling back to showing that copy's Settings window.
  - The new Settings and tray hooks.
- `vibetts/tray.py`: "Open reader" (shown while the server is running) and "Pause"/"Resume".
- `vibetts/settings_ui.py`: new Reader tab:
  - server settings and status;
  - Open reader, open library folder, extension ids;
  - extension folder plus the load-unpacked steps;
  - "Open books with VibeTTS": `.epub`/`.txt` Register and Remove, and "Choose default app…".
- `vibetts/fileassoc.py` (new): per-user `HKCU\Software\Classes` registration.
  - ProgID `VibeTTS.Book`, added to `OpenWithProgids`.
  - The `.txt`/`.epub` default value is never written.
  - Writes `Applications\VibeTTS.exe` when frozen, and calls `SHChangeNotify`.
  - Provides `register`, `unregister`, `registered_exts` and `open_default_apps`.

**Clients**
- `vibetts/web/` (new): the web reader. It is vanilla JS with no build step or CDN.
  - Library: upload by drag-and-drop or file picker, delete.
  - Reader:
    - chapter list, sentences as clickable spans;
    - a highlight that follows status via long-poll, then auto-advance to the next chapter;
    - "Playing elsewhere" when another source has taken over playback;
    - debounced saving of the position.
  - Controls: play/pause, Stop, previous/next sentence and chapter, voice, language, speed, font size, theme.
  - Keyboard: Space, ←/→, Esc.
  - `segment.js` is the shared sentence splitter: `Intl.Segmenter` with a regex fallback, and abbreviation joining ("Dr.", "e.g.", initials).
- `extension/` (new), Manifest V3:
  - The pinned `key` gives id `injaiiefkhoimpkeandbjfbnnjddaceh`.
  - Permissions are `activeTab` and `scripting`, with host permissions for 127.0.0.1 and localhost only.
  - Context menus: Read selection, Read article from here, Read whole article.
  - Shortcuts: Alt+Shift+R/P/S.
  - Service worker: calls the API and keeps a resumable long-poll.
  - Content script: finds the article, builds a DOM `Range` per sentence, and highlights with the CSS Custom Highlight API.
  - Popup: status, play/pause/stop/seek, voice, language, speed, port.
  - Icons are drawn from the tray icon.

**Packaging and docs**
- `packaging/VibeTTS.spec`: bundles `vibetts/web` and adds the hidden imports for the new modules.
- `scripts/build.ps1`: copies `extension/` next to the exe and builds `dist\VibeTTS-chrome-extension-<version>.zip`.
- `README.md`: new sections for the book reader, opening books from Windows, the Chrome extension and the reader settings; updated features, usage, build outputs and project layout.

**Tests**
- New:
  - `test_reading.py`
  - `test_library.py`
  - `test_reader_server.py`
  - `test_fileassoc.py`
  - `test_extension.py`: manifest checks, the two `segment.js` copies identical, `node --check`.
- Extended: `test_textprep.py`, `test_tts.py`, `test_docload.py`, `test_app.py`, `test_config.py`, `test_settings_ui.py`.

### Deviations from the plan
- **Windows "Open with" and `--open`** (plan section 8): added mid-implementation at the user's request.
- **`segment.js` is a classic script**: it sets `globalThis.VibeTTSSegment` instead of being an ES module, because content scripts injected with `chrome.scripting` cannot import.
- **Host permissions without a port**: the extension uses `http://127.0.0.1/*` and `http://localhost/*`, so a changed port keeps working.
- **Web reader Stop button**: added after review.
- **Adopting a session on reload**: the web reader picks up an active session for the same book when the page reloads.
- **Changing voice or speed while playing** restarts from the current sentence.
- **Extra session validation**: `ReadingSession.start` also rejects an unknown `source`, a non-object `meta`, and a Kokoro voice with a non-English `lang` (400).
- **`--open` conflicts**: `--open` cannot be combined with `--mcp-only`, `--say` or `--doc`.

### Verification
- `.venv\Scripts\python -m pytest -q`: **584 passed, 1 skipped**. The skip is the symlink-escape static test, because Windows does not allow creating symlinks here.
- `node --check` passes on all web and extension JS. `segment.js` was smoke-tested in Node: abbreviations, initials and Croatian quotes.
- Live dev app (temporary config, reader on 7724):
  - uploaded an EPUB through the API: title, author and 2 chapters from the nav TOC;
  - real Kokoro playback: status went loading → playing, items 1→4, then finished; pause/resume mid-sentence worked;
  - an out-of-range seek returned 400.
  - Security:
    - website Origin POST: 403;
    - POST with no Origin: 403;
    - DNS-rebinding Host: 403;
    - index page: 200.
  - `--open` handoff: a second launch POSTed `/api/open` to the running copy and exited 0; the same file was not added twice.
- `scripts\build.ps1 -SkipTests -NoModel` succeeded. `dist\VibeTTS\_internal\vibetts\web\*` and `dist\VibeTTS\extension\` are present, and `VibeTTS-chrome-extension-0.1.0.zip` contains the extension files.

**Not run (needs manual QA):**
- **The web reader UI in a browser.** Neither connected Chrome could load 127.0.0.1:7724; their tabs showed an error page while the server answered locally.
- **The Chrome extension loaded unpacked.** Not done: needs a local Chrome.
- **`fileassoc` against the real registry**, and Explorer "Open with" after registering.
- **The frozen exe run end to end.** Not done: the installed VibeTTS.exe was running and holds the single-instance mutex.

### Side effect during verification
Starting the dev app ran the existing `autostart.sync()`. It rewrote `HKCU\...\Run\VibeTTS` from the installed exe (`"D:\Programs\VibeTTS\VibeTTS.exe" --hidden`) to the dev `pythonw.exe run.pyw --hidden`. Restoring it was blocked by the permission classifier, so the user needs to restore it.

### Follow-ups / known limitations
- Upload shows a busy state, not a progress percentage.
- The extension cannot run on `chrome://` pages or the Web Store.
- The extension's highlight scrolling may not work inside nested scroll containers.
- Chrome may refuse a shortcut that another extension already holds.
- Windows does not let an app make itself the default; the user picks it under "Open with → Always".

Commit status: **uncommitted** on `VTF-07_book-reader-extension`.
