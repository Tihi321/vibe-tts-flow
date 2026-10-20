# VibeTTS

Read selected text aloud on Windows with a local voice. Select text in any app, press **Alt+Win**, and it is spoken right away; press again (or **Esc**) to stop. Runs [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (English) and [Supertonic 3](https://huggingface.co/Supertone/supertonic-3) (31 languages, including Croatian) on the CPU through ONNX Runtime: no GPU, no account, no cloud.

Sister app of [VibeFlow](https://github.com/Tihi321/vibe-whisper-flow) (voice → text). VibeTTS goes the other way (text → voice); text translation is planned next.

## Features

- System-wide hotkey: reads the current selection, falls back to the clipboard when nothing is selected
- Streams sentence by sentence: the first audio plays about 0.4 s after the text is grabbed, the rest is generated while it speaks (Kokoro runs ~8x faster than real time on a Ryzen AI Max+ 395)
- Two engines, two profiles: **Kokoro** (English, 28 American and British voices) and **Supertonic 3** (31 languages, 10 voices F1..F5 / M1..M5). Text that looks English is read with your English profile, anything else with your other-language profile (default Croatian); a tray switch forces one of them
- All audio is 44.1 kHz mono (Supertonic natively, Kokoro resampled up from 24 kHz)
- Speed, pause between sentences, volume, output device
- Export to an audio file (WAV or MP3, 44.1 kHz): type or paste text, or open a .txt, .md, .html, .docx, .epub or .pdf document (Settings > Export)
- Book reader in your browser (.epub and .txt): chapters, the sentence being read is highlighted, your place is remembered; a Chrome extension reads web pages aloud the same way
- Local MCP server so AI agents (Claude Code and others) can speak text or turn text and documents into audio files (off by default)
- Cleans markdown before reading (no "asterisk asterisk", web addresses become "link")
- Restores your clipboard after copying the selection (text, images, files, rich text)
- Status pill while reading, tray icon with language, voice and speed menus, Settings window
- Portable Windows build (`VibeTTS.exe`): no Python needed on the target machine

## Requirements

- Windows 10 / 11, Python 3.11+ (to run from source)
- About 120 MB for the Kokoro int8 model and voices, downloaded from the Settings window
- Optional: about 400 MB for Supertonic 3 (other languages, and English if you prefer it), downloaded from **Settings > Supertonic**

## Install

```powershell
git clone git@github.com:Tihi321/vibe-tts-flow.git
cd vibe-tts-flow
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# model + voices into .\models (or use the Download button in Settings > Kokoro)
python -m vibetts.model_setup            # add --model fp32 for the larger model
python -m vibetts.model_setup --supertonic   # optional: Supertonic 3 (other languages), ~400 MB into .\models\supertonic-3
```

## Run

```powershell
pythonw run.pyw            # background, no console window
python run.pyw --console   # prints the log to the terminal
python -m vibetts --say "Hello there"                 # speak once, no tray
python -m vibetts --say "Hello there" --out out.mp3   # write an audio file instead (.wav or .mp3; --wav still works)
python -m vibetts --doc C:\path\to\book.epub --out book.mp3   # a whole document to an audio file
python -m vibetts --open C:\path\to\book.epub        # open a book in the reader (starts the app if needed)
```

A speaker icon appears in the tray. Left-click it for Settings, right-click for the menu (language, voices, speed, stop, launch at login, quit). `--hidden` starts without opening the Settings window (used by launch at login). The first start opens Settings on the Kokoro tab when the model is missing: click **Download voice** (see below).

## Export to an audio file

Open **Settings > Export** (or tray > **Export audio…**). Type or paste text, or press **Open document…** to load a `.txt`, `.md`, `.html`, `.docx`, `.epub` or `.pdf` file. Only text-based PDFs work: scanned pages have no text to read. Pick a voice and speed. Then press **Export…** and choose where to save.

The file name's extension picks the format:

- `.mp3`: the default, 44.1 kHz mono (MPEG-1 Layer III), 64 kbps by default, about 0.5 MB per minute. Bitrates: 32, 48, 64, 96, 128, 160, 192, 256 or 320 kbps.
- `.wav`: 44.1 kHz 16-bit mono, about 5.3 MB per minute.

In the **Language** box pick `Auto` (the same routing as the hotkey), `English` or one of the 30 other languages. The format, MP3 bitrate and default folder (`~\Documents\VibeTTS`) are saved in `[export]` in `config.toml`.

The export runs in the background with a progress bar and **Cancel**; a cancelled or failed export leaves no file behind. Reading aloud has priority: if you press the hotkey while an export runs, the export pauses until the reading ends. Only one export runs at a time, whether it comes from the window or from an agent. Export ignores the playback volume.

## Agents (MCP)

VibeTTS can run a small [MCP](https://modelcontextprotocol.io) server inside the app, so AI agents can use your local voices. Turn it on in **Settings > Agents (MCP)** and press **Save**. The tab shows the status and the command to connect Claude Code:

```powershell
claude mcp add --transport http vibetts http://127.0.0.1:7723/mcp
```

The same setting as `.mcp.json`: `{"mcpServers": {"vibetts": {"type": "http", "url": "http://127.0.0.1:7723/mcp"}}}`. Any client that supports the Streamable HTTP transport works. The server only runs while VibeTTS is running.

**Headless (for scripts and agents).** `python -m vibetts --mcp-only [--port N]` serves the same MCP tools with no tray, hotkey or window, with MCP forced on (the port comes from `[mcp] port` unless `--port` is given). Stop it with Ctrl+C. It does not use the single-instance lock, so it cannot run next to the normal app on the same port; `speak` is not available there (use `synthesize_to_file` / `synthesize_batch`).

| Tool | What it does |
|---|---|
| `synthesize_to_file` | Turns `text` or a document `path` into a 44.1 kHz `.wav` / `.mp3` file. Optional: `output`, `format`, `voice`, `lang`, `speed`, `overwrite`. Returns the path, duration, size, voice and language |
| `speak` | Reads text aloud on your speakers, like the hotkey, and replaces what is playing. Optional: `voice`, `lang`, `speed`. With `wait = true` it returns when it has finished |
| `synthesize_batch` | Many short clips in one call: `items` = `[{text, output, voice?, lang?, speed?}]` (absolute `output` paths ending in `.wav` / `.mp3`), optional `overwrite`. The model is loaded once and stays loaded; up to 500 items; runs in the same one-at-a-time export queue (it pauses while the app reads aloud). Returns one `{output, duration_s, ok, error}` per item; a failing item does not stop the others |
| `list_voices` | `voices` (`id`, `label`, `engine`, `languages`), the 31 `languages` (`code`, `name`), the configured `english` and `other` profiles, `language_mode` and `supertonic_installed` |
| `stop` | Stops reading; `cancel_export = true` also cancels a running export |

`voice` takes a Kokoro id (`af_heart`, English only) or a Supertonic id (`supertonic:F1` .. `supertonic:M5`, needs the Supertonic download). `lang` is one of the 31 Supertonic codes (`en`, `hr`, `de`, `sl`, ...). Leave both out to get what the hotkey would do: English text is read with the English profile, anything else with the other-language profile. A `lang` without a `voice` picks the profile voice (`en` the English one, any other code the other-language voice in that language); a Kokoro `voice` with a non-English `lang` is an error.

**Security.** The server is off by default.

- **Who can reach it.** It listens on `127.0.0.1` only, so other computers cannot reach it. Requests whose `Host` or `Origin` is not localhost are refused, which stops web pages from calling it through DNS rebinding.
- **No password.** Any program on this computer can use it while it is on.
- **Where files go.** An `output` path can be absolute; a bare file name goes into the export folder. Only `.wav` and `.mp3` are written, and an existing file is only replaced with `overwrite = true`.
- **Size limit.** `[mcp] max_chars` (default 200000) limits the text of one call.

## Book reader

VibeTTS serves a small reader page on your own computer (`http://127.0.0.1:7724/`, on by default). The browser only shows the text; the voice plays through VibeTTS on your speakers, like the hotkey.

- **Open it:** tray > **Open reader** (or Settings > Reader > **Open reader**).
- **Add books:** drag a `.epub`, `.txt` (or `.md`, `.html`, `.docx`, `.pdf`) file onto the page, or use the file picker. Books are copied into the `library` folder next to the app (`[reader] library_folder`); the same file is only stored once.
- **Read:** the book opens with its chapter list on the left. Press play, or click a sentence to start there. The sentence being read is highlighted and scrolled into view, and the next chapter starts when one ends.
- **Resume:** the chapter and sentence are saved as you go; reopening a book continues from there.
- **Controls:** play/pause, previous/next sentence and chapter, voice (default: as configured), speed, font size, light or dark theme.
- **Keyboard:** **Space** plays or pauses, **Left** / **Right** move one sentence, **Esc** stops.
- **Pause from the tray:** tray > **Pause** / **Resume** works for any reading, including the hotkey.
- Pressing the hotkey (or an agent using `speak`) while a book is playing replaces it; the page then shows that it is playing elsewhere.

## Open books from Windows

Settings > **Reader** > **Open books with VibeTTS**: tick `.epub` (and `.txt` if you want it), press **Register**. This only adds VibeTTS to the **Open with** list for your user (no admin rights, nothing outside your own registry). Windows does not allow an app to make itself the default, so then either:

- right-click a book in Explorer > **Open with** > **VibeTTS** > **Always**, or
- press **Choose default app…** and pick VibeTTS in Windows Settings.

The book is imported and opened in the reader, whether VibeTTS was running or not (`VibeTTS.exe --open "C:\books\story.epub"` does the same from a shell). **Remove** undoes the registration. Your `.txt` default is never changed by VibeTTS.

## Chrome extension

The extension in the `extension` folder (also `dist\VibeTTS-chrome-extension-<version>.zip` after a build) reads web pages aloud through the running VibeTTS app and highlights the sentence being read.

1. Open `chrome://extensions` and turn on **Developer mode**.
2. Choose **Load unpacked** and pick the `extension` folder (Settings > Reader > **Open extension folder** opens it).
3. VibeTTS must be running with the reader turned on. The extension id is fixed (`injaiiefkhoimpkeandbjfbnnjddaceh`) and is already allowed in `[reader] extension_ids`.

Use it from the right-click menu:

| Menu item | What it reads |
|---|---|
| **Read selection** | Only the selected text |
| **Read article from here** | The article, starting at the sentence you clicked or selected |
| **Read whole article** | The whole article (the main text of the page, without menus and sidebars) |

Shortcuts (change them at `chrome://extensions/shortcuts`): **Alt+Shift+R** reads the selection or the article, **Alt+Shift+P** pauses or resumes, **Alt+Shift+S** stops. The toolbar popup shows whether VibeTTS is reachable and has play/pause, stop, previous/next sentence, **Read this page**, voice, speed and the port.

Only the extension with an id in `[reader] extension_ids` and pages served by the reader itself can call the API (see Security below).

## How to use

| Action | Result |
|---|---|
| Select text, press **Alt+Win** | Reads the selection aloud |
| Press **Alt+Win** with nothing selected | Reads the clipboard text (turn off in Settings) |
| **Alt+Win** or **Esc** while reading | Stops |
| Tray > **Language** | `Auto` (detect English), `English` (always the English profile) or the other language (always its profile) |
| Tray > **English voice** / **Croatian voice** (named after your other language) / **Speed** | Change the voices or speed (applies to the next read) |

Pill colours: blue animated dots = starting / loading the model, violet = reading, green = done, red text = error.

## Reader settings

```toml
[reader]
enabled = true               # local book reader and Chrome extension API (127.0.0.1 only)
port = 7724                  # 1024 .. 65535
library_folder = "library"   # imported books; relative paths are next to the app
max_chars = 500000           # longest text one read request may carry; 0 = no limit
extension_ids = ["injaiiefkhoimpkeandbjfbnnjddaceh"]   # Chrome extensions allowed to call the API
```

**Security.** The reader listens on `127.0.0.1` only and refuses requests whose `Host` is not localhost (DNS rebinding). A request that carries an `Origin` header must come from the reader page itself or from `chrome-extension://<id>` with an id in `extension_ids`; changing requests without an allowed `Origin` are refused, except `POST /api/open`, which a second `VibeTTS --open` launch uses (browsers always send `Origin` on cross-site posts, so a web page cannot use it). There is no password: any program on this computer can use the API while it is on. Logs record lengths and counts, never the text. Set `enabled = false` to turn it off.

## Hotkey

Edit it in Settings > General or in `config.toml`:

```toml
[hotkey]
read = "alt+win"     # other examples: "ctrl+shift+r"  "f9"  "ctrl_r+r"
```

Chords use the same names as VibeFlow: `ctrl` `alt` `shift` `win`, sides like `ctrl_r` `lwin`, keys `a`..`z`, `0`..`9`, `f1`..`f24`, `space`, `vk:<number>`. The keyboard hook is passive, so the keys still reach the focused app. Chords with `win` inject a no-op key so the Start menu does not open. Avoid `ctrl+alt` on keyboards that use AltGr (it is the same as Ctrl+Alt). The default `alt+win` does not clash with VibeFlow's `ctrl+win`.

## How the selection is read

VibeTTS waits until you release the hotkey (and any other modifier), saves the clipboard, sends a copy shortcut to the focused app, waits up to 400 ms for the clipboard to change, reads the text, then puts the old clipboard back.

- **Terminals** (Windows Terminal, conhost, Git Bash/mintty, ConEmu, PuTTY, WezTerm, Alacritty) get **Ctrl+Insert** instead of Ctrl+C, because Ctrl+C with nothing selected would interrupt the running program. The terminal panel inside an editor (VS Code, Zed) cannot be told apart from the editor, so it gets Ctrl+C: select something before pressing the hotkey there.
- **Editors** such as VS Code and Zed copy the current line when nothing is selected, so the hotkey reads that line.
- **Clipboard history (Win+V)** keeps a copy of the selected text, because the app itself does the copy. The restore of your previous clipboard is marked so it does not appear in the history again.
- If an app is slow to answer, raise `[selection].copy_timeout_ms`.

## Engines, profiles and language detection

| Engine | Languages | Voices | Size | Speed |
|---|---|---|---|---|
| Kokoro-82M | English (American, British) | 28 (`af_heart`, `bf_emma`, `am_michael`, `bm_george` ...) | 92 MB (int8) | ~8x real time |
| Supertonic 3 | 31: English, Croatian, German, Slovenian, Spanish, French, Russian, Japanese ... | 10: F1..F5, M1..M5 | ~400 MB | ~11x real time |

You set up two **profiles** in Settings > Voice (or `[speech]` in `config.toml`):

- **English profile:** engine Kokoro (default, voice `af_heart`) or Supertonic (voice F1..M5).
- **Other-language profile:** a Supertonic language (default Croatian, `hr`) and a Supertonic voice.

Every read decides once, up front: text that **looks English** uses the English profile, anything else the other-language profile. The guess is cheap and local: letters outside the Latin script or non-English diacritics (č ć š ž ä ö ü é ñ ...) mean "not English", otherwise the share of common English function words decides; very short or ambiguous text counts as English, so a Croatian name inside an English sentence does not switch voices. Croatian typed without diacritics is still caught when the text is long enough. Tray > **Language** (or Settings > Voice > Language) forces one profile: `Auto`, `English` or the other language. Without the Supertonic download, a read that needs it falls back to your Kokoro English voice, with a hint to download Supertonic.

Pick voices in the tray or Settings > Voice. `af_heart` (American, female) is the default and the best rated Kokoro voice; `af_bella` and `bf_emma` are other well-rated English voices, `am_michael` and `bm_george` are good male ones. Only English Kokoro voices are offered; other languages are Supertonic's job.

| Kokoro model | Size | Speed | Notes |
|---|---|---|---|
| `int8` (default) | 92 MB | ~8x real time | Same speed, nearly the same sound |
| `fp32` | 326 MB | ~8x real time | Original weights |

Kokoro files come from the [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) release `model-files-v1.0` and live in `models\` next to the app. `preload = true` loads the model(s) your profiles need at startup (about 1 s each; the running app uses about 260 MB of RAM) so the first read starts at once.

**Supertonic.** Download it once in **Settings > Supertonic** (about 400 MB from [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3), into `models\supertonic-3`, progress bar, resumable, Cancel) or with `python -m vibetts.model_setup --supertonic`. The same tab sets the quality (4 to 16 steps, default 8: fewer is faster, more sounds better) and deletes the files again. The Supertonic 3 model is released under the OpenRAIL-M license; read its terms before using the audio commercially.

**44.1 kHz.** Everything leaves the app at 44.1 kHz: Supertonic natively, Kokoro resampled up from 24 kHz with [soxr](https://github.com/dofuuz/python-soxr), so there is one output stream and one export format.

## Portable build (exe)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1             # -SkipTests to skip pytest
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -NoModel    # app-only zip, no model download
```

One run produces:

| Output | Size | Contents |
|---|---|---|
| `dist\VibeTTS\VibeTTS.exe` | | Ready to run, model included |
| `dist\VibeTTS-chrome-extension-<version>.zip` | small | The Chrome extension (also copied to `dist\VibeTTS\extension`) |
| `dist\VibeTTS-<version>-win64.zip` | ~56 MB | App only: the first start offers **Download voice** (about 121 MB); Supertonic is downloaded from Settings when you want it |
| `dist\VibeTTS-<version>-win64-with-model.zip` | ~140 MB | App + int8 model + voices, works offline |

The model files are downloaded once into `build\models` and reused by later builds. Unzip anywhere and run `VibeTTS.exe`. The exe is unsigned, so SmartScreen may warn: **More info**, then **Run anyway**. All data lives next to the exe: `config.toml`, `logs\`, `models\`. Each build deletes and recreates `dist\VibeTTS`, so copy the folder elsewhere before using it day to day.

## Getting the voice models from the app

Supertonic has its own **Supertonic** tab with the same download, delete and status controls (see above). When the Kokoro model is missing, the Settings window opens on the **Kokoro** tab with a **Voice model not installed** panel: click **Download voice** (progress bar, Cancel). The tray menu also shows **Download voice model…** until the engines your profiles need are installed. The **Model files** list on the same tab downloads or deletes single files (int8, fp32, voices) and **Open folder** shows the models folder. After a download the model loads straight away; no restart is needed.

## Permissions and caveats

- No special permission is needed for the keyboard hook, `SendInput` or the clipboard.
- **Elevated apps (UIPI):** the hotkey and the copy do not reach apps running as Administrator unless VibeTTS also runs elevated.
- **Antivirus / EDR** may flag keyboard hooks: allow-list `VibeTTS.exe` (or the venv's `python.exe` / `pythonw.exe`).
- **Esc** is observed passively, so it still reaches the app you are in.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Nothing selected" although text is selected | The app answered too slowly: raise `copy_timeout_ms`; or it is elevated (see above) |
| Reads the wrong thing in a terminal | Select the text first; terminals only copy a selection |
| No sound | Check Settings > Voice > Output device (blank = Windows default) and the log (`logs\vibetts.log`) |
| "Download the voice model" | Settings > Kokoro > Download voice, or `python -m vibetts.model_setup` |
| Croatian (or other) text is read by the English voice | Supertonic is not downloaded (Settings > Supertonic), or Tray > Language is set to `English` |
| English text is read by the other-language voice | The text had too few common English words to be sure (or has many accented letters): set Tray > Language to `English` |
| Slow first read | Turn on "Load the model at startup" |

## Privacy

Everything runs locally; the only network access is the optional model download from Hugging Face. Selected text is never written to disk (only its length goes to the log). Export writes only the audio file you choose; the text is not stored. The MCP server is off by default and only listens on 127.0.0.1. The book reader (on by default) also listens on 127.0.0.1 only and keeps your books and reading positions in the `library` folder next to the app. No telemetry.

## Project layout

```
run.pyw                  launcher (used by autostart)
config.example.toml      copied to config.toml on first run
scripts/build.ps1        builds the portable exe + zip
packaging/VibeTTS.spec   PyInstaller spec
vibetts/
  app.py                 wiring: hotkey -> selection -> speech; tray and settings hooks; export, MCP and reader bridges; --say/--doc/--open CLI
  export.py              streaming export to WAV / MP3 (lameenc), 44.1 kHz
  docload.py             text (and chapters) from .txt .md .html .docx .epub .pdf
  mcp_server.py          local MCP server (stdlib HTTP, JSON-RPC) and its tools
  reader_server.py       local reader API (stdlib HTTP): web reader files, playback, library, /api/open
  reading.py             reading session: items, current sentence, pause/seek, long-poll status
  library.py             book library: stored files, parsed chapters, reading positions
  fileassoc.py           "Open with VibeTTS" registration for .epub / .txt (HKCU)
  hotkeys.py             global hotkey listener (named bindings), chord parsing
  selection.py           copy the selection, snapshot and restore the clipboard
  textprep.py            markdown cleanup, sentence chunking, English-or-not guess
  tts.py                 Kokoro engine (soxr resampling to 44.1 kHz), streaming speaker, audio output, voice labels
  supertonic.py          Supertonic 3 engine (ONNX, 31 languages)
  model_setup.py         Kokoro and Supertonic model downloader (also a CLI)
  overlay.py             status pill (tkinter)
  tray.py                tray icon and menu
  settings_ui.py         Settings window (tkinter + sv-ttk)
  config.py              config handling, frozen-aware paths
  autostart.py           Run-key launch at login
  single_instance.py     one running copy; second launch shows the window
  web/                   the web reader (HTML, CSS, JS; no build step)
extension/               Chrome extension (Manifest V3, load unpacked)
tests/
```

## License

MIT (placeholder). Kokoro-82M weights: Apache 2.0. Supertonic 3 model: OpenRAIL-M (downloaded separately; the inference code in `supertonic.py` is ported from the MIT-licensed `supertonic` SDK).
