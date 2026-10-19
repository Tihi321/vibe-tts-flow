# VibeTTS

Read selected text aloud on Windows with a local voice. Select text in any app, press **Alt+Win**, and it is spoken right away; press again (or **Esc**) to stop. Runs [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on the CPU through ONNX Runtime: no GPU, no account, no cloud.

Sister app of [VibeFlow](https://github.com/Tihi321/vibe-whisper-flow) (voice → text). VibeTTS goes the other way (text → voice); text translation is planned next.

## Features

- System-wide hotkey: reads the current selection, falls back to the clipboard when nothing is selected
- Streams sentence by sentence: the first audio plays about 0.4 s after the text is grabbed, the rest is generated while it speaks (Kokoro runs ~8x faster than real time on a Ryzen AI Max+ 395)
- 41 voices: American and British English, plus Spanish, French, Hindi, Italian, Brazilian Portuguese
- Speed, pause between sentences, volume, output device
- Export to an audio file (WAV or MP3): type or paste text, or open a .txt, .md, .html, .docx, .epub or .pdf document (Settings > Export)
- Local MCP server so AI agents (Claude Code and others) can speak text or turn text and documents into audio files (off by default)
- Cleans markdown before reading (no "asterisk asterisk", web addresses become "link")
- Restores your clipboard after copying the selection (text, images, files, rich text)
- Status pill while reading, tray icon with voice and speed menus, Settings window
- Portable Windows build (`VibeTTS.exe`): no Python needed on the target machine

## Requirements

- Windows 10 / 11, Python 3.11+ (to run from source)
- About 120 MB for the int8 model and voices, downloaded from the Settings window

## Install

```powershell
git clone git@github.com:Tihi321/vibe-tts-flow.git
cd vibe-tts-flow
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# model + voices into .\models (or use the Download button in Settings > Kokoro)
python -m vibetts.model_setup            # add --model fp32 for the larger model
```

## Run

```powershell
pythonw run.pyw            # background, no console window
python run.pyw --console   # prints the log to the terminal
python -m vibetts --say "Hello there"                 # speak once, no tray
python -m vibetts --say "Hello there" --out out.mp3   # write an audio file instead (.wav or .mp3; --wav still works)
python -m vibetts --doc C:\path\to\book.epub --out book.mp3   # a whole document to an audio file
```

A speaker icon appears in the tray. Left-click it for Settings, right-click for the menu (voice, speed, stop, launch at login, quit). `--hidden` starts without opening the Settings window (used by launch at login). The first start opens Settings on the Kokoro tab when the model is missing: click **Download voice** (see below).

## Export to an audio file

Open **Settings > Export** (or tray > **Export audio…**). Type or paste text, or press **Open document…** to load a `.txt`, `.md`, `.html`, `.docx`, `.epub` or `.pdf` file. Only text-based PDFs work: scanned pages have no text to read. Pick a voice and speed. Then press **Export…** and choose where to save.

The file name's extension picks the format:

- `.mp3`: the default, 64 kbps mono, about 0.5 MB per minute.
- `.wav`: about 2.9 MB per minute.

The format, MP3 bitrate and default folder (`~\Documents\VibeTTS`) are saved in `[export]` in `config.toml`.

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
| `synthesize_to_file` | Turns `text` or a document `path` into a `.wav` / `.mp3` file. Optional: `output`, `format`, `voice`, `speed`, `overwrite`. Returns the path, duration and size |
| `speak` | Reads text aloud on your speakers, like the hotkey, and replaces what is playing. With `wait = true` it returns when it has finished |
| `synthesize_batch` | Many short clips in one call: `items` = `[{text, output, voice?, speed?}]` (absolute `output` paths ending in `.wav` / `.mp3`), optional `overwrite`. The model is loaded once and stays loaded; up to 500 items; runs in the same one-at-a-time export queue (it pauses while the app reads aloud). Returns one `{output, duration_s, ok, error}` per item; a failing item does not stop the others |
| `list_voices` | Voice ids with readable labels, and the current default voice |
| `stop` | Stops reading; `cancel_export = true` also cancels a running export |

**Security.** The server is off by default.

- **Who can reach it.** It listens on `127.0.0.1` only, so other computers cannot reach it. Requests whose `Host` or `Origin` is not localhost are refused, which stops web pages from calling it through DNS rebinding.
- **No password.** Any program on this computer can use it while it is on.
- **Where files go.** An `output` path can be absolute; a bare file name goes into the export folder. Only `.wav` and `.mp3` are written, and an existing file is only replaced with `overwrite = true`.
- **Size limit.** `[mcp] max_chars` (default 200000) limits the text of one call.

## How to use

| Action | Result |
|---|---|
| Select text, press **Alt+Win** | Reads the selection aloud |
| Press **Alt+Win** with nothing selected | Reads the clipboard text (turn off in Settings) |
| **Alt+Win** or **Esc** while reading | Stops |
| Tray > **Voice** / **Speed** | Change the voice or speed (applies to the next read) |

Pill colours: blue animated dots = starting / loading the model, violet = reading, green = done, red text = error.

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

## Voices and models

Pick a voice in the tray or Settings > Voice. `af_heart` (American, female) is the default and the best rated; `af_bella` and `bf_emma` are other well-rated English voices, `am_michael` and `bm_george` are good male ones. Japanese and Chinese voices exist in the voice file but need extra phonemizer packages, so they are not offered.

| Model | Size | Speed | Notes |
|---|---|---|---|
| `int8` (default) | 92 MB | ~8x real time | Same speed, nearly the same sound |
| `fp32` | 326 MB | ~8x real time | Original weights |

Files come from the [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) release `model-files-v1.0` and live in `models\` next to the app. `preload = true` loads the model at startup (about 1 s; the running app uses about 260 MB of RAM) so the first read starts at once.

## Portable build (exe)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1             # -SkipTests to skip pytest
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -NoModel    # app-only zip, no model download
```

One run produces:

| Output | Size | Contents |
|---|---|---|
| `dist\VibeTTS\VibeTTS.exe` | | Ready to run, model included |
| `dist\VibeTTS-<version>-win64.zip` | ~56 MB | App only: the first start offers **Download voice** (about 121 MB) |
| `dist\VibeTTS-<version>-win64-with-model.zip` | ~140 MB | App + int8 model + voices, works offline |

The model files are downloaded once into `build\models` and reused by later builds. Unzip anywhere and run `VibeTTS.exe`. The exe is unsigned, so SmartScreen may warn: **More info**, then **Run anyway**. All data lives next to the exe: `config.toml`, `logs\`, `models\`. Each build deletes and recreates `dist\VibeTTS`, so copy the folder elsewhere before using it day to day.

## Getting the voice model from the app

When the model is missing, the Settings window opens on the **Kokoro** tab with a **Voice model not installed** panel: click **Download voice** (progress bar, Cancel). The tray menu also shows **Download voice model…** until it is installed. The **Model files** list on the same tab downloads or deletes single files (int8, fp32, voices) and **Open folder** shows the models folder. After a download the model loads straight away; no restart is needed.

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
| Slow first read | Turn on "Load the model at startup" |

## Privacy

Everything runs locally. Selected text is never written to disk (only its length goes to the log). Export writes only the audio file you choose; the text is not stored. The MCP server is off by default and only listens on 127.0.0.1. No telemetry.

## Project layout

```
run.pyw                  launcher (used by autostart)
config.example.toml      copied to config.toml on first run
scripts/build.ps1        builds the portable exe + zip
packaging/VibeTTS.spec   PyInstaller spec
vibetts/
  app.py                 wiring: hotkey -> selection -> speech; tray and settings hooks; export, MCP bridge; --say/--doc CLI
  export.py              streaming export to WAV / MP3 (lameenc)
  docload.py             text from .txt .md .html .docx .epub .pdf
  mcp_server.py          local MCP server (stdlib HTTP, JSON-RPC) and its tools
  hotkeys.py             global hotkey listener (named bindings), chord parsing
  selection.py           copy the selection, snapshot and restore the clipboard
  textprep.py            markdown cleanup, sentence chunking
  tts.py                 Kokoro engine, streaming speaker, audio output, voice labels
  model_setup.py         model + voices downloader (also a CLI)
  overlay.py             status pill (tkinter)
  tray.py                tray icon and menu
  settings_ui.py         Settings window (tkinter + sv-ttk)
  config.py              config handling, frozen-aware paths
  autostart.py           Run-key launch at login
  single_instance.py     one running copy; second launch shows the window
tests/
```

## License

MIT (placeholder). Kokoro-82M weights: Apache 2.0.
