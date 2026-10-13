# VibeTTS

Read selected text aloud on Windows with a local voice. Select text in any app, press **Alt+Win**, and it is spoken right away; press again (or **Esc**) to stop. Runs [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on the CPU through ONNX Runtime: no GPU, no account, no cloud.

Sister app of [VibeFlow](https://github.com/Tihi321/vibe-whisper-flow) (voice → text). VibeTTS goes the other way (text → voice); text translation is planned next.

## Features

- System-wide hotkey: reads the current selection, falls back to the clipboard when nothing is selected
- Streams sentence by sentence: the first audio plays about 0.4 s after the text is grabbed, the rest is generated while it speaks (Kokoro runs ~8x faster than real time on a Ryzen AI Max+ 395)
- 41 voices: American and British English, plus Spanish, French, Hindi, Italian, Brazilian Portuguese
- Speed, pause between sentences, volume, output device
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

# model + voices into .\models (or use the Download button in Settings > Voice)
python -m vibetts.model_setup            # add --model fp32 for the larger model
```

## Run

```powershell
pythonw run.pyw            # background, no console window
python run.pyw --console   # prints the log to the terminal
python -m vibetts --say "Hello there"                 # speak once, no tray
python -m vibetts --say "Hello there" --wav out.wav   # write a WAV instead
```

A speaker icon appears in the tray. Left-click it for Settings, right-click for the menu (voice, speed, stop, launch at login, quit). `--hidden` starts without opening the Settings window (used by launch at login). The first start opens Settings on the Voice tab when the model is missing: click **Download voice** (see below).

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

When the model is missing, the Settings window opens on the **Voice** tab with a **Voice model not installed** panel: click **Download voice** (progress bar, Cancel). The tray menu also shows **Download voice model…** until it is installed. The **Model files** list on the same tab downloads or deletes single files (int8, fp32, voices) and **Open folder** shows the models folder. After a download the model loads straight away; no restart is needed.

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
| "Download the voice model" | Settings > Voice > Download, or `python -m vibetts.model_setup` |
| Slow first read | Turn on "Load the model at startup" |

## Privacy

Everything runs locally. Selected text is never written to disk (only its length goes to the log). No telemetry.

## Project layout

```
run.pyw                  launcher (used by autostart)
config.example.toml      copied to config.toml on first run
scripts/build.ps1        builds the portable exe + zip
packaging/VibeTTS.spec   PyInstaller spec
vibetts/
  app.py                 wiring: hotkey -> selection -> speech; tray and settings hooks; --say CLI
  hotkeys.py             global hotkey listener (named bindings), chord parsing
  selection.py           copy the selection, snapshot and restore the clipboard
  textprep.py            markdown cleanup, sentence chunking
  tts.py                 Kokoro engine, streaming speaker, audio output
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
