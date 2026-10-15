# VibeTTS

Read selected text aloud on Windows with a local voice. Select text in any app, press **Alt+Win**, and it is spoken right away; press again (or **Esc**) to stop. Runs [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on the CPU through ONNX Runtime: no GPU, no account, no cloud.

Sister app of [VibeFlow](https://github.com/Tihi321/vibe-whisper-flow) (voice → text). VibeTTS goes the other way (text → voice); text translation is planned next.

## Features

- System-wide hotkey: reads the current selection, falls back to the clipboard when nothing is selected
- Streams sentence by sentence: the first audio plays about 0.4 s after the text is grabbed, the rest is generated while it speaks (Kokoro runs ~8x faster than real time on a Ryzen AI Max+ 395)
- 41 voices: American and British English, plus Spanish, French, Hindi, Italian, Brazilian Portuguese
- Optional second engine, [OmniVoice](https://github.com/k2-fsa/OmniVoice) (CPU, installed from Settings, non-commercial weights): English, Croatian and more, custom voices from a description or a recording; text that looks Croatian is read as Croatian automatically
- Speed, pause between sentences, volume, output device
- Cleans markdown before reading (no "asterisk asterisk", web addresses become "link")
- Restores your clipboard after copying the selection (text, images, files, rich text)
- Status pill while reading, tray icon with engine, voice and speed menus, Settings window
- Portable Windows build (`VibeTTS.exe`): no Python needed on the target machine

## Requirements

- Windows 10 / 11, Python 3.11+ (to run from source)
- About 120 MB for the int8 model and voices, downloaded from the Settings window
- Optional, for OmniVoice: a one-time 2.9 GB download (3.5 GB with voice cloning), about 1.3 GB on disk, installed from Settings > OmniVoice

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

## OmniVoice engine (English, Croatian and more)

Kokoro is the default engine: small, fast, English plus a few other languages, but no Croatian. **OmniVoice** ([k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)) is an optional second engine. It is multilingual (including Croatian and English), supports custom voices, and runs on the CPU through ONNX Runtime. It is slower than Kokoro, about 1.4x faster than real time with the default settings on a 16-core Ryzen AI Max+ 395, and it is off until you install it.

**License.** The OmniVoice code is Apache-2.0, but the model weights are **CC-BY-NC**: non-commercial use only.

**Install it from the app.** Open **Settings > OmniVoice** and press **Install**. VibeTTS downloads the model once (about 2.9 GB, or 3.5 GB with voice cloning, from huggingface.co, with a progress bar and **Cancel**), converts it on your computer and keeps about 1.3 GB on disk, in `models\omnivoice\`. No Python, torch or other tools are needed. Tick **Include voice cloning** (+654 MB) if you want to clone voices from recordings; you can add it later. **Uninstall** removes the model files and keeps your voices. Everything else in the tab (voices, Create voice) is greyed out until OmniVoice is installed.

**Choose the engine.** Settings > Voice has an **Engine** choice, Kokoro or OmniVoice (OmniVoice is disabled until installed), and a voice list for the selected engine. The tray has an **Engine** submenu and a **Voice** menu grouped by Kokoro language plus **OmniVoice**; picking an OmniVoice voice there switches the engine. Settings are `engine = "kokoro" | "omnivoice"`, `voice` (Kokoro voice) and `omni_voice` (an `omni:<preset>` voice) in `config.toml`.

**English and Croatian.** Each request is checked once, before it is read: text that looks Croatian is read as Croatian, anything else as English, and the same decision picks the word spoken for web addresses ("poveznica" or "link").

| Engine | Croatian text (auto language on, OmniVoice installed) | Other text |
|---|---|---|
| Kokoro | the Croatian voice, language `hr` | the Kokoro `voice` |
| OmniVoice | the Croatian voice, language `hr` | `omni_voice`, language `en` |

The **Croatian voice** (Settings > Voice > Croatian text, `croatian_voice`) defaults to "Same as OmniVoice voice". With `auto_language = true` (the checkbox there), Croatian text is read with OmniVoice even when Kokoro is the engine; turn it off to keep Croatian text on the selected engine. Text counts as Croatian when it has Croatian letters (č ć đ š ž) together with Croatian function words, or enough common Croatian words without diacritics; short or mixed text stays with the normal voice. If OmniVoice is not installed, everything is read with the Kokoro voice instead (logged once).

**Speed and quality.** The voice preset adds a fixed cost per chunk, so OmniVoice requests are read in longer chunks after the first sentence. Slower CPUs are slower, and a CPU much slower than the one above can fall behind the speech, causing gaps between chunks. The first sentence starts after roughly 2-3 s. **Quality steps** (OmniVoice tab, `omnivoice_steps`, 4..16) trades speed for quality: fewer steps are faster and rougher, 16 sounds better but is roughly 2.5x slower than the default 6.

## Voices (OmniVoice tab)

Left alone, the model picks a different speaker every time, so a **voice** pins one: a short clip stored in `models\omnivoice\voices\<name>.*` and used as the speaker prompt for every sentence. The **Voices** list in Settings > OmniVoice shows each voice's name, kind (`default`, `description`, `recording`, `random`), description or source file, and the language of its sample. Select one and press **Play sample**, **Use as OmniVoice voice**, **Use for Croatian** or **Delete** (the `default` voice cannot be deleted; deleting a voice that is in use falls back to `omni:default`).

**Create voice** (below the list) makes a new one. Type a name (letters, digits, `_` and `-`, up to 32 characters; `default` is reserved), pick a tab, make the voice, press **Play** to listen and **Save**. **Use as OmniVoice voice after saving** is on by default. The new voice appears at once in the lists, the Voice combos and the tray. Saving under an existing name asks before replacing it. A custom voice reads exactly as fast as the `default` voice (the cloning encoders are only used while the voice is created). The first voice you create loads the model if it is not loaded yet.

- **From description.** One dropdown per category; leave a dropdown on `(any)` to leave it out, and leave all of them on `(any)` for a plain random voice. Vocabulary: gender `male` / `female`; age `child` / `teenager` / `young adult` / `middle-aged` / `elderly`; pitch `very low pitch` / `low pitch` / `moderate pitch` / `high pitch` / `very high pitch`; style `whisper`. **Sample language** (Croatian or English) is the language of the short clip the voice is made from. The seed picks which voice you get for a description: **Try another** rolls a new seed and generates, **Generate** uses the seed in the box. Takes about 5 s.
- **From recording.** Choose a WAV file of 3-8 s of clear speech and type its transcript, which must match what is said exactly (a wrong transcript gives a poor voice). Longer clips are refused: trim them to the part you type. Cloning needs the cloning files: if the tab says cloning is not installed, tick **Include voice cloning** and install it. **Only clone voices you have permission to use.**

**From the command line** (app's venv, OmniVoice installed):

```powershell
python -m vibetts.omnivoice make-voice ana --seed 7      # a random voice; play models\omnivoice\voices\ana.wav
python -m vibetts.omnivoice design-voice ana --instruct "female, young adult, low pitch" --seed 7
python -m vibetts.omnivoice design-voice ana --instruct "male, low pitch" --lang en   # English sample
python -m vibetts.omnivoice clone-voice marko --wav C:\path\to\me.wav --text "Dobar dan, ovo je moj glas."
python -m vibetts.omnivoice say "Hello there" --voice omni:ana --lang en
```

Each voice command writes `models\omnivoice\voices\<name>.*` and prints the voice id (`omni:<name>`), which you can set as `omni_voice` or `croatian_voice` in `config.toml`. Old configs that used `omni_hr:<name>` are migrated automatically (an OmniVoice `voice` becomes `engine = "omnivoice"` plus `omni_voice`).

## Run

```powershell
pythonw run.pyw            # background, no console window
python run.pyw --console   # prints the log to the terminal
python -m vibetts --say "Hello there"                 # speak once, no tray
python -m vibetts --say "Hello there" --wav out.wav   # write a WAV instead
```

A speaker icon appears in the tray. Left-click it for Settings, right-click for the menu (engine, voice, speed, stop, launch at login, quit). `--hidden` starts without opening the Settings window (used by launch at login). The first start opens Settings on the Kokoro tab when the model is missing: click **Download voice** (see below).

## How to use

| Action | Result |
|---|---|
| Select text, press **Alt+Win** | Reads the selection aloud |
| Press **Alt+Win** with nothing selected | Reads the clipboard text (turn off in Settings) |
| **Alt+Win** or **Esc** while reading | Stops |
| Tray > **Engine** / **Voice** / **Speed** | Change the engine, voice or speed (applies to the next read) |

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

Pick a voice in the tray or Settings > Voice (Kokoro voices when the engine is Kokoro). `af_heart` (American, female) is the default and the best rated; `af_bella` and `bf_emma` are other well-rated English voices, `am_michael` and `bm_george` are good male ones. Japanese and Chinese voices exist in the voice file but need extra phonemizer packages, so they are not offered.

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

## Developer: building the OmniVoice files with torch

End users never need this: the app installs OmniVoice itself (Settings > OmniVoice). `scripts/build_omnivoice.py` is the developer path. It uses torch to export the int4 language-model graph that ships with the app (`--export-graph`). Run it in its own throw-away venv, not the app's:

```powershell
python -m venv .venv-build
.venv-build\Scripts\activate
pip install -r scripts\omnivoice-build-requirements.txt
python scripts\build_omnivoice.py            # see --help for --export-graph and the other options
```

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
| Croatian text is read with the English voice | OmniVoice is not installed (Settings > OmniVoice shows it): press **Install**; also check "Switch to OmniVoice (Croatian)" is on in Settings > Voice |
| OmniVoice install fails or stops | Check the connection and press **Install** again; the error is shown and logged in `logs\vibetts.log` |
| OmniVoice speech has gaps between sentences | The CPU is too slow for the step count: lower Quality steps (min 4) |

## Privacy

Everything runs locally. Selected text is never written to disk (only its length goes to the log). No telemetry.

## Project layout

```
run.pyw                  launcher (used by autostart)
config.example.toml      copied to config.toml on first run
scripts/build.ps1        builds the portable exe + zip
scripts/build_omnivoice.py  developer tool: torch path that exports the OmniVoice graph (end users install from Settings)
packaging/VibeTTS.spec   PyInstaller spec
vibetts/
  app.py                 wiring: hotkey -> selection -> speech; tray and settings hooks; --say CLI
  hotkeys.py             global hotkey listener (named bindings), chord parsing
  selection.py           copy the selection, snapshot and restore the clipboard
  textprep.py            markdown cleanup, sentence chunking
  tts.py                 Kokoro engine, streaming speaker, audio output, voice labels
  omnivoice.py           OmniVoice ONNX engine (any language), voice presets
  omnivoice_setup.py     in-app OmniVoice installer (download + convert, no torch)
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

MIT (placeholder). Kokoro-82M weights: Apache 2.0. OmniVoice code: Apache 2.0, weights: CC-BY-NC (non-commercial).
