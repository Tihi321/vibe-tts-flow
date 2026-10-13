# VTF-01: Read selected text aloud (Kokoro)

Branch: `VTF-01_read-aloud` (from `master`). Not committed yet.

## Decisions (2026-10-13)

- Separate repo from VibeFlow (`vibe-whisper-flow`): VibeFlow is voice → text, this is text → voice, and text translation is planned here next. VibeFlow modules were copied (hotkeys, overlay, tray, settings patterns, config, autostart, single instance), not shared.
- Engine: Kokoro-82M via `kokoro-onnx` 0.6.1 on CPU (onnxruntime 1.31). English only was the requirement; other espeak-supported languages come for free. Japanese/Chinese voices are hidden (need extra phonemizer packages).
- Default model `int8` (92 MB): same speed as fp32 on this machine, much smaller.
- Hotkey `alt+win` (VibeFlow uses `ctrl+win`; `ctrl+alt` clashes with AltGr on the Croatian layout). Toggle: press to read, press again or Esc to stop.
- Selection via clipboard: snapshot all memory formats, Ctrl+C (Ctrl+Insert for terminal window classes, to avoid SIGINT), wait on `GetClipboardSequenceNumber`, restore. Fallback to clipboard text when nothing was copied.
- Streaming: `textprep` splits into sentences; first chunk is cut at a clause mark when ≤100 chars; producer thread synthesizes one chunk ahead; player writes 100 ms blocks so Stop is instant.
- `HotkeyListener` takes named bindings (`{"read": chord}`) so a `translate` hotkey can be added without restructuring.

## Measured on INFLAME-TIHI (Ryzen AI Max+ 395, CPU only)

- Kokoro: 6.5–8.6x real time (fp32 and int8 alike), model load ~0.75 s, warm-up synth added to preload.
- First audio 0.44 s after the request with the model loaded (`--say` CLI); selection grab 56 ms in Notepad.
- Running tray app: ~260 MB RAM.

## Verified

- 62 unit tests pass (`pytest -q`).
- `--say ... --wav` output transcribed by VibeFlow's whisper-cli word for word; markdown cleanup confirmed (no symbols, URL read as "link").
- Live playback through speakers via `--say` (first audio 0.44 s).
- Selection grab against Notepad: selected text returned, previous clipboard restored, nothing-selected falls back to clipboard.
- Tray app starts, listener + preload OK; Settings General and Voice tabs rendered and checked by screenshot.
- PyInstaller build: `dist\VibeTTS\VibeTTS.exe` synthesizes correctly (espeak DLL/data bundled), 131 MB folder / 56 MB zip.

## Not verified (needs the user)

- Pressing the real Alt+Win hotkey in apps (synthetic key events are ignored by the listener by design, so it cannot be driven from a script).
- Stop via hotkey/Esc during real reading, pill and tray icon colour while reading.
- Terminal Ctrl+Insert path (Windows Terminal, mintty), Electron apps (Slack, Discord), browsers.
- Clipboard restore of images/files and that the restore is skipped by Win+V history.
- Settings: Download button flow from an empty `models\`, voice/speed changes from the tray.

## Next

- Translation: second hotkey (`translate`) → selected text → LM Studio (OpenAI-compatible, like VibeFlow cleanup) → small window with the translation and Copy / Replace selection / Read aloud.

## 2026-10-13: in-app model download + exe build

- Settings > Voice: "Voice model not installed" panel (Download voice / progress / Cancel) shown while the active model or voices file is missing; "Open folder" button. Tray item "Download voice model…" while missing (opens Settings and starts the download). Old text banners for the missing model removed.
- `scripts/build.ps1`: one run makes `dist\VibeTTS` (ready to run, model included), the app-only zip (~56 MB) and the with-model zip (~140 MB). `-NoModel` skips the model; files cached in `build\models`. Replaced the `-BundleModel` switch.
- Verified: download flow end to end from source with an empty models folder (7.5 s, panel hides, engine reload via apply, status "Voice ready"), screenshots before/during/after; unzipped app-only exe opens on the download panel; with-model zip contains both files; `dist\VibeTTS\VibeTTS.exe` started and loaded the model; certifi `cacert.pem` bundled.
- Not verified: clicking Download inside the frozen exe (HTTPS from the frozen build), tray "Download voice model…" item.
