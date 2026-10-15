# VTF-05 changelog

## 2026-10-15

- `vibetts/mcp_server.py`: new tools `synthesize_batch` and `delete_voice` (`MAX_BATCH_ITEMS = 500`); `AgentApi` has `synthesize_batch` and `delete_voice`; the server instructions mention them.
- `vibetts/app.py`:
  - `TTSApp.export_batch(reqs, progress, cancel)`: one export slot for the whole batch, per-item results, `cancelled` for the rest after a cancel.
  - `AgentBridge.synthesize_batch` / `delete_voice`: per-item validation (absolute path, `.wav` / `.mp3`, text and size limit, `lang`, `speed`, known voice); invalid items fail alone.
  - `HeadlessApp` and `run_mcp_only(cfg, port)`; CLI flags `--mcp-only` and `--port N`.
- `README.md`: Agents (MCP) table and a "Headless" paragraph.
- Tests: `tests/test_mcp_server.py` (tool list, validation of both tools), `tests/test_app.py` (batch results, text limit, busy slot, cancel, delete_voice, headless bridge, CLI flags, `run_mcp_only`).
- Live check: `python -m vibetts --mcp-only --port 7733`, `synthesize_batch` from VN Studio made mp3 clips for 5 beats with designed voices; a Croatian line with a designed voice (`lang: hr`) made a 4.2 s mp3.
