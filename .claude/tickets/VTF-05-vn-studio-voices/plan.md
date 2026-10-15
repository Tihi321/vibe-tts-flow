# VTF-05: what VN Studio needs from VibeTTS

VN Studio (C:\projects\Personal\visual-novels-orchestrator, ticket VNO-01) makes spoken lines for visual novels. It uses VibeTTS through the MCP server: OmniVoice speaks English and Croatian, and voices are designed from a description. Branch `VTF-05_vn-studio-voices` (from `origin/master`).

## Scope

- [x] `synthesize_batch` MCP tool: many short clips in one call (`items: [{text, output, voice?, lang?, speed?}]`, absolute `output` ending in `.wav` / `.mp3`, `overwrite`). The model is loaded once and stays loaded. Per-item result `{output, duration_s, ok, error}`; a failing item does not stop the others. Per-item `max_chars` check, at most 500 items per call. One export slot (`TTSApp.export_batch`), so it does not run next to the hotkey reader's exports or other exports. Progress is logged (never the text).
- [x] `delete_voice(name)` MCP tool (`default` cannot be deleted; accepts `name` or `omni:<name>`).
- [x] `list_voices` already returns `kind`, `description` and sample `lang` for OmniVoice presets (no change).
- [x] Headless mode for agents: `python -m vibetts --mcp-only [--port N]` (no tray, hotkey, overlay or window; MCP forced on; no single-instance lock; `speak` refuses). Did not exist before.
- [x] README: new tools and headless mode.
- [x] pytest tests for all of it.

## Out of scope

Anything in VN Studio (its own ticket). The installed exe at `D:\Programs\VibeTTS` is an older build: it needs a rebuild to get the new tools.
