# CLI transcript fixtures

Real output of the arms (Codex, Hermes 0.23.0, OpenClaw 2026.11.2, Claude Code), captured on 2026-11-13 against
`scripts/fake_model_server.py`: a fake OpenAI-compatible model that calls one tool (a shell command or, with
`FAKE_TOOL=wallet_info`, Connect's `wallet_info` on the bench's mock Connect) and then answers. No model money is
spent and the formats are the CLIs' own. Re-capture after bumping a CLI's pin (`scripts/install-arms.sh`), then rerun
`tests/test_transcript.py`; a capture whose shape the arms read the same way may keep the earlier file. The current
pins (Codex 0.170.0, Claude Code 2.1.327, Hermes 0.23.2, OpenClaw 2026.11.5) were captured the same way on
2026-11-18 and read into the same events and stats as these files, so they were kept.

- `codex-shell.jsonl`, `codex-mcp.jsonl`: `codex exec --json` stdout, a shell command and a Connect MCP call.
- `hermes.jsonl`: `hermes chat --oneshot -Q --format stream-json` stdout.
- `openclaw-trajectory.jsonl`: `events.jsonl` from `openclaw sessions export-trajectory` after an `agent exec`.
- `openclaw-local.json`, `openclaw-local-error.json`, `openclaw-local-2turns.jsonl` (captured 2026-11-15):
  `openclaw agent --local --session-id <id> --json` stdout for a turn with one tool call (trimmed to the fields the
  arm reads: `payloads`, `meta.agentMeta`, `meta.toolSummary`, `meta.stopReason`), its error envelope with the
  model's endpoint unreachable, and the transcript lines of the export after two turns on the same session id (the
  second turn's request carried the first's history; paths replaced).
- `claude-code.jsonl`: `claude -p ... --output-format stream-json --verbose` stdout (ANTHROPIC_BASE_URL at the fake
  server, an isolated CLAUDE_CONFIG_DIR and a fake key: nothing reaches Anthropic).
