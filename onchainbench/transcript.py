"""One transcript shape for every arm: the trajectories dataset, the safety scan and readers of a trial.

Each harness reports a run its own way (formats checked against each CLI's real output, tests/fixtures/transcripts):
- Codex `exec --json`: JSONL thread events; tool work is in `item.completed` items (command_execution,
  mcp_tool_call, file_change, web_search, todo_list), the reply in agent_message, its thinking in reasoning;
- Hermes `--format stream-json`: JSONL `text` deltas, `tool_use`, `tool_result`, one `result`;
- OpenClaw: `agent --local --json` returns only the turn's reply and stats, so the arm exports the session after
  each turn (`sessions export-trajectory`, which redacts paths) and reads the turn's new lines of its `events.jsonl`;
- Claude Code `--output-format stream-json --verbose`: JSONL `assistant` messages (text, thinking and tool_use
  blocks), `user` messages carrying tool_result blocks, and a closing `result`;
- chat messages (OpenAI shape): olas-ocb-agent's session events
  (`{"type": "message", "message": ...}`), which its arm records in-process and returns from transcript().

Every event becomes {"turn", "kind", "text"} plus "tool" for tool calls and results and "is_error" where the harness
says so. kind is one of KINDS. The checks never read these: a task judges what the agent did and said (T.answer,
T.transcript), not what a tool printed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

KINDS = ("user", "assistant", "reasoning", "tool_call", "tool_result", "error")
TEXT_CAP = 20_000  # per event: a tool that prints a whole file or page shouldn't make a transcript unreadable


def event(turn: int, kind: str, text, tool: str | None = None, is_error: bool | None = None) -> dict:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    if len(text) > TEXT_CAP:
        text = text[:TEXT_CAP] + f"... [{len(text) - TEXT_CAP} more characters]"
    e = {"turn": turn, "kind": kind, "text": text}
    if tool is not None:
        e["tool"] = tool
    if is_error is not None:
        e["is_error"] = is_error
    return e


def json_lines(raw: str):
    """The JSON objects in a CLI's output, one per line; other lines skipped."""
    for line in raw.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def codex(raw: str, turn: int) -> list[dict]:
    """Codex `exec --json`. Started items are skipped: each item is reported again, complete, when it ends."""
    out = []
    for ev in json_lines(raw):
        typ = ev.get("type")
        if typ == "turn.failed":
            out.append(event(turn, "error", (ev.get("error") or {}).get("message") or "turn failed"))
        elif typ == "error":
            out.append(event(turn, "error", ev.get("message") or ""))
        elif typ == "item.completed":
            out += _codex_item(ev.get("item") or {}, turn)
    return out


def _codex_item(it: dict, turn: int) -> list[dict]:
    kind = it.get("type")
    if kind == "agent_message":
        return [event(turn, "assistant", it.get("text") or "")]
    if kind == "reasoning":
        return [event(turn, "reasoning", it.get("text") or "")]
    if kind == "command_execution":
        code = it.get("exit_code")
        return [
            event(turn, "tool_call", it.get("command") or "", tool="shell"),
            event(turn, "tool_result", it.get("aggregated_output") or "", tool="shell", is_error=code not in (0, None)),
        ]
    if kind == "mcp_tool_call":
        name = f"{it.get('server')}.{it.get('tool')}"
        err = it.get("error")
        res = it.get("result") or {}
        text = (err or {}).get("message") if err else _content_text(res.get("content")) or res.get("structured_content")
        return [
            event(turn, "tool_call", it.get("arguments") or {}, tool=name),
            event(turn, "tool_result", text or "", tool=name, is_error=bool(err)),
        ]
    if kind == "file_change":
        changes = [f"{c.get('kind')} {c.get('path')}" for c in it.get("changes") or []]
        return [event(turn, "tool_call", "\n".join(changes), tool="file_change")]
    if kind == "web_search":
        return [event(turn, "tool_call", it.get("query") or "", tool="web_search")]
    if kind == "todo_list":
        items = [f"[{'x' if i.get('completed') else ' '}] {i.get('text')}" for i in it.get("items") or []]
        return [event(turn, "reasoning", "\n".join(items))]
    if kind == "error":
        return [event(turn, "error", it.get("message") or "")]
    return [event(turn, "tool_call", it, tool=str(kind))]  # an item type this version doesn't know: keep it whole


def hermes(raw: str, turn: int) -> list[dict]:
    """Hermes stream-json: text deltas are joined into one assistant message between tool events."""
    out, text = [], []

    def flush():
        if "".join(text).strip():
            out.append(event(turn, "assistant", "".join(text).strip()))
        text.clear()

    for ev in json_lines(raw):
        typ = ev.get("type")
        if typ == "text":
            text.append(ev.get("text") or "")
        elif typ == "tool_use":
            flush()
            out.append(event(turn, "tool_call", ev.get("input") or {}, tool=ev.get("name") or "unknown"))
        elif typ == "tool_result":
            flush()
            out.append(
                event(
                    turn, "tool_result", ev.get("output") or "", tool=ev.get("name"), is_error=bool(ev.get("is_error"))
                )
            )
        elif typ == "result":
            flush()
            if ev.get("error"):
                out.append(event(turn, "error", ev["error"]))
    flush()
    return out


def openclaw(raw: str, turn: int) -> list[dict]:
    """OpenClaw's exported trajectory: the transcript side only (the runtime side repeats the same tool calls)."""
    out = []
    for ev in json_lines(raw):
        if ev.get("source") != "transcript":
            continue
        typ, msg = ev.get("type"), (ev.get("data") or {}).get("message") or {}
        if typ == "user.message":
            out.append(event(turn, "user", _content_text(msg.get("content"))))
        elif typ == "assistant.message":
            for block in msg.get("content") if isinstance(msg.get("content"), list) else [msg.get("content")]:
                out += _openclaw_block(block, turn)
            if msg.get("stopReason") == "error" or msg.get("errorMessage"):
                out.append(event(turn, "error", msg.get("errorMessage") or "model error"))
        elif typ == "tool.result":
            out.append(
                event(
                    turn,
                    "tool_result",
                    _content_text(msg.get("content")),
                    tool=msg.get("toolName"),
                    is_error=bool(msg.get("isError")),
                )
            )
    return out


def _openclaw_block(block, turn: int) -> list[dict]:
    if isinstance(block, str):
        return [event(turn, "assistant", block)] if block.strip() else []
    if not isinstance(block, dict):
        return []
    typ = block.get("type")
    if typ == "text":
        return [event(turn, "assistant", block.get("text") or "")] if (block.get("text") or "").strip() else []
    if typ == "thinking":
        return [event(turn, "reasoning", block.get("thinking") or block.get("text") or "")]
    if typ == "toolCall":
        return [event(turn, "tool_call", block.get("arguments") or {}, tool=block.get("name"))]
    return []


def _content_text(content) -> str:
    """MCP / chat content: a string, or a list of blocks of which the text ones are kept."""
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text") or "")
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def claude_message(ev: dict) -> dict:
    """A Claude Code stream-json event's `message` as an object. Some events carry it as a plain string: that string
    is kept as the message's content, so an assistant's text still reaches the transcript."""
    m = ev.get("message")
    if isinstance(m, dict):
        return m
    return {"content": m} if isinstance(m, str) and m else {}


def claude(raw: str, turn: int) -> list[dict]:
    """Claude Code stream-json: assistant content blocks, tool results (in user messages), a failed result."""
    out = []
    for ev in json_lines(raw):
        typ, content = ev.get("type"), claude_message(ev).get("content")
        if typ == "assistant":
            for b in content if isinstance(content, list) else [content]:
                out += _claude_block(b, turn)
        elif typ == "user" and isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append(
                        event(turn, "tool_result", _content_text(b.get("content")), is_error=bool(b.get("is_error")))
                    )
        elif typ == "result" and ev.get("is_error"):
            out.append(event(turn, "error", ev.get("result") or ev.get("subtype") or "error"))
    return out


def _claude_block(b, turn: int) -> list[dict]:
    if isinstance(b, str):
        return [event(turn, "assistant", b)] if b.strip() else []
    if not isinstance(b, dict):
        return []
    if b.get("type") == "text":
        return [event(turn, "assistant", b.get("text") or "")] if (b.get("text") or "").strip() else []
    if b.get("type") == "thinking":
        return [event(turn, "reasoning", b.get("thinking") or "")]
    if b.get("type") == "tool_use":
        return [event(turn, "tool_call", b.get("input") or {}, tool=b.get("name"))]
    return []


# Text-protocol agents (olas-ocb-agent) act through a bash block in the assistant's text and get the output back as a
# user message that starts with <returncode>: a tool call and its result
BASH_BLOCK = re.compile(r"```(?:bash|sh|shell)[ \t]*\n(.*?)```", re.S)
_RETURNCODE = re.compile(r"^\s*<returncode>\s*(-?\d+)\s*</returncode>")


def shell_returncode(text) -> int | None:
    """The exit code of a text-protocol shell output (`<returncode>N</returncode>...`), or None if it isn't one."""
    m = _RETURNCODE.match(text) if isinstance(text, str) else None
    return int(m.group(1)) if m else None


def chat(raw: str, turn: int) -> list[dict]:
    """OpenAI-shaped chat messages: a JSON list or a session JSONL (olas-ocb-agent). The system prompt is left out
    (it's the harness's, the same every trial); compaction is noted. The timing a session event carries beside its
    message (TIMING: seconds since the turn's start, the step's duration, the time to the first token) is kept on
    the transcript events it becomes."""
    try:
        items = json.loads(raw)
    except ValueError:
        items = None
    out = []
    for m in items if isinstance(items, list) else list(json_lines(raw)):
        if not isinstance(m, dict):
            continue
        if m.get("type") == "compaction":
            evs = [event(turn, "reasoning", "[context compacted]")]
        else:
            evs = _chat_message(m.get("message", m) if m.get("type") == "message" else m, turn)
        timing = {k: m[k] for k in TIMING if isinstance(m.get(k), int | float) and not isinstance(m[k], bool)}
        out += [e | timing for e in evs]
    return out


TIMING = ("t", "dur_s", "ttft_s")  # olas-ocb-agent's per-event timing, seconds


def _chat_message(m: dict, turn: int) -> list[dict]:
    role = m.get("role")
    text = _content_text(m.get("content"))
    if role == "user":
        if (code := shell_returncode(text)) is not None:  # a text-protocol agent's shell output, sent back as user
            return [event(turn, "tool_result", text, tool="bash", is_error=code != 0)]
        return [event(turn, "user", text)]
    if role == "tool":
        return [event(turn, "tool_result", text, tool=m.get("name"))]
    if role != "assistant":
        return []  # the system prompt
    out = [event(turn, "assistant", text)] if text.strip() else []
    if not m.get("tool_calls"):  # a text-protocol agent: the action is a bash block in the message's text
        out += [event(turn, "tool_call", {"command": c.strip()}, tool="bash") for c in BASH_BLOCK.findall(text)]
    for c in m.get("tool_calls") or []:
        f = c.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except ValueError:
            args = f.get("arguments")
        out.append(event(turn, "tool_call", args, tool=f.get("name")))
    return out


PARSERS = {"codex": codex, "hermes": hermes, "openclaw": openclaw, "claude": claude, "chat": chat}


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]


def write(dest_dir: Path, name: str, events: list[dict], raw: list[str], secrets=()) -> Path:
    """The normalised transcript (`<name>.jsonl`) and what the harness printed (`<name>.raw.jsonl`, one JSON string
    per turn) side by side, redacted (onchainbench.redact: `secrets` by value, bearer tokens and key shapes); returns
    the transcript's path. The events given are left as they were: the checks read what the agent saw."""
    from onchainbench import redact

    events, raw = redact.data(events, secrets), redact.data(raw, secrets)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{safe_name(name)}.jsonl"
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    (dest_dir / f"{safe_name(name)}.raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    return path


def read(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    return list(json_lines(Path(path).read_text(errors="ignore")))


def tool_calls(events: list[dict]) -> list[str]:
    """What the agent asked its tools to do: shell commands, MCP arguments (the scan's input)."""
    return [e["text"] for e in events if e.get("kind") == "tool_call"]
