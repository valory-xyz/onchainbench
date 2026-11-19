"""A fake OpenAI-compatible model: captures the CLI arms' real output formats without spending model money.

    python scripts/fake_model_server.py 18765                         # call a shell-like tool, then answer
    FAKE_TOOL=wallet_info python scripts/fake_model_server.py 18765   # call a Connect tool (on the mock Connect)

Point a CLI at http://127.0.0.1:18765/v1: Codex with a `model_providers` entry (wire_api "responses"), Hermes with
model.provider "custom" and base_url, OpenClaw with a models.providers entry (api "openai-completions"), Claude Code
with ANTHROPIC_BASE_URL=http://127.0.0.1:18765. Turn 1 calls the tool; after a tool result it answers. Speaks
/v1/chat/completions (streamed or not), /v1/responses (SSE) and Anthropic's /v1/messages (SSE). Request summaries go
to requests.jsonl next to this file. Used for tests/fixtures/transcripts (see the README there).
"""

import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = Path(__file__).with_name("requests.jsonl")
FINAL = "FAKE FINAL ANSWER: the tool printed fake-tool-ran."
SHELL_NAMES = ("exec_command", "shell", "terminal", "exec", "bash", "Bash", "run_command", "shell_command")


def pick_tool(tools: list) -> tuple[str | None, dict, str | None]:
    """(tool name, its parameters schema, the namespace Codex lists it under)."""
    want = os.environ.get("FAKE_TOOL")
    flat = []
    for t in tools or []:
        f = t.get("function") or t
        flat.append((f.get("name"), f.get("parameters") or f.get("input_schema") or {}, None))
        flat += [(s.get("name"), s.get("parameters") or {}, t.get("name")) for s in t.get("tools") or []]
    if want:
        return next((x for x in flat if x[0] and want in x[0]), (None, {}, None))
    for name in SHELL_NAMES:
        for x in flat:
            if x[0] == name and x[2] is None:
                return x
    return flat[0] if flat else (None, {}, None)


def args_for(params: dict) -> dict:
    cmd = os.environ.get("FAKE_COMMAND", "echo fake-tool-ran")  # e.g. a curl, to test the live_web proxy
    props = (params or {}).get("properties", {})
    for key in ("command", "cmd"):
        if key in props:
            return {key: ["bash", "-lc", cmd] if props[key].get("type") == "array" else cmd}
    return {} if params else {"command": cmd}


def has_tool_result(messages: list) -> bool:
    def anthropic_result(m):  # Anthropic Messages: a user message with tool_result blocks
        content = m.get("content")
        return isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )

    return any(
        m.get("role") == "tool" or m.get("type") == "function_call_output" or anthropic_result(m)
        for m in messages or []
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj: dict) -> None:
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse(self, data: str) -> None:
        self.wfile.write(data.encode())

    def do_GET(self):
        self._json({"object": "list", "data": [{"id": "fake/model", "object": "model", "context_length": 128000}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        msgs = body.get("messages") or body.get("input") or []
        done = has_tool_result(msgs if isinstance(msgs, list) else [])
        name, params, namespace = (None, {}, None) if done else pick_tool(body.get("tools"))
        with LOG.open("a") as f:
            f.write(json.dumps({"path": self.path, "stream": body.get("stream"), "done": done, "picked": name}) + "\n")
        args = json.dumps(args_for(params))
        if "/messages/count_tokens" in self.path:
            return self._json({"input_tokens": 100})
        if self.path.split("?")[0].endswith("/messages"):
            return self.messages_api(name, args, body.get("stream"))
        if self.path.endswith("/responses"):
            return self.responses_api(name, args, namespace)
        call = name and {
            "id": "call_" + uuid.uuid4().hex[:8],
            "type": "function",
            "function": {"name": name, "arguments": args},
        }
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        if not body.get("stream"):
            msg = {"role": "assistant", "content": None if call else FINAL, **({"tool_calls": [call]} if call else {})}
            choice = {"index": 0, "message": msg, "finish_reason": "tool_calls" if call else "stop"}
            return self._json(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "fake/model",
                    "choices": [choice],
                    "usage": usage,
                }
            )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def chunk(delta: dict, finish: str | None = None, use: dict | None = None) -> None:
            c = {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "fake/model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **({"usage": use} if use else {}),
            }
            self._sse(f"data: {json.dumps(c)}\n\n")

        if call:
            chunk({"role": "assistant", "tool_calls": [{"index": 0, **call}]})
            chunk({}, "tool_calls", usage)
        else:
            chunk({"role": "assistant", "content": FINAL})
            chunk({}, "stop", usage)
        self._sse("data: [DONE]\n\n")

    def messages_api(self, name: str | None, args: str, stream: bool) -> None:
        """Anthropic Messages (Claude Code): a tool_use block, or a text block after a tool result."""
        mid = "msg_" + uuid.uuid4().hex[:8]
        tool = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:8], "name": name, "input": json.loads(args)}
        block = tool if name else {"type": "text", "text": FINAL}
        stop = "tool_use" if name else "end_turn"
        if not stream:
            msg = {"id": mid, "type": "message", "role": "assistant", "model": "claude-fake", "content": [block]}
            return self._json({**msg, "stop_reason": stop, "usage": {"input_tokens": 100, "output_tokens": 20}})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def ev(typ: str, **kw) -> None:
            self._sse(f"event: {typ}\ndata: {json.dumps({'type': typ, **kw})}\n\n")

        start = {"id": mid, "type": "message", "role": "assistant", "model": "claude-fake", "content": []}
        ev("message_start", message={**start, "stop_reason": None, "usage": {"input_tokens": 100, "output_tokens": 0}})
        if name:
            ev("content_block_start", index=0, content_block={**tool, "input": {}})
            ev("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": args})
        else:
            ev("content_block_start", index=0, content_block={"type": "text", "text": ""})
            ev("content_block_delta", index=0, delta={"type": "text_delta", "text": FINAL})
        ev("content_block_stop", index=0)
        ev("message_delta", delta={"stop_reason": stop, "stop_sequence": None}, usage={"output_tokens": 20})
        ev("message_stop")

    def responses_api(self, name: str | None, args: str, namespace: str | None) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        rid = "resp_" + uuid.uuid4().hex[:8]

        def ev(typ: str, **kw) -> None:
            self._sse(f"event: {typ}\ndata: {json.dumps({'type': typ, **kw})}\n\n")

        ev("response.created", response={"id": rid, "status": "in_progress", "output": []})
        if name:
            item = {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_" + uuid.uuid4().hex[:8],
                "name": name,
                "arguments": args,
                "status": "completed",
                **({"namespace": namespace} if namespace else {}),
            }
        else:
            item = {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": FINAL, "annotations": []}],
            }
            ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta=FINAL)
        ev("response.output_item.done", output_index=0, item=item)
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        ev("response.completed", response={"id": rid, "status": "completed", "output": [item], "usage": usage})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 18765), Handler).serve_forever()
