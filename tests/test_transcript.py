"""The CLI arms' transcripts in one shape (onchainbench.transcript), checked against each CLI's real output."""

import json
from pathlib import Path

from onchainbench import transcript as TR

FIX = Path(__file__).parent / "fixtures" / "transcripts"


def kinds(events):
    return [(e["kind"], e.get("tool")) for e in events]


def test_codex_real_output_shell_and_mcp():
    ev = TR.codex((FIX / "codex-shell.jsonl").read_text(), 0)
    assert kinds(ev) == [("error", None), ("tool_call", "shell"), ("tool_result", "shell"), ("assistant", None)]
    assert "echo fake-tool-ran" in ev[1]["text"] and "fake-tool-ran" in ev[2]["text"] and not ev[2]["is_error"]
    assert ev[3]["text"].startswith("FAKE FINAL ANSWER")
    mcp = TR.codex((FIX / "codex-mcp.jsonl").read_text(), 2)
    call, result = (e for e in mcp if e.get("tool") == "pearl-connect.wallet_info")
    assert call["kind"] == "tool_call" and call["text"] == "{}" and call["turn"] == 2
    assert result["kind"] == "tool_result" and "0x5afe" in result["text"] and result["is_error"] is False


def test_hermes_real_output():
    ev = TR.hermes((FIX / "hermes.jsonl").read_text(), 1)
    assert kinds(ev) == [("tool_call", "terminal"), ("tool_result", "terminal"), ("assistant", None)]
    assert json.loads(ev[0]["text"]) == {"command": "echo fake-tool-ran"} and "fake-tool-ran" in ev[1]["text"]
    assert ev[2]["text"] == "FAKE FINAL ANSWER: the tool printed fake-tool-ran." and ev[2]["turn"] == 1


def test_openclaw_real_trajectory_export():
    ev = TR.openclaw((FIX / "openclaw-trajectory.jsonl").read_text(), 0)
    assert kinds(ev) == [("user", None), ("tool_call", "exec"), ("tool_result", "exec"), ("assistant", None)]
    assert (
        ev[0]["text"].startswith("What's in my safe?") and json.loads(ev[1]["text"])["command"] == "echo fake-tool-ran"
    )
    assert ev[2]["text"] == "fake-tool-ran" and ev[2]["is_error"] is False


def lines(*events):
    return "not json\n[1]\n" + "\n".join(json.dumps(e) for e in events)


def test_codex_every_item_kind():
    items = [
        {"type": "reasoning", "text": "thinking"},
        {"type": "command_execution", "command": "false", "aggregated_output": "", "exit_code": 1},
        {"type": "mcp_tool_call", "server": "s", "tool": "t", "arguments": {"a": 1}, "error": {"message": "denied"}},
        {"type": "mcp_tool_call", "server": "s", "tool": "u", "result": {"structured_content": {"x": 1}}},
        {"type": "file_change", "changes": [{"path": "a.py", "kind": "add"}]},
        {"type": "web_search", "query": "olas"},
        {"type": "todo_list", "items": [{"text": "one", "completed": True}, {"text": "two"}]},
        {"type": "error", "message": "warned"},
        {"type": "brand_new", "x": 1},
    ]
    ev = TR.codex(
        lines(
            *({"type": "item.completed", "item": i} for i in items),
            {"type": "item.started", "item": {"type": "agent_message", "text": "skipped"}},
            {"type": "turn.failed", "error": {"message": "rate limited"}},
            {"type": "turn.failed"},
            {"type": "error", "message": "stream closed"},
        ),
        0,
    )
    want = "reasoning tool_call tool_result tool_call tool_result tool_call tool_result tool_call tool_call reasoning"
    assert [e["kind"] for e in ev] == [*want.split(), "error", "tool_call", "error", "error", "error"]
    assert ev[2]["is_error"] and ev[4] == {
        "turn": 0,
        "kind": "tool_result",
        "text": "denied",
        "tool": "s.t",
        "is_error": True,
    }
    assert json.loads(ev[6]["text"]) == {"x": 1} and ev[7]["text"] == "add a.py" and ev[8]["text"] == "olas"
    assert ev[9]["text"] == "[x] one\n[ ] two" and ev[11]["tool"] == "brand_new"
    assert [e["text"] for e in ev[-3:]] == ["rate limited", "turn failed", "stream closed"]


def test_hermes_text_is_joined_between_tools_and_errors_kept():
    ev = TR.hermes(
        lines(
            {"type": "text", "text": "Let me "},
            {"type": "text", "text": "check."},
            {"type": "tool_use", "name": "terminal"},
            {"type": "tool_result", "output": "boom", "is_error": True},
            {"type": "text", "text": "   "},
            {"type": "result", "error": "provider down"},
            {"type": "text", "text": "late"},
        ),
        0,
    )
    assert [(e["kind"], e["text"]) for e in ev] == [
        ("assistant", "Let me check."),
        ("tool_call", "{}"),
        ("tool_result", "boom"),
        ("error", "provider down"),
        ("assistant", "late"),
    ]
    assert ev[1]["tool"] == "terminal" and ev[2]["is_error"] is True


def test_openclaw_blocks_and_errors():
    def tr(typ, msg, source="transcript"):
        return {"type": typ, "source": source, "data": {"message": msg}}

    ev = TR.openclaw(
        lines(
            tr("user.message", {"content": [{"type": "text", "text": "hi"}, "there", {"type": "image"}]}),
            tr("assistant.message", {"content": "plain answer"}),
            tr("assistant.message", {"content": "  "}),
            tr(
                "assistant.message",
                {
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": " "},
                        {"type": "text", "text": "ok"},
                        {"type": "image"},
                        7,
                    ],
                    "stopReason": "error",
                    "errorMessage": "429",
                },
            ),
            tr("assistant.message", {"content": [], "stopReason": "error"}),
            tr("tool.call", {"name": "exec"}, source="runtime"),
            tr("tool.result", {"toolName": "exec", "content": "out", "isError": True}),
            tr("session.custom", {}),
        ),
        3,
    )
    assert [(e["kind"], e["text"]) for e in ev] == [
        ("user", "hi\nthere"),
        ("assistant", "plain answer"),
        ("reasoning", "hmm"),
        ("assistant", "ok"),
        ("error", "429"),
        ("error", "model error"),
        ("tool_result", "out"),
    ]
    assert ev[-1]["is_error"] is True and ev[-1]["tool"] == "exec" and {e["turn"] for e in ev} == {3}


def test_events_are_capped_and_files_round_trip(tmp_path):
    e = TR.event(0, "tool_result", "x" * (TR.TEXT_CAP + 5))
    assert e["text"].endswith("[5 more characters]") and len(e["text"]) < TR.TEXT_CAP + 40
    assert TR.event(1, "tool_call", {"a": "é"}) == {"turn": 1, "kind": "tool_call", "text": '{"a": "é"}'}
    events = [TR.event(0, "user", "go"), TR.event(0, "tool_call", "ls", tool="shell")]
    path = TR.write(tmp_path / "t", "codex:openrouter/x?p=y__a-task__0", events, ['{"raw": 1}'])
    assert path.name == "codex_openrouter_x_p_y__a-task__0.jsonl" and TR.read(path) == events
    assert json.loads((path.parent / "codex_openrouter_x_p_y__a-task__0.raw.jsonl").read_text()) == '{"raw": 1}'
    assert TR.tool_calls(TR.read(path)) == ["ls"] and TR.read(None) == [] and TR.read(tmp_path / "missing") == []
    assert set(TR.KINDS) >= {e["kind"] for e in events} and TR.safe_name("a" * 200) == "a" * 120


def test_claude_code_real_stream_json():
    ev = TR.claude((FIX / "claude-code.jsonl").read_text(), 0)
    assert kinds(ev) == [("tool_call", "Bash"), ("tool_result", None), ("assistant", None)]
    assert json.loads(ev[0]["text"]) == {"command": "echo fake-tool-ran"} and ev[1]["text"] == "fake-tool-ran"
    assert ev[1]["is_error"] is False and ev[2]["text"].startswith("FAKE FINAL ANSWER")


def test_claude_code_blocks_and_failures():
    def msg(typ, content):
        return {"type": typ, "message": {"content": content}}

    ev = TR.claude(
        lines(
            msg("assistant", "plain"),
            msg("assistant", "  "),
            msg(
                "assistant",
                [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": " "}, {"type": "image"}, 3],
            ),
            msg(
                "user",
                [{"type": "tool_result", "content": [{"type": "text", "text": "denied"}], "is_error": True}, "x"],
            ),
            msg("user", "a plain user message"),
            {"type": "result", "is_error": True, "subtype": "error_max_turns"},
            {"type": "result", "is_error": True, "result": "API error"},
            {"type": "result", "is_error": False, "result": "fine"},
        ),
        2,
    )
    assert [(e["kind"], e["text"]) for e in ev] == [
        ("assistant", "plain"),
        ("reasoning", "hmm"),
        ("tool_result", "denied"),
        ("error", "error_max_turns"),
        ("error", "API error"),
    ]
    assert ev[2]["is_error"] is True and {e["turn"] for e in ev} == {2}


def test_chat_messages_with_native_tool_calls():
    """OpenAI-shaped messages with native tool calls: the call, its result, and no extra role is special any more
    (mini-swe-agent's `exit` role and format-error interrupts were its own)."""
    call = {"function": {"name": "bash", "arguments": json.dumps({"command": "echo hi"})}}
    msgs = [
        {"role": "system", "content": "harness"},
        {"role": "user", "content": "go", "extra": {"interrupt_type": "FormatError"}},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "name": "bash", "content": "hi"},
        {"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted"}},
    ]
    ev = TR.chat(json.dumps(msgs), 0)
    assert [(e["kind"], e.get("tool")) for e in ev] == [("user", None), ("tool_call", "bash"), ("tool_result", "bash")]
    assert json.loads(ev[1]["text"]) == {"command": "echo hi"} and ev[2]["text"] == "hi"


def test_chat_session_files_and_odd_messages(tmp_path):
    session = lines(
        {"type": "message", "message": {"role": "user", "content": "go"}},
        {"type": "state", "state": {}},
        {"type": "compaction", "messages": []},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [{"function": {"name": "bash", "arguments": "not json"}}],
            },
        },
        {"type": "message", "message": {"role": "tool", "name": "bash", "content": [{"type": "text", "text": "out"}]}},
        {"role": "narrator", "content": "?"},
    )
    ev = TR.chat(session, 1)
    assert [(e["kind"], e["text"]) for e in ev] == [
        ("user", "go"),
        ("reasoning", "[context compacted]"),
        ("assistant", "ok"),
        ("tool_call", "not json"),
        ("tool_result", "out"),
    ]
    assert TR.chat("[1, 2]", 0) == [] and TR.chat('{"role": "user", "content": "solo"}', 0)[0]["text"] == "solo"
