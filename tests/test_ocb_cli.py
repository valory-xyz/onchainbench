"""olas-ocb-agent as a CLI arm (`ocb-agent -p --output json`), driven by a stand-in executable: the command line the
bench builds, the JSON it reads (transcript, counts, refusals, listings, error kinds), resumes, killed turns, leak
counting, the session file's hand-over and the mech variant."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

from onchainbench import arms as A
from onchainbench import failures as F
from onchainbench import ocb_cli as O
from onchainbench.connect.policy import SessionCaps

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stand-in CLI is a script with a shebang")

STUB = """#!{py}
import json, os, sys, time
from pathlib import Path
plan = json.loads(Path(os.environ["FAKE_PLAN"]).read_text())
log = Path(os.environ["FAKE_PLAN"]).with_name("calls.jsonl")
n = len(log.read_text().splitlines()) if log.exists() else 0
a = sys.argv[1:]
with log.open("a") as f:
    f.write(json.dumps({{"argv": a, "key": os.environ.get("OPENROUTER_API_KEY"),
                         "proxy": os.environ.get("HTTPS_PROXY"), "xdg": os.environ.get("XDG_STATE_HOME")}}) + "\\n")
step = plan[min(n, len(plan) - 1)]
ws, sd = Path(a[a.index("--dir") + 1]), Path(a[a.index("--sessions-dir") + 1])
if step.get("token"):
    (ws / "agent").mkdir(exist_ok=True)
    shim = {{"mcpServers": {{"pearl-connect": {{"headers": {{"Authorization": "Bearer " + step["token"]}}}}}}}}
    (ws / "agent" / ".mcp.json").write_text(json.dumps(shim))
with (sd / "s1.jsonl").open("a") as f:
    for line in step.get("session", []):
        f.write((line if isinstance(line, str) else json.dumps(line)) + "\\n")
sys.stderr.write(step.get("stderr", ""))
sys.stderr.flush()
time.sleep(step.get("sleep", 0))
if "json" in step:
    out = dict(step["json"])
    out.setdefault("session_file", str(sd / "s1.jsonl"))
    print("noise before")
    print(json.dumps(out))
sys.exit(step.get("exit", 0))
"""

SYSTEM = {"type": "message", "message": {"role": "system", "content": "the system prompt"}}
USER = {"type": "message", "message": {"role": "user", "content": "pay the invoice"}}
CALL = {
    "type": "message",
    "message": {"role": "assistant", "content": "```bash\npearl wallet_info\n```"},
}
RESULT = {"type": "message", "message": {"role": "user", "content": "<returncode>0</returncode>\nok"}}
DONE = {"type": "message", "message": {"role": "assistant", "content": "DONE: paid"}}


def report(**kw):
    base = {
        "reply": "paid",
        "session_id": "s1",
        "done": True,
        "error": None,
        "model_calls": 2,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cached_tokens": 40,
        "cost_usd": 0.01,
        "cost_sources": {"billed": 2},
        "compactions": 0,
        "steps": 2,
        "policy_refusals": [],
        "broker_refusals": [],
        "tools_listed": 0,
        "events": [USER, CALL, RESULT, DONE],
    }
    return base | kw


def fake_t(tmp_path):
    trial = tmp_path / "pe2-trial"
    ws = trial / "ws"
    (ws / "agent").mkdir(parents=True)
    ledger = []
    s = types.SimpleNamespace(rug_pull=None, llm_backend=None, prepaid=0, mech_amount=lambda h: h * 100)
    fc = types.SimpleNamespace(
        s=s,
        _log=ledger.append,
        poisoned_listing=lambda: {"poisoned": ["wallet_info"], "rugged": False},
    )
    data = {"proxy_env": {"HTTPS_PROXY": "http://127.0.0.1:9999", "NO_PROXY": "localhost"}}
    return types.SimpleNamespace(ws=ws, fc=fc, data=data, chain="gnosis", ledger=ledger)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A stand-in `ocb-agent` and its plan: one step per turn (the last repeats)."""
    stub = tmp_path / "ocb-agent"
    stub.write_text(STUB.format(py=sys.executable))
    stub.chmod(0o755)
    plan = tmp_path / "plan.json"
    monkeypatch.setenv("FAKE_PLAN", str(plan))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-main")
    monkeypatch.delenv("OPENROUTER_PROVISIONING_KEY", raising=False)
    monkeypatch.setattr(A, "cli_bin", lambda kind: str(stub))
    monkeypatch.setattr(
        "onchainbench.pricing.price",
        lambda spec: {"input": 1e-7, "output": 2e-7, "cache_read": 1e-8, "cache_write": 1e-7},
    )

    def set_plan(*steps):
        plan.write_text(json.dumps(list(steps)))

    def calls():
        return [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]

    return types.SimpleNamespace(plan=set_plan, calls=calls)


def test_a_turn_builds_the_command_line_and_reads_the_json(tmp_path, cli):
    t = fake_t(tmp_path)
    t.fc.s.rug_pull = {"tool": ("sweep_funds", "drains"), "to": "0xRUG"}
    refusals = [
        {"tool": "safe_transaction", "args": {"target": "0xABC", "data": "0x12"}, "rule": "guardrail"},
        {"tool": "sweep_funds", "args": {"to": "0xother"}, "rule": "unknown tool"},
        {"tool": "sign_message", "args": {"digest": "0xDEAD"}, "rule": "sign"},
    ]
    cli.plan(
        {
            "json": report(events=[SYSTEM, USER, CALL, RESULT, DONE], broker_refusals=refusals, tools_listed=2),
            "token": "brokertok",
        }
    )
    s = O.OcbAgentSession(t, "olas-ocb-agent", "openrouter/deepseek/x?provider=fireworks", 500)
    assert s.send("pay the invoice") == "paid"
    (call,) = cli.calls()
    argv = call["argv"]

    def flag(name):
        return argv[argv.index(name) + 1]

    assert flag("--model") == "litellm:openrouter/deepseek/x?provider=fireworks" and flag("--approval") == "auto"
    assert flag("--agent-subdir") == "agent" and flag("--trusted-skill-hash") == O.skill_hash()
    assert flag("--step-limit") == "500" and flag("--cost-limit") == str(A.TRIAL_CAP_USD)
    assert flag("--max-native-usd") == flag("--max-mech-usd") == "inf" and flag("--max-value-calls") == "none"
    assert "--allow-sign-message" in argv and flag("--turn-seconds") == str(A.HARNESS_TURN_SECONDS)
    assert flag("--model-budget") == "1.0" and flag("--output") == "json" and argv[-2:] == ["-p", "pay the invoice"]
    assert json.loads(flag("--model-price"))["input_cost_per_token"] == 1e-7
    shell = [argv[i + 1] for i, x in enumerate(argv) if x == "--shell-env"]
    assert shell == ["HTTPS_PROXY=http://127.0.0.1:9999", "NO_PROXY=localhost"] and t.data["proxy_applied"]
    assert "--attended" not in argv and "--resume" not in argv
    # the model key in the CLI's environment, the proxy only for the agent's shell, its state in the trial
    assert call["key"] == "sk-or-main" and call["proxy"] is None and call["xdg"].endswith("ocb-agent/state")
    assert s.stats.model_calls == 2 and s.stats.prompt_tokens == 100 and s.stats.extra["cached_tokens"] == 40
    assert s.stats.cost_usd == 0.01 and s.stats.extra["cost_sources"] == {"billed": 2}
    assert s.model_calls_so_far() == 2
    events, raw = s.transcript()  # the system prompt is left out; the user message is the session's own
    assert [e["kind"] for e in events] == ["user", "assistant", "tool_call", "tool_result", "assistant"]
    assert len(raw) == 1 and "the system prompt" in raw[0]
    refused = [e for e in t.ledger if e["kind"] == "refusal"]
    assert [(e["tool"], e["to"], e["rule"]) for e in refused] == [
        ("safe_transaction", "0xabc", "guardrail"),
        ("sweep_funds", "0xrug", "unknown tool"),  # the rug pull's tool drains to its address
        ("sign_message", "", "sign"),
    ]
    assert refused[0]["data"] == "0x12" and refused[2]["digest"] == "0xdead"
    assert [e["kind"] for e in t.ledger].count("agent_tools_list") == 2 and t.data["agent_tools_listed"] == 2
    assert "brokertok" in t.data["secrets"]  # the agent's broker token: a leak of it counts
    assert "the OpenRouter key" in str(t.data["named_secrets"])


def test_turns_resume_the_session_and_add_the_difference(tmp_path, cli):
    t = fake_t(tmp_path)
    second = report(
        model_calls=5,
        prompt_tokens=300,
        completion_tokens=25,
        cached_tokens=100,
        cost_usd=0.03,
        policy_refusals=[{"tool": "safe_transaction", "allowed": False}],
        events=[USER, DONE],
        reply="answered",
    )
    cli.plan(
        {"json": report(policy_refusals=[{"tool": "x", "allowed": False}]), "session": [USER, DONE]},
        {"json": second, "session": [{"type": "state"}, USER, DONE]},
    )
    dest = tmp_path / "run-sessions"
    s = O.OcbAgentSession(t, "olas-ocb-agent", "openrouter/m", 500, unattended=False, sessions_dir=dest)
    assert s.send("one") == "paid" and s.send("two") == "answered"
    first, again = (c["argv"] for c in cli.calls())
    assert "--attended" in first and "--resume" not in first  # a simulated user answers: attended on every turn
    assert again[again.index("--resume") + 1] == "s1" and "--attended" in again
    assert s.stats.model_calls == 5 and s.stats.prompt_tokens == 300 and s.stats.completion_tokens == 25
    assert abs(s.stats.cost_usd - 0.03) < 1e-12 and s.stats.extra["cached_tokens"] == 100
    assert [r["tool"] for r in s.stats.extra["policy_refusals"]] == ["x", "safe_transaction"]
    assert s.seen[str(s.sessions / "s1.jsonl")] == 5
    s.crash_and_resume()
    st = s.close()  # the session file is copied where the run keeps it
    assert st.extra["resumes"] == 1 and st.extra["session"] == str(dest / "s1.jsonl")
    assert (dest / "s1.jsonl").read_text().count("\n") == 5


def test_the_cap_ends_the_trial_before_another_turn(tmp_path, cli):
    s = O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500)
    s.stats.cost_usd = A.TRIAL_CAP_USD
    assert s.send("more") == A.CAP_REACHED


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ("provider", F.PROVIDER_ERROR),
        ("harness", F.HARNESS_ERROR),
        ("timeout", F.TIMEOUT),
        ("terminated", F.BENCH_ERROR),  # 0.4.0 on SIGTERM: a stop from outside, infra
        ("model", F.OK),  # the only error that ends a turn normally: a budget or cap in the model backend
        ("something-new", F.HARNESS_ERROR),  # any value the bench doesn't know is the harness's own failure
    ],
)
def test_the_json_error_is_the_turns_failure_kind(tmp_path, cli, error, kind):
    cli.plan({"json": report(error=error, reply=f"[{error}] it stopped"), "exit": 1, "stderr": "trace"})
    s = O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500)
    r = s.send("go")
    assert getattr(r, "kind", F.OK) == kind and str(r) == f"[{error}] it stopped"
    assert s.transcript()[0][-1] == {"turn": 0, "kind": "error", "text": f"[{error}] it stopped"}
    if kind != F.OK:
        assert "trace" in r.detail


def test_a_start_failure_s_report_is_the_harness_s_and_counts_nothing(tmp_path, cli):
    """0.4.0: a CLI that can't start (a bad resume, a sandbox) still prints JSON: error "harness", no counts."""
    start = {"reply": "[error] SessionNotFound: s1", "session_id": "s1", "done": False, "error": "harness"}
    cli.plan({"json": report()}, {"json": {**start, "error_detail": "SessionNotFound: s1"}, "exit": 1})
    s = O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500)
    assert s.send("one") == "paid"
    r = s.send("two")
    assert r.kind == F.HARNESS_ERROR and r.detail == "SessionNotFound: s1" and s.stats.model_calls == 2
    assert s.totals.last["model_calls"] == 2  # the session's totals stay what it last reported


def test_an_exit_without_json_is_a_cli_error(tmp_path, cli):
    cli.plan({"exit": 1, "stderr": "Traceback ...\nlitellm.RateLimitError: slow down"}, {"exit": 2})
    s = O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500)
    r = s.send("go")
    assert r.kind == F.PROVIDER_ERROR and "RateLimitError" in r.detail
    r = s.send("again")
    assert r.kind == F.HARNESS_ERROR and r.detail == "exit 2, no output"
    events = s.transcript()[0]
    assert [e["kind"] for e in events] == ["user", "error", "user", "error"]


def test_a_killed_turn_counts_its_calls_from_the_session_file(tmp_path, cli, monkeypatch):
    cli.plan(
        {"json": report(events=[USER, DONE]), "session": [USER, DONE]},
        {"session": [{"type": "state"}, USER, CALL, "not json", {"type": "compaction"}, CALL], "sleep": 5},
    )
    monkeypatch.setattr(A, "TURN_SECONDS", 3)  # room for the stand-in to start and log under a loaded CI runner
    s = O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500)
    assert s.send("one") == "paid"
    r = s.send("two")
    assert r.kind == F.TIMEOUT and s.stats.model_calls == 2 + 2  # the two assistant messages it logged
    events = s.transcript()[0]
    turn1 = [e["kind"] for e in events if e["turn"] == 1]
    assert turn1 == ["user", "assistant", "tool_call", "reasoning", "assistant", "tool_call", "error"]
    # a turn killed before it logged anything: the user message and the error
    cli.plan({"sleep": 5})
    s = O.OcbAgentSession(fake_t(tmp_path / "b"), "olas-ocb-agent", "openrouter/m", 500)
    (s.sessions / "unreadable.jsonl").mkdir()  # read_text fails: skipped
    assert s.send("x").kind == F.TIMEOUT and [e["kind"] for e in s.transcript()[0]] == ["user", "error"]


def test_the_agent_s_event_timing_reaches_the_transcript(tmp_path, cli, monkeypatch):
    """olas-ocb-agent puts `t` (seconds since the turn's start), `dur_s` and `ttft_s` beside each event's message:
    the arm reads its turns as before, and the transcript events keep the timing, from the JSON and from the session
    file of a killed turn alike. An event without timing (an older agent) gets none."""
    timed_call = CALL | {"t": 2.5, "dur_s": 2.0, "ttft_s": 0.4}
    timed_result = RESULT | {"t": 4.0, "dur_s": 1.5}
    cli.plan(
        {"json": report(events=[USER | {"t": 0.0}, timed_call, timed_result, DONE])},
        {"session": [USER | {"t": 0.0}, CALL | {"t": 1.0, "dur_s": 0.75, "ttft_s": True}], "sleep": 5},
    )
    monkeypatch.setattr(A, "TURN_SECONDS", 3)
    s = O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500)
    assert s.send("one") == "paid" and s.stats.model_calls == 2
    assert s.send("two").kind == F.TIMEOUT
    got = [(e["turn"], e["kind"], e.get("t"), e.get("dur_s"), e.get("ttft_s")) for e in s.transcript()[0]]
    assert got == [
        (0, "user", 0.0, None, None),
        (0, "assistant", 2.5, 2.0, 0.4),
        (0, "tool_call", 2.5, 2.0, 0.4),
        (0, "tool_result", 4.0, 1.5, None),
        (0, "assistant", None, None, None),
        (1, "user", 0.0, None, None),
        (1, "assistant", 1.0, 0.75, None),  # a non-number (here a bool) isn't timing
        (1, "tool_call", 1.0, 0.75, None),
        (1, "error", None, None, None),
    ]


def test_the_broker_token_is_read_only_from_a_well_formed_shim(tmp_path, cli):
    t = fake_t(tmp_path)
    s = O.OcbAgentSession(t, "olas-ocb-agent", "openrouter/m", 500)
    (t.ws / "agent" / ".mcp.json").write_text("{not json")
    s._broker_token()
    (t.ws / "agent" / ".mcp.json").write_text(json.dumps({"mcpServers": {}}))
    s._broker_token()
    assert "secrets" not in t.data
    shim = {"mcpServers": {"pearl-connect": {"headers": {"Authorization": "Bearer tok"}}}}
    (t.ws / "agent" / ".mcp.json").write_text(json.dumps(shim))
    s._broker_token()
    s._broker_token()
    assert t.data["secrets"] == ["tok"]


def test_close_keeps_the_session_where_it_is_without_a_run_dir_and_says_when_it_cant_copy(tmp_path, cli):
    cli.plan({"json": report()})
    s = O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500)
    s.send("go")
    assert s.close().extra["session"] == str(s.sessions / "s1.jsonl")
    cli.plan({"json": report(session_file=str(tmp_path / "missing.jsonl"))})
    s = O.OcbAgentSession(fake_t(tmp_path / "b"), "olas-ocb-agent", "openrouter/m", 500, sessions_dir=tmp_path / "d")
    s.send("go")
    assert "FileNotFoundError" in s.close().extra["session_error"]
    s = O.OcbAgentSession(fake_t(tmp_path / "c"), "olas-ocb-agent", "openrouter/m", 500, sessions_dir=tmp_path / "e")
    assert "session" not in s.close().extra  # no turn ran: nothing to copy


def test_an_unpriced_model_gets_no_model_price(tmp_path, cli, monkeypatch):
    monkeypatch.setattr("onchainbench.pricing.price", lambda spec: None)
    cli.plan({"json": report()})
    O.OcbAgentSession(fake_t(tmp_path), "olas-ocb-agent", "openrouter/m", 500).send("go")
    assert "--model-price" not in cli.calls()[0]["argv"]


def test_caps_flags_follow_the_caps():
    assert O.caps_flags() == [
        "--max-native-usd",
        "inf",
        "--max-mech-usd",
        "inf",
        "--max-value-calls",
        "none",
        "--allow-sign-message",
    ]
    assert O.caps_flags(SessionCaps()) == [
        "--max-native-usd",
        "0.1",
        "--max-mech-usd",
        "1.0",
        "--max-value-calls",
        "20",
    ]


def test_chat_text_renders_a_native_tool_call_as_a_bash_block():
    call = {"function": {"arguments": json.dumps({"command": "pearl wallet_info"})}}
    assert O.chat_text({"content": "hi", "tool_calls": [call, {"function": {"arguments": "{bad"}}]}) == (
        "hi\n```bash\npearl wallet_info\n```"
    )
    assert O.chat_text({"content": None}) == ""


def _response(content, tool_calls=None, cost=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = {"cost": cost} if cost is not None else None
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def test_the_mech_variant_backs_the_chat_mech_in_the_bench(tmp_path, cli, monkeypatch):
    import litellm

    from onchainbench.fork_connect import LLM_MECH
    from onchainbench.mock_connect.server import ToolError

    seen = []

    def completion(**kw):
        seen.append(kw)
        if kw["messages"][0]["content"] == "fail":
            raise RuntimeError("upstream 503")
        call = types.SimpleNamespace(model_dump=lambda: {"function": {"arguments": '{"command": "ls"}'}})
        return _response("ok", [call], cost=0.5 if len(seen) == 1 else None)

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.25)
    t = fake_t(tmp_path)
    cli.plan({"json": report(error="model", reply="[model] mech model call failed")}, {"json": report()})
    s = O.OcbAgentSession(t, "olas-ocb-agent-mech", "openrouter/m?provider=fireworks", 500)
    assert t.fc.s.prepaid == 500000 and s.key is None
    llm = t.fc.s.llm_backend
    assert llm("hello") == "ok\n```bash\nls\n```" and llm([{"role": "user", "content": "hi"}]).startswith("ok")
    assert s.backend_spent == [0.75]
    assert seen[0]["model"] == "openrouter/m"
    assert seen[0]["extra_body"]["provider"] == {"only": ["fireworks"], "allow_fallbacks": False}
    with pytest.raises(RuntimeError):
        llm("fail")
    assert s.backend_errors == ["RuntimeError: upstream 503"]
    r = s.send("go")  # the earlier failure was cleared at the turn's start: this model error is the agent's cap
    argv = cli.calls()[0]["argv"]
    assert argv[argv.index("--model") + 1] == f"mech:chat@{LLM_MECH}" and "--model-price" not in argv
    assert cli.calls()[0]["key"] is None  # no model key: the bench's process calls the model
    assert r == "[model] mech model call failed" and getattr(r, "kind", F.OK) == F.OK
    s.backend_errors.append("RuntimeError: upstream 503")  # a backend failure in the turn: the provider's, infra
    assert s._turn(report(error="model", reply="[model] x"), "").kind == F.PROVIDER_ERROR
    s.backend_spent[0] = A.TRIAL_CAP_USD
    with pytest.raises(ToolError):
        llm("hello")
    assert s.close() is s.stats


def test_the_mech_backend_counts_nothing_for_an_unpriced_reply(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: _response("ok"))

    def unpriced(completion_response):
        raise ValueError("no price")

    monkeypatch.setattr(litellm, "completion_cost", unpriced)
    spent = [0.0]
    assert O.mech_backend("openrouter/m", spent, [])("hi") == "ok" and spent == [0.0]


def test_the_mech_backend_calls_the_model_as_the_agent_s_own_litellm_model_did(monkeypatch):
    """A stop at `<returncode>` (dropped once if the provider rejects it), three retries, the provider pin, and
    OpenRouter's charge in the usage only for an OpenRouter model."""
    import litellm

    seen = []

    def completion(**kw):
        seen.append(kw)
        if "stop" in kw and len(seen) == 1:
            raise litellm.exceptions.BadRequestError("stop sequences unsupported", model="m", llm_provider="x")
        return _response("ok", cost=0.1)

    monkeypatch.setattr(litellm, "completion", completion)
    assert O.mech_backend("openrouter/m?provider=fireworks", [0.0], [])("hi") == "ok"
    first, retry = seen
    assert first["stop"] == ["<returncode>"] and first["num_retries"] == 3 and first["model"] == "openrouter/m"
    assert first["extra_body"] == {
        "provider": {"only": ["fireworks"], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    assert "stop" not in retry and retry["num_retries"] == 3 and retry["extra_body"] == first["extra_body"]
    assert O.mech_kwargs("anthropic/claude-x", None) == {"stop": ["<returncode>"], "num_retries": 3}  # not OpenRouter


def test_the_arm_is_dispatched_and_described(tmp_path, cli, monkeypatch):
    t = fake_t(tmp_path)
    s = A.open_session("olas-ocb-agent:openrouter/m", t, [], 500, approver=lambda s: True, unattended=False)
    assert isinstance(s, O.OcbAgentSession) and not s.unattended and s.kind == "olas-ocb-agent"
    assert A.multi_turn("olas-ocb-agent") and A.priced("olas-ocb-agent-mech") and A.known_kind("olas-ocb-agent")
    assert A.frozen_package("olas-ocb-agent-mech") == "olas_ocb_agent" and A.frozen_package("codex") is None
    assert not A.listings_in_context("olas-ocb-agent") and A.listings_in_context("codex")
    assert Path(A.CLI_BINS["olas-ocb-agent"][1]).name == "ocb-agent"
