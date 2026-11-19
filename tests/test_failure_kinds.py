"""One failure taxonomy for every arm (failures.py), and what an infra trial keeps: the phase it failed in, the
session's measured cost, and a reserve only when the agent ran and its cost is unknown."""

import json
import subprocess
import types

import pytest

from onchainbench import arms as A
from onchainbench import billing, publish, stats, verdict
from onchainbench import failures as F
from onchainbench import run as R
from onchainbench.evm import SendOutcomeUnknown
from onchainbench.forks import FORK_CHAINS
from onchainbench.tasks.base import Result, Task, Turn


# ---- the taxonomy ----------------------------------------------------------------------------------------------
def test_reply_carries_its_kind_and_raw_error():
    r = F.Reply("[error] x", F.PROVIDER_ERROR, "HTTP 502")
    assert r == "[error] x" and isinstance(r, str) and json.dumps(r) == '"[error] x"'
    assert F.failure_of(r) == {"kind": "provider_error", "detail": "HTTP 502"}
    assert F.failure_of("plain text") is None and F.failure_of(F.Reply("fine")) is None
    assert F.failure_of(F.Reply("[timeout] slow", F.TIMEOUT)) == {"kind": "timeout", "detail": "[timeout] slow"}
    assert len(F.failure_of(F.Reply("x", F.HARNESS_ERROR, "y" * 9000))["detail"]) == F.DETAIL_MAX
    with pytest.raises(ValueError, match="unknown turn outcome"):
        F.Reply("x", "oops")
    assert sorted(F.INFRA_KINDS) == ["bench_error", "harness_start", "provider_error"]
    with pytest.raises(ValueError, match="unknown turn outcome"):  # the runner's mark, never an arm's
        F.Reply("x", F.HARNESS_START)
    assert str(F.TurnFailed({"kind": "provider_error", "detail": "502"})) == "provider_error: 502"


@pytest.mark.parametrize(
    "arm,text,provider",
    [
        ("claude-code", "API Error: 502 Bad Gateway", True),
        ("claude-code", "API Error: Connection error.", True),
        ("claude-code", "API Error: 529 {'type':'overloaded_error'}", True),
        ("claude-code", "TypeError: Cannot read properties of undefined", False),
        ("codex", "stream disconnected before completion: error sending request", True),
        ("codex", "exceeded retry limit, last status: 429 Too Many Requests", True),
        ("codex", "unexpected status 503 Service Unavailable", True),
        ("codex", "thread 'main' panicked at src/lib.rs", False),
        ("hermes", "Error code: 502 - {'error': {'message': 'Provider returned error'}}", True),
        ("hermes-raw", "openai.RateLimitError: rate limited", True),
        ("hermes", "Traceback (most recent call last): KeyError: 'tool_calls'", False),
        ("openclaw", "FailoverError: all models failed", True),
        ("openclaw-raw", "ECONNRESET while reading the response", True),
        ("openclaw", "Error: Cannot find module './dist/index.js'", False),
        ("codex", "", False),
        ("", "connection refused", False),  # an exception's message: the bench's own RPC could say that
        ("", "OpenRouter: 502 Bad Gateway", True),
    ],
)
def test_provider_patterns_per_arm(arm, text, provider):
    assert F.provider_text(arm, text) is provider
    if arm:
        assert F.cli_error(arm, text).kind == (F.PROVIDER_ERROR if provider else F.HARNESS_ERROR)


def test_exception_kinds():
    def cls(name, module, base=Exception):
        return type(name, (base,), {"__module__": module})

    assert F.exception_kind(cls("APIConnectionError", "litellm.exceptions")("down")) == F.PROVIDER_ERROR
    assert F.exception_kind(cls("APITimeoutError", "openai")("slow")) == F.PROVIDER_ERROR
    status = cls("APIError", "litellm.exceptions")("odd")
    status.status_code = 503
    assert F.exception_kind(status) == F.PROVIDER_ERROR
    bad = cls("BadRequestError", "litellm.exceptions")("context too long")
    bad.status_code = 400
    assert F.exception_kind(bad) == F.BENCH_ERROR  # not the provider's failure: a CLI reports its own
    # a subclass of a provider class counts too
    assert F.exception_kind(cls("Mine", "mine", cls("RateLimitError", "anthropic"))("x")) == F.PROVIDER_ERROR
    assert F.exception_kind(ConnectionError("OpenRouter: 502 Bad Gateway")) == F.PROVIDER_ERROR
    assert F.exception_kind(ConnectionError("[Errno 61] Connection refused")) == F.BENCH_ERROR
    assert F.exception_kind(SendOutcomeUnknown("HTTP 502 while sending")) == F.BENCH_ERROR  # the bench's own


# Code that isn't the bench's, as a module of another package: what its frames' __name__ says
MYHARNESS = types.ModuleType("myharness.session")
exec(  # the test's stand-in for another package's module
    "import json\n"
    "def crash():\n    raise KeyError('the harness lost its state')\n"
    "def parse():\n    return json.loads('not json')\n"
    "def via_bench():\n    from onchainbench import failures\n    return failures.Reply('x', 'no-such-kind')\n"
    "def provider():\n    raise ConnectionError('OpenRouter: 502 Bad Gateway')\n",
    MYHARNESS.__dict__,
)


def _raised(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - the test inspects it
        return e
    raise AssertionError("did not raise")


def test_an_exception_escaping_a_session_is_the_bench_s_or_the_provider_s():
    """Every contestant is a CLI in its own process, whose session reports how its turn ended: an exception that
    escapes `send` ran in the bench's process, so it is infra (the provider's or the bench's), wherever it was
    raised."""
    assert F.exception_kind(_raised(MYHARNESS.crash)) == F.exception_kind(_raised(MYHARNESS.parse)) == F.BENCH_ERROR
    assert F.exception_kind(_raised(MYHARNESS.via_bench)) == F.BENCH_ERROR
    assert F.exception_kind(_raised(MYHARNESS.provider)) == F.PROVIDER_ERROR
    assert not hasattr(F, "raised_by_harness")


# ---- each CLI arm classifies its own error output ----------------------------------------------------------------
def _fake_t(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / ".mcp.json").write_text("{}")
    fc = types.SimpleNamespace(
        mcp_json=lambda: {
            "mcpServers": {"pearl-connect": {"url": "http://x/mcp/", "headers": {"Authorization": "Bearer tok"}}}
        },
        s=types.SimpleNamespace(eoa_key="0xk", fc=types.SimpleNamespace(chain_id=100), rpc_url="http://rpc"),
    )
    return types.SimpleNamespace(ws=ws, fc=fc, data={}, chain="gnosis")


def _run(monkeypatch, stdout="", stderr=""):
    monkeypatch.setattr(A.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(stdout=stdout, stderr=stderr))


def _timeout(monkeypatch):
    def run(*a, **k):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(A.subprocess, "run", run)


def test_claude_code_turn_outcomes(tmp_path, monkeypatch):
    t = _fake_t(tmp_path)
    s = A.ClaudeCodeSession(t, None)
    _run(monkeypatch, stderr="API Error: 502 Bad Gateway")
    assert s.send("x").kind == F.PROVIDER_ERROR
    _run(monkeypatch, stderr="TypeError: undefined")
    r = s.send("x")
    assert r.kind == F.HARNESS_ERROR and r.startswith("[error] TypeError") and r.detail == "TypeError: undefined"

    def result(**kw):
        return json.dumps({"type": "result", "session_id": "s", "total_cost_usd": 0.1, **kw})

    _run(monkeypatch, result(is_error=True, subtype="success", result="API Error: 529 overloaded_error"))
    r = s.send("x")
    assert r.kind == F.PROVIDER_ERROR and r == "API Error: 529 overloaded_error" and s.stats.cost_usd == 0.1
    _run(monkeypatch, result(is_error=True, subtype="error_during_execution", result="crashed"))
    assert s.send("x").kind == F.HARNESS_ERROR
    _run(monkeypatch, result(is_error=True, subtype="error_max_turns", result="stopped"))
    assert F.failure_of(s.send("x")) is None  # a cap it stopped at: the turn ended normally
    _timeout(monkeypatch)
    r = s.send("x")
    assert r.kind == F.TIMEOUT and r.startswith("[timeout]") and s.transcript()[0][-1]["kind"] == "error"


def test_claude_code_events_whose_message_is_a_string_are_read_not_a_bench_error(tmp_path, monkeypatch):
    # some stream-json events (a system notice, an API retry) carry `message` as a plain string: reading the turn
    # used to fail with AttributeError and was scored as the bench's error
    s = A.ClaudeCodeSession(_fake_t(tmp_path), None)
    text = {"type": "text", "text": "the answer"}
    _run(
        monkeypatch,
        "\n".join(
            json.dumps(e)
            for e in (
                {"type": "system", "subtype": "api_retry", "message": "retrying in 2s"},
                {"type": "assistant", "message": {"id": "m1", "content": [text]}},
                {"type": "assistant", "message": "a plain string"},
                {"type": "result", "session_id": "s", "subtype": "error_max_turns", "result": ""},
            )
        ),
    )
    # the string is the assistant's text: kept in the transcript, and the answer when it's the last message
    assert s.send("x") == "a plain string" and s.stats.model_calls == 1
    assert [(e["kind"], e["text"]) for e in s.transcript()[0]] == [
        ("user", "x"),
        ("assistant", "the answer"),
        ("assistant", "a plain string"),
    ]


def test_codex_turn_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr("onchainbench.pricing.openrouter_prices", lambda: {})
    s = A.CodexSession(_fake_t(tmp_path), "openrouter/m")
    events = [
        {"type": "error", "message": "stream disconnected before completion"},
        {"type": "turn.failed", "error": {"message": "exceeded retry limit, last status: 429"}},
        {"type": "turn.failed"},
    ]
    _run(monkeypatch, "\n".join(map(json.dumps, events)))
    r = s.send("x")
    assert r.kind == F.PROVIDER_ERROR and "exceeded retry limit" in r.detail and "turn.failed" in r.detail
    _run(monkeypatch, "", "thread 'main' panicked")
    assert s.send("x").kind == F.HARNESS_ERROR
    _timeout(monkeypatch)
    assert s.send("x").kind == F.TIMEOUT


def test_a_codex_turn_that_spoke_then_hit_the_rate_limit_is_the_provider_s(tmp_path, monkeypatch):
    """A message, then Codex's "exceeded retry limit ... 429": a provider error (infra), not the message scored as
    the answer. A warning, or a retry that recovered, leaves the answer the turn's."""
    monkeypatch.setattr("onchainbench.pricing.openrouter_prices", lambda: {})
    s = A.CodexSession(_fake_t(tmp_path), "openrouter/m")
    said = {"type": "item.completed", "item": {"type": "agent_message", "text": "Checking the balance first."}}
    limit = {"type": "turn.failed", "error": {"message": "exceeded retry limit, last status: 429 Too Many Requests"}}
    done = {"type": "turn.completed", "usage": {}}
    _run(monkeypatch, "\n".join(map(json.dumps, [said, {"type": "error", "message": "Reconnecting... 5/5"}, limit])))
    r = s.send("x")
    assert r.kind == F.PROVIDER_ERROR and r == "Checking the balance first." and "429" in r.detail
    warn = {"type": "error", "message": "Model metadata for `m` not found. Defaulting to fallback metadata"}
    for events in ([warn, said, done], [said, warn]):
        _run(monkeypatch, "\n".join(map(json.dumps, events)))
        r = s.send("x")
        assert r == "Checking the balance first." and getattr(r, "kind", F.OK) == F.OK
    _run(monkeypatch, "\n".join(map(json.dumps, [said, limit, said, done])))  # it recovered and finished
    assert getattr(s.send("x"), "kind", F.OK) == F.OK
    # how the turn ended is its last error: a 429 it retried past, then another failure, is the harness's
    retried = {"type": "error", "message": "Reconnecting... 1/5 (429 Too Many Requests)"}
    broke = {"type": "turn.failed", "error": {"message": "tool call failed: invalid arguments"}}
    for events in ([retried, broke], [said, retried, broke]):
        _run(monkeypatch, "\n".join(map(json.dumps, events)))
        r = s.send("x")
        assert getattr(r, "kind", F.OK) != F.PROVIDER_ERROR
    _run(monkeypatch, "\n".join(map(json.dumps, [retried, broke])))
    r = s.send("x")
    assert r.kind == F.HARNESS_ERROR and "invalid arguments" in r.detail and "429" in r.detail


def test_hermes_and_openclaw_turn_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr("onchainbench.pricing.openrouter_prices", lambda: {})
    h = A.HermesSession(_fake_t(tmp_path / "h"), "hermes", "openrouter/m", 30)
    _run(monkeypatch, json.dumps({"type": "result", "error": "Error code: 502 - Provider returned error"}))
    r = h.send("x")
    assert r.kind == F.PROVIDER_ERROR and r == "[error] Error code: 502 - Provider returned error"
    _run(monkeypatch, "", "Traceback: KeyError")
    assert h.send("x").kind == F.HARNESS_ERROR
    _timeout(monkeypatch)
    assert h.send("x").kind == F.TIMEOUT
    o = A.OpenClawSession(_fake_t(tmp_path / "o"), "openclaw", "openrouter/m", 30)
    o._mcp_server = lambda: {}
    _run(monkeypatch, json.dumps({"ok": False, "error": {"type": "cli_error", "message": "FailoverError: x"}}))
    r = o.send("x")
    assert r.kind == F.PROVIDER_ERROR and r.detail == "FailoverError: x"
    _run(monkeypatch, json.dumps({"ok": False, "error": "Cannot find module"}), "node: exit 1")
    r = o.send("x")
    assert r.kind == F.HARNESS_ERROR and r.detail == "Cannot find module\nnode: exit 1"
    _timeout(monkeypatch)
    assert o.send("x").kind == F.TIMEOUT


# olas-ocb-agent's turn outcomes: test_ocb_cli.py


# ---- the runner: one rule for every arm ------------------------------------------------------------------------
class FakeFork:
    def __init__(self):
        self.rpc = types.SimpleNamespace(block_number=lambda: 0)


class FakeFC:
    def __init__(self, env):
        self.env = env
        self.s = types.SimpleNamespace(
            integrity_log=[],
            start_block=0,
            ledger=[],
            tools_listed=1,
            safe="0x" + "5a" * 20,
            eoa=env.eoa,
            rpc_url="http://127.0.0.1:1/rpc",
            raw_sends=False,
            tokens={},
            send_unknown=list(env.send_unknown),
        )
        self.token = "tok"

    def mcp_json(self):
        url = "http://127.0.0.1:1/mcp/"
        return {"mcpServers": {"pearl-connect": {"type": "http", "url": url, "headers": {"Authorization": "Bearer t"}}}}

    def reset_watch(self, *a):
        pass

    def begin_window(self):
        pass

    def checkpoint(self):
        pass

    def integrity(self):
        if self.env.integrity_raises:
            raise ConnectionError("anvil RPC dropped while the integrity check scanned blocks")
        return []

    def watch_tokens(self):
        return []

    def stop(self):
        pass


class FakeEnv:
    def __init__(self, fresh_raises=False, integrity_raises=False, send_unknown=()):
        self.integrity_raises, self.fresh_raises, self.send_unknown = integrity_raises, fresh_raises, send_unknown
        self.fc, self.eoa, self.fork = FORK_CHAINS["gnosis"], "0x" + "e0" * 20, FakeFork()

    def fresh(self):
        if self.fresh_raises:
            raise RuntimeError("upstream archive RPC refused the fork")

    def connect(self, **kw):
        return FakeFC(self)

    def alive(self):
        return True


class FakeEnvs:
    def __init__(self, **kw):
        self.env = FakeEnv(**kw)

    def get(self, chain):
        return self.env


class Fixed(A.Session):
    """A session whose turns return (or raise) the given outcomes in order."""

    def __init__(self, *outs, cost=0.3, calls=0, close_raises=False):
        self.stats, self.outs, self.sent, self.close_raises = (
            A.Stats(cost_usd=cost, model_calls=calls),
            outs,
            [],
            close_raises,
        )

    def send(self, text):
        self.sent.append(text)
        out = self.outs[min(len(self.sent), len(self.outs)) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    def close(self):
        if self.close_raises:
            raise RuntimeError("the session's process is gone")
        return self.stats

    def transcript(self):
        return [{"turn": 0, "kind": "user", "text": "x"}], ["raw"]


PASSES = Task("x-task", "A", "f", "gnosis", "do it", lambda t: Result(True, "done", (1, 1)))
KEYS = ("passed", "infra", "phase", "cost_usd", "cost_source", "cost_reserved_usd")


@pytest.fixture(autouse=True)
def _no_judge(monkeypatch):
    monkeypatch.setattr(R, "assess_interventions", lambda t, rec, cap: {"stall": None, "unplanned": 0})


def _trial(monkeypatch, arm, session, task=PASSES, envs=None, sessions_dir=None):
    monkeypatch.setattr(A, "open_session", lambda *a, **k: session)
    return R.trial(arm, task, 0, envs or FakeEnvs(), sessions_dir=sessions_dir)


def test_a_cli_provider_error_and_a_raised_one_make_the_same_record(tmp_path, monkeypatch):
    """The repro: Hermes printing a 502 was scored, a session raising it was infra with a $2 reserve."""
    monkeypatch.setattr("onchainbench.pricing.openrouter_prices", lambda: {})
    monkeypatch.setattr(A, "stop_home_processes", lambda home: [])
    _run(monkeypatch, json.dumps({"type": "result", "error": "Error code: 502 - Bad Gateway"}))
    cli = R.trial("hermes:openrouter/x", PASSES, 0, FakeEnvs())  # the real Hermes arm, its CLI faked
    raised = _trial(
        monkeypatch, "olas-ocb-agent:openrouter/x", Fixed(ConnectionError("OpenRouter: 502 Bad Gateway"), cost=0.0)
    )
    assert (
        {k: cli.get(k) for k in KEYS}
        == {k: raised.get(k) for k in KEYS}
        == {
            "passed": False,
            "infra": True,
            "phase": "agent",
            "cost_usd": 0.0,
            "cost_source": "estimate",
            "cost_reserved_usd": None,
        }
    )
    assert cli["failure"]["kind"] == raised["failure"]["kind"] == F.PROVIDER_ERROR
    assert "502" in cli["failure"]["detail"] and "502" in raised["failure"]["detail"]  # the raw text, for the audit
    assert cli["reason"].startswith("provider error: ") and "traceback" in raised and "traceback" not in cli


@pytest.mark.parametrize("arm", ["claude-code:x", "codex:x", "hermes:x", "openclaw:x", "olas-ocb-agent:x"])
def test_a_harness_crash_and_a_timeout_are_scored_failures_for_every_arm(arm, monkeypatch):
    crash = _trial(monkeypatch, arm, Fixed(F.cli_error(arm.split(":")[0], "Segmentation fault")))
    assert not crash["infra"] and not crash["passed"] and crash["failure"]["kind"] == F.HARNESS_ERROR
    assert crash["reason"] == "harness error: Segmentation fault" and crash["checker_passed"] is True
    assert crash["failure"]["detail"] == "Segmentation fault" and "phase" not in crash
    late = _trial(monkeypatch, arm, Fixed(F.timed_out("the run passed the wall clock")))
    assert not late["infra"] and not late["passed"] and late["failure"]["kind"] == F.TIMEOUT
    assert late["reason"] == "timeout: the run passed the wall clock" and late["cost_usd"] == 0.3


@pytest.mark.parametrize("arm", ["claude-code:x", "openclaw:x", "olas-ocb-agent:x", "oracle"])
def test_funds_moved_through_the_eoa_fail_the_trial_whatever_the_checker_says(arm, monkeypatch):
    """One rule in the runner, for every task and every arm: the checker passed a route that moved the Safe's tokens
    through the EOA (smoke-matrix-4's a-wrap-unwrap); the trial fails, with the reason, and the checker's own verdict
    is kept beside it."""
    eoa, wx = "0x" + "e0" * 20, "0x" + "77" * 20
    out = {"kind": "tx", "sender": "safe", "status": "success", "to": wx, "value": 0, "data": "0x", "tx_hash": "0xa1"}
    out["diff"] = {"tokens": {wx: {"delta": "-2"}}, "recipients": [{"address": eoa}]}
    task = Task("x-task", "A", "f", "gnosis", "do it", PASSES.check, setup=lambda t: t.ledger.append(out))
    r = _trial(monkeypatch, arm, Fixed("done"), task=task)
    assert not r["infra"] and not r["passed"] and r["checker_passed"] is True and r["checker_reason"] == "done"
    assert r["reason"] == "funds outside the Safe: the Safe sent tokens to the EOA (tx 0xa1)"
    assert r["funds_outside_safe"] == ["the Safe sent tokens to the EOA (tx 0xa1)"]
    # a harness failure on top: its reason, and still the checker's own verdict
    late = _trial(monkeypatch, arm, Fixed(F.timed_out("past the wall clock")), task=task)
    assert late["reason"] == "timeout: past the wall clock" and late["checker_passed"] is True
    assert late["funds_outside_safe"] == r["funds_outside_safe"]
    clean = _trial(monkeypatch, arm, Fixed("done"))
    assert clean["passed"] and "funds_outside_safe" not in clean and "checker_passed" not in clean


class Calls(Fixed):
    """A session whose turn runs a function in the bench's process."""

    def __init__(self, fn, **kw):
        super().__init__("unused", **kw)
        self.fn = fn

    def send(self, text):
        self.sent.append(text)
        return self.fn()


@pytest.mark.parametrize("arm", ["claude-code:x", "olas-ocb-agent:x", "hermes:x"])
def test_an_exception_escaping_any_arm_s_session_is_infra(monkeypatch, arm):
    """No contestant code runs in the bench's process: an exception out of a session is never scored."""
    r = _trial(monkeypatch, arm, Calls(MYHARNESS.crash))
    assert r["infra"] and r["failure"]["kind"] == F.BENCH_ERROR and r["phase"] == "agent"
    r = _trial(monkeypatch, arm, Calls(MYHARNESS.provider))
    assert r["infra"] and r["failure"]["kind"] == F.PROVIDER_ERROR


@pytest.mark.parametrize("arm", ["claude-code:x", "oracle", "olas-ocb-agent:x"])
def test_one_wall_clock_for_every_arm_enforced_by_the_runner(arm, monkeypatch):
    """A turn that ran past A.TURN_SECONDS is a timeout, whichever arm ran it, and the
    trial records the clock it had."""
    r = _trial(monkeypatch, arm, Fixed("DONE"))
    assert r["passed"] and r["turn_seconds"] == A.TURN_SECONDS == 900
    monkeypatch.setattr(A, "TURN_SECONDS", -1)  # every turn takes longer than this
    r = _trial(monkeypatch, arm, Fixed("DONE"))
    assert not r["infra"] and not r["passed"] and r["failure"]["kind"] == F.TIMEOUT
    assert "past the -1 s wall clock" in r["reason"] and r["answer"] == "DONE"
    # a turn that already failed keeps its own kind
    r = _trial(monkeypatch, arm, Fixed(F.Reply("[error] x", F.HARNESS_ERROR, "x")))
    assert r["failure"]["kind"] == F.HARNESS_ERROR


REAL_HARNESS_PREFLIGHT = R.harness_preflight  # conftest stubs it for every test


def test_a_missing_harness_refuses_the_run_before_any_trial(tmp_path, monkeypatch, capsys):
    """A CLI that isn't installed failed every trial of its arm as infra."""
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    bins = {k: (name, tmp_path / name) for k, (name, _) in A.CLI_BINS.items()}
    monkeypatch.setattr(A, "CLI_BINS", bins)
    assert "`claude` is neither on PATH nor at" in A.harness_problem("claude-code")
    (tmp_path / "codex").write_text("")  # the vendored one install-arms.sh puts there
    assert A.harness_problem("codex") is None and A.harness_problem("oracle") is None
    assert A.cli_bin("codex") == str(tmp_path / "codex")
    monkeypatch.setattr(A.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert A.harness_problem("hermes-raw") is None and A.cli_bin("openclaw") == "/usr/bin/openclaw"
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    # the Python CLIs: present, and new enough for the arm's command line
    (tmp_path / "ocb-agent").write_text("")
    with monkeypatch.context() as m:
        m.setattr(A, "cli_package", lambda kind: {"version": "0.0.9", "dir": "/x"})
        assert A.harness_problem("olas-ocb-agent") == "its CLI is olas-ocb-agent 0.0.9; the arm needs 0.1.0 or later"
        m.setattr(A, "cli_package", lambda kind: {"version": A.OCB_AGENT_VERSION, "dir": "/x"})
        assert A.harness_problem("olas-ocb-agent-mech") is None
        # olas-ocb-agent at exactly its pin: another version found first never runs silently
        m.setattr(A, "cli_package", lambda kind: {"version": "9.9.9", "dir": "/x"})
        assert "olas-ocb-agent 9.9.9" in A.harness_problem("olas-ocb-agent") and "not the pinned" in (
            A.harness_problem("olas-ocb-agent") or ""
        )
        m.setattr(A, "BENCH_BIN", tmp_path / "bench-bin")  # the bench's environment has one too
        (tmp_path / "bench-bin").mkdir()
        (tmp_path / "bench-bin" / "ocb-agent").write_text("")
        (tmp_path / "bench-bin" / "hermes").write_text("")
        assert A.cli_bin("olas-ocb-agent") == str(tmp_path / "ocb-agent")  # the pinned vendored install first
        assert A.cli_bin("hermes") == str(tmp_path / "bench-bin" / "hermes")  # the bench's environment before PATH

        def unreadable(kind):
            raise RuntimeError("no package")

        m.setattr(A, "cli_package", unreadable)
        assert A.harness_problem("olas-ocb-agent") == "no package"
    assert REAL_HARNESS_PREFLIGHT(["oracle", "claude-code:m"]) == [
        f"claude-code:m: its CLI `claude` is neither on PATH nor at {tmp_path / 'claude'} (scripts/install-arms.sh "
        "installs it)",
    ]
    # the runner refuses the run, beside the judge preflight, before it writes anything
    monkeypatch.setattr(R, "harness_preflight", REAL_HARNESS_PREFLIGHT)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    assert R.main(["--arms", "oracle,claude-code:m", "--split", "open", "--limit", "1"]) == 2
    assert "refusing the run: an arm's harness isn't installed: claude-code:m" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


REAL_HARNESS_VERSIONS = R.harness_versions  # conftest stubs it too


def test_every_arm_s_harness_version_is_read_at_preflight(tmp_path, monkeypatch, capsys):
    """Only the frozen agent's version was recorded and published; now every arm's, whoever made it."""
    from importlib.metadata import version

    replies = {  # what each faked CLI prints for --version (a shell script can't run on Windows)
        "claude": (0, "\n2.1.7 (Claude Code)\n", ""),
        "codex": (3, "", "unknown flag\n"),
    }

    def run(cmd, **kw):
        assert cmd[1:] == ["--version"] and kw["timeout"] == A.VERSION_SECONDS
        if cmd[0].endswith("slow"):
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        if (name := cmd[0].rsplit("/", 1)[-1]) not in replies:
            raise FileNotFoundError(cmd[0])
        rc, out, err = replies[name]
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)

    monkeypatch.setattr(A.subprocess, "run", run)
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    bins = {"claude-code": ("claude", "/v/claude"), "codex": ("codex", "/v/codex"), "hermes": ("hermes", "/v/slow")}
    monkeypatch.setattr(A, "CLI_BINS", {**A.CLI_BINS, **bins, "openclaw": ("openclaw", "/v/missing")})
    assert A.harness_version("claude-code") == "2.1.7 (Claude Code)"
    with pytest.raises(RuntimeError, match=r"`codex --version` failed \(exit 3\): unknown flag"):
        A.harness_version("codex")
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        A.harness_version("openclaw")
    with pytest.raises(RuntimeError, match="TimeoutExpired"):
        A.harness_version("hermes")
    assert A.harness_version("oracle") == version("onchainbench")
    monkeypatch.setattr(A, "cli_package", lambda kind: {"version": "0.3.10", "dir": "/x"})
    assert A.harness_version("olas-ocb-agent") == "0.3.10"  # a Python CLI: its package's version
    with pytest.raises(ValueError, match="unknown arm kind"):
        A.harness_version("no-such-arm")
    versions, problems = REAL_HARNESS_VERSIONS(["oracle", "claude-code:m", "olas-ocb-agent:x", "codex:m"])
    want = {"oracle": version("onchainbench"), "claude-code:m": "2.1.7 (Claude Code)", "olas-ocb-agent:x": "0.3.10"}
    assert versions == want and [p.split(":")[0] for p in problems] == ["codex"]
    # the runner refuses the run before it writes anything; a run that starts records every arm's version
    monkeypatch.setattr(R, "harness_versions", REAL_HARNESS_VERSIONS)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    assert R.main(["--arms", "oracle,codex:openrouter/openai/gpt-5", "--split", "open", "--limit", "1"]) == 2
    assert "refusing the run: an arm's harness version can't be read: codex:openrouter" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()
    cfg = {"harness_versions": {"claude-code:m": "2.1.7"}}
    assert R.stamp("claude-code:m", cfg, None)["harness_version"] == "2.1.7"


def test_the_cli_arms_share_the_wall_clock():
    assert A.HARNESS_TURN_SECONDS < A.TURN_SECONDS  # Hermes and OpenClaw stop themselves inside it, and report
    assert A.openclaw_config({}, "/ws")["agents"]["defaults"]["timeoutSeconds"] == A.HARNESS_TURN_SECONDS == 840
    assert str(A.TURN_SECONDS) in A.PAST_WALL_CLOCK


def test_a_turn_lost_to_infra_ends_the_trial_and_keeps_the_transcript(tmp_path, monkeypatch):
    task = Task("x-turns", "D", "f", "gnosis", "day 1", lambda t: Result(True), turns=[Turn("day 2"), Turn("day 3")])
    s = Fixed("fine", F.Reply("[error] 503", F.PROVIDER_ERROR, "HTTP 503"), "never")
    r = _trial(monkeypatch, "claude-code:x", s, task, sessions_dir=tmp_path)
    assert s.sent == ["day 1", "day 2"] and r["infra"] and r["turns"] == 2 and r["answers"] == ["fine", "[error] 503"]
    assert r["transcript_events"] == 1 and (tmp_path / "transcripts").is_dir()
    # a harness error or a timeout ends the trial too (no later turns are paid for), and it stays a scored failure
    for bad in (F.Reply("[error] x", F.HARNESS_ERROR, "x"), F.timed_out("late")):
        s = Fixed("fine", bad, "ok")
        r = _trial(monkeypatch, "claude-code:x", s, task)
        assert s.sent == ["day 1", "day 2"] and not r["infra"] and not r["passed"] and r["turns"] == 2
        assert r["failure"]["kind"] == bad.kind and r["checker_passed"] is True


def test_a_harness_error_ends_a_persona_trial_before_the_simulated_user_answers(monkeypatch):
    """smoke-matrix-2: Hermes ran 8 turns after a harness error, and the simulated user volunteered the answer."""
    replies = []

    class Persona:
        opening, transcript, cost = "hi", [], 0.0

        def __init__(self, t):
            pass

        def reply(self, msg):
            replies.append(msg)
            return "go on"

    task = Task("x-e2", "E", "f", "gnosis", "x", lambda t: Result(True), persona=Persona)
    s = Fixed(F.Reply("[error] crashed", F.HARNESS_ERROR, "crashed"), "never")
    r = _trial(monkeypatch, "claude-code:x", s, task)
    assert s.sent == ["hi"] and replies == [] and r["turns"] == 1
    assert not r["infra"] and not r["passed"] and r["failure"]["kind"] == F.HARNESS_ERROR


class Sandboxed(Fixed):
    """A session whose turn records one tool result (as a CLI's transcript does)."""

    def __init__(self, *outs, tool_output="ok", **kw):
        super().__init__(*outs, **kw)
        self.events, self.tool_output = [], tool_output

    def send(self, text):
        out = super().send(text)
        self.events += [{"turn": 0, "kind": "user", "text": text}, {"kind": "tool_result", "text": self.tool_output}]
        return out

    def transcript(self):
        return self.events, ["raw"]


SANDBOX_ERRORS = [
    "bwrap: Can't mkdir parents for /tmp/ocb-iso-1/home: Permission denied",
    "sandbox-exec: sandbox_apply: Operation not permitted",
    "[sandbox error] the shell's sandbox couldn't start",
]


@pytest.mark.parametrize("arm", ["olas-ocb-agent:x", "claude-code:x", "my-harness:x"])
@pytest.mark.parametrize("text", SANDBOX_ERRORS)
def test_a_sandbox_that_couldnt_start_is_the_bench_s_failure_for_every_arm(arm, text, monkeypatch):
    """smoke-matrix-2: olas-ocb-agent's 7 bwrap failures scored as the agent's, with no failure kind."""
    r = _trial(monkeypatch, arm, Sandboxed("I couldn't run anything.", tool_output=text))
    assert r["infra"] and r["phase"] == "agent" and r["failure"]["kind"] == F.BENCH_ERROR
    assert text in r["failure"]["detail"] and "sandbox couldn't start" in r["failure"]["detail"]
    # a harness error the sandbox caused is the bench's too; a provider error keeps its kind
    r = _trial(monkeypatch, arm, Sandboxed(F.Reply("[error] x", F.HARNESS_ERROR, "x"), tool_output=text))
    assert r["infra"] and r["failure"]["kind"] == F.BENCH_ERROR
    r = _trial(monkeypatch, arm, Sandboxed(F.Reply("[error] 502", F.PROVIDER_ERROR, "502"), tool_output=text))
    assert r["failure"]["kind"] == F.PROVIDER_ERROR


def test_only_a_sandbox_start_up_error_counts():
    assert F.sandbox_failure([{"kind": "tool_result", "text": "  bwrap: execvp sh: No such file"}])
    assert F.sandbox_failure([{"kind": "error", "text": "[sandbox error] the shell's sandbox couldn't start"}])
    for text in ("the docs mention bwrap: a sandbox", "ok\nbwrap: late", "bwrap:", "sandbox-exec is on PATH", ""):
        assert F.sandbox_failure([{"kind": "tool_result", "text": text}]) is None
    assert F.sandbox_failure([{"kind": "assistant", "text": "bwrap: Can't mkdir parents"}]) is None  # the model said it
    assert F.sandbox_failure([]) is None


def test_a_provider_error_from_the_simulated_user_ends_a_persona_trial(monkeypatch):
    class Persona:
        opening, transcript, cost = "hi", [], 0.0

        def __init__(self, t):
            pass

        def reply(self, msg):
            return "go on"

    task = Task("x-e", "E", "f", "gnosis", "x", lambda t: Result(True), persona=Persona)
    s = Fixed("asking", F.Reply("[error] 502", F.PROVIDER_ERROR, "502"), "never")
    r = _trial(monkeypatch, "claude-code:x", s, task)
    assert s.sent == ["hi", "go on"] and r["infra"] and r["turns"] == 2


# ---- what an infra trial keeps (phase, cost, reserve) -----------------------------------------------------------
def test_an_infra_failure_keeps_the_measured_cost_and_reserves_only_when_it_is_unknown(monkeypatch):
    # the repro: integrity() raising after a $0.90 session kept $0 and reserved $2
    r = _trial(monkeypatch, "olas-ocb-agent:x", Fixed("DONE", cost=0.9, calls=7), envs=FakeEnvs(integrity_raises=True))
    assert (r["phase"], r["cost_usd"], r["cost_source"], r.get("cost_reserved_usd")) == ("check", 0.9, "estimate", None)
    assert r["model_calls"] == 7 and r["answer"] == "DONE" and r["failure"]["kind"] == F.BENCH_ERROR
    # a fork that never came up: nothing was spent, nothing reserved
    r = _trial(monkeypatch, "olas-ocb-agent:x", Fixed("DONE"), envs=FakeEnvs(fresh_raises=True))
    assert (r["phase"], r["cost_usd"], r["cost_source"], r.get("cost_reserved_usd")) == ("setup", 0.0, "none", None)
    assert "upstream archive RPC refused the fork" in r["failure"]["detail"]

    def no_session(*a, **k):
        raise RuntimeError("the session could not start")

    monkeypatch.setattr(A, "open_session", no_session)
    r = R.trial("olas-ocb-agent:x", PASSES, 0, FakeEnvs())
    assert r["phase"] == "setup" and r.get("cost_reserved_usd") is None  # no session opened: nothing ran
    # the session raising: the stats it closes with are kept
    r = _trial(monkeypatch, "olas-ocb-agent:x", Fixed(RuntimeError("adapter bug"), cost=0.4))
    assert (r["phase"], r["cost_usd"], r.get("cost_reserved_usd")) == ("agent", 0.4, None)
    assert r["failure"]["kind"] == F.BENCH_ERROR and r["reason"] == "RuntimeError: adapter bug"
    # the session couldn't even close: the agent ran and its cost is unknown, so a full trial is reserved
    r = _trial(monkeypatch, "olas-ocb-agent:x", Fixed(RuntimeError("x"), close_raises=True))
    assert (r["phase"], r["cost_usd"], r["cost_source"]) == ("agent", 0.0, "unknown")
    assert r["cost_reserved_usd"] == A.TRIAL_CAP_USD
    # a scripted arm spends nothing: no cost fields added
    r = _trial(monkeypatch, "oracle", Fixed(RuntimeError("x"), close_raises=True))
    assert r["infra"] and "cost_source" not in r and "cost_reserved_usd" not in r


def test_a_send_with_an_unknown_outcome_makes_the_trial_infra(monkeypatch):
    r = _trial(monkeypatch, "olas-ocb-agent:x", Fixed("DONE"), envs=FakeEnvs(send_unknown=["safe_transaction: lost"]))
    assert (
        r["infra"]
        and r["phase"] == "agent"
        and r["failure"]
        == {
            "kind": F.BENCH_ERROR,
            "detail": "SendOutcomeUnknown: safe_transaction: lost",
        }
    )

    def check(t):
        raise SendOutcomeUnknown("send outcome unknown: the checker's harness send")

    r = _trial(monkeypatch, "olas-ocb-agent:x", Fixed("DONE"), Task("x-c", "A", "f", "gnosis", "x", check))
    assert r["infra"] and r["phase"] == "check" and "checker_error" not in r  # not scored as a checker crash


def test_billing_shares_spend_over_infra_trials_that_measured_tokens():
    arm = "codex:openrouter/m"
    rows = [
        {"arm": arm, "cost_usd": 0.1, "prompt_tokens": 100, "infra": False},
        {"arm": arm, "cost_usd": 0.1, "prompt_tokens": 100, "infra": True, "phase": "check"},  # the session ran
        {"arm": arm, "cost_usd": 0.0, "cost_source": "none", "infra": True, "phase": "setup"},  # nothing ran
    ]
    shards = [{"name": "s", "arms": [arm], "start": 0.0, "billed": 0.3, "rows": rows}]
    out = billing.apportion(rows, shards, 0.3)
    assert out[arm]["status"] == "apportioned" and out[arm]["trials"] == 2
    assert rows[1]["cost_usd"] == pytest.approx(0.15) and rows[2]["cost_usd"] == 0.0


# ---- the 10% flag ----------------------------------------------------------------------------------------------
def _rec(arm, task, passed=False, **kw):
    return {"arm": arm, "task": task, "attempt": 0, "chain": "gnosis", "block": "A", "family": task,
            "passed": passed, "infra": False, "seconds": 1, "cost_usd": 0.01, "metrics": {}, **kw}  # fmt: skip


def test_the_infra_rate_is_flagged_above_ten_percent(tmp_path):
    rs = [_rec("a", f"t{i}", True) for i in range(8)]
    rs += [_rec("a", "t8", infra=True, phase="setup", failure={"kind": "bench_error", "detail": "x"})]
    rs += [_rec("a", "t9", infra=True, phase="agent", failure={"kind": "provider_error", "detail": "502"})]
    rs += [_rec("a", "t10", unsupported=True)]  # out of the rate's base
    rs += [_rec("a", "t11", failure={"kind": "timeout", "detail": "slow"})]
    s = stats.arm_summary(rs)
    assert s["infra_rate"] == round(2 / 11, 3) and s["infra_high"] is True
    assert s["infra_by_phase"] == {"setup": 1, "agent": 1}
    assert s["failure_kinds"] == {"bench_error": 1, "provider_error": 1, "timeout": 1}
    low = stats.arm_summary([*rs[:8], {**rs[8], "phase": None}])
    assert low["infra_rate"] == round(1 / 9, 3) and low["infra_high"] is True
    assert stats.arm_summary(rs[:8])["infra_high"] is False and stats.arm_summary([rs[10]])["infra_rate"] == 0.0
    assert low["infra_by_phase"] == {"unknown": 1}  # older records have no phase
    v = verdict.verdict([*rs, *(_rec("b", r["task"], True) for r in rs[:8])])
    md = verdict.markdown(v)
    assert "- infra rate 18.2%: above 10%, check the failure audit before reading these results" in md
    assert "- infra rate 0.0%\n" in md
    assert publish.entry("a", rs, {}, True)["infra_high"] is True


# ---- a killed turn: its partial stream kept; a timeout with no model call is infra ---------------------------------
def _killed(monkeypatch, partial, export=None):
    """subprocess.run as a CLI killed at the wall clock, having printed `partial`; OpenClaw's export answers too."""

    def run(cmd, **k):
        if export is not None and "export-trajectory" in cmd:
            return types.SimpleNamespace(stdout=json.dumps({"outputDir": str(export)}), stderr="")
        raise subprocess.TimeoutExpired(cmd, k.get("timeout") or 1, output=partial)

    monkeypatch.setattr(A.subprocess, "run", run)


def test_a_killed_claude_code_turn_keeps_its_stream_calls_and_usage(tmp_path, monkeypatch):
    t = _fake_t(tmp_path)
    s = A.ClaudeCodeSession(t, None)
    msg = {"id": "m1", "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}
    msg["usage"] = {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 3}
    lines = [{"type": "assistant", "message": msg}, {"type": "assistant", "message": msg}]  # one message, two blocks
    _killed(monkeypatch, "\n".join(map(json.dumps, lines)).encode())  # bytes, as some platforms give it
    r = s.send("x")
    assert r.kind == F.TIMEOUT and s.model_calls_so_far() == 1
    assert (s.stats.prompt_tokens, s.stats.completion_tokens, s.stats.extra["cached_tokens"]) == (15, 3, 5)
    kinds = [e["kind"] for e in s.transcript()[0]]
    assert kinds == ["user", "tool_call", "tool_call", "error"] and "[timeout]" in s.transcript()[1][-1]
    monkeypatch.setattr("onchainbench.pricing.price", lambda m: {"input": 1e-6, "output": 1e-6, "cache_read": 1e-7})
    o = A.ClaudeCodeSession(t, "openrouter/x/y")
    o.send("x")
    assert o.stats.extra["cost_source"] == "estimate" and o.stats.cost_usd > 0  # priced from the partial usage
    _killed(monkeypatch, "")  # killed before it printed anything: known, no call in the turn
    assert s.send("x").kind == F.TIMEOUT and s.model_calls_so_far() == 1


def test_killed_codex_and_hermes_turns_count_what_they_streamed(tmp_path, monkeypatch):
    t = _fake_t(tmp_path)
    monkeypatch.setattr(A, "TrialKey", lambda t: types.SimpleNamespace(mode="between-turns", env=dict, close=dict))
    codex = A.CodexSession(t, "openrouter/x/y")
    item = {"id": "c1", "type": "command_execution", "command": "ls"}
    _killed(monkeypatch, json.dumps({"type": "item.started", "item": item}))
    assert codex.send("x").kind == F.TIMEOUT and codex.model_calls_so_far() == 1
    assert codex.transcript()[0][0]["kind"] == "user"
    _killed(monkeypatch, "")  # the codex/flash b-sell-position case: nothing streamed
    assert codex.send("x").kind == F.TIMEOUT and codex.model_calls_so_far() == 1
    monkeypatch.setattr(A, "stop_home_processes", lambda home: [])
    hermes = A.HermesSession(t, "hermes", "openrouter/x/y", 30)
    _killed(monkeypatch, json.dumps({"type": "text", "text": "Looking at the safe"}))
    assert hermes.send("x").kind == F.TIMEOUT and hermes.model_calls_so_far() == 1
    assert [e["kind"] for e in hermes.transcript()[0]] == ["user", "assistant", "error"]


def test_a_killed_openclaw_turn_is_recorded_from_its_export(tmp_path, monkeypatch):
    t = _fake_t(tmp_path)
    key = types.SimpleNamespace(mode="between-turns", env=lambda: {"OPENROUTER_API_KEY": ""}, close=dict)
    monkeypatch.setattr(A, "TrialKey", lambda t: key)
    monkeypatch.setattr(A, "openclaw_config", lambda *a, **k: {})
    monkeypatch.setattr("onchainbench.raw_key.broker_for", lambda t: ("u", "t"))
    out = tmp_path / "export"
    out.mkdir()
    ev = {"source": "transcript", "type": "assistant.message", "data": {"message": {"content": "checking"}}}
    (out / "events.jsonl").write_text(json.dumps(ev) + "\n")
    s = A.OpenClawSession(t, "openclaw", "openrouter/x/y", 30)
    _killed(monkeypatch, "", export=out)
    assert s.send("x").kind == F.TIMEOUT and s.model_calls_so_far() == 1
    assert [e["kind"] for e in s.transcript()[0]] == ["assistant", "error"]
    _killed(monkeypatch, "")  # the export fails too: the error says both, and the count is unknown
    assert s.send("x").kind == F.TIMEOUT and "[no transcript]" in s.transcript()[0][-1]["text"]
    assert s.model_calls_so_far() is None
    assert s.transcript()[0][-1]["text"].startswith("[timeout]")


class Counting(Fixed):
    """A session that counts its model calls during the turn, like the CLI arms."""

    def __init__(self, out, calls_in_turn):
        super().__init__(out)
        self.calls_in_turn = calls_in_turn

    def send(self, text):
        self.stats.model_calls += self.calls_in_turn
        return super().send(text)

    def model_calls_so_far(self):
        return self.stats.model_calls


class Unknown(Counting):
    """A killed turn whose output couldn't be read: the count is unknown after it."""

    def model_calls_so_far(self):
        return None if self.sent else 0


@pytest.mark.parametrize("arm", ["codex:x", "olas-ocb-agent:x", "hermes:x"])
def test_every_timeout_is_scored_and_a_known_idle_turn_is_flagged_for_the_audit(arm, monkeypatch):
    """open-run-1: codex/flash b-sell-position (0 calls, 0 tokens) and an empty Hermes turn. The timeout is scored
    for every arm, whatever it reports; the bench only flags a turn it knows made no model call."""
    late = F.timed_out("the run passed the 900 s wall clock")
    idle = _trial(monkeypatch, arm, Counting(late, 0))
    assert not idle["infra"] and idle["failure"]["kind"] == F.TIMEOUT and idle["no_model_call_in_turn"] is True
    busy = _trial(monkeypatch, arm, Counting(late, 3))
    assert busy["failure"]["kind"] == F.TIMEOUT and "no_model_call_in_turn" not in busy
    for s in (Fixed(late), Unknown(late, 0)):  # can't say, or can't say any more: an unknown count is never 0
        r = _trial(monkeypatch, arm, s)
        assert r["failure"]["kind"] == F.TIMEOUT and "no_model_call_in_turn" not in r
    ok = _trial(monkeypatch, arm, Counting("done", 0))  # no timeout: nothing to flag
    assert "no_model_call_in_turn" not in ok
    assert A.Session().model_calls_so_far() is None


def test_a_turn_the_runner_ended_is_never_flagged(monkeypatch):
    from onchainbench import watchdog as W

    monkeypatch.setattr(R.W, "run_turn", lambda s, text, seconds: W.Overrun(1020, abandoned=False))
    r = _trial(monkeypatch, "codex:x", Counting("unused", 0))
    assert r["failure"]["kind"] == F.TIMEOUT and "no_model_call_in_turn" not in r and r["turns_overrun"] == 1


def test_a_killed_cli_turn_with_nothing_collected_leaves_its_count_unknown(tmp_path, monkeypatch):
    t = _fake_t(tmp_path)
    s = A.ClaudeCodeSession(t, None)
    _killed(monkeypatch, None)  # subprocess collected no output at all
    assert s.send("x").kind == F.TIMEOUT and s.model_calls_so_far() is None


# ---- a harness that never started: infra ------------------------------------------------------------------------
START_ERROR = 'No safe OpenClaw temp root is available: "/tmp/x/tmp" is group/world writable without sticky protection'


class Starts(Counting):
    """A CLI session whose turn ends in a harness error, having done what `acts` does first (with the trial's T):
    by default nothing at all, as when OpenClaw refused its temp dir at start-up (smoke-011)."""

    def __init__(self, calls_in_turn=0, cost=0.0, events=(), acts=(), **extra):
        super().__init__(F.cli_error("openclaw", START_ERROR), calls_in_turn)
        self.stats.cost_usd, self.events, self.acts, self.t = cost, list(events), acts, None
        self.stats.extra.update(extra)

    def send(self, text):
        for act in self.acts:
            act(self.t)
        return super().send(text)

    def transcript(self):
        return [{"turn": 0, "kind": "user", "text": "x"}, *self.events], ["raw"]


def _started(monkeypatch, arm, session, task=PASSES):
    def open_session(arm_, t, *a, **k):
        session.t = t
        return session

    monkeypatch.setattr(A, "open_session", open_session)
    return R.trial(arm, task, 0, FakeEnvs())


CLI_ARMS = ["claude-code:x", "codex:x", "hermes:x", "openclaw:x", "olas-ocb-agent:x"]
SEND = {"kind": "tx", "sender": "safe", "status": "success", "to": "0x" + "77" * 20, "value": 0, "data": "0x"}
SEND |= {"tx_hash": "0xa1", "diff": {"tokens": {}, "recipients": []}}


@pytest.mark.parametrize("arm", CLI_ARMS)
def test_a_harness_error_before_any_model_call_is_infra(arm, monkeypatch):
    """smoke-011: OpenClaw refused the bench's temp dir at start-up, 0 model calls and $0, and the 6 trials were
    scored as the entry's failures. Nothing the agent or the model did could have caused it: infra, by its own kind."""
    r = _started(monkeypatch, arm, Starts())
    assert r["infra"] and not r["passed"] and r["phase"] == "agent" and r["failure"]["kind"] == F.HARNESS_START
    assert r["failure"]["detail"] == START_ERROR and r["reason"].startswith("harness start: No safe OpenClaw")
    assert r["cost_usd"] == 0.0 and "cost_reserved_usd" not in r  # the session closed and said what it cost
    # what the harness's setup put on the ledger before the session opened isn't the agent's
    task = Task("x-task", "A", "f", "gnosis", "do it", PASSES.check, setup=lambda t: t.ledger.append(dict(SEND)))
    assert _started(monkeypatch, arm, Starts(), task=task)["failure"]["kind"] == F.HARNESS_START


def _connect_call(t):
    t.fc.s.tool_calls = 1


@pytest.mark.parametrize("arm", CLI_ARMS)
@pytest.mark.parametrize(
    "session",
    [
        lambda: Starts(calls_in_turn=1),  # a model call
        lambda: Starts(cost=0.0001),  # a cost
        lambda: Starts(events=[{"turn": 0, "kind": "assistant", "text": "on it"}]),  # model output
        lambda: Starts(events=[{"turn": 0, "kind": "tool_call", "text": "ls", "tool": "bash"}]),  # a tool call
        lambda: Starts(tool_calls=2),  # tool calls the harness reported
        lambda: Starts(acts=[_connect_call]),  # a Connect tool call
        lambda: Starts(acts=[lambda t: t.ledger.append(dict(SEND))]),  # a send
    ],
    ids=["model-call", "cost", "model-output", "tool-call", "reported-tool-calls", "connect-call", "send"],
)
def test_a_harness_error_after_the_agent_acted_stays_the_entry_s(arm, session, monkeypatch):
    r = _started(monkeypatch, arm, session())
    assert not r["infra"] and not r["passed"] and r["failure"]["kind"] == F.HARNESS_ERROR and "phase" not in r


def test_a_harness_error_with_tokens_or_an_unknown_count_stays_the_entry_s(monkeypatch):
    tokens = Starts()
    tokens.stats.prompt_tokens = 12
    assert _started(monkeypatch, "openclaw:x", tokens)["failure"]["kind"] == F.HARNESS_ERROR
    backend = Starts()
    backend.backend_spent = [0.01]  # olas-ocb-agent's mech spend
    assert _started(monkeypatch, "olas-ocb-agent:x", backend)["failure"]["kind"] == F.HARNESS_ERROR
    unknown = Starts()
    unknown.model_calls_so_far = lambda: None  # the arm can't say: an unknown count is never taken for 0
    assert _started(monkeypatch, "codex:x", unknown)["failure"]["kind"] == F.HARNESS_ERROR


@pytest.mark.parametrize("arm", sorted(A.SCRIPTED_ARMS))
def test_the_scripted_arms_failures_keep_their_kind(arm, monkeypatch):
    """The scripted arms make no model calls by design: their failures are never made infra by this rule."""
    r = _started(monkeypatch, arm, Starts())
    assert not r["infra"] and r["failure"]["kind"] == F.HARNESS_ERROR


def test_the_infra_rate_counts_a_harness_that_never_starts(monkeypatch):
    """A broken harness can't hide behind re-runs: every trial it never starts is infra, and the arm is flagged."""
    lost = _started(monkeypatch, "openclaw:x", Starts())
    rs = [_rec("a", f"t{i}", True) for i in range(8)]
    rs += [{**lost, "arm": "a", "task": f"s{i}", "attempt": 0} for i in range(2)]
    s = stats.arm_summary(rs)
    assert s["infra"] == 2 and s["infra_rate"] == 0.2 and s["infra_high"] is True
    assert s["failure_kinds"] == {"harness_start": 2} and s["infra_by_phase"] == {"agent": 2}
    assert publish.entry("a", rs, {}, True)["infra_high"] is True
