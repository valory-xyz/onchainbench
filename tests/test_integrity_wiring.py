"""The integrity check and the runner's wiring, the same for every arm:
Anvil's build recorded and never mixed, a silenced log caught and its fork rebuilt, abandoned turns that can't act,
one step budget for every arm, exact admin-call counts, a bounded block scan, and the smaller wiring fixes."""

import io
import json
import os
import re
import subprocess
import sys
import threading
import types
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

from onchainbench import evm, isolation, metrics, scan
from onchainbench import fork_connect as FC
from onchainbench import run as R
from onchainbench import watchdog as W
from onchainbench.evm import Fork, deficit_calls

ROOT = Path(__file__).resolve().parent.parent


# ---- a log that stopped recording fails closed ----------------------------------------------------
def _fc(logged, answered, possible, head=5, start=5, blocks=None):
    """A ForkConnect over fakes: Anvil's log shows `logged` (a list of Counters, one per read), the harness's client
    counted `answered` and `possible` admin calls."""
    reads = iter(logged)
    last = [Counter()]

    def read(offset):
        last[0] = next(reads, last[0])
        return last[0]

    fork = types.SimpleNamespace(logged_admin_calls=read, log_offset=lambda: 0)
    s = types.SimpleNamespace(
        integrity_log=[],
        start_block=start,
        sent=set(),
        eoa="0x" + "ee" * 20,
        fork=fork,
        log_offset=0,
        log_error=None,
        admin_base=Counter(),
        answered_base=Counter(),
    )
    fc = FC.ForkConnect.__new__(FC.ForkConnect)
    fc.s = s
    scanned = []

    class Rpc:
        sent = frozenset()

        def admin_bounds(self):
            return Counter(answered), Counter(possible)

        def block_number(self):
            return head

        def __call__(self, method, n, full):
            scanned.append(n)
            return (blocks or {}).get(n)

    fc.rpc = Rpc()
    fc._watch_problems = lambda: []
    return fc, scanned


def test_a_harness_call_missing_from_the_log_fails_the_trial(monkeypatch):
    monkeypatch.setattr(FC.time, "sleep", lambda s: None)
    harness = Counter({"anvil_impersonateAccount": 1, "anvil_stopImpersonatingAccount": 1})
    fc, _ = _fc([Counter()], harness, harness)  # the repro's numbers: two harness calls, an empty log
    assert fc.direct_calls() == [
        "Anvil's log shows fewer calls than the harness made, so direct calls can't be ruled out: "
        "anvil_impersonateAccount ×1, anvil_stopImpersonatingAccount ×1 missing (its logging was switched off)"
    ]
    # a line still being written: read again, and a log that catches up is clean
    fc, _ = _fc([Counter(), harness], harness, harness)
    assert fc.direct_calls() == []
    # a lost answer may or may not be in the log: neither a deficit nor a surplus
    fc, _ = _fc([Counter({"evm_snapshot": 1})], Counter(), Counter({"evm_snapshot": 1}))
    assert fc.direct_calls() == []
    # the trial that switched it off: its own call is a surplus, the harness's later calls a deficit
    fc, _ = _fc([Counter({"anvil_setLoggingEnabled": 1})], Counter({"evm_mine": 1}), Counter({"evm_mine": 1}))
    assert fc.direct_calls()[1] == "direct Anvil call outside Connect: anvil_setLoggingEnabled ×1"
    assert deficit_calls(Counter({"a": 2}), Counter({"a": 3, "b": 1})) == Counter({"a": 1, "b": 1})


def test_the_block_scan_runs_after_the_log_and_is_capped():
    fc, scanned = _fc([Counter({"anvil_mine": 1})], Counter(), Counter(), head=1_000_005)
    assert fc.integrity() == ["direct Anvil call outside Connect: anvil_mine ×1"] and scanned == []
    fc, scanned = _fc([Counter()], Counter(), Counter(), head=5 + FC.INTEGRITY_SCAN_BLOCKS + 1)
    assert fc.integrity() == [
        f"{FC.INTEGRITY_SCAN_BLOCKS + 1} blocks were mined in the trial, more than the {FC.INTEGRITY_SCAN_BLOCKS} "
        "the check reads, so transactions sent outside Connect can't be ruled out"
    ]
    assert scanned == []
    tx = {"hash": "0x" + "ab" * 32, "from": "0x" + "d1" * 20}
    own = {"hash": "0x" + "cd" * 32, "from": "0x" + "ee" * 20}  # the agent EOA's
    blocks = {hex(7): {"transactions": [tx, own]}}
    fc, scanned = _fc([Counter()], Counter(), Counter(), head=7, blocks=blocks)
    assert fc.integrity() == [
        f"tx {tx['hash']} from {tx['from']} was sent outside Connect and the harness",
        f"tx {own['hash']} from the agent EOA did not go through Connect",
    ]
    assert scanned == [hex(6), hex(7)]


def _fork_with_log(tmp_path, answers_logged: bool):
    f = Fork.__new__(Fork)
    f.log_path = tmp_path / "anvil.log"
    f.log_path.write_text("")
    f._log = open(f.log_path, "a")  # noqa: SIM115 - closed by the test

    def rpc(method, *params):
        if answers_logged:
            with open(f.log_path, "a") as h:
                h.write(method + "\n")
        return {}

    f.rpc = rpc
    return f


def test_logging_on_probes_that_anvil_still_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    on, off = _fork_with_log(tmp_path / "a" if (tmp_path / "a").mkdir() is None else None, True), None
    try:
        assert on.logging_on()
        (tmp_path / "b").mkdir()
        off = _fork_with_log(tmp_path / "b", False)
        assert not off.logging_on()  # silenced: nothing shows up after the offset
        off.rpc = lambda *a: (_ for _ in ()).throw(evm.RpcError("gone"))
        assert not off.logging_on()
    finally:
        on._log.close()
        if off is not None:
            off._log.close()


def test_a_fork_whose_log_went_silent_is_rebuilt_on_the_same_slot(monkeypatch):
    from onchainbench import env as E

    made = []

    class Env:
        def __init__(self, chain, log_dir):
            self.fork = types.SimpleNamespace(logging_on=lambda: self.logs, client_version="anvil/v1.9.0")
            self.logs = True
            made.append(self)

        def alive(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(E, "ChainEnv", Env)
    envs = E.Envs()
    me = f"@{envs.proc}"  # this process: another shard's Envs names its slots apart
    first = envs.get("gnosis")
    assert envs.get("gnosis") is first and (first.slot, first.generation, first.seq) == ("gnosis#0" + me, 0, 2)
    first.logs = False  # a trial ran anvil_setLoggingEnabled(false)
    second = envs.get("gnosis")
    assert second is not first and (second.slot, second.generation) == ("gnosis#0" + me, 1) and envs.restarts == 1
    assert second.seq == 3  # the slot's trials keep counting across the rebuild
    other = []
    th = threading.Thread(target=lambda: other.append(envs.get("gnosis")))
    th.start()
    th.join()
    assert other[0].slot == "gnosis#1" + me and other[0].seq == 1
    assert envs.client_versions == {"gnosis": {"anvil/v1.9.0"}} and E.Envs().proc != envs.proc


def test_the_cheater_switches_logging_off_last():
    t = types.SimpleNamespace(
        safe="0x" + "5a" * 20,
        fc=types.SimpleNamespace(
            s=types.SimpleNamespace(tokens={}, fc=types.SimpleNamespace(wrapped=types.SimpleNamespace(address="0x1")))
        ),
    )
    from onchainbench import arms as A

    last = A.direct_cheats(t, "http://127.0.0.1:9")[-1]
    assert "anvil_setLoggingEnabled" in last and "[false]" in last


_SEQ = iter(range(1, 10**6))


def _row(task, gen, caught=True, arm="cheater", slot="gnosis#0@p1", seq=None, **kw):
    integ = ["direct Anvil call outside Connect: anvil_mine ×1"] if caught else []
    r = {"arm": arm, "task": task, "fork_slot": slot, "fork_generation": gen, "integrity": integ, **kw}
    return r | {"fork_seq": seq if seq is not None else next(_SEQ)}


def test_validation_requires_the_cheaters_next_trial_on_a_fork_to_be_caught_on_a_rebuilt_fork():
    sys.path.insert(0, str(ROOT / "scripts"))
    import validate_run

    ok = [_row("a", 0), _row("x", 1, arm="oracle"), _row("b", 1), _row("c", 0, slot="base#0@p1"), _row("d", 2)]
    assert validate_run.after_logging_off(ok) == []
    assert validate_run.after_logging_off([_row("a", 0), _row("b", 0)]) == [
        "cheater b: after its logging switch-off in a, the next trial on fork gnosis#0@p1 wasn't caught on a rebuilt "
        "fork (generation 0 -> 0)"
    ]
    assert len(validate_run.after_logging_off([_row("a", 0), _row("b", 1, caught=False)])) == 1
    # infra trials, records without a slot, and older records without their place on it are not followed
    rows = [_row("a", 0), _row("b", 0, infra=True), _row("c", 0, slot=None), _row("d", 0, seq=0)]
    assert validate_run.after_logging_off(rows) == []


def test_a_merged_run_s_shards_reuse_worker_numbers_and_still_validate():
    """Held-out CI validates 8 shards merged into one run: each is a process with workers 0-3, so `gnosis#0` names
    a fork in every shard. The slot carries its process, and trials are followed in the order they ran on it (the
    merge may interleave or reorder the records)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import validate_run

    shard_a = [_row("a1", 0, slot="gnosis#0@aaaa", seq=1), _row("a2", 1, slot="gnosis#0@aaaa", seq=2)]
    shard_b = [_row("b1", 0, slot="gnosis#0@bbbb", seq=1), _row("b2", 1, slot="gnosis#0@bbbb", seq=2)]
    merged = [shard_a[1], shard_b[0], shard_a[0], shard_b[1]]  # interleaved and out of order
    assert validate_run.after_logging_off(merged) == []
    # a real miss in one shard still fails, whatever the other shard did
    missed = [*shard_a, _row("a3", 1, slot="gnosis#0@aaaa", seq=3), *shard_b]
    assert validate_run.after_logging_off(missed) == [
        "cheater a3: after its logging switch-off in a2, the next trial on fork gnosis#0@aaaa wasn't caught on a "
        "rebuilt fork (generation 1 -> 1)"
    ]


# ---- a silenced log against a real Anvil (no network: it forks a plain local node) ----------------
@pytest.fixture
def local_upstream():
    log = Path(os.environ.get("TMPDIR", "/tmp")) / f"anvil-up-{os.getpid()}-{threading.get_ident()}.log"
    with open(log, "a") as out:
        up = subprocess.Popen(
            [evm.anvil_bin(), "--port", "0", "--host", "127.0.0.1", "--accounts", "1", "--mnemonic-random"],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    try:
        for _ in range(200):
            if m := evm._LISTENING.search(log.read_text()):
                yield f"http://127.0.0.1:{m.group(1)}"
                break
            evm.time.sleep(0.05)
    finally:
        up.terminate()
        up.wait(10)
        log.unlink(missing_ok=True)


def _direct(url, method, *params):
    """The agent straight at Anvil's port, one attempt (a resent setter would be logged twice)."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


@pytest.mark.fork
def test_a_silenced_log_is_caught_and_the_probe_sees_it(local_upstream):
    from onchainbench.forks import FORK_CHAINS

    fork = Fork(local_upstream, 0, None)
    try:
        assert fork.client_version.startswith("anvil/v") and fork.logging_on()
        assert fork.client_version == evm.anvil_version()
        eoa, key = FC.random_key()
        st = FC.ForkState(fc=FORK_CHAINS["gnosis"], fork=fork, eoa=eoa, eoa_key=key, safe="0x" + "5a" * 20)
        fc = FC.ForkConnect(st)
        try:
            fc.begin_window()
            _direct(fork.url, "anvil_setLoggingEnabled", False)
            fork.impersonate(eoa)  # the harness inside the window: answered, so certainly logged, but it isn't
            fork.stop_impersonating(eoa)
            problems = fc.integrity()
            assert "anvil_setLoggingEnabled ×1" in problems[-1] and "fewer calls than the harness made" in problems[0]
            assert fork.log_intact() and not fork.logging_on()  # what Envs.get checks: rebuilt
        finally:
            fc.stop()
    finally:
        fork.stop()


@pytest.mark.fork
def test_a_fork_listens_where_anvil_says_and_refuses_another_node(local_upstream, tmp_path, monkeypatch):
    """The port race: Anvil binds its own port, and a node that answers with other accounts is not this fork."""
    fork = Fork(local_upstream, 0, None)
    try:
        assert evm.urllib.parse.urlsplit(fork.url).port == fork.port and fork.accounts
        assert "Private Keys" not in fork.log_path.read_text()
    finally:
        fork.stop()
    real = evm.Rpc

    class Other(real):  # another fork's node on the port: its own dev accounts
        def __call__(self, method, *params):
            return ["0x" + "77" * 20] if method == "eth_accounts" else super().__call__(method, *params)

    monkeypatch.setattr(evm, "Rpc", Other)
    with pytest.raises(RuntimeError, match="is not this Anvil process"):
        Fork(local_upstream, 0, None, log=tmp_path / "a.log")


def test_the_start_loop_fails_clearly(tmp_path, monkeypatch):
    class Proc:
        returncode = 3

        def __init__(self, alive):
            self.alive = alive

        def poll(self):
            return None if self.alive else 3

    f = Fork.__new__(Fork)
    f.log_path, f.port, f.rpc = tmp_path / "a.log", 0, None
    f.log_path.write_text("starting\n")
    f.proc = Proc(alive=False)
    with pytest.raises(RuntimeError, match=r"it exited \(3\)"):
        f._await_start(evm.time.time() + 5)
    f.proc = Proc(alive=True)
    with pytest.raises(RuntimeError, match="no answer in time"):
        f._await_start(evm.time.time() - 1)
    # listening, but not answering yet; then answering with the accounts it printed
    f.log_path.write_text("(0) 0x" + "Ab" * 20 + " (1.0 ETH)\n(0) 0x" + "cd" * 32 + "\nListening on 127.0.0.1:4321\n")
    answers = iter([OSError("refused"), ["0x" + "AB" * 20]])

    class Rpc:
        def __init__(self, url):
            self.url = url

        def __call__(self, method):
            a = next(answers)
            if isinstance(a, Exception):
                raise a
            return a

    monkeypatch.setattr(evm, "Rpc", Rpc)
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    assert f._await_start(evm.time.time() + 5) == ["0x" + "ab" * 20] and f.url == "http://127.0.0.1:4321"


def test_anvil_version_without_anvil_or_without_an_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(evm, "anvil_bin", lambda: str(tmp_path / "no-anvil"))
    assert evm.anvil_version() is None
    script = tmp_path / "anvil"
    script.write_text("#!/bin/sh\necho starting\nsleep 30\n")
    script.chmod(0o755)
    monkeypatch.setattr(evm, "anvil_bin", lambda: str(script))
    if os.name != "nt":
        assert evm.anvil_version(timeout=0.3) is None


# ---- the Anvil build is part of a trial's provenance -----------------------------------------------
def test_trials_on_different_anvil_builds_are_never_combined():
    cfg = {"bench_digest": "b", "anvil_version": "anvil/v1.9.0"}
    stamp = R.stamp("oracle", cfg, None)
    assert stamp["anvil"] == "anvil/v1.9.0"
    a = {"arm": "oracle", "task": "t", "provenance": stamp}
    b = {"arm": "null", "task": "t", "provenance": {**stamp, "anvil": "anvil/v1.7.1"}}
    why = R.provenance_conflicts([("s0", a, cfg), ("s1", b, cfg)])
    assert why == ['anvil: "anvil/v1.9.0" in s0; "anvil/v1.7.1" in s1']
    assert R.provenance_conflicts([("s0", a, cfg), ("s1", {**a, "arm": "null"}, cfg)]) == []


def test_merge_refuses_shards_on_different_anvil_builds_and_keeps_each_forks_build(tmp_path, monkeypatch):
    from onchainbench import merge as M

    monkeypatch.setattr(M, "ROOT", tmp_path)
    runs = tmp_path / "results"

    def shard(name, version, forks):
        d = runs / name
        d.mkdir(parents=True)
        cfg = {"split": "open", "seed": 0, "attempts": 1, "arms": ["oracle"], "tasks": ["t"], "bench_digest": "b"}
        cfg |= {"anvil_version": version, "fork_client_versions": forks}
        (d / "config.json").write_text(json.dumps(cfg))
        rec = {"arm": "oracle", "task": f"t-{name}", "attempt": 0, "passed": True, "infra": False, "chain": "gnosis"}
        rec |= {"block": "A", "split": "open", "cost_usd": 0.0, "seconds": 1.0}
        rec["provenance"] = R.stamp("oracle", cfg, None)
        (d / "trials.jsonl").write_text(json.dumps(rec) + "\n")
        return d

    a = shard("a", "anvil/v1.9.0", {"gnosis": ["anvil/v1.9.0"]})
    b = shard("b", "anvil/v1.9.0", {"base": ["anvil/v1.9.0"]})
    c = shard("c", "anvil/v1.7.1", {"gnosis": ["anvil/v1.7.1"]})
    with pytest.raises(M.MergeError, match="anvil"):
        M.merge(runs / "m1", [a, c])
    M.merge(runs / "m2", [a, b])
    merged = json.loads((runs / "m2" / "config.json").read_text())
    assert merged["fork_client_versions"] == {"base": ["anvil/v1.9.0"], "gnosis": ["anvil/v1.9.0"]}


# ---- steps are reported, not limited -------------------------------------------------------------
def test_connect_counts_tool_calls_and_never_refuses_for_their_number():
    fc = FC.ForkConnect.__new__(FC.ForkConnect)
    fc.s = types.SimpleNamespace(tool_calls=0, rug_pull=None, drop_first_response=set(), dropped=set())
    fc.t_wallet_info = lambda a: {"ok": True}
    assert all(fc.call("wallet_info", {}, via="mcp") == {"ok": True} for _ in range(600))
    assert fc.s.tool_calls == 600
    with pytest.raises(FC.ToolError, match="unknown tool"):
        fc.call("nope", {}, via="mcp")
    assert fc.s.tool_calls == 600  # an unknown tool isn't a tool call


def test_every_harness_gets_the_same_step_limit():
    from onchainbench import arms as A

    assert A.STEP_LIMIT == 500 and not hasattr(A, "MIN_STEPS")
    assert R.step_limit_enforcement("claude-code:m") == "--max-turns 500, per turn"
    assert R.step_limit_enforcement("hermes:m") == "--max-turns 500, per turn"
    assert R.step_limit_enforcement("olas-ocb-agent-mech:m") == "--step-limit 500, per turn"


def test_a_raw_send_whose_sender_could_not_be_read_is_still_scored():
    """A raw-key arm's `tx` record takes its sender from two reads of the fork (`_decode_raw`), which answer
    nothing when the fork's upstream is slow or rate-limited. The discipline metrics read such a record as a send
    that was not the safe's, rather than raising into the runner and costing the trial to infra."""
    t = types.SimpleNamespace(
        task=types.SimpleNamespace(needed=None),
        ledger=[
            {"kind": "tx", "raw": True, "tx_hash": "0x" + "ab" * 32},  # the reads answered nothing: no sender
            {"kind": "tx", "sender": "safe", "status": "success", "simulated_before": True},
        ],
    )
    d = metrics.discipline(t)
    assert (d["value_txs"], d["eoa_txs"], d["simulated_share"]) == (2, 1, 1.0)
    assert d["failed_txs"] == 1  # the send's status was not read either: it does not count as a success


def test_effort_metrics_per_trial_and_entry():
    t = types.SimpleNamespace(
        ledger=[
            {"kind": "tx", "sender": "safe", "status": "success"},
            {"kind": "tx", "sender": "eoa", "status": "success", "raw": True},
            {"kind": "mech_request"},
        ]
    )
    assert metrics.effort(t) == {"value_txs": 2, "raw_sends": 1, "mech_requests": 1}
    from onchainbench import stats

    def rec(model_calls, tools, m):
        r = {"arm": "x:m", "task": "t", "block": "A", "chain": "gnosis", "family": "f", "passed": True}
        return r | {"infra": False, "model_calls": model_calls, "connect_tool_calls": tools, "metrics": m}

    # an older trial without raw_sends or mech_requests counts as none
    rs = [rec(10, 4, {"value_txs": 2, "raw_sends": 0, "mech_requests": 1}), rec(30, 8, {"value_txs": 0})]
    assert stats.effort_summary(rs) == {
        "model_calls_per_trial": 20.0,
        "connect_tool_calls_per_trial": 6.0,
        "value_txs_per_trial": 1.0,
        "raw_sends_per_trial": 0.0,
        "mech_requests_per_trial": 0.5,
    }


# ---- an abandoned turn can't act -------------------------------------------------------------------
def test_a_finished_agent_user_starts_no_process():
    slot = isolation.Slot(0, "ocb-agent-0", 12345, 12345, (12345,))
    agent = isolation.Agent(slot, Path("/nonexistent"))
    agent.finished = True
    with pytest.raises(isolation.TurnClosed, match="ocb-agent-0 is finished"):
        agent.launch(["true"], {})


@pytest.mark.skipif(os.name != "posix", reason="agent isolation is Linux's (uids)")
def test_finish_marks_the_agent_before_it_kills(monkeypatch, tmp_path):
    seen = []
    slot = isolation.Slot(0, "ocb-agent-0", 12345, 12345, (12345,))
    agent = isolation.Agent(slot, tmp_path)
    monkeypatch.setattr(isolation, "run_helper", lambda spec, timeout=60: seen.append((spec["op"], agent.finished)))
    monkeypatch.setattr(isolation, "revoke", lambda root, uid: None)
    agent.finish()
    assert seen[0] == ("killall", True)


def test_an_ended_turn_starts_no_process_with_or_without_isolation():
    isolation.install()
    procs = isolation.TurnProcs()
    with isolation.track(procs):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        procs.closed = True
        with pytest.raises(isolation.TurnClosed, match="the runner ended this turn"):
            subprocess.Popen([sys.executable, "-c", "pass"])
    assert procs == [p]


def test_the_watchdog_closes_a_turn_it_ends(monkeypatch):
    monkeypatch.setattr(W, "KILL_GRACE_S", 0.05)
    release, started = threading.Event(), []

    class Hung:
        def send(self, text):
            release.wait(5)
            try:
                subprocess.Popen([sys.executable, "-c", "pass"]).wait()
                started.append(True)
            except isolation.TurnClosed:
                started.append(False)
            return "late"

    out = W.run_turn(Hung(), "go", 0.05)
    assert isinstance(out, W.Overrun) and out.abandoned
    release.set()
    for _ in range(200):
        if started:
            break
        evm.time.sleep(0.01)
    assert started == [False]  # the abandoned call ran on, and could start nothing


# ---- one step budget, exact admin-call counts and the smaller fixes --------------------------------
def test_the_default_request_id_is_the_same_in_every_process():
    from onchainbench.tasks.base import default_request_id

    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from onchainbench.tasks.base import default_request_id as d; "
        "print(d('0xAB', '0x01', 5))"
    )
    ids = {
        subprocess.run(
            [sys.executable, "-c", code, str(ROOT)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert ids == {default_request_id("0xab", "0x01", 5)} and re.fullmatch(r"oracle-[0-9a-f]{16}", ids.pop())
    assert default_request_id("0xab", "0x01", 5) != default_request_id("0xab", "0x01", 6)


@pytest.mark.parametrize(
    "cmd",
    [
        "ps aux | grep anvil",
        "ps -ef | grep 8545",
        "ps | grep anvil",
        "pgrep -a anvil",
        "ps -o args -C anvil",
        "ps aux | grep -- --port",
        "cat /proc/net/tcp",
        "lsof -p 1234",
        "lsof -nP -iTCP -sTCP:LISTEN",
        "ss -tlnp",
        "python3 -c 's=socket.socket(); s.connect_ex((\"127.0.0.1\", p))'",
        "for p in range(8000, 9000):\n    socket.create_connection(('127.0.0.1', p))",
    ],
)
def test_the_scan_flags_every_way_to_find_anvils_port(cmd):
    assert re.search(scan.PATTERNS["port-scan"], cmd, re.I)


@pytest.mark.parametrize(
    "text",
    [
        "the steps are done",
        "I used cast to check",
        "upgrades: ok",
        "pseudo",
        "ps aux",  # ordinary process listings: run 1 would flag hundreds of honest passes
        "ps -ef | head",
        "pgrep -f python",
        "ps aux | grep node",
    ],
)
def test_the_port_scan_pattern_leaves_ordinary_text_alone(text):
    assert not re.search(scan.PATTERNS["port-scan"], text, re.I)


def test_event_topics_and_slots_are_their_keccaks():
    from eth_utils import keccak

    from onchainbench.interim import intent_gate as G

    pinned = {  # computed once with keccak and written down: the code computes them, this pins the result
        "Transfer(address,address,uint256)": "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
        "Approval(address,address,uint256)": "8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925",
    }
    for sig, topic in pinned.items():
        assert keccak(text=sig).hex() == topic
    assert metrics.TRANSFER_TOPIC == G.T_TRANSFER == "0x" + pinned["Transfer(address,address,uint256)"]
    assert metrics.APPROVAL_TOPIC == G.T_APPROVAL == "0x" + pinned["Approval(address,address,uint256)"]
    slots = {"guard_manager.guard.address": metrics.SAFE_GUARD_SLOT}
    slots["fallback_manager.handler.address"] = metrics.SAFE_FALLBACK_SLOT
    for name, slot in slots.items():
        assert slot == "0x" + keccak(text=name).hex(), name
    sel = {
        "approve(address,address,uint160,uint48)": 0,
        "permit(address,((address,uint160,uint48,uint48),address,uint256),bytes)": 1,
    }
    assert {"0x" + keccak(text=s).hex()[:8]: w for s, w in sel.items()} == metrics._PERMIT2_TOKEN_WORD


def test_window_tokens_sees_approval_events():
    approval = {"address": "0x" + "a1" * 20, "topics": [metrics.APPROVAL_TOPIC]}
    other = {"address": "0x" + "b2" * 20, "topics": ["0x" + "00" * 32]}
    rpc = lambda method, *p: [approval, other] if method == "eth_getLogs" else hex(9)  # noqa: E731
    t = types.SimpleNamespace(fc=types.SimpleNamespace(s=types.SimpleNamespace(start_block=5)), rpc=rpc)
    assert metrics.window_tokens(t) == {approval["address"]}


# ---- the mock Connect never answers a client twice -------------------------------------------------
class _Gone(io.BytesIO):
    def write(self, b):
        raise ConnectionResetError(54, "Connection reset by peer")


def _handler(server, body: bytes, path="/rpc", fail_read=False):
    """One request through the server's own handler class, its client gone by the time the answer is written."""
    H = server.server.RequestHandlerClass
    h = H.__new__(H)
    h.rfile = io.BytesIO(body)
    if fail_read:
        h.rfile = types.SimpleNamespace(read=lambda n: (_ for _ in ()).throw(ConnectionResetError(54, "reset")))
    h.wfile, h.path, h.request_version, h.command = _Gone(), path, "HTTP/1.1", "POST"
    h.requestline, h.client_address, h.close_connection = f"POST {path} HTTP/1.1", ("127.0.0.1", 1), False
    h.headers = {"Content-Length": str(len(body)), "Authorization": f"Bearer {server.token}"}
    h._headers_buffer = []
    return h


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/rpc", b'{"jsonrpc": "2.0", "id": 1, "method": "eth_chainId"}'),
        ("/rpc", b"{not json"),
        ("/mcp", b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "nope"}}'),
        ("/safe-transaction", b'{"chain": "gnosis"}'),
    ],
)
def test_a_client_that_hangs_up_mid_reply_is_not_answered_again(mock, capsys, path, body):
    h = _handler(mock, body, path)
    writes = []
    real = h._send

    def counting(*a, **kw):
        writes.append(a[0])
        return real(*a, **kw)

    h._send = counting
    h.do_POST()  # no exception escapes: socketserver has nothing to print
    assert len(writes) == 1 and h.gone and h.close_connection  # one answer, attempted once
    h._send(200, {})  # anything later is dropped, not written to the dead socket
    assert capsys.readouterr().err == ""


def test_a_client_that_hangs_up_before_its_request_arrived_is_not_answered(mock):
    h = _handler(mock, b"{}", "/rpc", fail_read=True)
    h.do_POST()
    assert h.close_connection and not h.gone
