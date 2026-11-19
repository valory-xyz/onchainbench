"""A first run without an OpenRouter key (the refusal names the fix, --no-judges
for a wiring run, the install command for an arm's harness), and Ctrl-C mid-run (in-flight trials marked
interrupted, a summary of what finished, the --fill command to resume, and --fill re-running them)."""

import json
import threading
import types

import pytest

from onchainbench import arms as A
from onchainbench import judge as J
from onchainbench import merge as M
from onchainbench import publish as P
from onchainbench import run as R
from onchainbench import stats as S
from onchainbench.failures import INTERRUPTED

TASK = "a-wrap-unwrap"


def rec(arm="oracle", attempt=0, passed=True, **kw):
    return {
        "arm": arm,
        "task": TASK,
        "attempt": attempt,
        "block": "A",
        "chain": "gnosis",
        "family": TASK,
        "split": "open",
        "passed": passed,
        "infra": False,
        "seconds": 1.0,
        "cost_usd": 0.0,
        "steps_done": 1,
        "steps_total": 1,
        "metrics": {},
        "reason": "ok",
        **kw,
    }


class FakeEnvs:
    def __init__(self, out=None):
        self.restarts, self.client_versions, self.stopped = 0, {}, threading.Event()

    def stop(self):
        self.stopped.set()


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """main() without forks: results under tmp_path, the trial faked by each test."""
    envs = FakeEnvs()
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "Envs", lambda out: envs)
    monkeypatch.setattr(R, "anvil_version", lambda: "anvil/test")
    monkeypatch.setattr(R, "provenance", lambda split: {"date": "2026-11-16", "bench_digest": "d"})
    monkeypatch.setattr(R, "key_usage", lambda settle=False: None)
    return types.SimpleNamespace(envs=envs, results=tmp_path / "results")


def test_ctrl_c_marks_trials_in_flight_interrupted_writes_the_summary_and_says_how_to_resume(
    runner, monkeypatch, capsys
):
    calls, lock, busy = [], threading.Lock(), threading.Semaphore(0)

    def trial(arm, task, k, envs, seed, sessions, tag):
        with lock:
            n = len(calls)
            calls.append(k)
        if n == 0:
            return rec(attempt=k)  # finished before the stop: scored
        busy.release()  # in flight: its forks stop under it, and it ends as the bench's infra failure
        assert envs.stopped.wait(10)
        return rec(attempt=k, passed=False, infra=True, reason="URLError: Connection refused")

    real_wait, first = R.cf.wait, [True]

    def ctrl_c(futures, return_when):  # Ctrl-C in the main thread, once both workers are in a trial
        if not first[0]:
            return real_wait(futures, return_when=return_when)
        first[0] = False
        assert busy.acquire(timeout=10) and busy.acquire(timeout=10)
        raise KeyboardInterrupt

    monkeypatch.setattr(R.cf, "wait", ctrl_c)
    monkeypatch.setattr(R, "trial", trial)
    args = ["--arms", "oracle", "--tasks", TASK, "--attempts", "4", "--workers", "2", "--run", "stopped"]
    assert R.main(args) == 130
    out = capsys.readouterr().out
    assert "interrupted: no new trials start" in out and "STOP" in out
    assert f"python -m onchainbench.run --arms oracle --tasks {TASK} --attempts 4 --workers 2 --fill stopped" in out
    assert len(calls) == 3  # the fourth trial never started
    d = runner.results / "stopped"
    rows = [json.loads(x) for x in (d / "trials.jsonl").read_text().splitlines()]
    assert sorted((r["passed"], bool(r.get("interrupted"))) for r in rows) == [
        (False, True),
        (False, True),
        (True, False),
    ]
    stopped = next(r for r in rows if r.get("interrupted"))
    assert not stopped["infra"] and stopped["failure"] == {
        "kind": INTERRUPTED,
        "detail": "URLError: Connection refused",
    }
    summary = json.loads((d / "summary.json").read_text())
    s = summary["arms"]["oracle"]
    assert summary["interrupted"] == 2 and s["interrupted"] == 2 and s["infra"] == 0 and s["infra_rate"] == 0.0
    assert s["trials"] == 3 and s["pass_mean"] == 1.0  # the interrupted trials aren't scored
    assert "judges" not in summary  # a normal run's summary is unchanged

    # --fill re-runs the interrupted trials and the missing one, and keeps the scored one
    monkeypatch.setattr(R, "trial", lambda arm, task, k, *a: rec(attempt=k))
    assert R.main(["--arms", "oracle", "--tasks", TASK, "--attempts", "4", "--fill", "stopped", "--run", "f"]) == 0
    assert "keeping 1 trials, running 3" in capsys.readouterr().out
    rows = [json.loads(x) for x in (runner.results / "f" / "trials.jsonl").read_text().splitlines()]
    assert sorted(r["attempt"] for r in rows) == [0, 1, 2, 3] and not any(r.get("interrupted") for r in rows)


def test_a_bench_error_still_raises_and_nothing_queued_starts(runner, monkeypatch):
    calls = []

    def trial(arm, task, k, *a):
        calls.append(k)
        raise RuntimeError("the bench's own bug")

    monkeypatch.setattr(R, "trial", trial)
    with pytest.raises(RuntimeError, match="own bug"):
        R.main(["--arms", "oracle", "--tasks", TASK, "--attempts", "3", "--workers", "1", "--run", "boom"])
    assert len(calls) == 1 and runner.envs.stopped.is_set() and J.OFF is None


def test_a_bench_error_stops_the_queue_even_before_the_main_thread_wakes(runner, monkeypatch):
    # a slow main thread (a busy Windows runner): the worker goes straight on to the queued trials before the main
    # thread is back from waiting to cancel them, and they must not start
    import time

    calls = []

    def trial(arm, task, k, *a):
        calls.append(k)
        raise RuntimeError("the bench's own bug")

    real_wait = R.cf.wait

    def slow_wait(*a, **kw):
        time.sleep(0.3)
        return real_wait(*a, **kw)

    monkeypatch.setattr(R, "trial", trial)
    monkeypatch.setattr(R.cf, "wait", slow_wait)
    with pytest.raises(RuntimeError, match="own bug"):
        R.main(["--arms", "oracle", "--tasks", TASK, "--attempts", "3", "--workers", "1", "--run", "slow"])
    assert calls == [0]


def test_resume_command_keeps_the_flags_but_run_and_fill():
    argv = ["--arms", "a:m?x=1|nogate", "--run", "r1", "--fill=old", "--run=r2", "--fill", "older", "--seed", "2"]
    assert R.resume_command(argv, "r1") == ("python -m onchainbench.run --arms 'a:m?x=1|nogate' --seed 2 --fill r1")


def test_interrupted_trials_are_neither_scored_nor_infra_anywhere(tmp_path, capsys):
    done, stopped = rec(), R.interrupted(rec(attempt=1, reason="URLError"))
    assert S.is_scored(done) and not S.is_scored(stopped) and not S.is_infra(stopped)
    # merge keeps any result over an interrupted trial of the same key, and an interrupted one only when alone
    assert M._rank(stopped) < M._rank(rec(infra=True)) < M._rank(done)
    # the validation run names it, rather than calling the oracle's trial a failure
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_run", R.ROOT / "scripts" / "validate_run.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    assert v.problems([done, stopped]) == [f"oracle {TASK}: interrupted (finish the run with --fill)"]
    from onchainbench.scan import scan_run

    assert len(scan_run([done, stopped])["trials"]) == 1
    from onchainbench.inspect_log import write_inspect_log

    (path,) = write_inspect_log(tmp_path, [done, stopped], types.SimpleNamespace(split="open", attempts=2))
    assert json.loads(path.read_text())["results"]["completed_samples"] == 1


def test_publish_refuses_an_interrupted_run_and_a_no_judges_run(tmp_path):
    arm = "claude-code:openrouter/x/y"

    def run_dir(name, rows, **cfg):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"split": "open", "run": name, **cfg}))
        (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return d

    stopped = run_dir("stopped", [rec(arm), R.interrupted(rec(arm, attempt=1))])
    with pytest.raises(P.PublishError, match=r"1 trials were interrupted.*--fill stopped"):
        P.build(stopped, "https://x")
    with pytest.raises(P.PublishError, match="--no-judges"):
        P.build(run_dir("wiring", [rec(arm)], no_judges=True), "https://x")
    with pytest.raises(P.PublishError, match="--no-judges"):  # a merged run carries the mark on its trials
        P.build(run_dir("merged", [R.judges_off(rec(arm))]), "https://x")


def test_the_preflight_refusal_names_the_fix(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    msg = R.preflight_refusal(["openrouter/m: AuthenticationError: 401"])
    assert "set OPENROUTER_API_KEY in .env (each run costs about a cent of judge calls)" in msg
    assert "--no-judges" in msg and msg.endswith("AuthenticationError: 401")
    monkeypatch.setenv("OPENROUTER_API_KEY", "set")
    msg = R.preflight_refusal(["openrouter/m: AuthenticationError: 401"])
    assert msg.startswith("refusing the run: a judge or simulated-user model failed") and "check it" in msg
    assert "set" not in msg.split("401")[0].split(": ")[-1]  # the key's value is never printed


def test_main_prints_the_preflight_fix(runner, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(R, "preflight_call", lambda m: (_ for _ in ()).throw(RuntimeError("401")))
    assert R.main(["--arms", "claude-code:openrouter/openai/gpt-5", "--tasks", TASK, "--run", "x"]) == 2
    assert "set OPENROUTER_API_KEY in .env" in capsys.readouterr().out
    assert not runner.results.exists()


def test_litellm_s_banner_is_suppressed(monkeypatch):
    import litellm

    from onchainbench import llm

    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="OK"), finish_reason="stop")], usage=None
    )
    monkeypatch.setattr(litellm, "completion", lambda **k: resp)
    monkeypatch.setattr(litellm, "completion_cost", lambda **k: 0.0)
    assert llm.complete("openrouter/anthropic/claude-haiku-5.5", [])[0] == "OK"
    assert litellm.suppress_debug_info is True


def test_no_judges_calls_no_judge_and_reports_judged_metrics_as_not_measured(runner, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(R, "preflight_call", calls.append)
    seen = {}

    def trial(arm, task, k, *a):
        assert J.OFF, "only --no-judges runs reach this fake: a judged one would call a real judge"
        with pytest.raises(J.JudgesOff):  # what a task's judge (or the hand-back judge) meets: refused, not called
            J.judge("t", "", "w", ["c"], "x")
        seen["off"] = J.OFF
        iv = {"stall": None, "corrections": 0, "duplicates_after_resume": 0, "failed_after_resume": False}
        iv |= {"handback": None, "handback_validated": None, "unplanned": 0}
        return rec(
            arm,
            attempt=k,
            passed=False,
            judgements=[{"error": R.NO_JUDGES, "kind": "handback", "cost": 0.0}],
            judge_errors=1,
            interventions=iv,
        )

    monkeypatch.setattr(R, "trial", trial)
    from onchainbench import pricing

    monkeypatch.setattr(pricing, "register_price", lambda m: True)
    arm = "claude-code:openrouter/openai/gpt-5"
    assert R.main(["--arms", arm, "--tasks", TASK, "--no-judges", "--run", "w"]) == 0
    assert calls == [] and seen["off"] == R.NO_JUDGES and J.OFF is None  # no preflight; judges back on after
    d = runner.results / "w"
    assert json.loads((d / "config.json").read_text())["no_judges"] is True
    (row,) = [json.loads(x) for x in (d / "trials.jsonl").read_text().splitlines()]
    assert row["judges_off"] and row["judgements"] == [] and "judge_errors" not in row
    summary = json.loads((d / "summary.json").read_text())
    assert summary["judges"] == R.NO_JUDGES and summary["judge_errors"] == {}
    iv = summary["arms"][arm]["autonomy"]["interventions"]
    assert iv["handbacks"] is None and iv["unplanned"] is None and iv["unplanned_per_pass"] is None
    assert "--no-judges" in iv["not_measured"] and iv["stalls"] == {}  # what needs no judge is still measured
    # a judged run won't --fill from it, nor it from a judged one
    capsys.readouterr()
    assert R.main(["--arms", arm, "--tasks", TASK, "--fill", "w", "--run", "w2"]) == 2
    assert "refusing --fill w: it ran without the judges" in capsys.readouterr().out
    assert R.main(["--arms", arm, "--tasks", TASK, "--no-judges", "--run", "w3"]) == 0
    assert R.main(["--arms", arm, "--tasks", TASK, "--fill", "w3", "--run", "w4"]) == 2
    monkeypatch.setattr(R, "trial", lambda arm, task, k, *a: rec(arm, attempt=k))  # a judged run: no judge called
    assert R.main(["--arms", "oracle", "--tasks", TASK, "--run", "j"]) == 0
    assert R.main(["--arms", "oracle", "--tasks", TASK, "--no-judges", "--fill", "j", "--run", "j2"]) == 2
    assert "it ran with the judges" in capsys.readouterr().out


def test_no_judges_refuses_tasks_that_need_the_simulated_user(runner, monkeypatch, capsys):
    from onchainbench import pricing

    monkeypatch.setattr(pricing, "register_price", lambda m: True)
    arm = "claude-code:openrouter/openai/gpt-5"
    assert R.main(["--arms", arm, "--blocks", "E", "--no-judges", "--run", "e"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("refusing --no-judges:") and "e-ambiguous-recipient" in out and "--blocks" in out
    assert not runner.results.exists()
    # scripted arms get a scripted user: nothing to refuse
    assert R.no_judges_refusal(["oracle"], [t for t in R.select("open", "", "E", "")]) is None


def test_an_arm_s_missing_harness_gets_its_install_command(runner, monkeypatch, capsys):
    from onchainbench import pricing

    monkeypatch.setattr(pricing, "register_price", lambda m: True)
    monkeypatch.setattr(A, "known_kind", lambda kind: kind != "olas-ocb-agent")
    assert R.main(["--arms", "olas-ocb-agent:openrouter/x?provider=y", "--run", "x"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("refusing unknown arms olas-ocb-agent:openrouter/x?provider=y")
    assert "install it with scripts/install-arms.sh --for 'olas-ocb-agent:openrouter/x?provider=y'" in out
    assert R.unknown_arms_refusal(["nope:m"]).count("install-arms") == 0  # a kind it doesn't know: no command
    monkeypatch.setattr(A, "known_kind", lambda kind: True)
    monkeypatch.setattr(R, "harness_preflight", lambda arms: ["codex:m: its CLI `codex` is missing"])
    assert R.main(["--arms", "oracle,codex:m", "--tasks", TASK, "--run", "x"]) == 2
    out = capsys.readouterr().out
    assert (
        "harness isn't installed: codex:m" in out
        and "codex:m: install it with scripts/install-arms.sh --for codex:m" in out
    )


def test_a_trial_taken_off_the_queue_as_ctrl_c_lands_never_starts(runner, monkeypatch, capsys):
    """A queued trial a worker already took when Ctrl-C landed (its cancel came too late) returns
    without starting, and builds no fork."""
    calls, busy = [], threading.Semaphore(0)

    def trial(arm, task, k, envs, *a):
        calls.append(k)
        busy.release()
        assert envs.stopped.wait(10)
        return rec(attempt=k, passed=False, infra=True, reason="URLError")

    real_wait, first = R.cf.wait, [True]

    def ctrl_c(futures, return_when):
        if not first[0]:
            return real_wait(futures, return_when=return_when)
        first[0] = False
        assert busy.acquire(timeout=10)
        raise KeyboardInterrupt

    monkeypatch.setattr(R.cf, "wait", ctrl_c)
    monkeypatch.setattr(R.cf.Future, "cancel", lambda self: False)  # every cancel too late: the race, every time
    monkeypatch.setattr(R, "trial", trial)
    args = ["--arms", "oracle", "--tasks", TASK, "--attempts", "3", "--workers", "1", "--run", "race"]
    assert R.main(args) == 130
    assert len(calls) == 1  # the two queued trials returned at the stop check
    rows = [json.loads(x) for x in (runner.results / "race" / "trials.jsonl").read_text().splitlines()]
    assert [r.get("interrupted") for r in rows] == [True]


def test_ctrl_c_while_trials_are_submitted_still_stops_the_forks(runner, monkeypatch, capsys):
    real_submit, n = R.cf.ThreadPoolExecutor.submit, [0]

    def submit(self, fn, *a):
        n[0] += 1
        if n[0] == 2:
            raise KeyboardInterrupt
        return real_submit(self, fn, *a)

    monkeypatch.setattr(R.cf.ThreadPoolExecutor, "submit", submit)
    monkeypatch.setattr(R, "trial", lambda arm, task, k, *a: rec(attempt=k))
    assert R.main(["--arms", "oracle", "--tasks", TASK, "--attempts", "3", "--run", "sub"]) == 130
    assert runner.envs.stopped.is_set() and "--fill sub" in capsys.readouterr().out


def test_stopped_envs_build_no_fork(monkeypatch):
    from onchainbench import env as E

    built = []

    class Chain:
        def __init__(self, chain, log_dir):
            built.append(self)
            self.fork, self.stopped = types.SimpleNamespace(client_version="v"), False

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(E, "ChainEnv", Chain)
    envs = E.Envs()
    envs.get("gnosis")
    envs.stop()
    with pytest.raises(E.EnvsStopped, match="no fork of base"):
        envs.get("base")
    assert len(built) == 1 and built[0].stopped
    # stopped while a fork was starting: that fork is stopped too, not left running
    envs = E.Envs()

    class Late(Chain):
        def __init__(self, chain, log_dir):
            super().__init__(chain, log_dir)
            envs._stopped = True  # stop() ran while this fork started

    monkeypatch.setattr(E, "ChainEnv", Late)
    with pytest.raises(E.EnvsStopped, match="while a fork of gnosis started"):
        envs.get("gnosis")
    assert built[-1].stopped and envs._all == []


def test_an_interrupted_twin_with_an_executed_loss_counts_in_the_safety_bound():
    loss = {"applicable": True, "executed": True, "attempted": 1, "prepared": 1, "blocked_by": {}, "exposed": True}
    loss |= {"executed_evidence": ["sent"], "near_misses": [], "blocked": 0, "failed": 0}
    twin = rec("claude-code:m", passed=False, attack="injection", attack_outcome=loss)
    clean = {**loss, "executed": False, "executed_evidence": []}
    s = S.attack_summary([R.interrupted(twin), R.interrupted({**twin, "attempt": 1, "attack_outcome": clean})])
    assert s["trials"] == 1 and s["executed_losses"] == 1  # the loss counts; the clean interrupted one stays out


def test_merge_refuses_judged_and_unjudged_shards(tmp_path):
    def shard(name, **cfg):
        d = tmp_path / name
        d.mkdir()
        base = {"split": "open", "seed": 0, "attempts": 1, "bench_digest": "d"}
        (d / "config.json").write_text(json.dumps({**base, **cfg}))
        (d / "trials.jsonl").write_text(json.dumps(rec(attempt=0 if name != "b" else 1)) + "\n")
        return d

    old, judged, unjudged = shard("a"), shard("b", no_judges=False), shard("c", no_judges=True)
    with pytest.raises(M.MergeError, match="no_judges"):
        M.merge(tmp_path / "out", [judged, unjudged])
    assert M._same({}, "no_judges") is M._same({"no_judges": False}, "no_judges") is False  # older configs: judged
    assert M.merge(tmp_path / "out2", [old, judged])["trials"] == 2
