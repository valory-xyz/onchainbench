"""The runner's hard wall clock on every turn (watchdog.py): a session whose turn hangs is ended at
A.TURN_SECONDS plus a grace, its processes killed, the turn a scored timeout, and the trial finishes."""

import os
import subprocess
import sys
import threading
import time
import types

import pytest

from onchainbench import arms as A
from onchainbench import failures as F
from onchainbench import isolation as I
from onchainbench import watchdog as W
from tests.test_failure_kinds import PASSES, FakeEnvs, _trial

posix = pytest.mark.skipif(os.name != "posix", reason="process groups and SIGKILL")


@pytest.fixture(autouse=True)
def _short_clock(monkeypatch):
    monkeypatch.setattr(A, "TURN_SECONDS", 0.3)
    monkeypatch.setattr(W, "TURN_GRACE_S", 0.0)
    monkeypatch.setattr(W, "KILL_GRACE_S", 0.5)
    yield
    subprocess.Popen = I._ORIG_POPEN


class Hangs(A.Session):
    """A session whose turn starts a child process and then never returns on its own."""

    def __init__(self, stuck=False, close_hangs=False):
        self.stats, self.children, self.stuck, self.close_hangs = A.Stats(cost_usd=0.1), [], stuck, close_hangs
        self.release = threading.Event()

    def send(self, text):
        self.children.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]))
        if self.stuck:  # ignores everything: only abandoning the call frees the runner
            self.release.wait(30)
            return "late"
        self.children[-1].wait()  # returns once its child is killed
        return "after the kill"

    def close(self):
        if self.close_hangs:
            self.release.wait(30)
        return self.stats


class Quick(A.Session):
    def __init__(self):
        self.stats = A.Stats(cost_usd=0.1)

    def send(self, text):
        return "DONE"


@posix
def test_a_turn_past_the_wall_clock_is_killed_and_scored_as_a_timeout(monkeypatch):
    s = Hangs()
    t0 = time.monotonic()
    r = _trial(monkeypatch, "stand-in:x", s, PASSES, FakeEnvs())
    assert time.monotonic() - t0 < 10  # the trial finished, not after the child's 60 s
    assert not r["infra"] and not r["passed"] and r["failure"]["kind"] == F.TIMEOUT
    assert "processes were killed" in r["reason"] and r["turns_overrun"] == 1 and r["turns_abandoned"] == 0
    assert s.children[0].poll() is not None  # its child is gone
    assert r["session_closed"] is True


@posix
def test_a_turn_that_never_returns_is_abandoned_and_its_close_bounded(monkeypatch):
    s = Hangs(stuck=True, close_hangs=True)
    t0 = time.monotonic()
    r = _trial(monkeypatch, "stand-in:x", s, PASSES, FakeEnvs())
    assert time.monotonic() - t0 < 10
    assert r["failure"]["kind"] == F.TIMEOUT and "never returned and was abandoned" in r["reason"]
    assert r["turns_abandoned"] == 1 and r["session_closed"] is False and r["cost_usd"] == pytest.approx(0.1)
    assert r["cost_after_abandon"] == "unknown"  # what the abandoned call spent later is counted nowhere
    assert s.children[0].poll() is not None
    s.release.set()  # let the abandoned threads end


def test_a_turn_in_time_is_untouched(monkeypatch):
    monkeypatch.setattr(A, "TURN_SECONDS", 30)
    r = _trial(monkeypatch, "stand-in:x", Quick(), PASSES, FakeEnvs())
    assert r["passed"] and r["answer"] == "DONE" and "turns_overrun" not in r


def test_an_exception_in_the_turn_reaches_the_runner_as_before():
    class Boom(Quick):
        def send(self, text):
            raise KeyError("x")

    with pytest.raises(KeyError):
        W.run_turn(Boom(), "go", 5)


def test_the_kill_goes_through_the_agent_user_under_isolation():
    killed = []
    session = types.SimpleNamespace(kill_agent_processes=lambda: killed.append(1))
    done = types.SimpleNamespace(poll=lambda: 0)  # already exited: left alone
    W.interrupt(session, [done])
    assert killed == [1]

    def broken():
        raise RuntimeError("helper gone")

    W.interrupt(types.SimpleNamespace(kill_agent_processes=broken), [])  # best effort: never raises
    abandoned = []
    stuck = types.SimpleNamespace(close=lambda: time.sleep(5), abandon=lambda: abandoned.append(1))
    assert W.close_bounded(stuck, "fallback") == ("fallback", False) and abandoned == [1]
    assert W.close_bounded(types.SimpleNamespace(close=lambda: "stats"), None) == ("stats", True)


def test_an_isolated_session_kills_its_user_and_gives_it_back_once(monkeypatch):
    calls = []
    monkeypatch.setattr(I, "run_helper", lambda spec, timeout=60: calls.append(spec["op"]))
    slot = I.Slot(0, "ocb-agent-0", 4242, 4243, (4243,))
    import contextlib

    agent = types.SimpleNamespace(slot=slot, finish=lambda: calls.append("finish"), active=contextlib.nullcontext)
    released = []
    s = I.IsolatedSession(types.SimpleNamespace(close=lambda: "stats"), agent, lambda: released.append(1))
    s.kill_agent_processes()
    s.abandon()
    assert s.close() == "stats"  # a late close after abandon(): the user went back once only
    assert calls == ["killall", "finish"] and released == [1]


def test_processes_started_in_a_turns_thread_are_tracked():
    I.install()
    procs = []
    with I.track(procs):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    q = subprocess.Popen([sys.executable, "-c", "pass"])  # outside a turn: not tracked
    q.wait()
    assert procs == [p]
