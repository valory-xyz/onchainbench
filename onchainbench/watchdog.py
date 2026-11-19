"""The runner's hard wall clock on a turn, the same for every arm.

A CLI arm's turn is a process the arm kills at its own limit, but the session around it (and a scripted arm's turn)
runs in the runner's own thread, so a hung turn would stall its whole shard. Every turn therefore runs in a thread of
its own with a deadline: A.TURN_SECONDS plus TURN_GRACE_S (an arm's own limit ends it first). At the deadline the
runner kills the processes the turn started (under isolation, every process of the trial's agent user; otherwise the
process trees started from the turn's thread), waits KILL_GRACE_S for the call to return, and abandons it (a daemon
thread) if it doesn't. The turn is a scored timeout either way, and a hung session's close is bounded the same way.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass

TURN_GRACE_S = 120.0  # past A.TURN_SECONDS before the runner ends a turn itself (OpenClaw's own limit is the same)
KILL_GRACE_S = 30.0  # after the kill, for the call to return before it is abandoned


@dataclass
class Overrun:
    """A turn the runner ended: whether its call came back after the kill, or was abandoned."""

    seconds: float
    abandoned: bool

    @property
    def detail(self) -> str:
        out = f"the turn passed the runner's {self.seconds:.0f} s wall clock; its processes were killed"
        return out + ("; the call never returned and was abandoned" if self.abandoned else "")


def _start(fn, procs: list) -> tuple[threading.Thread, dict]:
    """fn() in a daemon thread; Popen objects created in that thread are collected in procs."""
    from onchainbench import isolation

    isolation.install()  # so the thread's processes are tracked (outside an agent's context Popen is unchanged)
    box: dict = {}

    def target():
        with isolation.track(procs):
            try:
                box["value"] = fn()
            except BaseException as e:  # noqa: BLE001 - handed to the caller's thread
                box["error"] = e

    th = threading.Thread(target=target, daemon=True, name="onchainbench-turn")
    th.start()
    return th, box


def _result(box: dict):
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_turn(session, text: str, seconds: float):
    """session.send(text) with a hard deadline: its reply, or an Overrun once the runner had to end the turn. An
    ended turn's thread can start no process from then on (isolation.TurnClosed), so a call the runner abandons
    can't act after the trial moved on, whatever the arm's own loop does."""
    from onchainbench import isolation

    procs = isolation.TurnProcs()
    th, box = _start(lambda: session.send(text), procs)
    th.join(seconds)
    if not th.is_alive():
        return _result(box)
    procs.closed = True  # before the kill: nothing new starts while the old is killed
    interrupt(session, procs)
    th.join(KILL_GRACE_S)
    return Overrun(seconds, abandoned=th.is_alive())


def interrupt(session, procs: list) -> None:
    """Kill what a turn started: the agent user's processes under isolation (the session knows its user), and the
    process trees started from the turn's thread."""
    from onchainbench.connect.shell import _kill_tree

    kill = getattr(session, "kill_agent_processes", None)
    if callable(kill):
        with contextlib.suppress(Exception):  # best effort: the trees below go too
            kill()
    for p in procs:
        if p.poll() is None:
            _kill_tree(p.pid)


def close_bounded(session, fallback):
    """session.close(), waited for up to KILL_GRACE_S after a turn was abandoned: a hung arm must not keep its trial
    (and its agent user) forever. Returns (stats, closed)."""
    th, box = _start(session.close, [])
    th.join(KILL_GRACE_S)
    if not th.is_alive():
        return _result(box), True
    abandon = getattr(session, "abandon", None)
    if callable(abandon):
        with contextlib.suppress(Exception):
            abandon()  # under isolation: the user's processes, files and firewall go back as at close
    return fallback, False
