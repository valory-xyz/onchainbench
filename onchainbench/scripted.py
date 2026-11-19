"""The scripted arms (oracle, null, cheater, obeyer) on the benchmark's own Connect session.

They validate tasks and checkers, so they must not run on any contestant's code. A scripted session plays a list of
steps per turn, as olas-ocb-agent's loop did: each step is a shell command (or a function of the previous
command's output that returns one), and the last step is the turn's final message.
"""

from __future__ import annotations

from pathlib import Path

from onchainbench.connect.policy import SessionCaps
from onchainbench.connect.session import ConnectSession, open_session


class Script:
    """The current turn's steps; the runner loads each turn's (`load`)."""

    def __init__(self, steps: list | None = None):
        self.steps = list(steps or [])

    def load(self, steps: list) -> None:
        self.steps = list(steps)


class ScriptedSession:
    """A turn with no script left returns `finals` in order (the null arm: a refusal, then nothing further), or
    "done." for an oracle. Only arms that follow the oracle expose `oracle` (the runner loads each turn's steps
    into it); the null arm must not, or it would play the oracle's later turns."""

    def __init__(
        self,
        ws: Path,
        script: Script,
        step_limit: int,
        caps: SessionCaps,
        approver=None,
        unattended: bool = True,
        on_broker=None,
        finals: tuple[str, ...] | None = None,
    ):
        from onchainbench.arms import Stats

        self.ws, self.script, self.step_limit, self.caps = ws, script, step_limit, caps
        self.approver, self.unattended, self.on_broker = approver, unattended, on_broker
        self.finals = None if finals is None else list(finals)
        self.stats = Stats()
        self.cs = self._open()

    def _open(self) -> ConnectSession:
        return open_session(
            self.ws,
            self.caps,
            approval="ask" if self.approver else "auto",
            approver=self.approver,
            on_broker=self.on_broker,
        )

    def _final(self, text: str) -> str:
        return text if self.unattended else f"DONE: {text}"  # a chat turn shows the marker, as the agent loop did

    def send(self, text: str) -> str:
        out, ran = "", 0
        while self.script.steps:
            s = self.script.steps.pop(0)
            s = s(out) if callable(s) else s
            if not self.script.steps:
                return self._final(s)
            if ran >= self.step_limit:
                return f"[harness] step limit {self.step_limit} reached for this turn."
            out = self.cs.shell.run(s)["output"].rstrip("\n")
            ran += 1
        if self.finals is None:  # an oracle with nothing scripted this turn
            return self._final("done.")
        return self._final(self.finals.pop(0)) if self.finals else "I have nothing further to do."

    def crash_and_resume(self) -> None:
        """Drop the shell and broker; restart with the session's spend kept, as a resumed agent would."""
        p = self.cs.policy
        kept = (p.spent_native, p.spent_mech, p.value_calls, set(p.approved))
        self.cs.close()
        self.cs = self._open()
        q = self.cs.policy
        q.spent_native, q.spent_mech, q.value_calls, q.approved = kept

    def close(self):
        self.stats.extra = {"policy_refusals": [e for e in self.cs.policy.log if not e["allowed"]]}
        self.cs.close()
        return self.stats
