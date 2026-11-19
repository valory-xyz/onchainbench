"""A harness CLI as an OnchainBench arm: the session class to copy into onchainbench/arms.py (CONTRIBUTING.md,
"Harnesses"), with `my-harness` (beside this file) as the stand-in CLI it runs.

The runner prepares each trial and opens one session on it; it calls `send` for each user message (one for most
tasks, several for blocks D and E) and `close` at the end. Each turn starts the CLI once, in its own process: under
the bench's isolation it runs as the trial's agent user. What the trial gives the arm:
  t.ws    the workspace root: Connect's MCP config (`.mcp.json`), the brief (`AGENTS.md`, `CLAUDE.md`), the
          Connect skill and the agent's directory `agent/`, as a Pearl user's machine has them
  model   the model part of the arm spec (`my-harness:<model>`)
To add it: put the class in arms.py, name the kind in BUILTIN_KINDS, CLI_BINS (and MULTI_TURN if it resumes), and
dispatch it in `_open_session`; add its pinned install to scripts/install-arms.sh.
"""

from __future__ import annotations

import json
import subprocess

from onchainbench import arms as A
from onchainbench import failures as F
from onchainbench.arms import Session, Stats

ERROR_KINDS = {"provider": F.PROVIDER_ERROR, "harness": F.HARNESS_ERROR, "timeout": F.TIMEOUT}
TOTALS = ("model_calls", "prompt_tokens", "completion_tokens", "cost_usd")  # session totals, kept across --resume


class MyHarnessSession(A.CountsCallsLive, A.RecordsTranscript, Session):
    fmt = "chat"  # how transcript.PARSERS reads what the CLI printed: its events, in the OpenAI chat shape

    def __init__(self, t, model: str, step_limit: int, binp: str):
        self.t, self.model, self.step_limit, self.bin, self.sid = t, model, step_limit, binp, None
        self.stats, self.totals = Stats(), A.RunningTotals()
        self._init_transcript()
        self.key = A.TrialKey(t)  # the CLI's one model key: per trial, with a credit limit of the cap if it can
        # the runner's secrets removed; the model key and the trial's TMPDIR and logging proxy added
        self.env = A.env_with_key(self.key.env(), t)

    def send(self, text: str) -> str:
        if A.capped(self.stats):
            return A.CAP_REACHED
        cmd = [self.bin, "--dir", str(self.t.ws), "--model", self.model, "--step-limit", str(self.step_limit)]
        cmd += ["--cost-limit", str(A.TRIAL_CAP_USD), "--output", "json"]
        cmd += [*(["--resume", self.sid] if self.sid else []), "-p", text]
        try:
            p = subprocess.run(
                cmd,
                cwd=self.t.ws,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=A.TURN_SECONDS,  # the bench's one wall clock
                env=self.env,
                check=False,
            )
        except subprocess.TimeoutExpired as e:  # a scored failure
            self._record_killed(text, self._partial(e), f"[timeout] {A.PAST_WALL_CLOCK}")
            return F.timed_out(A.PAST_WALL_CLOCK)
        d = A.last_json(p.stdout)
        if "session_id" not in d:  # no reply: its error output says whose failure it was
            self._record_error(text, f"[error] {p.stderr[-1500:]}")
            return F.cli_error("my-harness", p.stderr[-F.DETAIL_MAX :] or f"exit {p.returncode}")
        self.sid = d["session_id"]
        self._record(None, json.dumps(d.get("events") or []))
        n = self.totals.turn(self.sid, {k: d.get(k) or 0 for k in TOTALS})  # this turn's share of the totals
        self.stats.model_calls += int(n["model_calls"])
        self.stats.prompt_tokens += int(n["prompt_tokens"])
        self.stats.completion_tokens += int(n["completion_tokens"])
        self.stats.cost_usd += float(n["cost_usd"])
        if (error := d.get("error")) in ERROR_KINDS:
            return F.Reply(d.get("reply") or f"[error] {error}", ERROR_KINDS[error], p.stderr[-F.DETAIL_MAX :] or error)
        return str(d.get("reply") or "")

    def close(self) -> Stats:
        self.key.close()
        return self.stats
