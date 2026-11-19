"""olas-ocb-agent as a CLI arm: `ocb-agent -p --output json`, one process per turn, like every contestant.

  olas-ocb-agent:M        the agent with LiteLLM model M (OpenRouter, provider pinned with ?provider=)
  olas-ocb-agent-mech:M   the agent buying its completions from the fork's chat mech, which M backs in the bench

The bench gives it the workspace, the task text, Connect's config (the workspace's .mcp.json, which the agent's own
broker fronts) and the limits on its command line, and reads what it did from the JSON object it prints:
- the transcript from the turn's `events` (a new session's first turn starts with the system prompt);
- the broker's refusals (`broker_refusals`) and the tool listings it delivered (`tools_listed`), into the ledger as
  the bench's own broker logs them (arms.log_refusal, arms.log_listing);
- counts, tokens and cost, which are session totals restored across `--resume`: each turn adds the difference;
- `error`: "provider", "harness" or "timeout" are the bench's failure kinds; "model" (a budget or cap inside the model
  backend) ends the turn normally. A non-zero exit without JSON is a CLI error (failures.cli_error);
- the broker's token, from the shim `.mcp.json` it writes in the agent's directory, so a leak of it counts.

What the model sees is what it saw when the agent ran inside the bench's process: the trusted Connect skill
(`--trusted-skill-hash`), the agent's directory `agent/` (`--agent-subdir`), the same limits, and the logging proxy for
its shell only (`--shell-env`). Multi-turn trials resume the session (`--resume`), with `--attended` where a simulated
user answers (block E), so each turn is a fresh process with the session's history, cwd and counts.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
from pathlib import Path

from onchainbench import arms as A
from onchainbench import failures as F
from onchainbench import transcript
from onchainbench.arms import Session, Stats
from onchainbench.connect.workspace import AGENT_SUBDIR
from onchainbench.env import SKILL
from onchainbench.tasks.base import T

MODEL_BUDGET_USD = 1.0  # marketplace model calls per trial (block D needs more than the agent's $0.05 default)
SESSION_KEYS = ("model_calls", "prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd")  # session totals
# the CLI's `error` as the bench's failure kinds: "terminated" is a stop from outside (SIGTERM), infra; "model" (a
# budget or cap inside the model backend) ends the turn normally; any other value is the harness's own failure
ERROR_KINDS = {
    "provider": F.PROVIDER_ERROR,
    "harness": F.HARNESS_ERROR,
    "timeout": F.TIMEOUT,
    "terminated": F.BENCH_ERROR,
}


def skill_hash() -> str:
    """The Connect skill the bench writes: trusted as instructions, as it was when the agent ran in-process."""
    return hashlib.sha256(SKILL.encode()).hexdigest()


def caps_flags(caps=A.EVAL_CAPS) -> list[str]:
    """The bench's caps (none) as the agent's command line."""

    def num(v):
        return "inf" if v is None else str(v)

    out = ["--max-native-usd", num(caps.max_native_usd), "--max-mech-usd", num(caps.max_mech_usd)]
    out += ["--max-value-calls", "none" if caps.max_value_calls is None else str(caps.max_value_calls)]
    return out + (["--allow-sign-message"] if caps.allow_sign_message else [])


def chat_text(out: dict) -> str:
    """What a text-only chat mech returns: the reply text, with any (unrequested) native tool call rendered as the
    fenced bash block a text agent expects. Some models emit tool calls with no tools declared; dropping them would
    leave an empty reply."""
    text = out.get("content") or ""
    cmds = []
    for c in out.get("tool_calls") or []:
        try:
            cmds.append(json.loads((c.get("function") or {}).get("arguments") or "{}").get("command"))
        except (ValueError, AttributeError):
            continue
    cmds = [c for c in cmds if c]
    return text + "".join(f"\n```bash\n{c}\n```" for c in cmds[:1])


def mech_kwargs(name: str, provider: str | None) -> dict:
    """The chat mech's completion arguments, as the agent's own LiteLLM model sends them: a stop at the shell
    output's `<returncode>`, three retries, the provider pin, and OpenRouter's charge in the usage (OpenRouter only)."""
    extra: dict = {}
    if provider:
        extra["provider"] = {"only": [provider], "allow_fallbacks": False}
    if name.startswith("openrouter/"):
        extra["usage"] = {"include": True}
    return {"stop": ["<returncode>"], "num_retries": 3, **({"extra_body": extra} if extra else {})}


def mech_backend(model: str, spent: list, errors: list):
    """The chat mech's model in the bench's process (fork Connect serves the mech): M with its provider pin, its
    cost what OpenRouter billed or tokens x the registered price, refused past the per-trial cap."""
    from onchainbench.llm import quiet_litellm
    from onchainbench.mock_connect.server import ToolError
    from onchainbench.pricing import billed, parse_spec

    litellm = quiet_litellm()
    name, opts = parse_spec(model)
    kwargs = mech_kwargs(name, opts.get("provider"))

    def complete(msgs):
        try:
            return litellm.completion(model=name, messages=msgs, **kwargs)
        except litellm.exceptions.BadRequestError:  # some providers reject stop sequences: once more without
            return litellm.completion(model=name, messages=msgs, **{k: v for k, v in kwargs.items() if k != "stop"})

    def llm(msgs) -> str:
        if spent[0] >= A.TRIAL_CAP_USD:
            raise ToolError(f"spending limit of ${A.TRIAL_CAP_USD:.2f} for this session reached")
        msgs = msgs if isinstance(msgs, list) else [{"role": "user", "content": msgs}]
        try:
            r = complete(msgs)
        except Exception as e:  # the provider failed: the turn's error names it
            errors.append(f"{type(e).__name__}: {e}")
            raise
        cost = billed(getattr(r, "usage", None))
        if cost is None:
            try:
                cost = float(litellm.completion_cost(completion_response=r) or 0.0)
            except Exception:  # noqa: BLE001 - an unpriced response counts nothing rather than fail the mech
                cost = 0.0
        spent[0] += cost
        m = r.choices[0].message
        calls = [c.model_dump() if hasattr(c, "model_dump") else c for c in (getattr(m, "tool_calls", None) or [])]
        return chat_text({"content": m.content, "tool_calls": calls})

    return llm


class OcbAgentSession(A.CountsCallsLive, A.RecordsTranscript, Session):
    fmt = "chat"  # the agent's session events: the user message is in them

    def __init__(
        self,
        t: T,
        kind: str,
        model: str,
        step_limit: int,
        unattended: bool = True,
        sessions_dir: Path | None = None,
    ):
        self.t, self.kind, self.model, self.step_limit, self.unattended = t, kind, model, step_limit, unattended
        self.mech = kind == "olas-ocb-agent-mech"
        self.sid = None
        self.stats, self.totals = Stats(), A.RunningTotals()
        self.stats.extra["policy_refusals"] = []
        self._init_transcript()
        self.dest = sessions_dir  # where the run keeps session files (isolation's own dir under isolation)
        self.dir = t.ws.parent / "ocb-agent"  # the trial's own: session files and state, which its agent user owns
        self.sessions = self.dir / "sessions"
        self.sessions.mkdir(parents=True, exist_ok=True)
        (self.dir / "state").mkdir(exist_ok=True)
        self.seen: dict[str, int] = {}  # session file -> lines already recorded (a killed turn's are read from it)
        self.bin = A.cli_bin(kind)
        self.backend_spent, self.backend_errors = [0.0], []
        extra = {"XDG_STATE_HOME": str(self.dir / "state")}  # its state stays in the trial
        if self.mech:
            from onchainbench.fork_connect import LLM_MECH

            self.key, self.spec = None, f"mech:chat@{LLM_MECH}"
            t.fc.s.llm_backend = mech_backend(model, self.backend_spent, self.backend_errors)
            t.fc.s.prepaid = t.fc.s.mech_amount(5000)  # a prepaid balance: model calls don't move the Safe's funds
        else:
            self.key, self.spec = A.TrialKey(t), f"litellm:{model}"
            self.stats.extra["cap_mode"] = self.key.mode
            extra |= self.key.env()  # the model key in the CLI's environment; the agent's shell gets an allowlist
        self.env = A.env_with_key(extra, t, proxy=False)  # the proxy goes to the agent's shell alone

    def _cmd(self, text: str) -> list[str]:
        cmd = [self.bin, "--dir", str(self.t.ws), "--model", self.spec, "--approval", "auto", "--sandbox", "auto"]
        cmd += ["--step-limit", str(self.step_limit), "--cost-limit", str(A.TRIAL_CAP_USD), *caps_flags()]
        cmd += ["--model-budget", str(MODEL_BUDGET_USD), "--turn-seconds", str(A.HARNESS_TURN_SECONDS)]
        cmd += ["--sessions-dir", str(self.sessions), "--agent-subdir", AGENT_SUBDIR]
        cmd += ["--trusted-skill-hash", skill_hash()]
        for k, v in A.proxy_env(self.t).items():
            cmd += ["--shell-env", f"{k}={v}"]
        if not self.mech:
            from onchainbench.pricing import price_entry

            if (entry := price_entry(self.model)) is not None:
                cmd += ["--model-price", json.dumps(entry)]
        cmd += ["--output", "json"]
        if not self.unattended:  # a simulated user answers (block E)
            cmd += ["--attended"]
        if self.sid:
            cmd += ["--resume", self.sid]
        return [*cmd, "-p", text]

    def send(self, text: str) -> str:
        if A.capped(self.stats):
            return A.CAP_REACHED
        self.backend_errors.clear()  # in place: the mech backend holds this list
        try:
            p = subprocess.run(
                self._cmd(text),
                cwd=self.t.ws,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=A.TURN_SECONDS,  # its own limit (--turn-seconds) ends the turn first, with its JSON
                env=self.env,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            self._killed(text, self._partial(e))
            return F.timed_out(A.PAST_WALL_CLOCK)
        finally:
            self._broker_token()
        d = A.last_json(p.stdout)
        if "session_id" not in d:
            detail = (p.stderr or p.stdout or f"exit {p.returncode}, no output")[-F.DETAIL_MAX :]
            self._record_error(text, f"[error] {detail[-1500:]}")
            return F.cli_error(self.kind, detail)
        return self._turn(d, p.stderr or "")

    def _turn(self, d: dict, stderr: str) -> str:
        self.sid = d["session_id"] or self.sid
        events = d.get("events") or []
        self._record(None, json.dumps(events))
        if path := d.get("session_file"):
            self._mark_seen(str(path))
        if "model_calls" in d:  # a start failure (a bad resume, a sandbox) reports no counts: nothing to diff
            n = self.totals.turn(self.sid, {k: d.get(k) or 0 for k in SESSION_KEYS})
            self.stats.model_calls += int(n["model_calls"])
            A.add_tokens(self.stats, int(n["prompt_tokens"]), int(n["completion_tokens"]), int(n["cached_tokens"]), 0)
            self.stats.cost_usd += float(n["cost_usd"])
        self.stats.extra["cost_sources"] = dict(d.get("cost_sources") or {})  # billed or estimate, per call
        self.stats.extra["compactions"] = int(d.get("compactions") or 0)
        self.stats.extra["policy_refusals"] += list(d.get("policy_refusals") or [])
        if path:
            self.stats.extra["session"] = str(path)
        for r in d.get("broker_refusals") or []:
            A.log_refusal(self.t, r.get("tool", ""), r.get("args") or {}, r.get("rule", ""))
        self.t.data.setdefault("agent_tools_listed", 0)
        for _ in range(int(d.get("tools_listed") or 0)):
            A.log_listing(self.t)
        reply, error = str(d.get("reply") or ""), d.get("error")
        if error is None:
            return reply
        self.events.append(transcript.event(self.turn - 1, "error", reply or f"[{error}]"))
        detail = str(d.get("error_detail") or "\n".join(x for x in (reply, stderr[-2000:]) if x))[-F.DETAIL_MAX :]
        if self.backend_errors:  # the mech arm's backend failed at the provider
            why = self.backend_errors[-1]
            return F.Reply(f"[error] {why}"[:1500], F.PROVIDER_ERROR, why)
        if error == "model":  # a budget or cap inside the model backend ended the turn, scored as it stands
            return reply
        kind = ERROR_KINDS.get(error, F.HARNESS_ERROR)
        return F.Reply(reply or f"[error] {error}", kind, detail or str(error))

    def _mark_seen(self, path: str) -> None:
        """The session file's lines so far are recorded: a later killed turn reads only what follows."""
        from onchainbench import isolation

        with contextlib.suppress(OSError):
            self.seen[path] = len(isolation.read_text(path).splitlines())

    def _killed(self, text: str, stdout: str) -> None:
        """A turn the bench killed at the wall clock has no JSON: what the session file gained since the last turn
        (the agent appends each event as it logs it) is the turn's transcript and its model calls."""
        from onchainbench import isolation

        new, read = [], False
        for p in sorted(self.sessions.glob("*.jsonl")):
            try:
                lines = isolation.read_text(p).splitlines()
            except OSError:
                continue
            read = True
            new += lines[self.seen.get(str(p), 0) :]
            self.seen[str(p)] = len(lines)
        events = []
        for line in new:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("type") in ("message", "compaction"):
                events.append(ev)
        self.calls_unknown = not read  # the session file is the count; without one the turn's calls are unknown
        self.stats.model_calls += sum(
            1 for e in events if e["type"] == "message" and (e.get("message") or {}).get("role") == "assistant"
        )
        why = f"[timeout] {A.PAST_WALL_CLOCK}"
        self._record_killed(None if events else text, stdout, why, parsed=json.dumps(events) if events else None)

    def _broker_token(self) -> None:
        """The agent's broker token, from the shim it writes for `pearl` (read as the agent: it is 0600), so a leak
        of it counts (redact.trial_secrets); and the broker the arm is behind, for the tool exposure rule."""
        from onchainbench import isolation

        try:
            shim = json.loads(isolation.read_text(self.t.ws / AGENT_SUBDIR / ".mcp.json"))
            auth = shim["mcpServers"]["pearl-connect"]["headers"]["Authorization"]
        except (OSError, ValueError, KeyError, TypeError):
            return
        token = str(auth).split()[-1]
        secrets = self.t.data.setdefault("secrets", [])
        if token and token not in secrets:
            secrets.append(token)

    def crash_and_resume(self) -> None:
        """Nothing runs between turns: the next turn is a new process resuming the saved session."""
        self.stats.extra["resumes"] = self.stats.extra.get("resumes", 0) + 1

    def close(self) -> Stats:
        """The session files go where the run keeps them (copied as the agent reads them), the key is deleted."""
        from onchainbench import isolation

        if self.key is not None:
            self.key.close()
        src = self.stats.extra.get("session")
        if self.dest is not None and src:
            try:
                text = isolation.read_text(src)
            except OSError as e:
                self.stats.extra["session_error"] = f"{type(e).__name__}: {e}"[:300]
            else:
                self.dest.mkdir(parents=True, exist_ok=True)
                out = self.dest / Path(src).name
                out.write_text(text)
                self.stats.extra["session"] = str(out)
        return self.stats
