"""One failure taxonomy for every arm. The arm that ran a turn says how it ended (a `Reply`); the runner applies one
rule to that kind, the same for every arm:

| kind             | what happened                                                          | the trial is   |
|------------------|------------------------------------------------------------------------|----------------|
| `provider_error` | the model API failed: HTTP 5xx or 429, a connection error or a timeout | infra          |
| `bench_error`    | the bench's own code failed: an adapter, the fork, the Connect double,  | infra          |
|                  | a judge, the simulated user                                            |                |
| `harness_error`  | the harness crashed, or exited with an error, for its own reasons      | scored, failed |
| `harness_start`  | a harness error before the agent did anything: no model call, no tool  | infra          |
|                  | call, no send (the runner's mark, never an arm's)                      |                |
| `timeout`        | a turn passed its wall clock                                           | scored, failed |
| `interrupted`    | the operator stopped the run (Ctrl-C) while the trial ran              | neither        |

**A harness that never started** is infra: the runner makes a `harness_error` `harness_start` when the trial's
session knows it made no model call (`Session.model_calls_so_far` is 0, not unknown), it recorded no tokens and no
cost, its transcript has no model output or tool call, it reported no tool calls, and the ledger has no entry from
the agent's window. Nothing the agent or the model did could have caused it. A harness error after any model call or
tool call stays the entry's (scored), and the scripted arms, which make no model calls by design, keep their kind.
`--fill` re-runs it like any infra trial; a harness that never starts shows in the arm's infra rate.

An interrupted trial is the runner's own mark, never an arm's: it is left out of the scores and of the infra rate,
and `--fill` re-runs it.

A timeout is a scored timeout for every arm, whatever the turn did. When the bench knows the turn made no model call
(`Session.model_calls_so_far` known before and after, and equal; the CLI arms count a killed turn's calls from its
partial output), the trial records `no_model_call_in_turn: true` for the failure audit to judge by hand. An unknown
count (output that couldn't be read, a turn the runner had to end) is never taken for 0: no flag.

An infra trial is re-run by `--fill` and left out of the scores, but what the agent did on chain still counts: a
twin lost to infra that shows a loss counts in the safety bound, and an infra trial that shows a loss
(stats.has_recorded_loss) is final: `--fill` keeps it and doesn't re-run it. Every failed or infra trial records the
raw error text (`failure.detail`), so the failure audit can check the classification.

**What an arm's session does.** `Session.send` returns the agent's answer: a plain `str` is a turn that ended
normally (a cap the agent ran into included). For a turn that failed it returns a `Reply`:
- the model API failed: `Reply(text, PROVIDER_ERROR, detail)`;
- the harness itself failed (it crashed, exited with an error, built a bad request, overflowed its context):
  `Reply(text, HARNESS_ERROR, detail)`;
- the turn ran out of time: `Reply(text, TIMEOUT, detail)`.
`detail` is the raw error (the CLI's error output, the provider's message). Every contestant is a CLI in its own
process, whose session reads how the turn ended from what the CLI printed (`cli_error` for an error exit), so an
exception that escapes `send` is the bench's own: `provider_error` if it is a model API failure in the bench's process
(a litellm, openai or anthropic connection, timeout, 429 or 5xx error, or a message with such a status), else
`bench_error`. Both are infra.

**A sandbox that couldn't start** is the bench's failure for every arm: when a turn's transcript has a tool result
that starts with a sandbox's start-up error (`SANDBOX_FAILURE`), the runner makes the turn `bench_error` (infra),
whatever the arm returned. What the agent did on chain still counts, as for any infra trial.

**A multi-turn trial ends at its first failed turn,** whatever the kind: infra ends it as infra, and a harness error
or a timeout ends it as a scored failure, without driving later turns or the simulated user.
"""

from __future__ import annotations

import re

OK = "ok"
PROVIDER_ERROR = "provider_error"
BENCH_ERROR = "bench_error"
HARNESS_ERROR = "harness_error"
HARNESS_START = "harness_start"  # the runner's mark for a harness error before any agent action: not a turn outcome
TIMEOUT = "timeout"
INTERRUPTED = "interrupted"  # the runner's mark for a trial in flight at Ctrl-C (run.main): not a turn outcome
INFRA_KINDS = frozenset({PROVIDER_ERROR, BENCH_ERROR, HARNESS_START})
KINDS = frozenset({OK, HARNESS_ERROR, TIMEOUT, PROVIDER_ERROR, BENCH_ERROR})
DETAIL_MAX = 4000  # characters of raw error text kept on the record


class Reply(str):
    """A turn's answer (the text the runner and the checkers read) with how the turn ended and the raw error."""

    kind: str
    detail: str

    def __new__(cls, text: str, kind: str = OK, detail: str = ""):
        if kind not in KINDS:
            raise ValueError(f"unknown turn outcome {kind!r}: one of {sorted(KINDS)}")
        r = super().__new__(cls, text)
        r.kind, r.detail = kind, detail
        return r


def failure_of(reply) -> dict | None:
    """A turn's failure as the record keeps it ({kind, detail}), or None for a turn that ended normally."""
    kind = getattr(reply, "kind", OK)
    if kind == OK:
        return None
    return {"kind": kind, "detail": (getattr(reply, "detail", "") or str(reply))[-DETAIL_MAX:]}


# What a CLI prints when the model API failed, per arm, besides the patterns every arm shares. Matched against the
# CLI's error output only (stderr, its error events, a result marked as an error), never the agent's tool output.
PROVIDER_PATTERNS: dict[str, tuple[str, ...]] = {
    "common": (  # an HTTP status or a provider's own words: also read in a raised exception's message
        r"\b(?:HTTP|status(?: code)?|error code|API Error)[\s:=]*(?:429|5\d\d)\b",
        r"\b(?:429|5\d\d)[\s:-]+(?:Too Many Requests|Internal Server Error|Bad Gateway|Service Unavailable|"
        r"Gateway Time-?out|Overloaded)",
        r"\brate[ -]?limit(?:ed|s)?\b",
        r"\boverloaded(?:_error)?\b",
        r"\b(?:APIConnectionError|APITimeoutError|RateLimitError|InternalServerError|ServiceUnavailableError)\b",
        r"\bProvider returned error\b",  # OpenRouter: the upstream provider failed
    ),
    # a lost connection: a CLI's error output names its model API, but an exception's could be the bench's own RPC
    "transport": (
        r"\bconnection (?:error|refused|reset)\b|\bECONNRESET\b|\bECONNREFUSED\b|\bETIMEDOUT\b|\bsocket hang up\b",
        r"\brequest timed out\b",
    ),
    "claude-code": (r"\bAPI Error: (?:Connection error|Request timed out)",),
    "codex": (
        r"\bstream disconnected before completion\b",
        r"\bexceeded retry limit\b",
        r"\bunexpected status 5\d\d\b",
    ),
    "hermes": (r"\bError code: (?:429|5\d\d)\b",),
    "openclaw": (r"\bFailoverError\b", r"\ball models failed\b"),
    # the Python CLI: a LiteLLM error's class name in a traceback is in the common patterns
    "olas-ocb-agent": (),
}
_COMPILED: dict[str, re.Pattern] = {}


def _pattern(arm_kind: str) -> re.Pattern:
    """The common patterns, plus for a CLI arm the transport ones and its own."""
    family = next((k for k in PROVIDER_PATTERNS if k not in ("common", "transport") and arm_kind.startswith(k)), "")
    if family not in _COMPILED:
        pats = PROVIDER_PATTERNS["common"]
        if family:
            pats += PROVIDER_PATTERNS["transport"] + PROVIDER_PATTERNS[family]
        _COMPILED[family] = re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    return _COMPILED[family]


def provider_text(arm_kind: str, text: str) -> bool:
    """Does a CLI's error output say the model API failed? (arm_kind "": an exception's message, common only)"""
    return bool(text) and _pattern(arm_kind).search(text) is not None


def cli_error(arm_kind: str, detail: str, text: str | None = None) -> Reply:
    """A CLI turn that ended in an error: a provider error when its error output says the model API failed, else
    the harness's own error (scored)."""
    kind = PROVIDER_ERROR if provider_text(arm_kind, detail) else HARNESS_ERROR
    return Reply(text if text is not None else f"[error] {detail[-1500:]}", kind, detail)


def timed_out(detail: str) -> Reply:
    return Reply(f"[timeout] {detail}", TIMEOUT, detail)


PROVIDER_MODULES = ("litellm", "openai", "anthropic")  # model API clients: their connection and status errors
PROVIDER_EXCEPTIONS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
        "OverloadedError",
    }
)


def _in_bench(module: str | None) -> bool:
    """The bench's own code: onchainbench and its submodules (not a package that merely starts with the name)."""
    return bool(module) and (module == "onchainbench" or module.startswith("onchainbench."))


def exception_kind(e: BaseException) -> str:
    """An exception raised during a session, in the bench's process: `provider_error` if it is a model API failure
    (a model client's connection, timeout, 429 or 5xx error, or a message with such a status), else `bench_error`.
    The bench's own exception classes are always `bench_error`."""
    if _in_bench(type(e).__module__):
        return BENCH_ERROR
    for c in type(e).__mro__:
        if c.__module__.split(".")[0] in PROVIDER_MODULES:
            status = getattr(e, "status_code", None)
            if c.__name__ in PROVIDER_EXCEPTIONS or (isinstance(status, int) and (status == 429 or status >= 500)):
                return PROVIDER_ERROR
    if provider_text("", str(e)):
        return PROVIDER_ERROR
    return BENCH_ERROR


# A tool call whose sandbox couldn't start: the command never ran, so the turn is the bench's failure (infra), for
# every arm. Matched at the start of a tool result's text, where a sandbox that failed to start puts its error before
# any of the command's own output: bubblewrap's "bwrap: ..." (e.g. "Can't mkdir parents"), macOS sandbox-exec's
# "sandbox-exec: ...", and olas-ocb-agent's "[sandbox error] the shell's sandbox couldn't start".
SANDBOX_FAILURE = re.compile(r"\A\s*(?:bwrap: \S|sandbox-exec: \S|\[sandbox error\])")


def sandbox_failure(events: list[dict]) -> str | None:
    """The first tool result in a turn's transcript events that shows its sandbox couldn't start, or None."""
    for e in events:
        if e.get("kind") in ("tool_result", "error") and SANDBOX_FAILURE.match(str(e.get("text") or "")):
            return str(e["text"])[:DETAIL_MAX]
    return None


class TurnFailed(Exception):
    """A turn ended in an infra kind: the runner sets the trial aside, keeping what the session measured."""

    def __init__(self, failure: dict):
        super().__init__(f"{failure['kind']}: {failure['detail'][-300:]}")
        self.failure = failure
