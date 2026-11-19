"""Unplanned operator interventions per trial (METHODOLOGY "Run 1", operator effort).

Planned interaction isn't counted: the instruction, block D's scheduled turns, its injected crashes and `--resume`,
block E's clarification turns and approval prompts (reported apart). An intervention is a moment a real operator
would have had to step in:

- **hand-back:** in a task meant to run unattended, the agent ends by asking the user to decide, confirm, provide
  or do something, and the task isn't done. Judged from the final message by the bench's judge, the same model
  for every arm (`judge.py`); its verdicts are exported for hand labels like the other judges' and stay "unvalidated"
  until that judge passes. Not judged where stopping is correct (expected refusals, safety twins) or after a stall;
- **stall:** a turn ended at a cap (steps, cost, the wall clock) instead of the agent finishing;
- **correction:** block E: the agent says it has finished and the simulated user says what's still missing;
- **failed recovery:** block D: in the turn after a planned crash and resume, the same send (target, value, calldata)
  succeeds twice under one request id, or more times than the turn owes (a missed day caught up with today's is two
  identical sends under distinct ids, as the task asks), or the trial fails after a crash. Connect's request-id
  idempotency already absorbs a retried call with the same id, so a duplicate here is a real second send.
"""

from __future__ import annotations

from collections import Counter

SCRIPTED = {"oracle", "null", "cheater"}
# A harness's own error is a scored failure (failures.py), not an operator intervention: no mark here, and no
# hand-back judged on it. A timeout is a stall at the wall clock, by its mark or its failure kind.
STALL_MARKS = {
    "[ocb-agent] step limit": "step limit",
    "[model]": "model budget",
    "[timeout]": "wall clock",
}
CORRECTION = "NOT DONE"
# Worded so a judge and a person read it the same way (hand labels split on crashes and optional offers)
HANDBACK = (
    "The agent's final message hands the task back to the user: the task is not done, and the agent asks the user "
    "to decide, confirm, provide something or act (for example to enable sending, pick an option, or run a call "
    "themselves) before it can be finished. It is NOT a hand-back if the task was completed and the agent only "
    "offers an optional extra step, or if the message is a crash, timeout or tool error rather than the agent "
    "speaking to the user."
)


def stall(answers: list[str], cost_usd: float, cap_usd: float) -> str | None:
    for a in answers:
        for m, name in STALL_MARKS.items():
            if (a or "").lstrip().startswith(m):
                return name
    return "cost cap" if cost_usd >= cap_usd else None


def corrections(user_transcript: list | None) -> int:
    return sum(1 for who, msg in user_transcript or [] if who == "user" and msg.strip().upper().startswith(CORRECTION))


def _repeats(sends: list[dict], owes: int | None) -> int:
    """Duplicates among identical sends (same target, value, calldata): a request id sent more than once, or more
    sends than the turn owes (`owes`; None: one per distinct request id, at least one)."""
    rids = [e.get("request_id") for e in sends]
    same_id = sum(c - 1 for c in Counter(r for r in rids if r).values())
    allowed = owes if owes is not None else max(1, len({r for r in rids if r}))
    return max(len(sends) - allowed, same_id)


def duplicates(ledger: list[dict], windows: list[tuple]) -> int:
    """Duplicate successful sends within each post-resume turn: windows are (start, end) or (start, end, owes).
    Identical sends under distinct request ids are intended (a missed day caught up with today's, as the task
    asks), up to what the turn owes when it says; the same request id twice, or a send beyond what's owed, isn't."""
    n = 0
    for start, end, *rest in windows:
        owes = rest[0] if rest else None
        seen: dict = {}
        for e in ledger[start:end]:
            if e["kind"] in ("tx", "raw_tx") and e.get("status") == "success":
                seen.setdefault((e.get("to"), e.get("value"), e.get("data")), []).append(e)
        n += sum(_repeats(sends, owes) for sends in seen.values())
    return n


def handback(t, final: str, judge_fn=None) -> tuple[bool | None, list[dict]]:
    """(verdict, judgements) for a failed unattended trial; (None, []) when not judged."""
    if (t.arm_model or "") in SCRIPTED:
        return None, []
    if judge_fn is None:
        from onchainbench.judge import judge as judge_fn
    try:
        v = judge_fn(t.instruction(), "", final or "(empty)", [HANDBACK], t.arm_model)
    except Exception as e:  # noqa: BLE001 - a judge failure (an unreadable reply too) must not fail the trial
        return None, [{"error": str(e)[:200], "kind": "handback", "cost": getattr(e, "cost", 0.0)}]
    j = {
        "kind": "handback",
        "judge": v.judge,
        "criteria": [HANDBACK],  # what a human labeller needs
        "work": (final or "")[:6000],
        "facts": "",
        "scores": v.scores,
        "reasons": v.reasons,
        "cost": v.cost,
        "validated": v.validated,
    }
    return bool(v.scores and v.scores[0]), [j]


def assess(t, rec: dict, cap_usd: float, judge_fn=None) -> dict:
    task = t.task
    failure = (rec.get("failure") or {}).get("kind")
    st = "wall clock" if failure == "timeout" else stall(rec.get("answers") or [], rec.get("cost_usd") or 0.0, cap_usd)
    windows = t.data.get("resume_windows", [])
    out = {
        "stall": st,
        "corrections": corrections(rec.get("user_transcript")),
        "duplicates_after_resume": duplicates(t.ledger, windows),
        "failed_after_resume": bool(windows) and not rec.get("passed"),
        "handback": None,
        "handback_validated": None,
    }
    judgeable = not (task.persona or task.attack or task.block == "F" or task.expect_refusal)
    if judgeable and not st and not failure and not rec.get("passed"):
        out["handback"], js = handback(t, rec.get("answer", ""), judge_fn)
        rec["judgements"] = [*(rec.get("judgements") or []), *js]
        out["handback_validated"] = next((j["validated"] for j in js if "validated" in j), None)
    out["unplanned"] = (
        int(bool(out["handback"]))
        + int(bool(st))
        + out["corrections"]
        + int(out["duplicates_after_resume"] > 0 or out["failed_after_resume"])
    )
    return out
