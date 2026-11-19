"""Each arm's billed cost, from the OpenRouter key's spend over its shards, shared out over its trials.

Most arms can't see what OpenRouter charged them (Codex, Hermes and Claude Code report tokens; the
trial's cost is an estimate). When a run is sharded one arm per shard, the key's spend over a shard is what that arm
cost plus the bench's own calls in it (judges and the simulated user, billed per call and recorded on the trials).
`merge` takes the difference and shares it out over the arm's trials in proportion to their tokens (their estimates
where the arm reports no tokens): those trials get `cost_source: "billed-apportioned"`, with the estimate kept as
`cost_estimate_usd`. Trials already billed keep their cost.

A shard's spend runs from its starting reading of the key to the next reading after it: the next shard's start, or
the reading `merge --key-wait` takes at the end. OpenRouter counts usage minutes late, so a shard's own closing
reading misses its last calls (the smoke run's Codex shard read $1.13 at its end and $1.41 by the next reading).

Nothing is attributed, and the estimates stay, when:
- the shards' readings overlap (shards ran at the same time on the key, so each reading holds the others' spend);
- a shard holds more than one arm, or the arm's calls don't go through the run's key (Anthropic directly, the mech,
  per-trial keys);
- a shard has no reading after it (no `--key-wait`, or the reading failed);
- the usage that arrived after a shard's own closing reading is more than LATE_MAX of what it read itself and more
  than LATE_FLOOR_USD (so a near-zero shard's few calls of lag don't fail the share test), or the
  billed total is outside BAND times the estimate: another user of the key, most likely, which the key's total
  can't tell apart from the run's own spend.
"""

from __future__ import annotations

from itertools import combinations

from onchainbench import arms as A

EPS = 1e-4  # USD: a shard records its billed reading rounded to 4 places
LATE_MAX = 0.5  # usage after a shard's own closing reading, as a share of that reading, still taken as its lag
# ... or up to this much in USD, whatever the share: a near-zero shard ($0.0006 read, $0.0006 late) is lag, not
# another user of the key
LATE_FLOOR_USD = 0.01


def late_max(own: float) -> float:
    """The most late usage still taken as a shard's own lag."""
    return max(LATE_MAX * own, LATE_FLOOR_USD)


BAND = (1 / 3, 3.0)  # billed over estimate: outside it, the key's spend is not taken as the arm's
SOURCE = "billed-apportioned"


def bench_billed(r: dict) -> float:
    """What the bench itself spent on a trial: its judges and its simulated user."""
    judges = sum(float(j.get("cost") or 0.0) for j in r.get("judgements") or [] if isinstance(j, dict))
    return judges + float(r.get("user_sim_cost_usd") or 0.0)


def own_cost(r: dict) -> float:
    """The arm's part of a trial's cost (`cost_usd` includes the simulated user's)."""
    return float(r.get("cost_usd") or 0.0) - float(r.get("user_sim_cost_usd") or 0.0)


def tokens(r: dict) -> int:
    return int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)


def overlaps(shards: list[dict]) -> dict[str, list[str]]:
    """{shard: the shards that ran at the same time on the key}, for shards whose readings span another's: a shard's
    span on the key runs from its starting reading to that plus its own billed reading."""
    closed = [s for s in shards if s.get("start") is not None and s.get("billed") is not None]
    out: dict = {}
    for a, b in combinations(closed, 2):
        if a["start"] < b["start"] + b["billed"] - EPS and b["start"] < a["start"] + a["billed"] - EPS:
            out.setdefault(a["name"], []).append(b["name"])
            out.setdefault(b["name"], []).append(a["name"])
    return out


def shard_spend(shards: list[dict], end: float | None) -> tuple[dict, dict[str, list[str]]]:
    """({shard name: {key_usd, own_usd, late_usd} or None}, overlaps(shards)) from each shard's starting reading
    (`start`), its own billed reading (`billed`), and the reading after the last shard (`end`). A shard that ran at
    the same time as another is None: its reading holds the other's spend."""
    timed = sorted((s for s in shards if s.get("start") is not None), key=lambda s: s["start"])
    both = overlaps(shards)
    out: dict = {s["name"]: None for s in shards}
    for i, s in enumerate(timed):
        if s["name"] in both:
            continue
        after = timed[i + 1]["start"] if i + 1 < len(timed) else end
        if after is None:
            continue  # no reading after this shard: its last calls may not be counted yet
        own = s.get("billed")
        spend = after - s["start"]
        late = spend - own if own is not None else None
        if late is not None and late > late_max(own) + EPS:
            out[s["name"]] = {"key_usd": None, "own_usd": own, "late_usd": round(late, 6)}
            continue
        out[s["name"]] = {"key_usd": round(spend, 6), "own_usd": own, "late_usd": round(late, 6) if late else 0.0}
    return out, both


def _via_key(arm: str, cap: dict) -> bool:
    """The arm's model calls go through the run's OpenRouter key (not Anthropic directly, the mech, or per-trial
    keys, whose spend the run's key doesn't see)."""
    return A.parse_arm(arm)[1].startswith("openrouter/") and cap.get(arm) != "per-trial key"


def apportion(records: list[dict], shards: list[dict], end: float | None, cap: dict | None = None) -> dict:
    """Share each arm's billed spend out over its trials (in place) where it can be attributed; returns, per arm,
    the billed total beside the estimate, and why not where it isn't attributed.

    `shards`: [{name, arms, start, billed, rows}], `rows` the shard's own trials (the bench's calls are counted from
    them, a retried trial's included); `end`: the key's reading after the last shard; `cap`: config's
    cap_enforcement."""
    cap = cap or {}
    spend, both = shard_spend(shards, end)
    by_arm: dict = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)
    out = {}
    for arm, rs in by_arm.items():
        mine = [s for s in shards if arm in (s.get("arms") or [])]
        estimate = sum(own_cost(r) for r in rs)
        rep: dict = {"estimate_usd": round(estimate, 4), "billed_usd": None, "shards": [s["name"] for s in mine]}
        out[arm] = rep
        if A.parse_arm(arm)[0] in A.SCRIPTED_ARMS:
            rep["status"] = "no model"
            continue
        why = None
        if not _via_key(arm, cap):
            why = "its model calls don't go through the run's key"
        elif shared := [s["name"] for s in mine if s["name"] in both]:  # its own shards' overlap, by name
            why = "; ".join(f"shard {n} ran at the same time as {', '.join(both[n])} on the key" for n in shared)
        elif not mine:
            why = "no shard lists it"
        elif any(len(s.get("arms") or []) > 1 for s in mine):
            why = "a shard holds several arms"
        elif missing := [s["name"] for s in mine if (spend.get(s["name"]) or {}).get("key_usd") is None]:
            late = [n for n in missing if spend.get(n)]
            why = (
                f"usage after {', '.join(late)} is more than {LATE_MAX:.0%} of its own reading "
                f"(and over {LATE_FLOOR_USD} USD): another user of the key?"
                if late
                else f"no reading of the key after {', '.join(missing)} (merge without --key-wait, or it failed)"
            )
        if why:
            rep |= {"status": "estimate kept", "reason": why}
            continue
        key = sum(spend[s["name"]]["key_usd"] for s in mine)
        bench = sum(bench_billed(r) for s in mine for r in s.get("rows") or [])
        billed = key - bench
        done = [r for r in rs if r.get("cost_source") == "billed"]
        todo = [r for r in rs if r.get("cost_source") != "billed"]
        weigh_tokens = sum(tokens(r) for r in todo) > 0
        weight = {id(r): tokens(r) if weigh_tokens else max(own_cost(r), 0.0) for r in todo}
        todo = [r for r in todo if weight[id(r)] > 0]
        left = billed - sum(own_cost(r) for r in done)
        rep |= {"key_usd": round(key, 4), "bench_usd": round(bench, 4), "billed_usd": round(billed, 4)}
        est = sum(own_cost(r) for r in todo)
        if not todo:
            rep |= {"status": "billed", "unattributed_usd": round(left, 4)}  # every trial billed already
            continue
        if left <= EPS:
            rep |= {"status": "estimate kept", "reason": f"nothing left for the unbilled trials ({left:.4f} USD)"}
            continue
        if est > EPS and not BAND[0] <= left / est <= BAND[1]:
            rep |= {
                "status": "estimate kept",
                "reason": f"billed {left:.4f} USD is {left / est:.2f}x the estimate: another user of the key?",
            }
            continue
        total = sum(weight[id(r)] for r in todo)
        for r in todo:
            share = left * weight[id(r)] / total
            r["cost_estimate_usd"] = r.get("cost_usd") or 0.0
            r["cost_usd"] = share + float(r.get("user_sim_cost_usd") or 0.0)
            r["cost_source"] = SOURCE
        rep |= {"status": "apportioned", "weight": "tokens" if weigh_tokens else "estimate", "trials": len(todo)}
    return out
