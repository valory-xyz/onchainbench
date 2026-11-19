"""Merge the shards of one run into a single run, as if it had run in one go.

    python -m onchainbench.merge [--key-wait SECONDS] [--expect-shards S1,S2,...] [--expect-arms A1,A2,...]
                                 <out-run> <shard-run> [<shard-run> ...]   # runs under results/

The shards must share the split, seed and attempts (the same (task, attempt, seed) gives the same parameters), and
their trials the versions they ran on: the bench's code and task data (`bench_digest`; a record without one is
refused, since it can't be compared), the held-out manifest and the forks' Anvil build (`anvil`, its
web3_clientVersion), and per arm the frozen agent's code digest and the price (each trial's `provenance`;
a record from before the stamps takes its shard's config.json values, so older shards merge as before when they
match, and a mix of older and newer is compared the same way).
Trials are keyed by (arm, task, attempt): a scored trial replaces an infra failure of the same key (and either replaces
an interrupted one), except an infra failure that shows a loss, which is final and kept over any other record of
the key (a loss once recorded stays recorded, as with `--fill`), and the same key
scored in two shards is an error (a re-run would silently replace a result), unless it is one trial that `--fill`
kept in both (the same `trial_dir`). The merged config.json lists every shard's arms and tasks, and unions every
per-arm map (prices, cap and step-limit enforcement); summary.json is recomputed from the merged trials. Both list
under `missing` the expected shards that aren't there (`--expect-shards`, else the directories given without a
config.json), shards with no trials and arms with none (`--expect-arms`, else every shard's listed arms), so a lost
shard shows instead of vanishing. Where each shard ran one arm, the key's spend over the arm's
shards is shared out over its trials (onchainbench/billing.py); `spending.by_arm` gives each arm's billed total
beside its estimate, or why it couldn't be attributed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from onchainbench.run import ROOT, key_usage, provenance_conflicts, relocate, spending
from onchainbench.stats import has_recorded_loss
from onchainbench.stats import is_infra as _infra
from onchainbench.stats import is_scored as _scored

# and every trial's provenance (run.provenance_conflicts); no_judges: judged and unjudged trials never mix in one run
SAME = ("split", "seed", "attempts", "no_judges")


def _same(cfg: dict, k: str):
    """A shard's value of k; a config from before --no-judges ran with the judges."""
    return bool(cfg.get(k)) if k == "no_judges" else cfg.get(k)


def _rank(r: dict) -> int:
    """Which of two records of one key a merge keeps (the later on a tie): an infra failure that shows a loss
    (stats.has_recorded_loss: final, as in --fill) over anything, then a scored or unsupported trial over an infra
    failure, and either over an interrupted one (in flight at Ctrl-C: no result at all)."""
    if r.get("interrupted"):
        return 0
    if _infra(r):
        return 3 if has_recorded_loss(r) else 1
    return 2


class MergeError(Exception):
    pass


def _read(d: Path) -> tuple[dict, list[dict]]:
    cfg = json.loads((d / "config.json").read_text())
    p = d / "trials.jsonl"  # a shard that ran no trial may not have written one
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
    return cfg, rows


def _per_arm_maps(cfgs: list[dict]) -> list[str]:
    """Config keys that map each arm to a value (prices, cap_enforcement, step_limit_enforcement, ...): a dict whose
    keys are all among its shard's arms."""
    keys = []
    for c in cfgs:
        arms = set(c.get("arms") or [])
        for k, v in c.items():
            if isinstance(v, dict) and v and set(v) <= arms and k not in keys:
                keys.append(k)
    return keys


def missing(
    shards: list[Path], rows: list[list[dict]], cfgs: list[dict], gone: list[str], expect_shards, expect_arms
) -> dict:
    """What the merged run should hold but doesn't: expected shards not given or without a config (`gone`), shards
    that produced no trials, and arms with no trial. Expected arms default to every shard's listed arms."""
    names = [d.name for d in shards]
    arms = list(expect_arms or dict.fromkeys(a for c in cfgs for a in c.get("arms") or []))
    seen = {r["arm"] for rs in rows for r in rs}
    return {
        "expected_shards": list(expect_shards) if expect_shards else [*names, *gone],
        "expected_arms": arms,
        "shards": list(dict.fromkeys([*(s for s in expect_shards or [] if s not in names), *gone])),
        "empty_shards": [n for n, rs in zip(names, rows, strict=True) if not rs],
        "arms": [a for a in arms if a not in seen],
    }


def merge(
    out: Path,
    shards: list[Path],
    key_wait_s: float = 0,
    expect_shards: list[str] | None = None,
    expect_arms: list[str] | None = None,
) -> dict:
    """key_wait_s: after the shards, wait this long and read the OpenRouter key once for the whole run's billed
    total (OpenRouter counts usage minutes late, so each shard's own reading falls short); 0 skips it.
    expect_shards / expect_arms: the shards and arms the run was launched with; any missing from the merge (a
    shard never uploaded, one with no trials, an arm with none) are listed in config.json and summary.json
    (`missing`) instead of vanishing. A shard directory without a config.json counts as missing."""
    gone = [d for d in shards if not (d / "config.json").is_file()]
    shards = [d for d in shards if d not in gone]
    if not shards:
        raise MergeError("no shards to merge")
    read = [_read(d) for d in shards]
    cfgs, rows = [c for c, _ in read], [r for _, r in read]
    for k in SAME:
        if len({json.dumps(_same(c, k)) for c in cfgs}) > 1:
            raise MergeError(f"shards differ in {k}: {[_same(c, k) for c in cfgs]}")
    tagged = [(d.name, r, c) for d, c, shard in zip(shards, cfgs, rows, strict=True) for r in shard]
    if why := provenance_conflicts(tagged):
        raise MergeError("shards ran on different versions: " + " | ".join(why))
    summaries = [json.loads((d / "summary.json").read_text()) if (d / "summary.json").is_file() else {} for d in shards]
    best: dict = {}
    for d, shard in zip(shards, rows, strict=True):
        for r in shard:
            key = (r["arm"], r["task"], r["attempt"])
            if key in best and _scored(r) and _scored(best[key][1]):
                if r.get("trial_dir") is None or r.get("trial_dir") != best[key][1].get("trial_dir"):
                    raise MergeError(
                        f"{key[0]} {key[1]} #{key[2]} is scored in both {best[key][0].name} and {d.name}: "
                        "a re-run would replace a result (drop one shard's trial)"
                    )
                continue  # the same trial, kept by --fill in both shards
            if key not in best or _rank(r) >= _rank(best[key][1]):
                best[key] = (d, r)
    out.mkdir(parents=True, exist_ok=True)
    records = [relocate(r, d, out) for d, r in best.values()]  # transcripts and session files come along
    cfg = {**cfgs[0], "run": out.name, "merged_from": [d.name for d in shards]}
    for k in ("arms", "tasks"):
        cfg[k] = list(dict.fromkeys(x for c in cfgs for x in c.get(k) or []))
    for k in _per_arm_maps(cfgs):  # prices, cap_enforcement, step_limit_enforcement, and any later per-arm map
        cfg[k] = {a: v for c in cfgs for a, v in (c.get(k) or {}).items()}
    forks: dict = {}  # each shard's forks' builds (provenance_conflicts already refused two different ones)
    for c in cfgs:
        for chain, versions in (c.get("fork_client_versions") or {}).items():
            forks[chain] = sorted({*forks.get(chain, []), *versions})
    if forks:
        cfg["fork_client_versions"] = dict(sorted(forks.items()))
    gaps = missing(shards, rows, cfgs, [d.name for d in gone], expect_shards, expect_arms)
    cfg["expected"] = {"shards": gaps.pop("expected_shards"), "arms": gaps.pop("expected_arms")}
    cfg["missing"] = gaps
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    from onchainbench.billing import apportion, shard_spend
    from onchainbench.stats import summarize

    starts = [c["key_usage_start_usd"] for c in cfgs if c.get("key_usage_start_usd") is not None]
    end = None
    if key_wait_s and starts:
        time.sleep(key_wait_s)
        end = key_usage(settle=True)
    meta = [
        {
            "name": d.name,
            "arms": c.get("arms") or [],
            "start": c.get("key_usage_start_usd"),
            "billed": (s.get("spending") or {}).get("key_billed_usd"),
            "rows": rs,
        }
        for d, c, s, rs in zip(shards, cfgs, summaries, rows, strict=True)
    ]
    by_arm = apportion(records, meta, end, cfg.get("cap_enforcement"))  # before the trials and summary are written
    (out / "trials.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    summary = summarize(records)
    summary["spending"] = spending(records, None)
    if key_wait_s and starts:
        summary["spending"]["key_billed_usd"] = round(end - min(starts), 4) if end is not None else None
    summary["spending"]["shards_key_billed_usd"] = {m["name"]: m["billed"] for m in meta}
    summary["spending"]["shards_key_spend"] = shard_spend(meta, end)[0]  # to the next reading: OpenRouter's lag
    summary["spending"]["by_arm"] = by_arm
    summary["missing"] = gaps  # shards and arms the run expected and the merge doesn't hold
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"trials": len(records), "shards": len(shards), **({"missing": gaps} if any(gaps.values()) else {})}


def main(argv: list[str]) -> int:
    wait, expect = 0.0, {"--expect-shards": None, "--expect-arms": None}
    while argv[:1] and argv[0] in ("--key-wait", *expect) and len(argv) > 1:
        if argv[0] == "--key-wait":
            wait = float(argv[1])
        else:
            expect[argv[0]] = [x for x in argv[1].split(",") if x]
        argv = argv[2:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    try:
        res = ROOT / "results"
        out, shards = res / argv[0], [res / s for s in argv[1:]]
        print(merge(out, shards, wait, expect["--expect-shards"], expect["--expect-arms"]))
    except (MergeError, OSError, KeyError, ValueError) as e:
        print(f"merge failed: {e}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
