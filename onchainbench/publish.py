"""Publish a run: the run JSON (onchainbench/schemas/run.schema.json), and for open runs the trajectories.

    python -m onchainbench.publish <run> --release-url URL [--bench-version X] [--prereg-commit SHA]
        [--hf-row-url URL] [--date YYYY-MM-DD] [--no-trials] [--verified]
        [--upload-release TAG] [--hf-results-repo valory/onchainbench-results]
        [--hf-trajectories-repo valory/onchainbench-trajectories]

Reads results/<run>/trials.jsonl and config.json and writes results/<run>/run.json (and trajectories.jsonl on the
open split), and SHA256SUMS over config.json, trials.jsonl, summary.json, inspect/*.json, trajectories.jsonl and
run.json (not yet sessions/ or the raw transcripts); run.json carries the others' sums as `artifacts_sha256`.
`--upload-release` attaches run.json, trajectories.jsonl and SHA256SUMS, and the Hugging Face uploads keep those
names under runs/<id>/; the other files stay in the run's archive, and the public run.json commits to them until
they are published. `python -m onchainbench.verify <dir>` checks a download. The scripted validation arms (oracle,
null, cheater, obeyer) are not entries. Older run files may carry a pair's `comparator` flag, or an entry's
`reference` and `reference_note` (an unranked reference baseline); the schema keeps them optional.

Entries are published as unverified (self-reported) by default. Only `--verified` marks them verified, which the
maintainers pass for a run Valory produced itself; a results file someone sends in is never verified by default.

Held-out runs never publish task ids or text: per-trial rows carry an opaque id (a salted hash; the salt is the
held-out package's PUBLIC-ID-SALT, so only the held-out repo can map ids back), no trajectories are written, and
the run's archive stays private. Every file is checked for held-out ids, id shapes and instruction text before it is
written (onchainbench.leaks), and the run JSON is validated against the run schema
(onchainbench/schemas/run.schema.json, the canonical copy).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

from onchainbench import arms as A
from onchainbench import leaks, stats, transcript

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = Path(__file__).with_name("schemas") / "run.schema.json"
SALT_FILE = "PUBLIC-ID-SALT"
# harnesses whose code is public
OPEN_HARNESSES = {"codex", "hermes", "hermes-raw", "openclaw", "openclaw-raw", "olas-ocb-agent", "olas-ocb-agent-mech"}
SPLITS = {"open": "open", "test": "heldout"}
SUMS = "SHA256SUMS"
ARCHIVE = ("config.json", "trials.jsonl", "summary.json")  # with inspect/*.json: what SHA256SUMS covers


class PublishError(Exception):
    """The run can't be published as asked (mixed splits, a leak, a schema violation, no salt)."""


def opaque_id(task: str, salt: str) -> str:
    return "h-" + hashlib.sha256(f"{salt}:{task}".encode()).hexdigest()[:12]


def heldout_salt() -> str:
    from onchainbench.tasks import heldout_dir

    d = heldout_dir()
    if d is None or not (d / SALT_FILE).is_file():
        raise PublishError(f"no {SALT_FILE} in the held-out package: held-out task ids can't be made opaque")
    return (d / SALT_FILE).read_text().strip()


def rate(rs: list[dict]) -> dict:
    """The per-task mean pass rate and its 95% interval (stats.pass_rate), normal on the family-clustered SE. Where
    that SE measures no spread there is no interval: `ci95` and `se` null, `ci_method` no_spread, and the site shows
    the task and family counts instead."""
    r = stats.pass_rate(rs)
    return {
        "mean": round(r["mean"], 4),
        "se": None if r["se"] is None else round(r["se"], 4),
        "ci95": None if r["ci95"] is None else [round(x, 4) for x in r["ci95"]],
        "ci_method": r["ci_method"],
        "n_tasks": r["n_tasks"],
        "n_families": r["n_families"],
        "n_trials": len(rs),
    }


def _grouped(rs: list[dict], key) -> dict:
    by = defaultdict(list)
    for r in rs:
        for k in key(r):
            by[k].append(r)
    return {k: rate(v) for k, v in sorted(by.items())}


def entry_name(arm: str) -> tuple[str, str, str | None]:
    """(harness, model, provider): `kind:model?provider=x|mod` -> the harness with its modifiers, the bare model."""
    kind, model, mods = A.parse_arm(arm)
    model, _, query = model.partition("?")
    provider = dict(kv.partition("=")[::2] for kv in query.split("&") if kv).get("provider")
    return kind + "".join(f"|{m}" for m in sorted(mods)), model, provider


def safety(rs: list[dict]) -> dict:
    """Executed losses per twin attempt and their one-sided 95% upper bound. With no twin attempts nothing was
    measured: `upper_95` is null and `safety_measured` false, never a bound of 100% (the site says "not measured").
    `funnel` holds the attack-outcome counts per twin trial (stats.attack_funnel), all null with no twin trial."""
    twins = [r for r in rs if r.get("attack")]
    s = stats.attack_summary(twins)
    by_class = {}
    for cls in sorted({r["attack"] for r in twins}):
        c = stats.attack_summary([r for r in twins if r["attack"] == cls])
        by_class[cls] = {
            "trials": c["trials"],
            "executed_losses": c["executed_losses"],
            "upper_95": c["executed_upper_95"],  # null with no attempts in the class
        }
    return {
        "executed_losses": s["executed_losses"],
        "failed": s["failed_onchain"],
        "trials": s["trials"],
        "upper_95": s["executed_upper_95"],  # null with no twin attempts
        "safety_measured": s["trials"] > 0,
        **({"attempted_rate": s["attempted_rate"]} if s["attempted_rate"] is not None else {}),
        "blocked_by": s["blocked_by"],
        "by_class": by_class,
        # the documented counts behind the attack-outcome chart: every field null with no twin trial (not measured)
        "funnel": s["funnel"] or dict.fromkeys(stats.FUNNEL_FIELDS),
    }


def entry(arm: str, rs: list[dict], cfg: dict, verified: bool, salt: str | None = None) -> dict:
    harness, model, provider = entry_name(arm)
    kind = A.parse_arm(arm)[0]
    scored = [r for r in rs if stats.is_scored(r)]
    summary = stats.arm_summary(rs)
    frozen = A.frozen_package(kind)
    e = {
        "harness": harness,
        "model": model,
        **({"provider": provider} if provider else {}),
        "open": kind in OPEN_HARNESSES,
        "verified": verified,
        "completion": rate(scored),
        # every trial, as the run report counts them: a twin lost to infra counts in the bound when it shows a loss
        # (stats.attack_summary), and the funnel still counts scored trials only
        "safety": safety(rs),
        "interventions_per_pass": summary["autonomy"]["interventions"]["unplanned_per_pass"],
        "cost_per_pass_usd": summary["cost_per_pass_usd"],
        "seconds_per_pass": summary["seconds_per_pass"],
        "infra_rate": summary["infra_rate"],
        "infra_high": summary["infra_high"],
        "by_block": _grouped(scored, lambda r: [r["block"]]),
        "by_chain": _grouped(scored, lambda r: [r["chain"]]),
        "by_protocol": _grouped(scored, lambda r: r.get("protocols") or []),
    }
    if summary["scored_on_subset"]:  # tasks it can't run: scored on the rest, shown with an asterisk and the count
        ids = summary["unsupported_tasks"]
        e |= {
            "scored_on_subset": True,
            "unsupported_tasks": len(ids),
            "unsupported_task_ids": sorted(opaque_id(t, salt) for t in ids) if salt else ids,
        }
    # what the entry spent its effort on (model calls, Connect tool calls, on-chain actions): reported, not limited
    e["effort"] = {k: v for k, v in summary["effort"].items() if v is not None}
    if summary["connect_tools_listed_share"] is not None:
        e["connect_tools_listed_share"] = summary["connect_tools_listed_share"]
    if frozen and (cfg.get("agent") or {}).get("digest"):
        e["frozen_digest"] = cfg["agent"]["digest"]
    if v := harness_version(arm, rs, cfg, frozen):
        e["harness_version"] = v
    return e


def harness_version(arm: str, rs: list[dict], cfg: dict, frozen: str | None) -> str | None:
    """Every entry's harness version, whoever made it: the one its trials ran with (provenance), else the run's
    preflight record, else a frozen agent's. Trials merged from shards on different versions list them all. Older
    runs without any have none."""
    seen = {(r.get("provenance") or {}).get("harness_version") for r in rs} - {None}
    if seen:
        return ", ".join(sorted(seen))
    if v := (cfg.get("harness_versions") or {}).get(arm):
        return v
    return (cfg.get("agent") or {}).get("version") if frozen else None


def pair_order(by_arm: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """Every pair of entries, as (a, b) with a the higher pass rate; the run report (onchainbench.verdict) compares
    the same pairs in the same order."""
    order = sorted(by_arm, key=lambda a: -rate(by_arm[a])["mean"])
    return list(combinations(order, 2))


def pairs(by_arm: dict[str, list[dict]]) -> list[dict]:
    """Every pair of ranked entries (pair_order): diff = mean(pass_b - pass_a) over shared tasks, family bootstrap
    (stats); the tasks only one of them ran are counted beside it."""
    out = []
    for a, b in pair_order(by_arm):
        ra, rb = by_arm[a], by_arm[b]
        p = stats.paired_bootstrap(ra, rb)
        if not p.get("tasks"):
            continue
        blocks = {}
        for blk in sorted({r["block"] for r in ra + rb}):
            q = stats.paired_bootstrap([r for r in ra if r["block"] == blk], [r for r in rb if r["block"] == blk])
            if q.get("tasks"):
                blocks[blk] = {k: q[k] for k in ("diff", "ci95", "ci_method", "tasks", "families")}
        out.append(
            {
                "a": a,
                "b": b,
                "diff": p["diff"],
                "ci95": p["ci95"],
                "ci_method": p["ci_method"],
                "tasks": p["tasks"],  # paired on the tasks both ran; the others are counted
                "families": p["families"],  # fewer than two: the interval can't show a difference
                "only_a": p["only_a"],
                "only_b": p["only_b"],
                "by_block": blocks,
            }
        )
    return out


def is_twin(r: dict) -> bool:
    """A safety twin's trial: its task has an attack (the runner records the task's `attack` on every twin trial)."""
    return bool(r.get("attack"))


def task_counts(rs: list[dict]) -> dict:
    """The run's size, as counts only (never ids), so a run without per-trial rows still says it. Over every task the
    run attempted: each task with any entrant trial record, scored or not (lost to infra, or one an entry can't run),
    so the counts match the task set even when all of a task's attempts were lost to infra; the scripted validation
    arms aren't entrants (build drops them). `no_signal` stays on scored tasks. `tasks` counts each best-of-N variant,
    `base_tasks` folds the variants into their base task (`variant_of`); `twins` counts the safety twins per base
    task, `twin_variants` each twin phrasing."""
    twins = [r for r in rs if is_twin(r)]

    def base(r):
        return r.get("variant_of") or r["task"]

    return {
        "tasks": len({r["task"] for r in rs}),
        "base_tasks": len({base(r) for r in rs}),
        "twins": len({base(r) for r in twins}),
        "twin_variants": len({r["task"] for r in twins}),
    }


def trial_safety(r: dict) -> str:
    """The trial row's stage (stats.attack_stage): the entry's `safety.funnel` counts these rows."""
    return stats.attack_stage(r)


def trial_rows(rs: list[dict], split: str, salt: str | None) -> list[dict]:
    rows = []
    for r in rs:
        if not stats.is_scored(r):
            continue
        row = {
            "task": opaque_id(r["task"], salt) if split == "heldout" else r["task"],
            "entry": r["arm"],
            "block": r["block"],
            "chain": r["chain"],
            "protocols": r.get("protocols") or [],
            "attempt": r["attempt"],
            "passed": bool(r["passed"]),
            "cost_usd": round(r.get("cost_usd") or 0.0, 6),
            "seconds": r.get("seconds") or 0.0,
            "safety": trial_safety(r),
            "twin": is_twin(r),
        }
        if r.get("steps_total"):
            row["partial"] = round(r["steps_done"] / r["steps_total"], 4)
        rows.append(row)
    return rows


def local_file(path: str, run_dir: Path | None) -> Path:
    """A trial's file in the run being published: the path the runner recorded is absolute on the machine that ran it
    (a CI runner), so a run downloaded and published elsewhere finds the file under its own sessions/ by the part
    after "sessions/". The recorded path itself where that isn't there (or without a run dir)."""
    p = path.replace("\\", "/")
    if run_dir is not None and "/sessions/" in p:
        here = run_dir / "sessions" / p.split("/sessions/", 1)[1]
        if here.is_file():
            return here
    return Path(path)


def trajectories(rs: list[dict], run_dir: Path | None = None) -> list[dict]:
    """One row per open-track trial, in the shared transcript shape: the transcript the harness recorded in-process
    (Session.transcript(), written by the runner), read from the run's own sessions/. An arm's own session file is
    never published: it is the agent's record, not the harness's, so a trial with only that has no published
    trajectory. A transcript the trial names but the run doesn't have is refused, never published empty."""
    out, missing = [], []
    for r in rs:
        if not r.get("transcript"):
            continue
        path = local_file(r["transcript"], run_dir)
        if not path.is_file():
            missing.append(f"{r['arm']} {r['task']} #{r['attempt']}")
            continue
        row = {"entry": r["arm"], "task": r["task"], "attempt": r["attempt"], "passed": bool(r.get("passed"))}
        row["events"] = transcript.read(path)
        out.append(row)
    if missing:
        raise PublishError(
            f"{len(missing)} trials name a transcript the run doesn't have (e.g. {missing[0]}): publish the run "
            "directory with its sessions/ (the merged artifact), so no trajectory is published empty"
        )
    return out


def bench_digest(rs: list[dict], cfg: dict) -> str | None:
    """The run's one bench_digest (freeze.bench_digest: the code and task data it ran on), read from its trials'
    provenance (a record from before the stamps: its config.json's); None when no trial records one. Trials that
    disagree, or some with a digest and some without, are refused: a published run is one version of the bench."""
    seen = {(r["provenance"] if isinstance(r.get("provenance"), dict) else cfg).get("bench_digest") for r in rs}
    if len(seen) > 1:
        raise PublishError(
            f"the trials ran on different bench code (bench_digest {sorted(map(str, seen))}): a run is published "
            "from one version of the bench (merge and --fill refuse to combine these)"
        )
    return next(iter(seen))


def build(run_dir: Path, release_url: str, **opt) -> tuple[dict, list[dict] | None]:
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg.get("split") not in SPLITS:
        raise PublishError(f"split {cfg.get('split')!r}: publish an open or a test run, never both mixed")
    split = SPLITS[cfg["split"]]
    recs = [json.loads(x) for x in (run_dir / "trials.jsonl").read_text().splitlines() if x.strip()]
    recs = [r for r in recs if A.parse_arm(r["arm"])[0] not in A.SCRIPTED_ARMS]
    if not recs:
        raise PublishError("no entrant trials: the scripted validation arms are not entries")
    if cfg.get("no_judges") or any(r.get("judges_off") for r in recs):
        raise PublishError(
            "the run was made with --no-judges, a wiring run: its judged metrics weren't measured, so it isn't a "
            "ranked result (run it again with the judges)"
        )
    if n := sum(1 for r in recs if r.get("interrupted")):
        raise PublishError(
            f"{n} trials were interrupted (the run was stopped): finish it first with "
            f"python -m onchainbench.run <its flags> --fill {cfg.get('run') or run_dir.name}"
        )
    by_arm = defaultdict(list)
    for r in recs:
        by_arm[r["arm"]].append(r)
    # held-out ids go out opaque: per-trial rows and the tasks an entry can't run need the salt
    need = not opt.get("no_trials") or any(r.get("unsupported") for r in recs)
    salt = heldout_salt() if split == "heldout" and need else None
    doc = {
        "schema_version": 1,
        "bench_version": opt.get("bench_version") or version("onchainbench"),
        "run_id": cfg.get("run") or run_dir.name,
        "split": split,
        "date": cfg.get("date") or opt.get("date"),
        "release_url": release_url,
        **{k: v for k, v in (("prereg_commit", opt.get("prereg_commit")), ("hf_row_url", opt.get("hf_row_url"))) if v},
        **{k: cfg[k] for k in ("manifest_sha256", "seed") if cfg.get(k) is not None},
        **({"bench_digest": d} if (d := bench_digest(recs, cfg)) else {}),
        "entries": [entry(a, rs, cfg, bool(opt.get("verified")), salt) for a, rs in by_arm.items()],
        "pairs": pairs({a: [r for r in rs if stats.is_scored(r)] for a, rs in by_arm.items()}),
    }
    # the tasks every entry passed on every attempt: named on open runs; on held-out runs opaque, and only counted
    # when the run publishes no per-trial rows (a list of ids would then be the only per-task data in the file)
    flat = stats.no_signal(by_arm)
    ids = flat.pop("tasks")
    if split == "open" or (salt and not opt.get("no_trials")):
        flat["tasks"] = sorted(opaque_id(t, salt) for t in ids) if split == "heldout" else ids
    doc["no_signal"] = flat
    doc["task_counts"] = task_counts(recs)  # on every run: the only task counts a run without trial rows has
    if not opt.get("no_trials"):
        doc["trials"] = trial_rows(recs, split, salt)
    return doc, (trajectories(recs, run_dir) if split == "open" else None)


def validate(doc: dict) -> None:
    import jsonschema

    errors = sorted(jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).iter_errors(doc), key=str)
    if errors:
        raise PublishError("the run JSON doesn't match the site's schema: " + "; ".join(e.message for e in errors[:5]))


def guard(text: str, what: str) -> None:
    from onchainbench.tasks import heldout_dir

    found = leaks.find(text, leaks.heldout_markers() if heldout_dir() else {})
    if found:
        raise PublishError(f"{what} would give the held-out set away: {'; '.join(found[:5])}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_sums(run_dir: Path, written: dict[str, bytes]) -> dict[str, str]:
    """path -> SHA-256 of ARCHIVE and inspect/*.json (those present) and of the files publish is about to write
    beside them, sorted by path. sessions/ and the raw transcripts aren't covered yet."""
    files = [run_dir / n for n in ARCHIVE] + sorted((run_dir / "inspect").glob("*.json"))
    sums = {p.relative_to(run_dir).as_posix(): sha256(p.read_bytes()) for p in files if p.is_file()}
    return dict(sorted((sums | {n: sha256(b) for n, b in written.items()}).items()))


def sums_file(sums: dict[str, str]) -> str:
    """The `sha256sum` format, so `sha256sum -c` reads it as well as onchainbench.verify."""
    return "".join(f"{h}  {n}\n" for n, h in sorted(sums.items()))


def write(run_dir: Path, doc: dict, traj: list[dict] | None) -> list[Path]:
    """run.json, trajectories.jsonl on open runs, and SHA256SUMS over them and archive_sums' files. run.json carries
    the sums of its siblings (`artifacts_sha256`), so the public run JSON commits to a held-out run's private
    archive too; SHA256SUMS adds run.json's own sum."""
    written = {}
    if traj is not None:
        lines = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traj)
        guard(lines, "trajectories.jsonl")
        written["trajectories.jsonl"] = lines.encode()
    doc["artifacts_sha256"] = archive_sums(run_dir, written)
    validate(doc)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    guard(text, "run.json")
    written["run.json"] = text.encode()
    sums = sums_file({**doc["artifacts_sha256"], "run.json": sha256(written["run.json"])})
    guard(sums, SUMS)
    written[SUMS] = sums.encode()
    out = [run_dir / n for n in ("run.json", "trajectories.jsonl", SUMS) if n in written]
    for p in out:
        p.write_bytes(written[p.name])
    return out


def upload_release(tag: str, files: list[Path], repo: str = "valory-xyz/onchainbench") -> None:
    subprocess.run(["gh", "release", "upload", tag, *map(str, files), "--clobber", "-R", repo], check=True)


def upload_hf(repo: str, path: Path, dest: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=dest, repo_id=repo, repo_type="dataset")


def _arg(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def main(argv: list[str]) -> int:
    if any(x in ("-h", "--help") for x in argv):
        print(__doc__.strip())
        return 0
    if not argv or argv[0].startswith("-") or "--release-url" not in argv:
        print(__doc__.strip())
        return 2
    run_dir = Path(argv[0]) if Path(argv[0]).is_dir() else ROOT / "results" / argv[0]
    opt = {k.replace("-", "_"): _arg(argv, f"--{k}") for k in ("bench-version", "prereg-commit", "hf-row-url", "date")}
    try:
        doc, traj = build(
            run_dir,
            _arg(argv, "--release-url"),
            no_trials="--no-trials" in argv,
            verified="--verified" in argv,
            **opt,
        )
        files = write(run_dir, doc, traj)
    except PublishError as e:
        print(f"refusing to publish: {e}")
        return 1
    print("wrote " + ", ".join(str(f) for f in files))
    if tag := _arg(argv, "--upload-release"):
        upload_release(tag, files)
        print(f"attached to release {tag}")
    # on Hugging Face under runs/<id>/ with their own names, so a download verifies as it is
    if repo := _arg(argv, "--hf-results-repo"):
        for name in ("run.json", SUMS):
            upload_hf(repo, run_dir / name, f"runs/{doc['run_id']}/{name}")
        print(f"uploaded run.json and {SUMS} to {repo}")
    if (repo := _arg(argv, "--hf-trajectories-repo")) and traj is not None:
        upload_hf(repo, run_dir / "trajectories.jsonl", f"runs/{doc['run_id']}/trajectories.jsonl")
        print(f"uploaded trajectories to {repo}")
    print(f"verify a download: python -m onchainbench.verify <dir>  (or sha256sum -c --ignore-missing {SUMS})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
