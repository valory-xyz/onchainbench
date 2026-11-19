"""Inspect-compatible logs: our runner, logs Inspect's viewer can open.

One EvalLog JSON per arm (`inspect view` reads `<run>/inspect/*.json`), following Inspect's log schema
(version 2): `eval` (task, model, dataset, config), `plan`, `results` (scores with metrics), `stats` and
`samples` (id, epoch, input, target, output, scores, metadata). Fields we don't have are omitted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from onchainbench.stats import is_scored


def _sample(r: dict) -> dict:
    return {
        "id": f"{r['chain']}/{r['task']}",
        "epoch": int(r.get("attempt", 0)) + 1,
        "input": r.get("task"),
        "target": "outcome checked on the fork",
        "messages": [{"role": "assistant", "content": a} for a in r.get("answers", [])],
        "output": {"completion": r.get("answer", "")},
        "scores": {"outcome": {"value": "C" if r.get("passed") else "I", "explanation": r.get("reason", "")}},
        "metadata": {
            k: r.get(k)
            for k in ("block", "family", "split", "params", "metrics", "cost_usd", "seconds", "infra", "attack")
        },
        "model_usage": {"input_tokens": r.get("prompt_tokens") or 0, "output_tokens": r.get("completion_tokens") or 0},
    }


def write_inspect_log(out: Path, records: list[dict], args) -> list[Path]:
    d = out / "inspect"
    d.mkdir(exist_ok=True)
    paths = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for arm in dict.fromkeys(r["arm"] for r in records):
        rs = [r for r in records if r["arm"] == arm]
        scored = [r for r in rs if is_scored(r)]
        acc = sum(r["passed"] for r in scored) / max(1, len(scored))
        log = {
            "version": 2,
            "status": "success",
            "eval": {
                "run_id": out.name,
                "task": "onchainbench",
                "task_version": 2,
                "created": now,
                "model": arm,
                "dataset": {
                    "name": f"onchainbench-{getattr(args, 'split', 'open')}",
                    "samples": len({(r["chain"], r["task"]) for r in rs}),
                },
                "config": {"epochs": getattr(args, "attempts", 1)},
            },
            "plan": {"name": "onchainbench", "steps": [{"solver": arm.partition(":")[0]}]},
            "results": {
                "total_samples": len(rs),
                "completed_samples": len(scored),
                "scores": [
                    {
                        "name": "outcome",
                        "scorer": "fork_outcome",
                        "metrics": {"accuracy": {"name": "accuracy", "value": round(acc, 4)}},
                    }
                ],
            },
            "stats": {"started_at": now, "completed_at": now},
            "samples": [_sample(r) for r in rs],
        }
        p = d / f"{arm.replace('/', '_').replace(':', '_').replace('|', '_').replace('?', '_')}.json"
        p.write_text(json.dumps(log, default=str))
        paths.append(p)
    return paths
