"""Task cards for the open track: what each public open task asks, checks and attacks, for the site's task pages and
the Hugging Face `onchainbench-open` dataset.

    python -m onchainbench.cards [--out task-cards.jsonl]

One JSON object per open task: id, block, family, chain, protocols, the instruction template and an example drawn
with seed 0, the check (its function, where it lives in this repo, its source, and its verdict labels where it
uses them), and for safety twins the attack class,
the base task and the attacker model's data fields. An example is null where the instruction names something the
task deploys at setup. Held-out tasks are refused, never exported, and the output is
leak-checked before it is written (onchainbench.leaks).
"""

from __future__ import annotations

import ast
import inspect
import json
import random
import sys
import textwrap
from pathlib import Path

from onchainbench import leaks
from onchainbench.arms import STEP_LIMIT
from onchainbench.tasks.base import Task


class CardError(Exception):
    """A card that must not be exported (a held-out task, or text that gives the held-out set away)."""


def checks_of(fn) -> list[str]:
    """The verdict labels in a check function: each ("label", condition) pair in its source, in order."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return []
    labels = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and isinstance(node.elts[0].value, str)
            and node.elts[0].value not in labels
        ):
            labels.append(node.elts[0].value)
    return labels


def check_of(fn) -> dict:
    """The check function: its name, its file and line in this repo (for a source link), its source and labels."""
    try:
        lines, start = inspect.getsourcelines(fn)
        path = Path(inspect.getsourcefile(fn)).resolve().relative_to(Path(__file__).resolve().parents[1])
    except (OSError, TypeError, ValueError):  # a builtin, or code outside this repo
        return {"function": getattr(fn, "__qualname__", repr(fn)), "labels": []}
    return {
        "function": fn.__qualname__,
        "path": f"{path.as_posix()}#L{start}",
        "source": textwrap.dedent("".join(lines)),
        "labels": checks_of(fn),
    }


def card(task: Task) -> dict:
    if task.split != "open":
        raise CardError(f"{task.id} is on the {task.split} split: only open-set tasks get cards")
    p = task.params(random.Random(0))
    try:
        example = task.text(p)
    except KeyError:  # the instruction names something setup deploys (an address on the fork): no offline example
        example = None
    c = {
        "id": task.id,
        "block": task.block,
        "family": task.family,
        "chain": task.chain,
        "protocols": list(task.protocols),
        "instruction_template": task.template,
        "instruction_example": example,
        "check": check_of(task.check),
        "step_limit": STEP_LIMIT,  # what every arm is given on every task (arms.STEP_LIMIT)
        "turns": len(task.turns) + 1,
        "simulated_user": task.persona is not None,
        "expect_refusal": task.expect_refusal,
        "lookalike": task.lookalike,
    }
    if task.attack or task.twin_of:
        threat = {k: v for k, v in (task.threat or {}).items() if isinstance(v, (str, int, float, bool, list))}
        c |= {"attack": task.attack, "twin_of": task.twin_of, "threat": threat}
    return c


def cards(tasks: list[Task]) -> list[dict]:
    seen, out = set(), []
    for t in tasks:
        key = t.variant_of or t.id
        if key not in seen:  # best-of-N variants share one card
            seen.add(key)
            out.append(card(t))
    return out


def export(tasks: list[Task], dest: Path) -> int:
    from onchainbench.tasks import heldout_dir

    lines = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cards(tasks))
    found = leaks.find(lines, leaks.heldout_markers() if heldout_dir() else {})
    if found:
        raise CardError("the cards would give the held-out set away: " + "; ".join(found[:5]))
    dest.write_text(lines)
    return lines.count("\n")


def main(argv: list[str]) -> int:
    from onchainbench.tasks import all_tasks

    dest = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("task-cards.jsonl")
    try:
        n = export([t for t in all_tasks() if t.split == "open"], dest)
    except CardError as e:
        print(f"refusing to export: {e}")
        return 1
    print(f"wrote {n} task cards to {dest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
