"""OnchainBench tasks: the open set (blocks A-G, written with the harness in view, tuning allowed; public) and the
held-out test tasks (written separately from the spec, frozen), which live in the private onchainbench-heldout
repo. The held-out set is found as the installed `onchainbench_heldout` package, or at the directory
ONCHAINBENCH_HELDOUT names; without it the bench runs the open set only."""

from __future__ import annotations

import copy
import importlib
import json
import os
import pkgutil
import sys
from pathlib import Path

from onchainbench.tasks.base import Task

OPEN_MODULES = [
    "block_a",
    "block_b",
    "block_c",
    "block_d",
    "block_e",
    "block_f",
    "block_f_payloads",
    "block_g",
    "block_chains",
    "block_ops",
    "block_a_admin",
]
CONTRIB_PREFIX = (
    "contrib_"  # contributed open-track tasks: one module per protocol, contrib_<protocol>.py (CONTRIBUTING.md)
)


def contrib_modules(d: Path | None = None) -> list[str]:
    """The contributed task modules next to the open blocks, in name order."""
    d = d or Path(__file__).resolve().parent
    return sorted(i.name for i in pkgutil.iter_modules([str(d)]) if i.name.startswith(CONTRIB_PREFIX))


HELDOUT_ENV = "ONCHAINBENCH_HELDOUT"  # a checkout's onchainbench_heldout/ dir, when the package isn't installed
HELDOUT_PKG = "onchainbench_heldout"


def _load(mod: str) -> list[Task]:
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name == mod:
            return []
        raise
    return list(getattr(m, "TASKS", []))


def heldout_dir() -> Path | None:
    """The held-out package's directory, or None when this checkout doesn't have the held-out set."""
    env = os.environ.get(HELDOUT_ENV)
    if env:
        d = Path(env).resolve()
        if str(d.parent) not in sys.path:
            sys.path.insert(0, str(d.parent))
    try:
        pkg = importlib.import_module(HELDOUT_PKG)
    except ModuleNotFoundError as e:
        if e.name != HELDOUT_PKG:
            raise
        return None
    return Path(next(iter(pkg.__path__)))


def tag(tasks: list[Task], heldout: Path | None) -> None:
    """Attach each task's protocols from PROTOCOLS.json (open here, held-out with its tasks). Every task needs an entry,
    so a new task can't reach the site's per-protocol results untagged."""
    table: dict = {}
    for d in (Path(__file__).resolve().parent, heldout):
        if d is not None and (d / "PROTOCOLS.json").is_file():
            table |= json.loads((d / "PROTOCOLS.json").read_text())
    missing = [t.id for t in tasks if t.id not in table]
    if missing:
        raise ValueError(f"tasks without protocols in PROTOCOLS.json: {missing}")
    for t in tasks:
        t.protocols = tuple(table[t.id])


def expand_variants(t: Task) -> list[Task]:
    """Best-of-N: a task with variants=N becomes N tasks `<id>~v<i>` in one family; params get `variant`."""
    if t.variants <= 1:
        return [t]
    out = []
    for i in range(t.variants):
        v = copy.copy(t)
        v.id, v.variant, v.variant_of, v.variants, v.variant_count = f"{t.id}~v{i}", i, t.id, 1, t.variants
        v.params = lambda rng, i=i, base=t.params: {**base(rng), "variant": i}
        out.append(v)
    return out


def all_tasks() -> list[Task]:
    out = []
    for m in OPEN_MODULES + contrib_modules():
        out += _load(f"onchainbench.tasks.{m}")
    if (d := heldout_dir()) is not None:
        for info in sorted(pkgutil.iter_modules([str(d)]), key=lambda i: i.name):
            for t in _load(f"{HELDOUT_PKG}.{info.name}"):
                t.split = "test"
                out.append(t)
    tag(out, heldout_dir())
    out = [v for t in out for v in expand_variants(t)]
    ids = [t.id for t in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate task ids: {sorted(dup)}")
    return out
