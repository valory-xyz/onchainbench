"""Freeze the held-out test tasks: record each test file's SHA-256 in the held-out package's MANIFEST.json.

    python -m onchainbench.freeze           # write the manifest (after validation)
    python -m onchainbench.freeze --check   # exit 1 if any test file changed since
    python -m onchainbench.freeze --commit  # the manifest's SHA-256, to publish before a run (the commitment)
    python -m onchainbench.freeze --agent 0.3.1 --reason "run 1"   # freeze the agent (METHODOLOGY "Run 1")
    python -m onchainbench.freeze --agent-check                    # exit 1 if the agent changed since
    python -m onchainbench.freeze --bench-digest   # this checkout's bench_digest, to verify a run's code

The runner refuses `--split test` when the check fails, so a test task can't be edited quietly after
results were seen; it also refuses test runs of the agent's arms when the agent changed since its freeze, and records
the frozen version and digest in each run's config.json, so the report can name the version tested.

Every run records the bench's code and task data as `bench_digest` (each trial's provenance, config.json and the
published run.json), not a git commit, so it can be checked against any copy of the code: check out the tag named in
the run's pre-registration (or install the bench from that checkout), run `--bench-digest`, and compare. It covers the
package's code, ABIs, contracts and task files; docs and tests don't change it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TEST: Path | None = None  # the held-out package dir; None: wherever the bench finds the held-out set
MANIFEST: Path | None = None
NO_HELDOUT = "no held-out set here: install onchainbench-heldout or set ONCHAINBENCH_HELDOUT"
AGENT: Path | None = None  # the agent package under test; None: the one the `ocb-agent` CLI runs
AGENT_MANIFEST: Path | None = None  # the agent freeze record, kept with the held-out set (run records are private)


def test_dir() -> Path | None:
    if TEST is not None:
        return TEST
    from onchainbench.tasks import heldout_dir

    return heldout_dir()


def manifest() -> Path | None:
    d = test_dir()
    return MANIFEST or (d / "MANIFEST.json" if d else None)


def agent_manifest() -> Path:
    if AGENT_MANIFEST is not None:
        return AGENT_MANIFEST
    d = test_dir()
    if d is None:
        raise FileNotFoundError(NO_HELDOUT)
    return d.parent / "AGENT-FROZEN.json"


def agent_dir() -> Path:
    """The installed olas-ocb-agent package: the one its CLI (`ocb-agent`, as the arm finds it) runs."""
    if AGENT is not None:
        return AGENT
    from onchainbench import arms as A

    if (why := A.harness_problem("olas-ocb-agent")) is not None:  # it reads the package, or says why it can't
        raise FileNotFoundError(f"olas-ocb-agent isn't installed: nothing to freeze or check ({why})")
    try:
        return Path(A.cli_package("olas-ocb-agent")["dir"])
    except RuntimeError as e:  # the CLI changed since the preflight read it
        raise FileNotFoundError(f"olas-ocb-agent's package can't be read: {e}") from e


def agent_files(root: Path | None = None) -> dict:
    """The agent's code and bundled config (the prompt, policy, skill trust list): everything its arm runs, in every
    subpackage (its `interim/` code included), keyed by the path inside the package. `root`: another package's."""
    root = root or agent_dir()
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted([*root.rglob("*.py"), *root.rglob("*.json")])
        if "__pycache__" not in p.parts
    }


BENCH_SUFFIXES = (".py", ".json", ".jsonl", ".sol", ".toml")  # the bench's code and task data, not its docs
# the compiler's working files, not what runs: the deployed artifacts (contracts/out/<X>.sol/<X>.json, the bytecode
# contracts/__init__.py deploys) are covered, their build-info and forge's cache aren't
BENCH_BUILD = {("contracts", "cache"), ("contracts", "out", "build-info")}


def bench_covers(rel: Path) -> bool:
    """A file (its path inside the package) that `bench_digest` covers."""
    if "__pycache__" in rel.parts or any(rel.parts[: len(b)] == b for b in BENCH_BUILD):
        return False
    return rel.suffix in BENCH_SUFFIXES


def bench_digest(root: Path | None = None) -> str:
    """The digest of the bench's code and task data (the onchainbench package's code, ABIs, contracts, their deployed
    build and task files; not its docs, tests or README): stamped on each trial, so a docs-only commit doesn't split a
    run."""
    root = root or Path(__file__).resolve().parent
    files = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and bench_covers(rel):
            files[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return agent_digest(files)


def package_digest(name: str) -> str | None:
    """The digest of the frozen agent's installed code (olas_ocb_agent, the package its CLI runs), as its freeze
    computes it, so it equals the freeze record's while unchanged; None for another package or when it isn't
    installed. Stamped on each trial of the agent's arms."""
    if name != "olas_ocb_agent":
        return None
    try:
        return agent_digest()
    except FileNotFoundError:
        return None


def agent_digest(files: dict | None = None) -> str:
    files = agent_files() if files is None else files
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def agent_check() -> list[str]:
    if not agent_manifest().is_file():
        return ["no AGENT-FROZEN.json: olas-ocb-agent is not frozen"]
    want, have = json.loads(agent_manifest().read_text())["files"], agent_files()
    return [
        f"olas_ocb_agent/{n}: changed since freezing"
        for n in sorted(set(want) | set(have))
        if want.get(n) != have.get(n)
    ]


def agent_frozen() -> dict:
    """The frozen version and digest (for a run's config.json), or {} when olas-ocb-agent isn't frozen or there is no
    held-out set here (a public checkout); "unchanged" is None when the agent isn't installed."""
    try:
        path = agent_manifest()
    except FileNotFoundError:
        return {}
    if not path.is_file():
        return {}
    m = json.loads(path.read_text())
    try:
        unchanged = not agent_check()
    except FileNotFoundError:
        unchanged = None
    return {"version": m["version"], "digest": m["digest"], "unchanged": unchanged}


def freeze_agent(version: str, reason: str | None = None) -> dict:
    old = json.loads(agent_manifest().read_text()) if agent_manifest().is_file() else {}
    history = list(old.get("history", []))
    if old.get("version"):
        history.append({"version": old["version"], "digest": old["digest"], "replaced_because": reason})
    files = agent_files()
    m = {"version": version, "digest": agent_digest(files), "reason": reason, "files": files}
    m |= {"history": history} if history else {}
    agent_manifest().write_text(json.dumps(m, indent=1) + "\n")
    return m


# What the held-out fingerprint covers. Version 2: every file of the package, in every subfolder (`__init__.py`,
# PROTOCOLS.json and data files included: the task loader imports the package and its modules and reads the protocol
# table), and the package's setup files next to it. Version 1 covered only the top-level task modules.
FINGERPRINT = 2
SETUP_FILES = ("pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in")  # beside the package dir, as "../<name>"


def _covered(rel: Path) -> bool:
    """A file the fingerprint covers: not the manifest itself, bytecode, or hidden files (.DS_Store and the like)."""
    if rel.as_posix() == "MANIFEST.json" or rel.suffix == ".pyc":
        return False
    return not any(part == "__pycache__" or part.startswith(".") for part in rel.parts)


def digest() -> dict:
    """The held-out set's files, keyed by their path in the package (setup files as "../<name>"), with SHA-256s."""
    root = test_dir()
    files = {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and _covered(p.relative_to(root))
    }
    files |= {f"../{n}": root.parent / n for n in SETUP_FILES if (root.parent / n).is_file()}
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sorted(files.items())}


REFREEZE = (
    "MANIFEST.json was frozen with fingerprint version {have}, which left out files the task loader reads "
    "(__init__.py, PROTOCOLS.json, subfolders, the package's setup files); version {want} covers them. Re-validate "
    "the held-out set and re-freeze: python -m onchainbench.freeze --reason '<why>'"
)


def check() -> list[str]:
    if test_dir() is None:
        return [NO_HELDOUT]
    if not manifest().is_file():
        return ["no MANIFEST.json: the test set is not frozen"]
    m = json.loads(manifest().read_text())
    if (have := m.get("fingerprint", 1)) != FINGERPRINT:
        return [REFREEZE.format(have=have, want=FINGERPRINT)]
    want, now = m["files"], digest()
    return [f"{n}: changed since freezing" for n in sorted(set(want) | set(now)) if want.get(n) != now.get(n)]


def freeze(reason: str | None = None) -> dict:
    """Record every test file's hash. A re-freeze keeps the history and says why (and which files it added)."""
    if test_dir() is None:
        raise FileNotFoundError(NO_HELDOUT)
    old = json.loads(manifest().read_text()) if manifest().is_file() else {}
    history = list(old.get("history", []))
    if old.get("files") and reason:
        history.append({"reason": reason, "added": sorted(set(digest()) - set(old["files"]))})
    m = {"fingerprint": FINGERPRINT, "files": digest(), **({"history": history} if history else {})}
    manifest().write_text(json.dumps(m, indent=1) + "\n")
    return m


def commitment() -> str:
    """The SHA-256 of the held-out manifest: published before a run, it shows later that the tasks weren't changed
    after results were seen (the site's pre-registration page)."""
    if test_dir() is None or not manifest().is_file():
        raise FileNotFoundError(NO_HELDOUT + " (or it isn't frozen)")
    return hashlib.sha256(manifest().read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    reason = argv[argv.index("--reason") + 1] if "--reason" in argv else None
    if argv and argv[0] == "--agent-check":
        problems = agent_check()
        print("\n".join(problems) or "frozen olas-ocb-agent unchanged")
        return 1 if problems else 0
    if argv and argv[0] == "--agent":
        m = freeze_agent(argv[1], reason)
        print(f"olas-ocb-agent {m['version']} frozen: {m['digest'][:16]}")
        return 0
    if argv and argv[0] == "--commit":
        print(commitment())
        return 0
    if argv and argv[0] == "--bench-digest":
        print(bench_digest())
        return 0
    if argv and argv[0] == "--check":
        problems = check()
        print("\n".join(problems) or "frozen test set unchanged")
        return 1 if problems else 0
    print(json.dumps(freeze(reason), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
