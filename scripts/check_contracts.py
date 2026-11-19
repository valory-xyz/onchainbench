"""Check that the committed contract build is what the sources compile to.

    scripts/check_contracts.py              # build onchainbench/contracts/src with forge, compare with out/
    scripts/check_contracts.py --forge PATH # another forge binary

Runs deploy the committed artifacts (onchainbench/contracts/out/<X>.sol/<X>.json), so they must match the sources.
The build runs into a temporary folder (the committed out/ is never touched) and each artifact is compared on what
deploys and what describes it: the ABI, the creation and runtime bytecode, and the compiler metadata (it holds each
source file's hash, so any edit to a source, comments included, shows). foundry.toml turns the CBOR metadata suffix
off, so the bytecode carries no metadata hash. Source maps and AST ids depend on the build's order, so they aren't
compared. Exit 1 on any difference, an artifact missing on either side, or a failed build.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CONTRACTS = Path(__file__).resolve().parent.parent / "onchainbench" / "contracts"


def fields(artifact: dict) -> dict:
    """The parts of a forge artifact that must match the sources."""
    return {
        "abi": artifact.get("abi"),
        "bytecode": (artifact.get("bytecode") or {}).get("object"),
        "deployedBytecode": (artifact.get("deployedBytecode") or {}).get("object"),
        "metadata": artifact.get("rawMetadata"),
    }


def artifacts(out: Path) -> dict[str, dict]:
    """Each artifact in a forge out/ folder (not build-info), keyed by `<X>.sol/<Name>.json`."""
    return {p.relative_to(out).as_posix(): fields(json.loads(p.read_text())) for p in sorted(out.glob("*.sol/*.json"))}


def compare(committed: dict[str, dict], built: dict[str, dict]) -> list[str]:
    """What differs between the committed artifacts and a fresh build (empty: they agree)."""
    problems = [f"{n}: committed, but the sources don't build it" for n in sorted(set(committed) - set(built))]
    problems += [f"{n}: built from the sources, but not committed" for n in sorted(set(built) - set(committed))]
    for n in sorted(set(committed) & set(built)):
        problems += [f"{n}: {k} differs from the build" for k in committed[n] if committed[n][k] != built[n][k]]
    return problems


def build(contracts: Path, dest: Path, forge: str = "forge") -> Path:
    """Build `contracts` (its foundry.toml) into `dest`; returns the out/ folder."""
    out = dest / "out"
    cmd = [forge, "build", "--root", str(contracts), "--out", str(out), "--cache-path", str(dest / "cache")]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"forge build failed:\n{res.stdout}\n{res.stderr}")
    return out


def main(argv: list[str], contracts: Path = CONTRACTS) -> int:
    forge = argv[argv.index("--forge") + 1] if "--forge" in argv else "forge"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            out = build(contracts, Path(tmp), forge)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            print(e)
            return 1
        committed = artifacts(contracts / "out")
        problems = compare(committed, artifacts(out))
    print("\n".join(problems) or f"contracts/out matches a fresh build of contracts/src ({len(committed)} artifacts)")
    if problems:
        print("rebuild: forge build --root onchainbench/contracts (with the pinned Foundry), then commit out/")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
