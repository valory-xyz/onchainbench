"""Fail if the public bench tree mentions the held-out set: any held-out task id, or a distinctive fragment of a
held-out instruction (held-out tasks are never published while in use), and any held-out id shape.

    python scripts/leak_check.py [--heldout DIR] [ROOT]   # ROOT defaults to this repo; DIR to the installed package

Without the held-out set (a public checkout) there are no ids or instructions to look for: only the id shapes are
checked, and the result says so.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from onchainbench.leaks import find, heldout_markers
from onchainbench.tasks import heldout_dir

SKIP_DIRS = {".git", ".venv", "vendor", "results", "__pycache__", "node_modules"}


def tracked_files(root: Path) -> list[Path]:
    r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=False)
    names = r.stdout.splitlines() if r.returncode == 0 else [str(p.relative_to(root)) for p in root.rglob("*")]
    return [root / n for n in names if not SKIP_DIRS & set(Path(n).parts) and (root / n).is_file()]


def scan(root: Path, markers: dict[str, str]) -> list[str]:
    found = []
    for p in tracked_files(root):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        found += [f"{p.relative_to(root)}: {x}" for x in find(text, markers)]
    return sorted(found)


def main(argv: list[str]) -> int:
    heldout = argv[argv.index("--heldout") + 1] if "--heldout" in argv else None
    rest = [a for i, a in enumerate(argv) if a != "--heldout" and (i == 0 or argv[i - 1] != "--heldout")]
    root = Path(rest[0] if rest else Path(__file__).resolve().parents[1]).resolve()
    markers = heldout_markers(heldout)
    found = scan(root, markers)
    if found:
        print("\n".join(found))
    elif heldout_dir() is None:
        print("no held-out set here: only held-out id shapes were checked (none found), not held-out ids or text")
    else:
        print(f"no held-out ids, instruction text or id shapes in the tree ({len(markers)} markers checked)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
