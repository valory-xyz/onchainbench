"""Check a downloaded run against its published SHA-256 sums.

    python -m onchainbench.verify <dir>

<dir> holds the run's SHA256SUMS (written by onchainbench.publish) and whichever of its files you downloaded, either
in place (results/<run>/) or flat, as a release download leaves them (inspect/x.json as x.json). Every file listed is
checked; one that isn't there is reported as not downloaded (a held-out run's archive stays private until its tasks
retire, and the public run.json commits to it meanwhile). A name that isn't a plain relative path inside <dir>
(`..`, absolute, a drive or a backslash) fails the check. When run.json is present SHA256SUMS must list it, and its
`artifacts_sha256` must agree with SHA256SUMS. Exit 0 when every file present matches and at least one was
checked, else 1.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SUMS = "SHA256SUMS"


def parse(text: str) -> dict[str, str]:
    """path -> sum from a `sha256sum` file (text or binary mode lines)."""
    out = {}
    for line in text.splitlines():
        if line.strip():
            h, _, name = line.partition(" ")
            out[name[1:] if name[:1] in " *" else name] = h.lower()
    return out


def safe(name: str) -> bool:
    """A relative path below the directory: no `..`, no absolute or drive path, no backslash."""
    parts = name.split("/")
    return bool(name) and "\\" not in name and ":" not in name and not name.startswith("/") and ".." not in parts


def find(d: Path, name: str) -> Path | None:
    root = d.resolve()
    for p in (d / name, d / Path(name).name):
        if p.is_file() and p.resolve().is_relative_to(root):  # a symlink can't lead out either
            return p
    return None


def committed(run: Path) -> dict | None:
    """run.json's `artifacts_sha256`, or None when run.json isn't a JSON object."""
    try:
        doc = json.loads(run.read_text())
    except (ValueError, UnicodeDecodeError):
        return None
    return doc.get("artifacts_sha256") if isinstance(doc, dict) else None


def check(d: Path) -> tuple[list[str], bool]:
    """One line per listed file (OK, FAILED, not downloaded or refused), and whether the download verifies."""
    sums = parse((d / SUMS).read_text())
    lines, ok, checked = [], True, 0
    for name, want in sums.items():
        if not safe(name):
            lines.append(f"{name}: refused, not a path inside the run")
            ok = False
            continue
        p = find(d, name)
        if p is None:
            lines.append(f"{name}: not downloaded")
            continue
        checked += 1
        good = hashlib.sha256(p.read_bytes()).hexdigest() == want
        ok &= good
        lines.append(f"{name}: {'OK' if good else 'FAILED'}")
    if (run := find(d, "run.json")) is not None:
        if "run.json" not in sums:
            lines.append(f"run.json: FAILED, not listed in {SUMS}")
            ok = False
        elif (c := committed(run)) is None:
            lines.append("run.json: FAILED, not a JSON object")
            ok = False
        elif c != {n: h for n, h in sums.items() if n != "run.json"}:
            lines.append(f"run.json: FAILED, its artifacts_sha256 doesn't match {SUMS}")
            ok = False
    return lines, ok and checked > 0


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0].startswith("-"):
        print(__doc__)
        return 2
    d = Path(argv[0])
    if not (d / SUMS).is_file():
        print(f"no {SUMS} in {d}")
        return 1
    lines, ok = check(d)
    print("\n".join(lines))
    print("verified" if ok else "NOT verified")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
