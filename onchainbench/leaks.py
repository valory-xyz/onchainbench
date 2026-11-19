"""What would give the held-out set away (held-out tasks are never published while in use): each
held-out task id, a distinctive fragment of each fixed instruction, and held-out id shapes. Used by
scripts/leak_check.py over the public tree and by the publish step over every file it writes."""

from __future__ import annotations

import os
import re

# the held-out id shape, beyond today's ids (renamed ones): every held-out id starts t-, t2-, t3- and so on, one
# prefix per authoring round, then its block letter. Open task ids never take this shape.
ID_PATTERN = re.compile(r"(?<![\w-])t\d*-[a-z]-[a-z][a-z0-9~-]*")


def heldout_markers(heldout: str | None = None) -> dict[str, str]:
    """marker -> what it is: each held-out task id, and a 48-character fragment of each fixed instruction."""
    if heldout:
        os.environ["ONCHAINBENCH_HELDOUT"] = heldout
    from onchainbench.tasks import all_tasks

    out = {}
    for t in all_tasks():
        if t.split != "test":
            continue
        out[(t.variant_of or t.id)] = "task id"
        if t.template is not None:  # a string instruction, or an AtStart one's string template
            words = re.sub(r"\{[^}]*\}", " ", t.template).split()
            frag = " ".join(words[4:12])
            if len(frag) >= 40:
                out[frag] = f"instruction of {t.variant_of or t.id}"
    return out


def find(text: str, markers: dict[str, str]) -> list[str]:
    """Each marker or held-out id shape in `text`, described (empty: nothing gives the held-out set away)."""
    found = set()
    if markers:
        pat = re.compile("|".join(sorted((re.escape(m) for m in markers), key=len, reverse=True)))
        found |= {f"{markers[m]} ({m[:40]})" for m in set(pat.findall(text))}
    found |= {f"held-out id shape ({m[:40]})" for m in set(ID_PATTERN.findall(text)) - set(markers)}
    return sorted(found)
