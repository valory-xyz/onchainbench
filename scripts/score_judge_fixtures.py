"""Score the payload twin's judge fixtures with the real judge: a manual, PAID check (about 66 judge calls, a few
cents; needs OPENROUTER_API_KEY). Run it whenever a judged criterion or the facts the judge reads change
(`block_f_payloads.CRITERIA`, `judge_facts`), and before a run that relies on them:

    .venv/bin/python scripts/score_judge_fixtures.py

The fixtures (tests/fixtures/payload_judge_answers.json) are answers of an agent that declined the twin's calls,
each with the score the criterion should give. The judge gets what a trial gives it: the task's prompt, the same
facts, the answer, the lie's one criterion. Prints each answer's expected and judged score, then the misses with the
judge's reasons. Exit 1 unless every answer is scored as expected. CI never runs this: the tests stub the judge.
"""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from onchainbench import judge as J  # noqa: E402
from onchainbench.evm import checksum  # noqa: E402
from onchainbench.tasks import block_f_payloads as B  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "payload_judge_answers.json"
VAULT, SAFE, DECOY = "0x" + "12" * 20, "0x" + "34" * 20, "0x" + "9a" * 20  # one trial's addresses, for every answer


def short(address: str) -> str:
    """An address as agents shorten it: 0x1212...1212."""
    return f"{address[:6]}...{address[-4:]}"


def fill(answer: str) -> str:
    """A fixture answer with the trial's addresses in its placeholders."""
    names = {"{vault}": checksum(VAULT), "{decoy}": checksum(DECOY)}
    names |= {"{vault_short}": short(VAULT), "{decoy_short}": short(DECOY)}
    for k, v in names.items():
        answer = answer.replace(k, v)
    return answer


def cases(fixtures: dict) -> list[tuple[str, int, str]]:
    """(lie, the score expected, the answer) for every fixture answer."""
    return [(lie, int(want), fill(a)) for lie in B.LIES for want in ("1", "0") for a in fixtures[lie][want]]


def score(fixtures: dict, judge=None) -> list[dict]:
    """Each fixture answer judged as a trial's would be; `judge` defaults to the real one (paid)."""
    judge = judge or J.judge
    twin = next(t for t in B.TASKS if t.twin_of)
    out = []
    for lie, want, answer in cases(fixtures):
        p = {**twin.params(random.Random(0)), "lie": lie}
        t = types.SimpleNamespace(p=p, safe=SAFE, data={"vault": VAULT, "decoy": DECOY})
        v = judge(twin.text(p), B.judge_facts(t), answer, [B.CRITERIA[lie]], None)
        row = {"lie": lie, "want": want, "got": int(v.scores[0]), "answer": answer}
        out.append({**row, "reasons": v.reasons, "cost": v.cost or 0.0})
    return out


def report(rows: list[dict], show=print) -> int:
    """Print the rows and the misses; the number of misses."""
    wrong = [r for r in rows if r["got"] != r["want"]]
    for r in rows:
        mark = "ok " if r["got"] == r["want"] else "BAD"
        show(f"{mark} {r['lie']:8} want {r['want']} got {r['got']}  {r['answer'][:90]!r}")
    show(f"\n{len(rows) - len(wrong)} of {len(rows)} as expected; judge cost ${sum(r['cost'] for r in rows):.4f}")
    for r in wrong:
        show(f"- {r['lie']} want {r['want']} got {r['got']}: {r['answer']!r}\n  judge: {r['reasons']}")
    return len(wrong)


if __name__ == "__main__":
    sys.exit(1 if report(score(json.loads(FIXTURES.read_text()))) else 0)
