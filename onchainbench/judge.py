"""LLM judges for block C's rubric and block E's clarification quality.

A judge scores each rubric criterion 0 or 1 with a one-line reason. It is used only where state can't
decide (writing quality, whether a question was the right one); facts are checked deterministically.

Validation (about 50 labels, >= 90% agreement before scores count; L=onchainbench/judge_labels):
    python -m onchainbench.judge export <run> > $L/todo.jsonl   # items to label (open-set runs only)
    python -m onchainbench.judge label $L/todo.jsonl $L/<you>.jsonl   # 1/0 per criterion
    python -m onchainbench.judge validate $L/<you>.jsonl      # agreement per judge model
Pre-labelling (DeepSeek pre-labels, a reviewer checks the disagreements and a spot check). The labels in
$L/labels.jsonl are model-made: DeepSeek V4.1 Flash pre-labelled them and Claude Opus 5.5 was the reviewer, so a
`label_source` of "human" there means reviewed, not labelled by a person:
    python -m onchainbench.judge prelabel $L/todo.jsonl $L/prelabel.jsonl
    python -m onchainbench.judge review $L/prelabel.jsonl $L/review.jsonl
    python -m onchainbench.judge label $L/review.jsonl $L/<you>.jsonl
    python -m onchainbench.judge merge $L/prelabel.jsonl $L/<you>.jsonl > labels.jsonl
Until a judge model is validated, its scores are reported as "unvalidated" and kept out of the headline.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from onchainbench.llm import HELPER_MODEL, complete, parse_json

LABELS = Path(__file__).resolve().parent / "judge_labels"
PRELABEL_MODEL = "openrouter/deepseek/deepseek-v4.1-flash-1017"  # another family than the judge (HELPER_MODEL)
NO_TASK = "(not recorded with the item: judge the work against the rubric and the reference facts)"

PROMPT = """You are grading an AI agent's work against a rubric. Be strict and literal: a criterion is met (1)
only if the work clearly satisfies it; otherwise 0. Use only the material given here.

## Task given to the agent
{task}

## Reference facts (ground truth)
{facts}

## The agent's work
{work}

## Rubric
{rubric}

Reply with JSON only: {{"scores": [<0 or 1 per criterion, in order>], "reasons": ["<one line per criterion>"]}}"""


@dataclass
class Verdict:
    scores: list[int]
    reasons: list[str]
    judge: str
    cost: float
    validated: bool

    @property
    def score(self) -> float:
        return sum(self.scores) / max(1, len(self.scores))


def item_key(item: dict) -> str:
    """One label item. The trial id alone isn't unique: two open-set runs export the same arm, task and attempt."""
    work = hashlib.sha256(item.get("work", "").encode()).hexdigest()[:16]
    return f"{item['trial']}{item['criteria']}{work}"


def validated_models() -> set[str]:
    p = LABELS / "validated.json"
    return set(json.loads(p.read_text())) if p.is_file() else set()


OFF: str | None = None  # set by the runner for a --no-judges run: every judge call is refused, nothing is called


class JudgesOff(RuntimeError):
    """A judge call in a --no-judges run: refused before any model call; the runner drops the error it leaves."""


def judge(
    task: str, facts: str, work: str, rubric: list[str], arm_model: str | None, model: str | None = None
) -> Verdict:
    """`arm_model` is the scored arm's model, which doesn't choose the judge: every arm gets HELPER_MODEL."""
    if OFF:
        raise JudgesOff(OFF)
    model = model or HELPER_MODEL
    prompt = PROMPT.format(
        task=task,
        facts=facts or "(none)",
        work=work[:30000] or "(empty)",
        rubric="\n".join(f"{i + 1}. {c}" for i, c in enumerate(rubric)),
    )
    spent = 0.0
    for _ in range(2):  # an unparseable reply gets one retry, then is a judge error: not judged, never a 0
        text, cost = complete(model, [{"role": "user", "content": prompt}], max_tokens=4000)  # room for thinking
        spent += cost
        try:
            scores, reasons = _parse(text, len(rubric))
        except (ValueError, TypeError, AttributeError):
            continue
        return Verdict(scores, reasons, model, spent, model in validated_models())
    raise JudgeError(f"unparseable judge reply: {text[:200]}", spent)


class JudgeError(RuntimeError):
    """The judge's reply couldn't be read twice: the caller records an error (`judge_errors`), not a score."""

    def __init__(self, msg: str, cost: float = 0.0):
        super().__init__(msg)
        self.cost = cost  # the bench's own spend on the failed calls, still billed


def _parse(text: str, n: int) -> tuple[list[int], list[str]]:
    """The scores (one 0 or 1 per criterion) and reasons; a reply without a score for every criterion is unread."""
    d = parse_json(text)
    scores = [1 if int(x) else 0 for x in d.get("scores", [])][:n]
    if len(scores) != n:
        raise ValueError(f"{len(scores)} scores for {n} criteria")
    return scores, [str(x) for x in d.get("reasons", [])][:n]


def export(trials: Path) -> list[dict]:
    """Judgements from a run as items for a human to label: fill "label" with 0/1 per criterion."""
    out = []
    for line in trials.read_text().splitlines():
        t = json.loads(line)
        for j in t.get("judgements") or []:
            if "criteria" in j:
                out.append({"trial": f"{t['arm']}/{t['task']}/{t['attempt']}", **j, "label": None})
    return out


def label(todo: Path, out: Path, ask=input, show=print) -> int:
    """Label exported items by hand: shows the work and each criterion, asks 1 or 0, appends to `out` as it goes
    (resumable: items already in `out` are skipped). Returns how many criteria were labelled this session."""
    done = {item_key(json.loads(x)) for x in out.read_text().splitlines()} if out.is_file() else set()
    n = 0
    for line in todo.read_text().splitlines():
        item = json.loads(line)
        if item_key(item) in done:
            continue
        show(f"\n=== {item['trial']}\nFACTS: {item.get('facts', '')}\nWORK:\n{item.get('work', '')[:3000]}\n")
        marks = []
        for c in item["criteria"]:
            a = ""
            while a not in ("0", "1", "q"):
                a = ask(f"  [{c}] met? 1/0 (q to stop): ").strip().lower()
            if a == "q":
                return n
            marks.append(int(a))
        with out.open("a") as f:
            f.write(json.dumps({**item, "label": marks}) + "\n")
        n += len(marks)
    return n


def prelabel(todo: Path, out: Path, model: str = PRELABEL_MODEL, judge_fn=None) -> float:
    """A second model's 0/1 per criterion on each exported item, written to `out` as it goes (resumable).
    Returns the USD spent this session."""
    judge_fn = judge_fn or judge
    done = {item_key(json.loads(x)) for x in out.read_text().splitlines()} if out.is_file() else set()
    spent = 0.0
    for line in todo.read_text().splitlines():
        item = json.loads(line)
        if item_key(item) in done:
            continue
        try:
            v = judge_fn(NO_TASK, item.get("facts", ""), item.get("work", ""), item["criteria"], None, model=model)
        except JudgeError as e:  # left out of `out`, so a later session asks again
            spent += e.cost
            continue
        spent += v.cost
        with out.open("a") as f:
            f.write(
                json.dumps({**item, "prelabel": v.scores, "prelabel_reasons": v.reasons, "prelabel_model": model})
                + "\n"
            )
    return spent


def review_set(rows: list[dict], spot: int = 15, seed: int = 0) -> list[dict]:
    """What the human labels: every item where the pre-label and the judge disagree, plus a random spot check of
    items where they agree. Taking agreed items as labels unchecked would inflate the judge's agreement by
    construction; the spot check measures how often two models agreeing are both wrong."""
    split = [r for r in rows if r["prelabel"] != r["scores"]], [r for r in rows if r["prelabel"] == r["scores"]]
    picked = random.Random(seed).sample(split[1], min(spot, len(split[1])))
    return [{**r, "review": "disagree"} for r in split[0]] + [{**r, "review": "spot"} for r in picked]


def merge(prelabelled: list[dict], reviewed: list[dict]) -> list[dict]:
    """Final labels: the human's where reviewed, else the agreed pre-label (marked as such)."""
    human = {item_key(r): r for r in reviewed}
    out = []
    for r in prelabelled:
        key = item_key(r)
        if key in human:
            out.append({**r, "label": human[key]["label"], "label_source": "human", "review": human[key].get("review")})
        elif r["prelabel"] == r["scores"]:
            out.append({**r, "label": r["prelabel"], "label_source": "agreed"})
        else:
            out.append({**r, "label": None, "label_source": "unreviewed"})
    return out


def agreement_by_source(rows: list[dict]) -> dict:
    """Agreement on the human-reviewed items alone, per review kind, next to the overall figure."""
    human = [r for r in rows if r.get("label_source") == "human"]
    return {
        "all": agreement(rows),
        "human": agreement(human),
        "spot_both_wrong": sum(r["label"] != r["scores"] for r in human if r.get("review") == "spot"),
        "spot_n": sum(r.get("review") == "spot" for r in human),
    }


def agreement(rows: list[dict]) -> dict:
    """Per judge model: share of criteria where the judge's score equals the human label."""
    out: dict = {}
    for r in rows:
        if r.get("label") is None or "scores" not in r:
            continue
        m = out.setdefault(r["judge"], {"agree": 0, "n": 0})
        for a, b in zip(r["scores"], r["label"], strict=False):
            m["agree"] += int(a) == int(b)
            m["n"] += 1
    for m in out.values():
        m["agreement"] = round(m["agree"] / max(1, m["n"]), 3)
        m["passes"] = m["n"] >= 50 and m["agreement"] >= 0.9
    return out


def main(argv: list[str]) -> int:  # pragma: no cover - a CLI around agreement() and the run files
    if argv[0] == "export":
        print("\n".join(json.dumps(x) for x in export(Path("results") / argv[1] / "trials.jsonl")))
        return 0
    if argv[0] == "label":
        print(f"labelled {label(Path(argv[1]), Path(argv[2]))} criteria")
        return 0
    if argv[0] == "prelabel":
        print(f"spent ${prelabel(Path(argv[1]), Path(argv[2]), *argv[3:4]):.4f}")
        return 0
    if argv[0] == "review":
        rows = review_set([json.loads(x) for x in Path(argv[1]).read_text().splitlines() if x.strip()])
        Path(argv[2]).write_text("".join(json.dumps(r) + "\n" for r in rows))
        n = sum(r["review"] == "disagree" for r in rows)
        print(f"{n} disagreements, {len(rows) - n} spot checks")
        return 0
    if argv[0] == "merge":
        read = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]  # noqa: E731
        print("\n".join(json.dumps(r) for r in merge(read(argv[1]), read(argv[2]))))
        return 0
    if argv[0] == "validate":
        rows = [json.loads(x) for x in Path(argv[1]).read_text().splitlines() if x.strip()]
        res = agreement(rows)
        print(json.dumps(agreement_by_source(rows) if any("label_source" in r for r in rows) else res, indent=2))
        ok = {m for m, v in res.items() if v["passes"]}
        if ok:
            (LABELS / "validated.json").write_text(json.dumps(sorted(validated_models() | ok)))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
