"""Per-arm billed cost from the key's spend over each arm's shards, shared out over its trials (billing.py)."""

import json

import pytest

from onchainbench import billing as B
from onchainbench import merge

CODEX = "codex:openrouter/openai/gpt-5.6-sol?provider=openai"
CC = "claude-code:openrouter/anthropic/claude-opus-5.5?provider=anthropic"
AGENT = "olas-ocb-agent:openrouter/deepseek/x"


def trial(arm, task, cost, tok=None, **kw):
    return {"arm": arm, "task": task, "attempt": 0, "cost_usd": cost, "prompt_tokens": tok, "completion_tokens": 0,
            "cost_source": "estimate", **kw}  # fmt: skip


def shard(name, arms, start, billed, rows=()):
    return {"name": name, "arms": list(arms), "start": start, "billed": billed, "rows": list(rows)}


def test_shard_spend_runs_to_the_next_reading():
    s = [shard("a", ["x"], 1.0, 0.5), shard("b", ["y"], 1.6, 1.0), shard("c", ["z"], None, None)]
    spend, why = B.shard_spend(s, 3.0)
    assert why == {} and spend["c"] is None
    assert spend["a"] == {"key_usd": 0.6, "own_usd": 0.5, "late_usd": 0.1}  # 0.1 arrived after a's own reading
    assert spend["b"] == {"key_usd": 1.4, "own_usd": 1.0, "late_usd": 0.4}
    assert B.shard_spend(s, None)[0]["b"] is None  # no reading after the last shard: not settled
    assert B.shard_spend([shard("a", ["x"], 1.0, None), shard("b", ["y"], 2.0, 0.1)], 2.1)[0]["a"]["key_usd"] == 1.0
    spend, why = B.shard_spend([shard("a", ["x"], 1.0, 0.5), shard("b", ["y"], 1.2, 0.5)], 2.0)
    assert spend == {"a": None, "b": None} and why == {"a": ["b"], "b": ["a"]}
    spend, _ = B.shard_spend([shard("a", ["x"], 1.0, 0.2)], 2.0)  # 0.8 after a reading of 0.2: not its lag
    assert spend["a"]["key_usd"] is None and spend["a"]["late_usd"] == 0.8


def test_an_arms_billed_spend_is_shared_out_by_tokens():
    judged = {"judgements": [{"cost": 0.02}, {"error": "down"}], "user_sim_cost_usd": 0.03}
    rows = [trial(CODEX, "t1", 0.4, 100), trial(CODEX, "t2", 1.6 + 0.03, 300, **judged)]
    rows += [trial(CODEX, "t3", 0.0, None, infra=True, cost_source="unknown")]  # nothing known: no share
    records = [dict(r) for r in rows]
    rep = B.apportion(records, [shard("s7", [CODEX], 2.0, 1.0, rows)], 3.05)
    # key 1.05 minus the bench's 0.05 (a judge and the simulated user) = 1.0 for the arm, by tokens 1:3
    assert rep[CODEX] == {
        "estimate_usd": 2.0,
        "billed_usd": 1.0,
        "shards": ["s7"],
        "key_usd": 1.05,
        "bench_usd": 0.05,
        "status": "apportioned",
        "weight": "tokens",
        "trials": 2,
    }
    t1, t2, t3 = records
    assert t1["cost_usd"] == pytest.approx(0.25) and t1["cost_estimate_usd"] == 0.4
    assert t2["cost_usd"] == pytest.approx(0.75 + 0.03) and t1["cost_source"] == t2["cost_source"] == B.SOURCE
    assert t3["cost_source"] == "unknown" and "cost_estimate_usd" not in t3
    # no tokens reported: the estimates are the weights
    records = [trial(CODEX, "t1", 0.2), trial(CODEX, "t2", 0.6)]
    rep = B.apportion(records, [shard("s", [CODEX], 0.0, 0.4, records)], 0.4)
    assert rep[CODEX]["weight"] == "estimate" and [r["cost_usd"] for r in records] == pytest.approx([0.1, 0.3])


def test_several_shards_of_one_arm_are_summed_and_billed_trials_kept():
    rows_a, rows_b = [trial(AGENT, "a1", 0.3, cost_source="billed")], [trial(AGENT, "d1", 0.2, 50)]
    records = rows_a + rows_b
    shards = [shard("A", [AGENT], 0.0, 0.3, rows_a), shard("D", [AGENT], 0.3, 0.25, rows_b)]
    rep = B.apportion(records, shards, 0.55)
    assert rep[AGENT]["key_usd"] == 0.55 and rep[AGENT]["status"] == "apportioned"
    assert records[0]["cost_usd"] == 0.3 and records[1]["cost_usd"] == pytest.approx(0.25)  # 0.55 - 0.3 billed
    billed = [trial(AGENT, "a1", 0.3, cost_source="billed")]
    rep = B.apportion(billed, [shard("A", [AGENT], 0.0, 0.31, billed)], 0.31)
    assert rep[AGENT]["status"] == "billed" and rep[AGENT]["unattributed_usd"] == 0.01


@pytest.mark.parametrize(
    ("arms", "shards", "end", "cap", "why"),
    [
        (["claude-code:claude-opus-5-5"], None, 1.0, {}, "don't go through the run's key"),
        ([CODEX], None, 1.0, {CODEX: "per-trial key"}, "don't go through the run's key"),
        ([CODEX], [shard("s", [CODEX], 0.0, 0.5), shard("t", [CC], 0.2, 0.5)], 1.0, {}, "at the same time"),
        ([CODEX], [shard("s", [CODEX, CC], 0.0, 1.0)], 1.0, {}, "several arms"),
        ([CODEX], [shard("s", [CC], 0.0, 1.0)], 1.0, {}, "no shard lists it"),
        ([CODEX], [shard("s", [CODEX], 0.0, 1.0)], None, {}, "no reading of the key after s"),
        ([CODEX], [shard("s", [CODEX], 0.0, 0.5)], 1.0, {}, "more than 50% of its own reading"),
        ([CODEX], [shard("s", [CODEX], 0.0, 1.0)], 1.0, {}, "3.33x the estimate"),  # the estimate is 0.3
        ([CODEX], [shard("s", [CODEX], 0.0, 0.01)], 0.01, {}, "0.03x the estimate"),
    ],
)
def test_the_estimate_stays_where_spend_cant_be_attributed(arms, shards, end, cap, why):
    records = [trial(a, "t1", 0.3, 10) for a in arms]
    shards = shards or [shard("s", arms, 0.0, 1.0)]
    rep = B.apportion(records, shards, end, cap)
    (r,) = rep.values()
    assert r["status"] == "estimate kept" and why in r["reason"]
    assert records[0]["cost_usd"] == 0.3 and records[0]["cost_source"] == "estimate"


def test_nothing_left_and_scripted_arms():
    records = [trial(AGENT, "a1", 0.5, cost_source="billed"), trial(AGENT, "a2", 0.1, 10), trial("oracle", "a1", 0)]
    rep = B.apportion(records, [shard("s", [AGENT], 0.0, 0.4), shard("o", ["oracle"], 0.4, 0.0)], 0.4)
    assert rep[AGENT]["status"] == "estimate kept" and rep[AGENT]["reason"].startswith("nothing left")
    assert rep["oracle"]["status"] == "no model"


def test_merge_reports_billed_beside_the_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(merge.time, "sleep", lambda s: None)
    monkeypatch.setattr(merge, "key_usage", lambda settle=False: 3.0)
    res = tmp_path / "results"
    res.mkdir()
    dirs = []
    for name, arm, start, billed in (("s6", CC, 1.0, 0.9), ("s7", CODEX, 2.0, 0.8)):
        d = res / name
        d.mkdir()
        cfg = {"split": "open", "seed": 0, "attempts": 1, "arms": [arm], "tasks": ["t1"], "key_usage_start_usd": start}
        cfg["bench_digest"] = "d"
        (d / "config.json").write_text(json.dumps(cfg))
        row = {**trial(arm, "t1", 1.0, 100), "chain": "gnosis", "block": "A", "passed": True, "infra": False}
        (d / "trials.jsonl").write_text(json.dumps({**row, "seconds": 1}) + "\n")
        (d / "summary.json").write_text(json.dumps({"spending": {"key_billed_usd": billed}}))
        dirs.append(d)
    merge.merge(res / "all", dirs, key_wait_s=60)
    s = json.loads((res / "all" / "summary.json").read_text())
    by = s["spending"]["by_arm"]
    assert by[CC]["billed_usd"] == 1.0 and by[CODEX]["billed_usd"] == 1.0 and by[CODEX]["estimate_usd"] == 1.0
    assert s["spending"]["shards_key_spend"]["s7"] == {"key_usd": 1.0, "own_usd": 0.8, "late_usd": 0.2}
    assert s["arms"][CODEX]["cost_estimate_usd"] == 1.0 and s["arms"][CODEX]["cost_sources"] == {B.SOURCE: 1}
    rows = [json.loads(x) for x in (res / "all" / "trials.jsonl").read_text().splitlines()]
    assert all(r["cost_source"] == B.SOURCE and r["cost_estimate_usd"] == 1.0 for r in rows)


def test_each_arm_gets_its_own_shards_overlap_reason_and_a_clean_shard_is_still_billed():
    """open-run-1: every arm's entry said "shards shard-0 and shard-1 ran at the same time", arms on shards 6 and 7
    too. The reason now names the arm's own shard, and a shard that overlapped nothing is still attributed."""
    rows = [trial(CODEX, "t1", 0.3, 10), trial(CC, "t1", 0.3, 10), trial(AGENT, "t1", 0.3, 10)]
    shards = [shard("s0", [CODEX], 0.0, 0.5), shard("s1", [CC], 0.2, 0.5), shard("s7", [AGENT], 1.0, 0.3)]
    rep = B.apportion(rows, shards, 1.3)
    assert rep[CODEX]["reason"] == "shard s0 ran at the same time as s1 on the key"
    assert rep[CC]["reason"] == "shard s1 ran at the same time as s0 on the key"
    assert rep[AGENT]["status"] == "apportioned" and rep[AGENT]["billed_usd"] == 0.3
