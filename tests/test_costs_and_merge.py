"""Cost accounting (sources, the infra reserve, the key's billed usage, judges' billed cost) and merging shards."""

import json
import types

import pytest

from onchainbench import arms as A
from onchainbench import llm, merge
from onchainbench import run as R


def test_cost_source_and_cap_enforcement(monkeypatch):
    assert R.cost_source({"cost_source": "harness"}) == "harness"
    assert R.cost_source({"cost_sources": {"billed": 3}}) == "billed"
    assert R.cost_source({"cost_sources": {"billed": 3, "estimate": 1}}) == "mixed"
    assert R.cost_source({}) == "estimate"
    monkeypatch.delenv("OPENROUTER_PROVISIONING_KEY", raising=False)
    assert R.cap_enforcement("oracle") == "none needed"
    assert R.cap_enforcement("codex:openrouter/m") == "between turns"
    assert R.cap_enforcement("olas-ocb-agent:m") == "in the turn"
    for m in (
        "claude-opus-5-5",
        "openrouter/anthropic/claude-opus-5.5",
        "openrouter/deepseek/deepseek-v4.1-flash-1017",
    ):
        assert R.cap_enforcement(f"claude-code:{m}") == "between turns"  # never shown its budget, Claude or not
    monkeypatch.setenv("OPENROUTER_PROVISIONING_KEY", "p")
    assert R.cap_enforcement("hermes:openrouter/m") == "per-trial key"


def test_key_usage_and_spending(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert R.key_usage() is None
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    usage = iter([1.25])
    monkeypatch.setattr(A, "_or_api", lambda m, url, key: {"data": {"usage": next(usage)}})
    assert R.key_usage() == 1.25
    monkeypatch.setattr(A, "_or_api", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    assert R.key_usage() is None  # informational: never fails a run
    # OpenRouter counts usage with a delay: at the end, read until two readings agree
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    reads = iter([1.0, 1.2, 1.25, 1.25])
    monkeypatch.setattr(A, "_or_api", lambda m, url, key: {"data": {"usage": next(reads)}})
    assert R.key_usage(settle=True) == 1.25
    reads = iter([1.0, None])
    monkeypatch.setattr(A, "_or_api", lambda m, url, key: {"data": {"usage": next(reads)}})
    assert R.key_usage(settle=True) == 1.0  # a failed reading keeps the last one
    monkeypatch.setattr(R, "KEY_SETTLE_S", (0, 2))
    reads = iter([1.0, 2.0, 3.0])
    monkeypatch.setattr(A, "_or_api", lambda m, url, key: {"data": {"usage": next(reads)}})
    assert R.key_usage(settle=True) == 3.0  # gives up after the last try
    records = [
        {"cost_usd": 0.5, "cost_source": "billed"},
        {"cost_usd": 0.25, "cost_source": "estimate"},
        {"cost_usd": 0.0, "cost_source": "unknown", "cost_reserved_usd": 2.0},
        {"cost_usd": 9.0, "cost_source": "billed", "filled_from": "prev"},
    ]
    monkeypatch.setattr(R, "key_usage", lambda settle=False: 3.0)
    s = R.spending(records, 2.0)
    assert s == {
        "this_run_usd": 0.75,
        "kept_from_fill_usd": 9.0,
        "by_source_usd": {"billed": 0.5, "estimate": 0.25, "unknown": 0.0},
        "reserved_for_infra_usd": 2.0,
        "key_billed_usd": 1.0,
    }
    assert R.spending([{}], None)["key_billed_usd"] is None and R.spending([{}], None)["by_source_usd"] == {"none": 0}


def test_judges_take_the_billed_cost(monkeypatch):
    calls = []

    def completion(**kw):
        calls.append(kw)
        msg = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage={"cost": 0.003})

    fake = types.SimpleNamespace(completion=completion, completion_cost=lambda **k: 1.0)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    monkeypatch.setattr("onchainbench.pricing.register_price", lambda m: True)
    assert llm.complete("openrouter/a/b", [{"role": "user", "content": "x"}]) == ("ok", 0.003)
    assert calls[-1]["extra_body"] == {"usage": {"include": True}}
    assert llm.complete("anthropic/x", [{"role": "user", "content": "x"}]) == ("ok", 0.003)
    assert "extra_body" not in calls[-1]


def _shard(root, name, rows, seed=0, arms=("a",)):
    d = root / name
    d.mkdir()
    cfg = {"split": "open", "seed": seed, "attempts": 1, "arms": list(arms), "tasks": ["t1"], "prices": {"a": 1}}
    cfg["bench_digest"] = "d"
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _row(arm, task, attempt=0, **kw):
    return {"arm": arm, "task": task, "attempt": attempt, "block": "A", "passed": True, "infra": False,
            "seconds": 1, "cost_usd": 0.1, **kw}  # fmt: skip


def test_merge_shards(tmp_path, monkeypatch):
    monkeypatch.setattr("onchainbench.stats.summarize", lambda rs: {"trials": len(rs)})  # stats has its own tests
    res = tmp_path / "results"
    res.mkdir()
    a = _shard(res, "a", [_row("a", "t1"), _row("a", "t2", infra=True, passed=False)], arms=("a",))
    b = _shard(
        tmp_path, "b", [_row("b", "t1"), _row("a", "t2", cost_usd=0.2), _row("a", "t1", infra=True)], arms=("b",)
    )
    (b / "summary.json").write_text(json.dumps({"spending": {"key_billed_usd": 0.3}}))
    out = res / "all"
    assert merge.merge(out, [a, b]) == {"trials": 3, "shards": 2}
    rows = {(r["arm"], r["task"]): r for r in map(json.loads, (out / "trials.jsonl").read_text().splitlines())}
    assert not rows[("a", "t2")]["infra"] and rows[("a", "t2")]["cost_usd"] == 0.2  # scored beats infra
    assert not rows[("a", "t1")]["infra"]  # a later infra failure doesn't replace a scored trial
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["arms"] == ["a", "b"] and cfg["merged_from"] == ["a", "b"] and cfg["run"] == "all"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["spending"]["shards_key_billed_usd"] == {"a": None, "b": 0.3}
    c = _shard(res, "c", [], seed=1)
    with pytest.raises(merge.MergeError, match="seed"):
        merge.merge(out, [a, c])
    with pytest.raises(merge.MergeError, match="no shards"):
        merge.merge(out, [])
    monkeypatch.setattr(merge, "ROOT", tmp_path)
    assert merge.main([]) == 2
    assert merge.main(["x", "a", "c"]) == 1
    assert merge.main(["y", "a"]) == 0


def test_relocate_copies_session_files_and_drops_dangling_paths(tmp_path):
    src, dst = tmp_path / "shard-0", tmp_path / "all"
    f = src / "sessions" / "transcripts" / "a-t1-0.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("{}\n")
    r = {"arm": "a", "transcript": "/home/runner/work/x/results/shard-0/sessions/transcripts/a-t1-0.jsonl"}
    moved = R.relocate({**r, "session": "C:\\run\\sessions\\gone.jsonl", "other": "kept"}, src, dst)
    assert moved["transcript"] == str(dst / "sessions" / "transcripts" / "a-t1-0.jsonl")
    assert (dst / "sessions" / "transcripts" / "a-t1-0.jsonl").read_text() == "{}\n"
    assert "session" not in moved and moved["other"] == "kept"  # a file that's gone isn't pointed at
    assert R.relocate(moved, dst, dst)["transcript"] == moved["transcript"]  # already in place: no copy needed
    assert R.relocate({"transcript": None}, src, dst) == {"transcript": None}


def test_merge_brings_the_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr("onchainbench.stats.summarize", lambda rs: {})
    res = tmp_path / "results"
    res.mkdir()
    a = _shard(res, "a", [_row("a", "t1", transcript="/x/results/a/sessions/transcripts/t.jsonl")])
    (a / "sessions" / "transcripts").mkdir(parents=True)
    (a / "sessions" / "transcripts" / "t.jsonl").write_text("x")
    merge.merge(res / "all", [a])
    row = json.loads((res / "all" / "trials.jsonl").read_text())
    assert row["transcript"] == str(res / "all" / "sessions" / "transcripts" / "t.jsonl")
    assert (res / "all" / "sessions" / "transcripts" / "t.jsonl").read_text() == "x"


def test_merge_reads_the_key_once_for_the_whole_run(tmp_path, monkeypatch):
    monkeypatch.setattr("onchainbench.stats.summarize", lambda rs: {})
    monkeypatch.setattr(merge.time, "sleep", lambda s: None)
    res = tmp_path / "results"
    res.mkdir()
    a, b = _shard(res, "a", [_row("a", "t1")]), _shard(res, "b", [_row("b", "t1")], arms=("b",))
    for d, start in ((a, 10.0), (b, 10.5)):
        cfg = json.loads((d / "config.json").read_text())
        (d / "config.json").write_text(json.dumps({**cfg, "key_usage_start_usd": start}))
    monkeypatch.setattr(merge, "key_usage", lambda settle=False: 12.25)
    merge.merge(res / "all", [a, b], key_wait_s=60)
    assert json.loads((res / "all" / "summary.json").read_text())["spending"]["key_billed_usd"] == 2.25
    monkeypatch.setattr(merge, "key_usage", lambda settle=False: None)  # no key or no network: unknown
    merge.merge(res / "all2", [a, b], key_wait_s=60)
    assert json.loads((res / "all2" / "summary.json").read_text())["spending"]["key_billed_usd"] is None
    monkeypatch.setattr(merge, "ROOT", tmp_path)
    assert merge.main(["--key-wait", "0", "all3", "a"]) == 0
