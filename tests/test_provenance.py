"""Every trial carries what it ran with (bench digest, held-out manifest, frozen agent's digest, price); `merge` and `--fill`
refuse to combine trials that differ, and a task scored in two shards is an error, not the later one winning."""

import json
import types

import pytest

from onchainbench import arms as A
from onchainbench import freeze, merge
from onchainbench import run as R

ARM = "olas-ocb-agent:m"


def _shard(root, name, rows, **cfg):
    d = root / name
    d.mkdir()
    base = {"split": "test", "seed": 0, "attempts": 1, "arms": [ARM], "tasks": [r["task"] for r in rows], "prices": {}}
    (d / "config.json").write_text(json.dumps({**base, **cfg}))
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _row(task, passed, **k):
    return {"arm": ARM, "task": task, "attempt": 0, "chain": "gnosis", "block": "A", "family": task,
            "passed": passed, "infra": False, "seconds": 1, "cost_usd": 0.1, "metrics": {}, **k}  # fmt: skip


def _cfg(digest, bench):
    return {"agent": {"version": "0.3.5", "digest": digest, "unchanged": True}, "bench_digest": bench,
            "manifest_sha256": "m-" + bench}  # fmt: skip


def test_merge_refuses_shards_of_different_versions(tmp_path, monkeypatch):
    """The repro: two shards across an agent re-freeze and a bench code change merged without a word."""
    monkeypatch.setattr("onchainbench.stats.summarize", lambda rs: {})
    a = _shard(tmp_path, "shard-before-refreeze", [_row("t1", False), _row("t2", False)], **_cfg("digest-OLD", "1111"))
    b = _shard(tmp_path, "shard-after-refreeze", [_row("t3", True), _row("t1", True)], **_cfg("digest-NEW", "2222"))
    with pytest.raises(merge.MergeError) as e:
        merge.merge(tmp_path / "merged", [a, b])
    msg = str(e.value)
    assert msg.startswith("shards ran on different versions: ")
    assert 'bench_digest: "1111" in shard-before-refreeze; "2222" in shard-after-refreeze' in msg
    assert f'agent_digest of {ARM}: "digest-OLD" in shard-before-refreeze; "digest-NEW" in shard-after' in msg
    assert "manifest_sha256" in msg and not (tmp_path / "merged").exists()


def test_a_task_scored_in_two_shards_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("onchainbench.stats.summarize", lambda rs: {})
    same = _cfg("d", "c")
    a = _shard(tmp_path, "a", [_row("t1", False, trial_dir="pe2-1"), _row("t2", False, infra=True)], **same)
    b = _shard(tmp_path, "b", [_row("t1", True, trial_dir="pe2-2"), _row("t2", True)], **same)
    with pytest.raises(merge.MergeError, match=f"{ARM} t1 #0 is scored in both a and b: a re-run would replace"):
        merge.merge(tmp_path / "out", [a, b])
    # one trial that --fill kept in both shards (the same trial_dir) is the same trial: kept once
    c = _shard(tmp_path, "c", [_row("t1", False, trial_dir="pe2-1", filled_from="a"), _row("t2", True)], **same)
    assert merge.merge(tmp_path / "out", [a, c]) == {"trials": 2, "shards": 2}
    rows = {r["task"]: r for r in map(json.loads, (tmp_path / "out" / "trials.jsonl").read_text().splitlines())}
    assert rows["t2"]["passed"] and not rows["t2"]["infra"]  # a scored trial still replaces an infra failure
    d = _shard(tmp_path, "d", [_row("t1", True)], **same)  # no trial_dir: can't be the same trial
    with pytest.raises(merge.MergeError, match="scored in both"):
        merge.merge(tmp_path / "out2", [a, d])


def test_stamped_and_older_records(tmp_path, monkeypatch):
    """Older records (no stamp) take their shard's config.json values: they merge with each other as before when
    those match, and with stamped records when they match too; anything else is refused."""
    monkeypatch.setattr("onchainbench.stats.summarize", lambda rs: {})
    old = _shard(tmp_path, "old", [_row("t1", True)], **_cfg("d", "c"))
    old2 = _shard(tmp_path, "old2", [_row("t2", True)], **_cfg("d", "c"))
    assert merge.merge(tmp_path / "o", [old, old2])["trials"] == 2
    stamp = {"bench_digest": "c", "manifest_sha256": "m-c", "agent_digest": "d", "price": None}
    new = _shard(tmp_path, "new", [_row("t3", True, provenance=stamp)], **_cfg("d", "c"))
    assert merge.merge(tmp_path / "o2", [old, new])["trials"] == 2
    moved = _shard(
        tmp_path, "moved", [_row("t4", True, provenance={**stamp, "agent_digest": "live"})], **_cfg("d", "c")
    )
    with pytest.raises(merge.MergeError, match=f'agent_digest of {ARM}: "d" in old; "live" in moved'):
        merge.merge(tmp_path / "o3", [old, moved])
    # an older config whose agent changed after its freeze: the frozen digest isn't what ran, so it's unknown
    changed = _shard(tmp_path, "changed", [_row("t5", True)], **{**_cfg("d", "c"), "agent": {"digest": "d"}})
    with pytest.raises(merge.MergeError, match='"d" in old; null in changed'):
        merge.merge(tmp_path / "o4", [old, changed])
    # a built-in arm has no agent digest, whatever its config says; prices are per arm
    cc = {**_row("t6", True), "arm": "codex:m"}
    p1 = _shard(tmp_path, "p1", [cc], **_cfg("d", "c"), prices={"codex:m": {"in": 1}})
    p2 = _shard(tmp_path, "p2", [{**cc, "task": "t7"}], **_cfg("d", "c"), prices={"codex:m": {"in": 2}})
    with pytest.raises(merge.MergeError, match=r'price of codex:m: \{"in": 1\} in p1; \{"in": 2\} in p2'):
        merge.merge(tmp_path / "o5", [p1, p2])
    assert R.trial_provenance(cc, {"agent": {"digest": "d", "unchanged": True}})["agent_digest"] is None


def test_arm_digests(tmp_path, monkeypatch):
    """The installed olas-ocb-agent package's digest (the one its CLI runs) for its arms; none for any other."""
    assert R.arm_digest("codex") is None and R.arm_digest("hermes") is None
    (tmp_path / "olas_ocb_agent").mkdir()
    (tmp_path / "olas_ocb_agent" / "agent.py").write_text("X = 1\n")
    monkeypatch.setattr(A, "cli_package", lambda kind: {"version": "0.3.10", "dir": str(tmp_path / "olas_ocb_agent")})
    monkeypatch.setattr(A, "harness_problem", lambda kind: None)
    d = R.arm_digest("olas-ocb-agent-mech")
    assert d == freeze.agent_digest(freeze.agent_files(tmp_path / "olas_ocb_agent")) and len(d) == 64
    assert freeze.agent_dir() == tmp_path / "olas_ocb_agent" and freeze.package_digest("other_pkg") is None

    def unreadable(kind):
        raise RuntimeError("no interpreter")

    monkeypatch.setattr(A, "cli_package", unreadable)
    assert R.arm_digest("olas-ocb-agent") is None  # can't be read: no digest, never a wrong one
    monkeypatch.setattr(A, "harness_problem", lambda kind: "not installed")
    with pytest.raises(FileNotFoundError, match="isn't installed"):
        freeze.agent_dir()


def _main_root(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "key_usage", lambda settle=False: None)
    _at(monkeypatch, "code-1")
    (tmp_path / "results").mkdir()


def test_python_packages_records_the_direct_dependencies_installed(monkeypatch):
    import importlib.metadata as md
    import platform

    have = R.python_packages()
    assert have["python"] == platform.python_version() and have["eth-account"] == md.version("eth-account")
    assert {"litellm", "pyyaml", "zstandard", "eth-hash"} <= set(have)
    assert "pytest" not in have and "jsonschema" not in have  # the extras' (dev, publish) aren't the bench's own
    monkeypatch.setattr(md, "requires", lambda dist: ["eth-abi>=5", "no-such-dist-xyz>=1", 'pytest>=8; extra == "dev"'])
    assert R.python_packages() == {"eth-abi": md.version("eth-abi"), "no-such-dist-xyz": None, "python": have["python"]}
    monkeypatch.undo()
    assert set(R.python_packages("no-such-dist-xyz")) == {"python"}  # not installed as a distribution


def _at(monkeypatch, digest):
    monkeypatch.setattr(R, "provenance", lambda split: {"date": "2026-11-15", "bench_digest": digest})


def test_every_trial_is_stamped_and_fill_refuses_other_versions(tmp_path, monkeypatch, capsys):
    _main_root(tmp_path, monkeypatch)
    task = R.select("open", "", "A", "", limit=1)[0]

    def trial(arm, t, k, envs, seed, sessions, run_tag=None):
        return {"arm": arm, "task": t.id, "attempt": k, "chain": t.chain, "block": t.block, "family": t.family,
                "passed": True, "infra": False, "seconds": 0.1, "cost_usd": 0.0, "metrics": {}}  # fmt: skip

    monkeypatch.setattr(R, "trial", trial)
    forks = {"gnosis": {"anvil/v1.9.0"}}  # each fork's build, written to config.json at the end
    monkeypatch.setattr(
        R, "Envs", lambda out: types.SimpleNamespace(stop=lambda: None, restarts=0, client_versions=forks)
    )
    monkeypatch.setattr(R, "anvil_version", lambda: "anvil/v1.9.0")
    assert R.main(["--arms", "oracle", "--tasks", task.id, "--run", "prev"]) == 0
    rows = [json.loads(x) for x in (tmp_path / "results" / "prev" / "trials.jsonl").read_text().splitlines()]
    cfg = json.loads((tmp_path / "results" / "prev" / "config.json").read_text())
    assert cfg["anvil_version"] == "anvil/v1.9.0" and cfg["fork_client_versions"] == {"gnosis": ["anvil/v1.9.0"]}
    assert cfg["python_packages"] == R.python_packages() and cfg["python_packages"]["litellm"]
    assert rows[0]["provenance"] == {
        "bench_digest": "code-1",
        "manifest_sha256": None,
        "anvil": "anvil/v1.9.0",  # the trial's fork's build, the run's when the trial had no fork
        "agent_digest": None,
        "price": None,
        "harness_version": None,  # the version probe is stubbed in tests
    }
    # --fill from the same versions keeps the trial
    assert R.main(["--arms", "oracle", "--tasks", task.id, "--run", "again", "--fill", "prev"]) == 0
    assert "keeping 1 trials, running 0" in capsys.readouterr().out
    # a code change since: refused, naming what differs
    _at(monkeypatch, "code-2")
    assert R.main(["--arms", "oracle", "--tasks", task.id, "--run", "after", "--fill", "prev"]) == 2
    out = capsys.readouterr().out
    assert "refusing --fill prev: its trials ran on other versions" in out
    assert 'bench_digest: "code-2" in this run; "code-1" in prev' in out
    # another Anvil build: refused too
    _at(monkeypatch, "code-1")
    monkeypatch.setattr(R, "anvil_version", lambda: "anvil/v1.10.0")
    assert R.main(["--arms", "oracle", "--tasks", task.id, "--run", "anvil", "--fill", "prev"]) == 2
    assert 'anvil: "anvil/v1.10.0" in this run; "anvil/v1.9.0" in prev' in capsys.readouterr().out


def test_merge_compares_the_bench_digest_and_refuses_records_without_one(tmp_path, monkeypatch):
    monkeypatch.setattr("onchainbench.stats.summarize", lambda rs: {})
    stamp = {"bench_digest": "code-1", "manifest_sha256": "m-c1", "agent_digest": "d", "price": None}

    def shard(name, task, **p):
        return _shard(tmp_path, name, [_row(task, True, provenance={**stamp, **p})], **_cfg("d", "c"))

    a, same, code = shard("a", "t1"), shard("same", "t2"), shard("code", "t3", bench_digest="code-2")
    assert merge.merge(tmp_path / "o1", [a, same])["trials"] == 2
    with pytest.raises(merge.MergeError, match='bench_digest: "code-1" in a; "code-2" in code'):
        merge.merge(tmp_path / "o2", [a, code])
    # a record without a digest (stamped with a commit only, or no stamp and none in its config): refused, even when
    # every other version matches, and even when no record has one
    commit_only = {k: v for k, v in stamp.items() if k != "bench_digest"} | {"bench_commit": "c1"}
    old = _shard(tmp_path, "old", [_row("t4", True, provenance=commit_only)], **_cfg("d", "c1"))
    bare = {k: v for k, v in _cfg("d", "c1").items() if k != "bench_digest"}
    older = _shard(tmp_path, "older", [_row("t5", True)], **bare)
    older2 = _shard(tmp_path, "older2", [_row("t6", True)], **bare)
    with pytest.raises(merge.MergeError, match="no bench_digest in old: trials there don't record the bench's code"):
        merge.merge(tmp_path / "o3", [a, old])
    with pytest.raises(merge.MergeError, match=r"no bench_digest in older, older2: .*\(run them again\)"):
        merge.merge(tmp_path / "o4", [older, older2])
    assert not (tmp_path / "o3").exists() and not (tmp_path / "o4").exists()


def test_the_bench_digest_covers_code_and_task_data_only(tmp_path):
    pkg = tmp_path / "onchainbench"
    (pkg / "tasks").mkdir(parents=True)
    (pkg / "contracts" / "out" / "build-info").mkdir(parents=True)
    (pkg / "contracts" / "out" / "A.sol").mkdir()
    (pkg / "contracts" / "cache").mkdir()
    (pkg / "__pycache__").mkdir()
    (pkg / "run.py").write_text("X = 1\n")
    (pkg / "tasks" / "block_a.py").write_text("T = 1\n")
    (pkg / "abi.json").write_text("{}")
    (pkg / "contracts" / "out" / "A.sol" / "A.json").write_text('{"bytecode": {"object": "0x00"}}')
    base = freeze.bench_digest(pkg)
    assert base == freeze.bench_digest(pkg) and len(base) == 64
    (pkg / "README.md").write_text("docs")  # docs, the compiler's working files and caches don't count
    (pkg / "tasks" / "NOTES.md").write_text("a doc beside the tasks")
    (pkg / "contracts" / "out" / "build-info" / "abc.json").write_text("{}")
    (pkg / "contracts" / "cache" / "solidity-files-cache.json").write_text("{}")
    (pkg / "__pycache__" / "run.cpython-312.pyc").write_bytes(b"x")
    (pkg / "__pycache__" / "x.json").write_text("{}")
    assert freeze.bench_digest(pkg) == base
    (pkg / "contracts" / "out" / "A.sol" / "A.json").write_text('{"bytecode": {"object": "0x01"}}')  # what deploys
    changed = freeze.bench_digest(pkg)
    assert changed != base
    (pkg / "tasks" / "block_a.py").write_text("T = 2\n")  # a task changed
    assert freeze.bench_digest(pkg) not in (base, changed)
    assert len(freeze.bench_digest()) == 64  # the installed bench's own


def test_a_docs_only_edit_leaves_the_bench_digest_unchanged(tmp_path):
    """The squashed public repo has other commits than the private one; the digest is the same for the same code,
    whatever the docs say."""
    pkg = tmp_path / "onchainbench"
    (pkg / "tasks").mkdir(parents=True)
    (pkg / "run.py").write_text("X = 1\n")
    (pkg / "README.md").write_text("how to run")
    (pkg / "tasks" / "GUIDE.md").write_text("how to write a task")
    base = freeze.bench_digest(pkg)
    (pkg / "README.md").write_text("how to run, reworded")
    (pkg / "tasks" / "GUIDE.md").write_text("how to write a task, with an example")
    (tmp_path / "README.md").write_text("the repo's README, outside the package")
    assert freeze.bench_digest(pkg) == base


def test_the_bench_digest_flag_prints_this_checkouts_digest(capsys):
    """`python -m onchainbench.freeze --bench-digest`: what a reader compares with a run's `bench_digest`."""
    assert freeze.main(["--bench-digest"]) == 0
    assert capsys.readouterr().out == freeze.bench_digest() + "\n"
