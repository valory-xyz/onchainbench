"""The publish step: the site's run JSON from a run's trials, opaque held-out ids, trajectories, leak and schema guards."""

import json
import sys
import types

import pytest
from conftest import needs_fixture_heldout

from onchainbench import arms as A
from onchainbench import freeze, transcript
from onchainbench import publish as P
from onchainbench import run as R

OUT = {"attempted": 1, "blocked": 0, "blocked_by": {}, "failed": 0, "prepared": 1, "executed": False}
OUT |= {"executed_evidence": [], "near_misses": [], "applicable": True, "exposed": True}


def rec(arm, task, passed, block="A", chain="gnosis", attempt=0, **kw):
    return {
        "arm": arm,
        "task": task,
        "passed": passed,
        "block": block,
        "chain": chain,
        "family": task,
        "attempt": attempt,
        "seconds": 10.0,
        "cost_usd": 0.01,
        "protocols": ["uniswap-v3"],
        "steps_done": 1 if passed else 0,
        "steps_total": 2,
        "metrics": {},
        **kw,
    }


def run_dir(tmp_path, records, split="open", **cfg):
    d = tmp_path / "run1"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"split": split, "run": "run1", "seed": 3, "date": "2026-11-14", **cfg}))
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return d


CODEX = "codex:openrouter/deepseek/x?provider=fireworks|nogate"
AGENT = "olas-ocb-agent:openrouter/deepseek/x"


@pytest.fixture
def frozen_agent():
    """The agent's arm is frozen by its package (arms.frozen_package)."""
    assert A.frozen_package("olas-ocb-agent") == "olas_ocb_agent"


def open_run(tmp_path):
    events = [transcript.event(0, "tool_call", "ls", tool="shell")]
    tr = transcript.write(tmp_path / "s", "codex__a1__0", events, ["raw"])
    agent_tr = transcript.write(tmp_path / "s", "agent__a1__0", [transcript.event(0, "assistant", "hi")], ["raw"])
    session = tmp_path / "agent.jsonl"  # the agent's own session file: never what gets published
    session.write_text(json.dumps({"type": "message", "message": {"role": "user", "content": "forged"}}) + "\n")
    return [
        rec(CODEX, "a-one", True, transcript=str(tr)),
        rec(CODEX, "a-two", False, block="B", chain="base"),
        rec(CODEX, "f-twin", True, block="F", attack="injection-page", attack_outcome={**OUT, "blocked": 1}),
        rec(CODEX, "a-one", True, infra=True),
        rec(AGENT, "a-one", True, transcript=str(agent_tr), session=str(session), connect_tools_listed=True),
        rec(AGENT, "a-three", True, session=str(session)),  # only the agent's file: no published trajectory
        rec(AGENT, "a-two", True, block="B", chain="base"),
        rec(AGENT, "f-twin", False, block="F", attack="injection-page", attack_outcome={**OUT, "executed": True}),
        rec("oracle", "a-one", True),
    ]


def test_an_entry_from_a_judges_off_run_has_null_interventions_and_validates(tmp_path, frozen_agent):
    """With judges off no hand-back is judged, so the entry's interventions per pass aren't measured: null, never a
    zero, and the schema takes it. publish.build still refuses such a run as a ranked result."""
    recs = [{**r, "judges_off": True} for r in open_run(tmp_path) if r["arm"] != "oracle"]
    with pytest.raises(P.PublishError, match="--no-judges"):
        P.build(run_dir(tmp_path, recs), "https://r")
    doc, _ = P.build(run_dir(tmp_path / "judged", [{**r, "judges_off": False} for r in recs]), "https://r")
    by = {r["arm"]: [] for r in recs}
    for r in recs:
        by[r["arm"]].append(r)
    doc["entries"] = [P.entry(arm, rs, {}, verified=False) for arm, rs in by.items()]
    assert [e["interventions_per_pass"] for e in doc["entries"]] == [None, None]
    P.validate(doc)


def test_an_open_run_publishes_entries_pairs_trials_and_trajectories(tmp_path, frozen_agent):
    versions = {CODEX: "codex-cli 0.98.0"}
    d = run_dir(
        tmp_path,
        open_run(tmp_path),
        agent={"version": "0.3.0", "digest": "abc"},
        bench_digest="c0ffee" * 10 + "c0ff",
        bench_commit="c0ffee",  # an older config's commit: never published
        harness_versions=versions,
    )
    doc, traj = P.build(d, "https://github.com/x/releases/tag/run1", prereg_commit="p1")
    assert doc["entries"][0]["harness_version"] == "codex-cli 0.98.0"  # not only the frozen agent's
    assert doc["bench_digest"] == "c0ffee" * 10 + "c0ff" and "bench_commit" not in doc
    P.validate(doc)
    assert doc["split"] == "open" and doc["run_id"] == "run1" and doc["seed"] == 3 and doc["prereg_commit"] == "p1"
    codex, agent = doc["entries"]
    assert (codex["harness"], codex["model"], codex["provider"]) == (
        "codex|nogate",
        "openrouter/deepseek/x",
        "fireworks",
    )
    assert codex["open"] and agent["open"] and agent["frozen_digest"] == "abc" and agent["harness_version"] == "0.3.0"
    assert codex["completion"]["n_tasks"] == 3 and codex["completion"]["n_trials"] == 3  # the infra trial is out
    assert codex["safety"]["blocked_by"] == {} and codex["safety"]["executed_losses"] == 0
    assert agent["safety"]["executed_losses"] == 1 and agent["safety"]["by_class"]["injection-page"]["trials"] == 1
    assert set(codex["by_block"]) == {"A", "B", "F"} and set(codex["by_protocol"]) == {"uniswap-v3"}
    assert agent["connect_tools_listed_share"] == 1.0 and "connect_tools_listed_share" not in codex
    (pair,) = doc["pairs"]
    # no pre-registered comparator: the pair is ordered by pass rate alone and carries no comparator flag
    rates = {e["harness"]: e["completion"]["mean"] for e in doc["entries"]}
    assert (pair["a"], pair["b"]) == (AGENT, CODEX) and rates["olas-ocb-agent"] > rates["codex|nogate"]
    assert "comparator" not in pair and set(pair["by_block"]) == {"A", "B", "F"}
    # the tasks every entry passed on every scored attempt, by name on an open run: a-three isn't one (codex never
    # ran it), nor a-two and f-twin (one entry failed each); the infra trial and the oracle's don't count
    assert doc["no_signal"] == {"count": 1, "of": 4, "tasks": ["a-one"]}
    P.validate(doc)
    P.validate({**doc, "pairs": [{**pair, "comparator": True}]})  # an older run file with the flag still validates
    assert {r["entry"] for r in doc["trials"]} == {CODEX, AGENT} and all(
        not r["task"].startswith("h-") for r in doc["trials"]
    )
    assert {r["safety"] for r in doc["trials"]} == {"n/a", "blocked", "executed"}
    # the run's size, and which rows are twins: by the task's attack, not its block
    assert doc["task_counts"] == {"tasks": 4, "base_tasks": 4, "twins": 1, "twin_variants": 1}
    assert {r["task"] for r in doc["trials"] if r["twin"]} == {"f-twin"}
    assert codex["safety"]["funnel"]["blocked"] == 1 and agent["safety"]["funnel"]["executed"] == 1
    assert [(t["entry"], [e["kind"] for e in t["events"]]) for t in traj] == [
        (CODEX, ["tool_call"]),
        (AGENT, ["assistant"]),
    ]
    files = P.write(d, doc, traj)
    assert [f.name for f in files] == ["run.json", "trajectories.jsonl", "SHA256SUMS"]
    assert json.loads(files[0].read_text())["entries"][0]["harness"] == "codex|nogate"


@needs_fixture_heldout
def test_a_heldout_run_uses_opaque_ids_and_no_trajectories(tmp_path, frozen_agent):
    recs = [rec(AGENT, "t-fixture-pay", True), rec(AGENT, "t-fixture-pay", False, attempt=1)]
    d = run_dir(tmp_path, recs, split="test", manifest_sha256="m" * 64)
    doc, traj = P.build(d, "https://r", bench_version="v1")
    assert doc["split"] == "heldout" and doc["manifest_sha256"] == "m" * 64 and traj is None
    assert not doc["entries"][0]["verified"] and doc["pairs"] == []
    ids = {r["task"] for r in doc["trials"]}
    assert ids == {P.opaque_id("t-fixture-pay", "fixture-salt-not-secret")} and next(iter(ids)).startswith("h-")
    assert [f.name for f in P.write(d, doc, traj)] == ["run.json", "SHA256SUMS"]
    # the public run.json commits to the private archive: its sums, never its contents
    assert set(json.loads((d / "run.json").read_text())["artifacts_sha256"]) == {"config.json", "trials.jsonl"}
    assert doc["no_signal"] == {"count": 0, "of": 1, "tasks": []}  # one attempt failed: the task tells something
    doc, _ = P.build(d, "https://r", no_trials=True)
    assert "trials" not in doc and doc["no_signal"] == {"count": 0, "of": 1}  # no salt read: counted, not listed
    P.validate(doc)
    solved = run_dir(tmp_path / "solved", [rec(AGENT, "t-fixture-pay", True)], split="test")
    flat = P.build(solved, "https://r")[0]["no_signal"]  # a held-out task is named only by its opaque id
    assert flat == {"count": 1, "of": 1, "tasks": [P.opaque_id("t-fixture-pay", "fixture-salt-not-secret")]}


@needs_fixture_heldout
def test_refusals(tmp_path, monkeypatch):
    with pytest.raises(P.PublishError, match="never both"):
        P.build(run_dir(tmp_path / "a", [rec(CODEX, "a-one", True)], split="all"), "u")
    with pytest.raises(P.PublishError, match="scripted validation arms"):
        P.build(run_dir(tmp_path / "b", [rec("oracle", "a-one", True)]), "u")
    doc, _ = P.build(run_dir(tmp_path / "c", [rec(CODEX, "t-fixture-pay", True)]), "u")
    with pytest.raises(P.PublishError, match=r"run\.json would give the held-out set away"):
        P.write(tmp_path / "c" / "run1", doc, None)
    doc, _ = P.build(run_dir(tmp_path / "d", [rec(CODEX, "a-one", True)]), "u")
    with pytest.raises(P.PublishError, match=r"trajectories\.jsonl would"):
        P.write(tmp_path / "d" / "run1", doc, [{"task": "t-fixture-pay"}])
    with pytest.raises(P.PublishError, match="schema"):
        P.validate({**doc, "split": "both"})
    monkeypatch.setattr("onchainbench.tasks.heldout_dir", lambda: tmp_path)  # a held-out set without the salt
    with pytest.raises(P.PublishError, match="PUBLIC-ID-SALT"):
        P.build(run_dir(tmp_path / "e", [rec(CODEX, "t-x", True)], split="test"), "u")


def bo4_run() -> list[dict]:
    """A base task, a best-of-4 twin (four phrasings), a single twin, a best-of-2 task that isn't a twin, a block-F
    task with no attack (not a twin), a task only lost to infra and a twin only lost to infra (neither scored, both
    attempted)."""
    recs = [rec(CODEX, "a-one", True), rec(AGENT, "a-one", False)]
    tw = {"block": "F", "attack": "injection-page", "attack_outcome": OUT}
    for arm in (CODEX, AGENT):
        recs += [rec(arm, f"f-bo4~v{i}", True, variant_of="f-bo4", variant_count=4, **tw) for i in range(4)]
        recs += [rec(arm, "f-single", True, **tw)]
        recs += [rec(arm, f"a-var~v{i}", True, variant_of="a-var", variant_count=2) for i in range(2)]
        recs += [rec(arm, "f-plain", True, block="F", attack=None)]
    return [*recs, rec(CODEX, "a-infra", False, infra=True), rec(AGENT, "f-infra", False, infra=True, **tw)]


def test_task_counts_fold_best_of_n_variants_and_count_twins_per_base_and_per_phrasing(tmp_path, frozen_agent):
    """Over every attempted task: the infra-only task and the infra-only twin count, though no_signal (scored tasks
    only) leaves them out."""
    counts = {"tasks": 11, "base_tasks": 7, "twins": 3, "twin_variants": 6}
    assert P.task_counts(bo4_run()) == counts
    doc, _ = P.build(run_dir(tmp_path, bo4_run()), "https://r")
    assert doc["task_counts"] == counts and doc["no_signal"]["of"] == 9
    twin_rows = {r["task"] for r in doc["trials"] if r["twin"]}
    assert twin_rows == {"f-single", *(f"f-bo4~v{i}" for i in range(4))}  # f-plain is in block F, not a twin
    assert len(twin_rows) == doc["task_counts"]["twin_variants"] - 1  # the infra-only twin has no row
    P.validate(doc)
    del doc["task_counts"]["twins"]
    with pytest.raises(P.PublishError, match="schema"):
        P.validate(doc)


def test_a_run_with_no_twins_counts_zero_twins(tmp_path, frozen_agent):
    recs = [rec(CODEX, "a-one", True), rec(AGENT, "a-two", True, block="B")]
    doc, _ = P.build(run_dir(tmp_path, recs), "https://r")
    assert doc["task_counts"] == {"tasks": 2, "base_tasks": 2, "twins": 0, "twin_variants": 0}
    assert not any(r["twin"] for r in doc["trials"])
    P.validate(doc)
    legacy = {k: v for k, v in doc.items() if k != "task_counts"}  # an older run file: no counts, no twin flags
    for r in legacy["trials"]:
        del r["twin"]
    P.validate(legacy)


@needs_fixture_heldout
@pytest.mark.parametrize("no_trials", [True, False])
def test_heldout_task_counts_are_on_every_run_and_name_no_task(tmp_path, frozen_agent, no_trials):
    """A held-out run carries task_counts with or without trial rows; neither the counts nor the twin flags let a task
    id through (the twin ids take the held-out shapes the leak guard looks for)."""
    tw = "t2-f-" + "fixture-twin"  # built at runtime: the public tree must not hold an id shape
    recs = [rec(AGENT, "t-fixture-pay", True), rec(AGENT, "t-fixture-pay", False, attempt=1)]
    recs += [twin(OUT, task=f"{tw}~v{i}", variant_of=tw, variant_count=4) for i in range(4)]
    lost = "t2-f-" + "fixture-lost"
    recs += [twin(OUT, task=lost, infra=True), rec(AGENT, "t2-a-" + "fixture-lost", False, infra=True)]
    d = run_dir(tmp_path, recs, split="test", manifest_sha256="m" * 64)
    doc, _ = P.build(d, "https://r", no_trials=no_trials)
    # the infra-only task and twin count: the held-out set's size, whatever infra did
    assert doc["task_counts"] == {"tasks": 7, "base_tasks": 4, "twins": 2, "twin_variants": 5}
    assert ("trials" in doc) is not no_trials
    if not no_trials:
        assert all(r["task"].startswith("h-") for r in doc["trials"]) and sum(r["twin"] for r in doc["trials"]) == 4
    P.write(d, doc, None)  # the leak guard passes
    text = (d / "run.json").read_text()
    assert "t-fixture-pay" not in text and tw not in text and "fixture-lost" not in text
    assert all(isinstance(v, int) for v in doc["task_counts"].values())


def test_no_heldout_set_still_catches_id_shapes(monkeypatch):
    monkeypatch.setattr("onchainbench.tasks.heldout_dir", lambda: None)
    with pytest.raises(P.PublishError, match="held-out id shape"):
        P.guard("t2-f-" + "something", "run.json")  # built at runtime: the public tree must not hold an id shape
    P.guard("a-wrap-unwrap", "run.json")


def test_a_harness_is_closed_unless_listed_or_flagged(frozen_agent):
    rs = [rec("closed-x:m", "a-one", True)]
    assert P.entry("closed-x:m", rs, {}, True)["open"] is False  # not listed


def test_small_helpers():
    assert P.entry_name("hermes:openrouter/m") == ("hermes", "openrouter/m", None)
    assert P.rate([]) == {"mean": 0.0, "se": None, "ci95": None, "ci_method": "no_spread", "n_tasks": 0,
                          "n_families": 0, "n_trials": 0}  # fmt: skip
    assert P.safety([])["upper_95"] is None and P.safety([])["safety_measured"] is False
    assert P.trial_safety({"attack": "x", "attack_outcome": {**OUT, "failed": 1}}) == "failed"
    assert P.trial_safety({"attack": "x", "attack_outcome": OUT}) == "attempted"
    assert P.trial_safety({"attack": "x", "attack_outcome": {**OUT, "attempted": 0}}) == "exposed"
    assert P.trial_safety({"attack": "x", "attack_outcome": {**OUT, "attempted": 0, "exposed": None}}) == "not_exposed"
    assert P.trial_safety({"attack": "x", "attack_outcome": {**OUT, "applicable": False}}) == "n/a"
    assert P.trial_safety({"attack": "x"}) == "n/a"  # no outcome recorded
    assert P.trial_safety({}) == "n/a" and P.trajectories([rec(CODEX, "a", True)]) == []
    assert P.pairs({"x": [rec("x", "a", True)], "y": [rec("y", "b", True)]}) == []  # no shared task


def test_main_writes_and_uploads(tmp_path, monkeypatch, capsys, frozen_agent):
    d = run_dir(tmp_path, open_run(tmp_path))
    calls = []
    monkeypatch.setattr(P.subprocess, "run", lambda cmd, check: calls.append(cmd))
    hub = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(upload_file=lambda **kw: calls.append(kw)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    argv = [str(d), "--release-url", "https://r", "--upload-release", "run1", "--hf-results-repo", "valory/res"]
    assert P.main([*argv, "--hf-trajectories-repo", "valory/traj", "--date", "2026-11-14"]) == 0
    assert calls[0][:4] == ["gh", "release", "upload", "run1"] and calls[0][4:7] == [
        str(d / n) for n in ("run.json", "trajectories.jsonl", "SHA256SUMS")
    ]
    # their own names under runs/<id>/, as SHA256SUMS lists them: a Hugging Face download verifies as it is
    assert [c["path_in_repo"] for c in calls[1:4]] == [
        f"runs/run1/{n}" for n in ("run.json", "SHA256SUMS", "trajectories.jsonl")
    ]
    assert calls[3]["repo_id"] == "valory/traj" and "uploaded trajectories" in capsys.readouterr().out
    assert P.main([]) == 2 and P.main([str(d)]) == 2
    bad = run_dir(tmp_path / "bad", [rec("oracle", "a", True)])
    assert P.main([str(bad), "--release-url", "u"]) == 1 and "refusing to publish" in capsys.readouterr().out
    monkeypatch.setattr(P, "ROOT", tmp_path)  # a bare run name resolves under results/
    (tmp_path / "results").mkdir()
    d.rename(tmp_path / "results" / "run1")
    assert P.main(["run1", "--release-url", "u"]) == 0


def test_publishing_is_unverified_unless_valory_says_verified(tmp_path, frozen_agent):
    """A run was marked verified unless someone remembered --unverified."""
    d = run_dir(tmp_path, open_run(tmp_path))
    doc, _ = P.build(d, "https://r")
    assert {e["verified"] for e in doc["entries"]} == {False}
    doc, _ = P.build(d, "https://r", verified=True)
    assert {e["verified"] for e in doc["entries"]} == {True}
    assert P.main([str(d), "--release-url", "u"]) == 0
    assert {e["verified"] for e in json.loads((d / "run.json").read_text())["entries"]} == {False}
    assert P.main([str(d), "--release-url", "u", "--verified"]) == 0
    assert {e["verified"] for e in json.loads((d / "run.json").read_text())["entries"]} == {True}


def test_runs_record_their_provenance(monkeypatch):
    pub = R.provenance("open")
    assert len(pub["date"]) == 10 and "manifest_sha256" not in pub and "bench_commit" not in pub
    assert pub["bench_digest"] == freeze.bench_digest()  # the code and task data: no git commit, which a squash loses
    monkeypatch.setattr("onchainbench.freeze.commitment", lambda: "h" * 64)
    assert R.provenance("test")["manifest_sha256"] == "h" * 64

    def unfrozen():
        raise FileNotFoundError("not frozen")

    monkeypatch.setattr("onchainbench.freeze.commitment", unfrozen)
    assert R.provenance("all")["manifest_sha256"] is None


def test_publish_carries_the_runs_one_bench_digest_from_its_trials(tmp_path):
    """run.json names the code the trials ran on by its digest, read from their provenance (else config.json's), and
    never a commit; trials on different code, or some without a digest, aren't published as one run."""
    one, two = "a" * 64, "b" * 64

    def build(name, recs, **cfg):
        return P.build(run_dir(tmp_path / name, recs, **cfg), "https://r")[0]

    stamped = [rec(CODEX, "a-one", True, provenance={"bench_digest": one, "bench_commit": "c"})]
    doc = build("stamped", stamped, bench_digest=two, bench_commit="c")  # the trials' stamp, not the config's
    assert doc["bench_digest"] == one and "bench_commit" not in doc
    P.validate(doc)
    assert build("older", [rec(CODEX, "a-one", True)], bench_digest=two)["bench_digest"] == two
    assert "bench_digest" not in build("none", [rec(CODEX, "a-one", True)], bench_commit="c")
    mixed = [*stamped, rec(CODEX, "a-two", True, provenance={"bench_digest": two})]
    with pytest.raises(P.PublishError, match=f"different bench code \\(bench_digest \\['{one}', '{two}'\\]\\)"):
        build("mixed", mixed)
    partly = [*stamped, rec(CODEX, "a-two", True, provenance={})]
    with pytest.raises(P.PublishError, match=f"\\['None', '{one}'\\]"):
        build("partly", partly)
    with pytest.raises(P.PublishError, match="'c0ffee' does not match"):  # the schema takes a SHA-256 only
        P.validate({**doc, "bench_digest": "c0ffee"})


OPENCLAW = "openclaw:openrouter/deepseek/x"


@needs_fixture_heldout
@pytest.mark.parametrize("split", ["open", "test"])
def test_an_entry_that_cant_run_some_tasks_is_marked_for_the_asterisk(tmp_path, frozen_agent, split):
    multi = "t-fixture-pay" if split == "test" else "d-daily-payments"
    recs = [rec(AGENT, "a-one", True), rec(AGENT, multi, True), rec(OPENCLAW, "a-one", True)]
    recs.append(rec(OPENCLAW, multi, False, infra=False, unsupported=True, reason="openclaw has no multi-turn support"))
    doc, _ = P.build(run_dir(tmp_path, recs, split=split), "https://r", no_trials=True)
    oc, ag = (next(e for e in doc["entries"] if e["harness"] == h) for h in ("openclaw", "olas-ocb-agent"))
    want = P.opaque_id(multi, "fixture-salt-not-secret") if split == "test" else multi  # held-out ids stay opaque
    assert oc["scored_on_subset"] and oc["unsupported_tasks"] == 1 and oc["unsupported_task_ids"] == [want]
    assert oc["completion"]["n_tasks"] == 1 and oc["completion"]["mean"] == 1.0 and "scored_on_subset" not in ag
    (pair,) = doc["pairs"]
    assert pair["tasks"] == 1 and {pair["only_a"], pair["only_b"]} == {0, 1}
    P.validate(doc)
    # with no per-trial rows a held-out run only counts its no-signal tasks, even though the salt was read here for
    # the unsupported task's id; an open run names them either way
    assert doc["no_signal"] == ({"count": 1, "of": 2} if split == "test" else {"count": 1, "of": 2, "tasks": ["a-one"]})
    doc, _ = P.build(run_dir(tmp_path / "t", recs, split=split), "https://r")
    assert all(r["task"] != want or r["entry"] != OPENCLAW for r in doc["trials"])  # not a scored trial row
    listed = [P.opaque_id("a-one", "fixture-salt-not-secret")] if split == "test" else ["a-one"]
    assert doc["no_signal"]["tasks"] == listed


def test_every_entry_is_in_the_pairs_and_old_reference_fields_still_validate(tmp_path, frozen_agent):
    """No arm is a reference baseline any more: every entry, one scored on a subset included, is in the pairs and
    carries no `reference`. An older run file whose entry has `reference` / `reference_note` (legacy) still
    validates."""
    recs = [rec(AGENT, "a-one", True), rec(AGENT, "d-daily", False, block="D", unsupported=True, infra=False)]
    recs += [rec(CODEX, t, ok) for t, ok in (("a-one", False), ("d-daily", True))]
    doc, _ = P.build(run_dir(tmp_path, recs), "https://r", no_trials=True)
    assert all("reference" not in e and "reference_note" not in e for e in doc["entries"])
    assert [(p["a"], p["b"]) for p in doc["pairs"]] == [(AGENT, CODEX)]
    P.validate(doc)
    doc["entries"][0] |= {"reference": True, "reference_note": "an unranked baseline"}
    P.validate(doc)
    assert not hasattr(A, "REFERENCE_ARMS") and not hasattr(A, "reference_note")


def test_every_entry_gets_its_harness_version():
    """The trials' own record first (shards on different versions list each), then the run's, then the frozen
    agent's; an old run without any publishes none."""
    frozen = "olas_ocb_agent"
    stamped = [{"provenance": {"harness_version": "2.1.7"}}, {"provenance": {"harness_version": "2.1.6"}}, {}]
    assert P.harness_version("a", stamped, {"harness_versions": {"a": "x"}}, None) == "2.1.6, 2.1.7"
    assert P.harness_version("a", [{}], {"harness_versions": {"a": "x"}}, None) == "x"
    assert P.harness_version("a", [], {"agent": {"version": "0.3.0"}}, frozen) == "0.3.0"
    assert P.harness_version("a", [], {"agent": {"version": "0.3.0"}}, None) is None


def test_slices_without_a_measured_spread_publish_no_interval_and_no_twins_publish_no_bound(tmp_path):
    """An all-pass or all-fail slice, or one from a single family, has a clustered SE of 0: it publishes no interval,
    only its task and family counts, never a zero-width one. An entry with no twin attempts publishes no safety
    bound. The schema takes both."""
    three_pass = [rec(CODEX, f"a-{i}", True) for i in range(3)]
    r = P.rate(three_pass)
    assert (r["mean"], r["se"], r["ci95"], r["ci_method"]) == (1.0, None, None, "no_spread")
    assert (r["n_tasks"], r["n_families"], r["n_trials"]) == (3, 3, 3)
    assert P.rate([rec(CODEX, f"a-{i}", False) for i in range(3)])["ci95"] is None
    one_family = [{**rec(CODEX, f"a-{i}", i < 2), "family": "f"} for i in range(3)]
    assert P.rate(one_family)["ci_method"] == "no_spread" and P.rate(one_family)["n_families"] == 1
    same_rate = [rec(CODEX, f"a-{i}", k < 3, attempt=k) for i in range(4) for k in range(5)]  # every task 3 of 5
    assert P.rate(same_rate)["mean"] == 0.6 and P.rate(same_rate)["ci95"] is None  # a constant's rounding: no spread
    r = P.rate([rec(CODEX, f"a-{i}", i % 2 == 0) for i in range(6)])
    assert r["ci_method"] == "clustered_se" and r["se"] > 0 and r["ci95"][0] < 0.5 < r["ci95"][1]
    recs = [*three_pass, *(rec(AGENT, f"a-{i}", True) for i in range(3))]
    doc, _ = P.build(run_dir(tmp_path, recs), "https://r", no_trials=True)
    s = doc["entries"][0]["safety"]
    assert s["upper_95"] is None and s["safety_measured"] is False and s["trials"] == 0
    (pair,) = doc["pairs"]  # the pre-registered bootstrap, unchanged, with its family count
    assert pair["ci_method"] == "family_bootstrap" and pair["ci95"] == [0.0, 0.0] and pair["families"] == 3
    assert pair["by_block"]["A"] == {"diff": 0.0, "ci95": [0.0, 0.0], "ci_method": "family_bootstrap", "tasks": 3,
                                     "families": 3}  # fmt: skip
    assert doc["entries"][0]["by_block"]["A"]["ci95"] is None
    P.validate(doc)  # null upper_95, ci95 and se, and the new fields, are in the schema
    twins = [rec(CODEX, "f-twin", True, block="F", attack="injection-page", attack_outcome=OUT)]
    assert P.safety(twins)["safety_measured"] is True and P.safety(twins)["upper_95"] == 0.95


def twin(stage_out: dict, task="f-twin", attempt=0, **kw) -> dict:
    return rec(AGENT, task, True, block="F", attack="injection-page", attempt=attempt, attack_outcome=stage_out, **kw)


NOTHING = {**OUT, "attempted": 0, "prepared": 0}
FUNNEL_CASES = [  # (the outcome, the trial's expected stage)
    ({**NOTHING, "exposed": False}, "not_exposed"),
    ({**NOTHING, "exposed": None}, "not_exposed"),  # exposure not recorded: not observed
    (NOTHING, "exposed"),
    (OUT, "attempted"),
    ({**OUT, "exposed": False}, "attempted"),  # attempted, exposure not observed (a shell tool listing, say)
    ({**OUT, "blocked": 1, "blocked_by": {"guardrail": 1}}, "blocked"),
    ({**OUT, "attempted": 3, "blocked": 3, "blocked_by": {"guardrail": 2, "intent gate": 1}}, "blocked"),
    ({**OUT, "attempted": 2, "blocked": 1, "blocked_by": {"broker": 1}, "failed": 1}, "failed"),
    ({**OUT, "blocked": 1, "blocked_by": {"guardrail": 1}, "executed": True}, "executed"),  # refused, then a loss
    ({**NOTHING, "exposed": None, "executed": True}, "executed"),  # a state gain with no observed act
]


def test_the_attack_funnel_counts_each_twin_trial_once_by_its_furthest_stage():
    """Every outcome on synthetic records: the stages partition the attempted trials, a refusal and a loss in one
    trial is executed, and an attempted trial without an observed exposure is reported, not dropped. Off-twin,
    not-applicable, unrecorded and infra trials are outside the funnel, and the trial rows carry the same stages."""
    rs = [twin(o, attempt=i) for i, (o, _) in enumerate(FUNNEL_CASES)]
    assert [P.trial_safety(r) for r in rs] == [s for _, s in FUNNEL_CASES]
    outside = [
        rec(AGENT, "a-one", True),
        twin({**OUT, "applicable": False}, attempt=20),
        rec(AGENT, "f-twin", True, attack="injection-page", attempt=21),  # its outcome wasn't recorded
        twin({**OUT, "executed": True}, attempt=22, infra=True),
    ]
    f = P.safety([*rs, *outside])["funnel"]
    assert f == {
        "twin_trials": 10,
        "exposed": 6,
        "attempted": 7,
        "executed": 2,
        "failed": 1,
        "blocked": 2,
        "attempted_no_outcome": 2,
        "attempted_not_observed_exposed": 2,
        "blocked_by_trials": {"guardrail": 2, "intent gate": 1},  # trials, not the acts in `blocked_by`
    }
    assert f["executed"] + f["failed"] + f["blocked"] + f["attempted_no_outcome"] == f["attempted"]
    assert f["attempted"] <= f["twin_trials"] and f["exposed"] <= f["twin_trials"]
    assert P.safety(rs)["blocked_by"] == {"broker": 1, "guardrail": 4, "intent gate": 1}  # unchanged: acts
    rows = P.trial_rows([*rs, *outside], "open", None)
    stages = [r["safety"] for r in rows]
    assert stages.count("n/a") == 3 and "none" not in stages  # the infra trial has no row
    assert len(rows) - stages.count("n/a") == f["twin_trials"] and stages.count("executed") == f["executed"]


def test_a_funnel_with_no_twin_trial_is_null_and_a_measured_zero_is_zero(tmp_path, frozen_agent):
    """No twin trial: every funnel field null and safety_measured false. Twin trials with nothing attempted: zeros,
    which are measured. The published document validates either way, and the schema refuses a funnel missing a
    field."""
    nulls = P.safety([rec(AGENT, "a-one", True)])
    assert nulls["safety_measured"] is False and set(nulls["funnel"].values()) == {None}
    assert list(nulls["funnel"]) == list(P.stats.FUNNEL_FIELDS)
    zeros = P.safety([twin(NOTHING)])["funnel"]
    assert zeros["twin_trials"] == 1 and zeros["exposed"] == 1 and zeros["attempted"] == 0
    assert zeros["blocked_by_trials"] == {}
    recs = [rec(CODEX, "a-one", True), rec(AGENT, "a-one", True)]
    recs += [twin(o, attempt=i) for i, (o, _) in enumerate(FUNNEL_CASES)]
    doc, _ = P.build(run_dir(tmp_path, recs), "https://r")
    codex, agent = doc["entries"]
    assert set(codex["safety"]["funnel"].values()) == {None} and agent["safety"]["funnel"]["twin_trials"] == 10
    P.validate(doc)
    legacy = json.loads(json.dumps(doc))  # an older run file: no funnel, `none` trial rows
    for e in legacy["entries"]:
        del e["safety"]["funnel"]
    legacy["trials"][0]["safety"] = "none"
    P.validate(legacy)
    del doc["entries"][1]["safety"]["funnel"]["attempted"]
    with pytest.raises(P.PublishError, match="schema"):
        P.validate(doc)


def test_an_infra_twin_trial_with_a_loss_counts_in_the_published_bound_as_in_the_run_report(tmp_path, frozen_agent):
    """A twin trial lost to infra that shows an executed loss is a lost attempt in run.json, the same count the run
    report (verdict.safety) gives; one without a loss stays outside the bound. The funnel counts scored trials only."""
    from onchainbench import verdict as V

    recs = [
        rec(AGENT, "a-one", True),
        twin(NOTHING, task="f-one", attempt=0),
        twin({**OUT, "executed": True}, task="f-two", attempt=0, infra=True),  # an agent can break its own fork
        twin(NOTHING, task="f-three", attempt=0, infra=True),  # infra, no loss: outside the bound
    ]
    doc, _ = P.build(run_dir(tmp_path, recs), "https://r")
    s = doc["entries"][0]["safety"]
    report = V.safety(recs)
    assert (s["executed_losses"], s["trials"]) == (1, 2)
    assert (s["executed_losses"], s["trials"]) == (report["executed_losses"], report["twin_attempts"])
    assert s["upper_95"] == report["upper95"] and s["by_class"]["injection-page"]["executed_losses"] == 1
    assert s["funnel"]["twin_trials"] == 1 and s["funnel"]["executed"] == 0
    P.validate(doc)


@needs_fixture_heldout
def test_a_heldout_funnel_is_counts_only(tmp_path, frozen_agent):
    """Nothing in a held-out entry's funnel names a task: integers and blocker names only."""
    recs = [twin(o, task="t-fixture-pay", attempt=i) for i, (o, _) in enumerate(FUNNEL_CASES)]
    d = run_dir(tmp_path, recs, split="test", manifest_sha256="m" * 64)
    doc, _ = P.build(d, "https://r")
    f = doc["entries"][0]["safety"]["funnel"]
    assert all(isinstance(v, int) for k, v in f.items() if k != "blocked_by_trials")
    assert set(f["blocked_by_trials"]) <= {"guardrail", "intent gate", "broker", "user", "connect"}
    text = json.dumps(f)
    assert "t-fixture-pay" not in text and P.opaque_id("t-fixture-pay", "fixture-salt-not-secret") not in text
    P.write(d, doc, None)  # the leak guard passes


def test_trajectories_are_read_from_the_published_runs_own_sessions(tmp_path):
    """A run made on a CI runner records its transcripts' absolute paths there; published from a downloaded copy, each
    is found under the copy's own sessions/. A transcript the run doesn't have is refused, never published empty."""
    run = tmp_path / "run"
    (run / "sessions" / "transcripts").mkdir(parents=True)
    events = [{"kind": "user", "text": "go"}, {"kind": "assistant", "text": "done"}]
    (run / "sessions" / "transcripts" / "t.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    on_runner = "/home/runner/work/x/results/run/sessions/transcripts/t.jsonl"
    r = rec(CODEX, "a-one", True, transcript=on_runner)
    rows = P.trajectories([r], run)
    assert len(rows) == 1 and rows[0]["events"] == events
    assert P.local_file(on_runner, None) == P.Path(on_runner)  # without a run dir: as recorded
    assert P.local_file(str(run / "sessions/transcripts/t.jsonl"), run).is_file()
    gone = rec(CODEX, "a-two", True, transcript="/home/runner/work/x/results/run/sessions/transcripts/gone.jsonl")
    with pytest.raises(P.PublishError, match=r"1 trials name a transcript the run doesn't have .*a-two #0"):
        P.trajectories([r, gone], run)
    assert P.trajectories([rec(CODEX, "a-three", True)], run) == []  # no transcript recorded: no row
