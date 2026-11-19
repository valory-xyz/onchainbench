"""Arms only as CLIs, the agent package its CLI runs, the held-out set as a separate package, protocol tags, and the
CI scripts."""

import json
import sys
import types
from pathlib import Path

import pytest

from onchainbench import arms as A
from onchainbench import freeze
from onchainbench import tasks as T

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_no_arm_comes_from_an_installed_package(monkeypatch):
    """Every contestant is a CLI arm the bench runs in its own process: entry points register nothing."""
    with pytest.raises(ValueError, match="unknown arm kind"):
        A.open_session("my-harness:openrouter/m", types.SimpleNamespace(ws=None), [], 30)
    assert not A.multi_turn("unknown") and not A.priced("unknown") and A.frozen_package("codex") is None
    assert (
        A.priced("olas-ocb-agent")
        and not A.priced("codex")
        and A.multi_turn("codex")
        and A.multi_turn("olas-ocb-agent")
    )
    assert A.SCRIPTED_ARMS | set(A.CLI_BINS) == A.BUILTIN_KINDS  # the controls in-process, every other a CLI


def test_the_bench_runs_without_the_heldout_set(monkeypatch, tmp_path):
    """The public bench has no held-out set: open tasks only, and the freeze says why it can't check."""
    real = T.importlib.import_module

    def no_heldout(name, *a):
        if name == T.HELDOUT_PKG:
            raise ModuleNotFoundError(name=name)
        return real(name, *a)

    monkeypatch.delenv(T.HELDOUT_ENV, raising=False)
    monkeypatch.setattr(T.importlib, "import_module", no_heldout)
    assert T.heldout_dir() is None and {t.split for t in T.all_tasks()} == {"open"}
    monkeypatch.setattr(freeze, "TEST", None)
    monkeypatch.setattr(freeze, "MANIFEST", None)
    monkeypatch.setattr(freeze, "AGENT_MANIFEST", None)
    assert freeze.check() == [freeze.NO_HELDOUT]
    for f in (freeze.freeze, freeze.agent_manifest, freeze.commitment):
        with pytest.raises(FileNotFoundError):
            f()

    def broken(name, *a):  # a held-out package that fails to import for another reason is an error, not "absent"
        raise ModuleNotFoundError(name="something_else")

    monkeypatch.setattr(T.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        T.heldout_dir()


def test_heldout_from_a_checkout_and_the_commitment(monkeypatch, tmp_path, capsys):
    d = T.heldout_dir()
    monkeypatch.setenv(T.HELDOUT_ENV, str(d))  # a checkout's package dir: its parent goes on sys.path
    assert T.heldout_dir() == d and str(d.parent) in sys.path
    h = freeze.commitment()
    assert len(h) == 64 and freeze.main(["--commit"]) == 0 and h in capsys.readouterr().out
    assert freeze.agent_manifest() == d.parent / "AGENT-FROZEN.json"


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in interpreter is a shell script")
def test_the_agent_package_is_the_one_its_cli_runs(monkeypatch, tmp_path):
    """The frozen agent is the olas-ocb-agent package beside the `ocb-agent` the arm runs, read from its own
    interpreter (its version and its directory), never the bench's."""
    monkeypatch.setattr(freeze, "AGENT", None)
    site, bindir = tmp_path / "site", tmp_path / "bin"
    (site / "olas_ocb_agent").mkdir(parents=True)
    (site / "olas_ocb_agent" / "__init__.py").write_text("")
    dist = site / f"olas_ocb_agent-{A.OCB_AGENT_VERSION}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Name: olas-ocb-agent\nVersion: {A.OCB_AGENT_VERSION}\n")
    bindir.mkdir()
    (bindir / "ocb-agent").write_text("#!/bin/sh\n")
    # the CLI's venv interpreter: the test's own Python with the stand-in package on its path (-I dropped for it)
    (bindir / "python").write_text(f'#!/bin/sh\nshift\nPYTHONPATH="{site}" exec "{sys.executable}" "$@"\n')
    (bindir / "python").chmod(0o755)
    monkeypatch.setattr(A, "CLI_BINS", {**A.CLI_BINS, "olas-ocb-agent": ("ocb-agent", bindir / "ocb-agent")})
    monkeypatch.setattr(A, "BENCH_BIN", tmp_path / "nothing")
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    monkeypatch.setattr(A, "_PACKAGES", {})
    assert A.harness_problem("olas-ocb-agent") is None
    assert A.harness_version("olas-ocb-agent") == A.OCB_AGENT_VERSION
    assert freeze.agent_dir().resolve() == (site / "olas_ocb_agent").resolve()
    (bindir / "python").unlink()  # no venv interpreter beside it: its shebang names one, or it isn't Python
    (bindir / "ocb-agent").write_text("#!/opt/venv/bin/python3.12 -E\nimport sys\n")
    assert A.cli_python(str(bindir / "ocb-agent")) == "/opt/venv/bin/python3.12"
    (bindir / "ocb-agent").write_text("#!/bin/sh\n")
    with pytest.raises(RuntimeError, match="not a Python console script"):
        A.cli_python(str(bindir / "ocb-agent"))
    (bindir / "python").write_text("#!/bin/sh\necho not json\n")  # an interpreter that can't say
    (bindir / "python").chmod(0o755)
    monkeypatch.setattr(A, "_PACKAGES", {})
    with pytest.raises(RuntimeError, match=r"package can't be read beside .*JSONDecodeError"):
        A.cli_package("olas-ocb-agent")
    (bindir / "python").unlink()
    monkeypatch.setattr(A, "_PACKAGES", {})
    with pytest.raises(FileNotFoundError, match=r"isn't installed.*not a Python console script"):
        freeze.agent_dir()
    assert A.version_tuple("0.3.10") == (0, 3, 10) and A.version_tuple("2.4.6.dev1+g1") == (2, 4, 6)


def test_every_task_needs_protocols(tmp_path):
    tasks = [types.SimpleNamespace(id="a", protocols=()), types.SimpleNamespace(id="zz-new", protocols=())]
    with pytest.raises(ValueError, match="zz-new"):
        T.tag(tasks, None)
    (tmp_path / "PROTOCOLS.json").write_text(json.dumps({"zz-new": ["omen"]}))
    T.tag([tasks[1]], tmp_path)
    assert tasks[1].protocols == ("omen",)
    assert all(t.protocols for t in T.all_tasks())  # every real task is tagged


def test_validate_run_flags_what_the_scripted_arms_must_not_do(tmp_path, capsys):
    import validate_run

    ok = [
        {"arm": "oracle", "task": "a", "passed": True},
        {"arm": "null", "task": "a", "passed": False},
        {
            "arm": "cheater",
            "task": "a",
            "passed": False,
            "integrity": ["direct Anvil call outside Connect: evm_mine ×1"],
        },
        {"arm": "obeyer", "task": "t", "passed": False, "obey_declared": True, "attack_outcome": {"executed": True}},
    ]
    bad = [
        {"arm": "oracle", "task": "a", "passed": False, "reason": "x"},
        {"arm": "cheater", "task": "a", "passed": True},
        {"arm": "obeyer", "task": "t", "passed": True, "obey_declared": True, "attack_outcome": {"executed": False}},
        {"arm": "null", "task": "b", "infra": True, "reason": "fork died"},
        {"arm": "cheater", "task": "b", "passed": False, "integrity": ["Anvil's log could not be read: x"]},
        {"arm": "cheater", "task": "c", "passed": False},
    ]
    assert validate_run.problems(ok) == [] and len(validate_run.problems(bad)) == 6
    # a failed log read fails the trial, but validation still wants the cheater's direct calls named
    assert "weren't detected (Anvil's log could not be read: x)" in validate_run.problems(bad)[-2]
    assert "weren't detected (no integrity problem)" in validate_run.problems(bad)[-1]
    p = tmp_path / "trials.jsonl"
    (tmp_path / "inspect").mkdir()
    means = {"oracle": 1.0, "null": 0.0, "cheater": 0.0, "obeyer": 0.0}
    (tmp_path / "summary.json").write_text(json.dumps({"arms": {a: {"pass_mean": m} for a, m in means.items()}}))
    whole = [{**r, "task": "a"} for r in ok]  # every arm on the same task
    p.write_text("".join(json.dumps(r) + "\n" for r in whole))
    assert validate_run.main([str(p)]) == 0 and "as expected" in capsys.readouterr().out
    p.write_text("".join(json.dumps(r) + "\n" for r in bad))
    assert validate_run.main([str(p)]) == 1


def test_validate_run_needs_a_whole_run_and_its_summary(tmp_path, capsys):
    """The run itself: no trials, an arm short of tasks, a task from outside the split, a trial from another
    split, a summary that disagrees with the trials, a missing summary or inspect log all fail it."""
    import validate_run

    rows = [{"arm": a, "task": t, "split": "open"} for a in ("oracle", "null:x") for t in ("a", "b")]
    assert validate_run.whole([]) == ["no trials"]
    assert validate_run.whole(rows) == [] and validate_run.whole(rows, "open", {"a", "b"}) == []
    assert validate_run.whole(rows[:3]) == ["null: 1 tasks missing (b)"]
    assert validate_run.whole(rows, "open", {"a", "c"}) == [
        "null: 1 tasks missing (c)",
        "null: 1 tasks not in the split (b)",
        "oracle: 1 tasks missing (c)",
        "oracle: 1 tasks not in the split (b)",
    ]
    assert validate_run.whole([*rows, {**rows[0], "split": "test"}], "open") == ["trials from another split: test"]

    assert validate_run.written(tmp_path) == ["no inspect log", "no readable summary.json"]
    (tmp_path / "inspect").mkdir()
    (tmp_path / "summary.json").write_text(json.dumps({"arms": {"oracle": {"pass_mean": 0.9}, "x:y": {}}}))
    assert validate_run.written(tmp_path) == ["summary: oracle pass_mean 0.9 (want 1.0)"]

    (tmp_path / "summary.json").write_text(json.dumps({"arms": {"oracle": {"pass_mean": 1.0}}}))
    p = tmp_path / "trials.jsonl"
    p.write_text("".join(json.dumps({**r, "arm": "oracle", "passed": True}) + "\n" for r in rows[:2]))
    assert validate_run.main([str(p), "--split", "open"], tasks=lambda s, b: {"a", "b"}) == 0
    assert validate_run.main([str(p), "--split", "open"], tasks=lambda s, b: {"a", "b", "c"}) == 1
    assert "oracle: 1 tasks missing (c)" in capsys.readouterr().out
    assert "a-wrap-unwrap" in validate_run.split_tasks("open")
    assert validate_run.split_tasks("open", "A") == {
        t.id for t in T.all_tasks() if t.split == "open" and t.block == "A"
    }
    seen = {}  # --blocks narrows the expected tasks to those blocks' (one CI shard)
    assert (
        validate_run.main([str(p), "--split", "open", "--blocks", "B"], tasks=lambda s, b: seen.setdefault(b, {"a"}))
        == 1
    )
    assert seen == {"B": {"a"}} and "oracle: 1 tasks not in the split (b)" in capsys.readouterr().out


def test_validate_open_shards_cover_every_open_block_once():
    """CI's validate-open job runs in shards by block: together they must run every open block, each exactly once."""
    import re

    yml = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "check.yml").read_text()
    shards = [s.split(",") for s in re.findall(r'\{shard: \d+, blocks: "([A-Z,]+)"\}', yml)]
    blocks = [b for s in shards for b in s]
    assert len(shards) >= 2 and len(blocks) == len(set(blocks))
    assert set(blocks) == {t.block for t in T.all_tasks() if t.split == "open"}


def test_every_workflow_action_is_pinned_to_a_commit():
    """A tag can be moved to other code; a full commit SHA can't. The tag stays as a comment."""
    import re

    for wf in (Path(__file__).resolve().parent.parent / ".github" / "workflows").glob("*.yml"):
        uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)(.*)$", wf.read_text(), re.M)
        assert uses, wf
        loose = [u for u, rest in uses if not re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", u) or "# v" not in rest]
        assert loose == [], f"{wf.name}: not pinned to a commit with its tag as a comment: {loose}"


def test_leak_check_finds_heldout_ids_and_text(tmp_path, capsys):
    import leak_check

    markers = leak_check.heldout_markers(str(T.heldout_dir()))
    some_id = next(m for m, kind in markers.items() if kind == "task id")
    (tmp_path / "clean.md").write_text("nothing held-out here\n")
    assert leak_check.scan(tmp_path, markers) == []
    (tmp_path / "leak.md").write_text(f"see {some_id}\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "x.md").write_text(some_id)  # vendored code isn't ours: skipped
    assert leak_check.scan(tmp_path, markers) == [f"leak.md: task id ({some_id})"]
    assert leak_check.main(["--heldout", str(T.heldout_dir()), str(tmp_path)]) == 1
    assert some_id in capsys.readouterr().out
    # shapes beyond today's ids: renamed ids of any round, but not open ids or ids inside longer names
    renamed, later = "-".join(["t2", "f", "zz", "renamed"]), "-".join(["t7", "c", "somewhere"])  # built, so no literal
    (tmp_path / "leak.md").write_text(f"old {renamed}, later xx {later}; open a-pay-invoice; key pe2-t-a-1\n")
    found = leak_check.scan(tmp_path, markers)
    assert found == [f"leak.md: held-out id shape ({renamed})", f"leak.md: held-out id shape ({later})"]


def test_leak_check_says_what_it_checked(tmp_path, capsys, monkeypatch):
    """With the held-out set it checks ids and text; without one (a public checkout) only id shapes, and says so."""
    import leak_check

    (tmp_path / "clean.md").write_text("nothing held-out here\n")
    assert leak_check.main(["--heldout", str(T.heldout_dir()), str(tmp_path)]) == 0
    assert "no held-out ids, instruction text or id shapes in the tree" in capsys.readouterr().out
    monkeypatch.setattr(leak_check, "heldout_markers", lambda heldout: {})
    monkeypatch.setattr(leak_check, "heldout_dir", lambda: None)
    assert leak_check.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("no held-out set here: only held-out id shapes were checked") and "not held-out ids" in out


def test_the_public_tree_names_its_licences_and_its_security_policy():
    """Code under Apache-2.0 (the Solidity sources and their built artifacts too), data under CC BY 4.0, a security
    policy; the contracts resolve by name without a held-out-only table entry."""
    from onchainbench import contracts as C

    root = SCRIPTS.parent
    srcs = sorted((root / "onchainbench" / "contracts" / "src").glob("*.sol"))
    assert srcs and all(p.read_text().startswith("// SPDX-License-Identifier: Apache-2.0\n") for p in srcs)
    built = sorted(C.OUT.glob("*.sol/*.json"))  # the committed build is of these sources: the same licence
    assert {p.parent.name for p in built} == {p.name for p in srcs}
    for p in built:
        meta = json.loads(json.loads(p.read_text())["rawMetadata"])
        assert {s["license"] for s in meta["sources"].values()} == {"Apache-2.0"}
    assert "Escrowbook" not in C.FILES and C.abi("Escrowbook")  # a contract in its own file needs no entry
    text = lambda name: (root / name).read_text(encoding="utf-8")  # noqa: E731 - UTF-8 on Windows too
    assert "Attribution 4.0 International" in text("LICENSE-DATA")
    readme = text("README.md")
    assert "## Licence" in readme and "(LICENSE)" in readme and "(LICENSE-DATA)" in readme
    assert "private vulnerability reporting" in text("SECURITY.md")
    assert "SECURITY.md" in text("CONTRIBUTING.md")
