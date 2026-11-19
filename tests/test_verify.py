"""A verifiable run archive: publish writes SHA256SUMS over every artifact, run.json carries its
siblings' sums, and onchainbench.verify checks a download in place or flat, as a release leaves it."""

import hashlib
import json
import shutil

import pytest
from test_publish import frozen_agent, open_run, run_dir  # noqa: F401 (frozen_agent: a fixture)

from onchainbench import publish as P
from onchainbench import verify as V

pytestmark = pytest.mark.usefixtures("frozen_agent")


def published(tmp_path):
    d = run_dir(tmp_path, open_run(tmp_path))
    (d / "summary.json").write_text("{}")
    (d / "inspect").mkdir()
    (d / "inspect" / "codex.json").write_text('{"version": 2}')
    doc, traj = P.build(d, "https://r")
    P.write(d, doc, traj)
    return d


def test_sha256sums_cover_every_artifact_and_run_json_carries_its_siblings(tmp_path):
    d = published(tmp_path)
    sums = V.parse((d / "SHA256SUMS").read_text())
    names = ["config.json", "inspect/codex.json", "run.json", "summary.json", "trajectories.jsonl", "trials.jsonl"]
    assert list(sums) == names
    assert all(sums[n] == hashlib.sha256((d / n).read_bytes()).hexdigest() for n in names)
    run = json.loads((d / "run.json").read_text())
    assert run["artifacts_sha256"] == {n: h for n, h in sums.items() if n != "run.json"}
    P.validate(run)
    assert (d / "SHA256SUMS").read_text().splitlines()[0] == f"{sums['config.json']}  config.json"  # sha256sum -c
    assert V.parse(f"{'A' * 64} *bin.json\n\n") == {"bin.json": "a" * 64}  # binary-mode lines, blank lines


def test_verify_in_place_and_a_flat_partial_download(tmp_path, capsys):
    d = published(tmp_path)
    assert V.main([str(d)]) == 0 and capsys.readouterr().out.endswith("verified\n")
    flat = tmp_path / "download"  # a release download: the files --upload-release attaches, flat
    flat.mkdir()
    for f in ("run.json", "SHA256SUMS", "trajectories.jsonl"):
        shutil.copy(d / f, flat)
    lines, ok = V.check(flat)
    assert ok and {"run.json: OK", "trajectories.jsonl: OK"} <= set(lines)
    assert {"inspect/codex.json: not downloaded", "trials.jsonl: not downloaded"} <= set(lines)
    (flat / "codex.json").write_bytes((d / "inspect" / "codex.json").read_bytes())  # a nested file, fetched flat
    assert "inspect/codex.json: OK" in V.check(flat)[0]


def test_verify_refuses_paths_outside_the_directory(tmp_path):
    """A SHA256SUMS naming `..`, absolute, drive or backslash paths fails, even when those files match; a symlink
    out of the directory is never followed."""
    d = tmp_path / "dl"
    d.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    h = hashlib.sha256(b"x").hexdigest()
    names = ["../secret.txt", str(outside), "..\\secret.txt", "C:secret.txt"]
    (d / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n in names))
    lines, ok = V.check(d)
    assert not ok and lines == [f"{n}: refused, not a path inside the run" for n in names]
    assert V.main([str(d)]) == 1
    (d / "link.txt").symlink_to(outside)
    (d / "SHA256SUMS").write_text(f"{h}  link.txt\n")
    assert V.check(d) == (["link.txt: not downloaded"], False)


def test_run_json_must_be_listed_and_parse(tmp_path):
    d = published(tmp_path)
    sums = (d / "SHA256SUMS").read_text().splitlines()
    (d / "SHA256SUMS").write_text("".join(x + "\n" for x in sums if not x.endswith(" run.json")))
    (d / "run.json").write_text((d / "run.json").read_text().replace('"open"', '"heldout"'))  # tampered, unlisted
    lines, ok = V.check(d)
    assert not ok and lines[-1] == "run.json: FAILED, not listed in SHA256SUMS"
    (d / "SHA256SUMS").write_text("".join(x + "\n" for x in sums))
    for corrupt in ("{", "[]", "\xff"):
        (d / "run.json").write_bytes(corrupt.encode("latin-1"))
        lines, ok = V.check(d)
        assert not ok and "run.json: FAILED" in lines and lines[-1] == "run.json: FAILED, not a JSON object"


def test_verify_fails_on_a_changed_file_or_a_disagreeing_run_json(tmp_path, capsys):
    d = published(tmp_path)
    (d / "trials.jsonl").write_text((d / "trials.jsonl").read_text() + "\n")
    lines, ok = V.check(d)
    assert not ok and "trials.jsonl: FAILED" in lines
    assert V.main([str(d)]) == 1 and "NOT verified" in capsys.readouterr().out
    only = tmp_path / "only"  # every file present matches, but SHA256SUMS no longer lists one run.json committed to
    only.mkdir()
    shutil.copy(d / "run.json", only)
    keep = [x for x in (d / "SHA256SUMS").read_text().splitlines() if x.endswith(" run.json")]
    (only / "SHA256SUMS").write_text("\n".join(keep) + "\n")
    lines, ok = V.check(only)
    assert not ok and lines == ["run.json: OK", "run.json: FAILED, its artifacts_sha256 doesn't match SHA256SUMS"]


def test_verify_needs_something_to_check(tmp_path, capsys):
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  trials.jsonl\n")
    assert V.check(tmp_path) == (["trials.jsonl: not downloaded"], False)  # nothing downloaded: nothing verified
    assert V.main([]) == 2 and V.main(["-h"]) == 2
    assert V.main([str(tmp_path / "none")]) == 1 and "no SHA256SUMS" in capsys.readouterr().out
