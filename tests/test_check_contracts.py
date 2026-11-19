"""scripts/check_contracts.py: the committed contract build (what runs deploy) matches the sources."""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_contracts as CC  # noqa: E402


def _art(code="0x6001", runtime="0x6002", meta='{"sources":{"src/A.sol":{"keccak256":"0x1"}}}', **extra):
    return {
        "abi": [{"type": "function", "name": "f"}],
        "bytecode": {"object": code, "sourceMap": extra.get("smap", "1:2:0")},
        "deployedBytecode": {"object": runtime, "sourceMap": extra.get("smap", "1:2:0")},
        "rawMetadata": meta,
        "id": extra.get("id", 0),
    }


def _out(root: Path, arts: dict) -> Path:
    for name, a in arts.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(a))
    (root / "build-info").mkdir(exist_ok=True)
    (root / "build-info" / "x.json").write_text("{}")  # not an artifact
    return root


def test_source_maps_and_ids_dont_count_but_bytecode_abi_and_metadata_do(tmp_path):
    committed = CC.artifacts(_out(tmp_path / "a", {"A.sol/A.json": _art()}))
    assert list(committed) == ["A.sol/A.json"]
    same = CC.artifacts(_out(tmp_path / "b", {"A.sol/A.json": _art(smap="9:9:9", id=7)}))
    assert CC.compare(committed, same) == []
    drift = CC.artifacts(_out(tmp_path / "c", {"A.sol/A.json": _art(runtime="0x6003", meta="{}")}))
    assert CC.compare(committed, drift) == [
        "A.sol/A.json: deployedBytecode differs from the build",
        "A.sol/A.json: metadata differs from the build",
    ]
    other = CC.artifacts(_out(tmp_path / "d", {"B.sol/B.json": _art()}))
    assert CC.compare(committed, other) == [
        "A.sol/A.json: committed, but the sources don't build it",
        "B.sol/B.json: built from the sources, but not committed",
    ]


def test_main_fails_on_drift_and_on_a_failed_build(tmp_path, monkeypatch, capsys):
    contracts = tmp_path / "contracts"
    _out(contracts / "out", {"A.sol/A.json": _art()})
    fresh = {"A.sol/A.json": _art()}
    monkeypatch.setattr(CC, "build", lambda c, dest, forge: _out(dest / "out", fresh))
    assert CC.main([], contracts) == 0 and "matches a fresh build" in capsys.readouterr().out
    fresh = {"A.sol/A.json": _art(code="0x60ff")}
    assert CC.main([], contracts) == 1 and "bytecode differs" in capsys.readouterr().out
    monkeypatch.undo()  # the real build, with a forge binary that isn't there
    assert CC.main(["--forge", str(tmp_path / "no-such-forge")], contracts) == 1


def test_a_failed_forge_build_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="forge build failed"):
        CC.build(tmp_path, tmp_path / "build", forge=sys.executable)  # not forge: exits non-zero


@pytest.mark.skipif(shutil.which("forge") is None or sys.platform == "win32", reason="needs Foundry's forge")
def test_the_committed_build_matches_the_sources(capsys):
    assert CC.main([]) == 0, capsys.readouterr().out
