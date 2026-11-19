"""The wheel ships every file the bench's digest covers: an installed bench has the same code and task data as a
checkout, so `python -m onchainbench.freeze --bench-digest` gives the same digest either way."""

import tomllib
from pathlib import Path

from onchainbench import freeze

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "onchainbench"


def _shipped(root: Path, patterns: list[str]) -> set[str]:
    """The data files setuptools' package-data patterns include, relative to the package: glob semantics, so `*`
    stays in one folder and `**` spans any number of them."""
    return {p.relative_to(root).as_posix() for pat in patterns for p in root.glob(pat) if p.is_file()}


def _covered(root: Path) -> list[str]:
    """The non-Python files `freeze.bench_digest` covers (setuptools ships the .py files of every package)."""
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and freeze.bench_covers(rel) and p.suffix != ".py":
            out.append(rel.as_posix())
    return out


def test_every_data_file_the_digest_covers_is_in_the_wheel():
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]
    shipped = _shipped(PKG, tool["setuptools"]["package-data"]["onchainbench"])
    missing = [f for f in _covered(PKG) if f not in shipped]
    assert not missing, f"in the digest but not in the wheel's package-data: {missing[:10]}"


def test_a_file_in_a_subfolder_a_pattern_misses_is_found(tmp_path):
    (tmp_path / "abis" / "gnosis").mkdir(parents=True)
    (tmp_path / "abis" / "top.json").write_text("{}")
    (tmp_path / "abis" / "gnosis" / "x.json").write_text("{}")
    assert _covered(tmp_path) == ["abis/gnosis/x.json", "abis/top.json"]
    assert _shipped(tmp_path, ["abis/*.json"]) == {"abis/top.json"}  # the subfolder isn't shipped
    assert _shipped(tmp_path, ["abis/**/*.json"]) == {"abis/top.json", "abis/gnosis/x.json"}


def test_the_deployed_contract_artifacts_are_in_the_digest_and_the_wheel():
    covered = _covered(PKG)
    assert "contracts/out/TestToken.sol/TestToken.json" in covered
    assert not [f for f in covered if f.startswith(("contracts/out/build-info/", "contracts/cache/"))]


def test_every_package_data_pattern_matches_a_file():
    """A pattern that matches nothing is dead (a folder that was removed) and hides what the wheel really ships."""
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]
    patterns = tool["setuptools"]["package-data"]["onchainbench"]
    assert [p for p in patterns if not _shipped(PKG, [p])] == []


def test_project_metadata_licences_and_sdist():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.11"  # datetime.UTC
    assert project["license"] == "Apache-2.0 AND CC-BY-4.0" and project["readme"] == "README.md"
    assert all((ROOT / f).is_file() for f in project["license-files"])
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")[:200]
    assert "CC BY 4.0" in (ROOT / "LICENSE-DATA").read_text(encoding="utf-8")
    assert project["urls"]["Repository"] == "https://github.com/valory-xyz/onchainbench"
    # the tests run from a checkout: the sdist leaves them out
    assert "prune tests" in (ROOT / "MANIFEST.in").read_text().splitlines()
