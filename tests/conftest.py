import json
import os
import socketserver
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# litellm loads a .env at import in its default DEV mode: one above the venv put a real OpenRouter key into the test
# process mid-suite (tests then depended on their order, and could reach a paid API). Never in the tests.
os.environ["LITELLM_MODE"] = "PRODUCTION"

# The local servers (relay, web corpus, live-web proxy, broker, mocks) poll for shutdown every 0.5 s by default, so
# each stop() waited up to that long: about a third of the suite's time. In the tests only; same behaviour otherwise.
socketserver.BaseServer.serve_forever.__defaults__ = (0.05,)

# pytest-xdist (scripts/check.sh: --dist loadgroup): each file runs on one worker (its module fixtures stay one), and
# the files that start forks or their relays share one worker. A chain's relay binds the chain's fixed port, one per
# process, and the fork cache needs that port: a second process gets a free port and no cache (cold, and
# test_a_second_fork_of_the_same_block_starts_warm fails).
FORK_FILES = {
    "test_fork_cache.py",
    "test_run_forks.py",
    "test_run_checks.py",
    "test_run_more.py",
    "test_failure_kinds.py",
    "test_integrity_log.py",
    "test_relay.py",
    "test_team_safe_tasks.py",
}


@pytest.hookimpl(tryfirst=True)  # before xdist's own hook reads the marks into each test's group (`@forks`): it was
# registered later, so without this it ran first, saw no mark, and spread the fork files over every worker, whose
# relays then raced for each chain's port (a fork of the loser started cold)
def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.path.name
        item.add_marker(pytest.mark.xdist_group("forks" if name in FORK_FILES else name))


from onchainbench.mock_connect.server import MockConnect, MockState  # noqa: E402

# The held-out set is private. Without it (public CI), the tests use a one-task stand-in, so the held-out
# machinery still runs; tests that need the real 130 tasks skip. ONCHAINBENCH_TEST_FIXTURE_HELDOUT=1 forces the stand-in.
FIXTURE_HELDOUT = Path(__file__).resolve().parent / "fixtures" / "heldout" / "onchainbench_heldout"
try:
    if os.environ.get("ONCHAINBENCH_TEST_FIXTURE_HELDOUT"):
        raise ModuleNotFoundError(name="onchainbench_heldout")
    import onchainbench_heldout  # noqa: F401

    REAL_HELDOUT = True
except ModuleNotFoundError:
    os.environ["ONCHAINBENCH_HELDOUT"] = str(FIXTURE_HELDOUT)
    REAL_HELDOUT = False
needs_real_heldout = pytest.mark.skipif(not REAL_HELDOUT, reason="needs the private held-out set")
# tests that assert the stand-in's own ids and salt (t-fixture-pay, fixture-salt-not-secret): CI always has the stand-in;
# with the real held-out set installed they'd see its ids instead (ONCHAINBENCH_TEST_FIXTURE_HELDOUT=1 forces the stand-in)
needs_fixture_heldout = pytest.mark.skipif(REAL_HELDOUT, reason="asserts the stand-in held-out set's ids")


@pytest.fixture
def mock():
    m = MockConnect(MockState())
    yield m
    m.stop()


@pytest.fixture
def connect_ws(tmp_path, mock):
    """A Connect-shaped workspace whose real .mcp.json points at the mock server."""
    ws = tmp_path / "persistent_data"
    ws.mkdir()
    (ws / ".mcp.json").write_text(json.dumps(mock.mcp_json()))
    (ws / "AGENTS.md").write_text("# You are a Pearl Connect agent (test brief)\n")
    sk = ws / ".agents" / "skills" / "pearl-connect"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: pearl-connect\ndescription: Act on-chain.\n---\n# body\n")
    return ws


@pytest.fixture(autouse=True)
def _no_openrouter_provisioning(monkeypatch):
    """Tests never create real per-trial OpenRouter keys (the CLI arms)."""
    monkeypatch.delenv("OPENROUTER_PROVISIONING_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_preflight_calls(monkeypatch):
    """Tests never call the judge and user-sim models (the run's preflight is mocked to succeed)."""
    from onchainbench import run

    monkeypatch.setattr(run, "preflight_call", lambda model: None)


@pytest.fixture(autouse=True)
def _harnesses_installed(monkeypatch):
    """The runner's harness preflight passes in tests, whose CLI arms are faked (tested in test_failure_kinds.py)."""
    from onchainbench import run

    monkeypatch.setattr(run, "harness_preflight", lambda arms: [])
    monkeypatch.setattr(run, "harness_versions", lambda arms: ({}, []))  # each CLI's --version: the same file


@pytest.fixture(autouse=True)
def _private_shared_tmp(tmp_path_factory, monkeypatch):
    """A runner under test sweeps a scratch dir, never the machine's real /tmp."""
    from onchainbench import run

    monkeypatch.setattr(run, "SHARED_TMP", tmp_path_factory.mktemp("shared-tmp"))
