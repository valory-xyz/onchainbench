"""Wallet adapters: the `wallet=` modifier, the built-in evm-mcp adapter, plugin adapters and the runner's checks."""

import pytest

from onchainbench import arms as A
from onchainbench import raw_key, wallets
from onchainbench import run as R


class EP:
    def __init__(self, obj):
        self.obj = obj

    def load(self):
        return self.obj


PLUGIN = wallets.Wallet("plug", "account {account} on {chain}", lambda t: {"command": "plug", "args": [], "env": {}})


@pytest.fixture
def plugin_wallet(monkeypatch):
    monkeypatch.setattr(wallets, "installed", lambda: {"plug": EP(PLUGIN)})


def test_wallet_name_from_the_modifier_and_the_raw_aliases():
    assert wallets.wallet_name("hermes", set()) is None  # Connect
    assert wallets.wallet_name("hermes", {"wallet=plug", "nogate"}) == "plug"
    assert wallets.wallet_name("hermes-raw", set()) == wallets.wallet_name("openclaw-raw", set()) == "evm-mcp"
    assert wallets.wallet_name("hermes-raw", {"wallet=plug"}) == "plug"  # an explicit wallet wins
    with pytest.raises(ValueError, match="one wallet per arm"):
        wallets.wallet_name("hermes", {"wallet=a", "wallet=b"})
    assert wallets.harness("openclaw-raw") == "openclaw" and wallets.harness("codex") == "codex"
    assert A.parse_arm("hermes:m|wallet=plug") == ("hermes", "m", {"wallet=plug"})


def test_builtin_and_plugin_adapters_load(plugin_wallet, monkeypatch, tmp_path):
    evm = wallets.load("evm-mcp")
    assert evm.name == "evm" and evm.brief is raw_key.RAW_BRIEF and evm.mcp_server is raw_key.evm_mcp_server
    monkeypatch.setattr(raw_key, "EVM_MCP", tmp_path / "missing.js")
    assert "install-arms.sh" in wallets.load("evm-mcp").problem()
    (tmp_path / "missing.js").write_text("")
    assert wallets.load("evm-mcp").problem() is None
    assert wallets.load("plug") is PLUGIN and PLUGIN.problem() is None and wallets.known("plug")
    with pytest.raises(ValueError, match="unknown wallet 'nope'"):
        wallets.load("nope")


def test_check_arm(plugin_wallet, monkeypatch):
    assert wallets.check_arm("codex", set()) is None  # no wallet: nothing to check
    assert wallets.check_arm("hermes", {"wallet=plug"}) is None
    assert "one wallet per arm" in wallets.check_arm("hermes", {"wallet=a", "wallet=b"})
    assert "takes no wallet adapter" in wallets.check_arm("claude-code", {"wallet=plug"})
    assert "unknown wallet 'nope'" in wallets.check_arm("openclaw", {"wallet=nope"})
    monkeypatch.setattr(raw_key, "EVM_MCP", raw_key.EVM_MCP.with_name("not-there.js"))
    assert "install-arms.sh" in wallets.check_arm("hermes-raw", set())


def test_the_runner_refuses_an_arm_whose_wallet_cant_run(plugin_wallet, capsys):
    assert R.main(["--arms", "codex:openrouter/x|wallet=plug", "--run", "pytest-run-x"]) == 2
    assert "takes no wallet adapter" in capsys.readouterr().out
    assert R.main(["--arms", "hermes:openrouter/x|wallet=nope", "--run", "pytest-run-x"]) == 2
    assert "unknown wallet 'nope'" in capsys.readouterr().out


def test_a_plugin_adapter_reaches_the_harness_config(plugin_wallet, tmp_path):
    class Fake:
        ws = tmp_path / "ws"

    Fake.ws.mkdir()
    s = A.open_session("openclaw:openrouter/x|wallet=plug", Fake, [], 10)
    assert s.wallet is PLUGIN and s._mcp_server() == {"plug": {"command": "plug", "args": [], "env": {}}}


def test_contributed_task_modules_are_found_by_name(tmp_path):
    from onchainbench import tasks as reg

    for f in ("contrib_zeta.py", "contrib_aave.py", "block_x.py", "helpers.py"):
        (tmp_path / f).write_text("")
    assert reg.contrib_modules(tmp_path) == ["contrib_aave", "contrib_zeta"]
    assert reg.contrib_modules() == []  # none contributed yet


def test_no_plugin_adapter_is_installed_here():
    assert wallets.installed() == {} and not wallets.known("plug") and wallets.known("evm-mcp")
