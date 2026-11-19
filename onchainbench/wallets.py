"""Wallet adapters: a wallet stack other than Connect, in front of a harness (CONTRIBUTING.md, "Wallet adapters").

An arm picks one with the `wallet=<name>` modifier, e.g. `hermes:M|wallet=evm-mcp`. Without one, the harness uses
Connect. `hermes-raw:M` and `openclaw-raw:M` are the same as `hermes:M|wallet=evm-mcp` and
`openclaw:M|wallet=evm-mcp`.

With a wallet, the trial's agent account is the fork's test EOA (`t.fc.s.safe`, its key `t.fc.s.eoa_key`); the
workspace has no Connect config, and the agent's RPC (`t.fc.s.rpc_url`) also takes `eth_sendRawTransaction`, so every
send is logged and checked like a Connect transaction. An adapter supplies:
  name        the `wallet=` value and the MCP server's name in the harness config
  brief       the workspace brief (AGENTS.md, CLAUDE.md), formatted with chain, chain_id, account, rpc and sym;
              it must say that "the safe" in the tasks means this account
  mcp_server  t -> a stdio MCP server entry ({"command", "args", "env"}) that signs with the trial's key and sends
              only through the trial's RPC
  problem     () -> None, or why the adapter can't run here (not installed, wrong version)
Other packages register adapters under the `onchainbench.wallets` entry points (name = the `wallet=` value; value =
a `Wallet`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

WALLET_GROUP = "onchainbench.wallets"
WALLET_HARNESSES = frozenset({"hermes", "openclaw"})  # the built-in harnesses that take a wallet adapter's MCP server


@dataclass(frozen=True)
class Wallet:
    name: str
    brief: str
    mcp_server: Callable[[object], dict]
    problem: Callable[[], str | None] = lambda: None


def _evm_mcp() -> Wallet:
    from onchainbench.raw_key import EVM_MCP, RAW_BRIEF, evm_mcp_server

    def problem() -> str | None:
        return None if EVM_MCP.is_file() else f"{EVM_MCP} missing: run scripts/install-arms.sh"

    return Wallet("evm", RAW_BRIEF, evm_mcp_server, problem)


BUILTIN = {"evm-mcp": _evm_mcp}
RAW_ALIASES = {"hermes-raw": "hermes", "openclaw-raw": "openclaw"}  # the older names, wallet=evm-mcp


def installed() -> dict:
    from importlib.metadata import entry_points

    return {ep.name: ep for ep in entry_points(group=WALLET_GROUP)}


def known(name: str) -> bool:
    return name in BUILTIN or name in installed()


def load(name: str) -> Wallet:
    if name in BUILTIN:
        return BUILTIN[name]()
    if (ep := installed().get(name)) is not None:
        return ep.load()
    raise ValueError(f"unknown wallet {name!r}: not built in, and no installed package registers it")


def wallet_name(kind: str, mods: set[str]) -> str | None:
    """The arm's wallet adapter, or None for Connect."""
    names = [m.partition("=")[2] for m in mods if m.startswith("wallet=")]
    if len(names) > 1:
        raise ValueError(f"one wallet per arm, got {', '.join(sorted(names))}")
    if names:
        return names[0]
    return "evm-mcp" if kind in RAW_ALIASES else None


def harness(kind: str) -> str:
    """The harness behind a kind (`hermes-raw` is Hermes)."""
    return RAW_ALIASES.get(kind, kind)


def check_arm(kind: str, mods: set[str]) -> str | None:
    """Why an arm's wallet can't run (unknown, not installed, or a harness that takes no wallet), or None."""
    try:
        name = wallet_name(kind, mods)
    except ValueError as e:
        return str(e)
    if name is None:
        return None
    if harness(kind) not in WALLET_HARNESSES:
        return f"{kind} takes no wallet adapter (only {', '.join(sorted(WALLET_HARNESSES))} do)"
    if not known(name):
        return f"unknown wallet {name!r}: not built in, and no installed package registers it"
    return load(name).problem()


__all__ = ["WALLET_GROUP", "WALLET_HARNESSES", "Wallet", "check_arm", "harness", "known", "load", "wallet_name"]
