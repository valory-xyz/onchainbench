"""Contracts the harness deploys on forks (look-alike tokens, synthetic protocols).

Sources in src/, built with Foundry (`cd onchainbench/contracts && forge build`; solc 0.8.26, pinned in
foundry.toml). The build output in out/ is committed, so runs need no compiler. A contract is looked up in the
source file of the same name unless FILES names another.
"""

from __future__ import annotations

import json
from pathlib import Path

from eth_abi import encode

OUT = Path(__file__).resolve().parent / "out"


FILES = {
    "StakeVault": "LongTail",
    "SavingsImpl": "LongTail",
    "Proxy1967": "LongTail",
    "OddSwap": "LongTail",
    "PolicyCoSigner": "TeamSafe",
    "SweepModule": "TeamSafe",
    "RecoveryModule": "TeamSafe",
    "PassGuard": "TeamSafe",
}


def artifact(name: str) -> dict:
    return json.loads((OUT / f"{FILES.get(name, name)}.sol" / f"{name}.json").read_text())


def bytecode(name: str, ctor_types: list[str] | None = None, ctor_args: list | None = None) -> str:
    code = artifact(name)["bytecode"]["object"]
    code = code if code.startswith("0x") else "0x" + code
    if ctor_types:
        code += encode(ctor_types, ctor_args or []).hex()
    return code


def abi(name: str) -> list:
    return artifact(name)["abi"]


def deploy(t, name: str, ctor_types: list[str] | None = None, ctor_args: list | None = None) -> str:
    """Deploy from a harness account; the address is recorded so the integrity check ignores the deployment."""
    a = t.fork.deploy(bytecode(name, ctor_types, ctor_args))
    return a


def deploy_token(t, name: str, symbol: str, decimals: int) -> str:
    return deploy(t, "TestToken", ["string", "string", "uint8"], [name, symbol, decimals])
