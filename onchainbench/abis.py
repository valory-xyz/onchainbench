"""The local cache of verified ABIs behind fork-connect's `get_abi` (a Sourcify cache;
see olas-ocb-agent's docs/DESIGN.md, "Principles, and why").

    python -m onchainbench.abis fetch gnosis 0xabc... 0xdef...   # fetch from Sourcify (follows EIP-1967 proxies)
    python -m onchainbench.abis signatures                       # rebuild abis/signatures.json

Unverified contracts (block G) are left out on purpose: the agent must work from bytecode and selectors.
"""

from __future__ import annotations

import json
import sys
import urllib.request

from eth_utils import keccak

from onchainbench.fork_connect import ABI_DIR, EIP1967_IMPL, _abi_type
from onchainbench.forks import FORK_CHAINS

COMMON = [
    "transfer(address,uint256)",
    "transferFrom(address,address,uint256)",
    "approve(address,uint256)",
    "balanceOf(address)",
    "allowance(address,address)",
    "decimals()",
    "symbol()",
    "name()",
    "totalSupply()",
    "deposit()",
    "withdraw(uint256)",
    "deposit(uint256,address)",
    "redeem(uint256,address,address)",
    "withdraw(uint256,address,address)",
    "multicall(bytes[])",
    "execTransaction(address,uint256,bytes,uint8,uint256,uint256,uint256,address,address,bytes)",
    "setApprovalForAll(address,bool)",
    "safeTransferFrom(address,address,uint256,uint256,bytes)",
    # block G's unverified StakeVault: public in signature lists (as on 4byte), absent from the ABI cache
    "stakeFor(address,uint256)",
    "exitShares(uint256,address)",
    "sharesOf(address)",
    "totalShares()",
    "asset()",
]


def fetch(chain: str, address: str) -> list | None:  # pragma: no cover - network; run by hand to refresh the cache
    cid = FORK_CHAINS[chain].chain_id
    url = f"https://sourcify.dev/server/v2/contract/{cid}/{address}?fields=abi"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pearl-eval"}), timeout=30) as r:
            return json.loads(r.read()).get("abi")
    except OSError:
        return None


def save(chain: str, address: str, abi: list) -> None:
    d = ABI_DIR / chain
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{address.lower()}.json").write_text(json.dumps(abi, separators=(",", ":")))


def fetch_many(chain: str, addresses: list[str]) -> None:  # pragma: no cover - network
    from onchainbench.evm import Rpc

    rpc = Rpc(FORK_CHAINS[chain].fork_url)
    for a in addresses:
        abi = fetch(chain, a)
        print(f"{chain} {a} {'ok' if abi else 'not verified'}")
        if abi:
            save(chain, a, abi)
        try:
            slot = rpc("eth_getStorageAt", a, EIP1967_IMPL, hex(FORK_CHAINS[chain].block))
        except Exception:  # noqa: BLE001 - not a proxy, or the RPC failed
            continue
        if int(slot, 16):
            impl = "0x" + slot[-40:]
            iabi = fetch(chain, impl)
            print(f"  implementation {impl} {'ok' if iabi else 'not verified'}")
            if iabi:
                save(chain, impl, iabi)


def signatures() -> dict:
    out = {"0x" + keccak(text=s)[:4].hex(): s for s in COMMON}
    for p in sorted(ABI_DIR.glob("*/*.json")):
        for item in json.loads(p.read_text()):
            if item.get("type") == "function":
                sig = f"{item['name']}({','.join(_abi_type(i) for i in item.get('inputs', []))})"
                out.setdefault("0x" + keccak(text=sig)[:4].hex(), sig)
    (ABI_DIR / "signatures.json").write_text(json.dumps(dict(sorted(out.items())), indent=0))
    return out


if __name__ == "__main__":  # pragma: no cover
    if sys.argv[1] == "fetch":
        fetch_many(sys.argv[2], sys.argv[3:])
    else:
        print(len(signatures()), "signatures")
