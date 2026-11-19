"""Fork configuration per chain the bench covers: archive RPC, pinned block, tokens and protocol addresses.

Blocks are pinned per chain (refreshed quarterly; every oracle must still pass after a refresh).
Anvil caches fetched state under ~/.foundry/cache/rpc/<chain>/<block>, so after the first run a fork
needs little or no network. Override the RPC with FORK_RPC_<CHAIN> (for example a paid archive node).

Addresses come from official sources (each protocol's own deployment docs) and were checked on the fork
(`python -m onchainbench.forks --check`).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from onchainbench.evm import Fork, calldata, checksum, decode

SAFE_141 = {
    "singleton": "0x29fcb43b46531bca003ddc8fcb67ffe91900c762",  # SafeL2 1.4.1
    "factory": "0x4e1dcf7ad4e460cfd30791ccc4f9c8a4f820ec67",
    "fallback": "0xfd0732dc9e303f09fcef3a7388ad10a83459ec99",
}


@dataclass(frozen=True)
class Token:
    symbol: str
    address: str
    decimals: int


@dataclass
class ForkChain:
    name: str
    chain_id: int
    rpc: str
    block: int
    native: str
    wrapped: Token
    tokens: dict = field(default_factory=dict)  # symbol -> Token
    protocols: dict = field(default_factory=dict)  # name -> address
    phase: int = 1  # 1: full block A/B/G tasks; 2: template reuse; 3: smoke only

    def __post_init__(self):
        self.protocols = {k: v.lower() for k, v in self.protocols.items()}  # the ledger keeps addresses lower-case

    @property
    def fork_url(self) -> str:
        return os.environ.get(f"FORK_RPC_{self.name.upper()}", self.rpc)

    def token(self, sym: str) -> Token:
        return self.wrapped if sym == self.wrapped.symbol else self.tokens[sym]


def _t(sym, a, d):
    return Token(sym, a.lower(), d)


FORK_CHAINS: dict[str, ForkChain] = {
    "gnosis": ForkChain(
        "gnosis",
        100,
        "https://rpc.gnosischain.com",
        49193000,
        "xDAI",
        _t("WXDAI", "0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d", 18),
        {
            "USDC": _t("USDC", "0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83", 6),
            "USDC.e": _t("USDC.e", "0x2a22f9c3b484c3629090FeED35F17Ff8F88f76F0", 6),
            "sDAI": _t("sDAI", "0xaf204776c7245bF4147c2612BF6e5972Ee483701", 18),
            "GNO": _t("GNO", "0x9C58BAcC331c9aa871AFD802DB6379a98e80CEdb", 18),
            "WETH": _t("WETH", "0x6A023CCd1ff6F2045C3309768eAd9E68F978f6e1", 18),
            "OLAS": _t("OLAS", "0xcE11e14225575945b8E6Dc0D4F2dD4C570f79d9f", 18),
        },
        {
            "aave_pool": "0xb50201558B00496A145fE76f7424749556E326D8",
            "uniswap_router": "0xc6D25285D5C5b62b7ca26D6092751A145D50e9Be",
            "uniswap_factory": "0xe32F7dD7e3f098D518ff19A22d5f028e076489B1",
            "uniswap_npm": "0xAE8fbE656a77519a7490054274910129c9244FA3",
            "conditional_tokens": "0xCeAfDD6bc0bEF976fdCd1112955828E00543c0Ce",
            "fpmm_factory": "0x9083A2B699c0a4AD06F63580BDE2635d26a3eeF0",
        },
    ),
    "polygon": ForkChain(
        "polygon",
        137,
        "https://polygon.gateway.tenderly.co",
        97024000,
        "POL",
        _t("WPOL", "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270", 18),
        {
            "USDC": _t("USDC", "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359", 6),
            "USDC.e": _t("USDC.e", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", 6),
            "USDT": _t("USDT", "0xc2132D05D31c914a87C6611C10748AEb04B58e8F", 6),
            "WETH": _t("WETH", "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", 18),
            "OLAS": _t("OLAS", "0xFEF5d947472e72Efbb2E388c730B7428406F2F95", 18),
        },
        {
            "aave_pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
            "uniswap_router": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
            "uniswap_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
            "uniswap_npm": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
            "uniswap_quoter": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        },
    ),
    "ethereum": ForkChain(
        "ethereum",
        1,
        "https://mainnet.gateway.tenderly.co",
        26378000,
        "ETH",
        _t("WETH", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
        {"USDC": _t("USDC", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6)},
        {
            "aave_pool": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
            "uniswap_router": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
            "uniswap_quoter": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        },
        phase=2,
    ),
    "base": ForkChain(
        "base",
        8453,
        "https://base.gateway.tenderly.co",
        55259000,
        "ETH",
        _t("WETH", "0x4200000000000000000000000000000000000006", 18),
        {"USDC": _t("USDC", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", 6)},
        {
            "aave_pool": "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
            "uniswap_router": "0x2626664c2603336E57B271c5C0b26F421741e481",
            "uniswap_quoter": "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
        },
        phase=2,
    ),
    "arbitrum": ForkChain(
        "arbitrum",
        42161,
        "https://arbitrum.gateway.tenderly.co",
        522731000,
        "ETH",
        _t("WETH", "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1", 18),
        {"USDC": _t("USDC", "0xaf88d065e77c8cC2239327C5EDb3A432268e5831", 6)},
        {
            "aave_pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
            "uniswap_router": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
            "uniswap_quoter": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        },
        phase=2,
    ),
    "optimism": ForkChain(
        "optimism",
        10,
        "https://optimism.gateway.tenderly.co",
        159321000,
        "ETH",
        _t("WETH", "0x4200000000000000000000000000000000000006", 18),
        {"USDC": _t("USDC", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", 6)},
        {
            "aave_pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
            "uniswap_router": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
            "uniswap_quoter": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        },
        phase=2,
    ),
    "celo": ForkChain(
        "celo",
        42220,
        "https://forno.celo.org",
        82341000,
        "CELO",
        # CELO is native and an ERC-20 at once (token duality); the "wrapped" entry is the CELO token itself
        _t("CELO", "0x471EcE3750Da237f93B8E339c536989b8978a438", 18),
        {"USDC": _t("USDC", "0xcebA9300f2b948710d2653dD7B07f33A8B32118C", 6)},
        {
            "aave_pool": "0x3E59A31363E2ad014dcbc521c4a0d5757d9f3402",
            "uniswap_router": "0x5615CDAb10dc425a742d643d949a7F474C01abc4",
            "uniswap_quoter": "0x82825d0554fA07f7FC52Ab63c961F330fdEFa8E8",
        },
        phase=2,
    ),
    "mode": ForkChain(
        "mode",
        34443,
        "https://mainnet.mode.network",
        47037000,
        "ETH",
        _t("WETH", "0x4200000000000000000000000000000000000006", 18),
        {"USDC": _t("USDC", "0xd988097fb8612cc24eeC14542bC03424c656005f", 6)},
        phase=3,
    ),
    "robinhood": ForkChain(
        "robinhood",
        4663,
        "https://robinhood.drpc.org",  # archive state; the chain's own RPC keeps no history
        111050085,
        "ETH",
        _t("WETH", "0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD73", 18),  # SwapRouter02.WETH9() at the pin
        protocols={"uniswap_router": "0xCaf681a66D020601342297493863E78C959E5cb2"},
        phase=3,
    ),
}

ERC20 = {
    "balanceOf": "balanceOf(address)",
    "transfer": "transfer(address,uint256)",
    "approve": "approve(address,uint256)",
    "allowance": "allowance(address,address)",
}


def deploy_safe(fork: Fork, owner: str, salt: int = 1) -> str:
    """A Safe 1.4.1 with one owner (threshold 1), created through the canonical factory on the fork."""
    init = calldata(
        "setup(address[],uint256,address,bytes,address,address,uint256,address)",
        [checksum(owner)],
        1,
        "0x" + "00" * 20,
        b"",
        checksum(SAFE_141["fallback"]),
        "0x" + "00" * 20,
        0,
        "0x" + "00" * 20,
    )
    r = fork.rpc.send(
        owner,
        SAFE_141["factory"],
        calldata(
            "createProxyWithNonce(address,bytes,uint256)",
            checksum(SAFE_141["singleton"]),
            bytes.fromhex(init[2:]),
            salt,
        ),
        gas=1_000_000,
    )
    if int(r["status"], 16) != 1:  # pragma: no cover - the canonical factory on a pinned fork
        raise RuntimeError("Safe deployment reverted")
    topic = "0x" + __import__("eth_utils").keccak(text="ProxyCreation(address,address)").hex()
    for log in r["logs"]:
        if log["topics"][0] == topic:
            return "0x" + log["topics"][1][-40:]
    raise RuntimeError("no ProxyCreation event")  # pragma: no cover - the factory always emits it


def owner_signature(owner: str) -> bytes:
    """A pre-validated signature (v=1): valid when the owner itself sends execTransaction."""
    return bytes.fromhex(owner[2:].rjust(64, "0")) + b"\x00" * 32 + b"\x01"


def exec_calldata(owner: str, to: str, value: int, data: str, operation: int = 0) -> str:
    return calldata(
        "execTransaction(address,uint256,bytes,uint8,uint256,uint256,uint256,address,address,bytes)",
        checksum(to),
        value,
        bytes.fromhex(data.removeprefix("0x")),
        operation,
        0,
        0,
        0,
        "0x" + "00" * 20,
        "0x" + "00" * 20,
        owner_signature(owner),
    )


def check(names: list[str]) -> int:  # pragma: no cover - a manual check against live RPCs
    bad = 0
    for n in names:
        c = FORK_CHAINS[n]
        if not c.block:
            print(f"{n}: no pinned block (set FORK_RPC_{n.upper()} and pin one)")
            continue
        f = Fork(c.fork_url, c.block, c.chain_id)
        try:
            items = {**{t.symbol: t.address for t in [c.wrapped, *c.tokens.values()]}, **c.protocols, **SAFE_141}
            for k, a in items.items():
                size = (len(f.code(a)) - 2) // 2
                ok = size > 0
                bad += not ok
                extra = ""
                if ok and k in {t.symbol for t in [c.wrapped, *c.tokens.values()]}:
                    tok = c.token(k)
                    d = decode(["uint8"], f.rpc.call(tok.address, "0x313ce567"))[0]
                    extra = f" decimals={d}" + ("" if d == tok.decimals else " MISMATCH")
                    bad += d != tok.decimals
                print(f"{n:9} {k:20} {a} {'ok' if ok else 'NO CODE'}{extra}")
        finally:
            f.stop()
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(check(sys.argv[2:] if len(sys.argv) > 2 else ["gnosis", "polygon"]))
