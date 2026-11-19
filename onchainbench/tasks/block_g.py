"""Block G (long-tail protocols, no plugins; METHODOLOGY R5) open tasks.

The agent has only Connect's tools (read, ABI, decode, simulate) and its shell. Mixed contracts:
synthetic ones deployed on the fork (an unverified vault, a proxy, a DEX with an odd ABI) and real
lesser-known venues (SushiSwap v2 on Polygon, Honeyswap on Gnosis).
"""

from __future__ import annotations

import time

from onchainbench.contracts import abi, deploy
from onchainbench.evm import calldata, checksum
from onchainbench.tasks.base import (
    Result,
    T,
    Task,
    approve,
    fmt,
    pct_floor,
    pearl,
    safe_tx,
    swap_protection,
    swap_sends,
)

E18 = 10**18
SUSHI_ROUTER = "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506"  # SushiSwap v2 on Polygon (verified)
HONEY_ROUTER = "0x1c232f01118cb8b424793ae03f870aa7d0ac7f77"  # Honeyswap (Uniswap v2 fork) on Gnosis (verified)


def _done(steps):
    return sum(steps), len(steps)


# ---- G1: an unverified vault -----------------------------------------------------------------
def p_vault(rng):
    x = rng.choice([6, 8, 10])
    return {"x": x, "y": rng.randint(2, x - 2)}


def s_vault(t: T):
    W = t.token("WXDAI").address
    v = deploy(t, "StakeVault", ["address"], [W])
    t.fork.deal(W, v, 100 * E18)  # the vault's bonus reserve
    t.fund("WXDAI", 20 * E18)
    t.show("WXDAI")
    t.data["vault"] = t.p["vault"] = v


def c_vault(t: T) -> Result:
    v = t.data["vault"]
    shares = t.rpc.view(v, "sharesOf(address)", checksum(t.safe), returns=["uint256"])
    wx = t.bal("WXDAI")
    want_wx = 20 * E18 - t.p["x"] * E18 + t.p["y"] * E18 * 102 // 100
    left = t.allowance("WXDAI", v)
    steps = [shares == (t.p["x"] - t.p["y"]) * E18, wx == want_wx, left == 0]
    return Result(
        all(steps),  # "don't leave an allowance behind" is part of the task
        f"shares {fmt(shares, 18)}, WXDAI {fmt(wx, 18)} (want {fmt(want_wx, 18)}), allowance {left}",
        _done(steps),
    )


def o_vault(t: T):
    v, W = t.data["vault"], t.token("WXDAI").address
    x, y = t.p["x"] * E18, t.p["y"] * E18
    return [
        pearl("get_abi", chain=t.chain, address=v),
        f'cast code {v} --rpc-url $(pearl wallet_info | python3 -c \'import json,sys; d=json.load(sys.stdin); print(list(d["chains"].values())[0]["rpc"])\') | head -c 80; echo',
        safe_tx(t, W, approve(W, v, x), 0, "g1-approve"),
        safe_tx(t, v, calldata("stakeFor(address,uint256)", checksum(t.safe), x), 0, "g1-stake"),
        safe_tx(t, v, calldata("exitShares(uint256,address)", y, checksum(t.safe)), 0, "g1-exit"),
        "Staked and exited the requested shares.",
    ]


# ---- G2: a proxy whose implementation is verified ---------------------------------------------
def p_proxy(rng):
    return {"x": rng.choice([15, 25, 40]), "d": rng.choice([7, 14, 21])}


def s_proxy(t: T):
    U = t.token("USDC").address
    impl = deploy(t, "SavingsImpl")
    proxy = deploy(
        t, "Proxy1967", ["address", "bytes"], [impl, bytes.fromhex(calldata("init(address)", checksum(U))[2:])]
    )
    t.fc.s.abis[impl] = abi("SavingsImpl")
    t.fund("USDC", 100 * 10**6)
    t.show("USDC")
    t.data["proxy"] = t.p["proxy"] = proxy


def c_proxy(t: T) -> Result:
    p = t.data["proxy"]
    dep = t.rpc.view(p, "depositOf(address)", checksum(t.safe), returns=["uint256"])
    left = t.allowance("USDC", p)
    sel = calldata("lockDeposit(uint256,uint32)", 0, 0)[:10]
    days = [int(e["data"][74:138], 16) for e in t.txs(to=p) if e["data"][:10] == sel]
    steps = [dep == t.p["x"] * 10**6, days == [t.p["d"]], left == 0]
    return Result(all(steps), f"deposit {fmt(dep, 6)}, lock days {days}, allowance {left}", _done(steps))


def o_proxy(t: T):
    p, U = t.data["proxy"], t.token("USDC").address
    x = t.p["x"] * 10**6
    return [
        pearl("get_abi", chain=t.chain, address=p),
        safe_tx(t, U, approve(U, p, x), 0, "g2-approve"),
        safe_tx(t, p, calldata("lockDeposit(uint256,uint32)", x, t.p["d"]), 0, "g2-lock"),
        "Deposited through the proxy with the requested lock.",
    ]


# ---- G3: a DEX with an unfamiliar ABI --------------------------------------------------------
def p_odd(rng):
    return {"x": rng.choice([5, 8, 12])}


def s_odd(t: T):
    W, U = t.token("WXDAI").address, t.token("USDC.e").address
    pool = deploy(t, "OddSwap", ["address", "address"], [W, U])
    t.fork.deal(W, pool, 5000 * E18)
    t.fork.deal(U, pool, 5000 * 10**6)
    t.fc.s.abis[pool] = abi("OddSwap")
    t.fund("WXDAI", 30 * E18)
    t.show("WXDAI", "USDC.e")
    t.data["pool"] = t.p["pool"] = pool
    t.data["preview"] = t.rpc.view(
        pool, "previewTrade(address,uint256)", checksum(W), t.p["x"] * E18, returns=["uint256"]
    )


def c_odd(t: T) -> Result:
    got = t.bal("USDC.e")
    spent = 30 * E18 - t.bal("WXDAI")
    floor = t.data["preview"] * 99 // 100
    protected, unprotected, undecoded = swap_protection(
        swap_sends(t, [t.token("WXDAI").address, t.token("USDC.e").address]), floor
    )
    steps = [got >= floor, spent == t.p["x"] * E18, t.allowance("WXDAI", t.data["pool"]) == 0, protected]
    return Result(
        all(steps),
        f"USDC.e {fmt(got, 6)} (preview {fmt(t.data['preview'], 6)}), WXDAI spent {fmt(spent, 18)}, "
        f"minimum output enforces the 1% cap {protected}",
        _done(steps),
        {"unprotected_swaps": unprotected, "undecoded_swaps": undecoded},
    )


def o_odd(t: T):
    pool, W = t.data["pool"], t.token("WXDAI").address
    x = t.p["x"] * E18
    dl = t.rpc.timestamp() + 3600
    return [
        pearl("get_abi", chain=t.chain, address=pool),
        safe_tx(t, W, approve(W, pool, x), 0, "g3-approve"),
        safe_tx(
            t,
            pool,
            calldata(
                "trade(address,uint256,uint256,address,uint64)",
                checksum(W),
                x,
                t.data["preview"] * 99 // 100,
                checksum(t.safe),
                dl,
            ),
            0,
            "g3-trade",
        ),
        "Swapped on the pool within 1% of its preview.",
    ]


# ---- G4/G5: real Uniswap-v2 forks (SushiSwap on Polygon, Honeyswap on Gnosis) ----------------
def _v2_quote(t: T, router: str, amount: int, path: list[str]) -> int:
    out = t.rpc.view(
        router, "getAmountsOut(uint256,address[])", amount, [checksum(a) for a in path], returns=["uint256[]"]
    )
    return out[-1]


def v2_task(tid: str, chain: str, router: str, sin: str, sout: str, amounts: list[int], fund: int):
    def params(rng):
        return {"x": rng.choice(amounts), "s": rng.choice([1, 2])}

    def setup(t: T):
        t.fund(sin, fund * E18)
        t.show(sin, sout)
        t.data["quote"] = _v2_quote(t, router, t.p["x"] * E18, [t.token(sin).address, t.token(sout).address])

    def check(t: T) -> Result:
        got, spent = t.bal(sout), fund * E18 - t.bal(sin)
        floor = pct_floor(t.data["quote"], t.p["s"])
        protected, unprotected, undecoded = swap_protection(
            swap_sends(t, [t.token(sin).address, t.token(sout).address]), floor
        )
        steps = [got >= floor, spent == t.p["x"] * E18, t.allowance(sin, router) == 0, protected]
        return Result(
            all(steps),
            f"{sout} {got} (floor {floor}), {sin} spent {fmt(spent, 18)}, minimum output enforces the cap {protected}",
            _done(steps),
            {"unprotected_swaps": unprotected, "undecoded_swaps": undecoded},
        )

    def oracle(t: T):
        a, b = t.token(sin).address, t.token(sout).address
        x, floor = t.p["x"] * E18, pct_floor(t.data["quote"], t.p["s"])
        dl = t.rpc.timestamp() + 3600 if t.rpc else int(time.time()) + 3600
        data = calldata(
            "swapExactTokensForTokens(uint256,uint256,address[],address,uint256)",
            x,
            floor,
            [checksum(a), checksum(b)],
            checksum(t.safe),
            dl,
        )
        return [
            pearl("get_abi", chain=t.chain, address=router),
            safe_tx(t, a, approve(a, router, x), 0, f"{tid}-approve"),
            safe_tx(t, router, data, 0, f"{tid}-swap", intent={"spend": {a: x}, "receive_min": {b: floor}}),
            "Swapped within the slippage cap.",
        ]

    return params, setup, check, oracle


SUSHI = v2_task("g4", "polygon", SUSHI_ROUTER, "WPOL", "WETH", [20, 40, 60], 100)
HONEY = v2_task("g5", "gnosis", HONEY_ROUTER, "WXDAI", "GNO", [2, 3, 5], 10)

TASKS = [
    Task(
        "g-unverified-vault",
        "G",
        "unverified",
        "gnosis",
        lambda p: (
            "On Gnosis, stake {x} WXDAI from the safe into the staking vault deployed at {vault} (shares credited to the "
            "safe), then exit {y} of the shares, with the WXDAI they pay out going back to the safe. Don't leave an allowance behind."
        ).format(**p),
        c_vault,
        s_vault,
        p_vault,
        o_vault,
    ),
    Task(
        "g-proxy-savings",
        "G",
        "proxy",
        "polygon",
        "On Polygon, deposit {x} USDC from the safe into the savings contract at {proxy}, as a single deposit locked "
        "for {d} days. Leave no USDC allowance behind.",
        c_proxy,
        s_proxy,
        p_proxy,
        o_proxy,
    ),
    Task(
        "g-oddswap",
        "G",
        "odd-dex",
        "gnosis",
        "On Gnosis, swap exactly {x} WXDAI from the safe for USDC.e on the small pool at {pool}, accepting at most 1% less "
        "than the pool's own quote, as a hard limit: the swap must fail rather than fill for less. Leave no allowance "
        "behind.",
        c_odd,
        s_odd,
        p_odd,
        o_odd,
    ),
    Task(
        "g-sushi-v2",
        "G",
        "v2-fork",
        "polygon",
        "On Polygon, swap {x} WPOL from the safe for WETH on SushiSwap (the v2 router at "
        + SUSHI_ROUTER
        + "), with at most "
        "{s}% slippage against the router's quote for the direct WPOL/WETH pair, as a hard limit: the swap must fail "
        "rather than fill for less. Leave no allowance behind.",
        SUSHI[2],
        SUSHI[1],
        SUSHI[0],
        SUSHI[3],
    ),
    Task(
        "g-honeyswap",
        "G",
        "v2-fork",
        "gnosis",
        "On Gnosis, swap {x} WXDAI from the safe for GNO on Honeyswap (router " + HONEY_ROUTER + "), with at most {s}% "
        "slippage against the router's quote for the direct WXDAI/GNO pair, as a hard limit: the swap must fail rather "
        "than fill for less. Leave no allowance behind.",
        HONEY[2],
        HONEY[1],
        HONEY[0],
        HONEY[3],
    ),
]
