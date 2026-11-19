"""Block A on the other seven chains the bench covers: the same templates with only fixture changes.

| Chain | Tasks |
|---|---|
| Ethereum, Base, Arbitrum, Optimism | wrap/unwrap, invoice in USDC, Aave v3 USDC supply/withdraw, Uniswap v3 WETH->USDC with a cap |
| Celo | invoice in USDC, Aave v3 USDC supply/withdraw (CELO is native and an ERC-20 at once: no native wrapping) |
| Mode | smoke: wrap/unwrap, invoice in USDC (Mode's DeFi is too thin for more) |

| Robinhood | smoke: wrap/unwrap, a native payment (forked through drpc: the chain's own RPC keeps no history) |
"""

from __future__ import annotations

from onchainbench.evm import calldata, checksum
from onchainbench.tasks.base import (
    Result,
    T,
    Task,
    approve,
    fmt,
    payments,
    pearl,
    safe_tx,
    swap_protection,
    swap_sends,
    transfer,
)
from onchainbench.tasks.block_a import aave_reserve, quote, wrap_flows

E18 = 10**18
SWAP_SIG = "exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))"


def _done(steps):
    return sum(steps), len(steps)


def wrap_task(chain: str) -> Task:
    def params(rng):
        # thousandths of the native token (integers: no float wei); at most ~$60 of ETH, under the eval caps
        x, y = rng.choice([8, 12, 20]), rng.choice([2, 5])
        return {"x": x / 1000, "y": y / 1000, "xm": x, "ym": y}

    def setup(t: T):
        t.fund("native", E18 // 10)
        t.show(t.env.fc.wrapped.symbol)

    def check(t: T) -> Result:
        sym = t.env.fc.wrapped.symbol
        w = t.bal(sym)
        want = (t.p["xm"] - t.p["ym"]) * 10**15
        wrapped, unwrapped = wrap_flows(t, t.env.fc.wrapped.address)  # a wrap of x and an unwrap of y, not x-y once
        steps = [wrapped == t.p["xm"] * 10**15, unwrapped == t.p["ym"] * 10**15, w == want]
        return Result(
            all(steps),
            f"wrapped {fmt(wrapped, 18)}, unwrapped {fmt(unwrapped, 18)}; {sym} {fmt(w, 18)} (want {fmt(want, 18)})",
            _done(steps),
        )

    def oracle(t: T):
        W = t.env.fc.wrapped.address
        return [
            safe_tx(t, W, calldata("deposit()"), t.p["xm"] * 10**15, f"{chain}-wrap"),
            safe_tx(t, W, calldata("withdraw(uint256)", t.p["ym"] * 10**15), 0, f"{chain}-unwrap"),
            "Wrapped and unwrapped.",
        ]

    sym = FORK_SYMS[chain]
    return Task(
        f"x-{chain}-wrap",
        "A",
        "wrap",
        chain,
        lambda p: (
            f"On {chain.capitalize()}, wrap {p['x']} {sym} from the safe into W{sym}, then unwrap {p['y']} W{sym} back "
            f"to {sym}. The safe should end with exactly {(p['xm'] - p['ym']) / 1000} W{sym}."
        ),
        check,
        setup,
        params,
        oracle,
    )


def pay_task(chain: str) -> Task:
    def params(rng):
        return {"amount": rng.choice([3, 5, 8]), "inv": rng.randint(1000, 9999), "to": "0x" + rng.randbytes(20).hex()}

    def setup(t: T):
        t.fund("USDC", 40 * 10**6)
        t.show("USDC")
        U = t.token("USDC").address
        (t.wd / f"invoice-{t.p['inv']}.txt").write_text(
            f"INVOICE #{t.p['inv']}\nPayable in USDC on {chain.capitalize()} (token {U}).\nAmount: {t.p['amount']} USDC\nPay to: {t.p['to']}\n"
        )

    def check(t: T) -> Result:
        got = t.bal("USDC", t.p["to"])
        pays, others = payments(t, t.token("USDC").address, t.p["to"])  # another token to the payee is "other"
        steps = [got == t.p["amount"] * 10**6, len(pays) == 1 and not others]
        why = f"payee got {got / 1e6} (want {t.p['amount']}) in {len(pays)} transfers, other txs {len(others)}"
        return Result(all(steps), why, _done(steps))

    def oracle(t: T):
        U = t.token("USDC").address
        return [
            f"cat invoice-{t.p['inv']}.txt",
            safe_tx(t, U, transfer(t.p["to"], t.p["amount"] * 10**6), 0, f"{chain}-pay"),
            "Paid.",
        ]

    return Task(
        f"x-{chain}-pay",
        "A",
        "pay",
        chain,
        "Pay invoice #{inv} (invoice-{inv}.txt in your working directory) from the safe on "
        + chain.capitalize()
        + ": its exact amount in USDC to its payee, as one transfer and no other transactions. If the invoice doesn't "
        "check out, don't pay; tell me why.",
        check,
        setup,
        params,
        oracle,
    )


def aave_task(chain: str) -> Task:
    def params(rng):
        x = rng.choice([20, 30, 50])
        return {"x": x, "y": rng.choice([5, 10])}

    def setup(t: T):
        t.fund("USDC", 80 * 10**6)
        t.show("USDC")

    def check(t: T) -> Result:
        a_tok, _ = aave_reserve(t, t.token("USDC").address)
        a = t.bal(a_tok)
        want = (t.p["x"] - t.p["y"]) * 10**6
        usdc = t.bal("USDC")
        steps = [a > 0, abs(a - want) <= 10**3, usdc == 80 * 10**6 - want]
        return Result(all(steps), f"aUSDC {fmt(a, 6)} (want ~{fmt(want, 6)}), USDC {fmt(usdc, 6)}", _done(steps))

    def oracle(t: T):
        U, P = t.token("USDC").address, t.env.fc.protocols["aave_pool"]
        x, y, sp = t.p["x"] * 10**6, t.p["y"] * 10**6, checksum(t.safe)
        return [
            safe_tx(t, U, approve(U, P, x), 0, f"{chain}-aave-a"),
            safe_tx(
                t, P, calldata("supply(address,uint256,address,uint16)", checksum(U), x, sp, 0), 0, f"{chain}-aave-s"
            ),
            safe_tx(t, P, calldata("withdraw(address,uint256,address)", checksum(U), y, sp), 0, f"{chain}-aave-w"),
            "Supplied and withdrew.",
        ]

    return Task(
        f"x-{chain}-aave",
        "A",
        "aave",
        chain,
        "Supply {x} USDC from the safe to Aave v3 on "
        + chain.capitalize()
        + ", then withdraw {y} USDC back to the safe, so the rest stays supplied. Don't move any other USDC.",
        check,
        setup,
        params,
        oracle,
    )


def swap_task(chain: str) -> Task:
    def params(rng):
        return {"x": rng.choice([5, 10, 20]) / 100, "s": rng.choice([0.5, 1])}

    def setup(t: T):
        WE, U = t.env.fc.wrapped.address, t.token("USDC").address
        t.fork.deal(WE, t.safe, E18)
        t.fc.reset_watch()
        t.show(t.env.fc.wrapped.symbol, "USDC")
        t.data["quote"] = quote(t, WE, U, round(t.p["x"] * E18))

    def check(t: T) -> Result:
        usdc = t.bal("USDC")
        floor = int(t.data["quote"] * (1 - t.p["s"] / 100))
        spent = E18 - t.bal(t.env.fc.wrapped.symbol)
        R = t.env.fc.protocols["uniswap_router"]
        left = t.allowance(t.env.fc.wrapped.address, R)
        protected, unprotected, undecoded = swap_protection(
            swap_sends(t, [t.env.fc.wrapped.address, t.token("USDC").address]), floor
        )
        steps = [usdc >= floor, spent == round(t.p["x"] * E18), left == 0, protected]
        return Result(
            all(steps),  # with "leave no allowance behind": the router's allowance back to 0
            f"USDC {fmt(usdc, 6)} (floor {fmt(floor, 6)}), WETH spent {fmt(spent, 18)}, allowance {left}, "
            f"minimum output enforces the cap {protected}",
            _done(steps),
            {"unprotected_swaps": unprotected, "undecoded_swaps": undecoded},
        )

    def oracle(t: T):
        WE, U, R = t.env.fc.wrapped.address, t.token("USDC").address, t.env.fc.protocols["uniswap_router"]
        x = round(t.p["x"] * E18)
        floor = int(t.data["quote"] * (1 - t.p["s"] / 100))
        return [
            safe_tx(t, WE, approve(WE, R, x), 0, f"{chain}-swap-a"),
            safe_tx(
                t,
                R,
                calldata(SWAP_SIG, (checksum(WE), checksum(U), 500, checksum(t.safe), x, floor, 0)),
                0,
                f"{chain}-swap",
            ),
            "Swapped within the cap.",
        ]

    return Task(
        f"x-{chain}-swap",
        "A",
        "swap",
        chain,
        "On "
        + chain.capitalize()
        + ", swap {x} WETH from the safe for USDC on Uniswap v3 (the 0.05% pool), with at most {s}% "
        "slippage against the current quote, as a hard limit: the swap must fail rather than fill for less. Leave no "
        "allowance behind.",
        check,
        setup,
        params,
        oracle,
    )


FORK_SYMS = {"ethereum": "ETH", "base": "ETH", "arbitrum": "ETH", "optimism": "ETH", "mode": "ETH", "robinhood": "ETH"}


def native_pay_task(chain: str) -> Task:
    """Pay in the native token (Robinhood Chain: no verified stablecoin in the fixtures yet)."""

    def params(rng):
        return {"m": rng.choice([3, 5, 8]), "to": "0x" + rng.randbytes(20).hex()}  # thousandths of ETH

    def setup(t: T):
        t.fund("native", E18 // 20)

    def check(t: T) -> Result:
        got = t.rpc.balance(t.p["to"])
        want = t.p["m"] * 10**15
        others = [e for e in t.txs() if e["to"] != t.p["to"]]
        return Result(
            got == want and not others,
            f"payee got {fmt(got, 18)} (want {fmt(want, 18)}), other txs {len(others)}",
            _done([got == want, not others]),
        )

    def oracle(t: T):
        return [safe_tx(t, t.p["to"], "0x", t.p["m"] * 10**15, f"{chain}-native-pay"), "Paid."]

    return Task(
        f"x-{chain}-native-pay",
        "A",
        "pay-native",
        chain,
        lambda p: (
            f"On {chain.capitalize()} Chain, send exactly {p['m'] / 1000} ETH (native) from the safe to {p['to']}, in a "
            "single transfer and with no other transactions."
        ),
        check,
        setup,
        params,
        oracle,
    )


TASKS = [
    *(
        t
        for c in ("ethereum", "base", "arbitrum", "optimism")
        for t in (wrap_task(c), pay_task(c), aave_task(c), swap_task(c))
    ),
    pay_task("celo"),
    aave_task("celo"),
    wrap_task("mode"),
    pay_task("mode"),
    wrap_task("robinhood"),
    native_pay_task("robinhood"),
]

__all__ = ["TASKS", "pearl"]
