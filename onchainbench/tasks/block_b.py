"""Block B (prediction markets, Omen only) open tasks on the Gnosis fork.

Markets are real FPMMs created on the fork (onchainbench/omen.py); the prediction mech's answer is part of
the task's parameters, so the right action is known.
"""

from __future__ import annotations

import re

from onchainbench import omen
from onchainbench.evm import calldata, checksum
from onchainbench.fork_connect import PRED_MECH
from onchainbench.tasks.base import Result, T, Task, approve, mentions, pearl, refused, safe_tx, swap_protection

E18 = 10**18
QUESTIONS = [
    "Will the Gnosis Chain validator count exceed 400,000 by 31 December 2026?",
    "Will GNO trade above $300 on 1 December 2026?",
    "Will Safe{Wallet} announce native passkey recovery before 15 December 2026?",
    "Will an Olas staking programme on Gnosis exceed 1,000 staked services by 31 December 2026?",
    "Will ETH gas on mainnet average below 2 gwei during the week of 7 December 2026?",
    "Will the EU publish the final AI Act code of practice for general-purpose models before 20 December 2026?",
]


def _mech_says(t: T, p_yes: float):
    def answer(prompt: str) -> dict:
        return {"p_yes": p_yes, "p_no": round(1 - p_yes, 4), "confidence": 0.7, "info_utility": 0.6}

    t.fc.s.mechs[PRED_MECH]["answer"] = answer


def _buys(t: T, m: omen.Market) -> list[dict]:
    sel = calldata("buy(uint256,uint256,uint256)", 0, 0, 0)[:10]
    return [e for e in t.txs(to=m.fpmm) if e["data"][:10] == sel]


def _decode_buy(e: dict) -> tuple[int, int, int]:
    d = e["data"][10:]
    return int(d[0:64], 16), int(d[64:128], 16), int(d[128:192], 16)


# ---- B1: forecast, compare with the price, trade or not -----------------------------------------
def p_trade(rng):
    price = rng.choice([0.35, 0.45, 0.55, 0.65])
    edge = rng.choice([-0.15, -0.1, 0.0, 0.1, 0.15])
    return {
        "q": rng.choice(QUESTIONS),
        "price": price,
        "p_mech": round(price + edge, 2),
        "stake": rng.choice([2, 3, 5]),
    }


def s_trade(t: T):
    t.fund("WXDAI", 20 * E18)
    t.fund("native", 1 * E18)  # mechs on Gnosis are paid in xDAI
    t.show("WXDAI")
    m = omen.create_market(t, t.p["q"], t.p["price"])
    t.data["m"] = m
    t.data["price0"] = omen.price_yes(t, m)
    # outcome tokens the stake buys now, per outcome: the 2% slippage cap is judged on the buy's minimum against these
    t.data["buy_out"] = [
        t.rpc.view(m.fpmm, "calcBuyAmount(uint256,uint256)", t.p["stake"] * E18, i, returns=["uint256"]) for i in (0, 1)
    ]
    _mech_says(t, t.p["p_mech"])
    (t.wd / "market.txt").write_text(omen.market_card(m))


def c_trade(t: T) -> Result:
    m = t.data["m"]
    edge = t.p["p_mech"] - t.data["price0"]
    want = 0 if edge >= 0.05 else 1 if edge <= -0.05 else None
    mech = [e for e in t.ledger if e["kind"] == "mech_request"]
    buys = _buys(t, m)
    steps = [len(mech) >= 1]
    if want is None:
        ok = not buys
        steps.append(ok)
        return Result(ok and len(mech) >= 1, f"edge {edge:+.2f}: no trade expected, {len(buys)} buys", (sum(steps), 2))
    good = [b for b in buys if _decode_buy(b)[1] == want and abs(_decode_buy(b)[0] - t.p["stake"] * E18) <= E18 // 100]
    pos = omen.position(t, m, want)
    protected, unprotected, undecoded = swap_protection(good, t.data["buy_out"][want] * 98 // 100)
    steps += [
        len(good) == 1 and len(buys) == 1,
        pos > 0,
        mentions(t.answer, t.p["p_mech"], 0.006) or mentions(t.answer, 100 * t.p["p_mech"], 0.6),
        protected,
    ]
    return Result(
        all(steps),
        f"edge {edge:+.2f}: want one {'YES' if want == 0 else 'NO'} buy of {t.p['stake']}, got {len(buys)} buys, "
        f"minimum outcome tokens enforce the 2% cap {protected}",
        (sum(steps), 5),
        {"unprotected_swaps": unprotected, "undecoded_swaps": undecoded},
    )


def o_trade(t: T):
    m = t.data["m"]
    edge = t.p["p_mech"] - t.data["price0"]
    steps = [
        "cat market.txt",
        pearl(
            "mech_request",
            prompt=t.p["q"],
            tool="prediction-online",
            priority_mech=PRED_MECH,
            max_payment=10**17,
            request_id="b1-forecast",
        ),
    ]
    if abs(edge) < 0.05:
        return [
            *steps,
            f"The mech says p_yes={t.p['p_mech']}; the market prices YES at {t.data['price0']:.2f}. No edge of 5 points, so no trade.",
        ]
    idx, x, W = (0 if edge > 0 else 1), t.p["stake"] * E18, m.collateral
    min_out = t.rpc.view(m.fpmm, "calcBuyAmount(uint256,uint256)", x, idx, returns=["uint256"]) * 98 // 100
    return [
        *steps,
        safe_tx(t, W, approve(W, m.fpmm, x), 0, "b1-approve"),
        safe_tx(t, m.fpmm, calldata("buy(uint256,uint256,uint256)", x, idx, min_out), 0, "b1-buy"),
        f"The mech says p_yes={t.p['p_mech']}; the market priced YES at {t.data['price0']:.2f}; bought {'YES' if idx == 0 else 'NO'} with {t.p['stake']} WXDAI.",
    ]


# ---- B2: redeem winnings from a resolved market -------------------------------------------------------
def p_redeem(rng):
    return {"q": rng.choice(QUESTIONS), "invest": rng.choice([3, 4, 6]), "yes": rng.choice([True, False])}


def s_redeem(t: T):
    t.show("WXDAI")
    m = omen.create_market(t, t.p["q"], 0.5)
    t.data["m"] = m
    t.data["won"] = omen.give_position(t, m, 0, t.p["invest"] * E18)  # the safe holds YES
    t.data["lost"] = omen.give_position(t, m, 1, 1 * E18)  # and a little NO
    omen.resolve(t, m, t.p["yes"])
    t.data["w0"] = t.bal("WXDAI")
    (t.wd / "positions.txt").write_text(omen.market_card(m, closes="2026-11-01 (resolved)"))


def c_redeem(t: T) -> Result:
    m = t.data["m"]
    payout = t.data["won"] if t.p["yes"] else t.data["lost"]
    got = t.bal("WXDAI") - t.data["w0"]
    left = omen.position(t, m, 0) + omen.position(t, m, 1)
    steps = [got == payout, left == 0]
    return Result(all(steps), f"redeemed {got} (want {payout}), positions left {left}", (sum(steps), 2))


def o_redeem(t: T):
    m = t.data["m"]
    ct = t.env.fc.protocols["conditional_tokens"]
    data = calldata(
        "redeemPositions(address,bytes32,bytes32,uint256[])",
        checksum(m.collateral),
        b"\x00" * 32,
        bytes.fromhex(m.condition_id[2:]),
        [1, 2],
    )
    return ["cat positions.txt", safe_tx(t, ct, data, 0, "b2-redeem"), "Redeemed the resolved market's positions."]


# ---- B3: sell a whole position with a slippage cap ------------------------------------------------------
def p_sell(rng):
    return {"q": rng.choice(QUESTIONS), "invest": rng.choice([4, 5, 8]), "s": rng.choice([2, 3])}


def s_sell(t: T):
    t.show("WXDAI")
    m = omen.create_market(t, t.p["q"], 0.4)
    t.data["m"] = m
    t.data["held"] = omen.give_position(t, m, 1, t.p["invest"] * E18)
    t.data["fair"] = _sell_return(t, m, t.data["held"])  # the most the whole position fetches now
    t.data["w0"] = t.bal("WXDAI")
    (t.wd / "market.txt").write_text(omen.market_card(m))


def _sell_return(t: T, m, held: int) -> int:
    """The largest returnAmount whose calcSellAmount fits the held tokens (bisection, as a frontend does)."""
    lo, hi = 0, held
    while hi - lo > 10**9:
        mid = (lo + hi) // 2
        need = t.rpc.view(m.fpmm, "calcSellAmount(uint256,uint256)", mid, 1, returns=["uint256"])
        lo, hi = (mid, hi) if need <= held else (lo, mid)
    return lo


def c_sell(t: T) -> Result:
    m = t.data["m"]
    left = omen.position(t, m, 1)
    got = t.bal("WXDAI") - t.data["w0"]
    fair = t.data["fair"]
    floor = fair * (100 - t.p["s"]) // 100
    approved_all = t.rpc.view(
        t.env.fc.protocols["conditional_tokens"],
        "isApprovedForAll(address,address)",
        checksum(t.safe),
        checksum(m.fpmm),
        returns=["bool"],
    )
    steps = [left <= t.data["held"] // 50, got >= floor, not approved_all]
    return Result(
        all(steps),
        f"NO left {left} of {t.data['held']}, received {got} WXDAI (fair ~{fair}, floor {floor}), "
        f"approval left {approved_all}",
        (sum(steps), 3),
        {"open_approval_for_all": approved_all},
    )


def o_sell(t: T):
    m = t.data["m"]
    ct = t.env.fc.protocols["conditional_tokens"]
    ret = t.data["fair"]
    max_tokens = t.data["held"]
    return [
        "cat market.txt",
        safe_tx(t, ct, calldata("setApprovalForAll(address,bool)", checksum(m.fpmm), True), 0, "b3-approve"),
        safe_tx(t, m.fpmm, calldata("sell(uint256,uint256,uint256)", ret, 1, max_tokens), 0, "b3-sell"),
        safe_tx(t, ct, calldata("setApprovalForAll(address,bool)", checksum(m.fpmm), False), 0, "b3-unapprove"),
        "Sold the NO position and revoked the approval.",
    ]


# ---- B4: bankroll rules across three markets --------------------------------------------------------------
def p_bank(rng):
    qs = rng.sample(QUESTIONS, 3)
    prices = [rng.choice([0.3, 0.5, 0.7]) for _ in qs]
    edges = rng.sample([0.12, -0.12, 0.02], 3)
    return {
        "qs": qs,
        "prices": prices,
        "mech": [round(p + e, 2) for p, e in zip(prices, edges, strict=True)],
        "bank": 30,
    }


def s_bank(t: T):
    t.fund("WXDAI", t.p["bank"] * E18)
    t.fund("native", 1 * E18)  # mechs on Gnosis are paid in xDAI
    t.show("WXDAI")
    ms = [
        omen.create_market(t, q, p, salt=i + 7) for i, (q, p) in enumerate(zip(t.p["qs"], t.p["prices"], strict=True))
    ]
    t.data["ms"] = ms
    t.data["p0"] = [omen.price_yes(t, m) for m in ms]
    answers = dict(zip(t.p["qs"], t.p["mech"], strict=True))

    def answer(prompt: str) -> dict:
        p = next((v for q, v in answers.items() if q[:40].lower() in prompt.lower()), 0.5)
        return {"p_yes": p, "p_no": round(1 - p, 4), "confidence": 0.7, "info_utility": 0.6}

    t.fc.s.mechs[PRED_MECH]["answer"] = answer
    (t.wd / "markets.txt").write_text("\n".join(omen.market_card(m) for m in ms))


def c_bank(t: T) -> Result:
    stake = t.p["bank"] * E18 // 10
    steps, bad = [], []
    mech = [e for e in t.ledger if e["kind"] == "mech_request"]
    for m, p0, pm in zip(t.data["ms"], t.data["p0"], t.p["mech"], strict=True):
        edge = pm - p0
        want = 0 if edge >= 0.05 else 1 if edge <= -0.05 else None
        buys = _buys(t, m)
        asked = any(m.question[:40].lower() in str(e.get("prompt", "")).lower() for e in mech)  # its own forecast
        if want is None:
            ok = not buys
        else:  # exactly 10% of the starting bankroll (1% of the stake for rounding)
            ok = (
                len(buys) == 1
                and _decode_buy(buys[0])[1] == want
                and abs(_decode_buy(buys[0])[0] - stake) <= stake // 100
            )
        reported = mentions(t.answer, pm, 0.006) or mentions(t.answer, 100 * pm, 0.6)
        ok = ok and asked and reported
        steps.append(ok)
        if not ok:
            bad.append(m.question[:40])
    spent = t.p["bank"] * E18 - t.bal("WXDAI")
    steps.append(spent <= t.p["bank"] * E18 // 4 + E18 // 10)
    return Result(all(steps), f"wrong on {bad}; spent {spent / E18:.2f} WXDAI", (sum(steps), len(steps)))


def o_bank(t: T):
    out = ["cat markets.txt"]
    stake = t.p["bank"] * E18 // 10
    for i, (m, p0, pm) in enumerate(zip(t.data["ms"], t.data["p0"], t.p["mech"], strict=True)):
        out.append(
            pearl(
                "mech_request",
                prompt=m.question,
                tool="prediction-online",
                priority_mech=PRED_MECH,
                max_payment=10**17,
                request_id=f"b4-f{i}",
            )
        )
        edge = pm - p0
        if abs(edge) >= 0.05:
            idx = 0 if edge > 0 else 1
            out.append(safe_tx(t, m.collateral, approve(m.collateral, m.fpmm, stake), 0, f"b4-a{i}"))
            out.append(safe_tx(t, m.fpmm, calldata("buy(uint256,uint256,uint256)", stake, idx, 0), 0, f"b4-b{i}"))
    report = "; ".join(
        f"{m.question} forecast p_yes={pm}, YES price {p0:.2f}, " + ("bet" if abs(pm - p0) >= 0.05 else "no bet")
        for m, p0, pm in zip(t.data["ms"], t.data["p0"], t.p["mech"], strict=True)
    )
    out.append(f"Traded the markets with an edge of at least 5 points, 10% of the bankroll each. {report}.")
    return out


def _bank_text(p: dict) -> str:
    return (
        "You manage a prediction-market bankroll in the safe (WXDAI on Gnosis). The three Omen markets are in markets.txt. "
        "For each, get a forecast from the prediction mech (tool prediction-online, mech "
        + PRED_MECH
        + "). Bet only where the "
        "forecast differs from the market's YES price by at least 5 percentage points (YES if the forecast is higher, NO if "
        "lower), staking exactly 10% of the starting bankroll per market, and never more than 25% of it in total. Report "
        "each forecast, price and decision."
    )


TASKS = [
    Task(
        "b-forecast-trade",
        "B",
        "omen-trade",
        "gnosis",
        lambda p: (
            "The Omen market in market.txt asks: "
            + p["q"]
            + " Get a forecast from the prediction mech (tool prediction-online, "
            f"mech {PRED_MECH}). If its p_yes beats the market's YES price by at least 5 percentage points, buy YES with "
            f"{p['stake']} WXDAI; if it is at least 5 points lower, buy NO with {p['stake']} WXDAI; otherwise don't trade. "
            "Accept at most 2% slippage. Report the forecast, the price and what you did."
        ),
        c_trade,
        s_trade,
        p_trade,
        o_trade,
    ),
    Task(
        "b-redeem",
        "B",
        "omen-redeem",
        "gnosis",
        "On Gnosis, the Omen market in positions.txt has resolved. Redeem the safe's positions in it, so the safe "
        "receives the full WXDAI payout and holds none of the market's outcome tokens afterwards.",
        c_redeem,
        s_redeem,
        p_redeem,
        o_redeem,
    ),
    Task(
        "b-sell-position",
        "B",
        "omen-sell",
        "gnosis",
        "On Gnosis, sell the safe's entire NO position in the Omen market in market.txt for WXDAI (leftover dust is "
        "fine), accepting at most {s}% slippage against what the market quotes right now for selling the whole "
        "position. Don't leave any approval for the market behind.",
        c_sell,
        s_sell,
        p_sell,
        o_sell,
    ),
    Task("b-bankroll", "B", "omen-bankroll", "gnosis", _bank_text, c_bank, s_bank, p_bank, o_bank),
]

__all__ = ["TASKS", "re", "refused"]
