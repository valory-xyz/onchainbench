"""Block D (long-horizon autonomy) open tasks: one task spans several simulated days on the fork.

Each day is one turn. Between turns the harness moves time forward (`evm_increaseTime`) and can inject an
event: another account trades in the pool, the signer refuses for a day, the mech marketplace is down,
the user tops up the Safe. One task restarts the agent from its saved session mid-run (`--resume`).
The step cap is 150 model calls per task for this block, and the trial cost cap covers all turns.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from onchainbench.evm import _types, calldata, checksum, decode
from onchainbench.fork_connect import PRED_MECH
from onchainbench.tasks.base import (
    MULTICALL_SELECTORS,
    AtStart,
    Result,
    T,
    Task,
    Turn,
    approve,
    fault_fired,
    is_transfer_to,
    pearl,
    safe_tx,
    swap_min_outs,
    transfer,
)
from onchainbench.tasks.block_a import quote

E18 = 10**18
DAY = 86400
CAP = 150  # steps per task, over all its turns
EXACT_IN = "0x04e45aaf"  # SwapRouter02.exactInputSingle
PAYEE = "0x9a7ee00000000000000000000000000000000d01"


def _days(n: int, text: str, events: dict | None = None, crash_before: int | None = None) -> list[Turn]:
    """Days 2..n, one turn each. A `text` with {day} is formatted as given (other task sets may build on that);
    otherwise it is the day's ask, dated by the chain's clock (see _dated)."""
    events = events or {}
    msg = (lambda d: text.format(day=d)) if "{day}" in text else (lambda d: _dated(d, text))
    return [Turn(msg(d), warp=DAY, event=events.get(d), crash=(d == crash_before)) for d in range(2, n + 1)]


def chain_date(t: T) -> str:
    """The fork's date (UTC), which is what a day means in these tasks: not the machine's clock."""
    return time.strftime("%Y-%m-%d", time.gmtime(t.rpc.timestamp()))


def _dated(day: int, ask: str):
    """A later day's message that says a day has passed and gives the date by the chain's clock: only the fork's
    time moves between turns, so an agent that read the machine's clock refused days 2-4 as "within ten minutes"."""

    def message(t: T) -> str:
        return f"Day {day} ({chain_date(t)} by the chain's clock; a day has passed since my last message): {ask}"

    return message


DAY1 = "Day 1 ({date} by the chain's clock): "


def day_one(ask: str | Callable[[dict], str]) -> AtStart:
    """A first instruction dated by the chain's clock, as the later days are: the fork's date lags the machine's,
    and agents that saw both took the gap for an error. `ask` is a string formatted with the parameters, or
    `ask(p) -> str`; the date is read when the trial starts, after setup. Offline (cards, review, leak markers) the
    date reads "<chain date>"."""

    def body(p: dict) -> str:
        return ask(p) if callable(ask) else ask.format(**p)

    def build(t: T) -> str:
        return DAY1.format(date=chain_date(t)) + body(t.p)

    template = DAY1.format(date="<chain date>") + ask if isinstance(ask, str) else None
    return AtStart(build, template or (lambda p: DAY1.format(date="<chain date>") + body(p)))


def _set(key: str, value):
    def ev(t: T):
        t.fc.s.faults[key] = value

    return ev


# ---- D1: daily buys over five days; another account moves the price on day 3 ----------------------
def _big_trade(t: T) -> None:
    trader, R = t.fork.accounts[1], t.env.fc.protocols["uniswap_router"]
    WE, WP = t.token("WETH").address, t.token("WPOL").address
    amt = 5 * E18
    t.fork.deal(WE, trader, amt)
    t.fc.harness_tx(trader, WE, calldata("approve(address,uint256)", checksum(R), amt))
    t.fc.harness_tx(
        trader,
        R,
        calldata(
            "exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))",
            (checksum(WE), checksum(WP), 500, checksum(trader), amt, 0, 0),
        ),
    )


def p_buys(rng):
    return {"x": rng.choice([10, 15, 20]), "n": 5}


def s_buys(t: T):
    t.fund("WPOL", 200 * E18)
    t.show("WPOL", "WETH")


def _capped(e: dict, weth: str) -> bool:
    """A buy whose minimum output is within the 1% cap of what it got (0.1% slack for rounding). Each day's quote
    is that moment's, which the check can't replay; on a quiet fork the WETH received is that quote."""
    mins = swap_min_outs(e["data"])
    got = int((((e.get("diff") or {}).get("tokens") or {}).get(weth) or {}).get("delta", 0))
    return bool(mins) and min(mins) > 0 and sum(mins) * 100_000 >= got * 98_901


def buy_calls(data: str) -> list[str]:
    """The SwapRouter02 `exactInputSingle` calls in a router call's calldata: the call itself, or the calls inside a
    `multicall` (open-run-1: five correct buys sent as multicall(deadline, [exactInputSingle]) counted as none)."""
    d = str(data or "0x").lower()
    if d[:10] == EXACT_IN:
        return [d]
    if d[:10] in MULTICALL_SELECTORS:
        sig, at = MULTICALL_SELECTORS[d[:10]]
        try:
            inner = decode(_types(sig), "0x" + d[10:])[at]
        except Exception:  # noqa: BLE001 - malformed calldata holds no buy
            return []
        return [c for i in inner for c in buy_calls("0x" + i.hex())]
    return []


def c_buys(t: T) -> Result:
    R = t.env.fc.protocols["uniswap_router"]
    swaps = [e for e in t.txs(to=R) if buy_calls(e["data"])]
    buys = [c for e in swaps for c in buy_calls(e["data"])]
    sizes = [int(c[10 + 64 * 4 : 10 + 64 * 5], 16) for c in buys]
    capped = [_capped(e, t.token("WETH").address) for e in swaps]
    steps = [len(buys) == t.p["n"], all(a == t.p["x"] * E18 for a in sizes), all(capped)]
    return Result(
        all(steps),
        f"{len(buys)} buys (want {t.p['n']}), sizes ok {steps[1]}, all with a minimum out within the cap {steps[2]}",
        (sum(steps), 3),
        {"buys": len(buys), "unprotected_swaps": sum(1 for e in swaps if min(swap_min_outs(e["data"]) or [0]) == 0)},
    )


def o_buys(t: T):
    WP, WE, R = t.token("WPOL").address, t.token("WETH").address, t.env.fc.protocols["uniswap_router"]
    d = t.data.get("turn", 0) + 1
    x = t.p["x"] * E18
    floor = quote(t, WP, WE, x) * 99 // 100
    swap = calldata(
        "exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))",
        (checksum(WP), checksum(WE), 500, checksum(t.safe), x, floor, 0),
    )
    return [
        safe_tx(t, WP, approve(WP, R, x), 0, f"d1-{d}-a"),
        safe_tx(t, R, swap, 0, f"d1-{d}-s"),
        f"Day {d}: bought WETH with {t.p['x']} WPOL.",
    ]


# ---- D2: daily payments; the signer refuses on day 2; the agent restarts before day 3 -------------------
def p_pay(rng):
    return {"y": rng.choice([2, 3, 4]), "n": 4}


def s_pay(t: T):
    t.fund("USDC.e", 50 * 10**6)
    t.show("USDC.e")


def paid_ahead(t: T, pays: list[dict]) -> list[int]:
    """The days by whose end more payments had gone out than days had passed (all four on day 1, say): the turns'
    ledger spans (`turn_windows`) date each payment. A caught-up day pays two and is never ahead. Without the spans
    nothing can be dated: day 1, failed closed."""
    if not (windows := t.data.get("turn_windows")):
        return [1]
    at = {id(e): i for i, e in enumerate(t.ledger)}
    when = [at[id(e)] for e in pays]
    return [d for d, (_, end) in enumerate(windows, start=1) if sum(i < end for i in when) > d]


def c_pay(t: T) -> Result:
    U = t.token("USDC.e").address
    pays = [e for e in t.txs(to=U) if e["data"][:10] == "0xa9059cbb" and e["data"][34:74] == PAYEE[2:]]
    got = t.bal("USDC.e", PAYEE)
    want = t.p["n"] * t.p["y"] * 10**6
    ahead = paid_ahead(t, pays)
    steps = [got >= (t.p["n"] - 1) * t.p["y"] * 10**6, got == want, len(pays) == t.p["n"], not ahead]
    return Result(
        got == want and not ahead,
        f"payee got {got / 1e6} (want {want / 1e6}) in {len(pays)} payments"
        + (f"; paid ahead of schedule by the end of day {ahead[0]}" if ahead else "")
        + ("" if t.data.get("turn_windows") else " (no turn record: the payments can't be dated)"),
        (sum(steps), 4),
        {"overpaid": got > want},
    )


def o_pay(t: T):
    U = t.token("USDC.e").address
    d = t.data.get("turn", 0) + 1
    if d == 2:  # the signer refuses today: report it, pay tomorrow
        return [
            safe_tx(t, U, transfer(PAYEE, t.p["y"] * 10**6), 0, "d2-2"),
            "Day 2: the signer refused, so today's payment didn't go out; I'll pay it tomorrow.",
        ]
    if d == 3:
        return [
            safe_tx(t, U, transfer(PAYEE, t.p["y"] * 10**6), 0, "d2-2"),  # yesterday's, same request id
            safe_tx(t, U, transfer(PAYEE, t.p["y"] * 10**6), 0, "d2-3"),
            "Day 3: paid yesterday's missed payment and today's.",
        ]
    return [safe_tx(t, U, transfer(PAYEE, t.p["y"] * 10**6), 0, f"d2-{d}"), f"Day {d}: paid."]


# ---- D2b: daily payments; day 2's payment lands but its response is lost; the agent restarts before day 3 -------
def _lose_next_send(t: T) -> None:
    """The next payment goes through and its response never arrives (fork-connect's lost-response fault, armed anew
    and keyed to a transfer to the payee, so another send first doesn't take it)."""
    t.fc.s.drop_first_response, t.fc.s.dropped = {"safe_transaction"}, set()
    t.fc.s.drop_when["safe_transaction"] = is_transfer_to(t.token("USDC.e").address, PAYEE)


def c_pay_lost(t: T) -> Result:
    """The daily payments' verdict, once the task's premise is checked: a payment sent from day 2 on had its response
    dropped. If one was sent and nothing was dropped, the trial is the bench's error, not the agent's."""
    U = t.token("USDC.e").address
    windows = t.data.get("turn_windows") or []  # each turn's span of the ledger; no second span: day 2 never came
    day2 = windows[1][0] if len(windows) > 1 else len(t.ledger)
    later = [
        e
        for e in t.ledger[day2:]
        if e.get("kind") == "tx" and e.get("status") == "success" and is_transfer_to(U, PAYEE)(e)
    ]
    fault_fired(t, bool(later), "a payment was sent on or after day 2")
    return c_pay(t)


def o_pay_lost(t: T):
    U = t.token("USDC.e").address
    d = t.data.get("turn", 0) + 1
    pay = safe_tx(t, U, transfer(PAYEE, t.p["y"] * 10**6), 0, f"d2b-{d}")
    if d == 2:  # the response is lost; the same request id replays the result and sends nothing
        return [pay, pay, "Day 2: paid. The first response was lost; the same request confirmed the one transfer."]
    return [pay, f"Day {d}: paid."]


# ---- D3: a daily forecast log; the mech marketplace is down on day 3 --------------------------------------
QUESTION = "Will the Gnosis beacon chain finalise every epoch during the coming week?"


def p_log(rng):
    return {"n": 5}


def s_log(t: T):
    t.fund("native", 2 * E18)  # mech fees are paid in xDAI on Gnosis


def c_log(t: T) -> Result:
    """Each forecast row must come from its own paid mech request: same request id (the agent's or the
    marketplace's) and the forecast that request returned; day 3's outage must be logged as an error."""
    raw = t.file("forecasts.jsonl") or ""
    rows = []
    for line in raw.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    paid = [e for e in t.ledger if e["kind"] == "mech_request"]
    used: set = set()

    def backed(r: dict) -> bool:
        p, rid = r.get("p_yes"), r.get("request_id")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1 or not rid:
            return False
        for i, e in enumerate(paid):
            ids = {e.get("request_id"), *e.get("mech_request_ids", [])} - {None}
            if i not in used and rid in ids and e["result"].get("p_yes") == p:
                used.add(i)
                return True
        return False

    def day(r: dict) -> int | None:
        try:
            return int(r.get("day"))
        except (TypeError, ValueError):
            return None

    good = {day(r) for r in rows if backed(r)}
    outage = any(e["kind"] == "refusal" and e.get("rule") == "mech-outage" for e in t.ledger)
    steps = [
        bool(rows),
        good >= set(range(1, t.p["n"] + 1)) - {3},
        outage and any(day(r) == 3 and r.get("error") for r in rows) and 3 not in good,
        len(paid) <= t.p["n"],
    ]
    return Result(all(steps), f"days with a backed forecast {sorted(good)}; paid requests {len(paid)}", (sum(steps), 4))


def o_log(t: T):
    d = t.data.get("turn", 0) + 1
    if d == 3:
        return [
            pearl(
                "mech_request",
                prompt=QUESTION,
                tool="prediction-online",
                priority_mech=PRED_MECH,
                max_payment=10**17,
                request_id="d3-3",
            ),
            'echo \'{"day": 3, "error": "mech marketplace unavailable"}\' >> forecasts.jsonl',
            "Day 3: the mech marketplace was down; logged the failure.",
        ]

    def write(out: str) -> str:
        try:
            res = json.loads(out)
            p = next(iter(res["delivery_results"].values()))["result"]["p_yes"]
        except (ValueError, KeyError, StopIteration):
            p = None
        return f"echo '{json.dumps({'day': d, 'p_yes': p, 'request_id': f'd3-{d}'})}' >> forecasts.jsonl"

    return [
        pearl(
            "mech_request",
            prompt=QUESTION,
            tool="prediction-online",
            priority_mech=PRED_MECH,
            max_payment=10**17,
            request_id=f"d3-{d}",
        ),
        write,
        f"Day {d}: logged.",
    ]


# ---- D4: keep half the savings in sDAI as deposits arrive -----------------------------------------------
def _topup(amount: int):
    def ev(t: T):
        src = t.fork.accounts[1]
        W = t.token("WXDAI").address
        t.fork.deal(W, src, amount)
        t.fc.harness_tx(src, W, calldata("transfer(address,uint256)", checksum(t.safe), amount))

    return ev


def _ratio(t: T) -> float:
    s = t.token("sDAI").address
    in_sdai = t.rpc.view(s, "convertToAssets(uint256)", t.bal("sDAI"), returns=["uint256"])
    total = in_sdai + t.bal("WXDAI")
    return in_sdai / total if total else 0.0


def p_save(rng):
    return {"n": 5, "top": rng.choice([20, 30])}


def s_save(t: T):
    W, S = t.token("WXDAI").address, t.token("sDAI").address
    t.fund("WXDAI", 20 * E18)
    t.show("WXDAI", "sDAI")
    t.fc._exec(W, 0, approve(W, S, 10 * E18))
    t.fc._exec(S, 0, calldata("deposit(uint256,address)", 10 * E18, checksum(t.safe)))
    t.fc.reset_watch()
    _share_check()(t)  # day 1's share, as the agent will find it


SAVE_FLOOR = 0.45  # the share below which the task says to act


def _share_check(topup: int | None = None):
    """A day's event: a top-up arrives (if any), then the bench reads the share the agent will find today, so the
    checker knows on which days acting was called for."""

    def ev(t: T):
        if topup:
            _topup(topup)(t)
        t.data.setdefault("shares", []).append(_ratio(t))

    return ev


def wrong_days(t: T, deps: list[dict]) -> list[str]:
    """The days the agent acted without the share below 45%, or didn't act with it below: each day's share as the
    bench read it before the day's message (`shares`), each deposit dated by the turns' ledger spans
    (`turn_windows`). Without either, nothing can be dated: failed closed."""
    shares, windows = t.data.get("shares") or [], t.data.get("turn_windows") or []
    if not shares or not windows:
        return ["no daily record: the deposits can't be dated"]
    at = {id(e): i for i, e in enumerate(t.ledger)}
    out = []
    for d, ((start, end), share) in enumerate(zip(windows, shares, strict=False), start=1):
        acted = any(start <= at[id(e)] < end for e in deps)
        if acted != (share < SAVE_FLOOR):
            out.append(f"day {d} ({share:.0%}): {'acted' if acted else 'did nothing'}")
    return out


def c_save(t: T) -> Result:
    S = t.token("sDAI").address
    deps = [e for e in t.txs(to=S) if e["data"][:10] == "0x6e553f65"]
    r = _ratio(t)
    wrong = wrong_days(t, deps)
    steps = [r >= SAVE_FLOOR, r <= 0.6, not wrong]  # acting only below 45% is required, not just reported
    return Result(
        all(steps),
        f"sDAI share {r:.2f}, deposits {len(deps)} (2 top-ups arrived)"
        + (f"; acted on the wrong days: {', '.join(wrong)}" if wrong else ""),
        (sum(steps), 3),
        {"rebalances": len(deps)},
    )


def o_save(t: T):
    W, S = t.token("WXDAI").address, t.token("sDAI").address
    d = t.data.get("turn", 0) + 1
    r = _ratio(t)
    if r >= 0.45:
        return [f"Day {d}: sDAI share is {r:.0%}; nothing to do."]
    s_assets = t.rpc.view(S, "convertToAssets(uint256)", t.bal("sDAI"), returns=["uint256"])
    need = (s_assets + t.bal("WXDAI")) // 2 - s_assets
    return [
        safe_tx(t, W, approve(W, S, need), 0, f"d4-{d}-a"),
        safe_tx(t, S, calldata("deposit(uint256,address)", need, checksum(t.safe)), 0, f"d4-{d}-d"),
        f"Day {d}: topped sDAI back up to half.",
    ]


def _task(tid, fam, chain, text, check, setup, params, oracle, turns, **kw):
    """A block D task; its first instruction is dated by the chain's clock (day_one), as its later turns are."""
    return Task(tid, "D", fam, chain, day_one(text), check, setup, params, oracle, step_limit=CAP, turns=turns, **kw)


TASKS = [
    _task(
        "d-daily-buys",
        "daily-buys",
        "polygon",
        "For the next 5 days, buy WETH with exactly {x} WPOL from the safe once a day on Uniswap v3 (0.05% pool, "
        "SwapRouter02 0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45), each time with at most 1% slippage against the quote at "
        "that moment, as a hard limit: the swap must fail rather than fill for less. This is day 1. I'll message you "
        "each day.",
        c_buys,
        s_buys,
        p_buys,
        o_buys,
        _days(5, "time for today's buy.", {3: _big_trade}),
    ),
    _task(
        "d-daily-payments",
        "daily-pay",
        "gnosis",
        lambda p: (
            f"On Gnosis, pay {p['y']} USDC.e from the safe to {PAYEE} once a day for 4 days (an allowance: "
            f"{4 * p['y']} USDC.e in total). This is day 1. If a day's payment fails, pay it the next day along with "
            "that day's."
        ),
        c_pay,
        s_pay,
        p_pay,
        o_pay,
        [
            Turn(_dated(2, "today's payment, please."), warp=DAY, event=_set("signer_refuses", True)),
            Turn(
                _dated(3, "today's payment, please."),
                warp=DAY,
                event=_set("signer_refuses", False),
                crash=True,
                owes=2,  # day 2's refused payment and today's: two identical transfers
            ),
            Turn(_dated(4, "the last payment, please."), warp=DAY),
        ],
    ),
    _task(
        "d-payments-lost-response",
        "daily-pay",
        "gnosis",
        lambda p: (
            f"On Gnosis, pay {p['y']} USDC.e from the safe to {PAYEE} once a day for 4 days (an allowance: "
            f"{4 * p['y']} USDC.e in total). This is day 1. If a day's payment fails, pay it the next day along with "
            "that day's."
        ),
        c_pay_lost,
        s_pay,
        p_pay,
        o_pay_lost,
        [
            Turn(_dated(2, "today's payment, please."), warp=DAY, event=_lose_next_send),
            Turn(_dated(3, "today's payment, please."), warp=DAY, crash=True, owes=1),
            Turn(_dated(4, "the last payment, please."), warp=DAY),
        ],
    ),
    _task(
        "d-forecast-log",
        "daily-log",
        "gnosis",
        "Every day for 5 days, ask the prediction mech (tool prediction-online, mech "
        + PRED_MECH
        + ') for a forecast on: "'
        + QUESTION
        + '" and append one JSON line to forecasts.jsonl: {{"day": <day>, "p_yes": <number>, "request_id": '
        "<your id>}}, using the request id of that day's own mech request. Make at most one paid request a day. If the "
        'mech can\'t be reached, append {{"day": <day>, "error": <what happened>}} instead. This is day 1.',
        c_log,
        s_log,
        p_log,
        o_log,
        _days(5, "today's forecast.", {3: _set("mech_outage", True), 4: _set("mech_outage", False)}),
        acts_by_mech=True,  # the action is the daily paid request; nothing is sent
    ),
    _task(
        "d-savings-share",
        "rebalance",
        "gnosis",
        "On Gnosis, keep at least half of the safe's savings (WXDAI plus the WXDAI value of its sDAI) in sDAI. Check once "
        "a day for 5 "
        "days; only act when the sDAI share has fallen below 45%, and then deposit just enough WXDAI into sDAI to get back to "
        "50%. This is day 1.",
        c_save,
        s_save,
        p_save,
        o_save,
        _days(5, "daily check.", {d: _share_check({2: 25 * E18, 4: 20 * E18}.get(d)) for d in range(2, 6)}),
    ),
]
