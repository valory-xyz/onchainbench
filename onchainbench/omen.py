"""Omen on the Gnosis fork (block B): binary markets created through the real ConditionalTokens and
FPMMDeterministicFactory, with a harness account as the oracle, so tasks can resolve markets on the fork.

Outcome 0 is "Yes", outcome 1 is "No" (Omen's convention). Collateral is WXDAI.
A market's YES price is set by the funding distribution: price_yes = b_no / (b_yes + b_no).
"""

from __future__ import annotations

from dataclasses import dataclass

from eth_utils import keccak

from onchainbench.evm import calldata, checksum, decode

E18 = 10**18


@dataclass
class Market:
    fpmm: str
    question: str
    question_id: str
    condition_id: str
    oracle: str
    collateral: str
    positions: tuple[int, int]  # ERC-1155 ids of YES and NO


def _ct(t) -> str:
    return t.env.fc.protocols["conditional_tokens"]


def create_market(
    t, question: str, p_yes: float, liquidity: int = 200 * E18, salt: int | None = None, fee: int = 2 * 10**16
) -> Market:
    fork, ct = t.fork, _ct(t)
    oracle = fork.accounts[2]
    W = t.env.fc.wrapped.address
    qid = "0x" + keccak(text=f"{question}:{salt}").hex()
    fork.as_(
        oracle, ct, calldata("prepareCondition(address,bytes32,uint256)", checksum(oracle), bytes.fromhex(qid[2:]), 2)
    )
    cid = "0x" + keccak(bytes.fromhex(oracle[2:]) + bytes.fromhex(qid[2:]) + (2).to_bytes(32, "big")).hex()
    creator = fork.accounts[1]
    fork.deal(W, creator, liquidity)
    factory = t.env.fc.protocols["fpmm_factory"]
    fork.as_(creator, W, calldata("approve(address,uint256)", checksum(factory), liquidity))
    k = 10**6
    hint = [round((1 - p_yes) * k), round(p_yes * k)]
    r = fork.as_(
        creator,
        factory,
        calldata(
            "create2FixedProductMarketMaker(uint256,address,address,bytes32[],uint256,uint256,uint256[])",
            salt if salt is not None else int(qid, 16) % 2**64,
            checksum(ct),
            checksum(W),
            [bytes.fromhex(cid[2:])],
            fee,
            liquidity,
            hint,
        ),
    )
    topic = (
        "0x" + keccak(text="FixedProductMarketMakerCreation(address,address,address,address,bytes32[],uint256)").hex()
    )
    fpmm = next(
        "0x" + decode(["address"], "0x" + log["data"][2:66])[0][2:].lower()
        for log in r["logs"]
        if log["topics"][0] == topic
    )
    pos = []
    for index_set in (1, 2):
        coll = t.rpc.view(
            ct,
            "getCollectionId(bytes32,bytes32,uint256)",
            b"\x00" * 32,
            bytes.fromhex(cid[2:]),
            index_set,
            returns=["bytes32"],
        )
        pos.append(t.rpc.view(ct, "getPositionId(address,bytes32)", checksum(W), coll, returns=["uint256"]))
    m = Market(fpmm.lower(), question, qid, cid, oracle, W, (pos[0], pos[1]))
    t.fc.s.abis[m.fpmm] = fpmm_abi(t)
    return m


def fpmm_abi(t) -> list | None:
    from onchainbench.fork_connect import load_abi

    master = t.rpc.view(t.env.fc.protocols["fpmm_factory"], "implementationMaster()", returns=["address"]).lower()
    return load_abi("gnosis", master)


def price_yes(t, m: Market) -> float:
    b_yes = position(t, m, 0, m.fpmm)
    b_no = position(t, m, 1, m.fpmm)
    return b_no / (b_yes + b_no)


def position(t, m: Market, idx: int, who: str | None = None) -> int:
    return t.rpc.view(
        _ct(t), "balanceOf(address,uint256)", checksum(who or t.safe), m.positions[idx], returns=["uint256"]
    )


def give_position(t, m: Market, idx: int, invest: int) -> int:
    """A harness trader buys outcome `idx` for `invest` WXDAI and hands the tokens to the Safe."""
    trader = t.fork.accounts[1]
    W, fork = m.collateral, t.fork
    fork.deal(W, trader, invest)
    fork.as_(trader, W, calldata("approve(address,uint256)", checksum(m.fpmm), invest))
    fork.as_(trader, m.fpmm, calldata("buy(uint256,uint256,uint256)", invest, idx, 0))
    amt = position(t, m, idx, trader)
    fork.as_(
        trader,
        _ct(t),
        calldata(
            "safeTransferFrom(address,address,uint256,uint256,bytes)",
            checksum(trader),
            checksum(t.safe),
            m.positions[idx],
            amt,
            b"",
        ),
    )
    return amt


def resolve(t, m: Market, yes: bool) -> None:
    payouts = [1, 0] if yes else [0, 1]
    t.fork.as_(
        m.oracle, _ct(t), calldata("reportPayouts(bytes32,uint256[])", bytes.fromhex(m.question_id[2:]), payouts)
    )


def market_card(m: Market, p: float | None = None, closes: str = "2026-12-31") -> str:
    return (
        f"Omen market {m.fpmm}\n  question: {m.question}\n  outcomes: 0 = Yes, 1 = No\n  collateral: WXDAI {m.collateral}\n"  # noqa: E501 - a literal command or config line
        f"  conditionId: {m.condition_id}\n  closes: {closes}\n"
    )
