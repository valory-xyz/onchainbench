"""Paths the real held-out set exercises, tested directly, so the public bench (with the one-task stand-in) still
reaches full coverage."""

import json
import time
import types

from onchainbench import review
from onchainbench.fork_connect import ForkConnect
from onchainbench.mock_connect.server import LLM_MECH, LOST_RESPONSE, SAFE, USDC, MockConnect, MockState


def test_review_pack_renders_whatever_held_out_set_is_installed():
    tasks = review.sample()
    assert tasks and all(t.split == "test" for t in tasks)
    assert review.render(tasks).count("**Verdict:**") == len(tasks)


def test_intent_gate_ignores_burns_but_not_undeclared_payees():
    fake = types.SimpleNamespace(_token_key=lambda k: k, _sym=lambda t: t)
    zero = "0x" + "00" * 20
    diff = {
        "ok": True,
        "tokens": {},
        "native": 0,
        "approvals": [],
        "recipients": [{"address": zero, "contract": False}, {"address": "0xabc", "contract": False}],
    }
    assert ForkConnect.check_intent(fake, diff, {"recipients": ["0xabc"]}) == []
    assert "undeclared-recipient" in ForkConnect.check_intent(fake, diff, {"recipients": []})[0]


def test_mock_usdc_transfer_unknown_rpc_and_a_settled_marketplace_request():
    m = MockConnect(MockState())
    try:
        before = m.s.usdc[SAFE]
        to = "0x" + "12" * 20
        data = "0xa9059cbb" + to[2:].rjust(64, "0") + hex(1_000_000)[2:].rjust(64, "0")
        m.call("safe_transaction", {"chain": "gnosis", "target": USDC, "value": 0, "data": data}, via="mcp")
        assert m.s.usdc[SAFE] == before - 1_000_000 and m.s.usdc[to] == 1_000_000
        assert m.rpc({"method": "eth_blockNumber", "id": 3})["error"]["message"] == "unsupported"
        m.s.connect_settle_delay = 0.01  # an off-chain request settles at the nonce it was signed at
        m.s.llm_backend = lambda msgs: "hi"
        args = {"prompt": json.dumps([{"role": "user", "content": "x"}]), "tool": "chat", "priority_mech": LLM_MECH}
        m.call("mech_request", {**args, "max_payment": 10**18, "chain": "gnosis"}, via="mcp")
        time.sleep(0.2)
        assert m.s.market_nonce == 1 and m.s.connect_settle_failures == 0
    finally:
        m.stop()


def test_mock_mcp_reports_a_lost_response_as_an_error(mock):
    import urllib.request

    mock.s.drop_first_response.add("wallet_info")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "wallet_info", "arguments": {}}}
    req = urllib.request.Request(
        mock.url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {mock.token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        res = json.loads(r.read())["result"]
    assert res["isError"] and res["content"][0]["text"] == LOST_RESPONSE


def test_review_pack_samples_every_block_then_at_random(monkeypatch):
    """The public stand-in has one task, so build a larger set: every block is drawn, then the rest at random."""

    def task(tid, block="A", split="test", variant_of=None):
        return types.SimpleNamespace(id=tid, block=block, split=split, variant_of=variant_of)

    # synthetic ids with digits, outside the held-out id shapes the leak check looks for
    tasks = [task(f"x1-{b}{i}", block=b) for b in "ABCDEFG" for i in range(3)]
    tasks += [task("x2-open", split="open"), task("x3-variant", variant_of="x1-A0")]
    monkeypatch.setattr(review, "all_tasks", lambda: tasks)
    picked = review.sample(n=10)
    assert len(picked) == 10 and {t.block for t in picked} == set("ABCDEFG")
    assert all(t.split == "test" and not t.variant_of for t in picked) and picked == sorted(picked, key=lambda t: t.id)
    assert len(review.sample(n=100)) == 21  # a smaller set gives what it has


def test_review_pack_handles_setup_filled_instructions_and_missing_source():
    import copy

    t = copy.copy(review.sample()[0])

    def needs_setup(p):
        raise KeyError("deployed address")

    t.text = needs_setup
    assert "[some values come from setup]" in review.render([t]) and review._src(None) == "(no source)"
