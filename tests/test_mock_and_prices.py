"""Mock Connect and model prices, against the bench's own modules."""

import json
import urllib.error
import urllib.request

import pytest

from onchainbench import pricing
from onchainbench.mock_connect.server import (
    DEAR_MECH,
    EOA,
    LLM_MECH,
    PRED_MECH,
    SAFE,
    USDC,
    MockConnect,
    MockState,
    ToolError,
)


def _http(url, token, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_openrouter_prices_and_register_price(monkeypatch):
    models = {"data": [{"id": "a/b", "pricing": {"prompt": "0.001", "completion": "0.002"}}]}
    eps = {
        "data": {
            "endpoints": [
                {
                    "provider_name": "Fireworks",
                    "tag": "fireworks/us",
                    "pricing": {"prompt": "0.009", "completion": "0.01"},
                },
                {
                    "provider_name": "Fireworks",
                    "tag": "fireworks",
                    "pricing": {
                        "prompt": "0.003",
                        "completion": "0.004",
                        "input_cache_read": "0.0001",
                        "input_cache_write": "0.005",
                    },
                },
            ]
        }
    }

    class Resp:
        def __init__(self, url):
            self.body = eps if "/endpoints" in url else models

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    pricing._PRICES.clear(), pricing._ENDPOINTS.clear()
    monkeypatch.setattr(pricing.urllib.request, "urlopen", lambda url, **k: Resp(url))
    import litellm

    # pinned: the endpoint whose tag is the slug (not fireworks/us), with its cache-read and cache-write prices
    assert pricing.price("openrouter/a/b?provider=fireworks") == {
        "input": 0.003,
        "output": 0.004,
        "cache_read": 0.0001,
        "cache_write": 0.005,
        "source": "endpoint fireworks",
    }
    assert pricing.price("openrouter/a/b?provider=fireworks/us")["source"] == "endpoint fireworks/us"
    listed = {"input": 0.001, "output": 0.002, "cache_read": 0.001, "cache_write": 0.001, "source": "list"}
    assert pricing.price("openrouter/a/b") == listed  # no cache prices: reads and writes at the input price
    assert pricing.price("openrouter/a/b?provider=nobody")["source"] == "list"  # no such provider: the list price
    assert pricing.price("anthropic/x") is None and pricing.price("openrouter/unknown") is None
    assert pricing.cost_from_tokens("openrouter/a/b?provider=fireworks", 100, 10, cached=40) == pytest.approx(
        60 * 0.003 + 40 * 0.0001 + 10 * 0.004
    )
    # input written to the cache at the write price (Anthropic, OpenAI): part of the prompt, like the cached input
    assert pricing.cost_from_tokens("openrouter/a/b?provider=fireworks", 100, 10, 40, 20) == pytest.approx(
        40 * 0.003 + 40 * 0.0001 + 20 * 0.005 + 10 * 0.004
    )
    assert pricing.cost_from_tokens("openrouter/unknown", 100, 10) == 0.0
    assert pricing.register_price("openrouter/a/b?provider=fireworks")  # caps bind during the trial
    assert litellm.cost_per_token(model="openrouter/a/b", prompt_tokens=1000, completion_tokens=10) == (
        pytest.approx(3.0),
        pytest.approx(0.04),
    )
    assert not pricing.register_price("openrouter/unknown")  # not listed: nothing registered, no error
    pricing._PRICES.clear(), pricing._ENDPOINTS.clear()
    monkeypatch.setattr(pricing.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert pricing.openrouter_prices() == {} and pricing.endpoints("a/b") == []
    assert pricing.billed({"cost": 0.5}) == 0.5 and pricing.billed(None) is None and pricing.billed({}) is None
    assert pricing.billed(type("U", (), {"cost": None, "model_extra": {"cost": 0.25}})()) == 0.25
    monkeypatch.setattr(pricing, "price", lambda s: {"input": 1.0, "output": 0.0, "cache_read": 0.0})
    assert pricing.cost_from_tokens("x", 10, 0, 0, 4) == 10.0  # prices saved before writes were priced


def test_parse_spec_splits_a_model_spec_into_its_name_and_options():
    """The model spec's options (`?provider=...&flag`), as the chat mech's backend reads them."""
    assert pricing.parse_spec("openrouter/x?provider=fireworks") == ("openrouter/x", {"provider": "fireworks"})
    assert pricing.parse_spec("openrouter/x") == ("openrouter/x", {})
    assert pricing.parse_spec("m?a=1&b=false&c&d=True&e=0&&f=x=y") == (
        "m",
        {"a": True, "b": False, "c": True, "d": True, "e": False, "f": "x=y"},
    )


def test_mock_connect_http_routes(mock):
    base = mock.url.removesuffix("/mcp/")
    assert _http(base + "/wallet", "bad")[0] == 401
    assert _http(base + "/nope", mock.token)[0] == 404
    assert _http(base + "/nope", mock.token, "POST", {})[0] == 404
    mock.s.drop_first_response.add("sign_message")
    assert _http(base + "/sign-message", mock.token, "POST", {"digest": "0x1"})[0] == 504
    assert _http(base + "/safe-transaction", mock.token, "POST", {"to": SAFE, "value": 1})[0] == 400


def test_mock_connect_tool_semantics(mock):
    m = mock
    with pytest.raises(ToolError, match="no-self-call"):
        m.call("safe_transaction", {"target": SAFE, "value": 1}, via="mcp")
    m.s.guardrail_mode, m.s.whitelist = "whitelist", {"0x" + "44" * 20}
    data = "0xa9059cbb" + ("44" * 20).rjust(64, "0") + hex(1)[2:].rjust(64, "0")
    assert m.call("preflight_transaction", {"target": USDC, "data": data}, via="mcp")["permitted"]
    refused = m.call("preflight_transaction", {"target": "0x" + "99" * 20, "value": 1}, via="mcp")
    assert not refused["permitted"] and "whitelist" in refused["rule"]
    m.s.guardrail_mode = "unrestricted"
    with pytest.raises(ToolError, match="insufficient native"):
        m.call("safe_transaction", {"target": "0x" + "11" * 20, "value": 10**30}, via="mcp")
    big = "0xa9059cbb" + ("11" * 20).rjust(64, "0") + hex(10**12)[2:].rjust(64, "0")
    with pytest.raises(ToolError, match="exceeds balance"):
        m.call("safe_transaction", {"target": USDC, "data": big}, via="mcp")
    approve = "0x095ea7b3" + ("11" * 20).rjust(64, "0") + hex(5)[2:].rjust(64, "0")
    out = m.call("safe_transaction", {"target": USDC, "data": approve, "request_id": "ap"}, via="mcp")
    assert m.call("safe_transaction", {"target": USDC, "data": approve, "request_id": "ap"}, via="mcp") == out
    assert m.s.ledger[-1]["decoded"]["erc20_approve"]["amount"] == 5
    m.s.native[EOA] = 10**18
    tx = m.call("send_transaction", {"to": "0x" + "11" * 20, "value": 1}, via="mcp")
    assert m.call("transaction_status", {"tx_hash": tx["tx_hash"]}, via="mcp")["status"] == "success"
    assert m.call("mech_tools", {}, via="mcp")["mechs"]
    with pytest.raises(ToolError, match="unknown mech"):
        m.call("mech_tools", {"priority_mech": "0x1"}, via="mcp")
    with pytest.raises(ToolError, match="no mech serves"):
        m.call("mech_request", {"prompt": "q", "tool": "nope"}, via="mcp")
    with pytest.raises(ToolError, match="refused before payment"):
        m.call("mech_request", {"prompt": "q", "tool": "superforecaster", "max_payment": 1}, via="mcp")
    with pytest.raises(ToolError, match="auto_deposit is off"):
        m.call("mech_request", {"prompt": "q", "tool": "prediction-online", "auto_deposit": False}, via="mcp")
    m.s.native[SAFE] = 0
    with pytest.raises(ToolError, match="too little xDAI"):
        m.call("mech_request", {"prompt": "q", "tool": "prediction-online"}, via="mcp")
    m.s.native[SAFE], m.s.prepaid = 2 * 10**18, 10**18
    with pytest.raises(ToolError, match="no backend"):
        m.call("mech_request", {"prompt": "q", "tool": "chat", "priority_mech": LLM_MECH}, via="mcp")
    m.s.llm_backend = lambda p: f"got {p}"
    r = m.call("mech_request", {"prompt": "plain text", "tool": "chat", "priority_mech": LLM_MECH}, via="mcp")
    assert "got plain text" in json.dumps(r)
    r = m.call(
        "mech_request",
        {"prompt": "q", "tool": "superforecaster", "priority_mech": DEAR_MECH, "max_payment": 10**18},
        via="mcp",
    )
    assert next(iter(r["delivery_results"].values()))["result"]["p_yes"] == 0.5
    again = {
        "prompt": "q",
        "tool": "superforecaster",
        "priority_mech": DEAR_MECH,
        "max_payment": 10**18,
        "request_id": "same",
    }
    first = m.call("mech_request", again, via="mcp")
    assert m.call("mech_request", again, via="mcp") == first  # same request_id: no second payment
    with pytest.raises(ToolError, match="no pending"):
        m.call("mech_result", {"request_id": "x"}, via="mcp")
    with pytest.raises(ToolError, match="unknown tool"):
        m.call("bogus", {}, via="mcp")


def test_mock_connect_settlement_failure_is_counted(mock):
    import time

    mock.s.connect_settle_delay = 0.05
    mock.s.prepaid = 10**18
    mock.call("mech_request", {"prompt": "q", "tool": "prediction-online", "priority_mech": PRED_MECH}, via="mcp")
    mock.s.market_nonce = 99  # another payer took the slot first
    time.sleep(0.2)
    assert mock.s.connect_settle_failures == 1


# ---- mock facilitator ------------------------------------------------------------------


def test_malformed_agent_input_gets_an_error_not_a_dropped_connection():
    """In an earlier comparison run, an intent's spend as a list crashed the handler, so the agent saw only a closed socket."""
    import json
    import urllib.request

    mock = MockConnect(MockState())

    def bad_call(tool, a, via):
        return a["intent"]["spend"].items()  # a list has no .items()

    mock.call = bad_call
    base = mock.url.removesuffix("/mcp/")

    def post(path, body, auth=True):
        h = {"Content-Type": "application/json"} | ({"Authorization": f"Bearer {mock.token}"} if auth else {})
        req = urllib.request.Request(base + path, json.dumps(body).encode(), h)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        args = {"intent": {"spend": ["0xabc", 1]}}
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "safe_transaction", "arguments": args},
        }
        code, r = post("/mcp", msg)
        assert code == 200 and r["result"]["isError"]
        assert "invalid request: AttributeError" in r["result"]["content"][0]["text"]
        code, r = post("/safe-transaction", args)
        assert code == 400 and r["detail"].startswith("invalid request")
        code, r = post("/rpc", "not an object", auth=False)
        assert code == 200 and r["error"]["code"] == -32600
        assert [e["where"] for e in mock.s.ledger if e["kind"] == "bad_input"] == [
            "safe_transaction",
            "safe_transaction",
            "rpc",
        ]
    finally:
        mock.stop()
