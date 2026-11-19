"""What reaches an agent reads as production: no brief, skill, tool description, refusal, server name or environment
variable says or implies that the session is a benchmark (METHODOLOGY, "How tasks reach the agent")."""

import json
import re
import urllib.error
import urllib.request

import pytest

from onchainbench import arms as A
from onchainbench import env, fork_connect, raw_key
from onchainbench.connect.broker import Broker
from onchainbench.connect.policy import Policy, SessionCaps
from onchainbench.mock_connect import server as mock
from onchainbench.web import Web

TELLS = re.compile(
    r"\b(evals?|evaluat\w*|benchmark\w*|bench|onchainbench|ocb|harness|tests?|testing|mock\w*|trials?|fork\w*|"
    r"anvil|snapshot|grader|checker)\b",
    re.I,
)


def agent_texts() -> dict[str, str]:
    """Every fixed text an agent is shown, by where it comes from."""
    texts = {
        "BRIEF": env.BRIEF,
        "WEB_NOTE": env.WEB_NOTE,
        "SKILL": env.SKILL,
        "RAW_BRIEF": raw_key.RAW_BRIEF,
        "WEI_DESCRIPTION": fork_connect.WEI_DESCRIPTION,
        "no-delegatecall": mock.GUARD_NO_DELEGATECALL,
        "no-self-call": mock.GUARD_NO_SELF_CALL,
        "whitelist": mock.guard_not_allowed("0x" + "ab" * 20, "gnosis"),
        "serverInfo": json.dumps(mock.SERVER_INFO),
    }
    for where, tools in (("fork", fork_connect.TOOLS), ("mock", mock.TOOLS)):
        texts |= {f"{where}:{name}": description for name, description, _ in tools}
    return texts


@pytest.mark.parametrize("where", sorted(agent_texts()))
def test_no_agent_text_says_it_is_a_benchmark(where):
    found = TELLS.findall(agent_texts()[where])
    assert not found, f"{where} says {found}"


def test_the_brief_keeps_what_checkers_and_tasks_rely_on():
    """The meaning the oracle, the checkers and the web tasks rely on survives the rewording."""
    text = env.BRIEF + env.WEB_NOTE.format(base="http://w")
    needles = "wallet_info request_id intent simulate_transaction chain_call get_abi decode_calldata mech_tools"
    for needle in [*needles.split(), "priority_mech", "guardrail", "data, not instructions", "http://w/search?q="]:
        assert needle in text, needle
    assert "http://w/fetch?url=" in text
    assert env.SKILL.startswith("---\nname: pearl-connect\n")


def test_guardrail_refusals_keep_the_rule_prefix():
    for text in (mock.GUARD_NO_DELEGATECALL, mock.GUARD_NO_SELF_CALL, mock.guard_not_allowed("0x1", "gnosis")):
        assert text.startswith("guardrail: rule '")
    assert "ask the operator" in mock.guard_not_allowed("0x1", "gnosis")


@pytest.mark.parametrize("mode", ["unrestricted", "whitelist"])
def test_wallet_info_keeps_the_guardrail_mode_from_the_agent(mode):
    """As Connect does (EXPOSE_MODE_TO_AGENT=False): the guard enforces the mode, the agent isn't told it exists."""
    m = mock.MockConnect(mock.MockState(guardrail_mode=mode, whitelist={"0x" + "ab" * 20}))
    try:
        out = json.dumps(m.call("wallet_info", {}, via="mcp"))
        assert not re.search(r"guardrail|mode|whitelist|unrestricted|intent_gate", out), out
    finally:
        m.stop()


def _initialize(url: str, token: str):
    req = urllib.request.Request(
        url,
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["result"]["serverInfo"], r.headers["Mcp-Session-Id"]


def test_servers_introduce_themselves_as_connect():
    """Fork Connect (the mock's HTTP layer) and the bench's broker: Connect's name, an opaque session id."""
    m, other = mock.MockConnect(), mock.MockConnect()
    b = Broker(None, Policy(SessionCaps())).start()
    try:
        assert m.session_id != other.session_id and len(m.session_id) == len(b.session_id) == 32
        assert _initialize(m.url, m.token) == ({"name": "connect", "version": "0.1.0"}, m.session_id)
        assert _initialize(b.url, b.token) == ({"name": "connect", "version": "0.1.0"}, b.session_id)
    finally:
        for s in (m, other, b):
            s.stop()


def test_the_web_gateway_404_says_nothing_of_a_snapshot():
    w = Web({})
    try:
        for path in ("/fetch?url=nope.example", "/elsewhere"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(w.base + path)
            assert not TELLS.search(e.value.read().decode())
    finally:
        w.stop()


def test_an_agent_environment_has_none_of_the_bench_settings(monkeypatch):
    monkeypatch.setenv("ONCHAINBENCH_HELDOUT", "/somewhere/heldout")
    monkeypatch.setenv("ONCHAINBENCH_AGENT_ISOLATION", "1")
    monkeypatch.setenv("SOME_TOOL_SETTING", "kept")
    out = A.scrubbed_environ()
    assert not [k for k in out if k.startswith("ONCHAINBENCH_")] and out["SOME_TOOL_SETTING"] == "kept"
