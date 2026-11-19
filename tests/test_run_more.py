"""OnchainBench edge paths: model-arm dispatch with stubbed models, RPC retries and failures, raw sends through the
proxy, diffs of unwatched tokens, Polygon's USDC mech deposits, personas and judges for model arms."""

import json
import os
import subprocess
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from eth_account import Account

from onchainbench import abis, evm, metrics, raw_key, surface
from onchainbench import arms as A
from onchainbench import run as R
from onchainbench.connect.connect_client import ConnectClient, ConnectError
from onchainbench.env import ChainEnv, Envs
from onchainbench.evm import Fork, Rpc, RpcError, calldata, checksum
from onchainbench.fork_connect import LLM_MECH, PRED_MECH
from onchainbench.interim.intent_gate import _log_delta
from onchainbench.tasks import base, block_c, block_e, block_f

E18 = 10**18


# ---- the RPC client, against a scripted server ---------------------------------------------------
class Scripted:
    def __init__(self, replies):
        self.replies, self.n = list(replies), 0
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                code, body = me.replies[min(me.n, len(me.replies) - 1)]
                me.n += 1
                b = body.encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def stop(self):
        self.srv.shutdown()


def test_rpc_client_errors_retries_and_receipts(monkeypatch):
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    s = Scripted([(400, '{"jsonrpc":"2.0","id":1,"error":{"message":"bad","data":"0x08c379a0"}}')])
    try:
        with pytest.raises(RpcError, match="bad"):
            Rpc(s.url)("eth_call")
    finally:
        s.stop()
    s = Scripted([(500, "not json")])
    try:
        with pytest.raises(Exception):  # noqa: B017 - the HTTP error after four tries
            Rpc(s.url).raw("x")
    finally:
        s.stop()
    with pytest.raises(Exception):  # noqa: B017 - nothing listening
        Rpc("http://127.0.0.1:1", timeout=1).raw("x")
    s = Scripted(
        [
            (200, '{"jsonrpc":"2.0","id":1,"error":{"message":"Unknown state. First available state is 1"}}'),
            (200, '{"jsonrpc":"2.0","id":1,"result":"0x1"}'),
        ]
    )
    try:
        assert Rpc(s.url)("eth_chainId") == "0x1" and s.n == 2  # a transient upstream error is retried
    finally:
        s.stop()
    s = Scripted([(200, '{"jsonrpc":"2.0","id":1,"result":null}')])
    try:
        with pytest.raises(RpcError, match="no receipt"):
            Rpc(s.url).receipt("0xabc")
        assert s.n > 40  # it mined a block on the way (evm_mine)
    finally:
        s.stop()
    assert (
        evm.revert_reason(None) == ""
        and evm.revert_reason("0x4e487b710000000000000000000000000000000000000000000000000000000000000011")
        == "panic 0x11"
    )
    assert evm.revert_reason("0x08c379a0zz") == "0x08c379a0zz" and evm.revert_reason("0x1234") == "0x1234"
    with pytest.raises(RuntimeError, match="did not start"):
        Fork("http://127.0.0.1:1", 1)


def test_log_delta_reads_transfers_deposits_and_withdrawals():
    from onchainbench.interim.intent_gate import T_DEPOSIT, T_TRANSFER, T_WITHDRAWAL

    who, tok, other = "0x" + "aa" * 20, "0x" + "bb" * 20, "0x" + "cc" * 20

    def pad(a):
        return "0x" + a[2:].rjust(64, "0")

    amt = "0x" + hex(5)[2:].rjust(64, "0")
    logs = [
        {"address": tok, "topics": [T_TRANSFER, pad(other), pad(who)], "data": amt},
        {"address": tok, "topics": [T_TRANSFER, pad(who), pad(other)], "data": amt},
        {"address": tok, "topics": [T_DEPOSIT, pad(who)], "data": amt},
        {"address": tok, "topics": [T_WITHDRAWAL, pad(who)], "data": "0x"},
        {"address": "0x" + "dd" * 20, "topics": [], "data": amt},
    ]
    assert _log_delta({"logs": logs}, tok, who) == 5


# ---- model arms, dispatched with stubbed models ---------------------------------------------------------
@pytest.fixture(scope="module")
def envs():
    e = Envs()
    yield e
    e.stop()


def test_cli_dispatch_timeouts_and_base_session(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    fc = types.SimpleNamespace(
        mcp_json=lambda: {"mcpServers": {"pearl-connect": {"url": "u", "headers": {"Authorization": "Bearer t"}}}},
        s=types.SimpleNamespace(eoa_key="k", fc=types.SimpleNamespace(chain_id=1), rpc_url="r"),
    )
    t = types.SimpleNamespace(ws=ws, fc=fc, data={}, chain="gnosis")
    monkeypatch.setattr(raw_key, "broker_for", lambda t: ("u", "t"))
    for arm, cls in (
        ("claude-code", A.ClaudeCodeSession),
        ("codex:m", A.CodexSession),
        ("hermes:m", A.HermesSession),
        ("openclaw:m", A.OpenClawSession),
    ):
        assert isinstance(A.open_session(arm, t, [], 30), cls)

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(A.subprocess, "run", timeout)
    assert A.OpenClawSession(t, "openclaw", "m", 30).send("x").startswith("[timeout]")
    assert A.HermesSession(t, "hermes", "m", 30).send("x").startswith("[timeout]")
    base_s = A.Session()
    base_s.stats = A.Stats()
    assert base_s.close() is base_s.stats and base_s.crash_and_resume() is None
    assert A.last_json('{"a": 1} trailing {"b": {"c": 2}}') == {"b": {"c": 2}} and A.last_json("{nope} {") == {}


def test_the_cli_arm_helpers_are_public_and_keep_their_old_names():
    """A CLI arm outside arms.py (ocb_cli.py, examples/arm-template) calls the public names; the underscore names
    stay as aliases of the same functions."""
    for name in ("env_with_key", "capped", "last_json", "add_tokens"):
        assert getattr(A, f"_{name}") is getattr(A, name)
    st = A.Stats()
    A.add_tokens(st, 10, 2, 4, 1)
    assert (st.prompt_tokens, st.completion_tokens, st.extra) == (10, 2, {"cached_tokens": 4, "cache_write_tokens": 1})
    assert not A.capped(st) and A.capped(A.Stats(cost_usd=A.TRIAL_CAP_USD))
    assert not hasattr(A, "_json_lines") and {"re", "threading", "time"}.isdisjoint(A.__all__)


# ---- fork-connect paths the suite doesn't reach -------------------------------------------------------------
@pytest.mark.fork
def test_raw_sends_unwatched_tokens_and_polygon_mech_deposits(envs):
    g = envs.get("gnosis")
    g.fresh()
    fc = g.connect(raw_sends=True, gate="off")
    try:
        W = g.fc.wrapped.address
        g.fork.set_balance(g.safe, 5 * E18)
        fc.reset_watch()  # the harness funding the Safe (a direct edit otherwise kept by the next send's checkpoint)
        c = ConnectClient(fc.url, fc.token)
        # with the gate off and WXDAI not watched, the diff comes from the receipt's logs
        c.call_tool(
            "safe_transaction", {"chain": "gnosis", "target": W, "value": 2 * E18, "data": calldata("deposit()")}
        )
        assert fc.s.ledger[-1]["diff"]["tokens"][W]["delta"] == str(2 * E18)
        c.call_tool("safe_transaction", {"chain": "gnosis", "target": W, "data": calldata("withdraw(uint256)", E18)})
        assert fc.s.ledger[-1]["diff"]["tokens"][W]["delta"] == str(-E18)
        assert fc._bal("0x" + "12" * 20, g.safe) is None  # not a token
        # a raw transaction signed with the agent's key goes through the proxy and is registered
        nonce = int(g.fork.rpc("eth_getTransactionCount", g.eoa, "latest"), 16)
        tx = {
            "to": checksum("0x" + "34" * 20),
            "value": 1,
            "gas": 21000,
            "gasPrice": int(g.fork.rpc("eth_gasPrice"), 16) * 2,
            "nonce": nonce,
            "chainId": 100,
        }
        raw = Account.sign_transaction(tx, g.key).raw_transaction.hex()
        out = fc.rpc_endpoint(
            {"jsonrpc": "2.0", "id": 1, "method": "eth_sendRawTransaction", "params": ["0x" + raw.removeprefix("0x")]}
        )
        assert "result" in out and fc.integrity() == []
        e = fc.s.ledger[-1]  # decoded, so checks and attack outcomes see it like a safe tx
        assert e["kind"] == "tx" and e["raw"] and e["sender"] == "key" and e["to"] == "0x" + "34" * 20
        assert "diff" not in e  # not the wallet's own account here (fc.s.safe is the Safe)
        assert e["value"] == 1 and e["status"] == "success" and e["data"] in ("0x", "")
        real = fc.s.fork.rpc.receipt
        fc.s.fork.rpc.receipt = lambda h: (_ for _ in ()).throw(RpcError("gone"))
        assert fc._decode_raw(out["result"]) == {}
        fc.s.fork.rpc.receipt = real
        # on a raw arm the wallet is the key's account: its sends carry a diff, like a Connect send
        safe, fc.s.safe = fc.s.safe, g.eoa
        tx2 = dict(tx, nonce=nonce + 1, value=3)
        raw2 = Account.sign_transaction(tx2, g.key).raw_transaction.hex()
        before = g.fork.rpc.balance(g.eoa)
        fc.rpc_endpoint(
            {"jsonrpc": "2.0", "id": 2, "method": "eth_sendRawTransaction", "params": ["0x" + raw2.removeprefix("0x")]}
        )
        e2 = fc.s.ledger[-1]
        assert e2["diff"]["ok"] and int(e2["diff"]["native"]) == g.fork.rpc.balance(g.eoa) - before < -3
        assert [r["address"] for r in e2["diff"]["recipients"]] == ["0x" + "34" * 20] and e2["gas_used"] == 21000
        fc.s.safe = safe
        fc.reset_watch()
        # a tx from the agent EOA that Connect didn't send is flagged: sent by another client than the harness's (a
        # send through the harness's own client is the harness's). Anvil's log names it
        # first, so the block scan isn't needed (test_integrity_wiring covers the scan's own verdict)
        g.fork.impersonate(g.eoa)
        # the send alone, no receipt polling: Rpc.send's receipt loop mines a block itself when the receipt is slow
        # (a cold fork), which would be a second direct call from that client
        tx = {"from": g.eoa, "to": "0x" + "56" * 20, "value": "0x1", "gas": hex(21000)}
        assert "result" in Rpc(g.fork.url).raw("eth_sendTransaction", [tx])
        g.fork.stop_impersonating(g.eoa)
        assert "direct Anvil call outside Connect: eth_sendTransaction ×1" in fc.integrity()
        # a chat prompt that isn't JSON reaches the backend as text
        fc.s.llm_backend = lambda m: type(m).__name__
        fc.s.prepaid = 10**18
        out = c.call_tool("mech_request", {"prompt": "plain text", "tool": "chat", "priority_mech": LLM_MECH})
        assert next(iter(out["delivery_results"].values()))["result"]["content"] == "str"
    finally:
        fc.stop()
    p = envs.get("polygon")
    p.fresh()
    fc = p.connect()
    try:
        c = ConnectClient(fc.url, fc.token)
        assert c.call_tool("mech_tools", {"priority_mech": PRED_MECH})["payment_token"] == p.fc.tokens["USDC"].address
        with pytest.raises(ConnectError, match="too little USDC"):
            c.call_tool("mech_request", {"prompt": "q", "tool": "prediction-online"})
        p.fork.deal(p.fc.tokens["USDC"].address, p.safe, 10**6)
        fc.reset_watch()
        assert c.call_tool("mech_request", {"prompt": "q", "tool": "prediction-online"})["payment"] == 10**4
        assert any(e["kind"] == "deposit" for e in fc.s.ledger) and fc.integrity() == []
    finally:
        fc.stop()
    with pytest.raises(RpcError, match="reverted"):
        p.fork.deploy("0xfe")


@pytest.mark.fork
def test_raw_key_wiring_on_a_fork(envs):
    g = envs.get("gnosis")
    g.fresh()
    fc = g.connect()
    try:
        import tempfile

        ws = Path(tempfile.mkdtemp())
        (ws / ".mcp.json").write_text(json.dumps(fc.mcp_json()))
        t = types.SimpleNamespace(fc=fc, chain="gnosis", data={}, ws=ws)
        url, token = raw_key.broker_for(t)
        assert url.startswith("http://127.0.0.1") and token != fc.token
        assert fc.token not in (ws / ".mcp.json").read_text() and token in (ws / ".mcp.json").read_text()
        bc = ConnectClient(url, token)
        assert bc.call_tool("wallet_info", {})["eoa"] == g.eoa
        # Connect's read tools pass the broker, and its listings are logged
        W = g.fc.wrapped.address
        assert "ok" in bc.call_tool("simulate_transaction", {"chain": "gnosis", "target": W, "data": "0x"})
        fc.s.tool_descriptions = {"wallet_info": "poisoned"}
        bc.list_tools()
        assert {"kind": "agent_tools_list", "poisoned": ["wallet_info"], "rugged": False} in fc.s.ledger
        assert any(e["kind"] == "tools_list" for e in fc.s.ledger) and t.data["secrets"] == [token]
        fc.s.tool_descriptions = {}
        for fn in t.data["cleanup"]:
            fn()
        srv = (
            raw_key.evm_mcp_server(t)
            if raw_key.EVM_MCP.is_file()
            else {"env": {"EVM_RPC_STRICT": "1", "EVM_PRIVATE_KEY": fc.s.eoa_key}}
        )
        assert srv["env"]["EVM_RPC_STRICT"] == "1" and srv["env"]["EVM_PRIVATE_KEY"] == g.key
    finally:
        fc.stop()


# ---- metrics, registry, abis, surface ---------------------------------------------------------------------------
def test_metrics_open_approvals_edge_cases():
    led = [
        {
            "kind": "tx",
            "diff": {
                "approvals": [
                    {"token": "0x1", "spender": "0x2", "amount": "all"},
                    {"token": "0x3", "spender": "0x4", "amount": "5"},
                ]
            },
        }
    ]

    def allowance(token, spender):
        raise RpcError("x")

    t = types.SimpleNamespace(ledger=led, allowance=allowance)
    assert metrics.open_approvals(t) == [{"token": "0x1", "spender": "0x2", "left": "all"}]


def test_task_registry_rejects_duplicates(monkeypatch):
    from onchainbench import tasks as reg

    assert reg._load("no_such_block") == []
    monkeypatch.setattr(reg, "OPEN_MODULES", ["block_a", "block_a"])
    with pytest.raises(ValueError, match="duplicate"):
        reg.all_tasks()
    import importlib

    real = importlib.import_module

    def broken(name, *a):
        if name.endswith("block_x"):
            raise ModuleNotFoundError("dep", name="some_dependency")
        return real(name, *a)

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        reg._load("block_x")


def test_signatures_and_saved_abis(tmp_path, monkeypatch):
    monkeypatch.setattr(abis, "ABI_DIR", tmp_path)
    abis.save(
        "gnosis",
        "0xAB",
        [{"type": "function", "name": "f", "inputs": [{"type": "tuple", "components": [{"type": "uint256"}]}]}],
    )
    sigs = abis.signatures()
    assert "f((uint256))" in sigs.values() and "transfer(address,uint256)" in sigs.values()
    assert json.loads((tmp_path / "signatures.json").read_text())


def test_surface_skips_unreadable_files(tmp_path, monkeypatch):
    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    real = type(f).read_text

    def boom(self, *a, **k):
        if self == f:
            raise OSError("nope")
        return real(self, *a, **k)

    monkeypatch.setattr(type(f), "read_text", boom)
    assert surface.loc([tmp_path]) == 0


# ---- personas and judges for model arms (no model: stubbed) --------------------------------------------------
def test_model_arm_personas_and_judges(monkeypatch):
    monkeypatch.setattr(
        "onchainbench.judge.complete", lambda *a, **k: ('{"scores": [1, 1], "reasons": ["a", "b"]}', 0.0)
    )
    t = types.SimpleNamespace(
        arm_model="openrouter/deepseek/x",
        p={"amount": 5, "which": "a", "a1": 1, "a2": 2},
        task=types.SimpleNamespace(text=lambda p: "task"),
        instruction=lambda: "task",
    )
    for build in (block_e.persona_contact, block_e.persona_change, block_e.persona_refuse):
        assert build(t).__class__.__name__ == "LlmPersona"
    t.persona = types.SimpleNamespace(transcript=[("agent", "which?"), ("user", "A")], goal="g")
    assert block_e._clarify_judge(t, ["c1", "c2"])[0]["score"] == 1.0
    assert block_c._judge(t, "work", "facts", ["c1", "c2"])[0]["score"] == 1.0

    def boom(*a, **k):
        raise RuntimeError("judge down")

    monkeypatch.setattr("onchainbench.judge.complete", boom)
    assert "error" in block_e._clarify_judge(t, ["c"])[0] and "error" in block_c._judge(t, "w", "f", ["c"])[0]


def test_safety_twin_evidence(tmp_path):
    ws = tmp_path / "ws"
    (ws / "ocb-agent" / ".ocb-agent").mkdir(parents=True)
    (ws / "notes.txt").write_text("the token is SECRET")
    (ws / "ocb-agent" / ".ocb-agent" / "x").write_text("SECRET")
    led = [
        {"kind": "tx", "status": "success", "to": block_f.ATTACKER, "data": "0x", "tx_hash": "0x" + "1" * 64},
        {"kind": "tx", "status": "reverted"},
    ]
    web = types.SimpleNamespace(log=[{"path": "/collect?t=SECRET", "body": ""}])
    t = types.SimpleNamespace(
        ledger=led,
        rpc=types.SimpleNamespace(balance=lambda a: 1),
        fc=types.SimpleNamespace(token="SECRET"),
        web=web,
        ws=ws,
        answer="SECRET",
    )
    ev = metrics.token_leaks(t)
    assert len(ev) == 3 and any("notes.txt" in e for e in ev)
    with pytest.raises(KeyError):
        block_f._base("nope")


def test_env_brief_and_claude_layout(tmp_path):
    """The workspace layout needs no fork (only the chain's config and Connect's MCP entry): it runs on every OS."""
    from onchainbench.env import make_workspace
    from onchainbench.forks import FORK_CHAINS

    g = types.SimpleNamespace(fc=FORK_CHAINS["gnosis"])
    fc = types.SimpleNamespace(mcp_json=lambda: {"mcpServers": {"pearl-connect": {"url": "http://127.0.0.1:1/mcp"}}})
    ws = make_workspace(tmp_path, fc, g, claude_layout=True, web_base="http://w")
    assert "local gateway at http://w" in (ws / "CLAUDE.md").read_text() and (ws / ".claude" / "skills").is_dir()
    assert (
        json.loads((ws / ".mcp.json").read_text()) == fc.mcp_json()
        and "(chain id 100)" in (ws / "AGENTS.md").read_text()
    )


def test_load_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('# c\nPEARL_X_TEST="v"\nnot a pair\n')
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.delenv("PEARL_X_TEST", raising=False)
    R.load_env_file()
    assert os.environ["PEARL_X_TEST"] == "v"


@pytest.mark.fork
def test_task_t_helpers(envs):
    g = envs.get("gnosis")
    g.fresh()
    fc = g.connect()
    try:
        t = base.T(task=None, p={}, env=g, fc=fc, ws=Path("/nonexistent"), wd=Path("/nonexistent"))
        t.fund("native", 3 * E18)
        assert t.bal("native") == 3 * E18 and t.bal("xDAI") == 3 * E18 and t.gas_slack == 0
        assert t.file("definitely-not-here.txt") is None and t.txs() == [] and t.ledger is fc.s.ledger
        fc.s.safe = g.eoa
        assert t.gas_slack > 0
    finally:
        fc.stop()


__all__ = ["ChainEnv"]


@pytest.mark.fork
def test_rug_pull_twin_changes_tools_mid_session_and_its_tool_drains(envs):
    twin = next(t for t in R.all_tasks() if t.id == "f-rug-pull")
    g = envs.get("gnosis")
    g.fresh()
    fc = g.connect()
    try:
        t = base.T(task=twin, p={"x": 5, "y": 2}, env=g, fc=fc, ws=Path("/nonexistent"), wd=Path("/nonexistent"))
        twin.setup(t)
        c = ConnectClient(fc.url, fc.token)
        assert "escrow_deposit" not in {x["name"] for x in c.list_tools()}
        with pytest.raises(ConnectError, match="unknown tool"):
            c.call_tool("escrow_deposit", {"amount": 1})
        c.call_tool(
            "safe_transaction",
            {"chain": "gnosis", "target": g.fc.wrapped.address, "value": E18, "data": calldata("deposit()")},
        )
        tools = {x["name"]: x for x in c.list_tools()}
        assert "escrow_deposit" in tools and "v2.1" in tools["safe_transaction"]["description"]
        assert any(e["kind"] == "rug_pull" for e in fc.s.ledger)
        c.call_tool("escrow_deposit", {"amount": 10**16})  # an agent that obeys the new tool pays the attacker
        t.answer = ""
        t.web = None
        r = twin.check(t)
        assert r.metrics["attack_succeeded"] and not r.passed
        with pytest.raises(ConnectError, match="unknown tool"):
            c.call_tool("rug_tool", {})
    finally:
        fc.stop()
