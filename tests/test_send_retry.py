"""A send is never sent twice blind: after a lost answer `Rpc` asks the chain whether it landed (its hash, or "send
outcome unknown", an infra trial); only reads and idempotent harness setters are retried after a transport error."""

import json
import shutil
import socket
import subprocess
import threading
import time
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from eth_account import Account

from onchainbench import evm
from onchainbench import fork_connect as FC
from onchainbench.evm import Rpc, RpcError, SendOutcomeUnknown
from onchainbench.mock_connect.server import ToolError

needs_anvil = pytest.mark.skipif(shutil.which("anvil") is None, reason="needs Foundry's anvil")
ANVIL_KEY0 = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Anvil's default account 0


class Proxy:
    """A JSON-RPC server in front of `upstream` (or answering itself): `rule(req)` returns None to forward,
    "late" to forward and answer after the client gave up, "drop" to answer nothing, or a reply to send instead."""

    def __init__(self, upstream: str | None, rule=lambda req: None):
        self.rule, self.seen = rule, []
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                req = json.loads(body)
                proxy.seen.append(req["method"])
                what = proxy.rule(req)
                if what == "drop":
                    time.sleep(1.5)  # past the client's timeout, and nothing forwarded
                    return
                if isinstance(what, dict):
                    out = json.dumps({"jsonrpc": "2.0", "id": req["id"], **what}).encode()
                else:
                    r = urllib.request.Request(upstream, body, {"Content-Type": "application/json"})
                    out = urllib.request.urlopen(r, timeout=30).read()
                    if what == "late":
                        time.sleep(1.5)  # upstream already mined it; only the answer is late
                    elif callable(what):
                        out = json.dumps({"jsonrpc": "2.0", "id": req["id"], **what(json.loads(out))}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                except (BrokenPipeError, ConnectionResetError):  # the client gave up
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="module")
def anvil():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = subprocess.Popen(
        ["anvil", "--port", str(port), "--silent", "--accounts", "2", "--balance", "100"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    rpc = Rpc(url, timeout=1)
    for _ in range(100):
        try:
            rpc("eth_chainId")
            break
        except OSError:
            time.sleep(0.1)
    yield url
    p.terminate()
    p.wait(10)


def _first_send(kind):
    """A rule for the first `kind` request only."""
    done = []

    def rule(req):
        if req["method"] == kind and not done:
            done.append(1)
            return rule.what
        return None

    return rule


@needs_anvil
def test_a_send_whose_answer_is_late_mines_once_and_returns_its_hash(anvil):
    """The repro: a delaying proxy in front of Anvil, one rpc.send(): one transaction, the right hash."""
    rule = _first_send("eth_sendTransaction")
    rule.what = "late"
    proxy = Proxy(anvil, rule)
    try:
        rpc = Rpc(proxy.url, timeout=0.5)
        frm, to = rpc("eth_accounts")
        nonce0, bal0, b0 = rpc("eth_getTransactionCount", frm, "latest"), rpc.balance(to), rpc.block_number()
        r = rpc.send(frm, to, "0x", 10**18)
        assert int(rpc("eth_getTransactionCount", frm, "latest"), 16) - int(nonce0, 16) == 1  # one tx mined
        assert rpc.balance(to) - bal0 == 10**18 and rpc.block_number() == b0 + 1
        first = rpc("eth_getBlockByNumber", hex(b0 + 1), True)["transactions"][0]
        assert r["transactionHash"] == first["hash"]  # the caller gets the hash of the one that landed
        assert proxy.seen.count("eth_sendTransaction") == 1
    finally:
        proxy.stop()


@needs_anvil
def test_a_send_that_did_not_land_raises_send_outcome_unknown(anvil):
    rule = _first_send("eth_sendTransaction")
    rule.what = "drop"
    proxy = Proxy(anvil, rule)
    try:
        rpc = Rpc(proxy.url, timeout=0.5)
        frm, to = rpc("eth_accounts")
        nonce0 = rpc("eth_getTransactionCount", frm, "latest")
        with pytest.raises(SendOutcomeUnknown, match="send outcome unknown: eth_sendTransaction's answer was lost"):
            rpc.send(frm, to, "0x", 1)
        assert rpc("eth_getTransactionCount", frm, "latest") == nonce0  # never sent again
        assert proxy.seen.count("eth_sendTransaction") == 1
        assert isinstance(SendOutcomeUnknown("x"), RpcError)
    finally:
        proxy.stop()


@needs_anvil
def test_a_raw_send_is_found_by_its_hash(anvil):
    rpc0 = Rpc(anvil)
    to = evm.checksum(rpc0("eth_accounts")[1])
    acct = Account.from_key(ANVIL_KEY0)
    nonce = int(rpc0("eth_getTransactionCount", acct.address, "latest"), 16)
    tx = {"to": to, "value": 5, "gas": 21000, "gasPrice": 10**10, "nonce": nonce, "chainId": 31337}
    raw = "0x" + Account.sign_transaction(tx, ANVIL_KEY0).raw_transaction.hex().removeprefix("0x")
    rule = _first_send("eth_sendRawTransaction")
    rule.what = "late"
    proxy = Proxy(anvil, rule)
    try:
        out = Rpc(proxy.url, timeout=0.5).raw("eth_sendRawTransaction", [raw])
        assert rpc0("eth_getTransactionByHash", out["result"])["nonce"] == hex(nonce)
        assert proxy.seen.count("eth_sendRawTransaction") == 1
        # a raw send that never arrived: not on the chain, so its outcome is unknown
        tx2 = {**tx, "nonce": nonce + 1}
        raw2 = Account.sign_transaction(tx2, ANVIL_KEY0).raw_transaction.hex()
        rule2 = _first_send("eth_sendRawTransaction")
        rule2.what = "drop"
        proxy.rule = rule2
        with pytest.raises(SendOutcomeUnknown, match="isn't on the chain yet"):
            Rpc(proxy.url, timeout=0.5).raw("eth_sendRawTransaction", ["0x" + raw2.removeprefix("0x")])
    finally:
        proxy.stop()


@needs_anvil
def test_a_transient_error_on_a_send_is_retried_only_while_it_did_not_land(anvil, monkeypatch):
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    rpc0 = Rpc(anvil)
    frm, to = rpc0("eth_accounts")
    # the node refused it (nothing mined): sent again, mined once
    rule = _first_send("eth_sendTransaction")
    rule.what = {"error": {"code": -32603, "message": "failed to get account for 0x..."}}
    proxy = Proxy(anvil, rule)
    try:
        rpc = Rpc(proxy.url)
        b0 = rpc.block_number()
        assert rpc.send(frm, to, "0x", 1)["status"] == "0x1" and rpc.block_number() == b0 + 1
        assert proxy.seen.count("eth_sendTransaction") == 2
        # it landed but the answer came back as a transient error: not sent again, its hash returned
        rule = _first_send("eth_sendTransaction")
        rule.what = lambda out: {"error": {"code": -32603, "message": "error sending request: upstream"}}
        proxy.rule, proxy.seen = rule, []
        r = rpc.send(frm, to, "0x", 1)
        assert rpc.block_number() == b0 + 2 and proxy.seen.count("eth_sendTransaction") == 1
        assert r["blockNumber"] == hex(b0 + 2)
        # an explicit nonce: the mark is that nonce, no read before the send
        n = int(rpc("eth_getTransactionCount", frm, "latest"), 16)
        rule = _first_send("eth_sendTransaction")
        rule.what = "late"
        proxy.rule, proxy.seen = rule, []
        Rpc(proxy.url, timeout=0.5).send(frm, to, "0x", 1, nonce=n)
        assert proxy.seen[0] == "eth_sendTransaction" and rpc.block_number() == b0 + 3
    finally:
        proxy.stop()


@needs_anvil
def test_a_receipt_lookup_that_errors_before_the_send_is_mined_is_asked_again(anvil):
    """A fork answers a receipt lookup of a transaction it hasn't mined yet from its upstream RPC, and a public RPC
    sometimes fails it (CI: the oracle's d-daily-payments failed integrity). The error is not the send's: the
    transaction still lands, so the send polls on and returns its receipt instead of raising."""
    rule = _first_send("eth_getTransactionReceipt")
    rule.what = {"error": {"code": -32603, "message": "error code -32000: request timed out"}}  # not TRANSIENT
    proxy = Proxy(anvil, rule)
    try:
        rpc = Rpc(proxy.url)
        frm, to = rpc("eth_accounts")
        nonce0, b0 = int(rpc("eth_getTransactionCount", frm, "latest"), 16), rpc.block_number()
        r = rpc.send(frm, to, "0x", 1)
        assert r["status"] == "0x1" and r["transactionHash"].lower() in rpc.sent
        assert int(rpc("eth_getTransactionCount", frm, "latest"), 16) == nonce0 + 1 and rpc.block_number() == b0 + 1
        assert proxy.seen.count("eth_getTransactionReceipt") >= 2 and proxy.seen.count("eth_sendTransaction") == 1
    finally:
        proxy.stop()


def test_a_send_that_is_never_mined_has_an_unknown_outcome(monkeypatch):
    """Lookups that keep failing, as errors or lost answers, and a mine that fails: the send may still land, so its
    outcome is unknown (an infra trial), never a plain error the agent would read as "not sent"."""
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    rpc = Rpc("http://unused")
    polls = []

    def raw(method, params=None):
        polls.append(method)
        if polls.count(method) % 2:  # every other lookup's answer lost; the last one an error reply
            raise OSError("connection reset")
        return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "request timed out"}}

    def mine(self, method, *params):
        polls.append(method)
        raise RpcError("mine failed")

    monkeypatch.setattr(rpc, "raw", raw)
    monkeypatch.setattr(Rpc, "__call__", mine)
    with pytest.raises(SendOutcomeUnknown, match=r"no receipt for 0xabc \(last lookup: request timed out\)"):
        rpc.receipt("0xabc")
    assert polls.count("eth_getTransactionReceipt") == 200 and polls.count("evm_mine") == 1


def test_only_reads_and_idempotent_setters_are_retried(monkeypatch):
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    assert "eth_sendTransaction" not in evm.RETRY_METHODS and "evm_mine" not in evm.RETRY_METHODS
    assert {"eth_call", "eth_getBalance", "eth_blockNumber", "eth_chainId", "eth_estimateGas", "net_version"} <= (
        evm.RETRY_METHODS
    )
    assert "evm_snapshot" in evm.RETRY_METHODS and "anvil_setStorageAt" in evm.RETRY_METHODS
    proxy = Proxy(None, lambda req: "drop")
    try:
        rpc = Rpc(proxy.url, timeout=0.3)
        with pytest.raises(OSError):
            rpc.raw("evm_mine")
        assert proxy.seen == ["evm_mine"]  # one attempt: a mine done twice is two blocks
        proxy.seen.clear()
        with pytest.raises(OSError):
            rpc.raw("eth_blockNumber")
        assert proxy.seen == ["eth_blockNumber"] * 4
    finally:
        proxy.stop()
    # a transient error reply: the node did nothing, so any method is asked again (a read, a time jump, a revert)
    proxy = Proxy(None, lambda req: {"error": {"code": -1, "message": "Unknown state"}})
    try:
        rpc = Rpc(proxy.url)
        for method, params in (("evm_increaseTime", ["0x1"]), ("evm_revert", ["0x1"]), ("eth_getBalance", ["0x1"])):
            proxy.seen.clear()
            with pytest.raises(RpcError, match="Unknown state"):
                rpc(method, *params)
            assert proxy.seen == [method] * 6
    finally:
        proxy.stop()


@needs_anvil
def test_a_mine_answered_with_a_transient_error_is_retried_and_mines_once(anvil):
    """CI's block D warps: Anvil answered evm_mine with "failed to apply blockhash contract call: database error:
    failed to get storage" when the fork's upstream failed. An error reply means no block was mined: asked again."""
    msg = "failed to apply beacon root contract call at 0x00: database error: failed to get storage"
    rule = _first_send("evm_mine")
    rule.what = {"error": {"code": -32603, "message": msg}}
    proxy = Proxy(anvil, rule)
    try:
        rpc = Rpc(proxy.url)
        b0 = rpc.block_number()
        rpc("evm_mine")
        assert rpc.block_number() == b0 + 1 and proxy.seen.count("evm_mine") == 2
        # a lost answer to a mine is not repeated: it may have mined
        rule = _first_send("evm_mine")
        rule.what = "late"
        proxy.rule, proxy.seen = rule, []
        with pytest.raises(OSError):
            Rpc(proxy.url, timeout=0.5)("evm_mine")
        assert proxy.seen == ["evm_mine"] and rpc.block_number() == b0 + 2
    finally:
        proxy.stop()


def test_when_the_chain_cannot_tell_the_outcome_is_unknown(monkeypatch):
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    lost = OSError("timed out")
    rpc = Rpc("http://unused")

    def post(method, params, tries):
        raise lost

    monkeypatch.setattr(rpc, "_post", post)
    with pytest.raises(SendOutcomeUnknown, match="a malformed send"):
        rpc.raw("eth_sendRawTransaction", ["0xnothex"])  # its hash can't be computed: nothing to look for
    with pytest.raises(SendOutcomeUnknown, match="a malformed send"):
        rpc.raw("eth_sendTransaction", [{"to": "0x1"}])  # no sender
    frm = "0x" + "aa" * 20
    answers = {"eth_getTransactionCount": "0x6", "eth_blockNumber": "0x2"}  # the nonce moved past 5 ...
    block = {"transactions": [{"from": "0x" + "bb" * 20, "nonce": "0x5", "hash": "0xother"}]}

    def chain(self, m, *p):
        if isinstance(answers.get(m), Exception):
            raise answers[m]
        return block if m == "eth_getBlockByNumber" else answers[m]

    monkeypatch.setattr(Rpc, "__call__", chain)
    with pytest.raises(SendOutcomeUnknown, match=r"nonce 5 of 0xa+ was taken, but no recent block holds it"):
        rpc.raw("eth_sendTransaction", [{"from": frm, "nonce": "0x5"}])
    answers["eth_blockNumber"] = OSError("refused")  # ... and the chain can't be read
    with pytest.raises(SendOutcomeUnknown, match="the chain can't tell: refused"):
        rpc.raw("eth_sendTransaction", [{"from": frm, "nonce": "0x5"}])
    answers["eth_getTransactionByHash"] = RpcError("down")
    with pytest.raises(SendOutcomeUnknown, match="down"):
        rpc.raw("eth_sendRawTransaction", ["0x" + "ab" * 10])
    # a transient error that keeps coming while the chain shows nothing landed: given up after six tries
    answers["eth_getTransactionCount"] = "0x9"
    monkeypatch.setattr(rpc, "_post", lambda m, p, t: {"error": {"message": "Unknown state"}})
    assert rpc._send("eth_sendTransaction", [{"from": frm, "nonce": "0x9"}]) == {"error": {"message": "Unknown state"}}


def test_connect_marks_a_send_with_an_unknown_outcome_for_the_runner():
    """A safe_transaction or a relayed raw send whose outcome is unknown: recorded on the trial's state, so the
    runner makes the trial infra instead of letting integrity() blame the agent."""

    def unknown(*a):
        raise SendOutcomeUnknown("send outcome unknown: lost")

    s = types.SimpleNamespace(
        rug_pull=None,
        send_unknown=[],
        tool_calls=0,
        raw_sends=True,
        drop_first_response=set(),
        fork=types.SimpleNamespace(rpc=types.SimpleNamespace(raw=unknown)),
    )
    me = types.SimpleNamespace(s=s, t_safe_transaction=unknown, checkpoint=lambda: None, _log=lambda e: None)
    with pytest.raises(SendOutcomeUnknown):
        FC.ForkConnect.call(me, "safe_transaction", {}, "mcp")
    with pytest.raises(SendOutcomeUnknown):
        FC.ForkConnect.rpc_endpoint(me, {"method": "eth_sendRawTransaction", "params": ["0x"], "id": 1})
    assert s.send_unknown == [
        "safe_transaction: send outcome unknown: lost",
        "rpc eth_sendRawTransaction: send outcome unknown: lost",
    ]
    with pytest.raises(ToolError):
        FC.ForkConnect.call(me, "no_such_tool", {}, "mcp")
    assert FC.ForkState.__dataclass_fields__["send_unknown"].default_factory() == []
