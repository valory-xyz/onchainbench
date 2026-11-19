"""The integrity check's second half: Anvil logs every request's method, the harness
counts its own admin calls, and any admin call in the log beyond the harness's is a direct call to Anvil's port.
No fork here; the fork test is in test_run_forks.py."""

import re
import sys
import types
from collections import Counter

import pytest

from onchainbench import arms as A
from onchainbench import evm, scan
from onchainbench.evm import Fork, admin_calls_in_log, is_admin, surplus_calls

LOG = """Listening on 127.0.0.1:8545
anvil_setBalance
eth_call
evm_increaseTime\r
anvil_mine

    Block Number: 1
    Block Hash: 0x97df
    Block Time: "Sat, 14 Nov 2026 17:51:28 +0000"

eth_sendRawTransaction
RPC request failed:
    Request: EthSendRawTransaction(0x00)
anvil_setBalance
evm_snapshot
"""


def test_admin_methods_and_the_log_parse():
    assert is_admin("anvil_setStorageAt") and is_admin("evm_mine") and is_admin("hardhat_setBalance")
    assert is_admin("debug_setHead") and is_admin("eth_sendTransaction") and is_admin("eth_sendUnsignedTransaction")
    assert not is_admin("eth_call") and not is_admin("debug_traceTransaction") and not is_admin("eth_getBalance")
    assert admin_calls_in_log(LOG) == Counter(
        {"anvil_setBalance": 2, "evm_increaseTime": 1, "anvil_mine": 1, "eth_sendRawTransaction": 1, "evm_snapshot": 1}
    )
    # Anvil logs hardhat_* under its own name; indented lines and prose are not methods
    assert admin_calls_in_log("hardhat_setBalance\n  anvil_mine\nanvil_mine now\n") == Counter({"anvil_setBalance": 1})


def test_only_a_surplus_over_the_harness_counts():
    logged = Counter({"anvil_setBalance": 3, "evm_mine": 1, "evm_snapshot": 2})
    harness = Counter({"anvil_setBalance": 2, "evm_mine": 1, "evm_snapshot": 4})  # a harness call Anvil didn't log
    assert surplus_calls(logged, harness) == Counter({"anvil_setBalance": 1})
    assert surplus_calls(Counter(), harness) == Counter()


def _fork_writing(p):
    """A Fork with only its log: the harness's own handle on the file Anvil writes."""
    f = Fork.__new__(Fork)
    f.log_path, f._log = p, open(p, "a")  # noqa: SIM115 - closed by the test
    return f


def test_the_log_is_read_from_the_trials_offset(tmp_path):
    p = tmp_path / "anvil.log"
    p.write_bytes(b"anvil_setBalance\n")  # bytes: Windows text mode would write \r\n
    f = _fork_writing(p)
    try:
        start = Fork.log_offset(f)
        assert start == len(b"anvil_setBalance\n") and f.log_intact()
        with open(p, "a") as h:
            h.write("evm_mine\nanvil_mine\n")
        assert Fork.logged_admin_calls(f, start) == Counter({"evm_mine": 1, "anvil_mine": 1})
        p.write_text("")  # cut short: someone edited the log
        with pytest.raises(ValueError, match="shrank"):
            Fork.logged_admin_calls(f, start)
        assert f.log_intact()  # the same file, cut short: integrity fails, the fork stays
    finally:
        f._log.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows can't delete a file another handle has open")
def test_a_deleted_or_replaced_log_cant_be_read_and_the_fork_is_rebuilt(tmp_path):
    """Deleting the log, or putting another file at its path, must not pass as clean."""
    p = tmp_path / "anvil.log"
    p.write_text("eth_call\n")
    f = _fork_writing(p)
    try:
        start = f.log_offset()
        p.unlink()
        assert f.log_offset() == start  # the harness's own handle still knows the size
        assert not f.log_intact()
        with pytest.raises(OSError):
            f.logged_admin_calls(start)
        p.write_text("eth_call\n" * 100)  # a stand-in, longer than the trial's start: still not Anvil's file
        assert not f.log_intact()
        with pytest.raises(FileNotFoundError, match="replaced"):
            f.logged_admin_calls(start)
    finally:
        f._log.close()


def test_a_log_that_cant_be_read_fails_integrity_and_the_harness_s_own_sends_are_trusted():
    """The direct-call check fails closed, and a dev account's transaction counts only when
    the harness's own client sent it (its keys were in Anvil's start-up output)."""
    from collections import Counter as C

    from onchainbench.fork_connect import ForkConnect

    def raise_(e):
        raise e

    harness, dev, eoa = "0x" + "11" * 32, "0x" + "22" * 32, "0x" + "ee" * 20
    blocks = {
        hex(6): {"transactions": [{"hash": harness, "from": "0x" + "d1" * 20}]},
        hex(7): {"transactions": [{"hash": dev, "from": "0x" + "d1" * 20}]},
    }
    fork = types.SimpleNamespace(accounts=["0x" + "d1" * 20], log_offset=lambda: 10)  # the dev account sent both
    s = types.SimpleNamespace(
        integrity_log=[],
        start_block=5,
        sent=set(),
        eoa=eoa,
        fork=fork,
        log_offset=10,
        log_error=None,
        admin_base=C(),
        answered_base=C(),
    )
    fc = ForkConnect.__new__(ForkConnect)
    fc.s = s

    class R:
        admin_calls, sent = C(), {harness}

        def admin_bounds(self):
            return C(), C()

        def block_number(self):
            return 7

        def __call__(self, method, n, full):
            return blocks.get(n)

    fc.rpc = R()
    fc._watch_problems = lambda: []
    fork.logged_admin_calls = lambda o: C()
    assert fc.integrity() == [f"tx {dev} from {'0x' + 'd1' * 20} was sent outside Connect and the harness"]
    fork.logged_admin_calls = lambda o: raise_(FileNotFoundError("gone"))
    assert fc.direct_calls() == [
        "Anvil's log was deleted or made unreadable, so direct calls can't be ruled out: FileNotFoundError: gone"
    ]
    fork.log_offset = lambda: raise_(PermissionError("denied"))
    fc.begin_window()
    assert s.log_offset is None and fc.direct_calls() == [
        "Anvil's log could not be read, so direct calls can't be ruled out: PermissionError: denied"
    ]


def test_the_rpc_client_counts_its_admin_calls_exactly(monkeypatch):
    """A call Anvil answered is in its log for certain, a lost answer may be, a refused
    connection never is. The surplus check allows the possible ones, the deficit check expects the certain ones."""
    import urllib.error

    sent = []

    class Resp:
        def __init__(self, req):
            sent.append(req)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"result": "0x1"}'

    refused = urllib.error.URLError(ConnectionRefusedError(61, "refused"))
    tries = iter([ConnectionResetError("lost"), refused, None, None, TimeoutError("slow"), None])

    def urlopen(req, timeout):
        if (e := next(tries)) is not None:
            raise e
        return Resp(req)

    monkeypatch.setattr(evm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    r = evm.Rpc("http://127.0.0.1:1")
    assert r("anvil_setBalance", "0x01", "0x1") == "0x1" and r("eth_call", {}) == "0x1"
    assert r("anvil_setStorageAt", "0x01", "0x0", "0x1") == "0x1"
    answered, possible = r.admin_bounds()
    assert answered == Counter({"anvil_setBalance": 1, "anvil_setStorageAt": 1})
    # the reset and the timeout may have reached Anvil; the refused connection didn't
    assert possible == Counter({"anvil_setBalance": 2, "anvil_setStorageAt": 2}) == r.admin_calls
    assert evm.never_arrived(refused) and evm.never_arrived(ConnectionRefusedError())
    assert not evm.never_arrived(TimeoutError()) and not evm.never_arrived(urllib.error.URLError("dns"))


def test_an_http_error_reply_counts_as_answered(monkeypatch):
    import io
    import urllib.error

    def urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 500, "x", {}, io.BytesIO(b'{"error": {"message": "nope"}}'))

    monkeypatch.setattr(evm.urllib.request, "urlopen", urlopen)
    r = evm.Rpc("http://127.0.0.1:1")
    assert r.raw("evm_mine")["error"]["message"] == "nope"
    assert r.admin_bounds() == (Counter({"evm_mine": 1}), Counter({"evm_mine": 1}))


def test_the_rpc_client_remembers_the_transactions_it_sent():
    class R(evm.Rpc):
        def __call__(self, method, *params):
            return "0xABC" if method == "eth_sendTransaction" else {"transactionHash": "0xabc", "status": "0x1"}

        def raw(self, method, params=None):
            return {"result": {"transactionHash": "0xabc", "status": "0x1"}}

    r = R("http://127.0.0.1:1")
    assert r.send("0x" + "11" * 20, "0x" + "22" * 20)["status"] == "0x1" and r.sent == {"0xabc"}


def test_a_receipt_lookup_gives_up_fast_when_the_node_stops_answering(monkeypatch):
    """A node that answers no lookup at all (each already retried): the outcome is unknown after a few, not after
    every poll's retries (minutes). An error the node answers with (its upstream) still means "not mined yet"."""
    calls = []

    class R(evm.Rpc):
        def raw(self, method, params=None):
            calls.append(method)
            raise ConnectionRefusedError("refused")

    monkeypatch.setattr(evm.time, "sleep", lambda s: None)
    with pytest.raises(evm.SendOutcomeUnknown, match="refused"):
        R("http://127.0.0.1:1").receipt("0xabc")
    assert calls == ["eth_getTransactionReceipt"] * evm.RECEIPT_TRANSPORT_FAILURES

    answers = iter([OSError("lost"), OSError("lost"), {"error": {"message": "upstream timeout"}}, OSError("lost")])

    class Flaky(evm.Rpc):
        def raw(self, method, params=None):
            a = next(answers, {"result": {"status": "0x1"}})
            if isinstance(a, Exception):
                raise a
            return a

    assert Flaky("http://127.0.0.1:1").receipt("0xabc") == {"status": "0x1"}


def test_anvil_starts_with_random_accounts_and_its_start_up_output_is_cut_from_the_log(tmp_path, monkeypatch):
    """No seed on the command line, and the keys Anvil prints never reach the log."""
    started = {}

    class Proc:
        def __init__(self, cmd, stdout, stderr):
            started["cmd"] = cmd
            stdout.write(
                "Available Accounts\n(0) 0x" + "D1" * 20 + " (1.000000000000000000 ETH)\n"
                "Private Keys\n(0) 0x" + "ab" * 32 + "\nMnemonic: one two three\nListening on 127.0.0.1:8545\n"
            )
            stdout.flush()

        def poll(self):
            return None

    class R:
        def __init__(self, url):
            self.admin_calls = Counter()

        def __call__(self, method, *params):
            return {"eth_accounts": ["0x" + "D1" * 20], "web3_clientVersion": "anvil/v1.9.0"}.get(method, "0x64")

    monkeypatch.setattr(evm.subprocess, "Popen", Proc)
    monkeypatch.setattr(evm, "Rpc", R)
    monkeypatch.setattr(evm, "UpstreamRelay", lambda url: types.SimpleNamespace(url="http://127.0.0.1:2", fixed=False))
    log = tmp_path / "anvil.log"
    log.write_text("an older run's lines\n")
    f = Fork("http://127.0.0.1:3", 1, log=log)
    try:
        assert "--mnemonic-random" in started["cmd"] and "--mnemonic-seed-unsafe" not in started["cmd"]
        assert log.read_text() == "" and f.log_offset() == 0 and f.accounts == ["0x" + "d1" * 20]
        cmd = started["cmd"]  # the port is Anvil's own choice, read from its output (no race for a released one)
        assert cmd[cmd.index("--port") + 1] == "0" and f.port == 8545 and f.client_version == "anvil/v1.9.0"
    finally:
        f._log.close()


def test_the_safe_owner_key_is_random_per_fork(monkeypatch):
    """The owner key was keccak("onchainbench:agent-eoa:<chain>"), computable from the code."""
    from eth_account import Account

    from onchainbench import env as E
    from onchainbench.fork_connect import random_key

    (a1, k1), (a2, k2) = random_key(), random_key()
    assert a1 != a2 and k1 != k2 and Account.from_key(k1).address.lower() == a1
    calls = []
    fork = types.SimpleNamespace(
        set_balance=lambda a, w: calls.append(("fund", a)),
        impersonate=lambda a: None,
        stop_impersonating=lambda a: None,
        snapshot=lambda: "0x1",
    )
    monkeypatch.setattr(E, "start_fork", lambda fc, log: fork)
    monkeypatch.setattr(E, "deploy_safe", lambda f, owner: "safe-of-" + owner)
    envs = [E.ChainEnv("gnosis") for _ in range(2)]
    assert envs[0].eoa != envs[1].eoa and envs[0].safe == "safe-of-" + envs[0].eoa
    assert Account.from_key(envs[0].key).address.lower() == envs[0].eoa


def test_the_cheater_also_cheats_without_a_transaction(monkeypatch):
    wrapped = types.SimpleNamespace(address="0x" + "E9" * 20)
    shown = types.SimpleNamespace(address="0x" + "2A" * 20)
    s = types.SimpleNamespace(tokens={}, fc=types.SimpleNamespace(wrapped=wrapped))
    t = types.SimpleNamespace(safe="0x" + "5a" * 20, fc=types.SimpleNamespace(s=s))
    out = A.direct_cheats(t, "http://127.0.0.1:9")
    assert [m for c in out for m in ("anvil_setStorageAt", "evm_increaseTime", "evm_mine", "anvil_mine") if m in c] == [
        "anvil_setStorageAt",
        "evm_increaseTime",
        "evm_mine",
        "anvil_mine",
    ]
    assert wrapped.address.lower() in out[0] and all(c.endswith("http://127.0.0.1:9") for c in out)
    # a token setup dealt (its balance slot is known) is the one written, Vyper layout included
    s.tokens = {"W": wrapped, "U": shown}
    monkeypatch.setitem(evm._SLOT_CACHE, shown.address.lower(), (3, True))
    first = A.direct_cheats(t, "http://127.0.0.1:9")[0]
    assert shown.address.lower() in first and first != out[0]


def test_the_scan_flags_edits_of_anvils_log():
    edits = scan.PATTERNS["edits-harness"]
    assert re.search(edits, "truncate -s 0 ../results/r/anvil-gnosis-123.log")
    assert re.search(edits, "sed -i /anvil_setBalance/d /tmp/anvil-x1.log")
    assert not re.search(edits, "cat anvil-gnosis-123.log")
