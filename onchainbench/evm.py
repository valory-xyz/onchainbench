"""EVM plumbing for OnchainBench: a JSON-RPC client, ABI helpers and an Anvil fork manager.

The harness talks to Anvil directly (cheat methods included); the agent never does: it gets the
allowlisting proxy in fork_connect.py.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address

ZERO = "0x" + "00" * 20
MAX_UINT = 2**256 - 1


TRANSIENT = (
    "Unknown state",
    "failed to get account",
    "failed to get storage",
    "failed to get block",
    "error sending request",
    "database error",  # Anvil's fork database failing to fetch upstream state (a mine's system calls included)
    "429",
)


# Methods that change a fork's state without the agent's Safe or EOA signing anything, or that send as an account the
# agent doesn't hold: Anvil's cheat namespaces (hardhat_* is an alias Anvil logs as anvil_*) and the sends that skip
# Connect. The integrity check counts them in Anvil's log against the harness's own calls (`Rpc.admin_calls`).
ADMIN_PREFIXES = ("anvil_", "evm_", "hardhat_", "debug_set")
ADMIN_METHODS = frozenset({"eth_sendTransaction", "eth_sendUnsignedTransaction", "eth_sendRawTransaction"})
_LOGGED_METHOD = re.compile(r"[a-z][A-Za-z0-9]*_[A-Za-z0-9_]+")  # a line of Anvil's log that is a request's method


def is_admin(method: str) -> bool:
    return method.startswith(ADMIN_PREFIXES) or method in ADMIN_METHODS


def admin_calls_in_log(text: str) -> Counter:
    """Admin methods in a stretch of Anvil's log: without --silent, Anvil prints each request's method on its own line
    (its other output is indented or has spaces)."""
    return Counter(
        "anvil_" + m.removeprefix("hardhat_") if m.startswith("hardhat_") else m
        for line in text.splitlines()
        if _LOGGED_METHOD.fullmatch(m := line.rstrip()) and is_admin(m)
    )


def surplus_calls(logged: Counter, harness: Counter) -> Counter:
    """Admin calls Anvil received beyond the harness's own: someone else reached Anvil's port."""
    return Counter({m: n - harness.get(m, 0) for m, n in logged.items() if n > harness.get(m, 0)})


def deficit_calls(logged: Counter, answered: Counter) -> Counter:
    """Harness admin calls Anvil answered but its log doesn't show: the log stopped recording (an agent's
    anvil_setLoggingEnabled), so it can't rule out anything."""
    return Counter({m: n - logged.get(m, 0) for m, n in answered.items() if n > logged.get(m, 0)})


def never_arrived(e: BaseException) -> bool:
    """A transport error that means the request never reached the server (the connection was refused), so it
    can't be in Anvil's log; any other (a timeout, a reset) may have arrived."""
    reason = getattr(e, "reason", e)  # URLError wraps the socket's error
    return isinstance(reason, ConnectionRefusedError)


# The client hung up under a server: a reset, a write to a closed connection, or (Windows' WinError 10053, which
# Python raises as ConnectionAbortedError) the local stack aborting a connection the peer had already closed.
PEER_GONE = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)


class BusyHTTPServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer for the bench's local servers (the fork relay, the web corpus). The standard library
    lets only 5 connections wait to be accepted: under load (parallel trials, a fork's burst of state fetches) the
    rest are reset, which looks like a flaky public RPC rather than our own server."""

    request_queue_size = 128
    daemon_threads = True
    peer_resets = 0  # connections the client reset or closed under us, counted instead of printed

    def handle_error(self, request, client_address) -> None:
        """A client that hung up (a fork Anvil stopped at its trial's end resets the keep-alive connections it held
        to the relay, while their threads wait for the next request) is counted, not printed: no request was lost,
        as its sender is gone. Any other error in a handler is printed as the standard library does."""
        if isinstance(sys.exc_info()[1], PEER_GONE):
            self.peer_resets += 1
            return
        super().handle_error(request, client_address)


class RpcError(Exception):
    def __init__(self, message: str, data=None):
        super().__init__(message)
        self.data = data


class SendOutcomeUnknown(RpcError):
    """A send's answer was lost and the chain doesn't show that it landed: the trial is the bench's failure (infra)."""


# Sent again after a transport error (the answer lost, the request may have arrived): reads, and harness setters whose
# second call leaves the state the first did. Nothing else is retried: a send or a mine done twice is two of them.
RETRY_METHODS = frozenset(
    {
        "eth_chainId",
        "net_version",
        "web3_clientVersion",
        "eth_blockNumber",
        "eth_syncing",
        "eth_accounts",
        "eth_call",
        "eth_estimateGas",
        "eth_createAccessList",
        "eth_gasPrice",
        "eth_maxPriorityFeePerGas",
        "eth_feeHistory",
        "eth_getAccount",
        "eth_getAccountInfo",
        "eth_getBalance",
        "eth_getCode",
        "eth_getStorageAt",
        "eth_getTransactionCount",
        "eth_getProof",
        "eth_getBlockByNumber",
        "eth_getBlockByHash",
        "eth_getBlockReceipts",
        "eth_getTransactionByHash",
        "eth_getTransactionReceipt",
        "eth_getTransactionByBlockNumberAndIndex",
        "eth_getTransactionByBlockHashAndIndex",
        "eth_getLogs",
        "debug_traceTransaction",
        "trace_transaction",
        "anvil_nodeInfo",
        "evm_snapshot",  # a second snapshot is never reverted to: harmless
        "anvil_setBalance",
        "anvil_setCode",
        "anvil_setNonce",
        "anvil_setStorageAt",
        "anvil_impersonateAccount",
        "anvil_stopImpersonatingAccount",
    }
)
SEND_METHODS = frozenset({"eth_sendTransaction", "eth_sendRawTransaction"})
RECEIPT_TRANSPORT_FAILURES = 3  # receipt lookups in a row the node itself doesn't answer before the outcome is unknown
LANDED_SCAN_BLOCKS = 64  # how far back from the head a send that landed is looked for (Anvil mines one per send)


def selector(sig: str) -> str:
    return "0x" + keccak(text=sig)[:4].hex()


def _types(sig: str) -> list[str]:
    inner = sig[sig.index("(") + 1 : sig.rindex(")")]
    out, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    return [t for t in [*out, cur] if t]


def calldata(sig: str, *args) -> str:
    """calldata("transfer(address,uint256)", to, amount)."""
    return selector(sig) + abi_encode(_types(sig), list(args)).hex()


def decode(types: list[str], data: str):
    raw = bytes.fromhex(data.removeprefix("0x"))
    return abi_decode(types, raw)


def checksum(a: str) -> str:
    return to_checksum_address(a)


def addr(a: str) -> str:
    return a.lower()


def revert_reason(data: str | None) -> str:
    """Decode Error(string) / Panic(uint256) revert data, else return it raw."""
    if not data or not isinstance(data, str):
        return ""
    if data.startswith("0x08c379a0"):
        try:
            return decode(["string"], "0x" + data[10:])[0]
        except Exception:  # noqa: BLE001 - malformed revert data: show it raw
            return data
    if data.startswith("0x4e487b71"):
        return f"panic {int(data[10:], 16):#x}"
    return data


class Rpc:
    def __init__(self, url: str, timeout: float = 120):
        self.url, self.timeout, self._id = url, timeout, 0
        self._lock = threading.Lock()
        # The integrity check's counts of this client's admin calls (`admin_bounds`): Anvil logs a request's method
        # before it answers, so a call it answered is in its log; a call whose answer was lost (a timeout, a reset
        # connection) may be; a refused connection never reached it and is not counted.
        self.admin_answered: Counter = Counter()  # in Anvil's log for certain
        self.admin_calls: Counter = Counter()  # possibly in it: the answered ones and the lost ones
        self.sent: set[str] = set()  # hashes of the transactions this client sent with `send` (the integrity check)

    def raw(self, method: str, params: list | None = None) -> dict:
        """One JSON-RPC call. After a transport error only RETRY_METHODS are sent again; a send is never sent twice
        blind (`_send`)."""
        params = params or []
        if method in SEND_METHODS:
            return self._send(method, params)
        return self._post(method, params, 4 if method in RETRY_METHODS else 1)

    def _post(self, method: str, params: list, tries: int) -> dict:
        with self._lock:
            self._id += 1
            mid = self._id
        body = json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = r.read()
            except urllib.error.HTTPError as e:  # a JSON-RPC error with an HTTP status: Anvil answered
                self._count_admin(method, answered=True)
                try:
                    return json.loads(e.read())
                except ValueError:
                    if attempt == tries - 1:
                        raise
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if not never_arrived(e):  # a lost answer: Anvil may have logged the call
                    self._count_admin(method, answered=False)
                if attempt == tries - 1:
                    raise
            else:
                self._count_admin(method, answered=True)
                return json.loads(out)
            time.sleep(0.5 * (attempt + 1))
        raise RpcError("unreachable")  # pragma: no cover - the loop returns or raises

    def _count_admin(self, method: str, answered: bool) -> None:
        if is_admin(method):
            with self._lock:
                self.admin_calls[method] += 1
                if answered:
                    self.admin_answered[method] += 1

    def admin_bounds(self) -> tuple[Counter, Counter]:
        """(admin calls certainly in Anvil's log, admin calls possibly in it), both copies."""
        with self._lock:
            return Counter(self.admin_answered), Counter(self.admin_calls)

    def _send(self, method: str, params: list) -> dict:
        """A send. Anvil mines before it answers, so a lost answer (a fork fetching cold state past the timeout) may
        hide a send that landed: the chain says whether it did, and a send that may have landed is never repeated.
        After a transport error: the landed transaction's hash, or SendOutcomeUnknown. After a TRANSIENT error reply
        (the node answered, so the request is over): sent again only while the chain shows it didn't land."""
        mark = self._send_mark(method, params)
        for attempt in range(6):
            try:
                out = self._post(method, params, 1)
            except (OSError, ValueError) as e:  # URLError, a socket timeout, an unreadable answer
                why = f"{method}'s answer was lost ({type(e).__name__}: {e})"
                if (h := self._landed(mark, why)) is not None:
                    return {"jsonrpc": "2.0", "id": None, "result": h}
                raise SendOutcomeUnknown(f"send outcome unknown: {why}, and it isn't on the chain yet") from e
            msg = str((out.get("error") or {}).get("message", ""))
            if attempt == 5 or not any(k in msg for k in TRANSIENT):
                return out
            if (h := self._landed(mark, f"{method} failed with {msg!r}")) is not None:
                return {"jsonrpc": "2.0", "id": out.get("id"), "result": h}
            time.sleep(0.5 * (attempt + 1))
        raise RpcError("unreachable")  # pragma: no cover - the loop returns

    def _send_mark(self, method: str, params: list) -> tuple | None:
        """What shows that a send landed: a raw transaction's hash (the keccak of its bytes), or the sender and the
        nonce the node gives it. None for a send the node will refuse as malformed."""
        try:
            if method == "eth_sendRawTransaction":
                return ("hash", "0x" + keccak(hexstr=params[0]).hex())
            frm, n = params[0]["from"].lower(), params[0].get("nonce")
            n = n if n is not None else self("eth_getTransactionCount", frm, "pending")
            return ("nonce", frm, int(n, 16))
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return None

    def _landed(self, mark: tuple | None, why: str) -> str | None:
        """The landed send's hash, or None while the chain shows it didn't land; SendOutcomeUnknown when the chain
        can't tell (it took the nonce but the transaction isn't in a recent block, or it can't be read)."""
        try:
            if mark is None:
                raise ValueError("a malformed send")
            if mark[0] == "hash":
                return mark[1] if self("eth_getTransactionByHash", mark[1]) else None
            _, frm, nonce = mark
            if int(self("eth_getTransactionCount", frm, "pending"), 16) <= nonce:
                return None
            head = self.block_number()
            for n in range(head, max(head - LANDED_SCAN_BLOCKS, -1), -1):
                for tx in (self("eth_getBlockByNumber", hex(n), True) or {}).get("transactions", []):
                    if tx["from"].lower() == frm and int(tx["nonce"], 16) == nonce:
                        return tx["hash"]
            raise ValueError(f"nonce {nonce} of {frm} was taken, but no recent block holds it")
        except (OSError, ValueError, RpcError) as e:
            raise SendOutcomeUnknown(f"send outcome unknown: {why}; the chain can't tell: {e}") from e

    def __call__(self, method: str, *params):
        # a fork's upstream archive RPC may hit a node without the state. An error reply means the node did nothing
        # (no block mined, no time advanced), so any method is asked again; sends retry inside `_send`, which also
        # checks the chain first. Only a lost answer is limited to RETRY_METHODS (`raw`).
        tries = 1 if method in SEND_METHODS else 6
        for attempt in range(tries):
            out = self.raw(method, list(params))
            msg = str((out.get("error") or {}).get("message", ""))
            if attempt == tries - 1 or not any(k in msg for k in TRANSIENT):
                break
            time.sleep(0.5 * (attempt + 1))
        if "error" in out:
            e = out["error"]
            raise RpcError(e.get("message", str(e)), e.get("data"))
        return out["result"]

    # ---- conveniences ----
    def call(self, to: str, data: str, frm: str | None = None, block="latest") -> str:
        tx = {"to": to, "data": data}
        if frm:
            tx["from"] = frm
        return self("eth_call", tx, block)

    def view(self, to: str, sig: str, *args, returns: list[str] | None = None):
        out = self.call(to, calldata(sig, *args))
        if returns is None:
            return out
        vals = decode(returns, out)
        return vals[0] if len(vals) == 1 else vals

    def balance(self, a: str) -> int:
        return int(self("eth_getBalance", a, "latest"), 16)

    def erc20_balance(self, token: str, who: str) -> int:
        return self.view(token, "balanceOf(address)", checksum(who), returns=["uint256"])

    def block_number(self) -> int:
        return int(self("eth_blockNumber"), 16)

    def timestamp(self) -> int:
        return int(self("eth_getBlockByNumber", "latest", False)["timestamp"], 16)

    def send(
        self,
        frm: str,
        to: str | None,
        data: str = "0x",
        value: int = 0,
        gas: int | None = None,
        nonce: int | None = None,
    ) -> dict:
        """eth_sendTransaction from an unlocked or impersonated account; returns the receipt (automine)."""
        tx = {"from": frm, "data": data, "value": hex(value)}
        if to:
            tx["to"] = to
        if gas:
            tx["gas"] = hex(gas)
        if nonce is not None:
            tx["nonce"] = hex(nonce)
        h = self("eth_sendTransaction", tx)
        with self._lock:
            self.sent.add(str(h).lower())
        return self.receipt(h)

    def receipt(self, h: str) -> dict:
        """The receipt of a transaction the node accepted, polled until it is mined. A fork answers a lookup of one it
        hasn't mined yet from its upstream RPC, which never has it: an error there (a public RPC timing out or
        rate-limiting) means only "not mined yet", and the transaction still lands. Giving up on it would leave a
        send that landed reported as failed. Never mined: SendOutcomeUnknown (it may land later), as when the node
        itself stops answering (RECEIPT_TRANSPORT_FAILURES lookups in a row, each already retried)."""
        last, lost = None, 0
        for i in range(200):
            try:
                out = self.raw("eth_getTransactionReceipt", [h])
                lost = 0
            except (OSError, ValueError) as e:  # a lost answer, after the reads' retries
                out, last, lost = {}, e, lost + 1
                if lost >= RECEIPT_TRANSPORT_FAILURES:
                    break
            if out.get("result"):
                return out["result"]
            if "error" in out:
                last = out["error"].get("message", out["error"])
            if i == 40:  # still pending after 2 s: mine a block (a queued tx after a revert)
                with contextlib.suppress(RpcError, OSError):  # it may have mined: the polling tells
                    self("evm_mine")
            time.sleep(0.05)
        raise SendOutcomeUnknown(
            f"send outcome unknown: no receipt for {h}" + (f" (last lookup: {last})" if last else "")
        )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# What Anvil's fork backend reads from its upstream. The relay's localhost URL is on Anvil's command line (and in
# anvil_nodeInfo), so an unsandboxed agent can reach it: anything else (eth_sendRawTransaction to the real chain,
# costly or unbounded calls on a paid node) is refused.
RELAY_METHODS = frozenset(
    {
        "eth_chainId",
        "net_version",
        "eth_blockNumber",
        "anvil_nodeInfo",  # Anvil asks at startup whether its upstream is another Anvil (read-only)
        "eth_getAccountInfo",  # Anvil's balance, nonce and code in one call (it falls back if refused)
        "eth_getAccount",
        "eth_getBalance",
        "eth_getCode",
        "eth_getStorageAt",
        "eth_getTransactionCount",
        "eth_getProof",
        "eth_getBlockByNumber",
        "eth_getBlockByHash",
        "eth_getBlockReceipts",
        "eth_getTransactionByHash",
        "eth_getTransactionReceipt",
        "eth_getTransactionByBlockNumberAndIndex",
        "eth_getTransactionByBlockHashAndIndex",
        "eth_getLogs",
        "eth_call",
        "eth_estimateGas",
        "eth_createAccessList",
        "eth_gasPrice",
        "eth_maxPriorityFeePerGas",
        "eth_feeHistory",
        "debug_traceTransaction",
        "trace_transaction",
    }
)


TCP_STATES = {"01": "ESTABLISHED", "06": "TIME_WAIT", "08": "CLOSE_WAIT", "0A": "LISTEN"}


def port_holders(port: int, tables: tuple[str, ...] = ("/proc/net/tcp", "/proc/net/tcp6")) -> list[str]:
    """The sockets on this machine whose local port is `port` (Linux's socket tables; none elsewhere): their state,
    the peer, the owner's uid and, where this user may look, the process that holds it, so a relay that couldn't
    bind its chain's port says what held it. A diagnostic only: a line it can't read is skipped, never raised (the
    relay falls back to a free port whatever this finds)."""
    out = []
    for table in tables:
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            try:
                if len(f) <= 9 or int(f[1].rsplit(":", 1)[1], 16) != port:
                    continue
                peer = int(f[2].rsplit(":", 1)[1], 16)
            except (ValueError, IndexError):
                continue
            who = _socket_owner(f[9])
            out.append(f"{TCP_STATES.get(f[3], f[3])} to port {peer} (uid {f[7]}{', ' + who if who else ''})")
    return out


def _socket_owner(inode: str, proc: Path = Path("/proc")) -> str | None:
    """`pid <n>: <command line>` of the process holding socket `inode`, or None if none this user can see."""
    target = f"socket:[{inode}]"
    for d in proc.glob("[0-9]*"):
        try:
            if any(os.readlink(fd) == target for fd in (d / "fd").iterdir()):
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                me = " (this process)" if d.name == str(os.getpid()) else ""
                return f"pid {d.name}{me}: {cmd[:120]}"
        except OSError:
            continue
    return None


def _relay_refusals(body: bytes) -> tuple[bytes, list[str]] | None:
    """The JSON-RPC error reply for a request the relay won't forward and the methods refused, or None to forward
    it. A batch is refused whole if any call in it is."""
    try:
        req = json.loads(body)
    except ValueError:
        return json.dumps(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        ).encode(), ["(unparsable)"]
    calls = req if isinstance(req, list) else [req]
    methods = [str(c.get("method")) if isinstance(c, dict) else "(not a call)" for c in calls]
    if all(m in RELAY_METHODS for m in methods):
        return None
    errors = [
        {
            "jsonrpc": "2.0",
            "id": c.get("id") if isinstance(c, dict) else None,
            "error": {"code": -32601, "message": "method not available through the fork relay"},
        }
        for c in calls
    ]
    return json.dumps(errors if isinstance(req, list) else errors[0]).encode(), [
        m for m in methods if m not in RELAY_METHODS
    ]


class UpstreamRelay:
    """A localhost relay in front of a fork's upstream RPC, so Anvil's command line (which `ps` shows any process on
    the machine, unsandboxed arms included) carries http://127.0.0.1:<port>, not a paid node's URL and its key.
    Each of Anvil's connections keeps its own upstream connection alive, as Anvil's own client would. Only the
    read methods a fork needs (RELAY_METHODS) are forwarded; `refused` keeps the methods it turned away."""

    def __init__(self, upstream: str, timeout: float = 60, port: int = 0):
        """`port`: the chain's fixed port (see shared_relay), or 0 for a free one; a fixed port another process holds
        gives a free one too (`fixed` says which). Never shared between servers: no SO_REUSEPORT, whose load
        balancing would hand one fork's connections to another's relay, reset when that one stops."""
        import base64
        import http.client
        from http.server import BaseHTTPRequestHandler

        relay, self.refused, self.forwarded, self.upstream = self, [], 0, upstream
        u = urllib.parse.urlsplit(upstream)
        conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        auth = {}
        if u.username:  # https://user:pass@host: the credentials go in a header, as Anvil would send them
            cred = f"{urllib.parse.unquote(u.username)}:{urllib.parse.unquote(u.password or '')}"
            auth = {"Authorization": "Basic " + base64.b64encode(cred.encode()).decode()}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            up = None  # this client connection's upstream connection

            def forward(self, body: bytes, headers: dict) -> tuple[int, bytes]:
                for attempt in (0, 1):  # a kept-alive upstream connection may have closed: once more on a fresh one
                    if self.up is None:
                        self.up = conn(u.hostname, u.port, timeout=timeout)
                    try:
                        self.up.request("POST", path, body, headers)
                        r = self.up.getresponse()
                        return r.status, r.read()  # 429 and 5xx pass through: Anvil retries those
                    except (http.client.HTTPException, OSError) as e:
                        self.up.close()
                        self.up = None
                        if attempt:
                            return 502, json.dumps({"error": f"upstream unreachable: {type(e).__name__}"}).encode()
                raise AssertionError("unreachable")  # pragma: no cover

            def do_POST(self):  # the http.server API
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                # Anvil's own User-Agent: some public RPCs' CDNs refuse Python's (403, Cloudflare error 1010)
                ua = self.headers.get("User-Agent") or "onchainbench-relay"
                if (no := _relay_refusals(body)) is not None:
                    out, refused = no
                    code, relay.refused = 200, relay.refused + refused
                else:
                    relay.forwarded += 1  # requests that reached the upstream (a warm cache makes few)
                    code, out = self.forward(body, {"Content-Type": "application/json", "User-Agent": ua, **auth})
                with contextlib.suppress(*PEER_GONE):  # Anvil gave up on it (its timeout)
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)

            def finish(self):
                super().finish()
                if self.up is not None:
                    self.up.close()

            def log_message(self, *a):  # quiet
                pass

        class Fixed(BusyHTTPServer):
            allow_reuse_address = os.name != "nt"  # on Windows SO_REUSEADDR would let a second server take the port

        self.bind_error = None  # why the chain's fixed port couldn't be bound, and what held it (a cold cache)
        try:
            self.server = (Fixed if port else BusyHTTPServer)(("127.0.0.1", port), Handler)
        except OSError as e:  # another socket holds the chain's port: a free port, and a cold cache
            self.bind_error = f"{type(e).__name__}: {e}; held by {port_holders(port) or 'nothing visible'}"
            self.server = BusyHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.fixed = bool(port) and self.server.server_address[1] == port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


# One relay per chain in this process, on the chain's fixed port, used by every fork of it and stopped with the last:
# Foundry >= 1.9 keeps the fork URL in its cache's identity (a cache copied to another URL's name is thrown away), so
# the URL has to stay the same for a fork to find the cache a previous one left. port -> [relay, forks using it]
_RELAYS: dict[int, list] = {}
_RELAYS_LOCK = threading.Lock()


def shared_relay(upstream: str, port: int) -> UpstreamRelay:
    with _RELAYS_LOCK:
        entry = _RELAYS.get(port)
        if entry is not None and entry[0].upstream == upstream:
            entry[1] += 1
            return entry[0]
        relay = UpstreamRelay(upstream, port=port if entry is None else 0)  # another upstream on it: a cold cache
        if relay.fixed:
            _RELAYS[port] = [relay, 1]
        return relay


def release_relay(relay: UpstreamRelay) -> None:
    """A fork is done with its relay: stopped once no fork of its chain uses it."""
    port = relay.server.server_address[1]
    with _RELAYS_LOCK:
        entry = _RELAYS.get(port)
        if entry is not None and entry[0] is relay:
            entry[1] -= 1
            if entry[1]:
                return
            del _RELAYS[port]
    relay.stop()


def anvil_bin() -> str:
    return shutil.which("anvil") or os.path.expanduser("~/.foundry/bin/anvil")


def anvil_version(timeout: float = 30) -> str | None:
    """The installed Anvil's web3_clientVersion, which every fork of the run reports (`Fork.client_version`), from a
    throwaway local node; None without Anvil. The run records it: gas estimation, traces and the request log are
    the build's, so trials on different builds are never combined (run.provenance_conflicts)."""
    if not os.path.exists(anvil_bin()):
        return None
    fd, name = tempfile.mkstemp(prefix="anvil-version-", suffix=".log")
    os.close(fd)
    log = Path(name)
    with open(log, "a") as out:  # append: the node writes at the end whatever this process reads
        proc = subprocess.Popen(
            [anvil_bin(), "--port", "0", "--host", "127.0.0.1", "--accounts", "1", "--mnemonic-random"],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.time() + timeout
        while proc.poll() is None and time.time() < deadline:
            if m := _LISTENING.search(log.read_text(errors="replace")):
                return str(Rpc(f"http://127.0.0.1:{m.group(1)}", timeout=10)("web3_clientVersion"))
            time.sleep(0.1)
        return None
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(10)
        log.unlink(missing_ok=True)


START_SECONDS = 120  # for Anvil to fetch the fork block and listen
_LISTENING = re.compile(r"^Listening on \S*?:(\d+)\s*$", re.M)  # Anvil's last start-up line: the port it bound
_PRINTED_ACCOUNT = re.compile(r"^\(\d+\) (0x[0-9a-fA-F]{40}) \(", re.M)  # "(0) 0xabc... (1.000 ETH)"
LOG_PROBE = "anvil_nodeInfo"  # a harness admin call that changes nothing: the probe that Anvil still logs


class Fork:
    """One Anvil process forking `fork_url` at `block`. Snapshot/revert between trials."""

    def __init__(self, fork_url: str, block: int, chain_id: int | None = None, log: Path | None = None):
        anvil = anvil_bin()
        if not fork_url.startswith(("http://", "https://")):  # a ws:// upstream would bypass the relay
            raise ValueError("fork URLs must be http(s): the relay that keeps the URL off Anvil's command line is HTTP")
        from onchainbench.fork_util import ForkCache, relay_port

        # this process's relay for the chain, on its fixed port: the same fork URL, so the cache is found again
        self.relay = shared_relay(fork_url, relay_port(chain_id)) if chain_id else UpstreamRelay(fork_url)
        self.cache = ForkCache(chain_id, block, self.relay.url) if self.relay.fixed else None
        self.warm = bool(self.cache and self.cache.warm())
        cmd = [
            anvil,
            "--fork-url",
            self.relay.url,
            "--fork-block-number",
            str(block),
            # Anvil binds a free port itself and prints it (`_await_start`): a port picked here and released could be
            # taken by another fork's Anvil while this one waits for the cache lock (warm()), and that one would answer
            "--port",
            "0",
            "--host",
            "127.0.0.1",
            # the dev accounts' keys come from a mnemonic Anvil draws itself, new for every fork: none is on the
            # command line, and the start-up output that prints them is cut from the log below
            "--mnemonic-random",
            "--accounts",
            "4",
            "--balance",
            "1",
            "--compute-units-per-second",
            os.environ.get("FORK_CUPS", "200"),  # stay under public RPCs' rate limits
            "--retries",
            "10",
            "--timeout",
            "30000",
        ]
        if chain_id:
            cmd += ["--chain-id", str(chain_id)]
        # Anvil's log (not --silent) lists every request's method: the integrity check reads it for admin calls the
        # harness didn't make. One file per fork; a temporary one when the caller keeps none.
        self._tmp_log = log is None
        if log is None:
            fd, name = tempfile.mkstemp(prefix="anvil-", suffix=".log")
            os.close(fd)
            log = Path(name)
        self.log_path = Path(log)
        # append mode: Anvil always writes at the end, so cutting the start-up output (below) leaves no gap
        self._log = open(self.log_path, "a")  # noqa: SIM115 - closed in stop()
        self._log.truncate(0)
        from onchainbench import isolation

        with isolation.harness():  # the harness's, even if started inside an agent's turn: never the agent's uid
            self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self.port, self.url, self.rpc = 0, "", None
        host = urllib.parse.urlsplit(fork_url).hostname  # not the URL: a paid node's has its key
        try:
            self.accounts = self._await_start(time.time() + START_SECONDS)
        except RuntimeError as e:
            self.stop()
            raise RuntimeError(f"anvil did not start (fork of {host} @ {block}): {e}") from None
        # Anvil prints the dev accounts' private keys and mnemonic before it listens: cut them from the log the
        # integrity check reads, which an agent outside isolation can read too
        self._log.truncate(0)
        self.client_version = str(self.rpc("web3_clientVersion"))  # the run records it (the build decides behaviour)

    def _await_start(self, deadline: float) -> list[str]:
        """Wait for this process's Anvil to listen; its dev accounts. The port comes from Anvil's own start-up
        output, and the node that answers there must be this process: running, with the accounts it printed."""
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"it exited ({self.proc.returncode})")
            text = self.log_path.read_text(errors="replace")
            if not self.port and (m := _LISTENING.search(text)):
                self.port = int(m.group(1))
                self.url = f"http://127.0.0.1:{self.port}"
                self.rpc = Rpc(self.url)
            if self.rpc is not None:
                try:
                    answered = [a.lower() for a in self.rpc("eth_accounts")]
                except Exception:  # noqa: BLE001 - not answering yet
                    answered = None
                if answered is not None:
                    printed = [a.lower() for a in _PRINTED_ACCOUNT.findall(text)]
                    if self.proc.poll() is not None or not printed or answered != printed:
                        raise RuntimeError(f"the node on port {self.port} is not this Anvil process")
                    return answered
            if time.time() > deadline:
                raise RuntimeError("no answer in time")
            time.sleep(0.2)

    def _end(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)  # Anvil writes its fork cache as it exits
            except subprocess.TimeoutExpired:  # pragma: no cover - anvil ignores SIGTERM only when wedged
                self.proc.kill()

    def stop(self):
        if self.cache is not None:
            self.cache.keep(self._end)
        else:
            self._end()
        self._log.close()
        if self._tmp_log:
            self.log_path.unlink(missing_ok=True)
        if self.relay is not None:
            release_relay(self.relay)  # stopped with the chain's last fork in this process
            self.relay = None

    # ---- Anvil's log: admin calls (the integrity check) ----
    def log_offset(self) -> int:
        """The log's size, from the harness's own handle on the file Anvil writes (not its path, which an agent can
        delete or replace)."""
        return os.fstat(self._log.fileno()).st_size

    def _same_file(self, st: os.stat_result) -> bool:
        own = os.fstat(self._log.fileno())
        return (st.st_dev, st.st_ino) == (own.st_dev, own.st_ino)

    def log_intact(self) -> bool:
        """The log's path still names the file Anvil writes, and it can be read: a fork whose log was deleted or
        replaced is rebuilt before its next trial (`Envs.get`), so the next agent isn't failed for it."""
        try:
            with open(self.log_path, "rb") as f:
                return self._same_file(os.fstat(f.fileno()))
        except OSError:
            return False

    def logging_on(self) -> bool:
        """Anvil still logs each request: a harmless admin call shows up in the log after its current end. One
        `anvil_setLoggingEnabled(false)` silences it for good (re-enabling isn't logged), so a fork that fails this is
        rebuilt before its next trial (`Envs.get`): otherwise later trials' direct calls would go unseen."""
        try:
            offset = self.log_offset()
            self.rpc(LOG_PROBE)
            for _ in range(10):  # Anvil logs a request before it answers; a short grace for the file write
                if self.logged_admin_calls(offset)[LOG_PROBE]:
                    return True
                time.sleep(0.05)
        except (OSError, ValueError, RpcError):
            return False
        return False

    def logged_admin_calls(self, offset: int) -> Counter:
        """Admin calls in Anvil's log since `offset`. Raises OSError if the log can't be read (deleted, unreadable, or
        its path names another file than the one Anvil writes), ValueError if it is shorter than `offset` (someone
        truncated it)."""
        with open(self.log_path, "rb") as f:
            if not self._same_file(os.fstat(f.fileno())):
                raise FileNotFoundError(f"{self.log_path} is no longer the file Anvil writes (replaced)")
            f.seek(0, os.SEEK_END)
            if f.tell() < offset:
                raise ValueError(f"Anvil's log shrank below the trial's start ({f.tell()} < {offset} bytes)")
            f.seek(offset)
            return admin_calls_in_log(f.read().decode(errors="replace"))

    # ---- cheats: harness only ----
    def snapshot(self) -> str:
        return self.rpc("evm_snapshot")

    def revert(self, snap: str) -> None:
        if not self.rpc("evm_revert", snap):
            raise RpcError(f"evm_revert {snap} failed")

    def impersonate(self, a: str) -> None:
        self.rpc("anvil_impersonateAccount", a)

    def stop_impersonating(self, a: str) -> None:
        self.rpc("anvil_stopImpersonatingAccount", a)

    def set_balance(self, a: str, wei: int) -> None:
        self.rpc("anvil_setBalance", a, hex(wei))

    def as_(self, who: str, to: str, data: str, value: int = 0) -> dict:
        """Send a tx as any address (impersonated, gas topped up). The top-up is taken back afterwards, so a harness
        event doesn't fund the address it acts as (an attack target gaining native counts as an executed loss)."""
        before = self.rpc.balance(who)
        topup = 1000 * 10**18 + value - before if before < 100 * 10**18 + value else 0
        if topup:
            self.set_balance(who, before + topup)  # gas at any chain's gas price
        self.impersonate(who)
        try:
            r = self.rpc.send(
                who, to, data, value, gas=8_000_000, nonce=int(self.rpc("eth_getTransactionCount", who, "latest"), 16)
            )
        finally:
            self.stop_impersonating(who)
            if topup:  # what it had, less what it sent and its gas, plus anything it received
                self.set_balance(who, max(self.rpc.balance(who) - topup, 0))
        if int(r["status"], 16) != 1:
            raise RpcError(f"harness tx as {who} to {to} reverted")
        return r

    def warp(self, seconds: int) -> None:
        self.rpc("evm_increaseTime", hex(seconds))
        self.rpc("evm_mine")

    def deal(self, token: str, who: str, amount: int) -> None:
        """Set an ERC-20 balance by finding the balance mapping's slot (Solidity and Vyper layouts)."""
        token = addr(token)
        cache = _SLOT_CACHE.get(token)
        candidates = [cache] if cache else [(s, v) for s in range(0, 60) for v in (False, True)]
        target = self.rpc.erc20_balance(token, who)
        probe = target + 123456789
        for slot, vyper in candidates:
            key = (
                abi_encode(["uint256", "address"], [slot, who])
                if vyper
                else abi_encode(["address", "uint256"], [who, slot])
            )
            loc = "0x" + keccak(key).hex()
            old = self.rpc("eth_getStorageAt", token, loc, "latest")
            self.rpc("anvil_setStorageAt", token, loc, "0x" + hex(probe)[2:].rjust(64, "0"))
            if self.rpc.erc20_balance(token, who) == probe:
                self.rpc("anvil_setStorageAt", token, loc, "0x" + hex(amount)[2:].rjust(64, "0"))
                _SLOT_CACHE[token] = (slot, vyper)
                return
            self.rpc("anvil_setStorageAt", token, loc, old)
        raise RpcError(f"could not find the balance slot of {token}")  # pragma: no cover - suite tokens use mappings

    def code(self, a: str) -> str:
        return self.rpc("eth_getCode", a, "latest")

    def deploy(self, bytecode: str, frm: str | None = None) -> str:
        frm = frm or self.accounts[3]
        self.set_balance(frm, 1000 * 10**18)  # gas, whatever the chain's gas price
        r = self.rpc.send(frm, None, bytecode, gas=8_000_000)
        if int(r["status"], 16) != 1:
            raise RpcError("deploy reverted")
        return r["contractAddress"].lower()


_SLOT_CACHE: dict = {}
