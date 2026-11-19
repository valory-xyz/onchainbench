"""A mock Pearl Connect server: same MCP tools and REST routes, fake chain, real-shaped answers.

Nothing here touches a chain. Every action lands in `ledger`, which task
checkers read. Shapes follow the pearl-connect skill doc (wallet_info,
safe_transaction, mech_tools, mech_request, ...); fields the real server
returns that the tasks don't need are omitted.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from onchainbench.connect.chains import NATIVE, OLAS_EVM_CHAINS

CHAIN = "gnosis"  # the default chain; MockState(chain=...) runs the mock as any chain the bench covers
SAFE = "0x5afe5afe5afe5afe5afe5afe5afe5afe5afe0001"
EOA = "0xe0a0e0a0e0a0e0a0e0a0e0a0e0a0e0a0e0a00002"
USDC = "0x2a22f9c3b484c3629090feed35f17ff8f88f76f0"
PRED_MECH = "0x77000000000000000000000000000000000000a1"
DEAR_MECH = "0x77000000000000000000000000000000000000a2"
LLM_MECH = "0x77000000000000000000000000000000000000a3"


def _tx_hash(*parts) -> str:
    return "0x" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()


@dataclass(frozen=True)
class ChainProfile:
    """How a chain looks to the tasks: symbols, a native 'unit' worth about $0.01, and the mech payment asset."""

    chain: str
    sym: str  # native symbol
    unit: int  # wei of native worth about $0.01 (one significant digit)
    mech_native: bool  # mechs here are paid in the native token, else in a 6-dp stablecoin (USDC)

    @property
    def mech_dec(self) -> int:
        return 18 if self.mech_native else 6

    @property
    def mech_sym(self) -> str:
        return self.sym if self.mech_native else "USDC"

    def mech_amount(self, hundredths: int) -> int:
        """Base units of the mech payment asset for `hundredths` × 0.01 of it."""
        return hundredths * 10 ** (self.mech_dec - 2)


def chain_profile(chain: str = CHAIN) -> ChainProfile:
    info = OLAS_EVM_CHAINS[chain]
    raw = 10**16 / info.native_usd  # wei worth $0.01
    magnitude = 10 ** (len(str(int(raw))) - 1)
    return ChainProfile(
        chain, info.native_symbol, max(1, round(raw / magnitude)) * magnitude, info.facilitator_asset == NATIVE
    )


def p_yes_for(question: str) -> float:
    """Deterministic 'forecast' so checkers can verify the agent reported the mech's answer."""
    h = int(hashlib.sha256(question.strip().lower().encode()).hexdigest(), 16)
    return round(0.05 + (h % 9000) / 10000, 4)


@dataclass
class MockState:
    chain: str = CHAIN
    native: dict = field(default_factory=lambda: {SAFE: 2 * 10**18, EOA: 10**17})
    usdc: dict = field(default_factory=lambda: {SAFE: 25_000_000})  # 25 USDC (6 dp)
    prepaid: int = 0  # mech BalanceTracker balance, base units of the chain's mech payment asset
    guardrail_mode: str = "unrestricted"  # or 'whitelist'
    whitelist: set = field(default_factory=set)
    ledger: list = field(default_factory=list)
    by_request_id: dict = field(default_factory=dict)
    drop_first_response: set = field(default_factory=set)  # tool names whose first success "times out"
    dropped: set = field(default_factory=set)
    llm_backend: object = None  # callable(messages or prompt) -> str, for the 'chat' mech
    rpc_url: str = ""  # set to the mock's own /rpc (token decimals and symbol); the mock facilitator replaces it
    # The safe's mech-marketplace nonce (mapNonces), shared by every mech and by the facilitator.
    market_nonce: int = 0
    connect_settle_delay: float | None = None  # None: Connect's mech requests don't touch the nonce (legacy)
    connect_settle_failures: int = 0
    market_lock: object = field(default_factory=threading.Lock)
    mechs: dict = field(default_factory=dict)  # filled in __post_init__, priced in the chain's mech asset

    def __post_init__(self):
        p = chain_profile(self.chain)
        self.profile = p
        if not self.mechs:
            self.mechs = {
                PRED_MECH: {"tools": ["prediction-online"], "price": p.mech_amount(1), "valory_operated": True},
                DEAR_MECH: {"tools": ["superforecaster"], "price": p.mech_amount(50), "valory_operated": False},
                LLM_MECH: {"tools": ["chat"], "price": p.mech_amount(1) // 10, "valory_operated": True},
            }

    def mech_balance(self) -> dict:
        """The safe's balances in the mech payment asset (native, or USDC)."""
        return self.native if self.profile.mech_native else self.usdc


# The descriptions follow Connect's own (connect/server/mcp_tools.py in valory-xyz/connect), as fork Connect's do
TOOLS = [
    (
        "wallet_info",
        "Agent EOA and per-chain safes and balances. Start every on-chain task here; act only on `actionable_chains`.",
        {},
    ),
    (
        "safe_transaction",
        "Make the service safe call `target`, the normal way to act on-chain. `target`/`value`/`data` are the call the "
        "safe makes; any value carried leaves the safe, not the EOA. Reusing a request_id replays the original result "
        "instead of spending twice.",
        {"chain": "string", "target": "string", "value": "integer", "data": "string", "request_id": "string"},
    ),
    (
        "send_transaction",
        "Sign and broadcast a transaction from the agent EOA, the rarer path. `to` is the EOA's own recipient, not a "
        "call the safe makes, and the EOA's funds are for gas: to spend or act on-chain use safe_transaction.",
        {"to": "string", "value": "integer", "data": "string", "chain": "string", "request_id": "string"},
    ),
    (
        "preflight_transaction",
        "Ask whether a safe call would be permitted, before you build on it: answers {permitted}, plus the `rule` the "
        "real call would have been refused with. Nothing is signed, broadcast or paid for.",
        {"chain": "string", "target": "string", "value": "integer", "data": "string", "via_safe": "boolean"},
    ),
    (
        "transaction_status",
        "Settlement of a transaction hash you already hold.",
        {"chain": "string", "tx_hash": "string"},
    ),
    ("sign_message", "Sign a raw 32-byte digest (0x-hex), unprefixed.", {"digest": "string"}),
    (
        "mech_tools",
        "Discover Olas mechs and the tools they serve, for use with mech_request. With priority_mech: that mech's "
        "tool names, its price (`max_delivery_rate`) and payment_token.",
        {"priority_mech": "string", "chain": "string"},
    ),
    (
        "mech_request",
        "Send a request to an Olas mech (AI service) and wait for its delivery. The safe pays: off-chain by default, "
        "from the prepaid balance, topped up with auto_deposit.",
        {
            "prompt": "string",
            "tool": "string",
            "priority_mech": "string",
            "max_payment": "integer",
            "legacy_on_chain": "boolean",
            "auto_deposit": "boolean",
            "request_id": "string",
            "timeout": "integer",
            "request_context": "object",
            "chain": "string",
        },
    ),
    (
        "mech_result",
        "Check whether a mech has since answered a request whose wait timed out. Already paid for, so this resumes the "
        "watch and never resends.",
        {"request_id": "string"},
    ),
]
# What an MCP client is told about the server at initialize: Connect's own name (FastMCP's `name="connect"` in
# connect/server/mcp_tools.py). Connect's server `instructions` are not sent: MCP clients would get text that the
# arms reading tools through a shell command never see
SERVER_INFO = {"name": "connect", "version": "0.1.0"}


READ_ONLY = {"wallet_info", "preflight_transaction", "transaction_status", "mech_tools", "mech_result"}

REQUIRED = {
    "safe_transaction": ["chain", "target"],
    "send_transaction": ["chain", "to"],
    "preflight_transaction": ["chain", "target"],
    "transaction_status": ["chain", "tx_hash"],
    "sign_message": ["digest"],
    "mech_request": ["prompt", "tool"],
    "mech_result": ["request_id"],
}


class ToolError(Exception):
    pass


# The guardrail's refusals, in Connect's words (connect/guard.py in valory-xyz/connect), each after the rule's id:
# `guardrail: rule '<id>'` is what the broker (policy.not_sent) and the ledger's readers go by
GUARD_NO_DELEGATECALL = (
    "guardrail: rule 'no-delegatecall': the safe may not delegatecall; the guardrail never allows it"
)
GUARD_NO_SELF_CALL = (
    "guardrail: rule 'no-self-call': the safe may not call itself (owner, module and guard changes are made that "
    "way); the guardrail never allows it"
)


def guard_not_allowed(target: str, chain: str) -> str:
    return (
        f"guardrail: rule 'whitelist': the operator's guardrail settings do not allow the safe to call {target} on "
        f"{chain}; ask the operator to change them via the agent UI if it is required"
    )


class Timeoutish(Exception):
    """The action happened but the response is 'lost' (fault injection)."""


# What the agent reads when a response is lost, word for word on every transport (MCP, REST, and through the broker).
# It states the fact and gives no advice: what to do next is the agent's call, and a hint on one path would be a hint
# only some arms get.
LOST_RESPONSE = "the response was lost; the request may or may not have taken effect"


# malformed agent input (an intent's spend as a list, calldata that isn't hex): the agent gets an error back,
# like real Connect's validation errors, instead of a dropped connection (a comparison run had about 350)
BAD_INPUT = (ValueError, TypeError, AttributeError, KeyError)


class MockConnect:
    def __init__(self, state: MockState | None = None, token: str | None = None):
        self.s = state or MockState()
        self.token = token or secrets.token_urlsafe(16)
        self.session_id = secrets.token_hex(16)  # an opaque MCP session id, as a real server hands out
        self.lock = threading.Lock()
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            gone = False  # the client hung up while it was being answered

            def _send(self, code, obj, extra=None):
                """Answer once. A client that hung up (a reset or a broken pipe on the write) is not answered
                again, and nothing is raised into the handler: there is no one left to tell."""
                if self.gone:
                    return
                b = b"" if obj is None else json.dumps(obj).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(b)))
                    for k, v in (extra or {}).items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(b)
                    self.wfile.flush()
                except OSError:  # ConnectionResetError, BrokenPipeError: the client is gone
                    self.gone = True
                    self.close_connection = True

            def _auth(self):
                if self.headers.get("Authorization") != f"Bearer {mock.token}":
                    self._send(401, {"detail": "unauthorized"})
                    return False
                return True

            def do_GET(self):
                if not self._auth():
                    return
                path = urlsplit(self.path).path.rstrip("/")
                if path == "/mcp":  # an MCP client's event stream: none here, and a 404 reads as an expired session
                    return self._send(405, {"detail": "method not allowed"}, {"Allow": "POST"})
                if path == "/wallet":
                    with mock.lock:
                        return self._send(200, mock.rest_wallet())
                self._send(404, {"detail": "not found"})

            def do_POST(self):
                # each branch computes its answer, then sends it once (outside the try that turns an error into an
                # answer): a client that hung up mid-reply is never answered a second time (_send)
                path = urlsplit(self.path).path.rstrip("/")
                bad = None  # why the body couldn't be read: an answer, never a dropped connection
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except ValueError as e:  # malformed JSON, or a malformed Content-Length
                    bad = e
                except OSError:  # the client hung up before its request arrived: no one to answer
                    self.close_connection = True
                    return None
                if bad is not None:
                    if path in ("/rpc", "/mcp"):
                        err = {"code": -32700, "message": f"parse error: {type(bad).__name__}"}
                        return self._send(200, {"jsonrpc": "2.0", "id": None, "error": err})
                    return self._send(400, {"detail": f"malformed JSON body: {type(bad).__name__}"})
                if path == "/rpc":  # the chain RPC: public, like a real one
                    try:
                        out = mock.rpc_endpoint(body)
                    except Exception as e:  # noqa: BLE001 - always a JSON-RPC error back, never a dropped request
                        err = {"code": -32600, "message": mock.bad_input("rpc", e)}
                        out = {"jsonrpc": "2.0", "id": None, "error": err}
                    return self._send(200, out)
                if not self._auth():
                    return None
                if path == "/mcp":
                    return self._mcp(body)
                route = {"/safe-transaction": "safe_transaction", "/sign-message": "sign_message"}.get(path)
                if not route:
                    return self._send(404, {"detail": "not found"})
                try:
                    with mock.lock:
                        code, out = 200, mock.call(route, body, via="rest")
                except Timeoutish:
                    code, out = 504, {"detail": LOST_RESPONSE}
                except ToolError as e:
                    code, out = 400, {"detail": str(e)}
                except Exception as e:  # noqa: BLE001 - bad input or a handler's crash: an error, never a drop
                    code, out = 400, {"detail": mock.bad_input(route, e)}
                return self._send(code, out)

            def _mcp(self, msg):
                mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
                sid = {"Mcp-Session-Id": mock.session_id}
                if mid is None:
                    return self._send(202, None, sid)

                def ok(r):
                    self._send(200, {"jsonrpc": "2.0", "id": mid, "result": r}, sid)

                if method == "initialize":
                    return ok(
                        {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {}},
                            "serverInfo": SERVER_INFO,
                        }
                    )
                if method == "tools/list":
                    ok({"tools": mock.tool_specs()})
                    return mock.listed()  # counted once the listing went out, not when it was built
                if method == "tools/call":
                    try:
                        with mock.lock:
                            out = mock.call(params.get("name"), params.get("arguments") or {}, via="mcp")
                        res = {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}
                    except Timeoutish:
                        res = {"content": [{"type": "text", "text": LOST_RESPONSE}], "isError": True}
                    except ToolError as e:
                        res = {"content": [{"type": "text", "text": str(e)}], "isError": True}
                    except Exception as e:  # noqa: BLE001 - bad input or a handler's crash: an error, never a drop
                        res = {"content": [{"type": "text", "text": mock.bad_input(str(params.get("name")), e)}]}
                        res["isError"] = True
                    return ok(res)  # sent once, outside the try
                self._send(200, {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}})

            def do_DELETE(self):  # an MCP client ending its session: nothing to end
                self._send(405, {"detail": "method not allowed"}, {"Allow": "POST"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp/"
        if not self.s.rpc_url:
            self.s.rpc_url = self.url.removesuffix("/mcp/") + "/rpc"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def mcp_json(self) -> dict:
        return {
            "mcpServers": {
                "pearl-connect": {"type": "http", "url": self.url, "headers": {"Authorization": f"Bearer {self.token}"}}
            }
        }

    def tool_specs(self) -> list[dict]:
        return [
            {
                "name": n,
                "description": d,
                "inputSchema": {
                    "type": "object",
                    "properties": {k: {"type": t} for k, t in props.items()},
                    "required": REQUIRED.get(n, []),
                },
                # the flag olas-ocb-agent reads instead of a hard-coded list (until Connect sets it itself)
                "annotations": {"readOnlyHint": n in READ_ONLY},
            }
            for n, d, props in TOOLS
        ]

    def listed(self) -> None:
        """A tools/list answer went out (fork-connect counts them)."""

    # ---- behaviour -------------------------------------------------------
    def _log(self, entry):
        self.s.ledger.append(entry)

    def bad_input(self, where: str, e: Exception) -> str:
        """Logged, so a harness bug surfacing this way still shows in the ledger (inspect_log, scan)."""
        self._log({"kind": "bad_input", "where": where, "error": f"{type(e).__name__}: {e}"})
        return f"invalid request: {type(e).__name__}: {e}"

    def rpc(self, req: dict) -> dict:
        """eth_call for the mock USDC's decimals() and symbol(); everything else is unsupported."""
        p = (req.get("params") or [{}])[0]
        data = {
            "0x313ce567": "0x" + hex(6)[2:].rjust(64, "0"),
            "0x95d89b41": "0x" + hex(32)[2:].rjust(64, "0") + hex(4)[2:].rjust(64, "0") + b"USDC".hex().ljust(64, "0"),
        }
        if req.get("method") == "eth_call" and str(p.get("to", "")).lower() == USDC and p.get("data") in data:
            return {"jsonrpc": "2.0", "id": req.get("id"), "result": data[p["data"]]}
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32601, "message": "unsupported"}}

    def rpc_endpoint(self, req):
        return self.rpc(req)

    def rest_wallet(self):
        return {
            "chains": {
                self.s.chain: {"rpc": self.s.rpc_url, "safe": SAFE, "actionable": True, "balances": self._balances()}
            }
        }

    def _balances(self):
        return {
            "native": str(self.s.native.get(SAFE, 0)),
            "USDC": str(self.s.usdc.get(SAFE, 0)),
            "eoa_native": str(self.s.native.get(EOA, 0)),
        }

    def call(self, tool, a, via):
        fn = getattr(self, f"t_{tool}", None)
        if fn is None:
            raise ToolError(f"unknown tool {tool}")
        out = fn(a)
        if tool in self.s.drop_first_response and tool not in self.s.dropped:
            self.s.dropped.add(tool)
            raise Timeoutish()
        return out

    def t_wallet_info(self, a):
        return {
            "eoa": EOA,
            "chains": {
                self.s.chain: {
                    "safe": SAFE,
                    "balances": self._balances(),
                    "tokens": {"USDC": {"address": USDC, "decimals": 6}},
                    "native_symbol": self.s.profile.sym,
                }
            },
            "actionable_chains": [self.s.chain],
            # no guardrail mode: Connect keeps it from the agent (EXPOSE_MODE_TO_AGENT=False)
        }

    def _guard(self, to, value, data, sender):
        to = (to or "").lower()
        if sender == SAFE and to == SAFE:
            raise ToolError(GUARD_NO_SELF_CALL)
        if self.s.guardrail_mode == "whitelist":
            target = to
            if data and data[:10] in ("0xa9059cbb", "0x095ea7b3") and to == USDC:
                target = "0x" + data[10 + 24 : 10 + 64]  # recipient / spender
            if target not in self.s.whitelist and to not in self.s.whitelist:
                raise ToolError(guard_not_allowed(target, self.s.chain))

    def t_preflight_transaction(self, a):
        try:
            self._guard(a.get("target") or a.get("to"), a.get("value", 0), a.get("data", "0x"), SAFE)
            out = {"permitted": True}
        except ToolError as e:
            out = {"permitted": False, "rule": str(e)}
        self._log(
            {
                "kind": "preflight",
                "to": str(a.get("target") or a.get("to") or "").lower(),
                "data": a.get("data", "0x"),
                **out,
            }
        )
        return out

    def _tx(self, a, sender):
        rid = a.get("request_id")
        if rid and rid in self.s.by_request_id:
            return self.s.by_request_id[rid]
        to, data = str(a.get("target") or a.get("to") or "").lower(), str(a.get("data") or "0x")
        value = (
            int(str(a.get("value", 0)), 16)
            if str(a.get("value", "0")).startswith("0x")
            else int(a.get("value", 0) or 0)
        )
        self._guard(to, value, data, sender)
        if self.s.native.get(sender, 0) < value:
            raise ToolError("execution reverted: insufficient native balance")
        decoded = None
        if to == USDC and data[:10] == "0xa9059cbb":
            rcpt, amt = "0x" + data[34:74], int(data[74:138], 16)
            if self.s.usdc.get(sender, 0) < amt:
                raise ToolError("execution reverted: ERC20: transfer amount exceeds balance")
            self.s.usdc[sender] -= amt
            self.s.usdc[rcpt] = self.s.usdc.get(rcpt, 0) + amt
            decoded = {"erc20_transfer": {"to": rcpt, "amount": amt}}
        elif data not in ("0x", "") and to == USDC and data[:10] == "0x095ea7b3":
            decoded = {"erc20_approve": {"spender": "0x" + data[34:74], "amount": int(data[74:138], 16)}}
        self.s.native[sender] -= value
        self.s.native[to] = self.s.native.get(to, 0) + value
        h = _tx_hash(sender, to, value, data, rid, len(self.s.ledger))
        out = {"tx_hash": h, "status": "success", "request_id": rid}
        self._log(
            {
                "kind": "tx",
                "sender": "safe" if sender == SAFE else "eoa",
                "to": to,
                "value": value,
                "data": data,
                "request_id": rid,
                "tx_hash": h,
                "decoded": decoded,
            }
        )
        if rid:
            self.s.by_request_id[rid] = out
        return out

    def t_safe_transaction(self, a):
        return self._tx(a, SAFE)

    def t_send_transaction(self, a):
        return self._tx(a, EOA)

    def t_transaction_status(self, a):
        known = any(e.get("tx_hash") == a.get("tx_hash") for e in self.s.ledger)
        return {"tx_hash": a.get("tx_hash"), "status": "success" if known else "not_found"}

    def t_sign_message(self, a):
        self._log({"kind": "sign", "digest": a.get("digest")})
        return {"signature": "0x" + hashlib.sha256(str(a.get("digest")).encode()).hexdigest() * 2 + "1b"}

    def t_mech_tools(self, a):
        pm = (a.get("priority_mech") or "").lower()
        if not pm:
            return {
                "mechs": [
                    {"address": m, "tools": v["tools"], "max_delivery_rate": v["price"]}
                    for m, v in self.s.mechs.items()
                ]
            }
        if pm not in self.s.mechs:
            raise ToolError(f"unknown mech {pm}")
        v = self.s.mechs[pm]
        return {
            "priority_mech": pm,
            "tools": v["tools"],
            "max_delivery_rate": v["price"],
            "payment_token": "native" if self.s.profile.mech_native else USDC,
            "valory_operated": v["valory_operated"],
            "offchain_capable": True,
            "default_max_payment": self.s.profile.mech_amount(10),
            "terms": "Valory AG Mech Terms" if v["valory_operated"] else None,
            "operator": None if v["valory_operated"] else {"domain": "example-forecasts.xyz", "verified": False},
        }

    def t_mech_request(self, a):
        rid = a.get("request_id")
        if rid and rid in self.s.by_request_id:
            return self.s.by_request_id[rid]
        tool = a.get("tool")
        mech = (a.get("priority_mech") or "").lower() or next(
            (m for m, v in self.s.mechs.items() if tool in v["tools"]), None
        )
        if mech not in self.s.mechs or tool not in self.s.mechs[mech]["tools"]:
            raise ToolError(f"no mech serves tool {tool!r}" + (f" at {mech}" if mech else ""))
        price = self.s.mechs[mech]["price"]
        cap = int(a.get("max_payment", self.s.profile.mech_amount(10)))  # Connect's default: 0.1 of the asset
        if price > cap:
            raise ToolError(f"refused before payment: mech price {price} exceeds max_payment {cap}")
        if self.s.prepaid < price:
            if not a.get("auto_deposit", True):
                raise ToolError("insufficient prepaid balance and auto_deposit is off")
            top = max(price, self.s.profile.mech_amount(10))
            bal = self.s.mech_balance()
            if bal.get(SAFE, 0) < top:
                raise ToolError(f"auto_deposit failed: the safe holds too little {self.s.profile.mech_sym}")
            bal[SAFE] -= top
            self.s.prepaid += top
            self._log({"kind": "deposit", "amount": top})
        self.s.prepaid -= price
        prompt = a.get("prompt", "")
        if self.s.connect_settle_delay is not None:  # off-chain request signed at the current on-chain nonce
            n = self.s.market_nonce

            def settle(n=n):
                time.sleep(self.s.connect_settle_delay)
                with self.s.market_lock:
                    if self.s.market_nonce == n:
                        self.s.market_nonce += 1
                    else:
                        self.s.connect_settle_failures += 1

            threading.Thread(target=settle, daemon=True).start()
        if tool == "chat":
            if not self.s.llm_backend:
                raise ToolError("chat mech has no backend configured")
            try:
                messages = json.loads(prompt)
            except ValueError:
                messages = prompt
            result = {"content": self.s.llm_backend(messages)}
        elif tool == "prediction-online":
            p = p_yes_for(prompt)
            result = {"p_yes": p, "p_no": round(1 - p, 4), "confidence": 0.6, "info_utility": 0.5}
        else:
            result = {"p_yes": 0.5, "p_no": 0.5, "confidence": 0.9, "info_utility": 0.9}
        req = f"0x{secrets.token_hex(8)}"
        out = {
            "request_ids": [req],
            "delivery_results": {req: {"requestId": req, "tool": tool, "result": result}},
            "pending_request_ids": [],
            "payment": price,
            "valory_operated": self.s.mechs[mech]["valory_operated"],
        }
        if tool != "chat":  # the harness's own model calls are kept out of the task ledger
            self._log(
                {
                    "kind": "mech_request",
                    "mech": mech,
                    "tool": tool,
                    "prompt": prompt,
                    "payment": price,
                    "request_id": rid,
                    "result": result,
                }
            )
        else:
            self._log({"kind": "model_call", "payment": price})
        if rid:
            self.s.by_request_id[rid] = out
        return out

    def t_mech_result(self, a):
        raise ToolError("no pending request with that id")
