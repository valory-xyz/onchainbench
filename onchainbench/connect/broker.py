"""Local broker between an arm and the trial's Connect: the benchmark's own copy, for the scripted arms and the
unsandboxed CLI arms (its own token instead of the real one; the bench sets no caps).

The harness process reads the real .mcp.json (as Claude Code does) and keeps
the connect token in memory. The agent only ever sees a shim .mcp.json that
points here with a broker token. The broker speaks the same MCP and REST API
as the connect server, so Connect's skills, signer_client.py and the `pearl`
CLI work unchanged; every value call passes the session Policy first.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

from onchainbench.evm import BusyHTTPServer
from onchainbench.mock_connect.server import SERVER_INFO

from .connect_client import ConnectClient, ConnectError, decode_tool_result
from .policy import REST_READ_ROUTES, REST_VALUE_ROUTES, Policy, not_sent

KILLED = "broker: the command that made this call was stopped (timed out) before it was sent; not sent"
IN_FLIGHT = (
    "broker: these calls had already been sent to Connect when the command was stopped; "
    "they may or may not have taken effect: {ids}"
)


def rpc_error(e: Exception) -> dict:
    """A JSON-RPC error for a request the broker couldn't handle: malformed JSON, or a crash."""
    code = -32700 if isinstance(e, json.JSONDecodeError) else -32603
    return {"code": code, "message": broker_error(e)}


def broker_error(e: Exception) -> str:
    return str(e) if isinstance(e, ConnectError) else f"broker: could not handle the request: {type(e).__name__}: {e}"


def rest_error(e: Exception) -> tuple[int, dict]:
    """A REST answer for an upstream failure: Connect's own status and detail when it sent one (a 504 lost response
    stays a 504, not a 400 that reads as a rejected request), else 400 with the broker's text."""
    if isinstance(e, ConnectError) and e.status is not None:
        return e.status, {"detail": e.detail}
    return 400, {"detail": broker_error(e)}


class Broker:
    def __init__(self, upstream: ConnectClient, policy: Policy, host: str = "127.0.0.1", port: int = 0):
        self.upstream = upstream
        self.policy = policy
        self.token = secrets.token_urlsafe(24)
        self.session_id = secrets.token_hex(16)  # opaque, as Connect's: the agent's client sees it
        self._lock = threading.Lock()  # one upstream MCP session; serialize calls on it
        self.epoch = 0  # bumped when the shell kills a command: its calls not yet sent are refused
        self.in_flight = 0  # POSTs being handled (approvals, gate waits, upstream calls); see busy()
        self._count_lock = threading.Lock()
        self._sending: dict = {}  # calls sent upstream and not yet answered: their request ids
        self.last_in_flight: str | None = None  # what the last kill found already sent (told to the agent)
        self.on_refusal = None  # called (tool, args, rule) when policy refuses a call; the eval logs these
        self.on_list = None  # called (tools) when the agent lists tools: the eval records what it saw
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, obj: dict | None, extra: dict | None = None):
                body = b"" if obj is None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                if self.headers.get("Authorization", "") != f"Bearer {broker.token}":
                    self._send(401, {"detail": "unauthorized"})
                    return False
                return True

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def do_GET(self):
                if not self._authed():
                    return
                path = urlsplit(self.path).path.rstrip("/")
                if path == "/mcp":  # an MCP client opening its event stream: 405 means "none"; a 404 reads as an
                    return self._send(405, {"detail": "method not allowed"}, {"Allow": "POST"})  # expired session
                if path not in REST_READ_ROUTES:
                    return self._send(404, {"detail": f"broker: route {path} not exposed"})
                try:
                    with broker._lock:
                        out = broker.upstream.rest("GET", self.path)
                    self._send(200, out)
                except Exception as e:  # noqa: BLE001 - ConnectError or a crash: always an answer, never a drop
                    self._send(*rest_error(e))

            def do_DELETE(self):  # an MCP client ending its session: the broker keeps one upstream session
                self._send(405, {"detail": "method not allowed"}, {"Allow": "POST"})

            def do_POST(self):
                with broker.counting():
                    try:
                        self._post()
                    except Exception as e:  # noqa: BLE001 - a malformed body or a crash: a JSON-RPC error back
                        with contextlib.suppress(OSError):  # unless the client is gone
                            self._send(200, {"jsonrpc": "2.0", "id": None, "error": rpc_error(e)})

            def _post(self):
                if not self._authed():
                    return
                path = urlsplit(self.path).path.rstrip("/")
                body = self._body()
                if path == "/mcp":
                    return self._mcp(body)
                if path in REST_VALUE_ROUTES:
                    epoch = broker.epoch
                    d = broker.policy.check(REST_VALUE_ROUTES[path], body)
                    if not d.allowed:
                        broker.refused(REST_VALUE_ROUTES[path], body, d.rule)
                        return self._send(400, {"detail": d.rule})
                else:  # reads are GET-only (/wallet); nothing else is forwarded
                    return self._send(404, {"detail": f"broker: route {path} not exposed"})
                try:
                    out = broker.send_upstream(epoch, body, lambda: broker.upstream.rest("POST", path, body))
                except ConnectError as e:
                    status, out = rest_error(e)  # Connect's status and detail passed on; 400 for the broker's own
                    if not_sent(e.detail if e.status is not None else str(e).partition(": ")[2]):
                        broker.policy.release(d)
                    return self._send(status, out)
                if out is None:
                    broker.policy.release(d)
                    return self._send(400, {"detail": broker.killed_text()})
                self._send(200, out)

            def _mcp(self, msg: dict):
                mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
                sid = {"Mcp-Session-Id": broker.session_id}
                if mid is None:  # notification
                    return self._send(202, None, sid)

                def ok(result):
                    self._send(200, {"jsonrpc": "2.0", "id": mid, "result": result}, sid)

                if method == "initialize":
                    return ok(
                        {
                            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                            "capabilities": {"tools": {}},
                            "serverInfo": SERVER_INFO,  # Connect's own name: the agent's client shows it
                        }
                    )
                if method == "ping":
                    return ok({})
                try:
                    if method == "tools/list":
                        with broker._lock:
                            tools = broker.upstream.list_tools()
                        # reported before the answer goes out: once the agent holds the listing, the trial's ledger
                        # already has it (reported after, a ledger read right then could miss it)
                        if broker.on_list is not None:
                            with contextlib.suppress(Exception):  # reporting must never change what the agent gets
                                broker.on_list(tools)
                        return ok({"tools": tools})
                    if method == "tools/call":
                        name, args = params.get("name", ""), params.get("arguments") or {}
                        epoch = broker.epoch
                        d = broker.policy.check(name, args)
                        if not d.allowed:
                            broker.refused(name, args, d.rule)
                            return ok({"content": [{"type": "text", "text": d.rule}], "isError": True})
                        out = broker.send_upstream(epoch, args, lambda: broker.upstream.call_tool_raw(name, args))
                        if out is None:  # the command was killed while this call waited: not sent
                            broker.policy.release(d)
                            return ok({"content": [{"type": "text", "text": broker.killed_text()}], "isError": True})
                        if out.get("isError") and not_sent(decode_tool_result(out)):  # Connect refused it unsent
                            broker.policy.release(d)
                        return ok(out)
                except ConnectError as e:
                    return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})
                except Exception as e:  # noqa: BLE001 - bad arguments or a crash: a JSON-RPC error, never a drop
                    return self._send(200, {"jsonrpc": "2.0", "id": mid, "error": rpc_error(e)}, sid)
                self._send(
                    200, {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"{method} not supported"}}
                )

        self.server = BusyHTTPServer((host, port), Handler)  # a backlog of 128: 5 reset concurrent clients
        self.url = f"http://{host}:{self.server.server_address[1]}/mcp/"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def refused(self, tool: str, args: dict, rule: str) -> None:
        if self.on_refusal is not None:
            with contextlib.suppress(Exception):  # reporting must never change what the agent gets back
                self.on_refusal(tool, args, rule)

    @contextlib.contextmanager
    def counting(self):
        with self._count_lock:
            self.in_flight += 1
        try:
            yield
        finally:
            with self._count_lock:
                self.in_flight -= 1

    def send_upstream(self, epoch: int, args: dict, send):
        """Make one value or tool call upstream, one at a time; None if the command that made it was killed since
        `epoch` (checked under the lock, right before sending: a call queued behind a slow send is not sent late)."""
        with self._lock:
            if self.epoch != epoch:
                return None
            key = object()
            with self._count_lock:
                self._sending[key] = str((args or {}).get("request_id") or "(a call without a request_id)")
            try:
                return send()
            finally:
                with self._count_lock:
                    self._sending.pop(key, None)

    def killed_text(self) -> str:
        return KILLED + (f". {self.last_in_flight}" if self.last_in_flight else "")

    def command_killed(self) -> str | None:
        """The shell killed the running command: its calls not yet sent will be refused. Returns, for the agent, the
        request ids of calls already sent at that moment (their results are lost with the command), or None."""
        with self._count_lock:
            self.epoch += 1
            ids = list(self._sending.values())
        self.last_in_flight = IN_FLIGHT.format(ids=", ".join(ids)) if ids else None
        return self.last_in_flight

    def busy(self) -> bool:
        """A call is waiting in the broker: the agent's shell command shouldn't time out under it."""
        return self.in_flight > 0

    def start(self) -> Broker:
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def shim_mcp_json(self) -> dict:
        return {
            "mcpServers": {
                "pearl-connect": {
                    "type": "http",
                    "url": self.url,
                    "headers": {"Authorization": f"Bearer {self.token}"},
                }
            }
        }
