"""Minimal client for the Pearl Connect server (MCP over streamable HTTP, plus its REST routes).

Used by three callers: the broker (upstream to the real connect server), the
`pearl` CLI (to the broker, via the shim .mcp.json) and MechModel (model calls
bought from the marketplace). Stdlib only, so it runs inside any sandbox.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


class ConnectError(RuntimeError):
    """A refusal or failure reported by the server (HTTP error or JSON-RPC/tool error). An HTTP error keeps its
    `status` and the server's own `detail`, so a broker can pass both on unchanged."""

    def __init__(self, message: str, status: int | None = None, detail=None):
        super().__init__(message)
        self.status, self.detail = status, detail


def find_mcp_config(start: str | os.PathLike | None = None, server: str = "pearl-connect") -> tuple[str, str]:
    """Walk from `start` (default cwd) upwards to the first .mcp.json; return (url, token).

    Same discovery as Connect's signer_client.py, so a shim .mcp.json in the
    agent's directory shadows the real one in the workspace root.
    """
    here = Path(start or os.getcwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / ".mcp.json"
        if path.is_file():
            entry = json.loads(path.read_text())["mcpServers"][server]
            token = entry.get("headers", {}).get("Authorization", "").removeprefix("Bearer ")
            return entry["url"], token
    raise FileNotFoundError(".mcp.json not found in cwd or parents")


def base_url_of(mcp_url: str) -> str:
    """http://127.0.0.1:8716/mcp/ -> http://127.0.0.1:8716 (where the REST routes live)."""
    url = mcp_url.rstrip("/")
    return url[: -len("/mcp")] if url.endswith("/mcp") else url


class ConnectClient:
    def __init__(self, mcp_url: str, token: str, timeout: float = 960.0):
        self.mcp_url = mcp_url if mcp_url.endswith("/") else mcp_url + "/"
        self.base_url = base_url_of(mcp_url)
        self.token = token
        self.timeout = timeout
        self.session_id: str | None = None
        self._next_id = 0
        self._initialized = False

    @classmethod
    def from_mcp_json(cls, start=None) -> ConnectClient:
        return cls(*find_mcp_config(start))

    # ---- HTTP plumbing -------------------------------------------------
    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.token}",
        }
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _http(self, url: str, payload: dict | None, method: str = "POST") -> tuple[int, dict, str]:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - local URL
                return resp.status, dict(resp.headers), resp.read().decode()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except (ValueError, AttributeError):
                detail = body
            raise ConnectError(f"HTTP {e.code}: {detail}", status=e.code, detail=detail) from None
        except OSError as e:  # refused, reset, timed out: the connect server isn't reachable
            raise ConnectError(f"connect server unreachable: {e}") from None

    @staticmethod
    def _parse_rpc_body(headers: dict, body: str) -> dict | None:
        ctype = {k.lower(): v for k, v in headers.items()}.get("content-type", "")
        if not body.strip():
            return None
        if "text/event-stream" in ctype:
            last = None
            for line in body.splitlines():
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk:
                        msg = json.loads(chunk)
                        if "result" in msg or "error" in msg:
                            last = msg
            return last
        return json.loads(body)

    def _rpc(self, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notify:
            self._next_id += 1
            payload["id"] = self._next_id
        _status, headers, body = self._http(self.mcp_url, payload)
        sid = {k.lower(): v for k, v in headers.items()}.get("mcp-session-id")
        if sid:
            self.session_id = sid
        if notify:
            return None
        msg = self._parse_rpc_body(headers, body)
        if msg is None:
            raise ConnectError(f"empty response to {method}")
        if "error" in msg:
            raise ConnectError(f"{method}: {msg['error'].get('message', msg['error'])}")
        return msg["result"]

    # ---- MCP -----------------------------------------------------------
    def initialize(self) -> None:
        if self._initialized:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "onchainbench", "version": "0.1"},
            },
        )
        self._rpc("notifications/initialized", notify=True)
        self._initialized = True

    def list_tools(self) -> list[dict]:
        self.initialize()
        return self._rpc("tools/list", {})["tools"]

    def call_tool_raw(self, name: str, arguments: dict) -> dict:
        """Return the MCP CallToolResult as-is (content, isError, structuredContent)."""
        self.initialize()
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def call_tool(self, name: str, arguments: dict | None = None):
        """Call a tool and decode its result; raise ConnectError if the tool reports an error."""
        res = self.call_tool_raw(name, arguments or {})
        value = decode_tool_result(res)
        if res.get("isError"):
            raise ConnectError(value if isinstance(value, str) else json.dumps(value))
        return value

    # ---- REST (signer_client.py routes) ----------------------------------
    def rest(self, method: str, path: str, payload: dict | None = None) -> dict:
        _, _, body = self._http(self.base_url + path, payload, method=method)
        return json.loads(body) if body.strip() else {}


def decode_tool_result(res: dict):
    if "structuredContent" in res and res["structuredContent"] is not None:
        return res["structuredContent"]
    texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
    text = "\n".join(texts)
    try:
        return json.loads(text)
    except ValueError:
        return text
