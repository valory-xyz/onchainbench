"""Which live hosts a trial's agent reached (a safeguard for live sources): agents have live web access while the
research tasks are graded against a frozen snapshot, so results that lean on the live web must be visible.

Each trial gets its own forward proxy that only logs: HTTPS `CONNECT` tunnels are relayed untouched (no TLS
interception), plain HTTP is forwarded as sent. The arms' processes find it through the standard proxy variables;
local services (the fork's RPC, Connect, the web corpus) bypass it through NO_PROXY. A harness that ignores proxy
variables isn't logged, so a trial records whether its arm was given the proxy (`domains_logged`), not that nothing
was reached.
"""

from __future__ import annotations

import contextlib
import ipaddress
import re
import select
import socket
import socketserver
import threading
from collections import Counter

# model and harness infrastructure: reached on every trial, not the agent's research
INFRA_SUFFIXES = (
    "openrouter.ai",
    "anthropic.com",
    "claude.ai",
    "openai.com",
    "chatgpt.com",
    "fireworks.ai",
    "deepseek.com",
    "googleapis.com",
    "sentry.io",
    "statsig.com",
)
# what a harness contacts on its own, before the agent does anything (seen in the proxy smoke test, 2026-11-14:
# Codex's update and telemetry checks); kept in live_domains, left out of live_web
# (a host or its subdomains; the smoke run of 2026-11-14 showed Hermes's start-up fetches and Claude Code's
# non-essential traffic, which CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC also turns off)
_HERMES = ("models.dev", "nousresearch.com", "nousresearch.github.io")
HARNESS_HOSTS = {
    "codex": ("github.com",),
    "claude-code": ("api.anthropic.com",),
    "hermes": _HERMES,
    "hermes-raw": _HERMES,
}
# lookups a tool makes on its own when the agent runs it, not the agent browsing: Foundry's `cast` resolves
# selectors and verified sources (sourcify.dev, api.4byte.sourcify.dev; openchain.xyz in older versions). Kept in
# live_domains, reported apart as tool_lookups, left out of live_web
TOOL_LOOKUPS = ("sourcify.dev", "openchain.xyz")
NO_PROXY = "localhost,127.0.0.1,::1,0.0.0.0"
LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}  # the trial's own services, reached through the proxy anyway
RELAY_IDLE_SECONDS = 120


def _under(host: str, suffixes) -> bool:
    host = host.lower().rstrip(".")
    return any(host == s or host.endswith("." + s) for s in suffixes)


def is_infra(host: str) -> bool:
    return _under(host, INFRA_SUFFIXES)


def is_tool_lookup(host: str) -> bool:
    return _under(host, TOOL_LOOKUPS)


def is_loopback(host: str) -> bool:
    h = host.lower().strip("[]")
    return h in LOOPBACK or h.startswith("127.")


_LABEL = re.compile(r"(?!-)[a-z0-9_-]{1,63}(?<!-)", re.ASCII)  # `_` too: real names use it (_dmarc, my_host)


def is_web_host(host: str) -> bool:
    """A name the web can have: an IP address, or a dotted DNS name. A single word isn't: open-run-1 logged "role",
    "was" and "vqr" as hosts, the words of a mis-quoted `curl` argument that curl tried as URLs through the proxy."""
    h = host.lower().strip("[]").rstrip(".")
    with contextlib.suppress(ValueError):
        ipaddress.ip_address(h.split("%", 1)[0])
        return True
    labels = h.split(".")
    return len(labels) > 1 and all(_LABEL.fullmatch(x) for x in labels)


def is_local_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%", 1)[0])
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped is not None:
        a = a.ipv4_mapped
    return a.is_loopback or a.is_unspecified or a.is_link_local


def public_address(host: str, port: int) -> tuple[str, int] | None:
    """host's first address, or None when any of them is this machine's (loopback, unspecified, link-local) or the
    name doesn't resolve: a name that points at 127.0.0.1 can't reach the harness's ports either."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return None
    ips = [i[4][0] for i in infos]
    if not ips or any(is_local_ip(ip) for ip in ips):
        return None
    return ips[0], port


def _relay(a: socket.socket, b: socket.socket) -> None:
    socks = [a, b]
    while True:
        ready, _, _ = select.select(socks, [], [], RELAY_IDLE_SECONDS)
        if not ready:
            return
        for s in ready:
            data = s.recv(65536)
            if not data:
                return
            (b if s is a else a).sendall(data)


def _target(line: str, headers: list[str]) -> tuple[str, int, bool]:
    """(host, port, is CONNECT) from a proxy request line: `CONNECT host:443 HTTP/1.1` or `GET http://host/ ...`."""
    method, uri, _ = line.split(" ", 2)
    if method.upper() == "CONNECT":
        host, _, port = uri.rpartition(":")
        return host.strip("[]"), int(port or 443), True
    rest = uri.split("://", 1)[-1]
    hostport = rest.split("/", 1)[0] or next(
        (h.split(":", 1)[1].strip() for h in headers if h.lower().startswith("host:")), ""
    )
    host, _, port = hostport.partition(":")
    return host, int(port or 80), False


class DomainLog:
    """A per-trial logging proxy: `env()` for the arm's processes, `hosts` what they reached."""

    def __init__(self, refuse_local: bool | None = None):
        """refuse_local: no tunnel to this machine's own addresses (default: under agent isolation, where the
        proxy, running as the harness, would otherwise relay an agent to the ports the firewall refuses it)."""
        if refuse_local is None:
            from onchainbench import isolation

            refuse_local = isolation.requested()
        self.refuse_local = refuse_local
        self.refused: Counter = Counter()
        self.hosts: Counter = Counter()
        self._lock = threading.Lock()
        log = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                with contextlib.suppress(OSError, ValueError):
                    log._serve(self.request)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 65536:
            chunk = client.recv(4096)
            if not chunk:
                return
            head += chunk
        text = head.split(b"\r\n\r\n", 1)[0].decode("latin-1")
        line, *headers = text.split("\r\n")
        host, port, tunnel = _target(line, headers)
        with self._lock:
            self.hosts[host.lower()] += 1
        if self.refuse_local:
            addr = public_address(host, port)
            if addr is None:
                with self._lock:
                    self.refused[host.lower()] += 1
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            upstream = socket.create_connection(addr, timeout=30)  # the address checked, not a fresh lookup
        else:
            upstream = socket.create_connection((host, port), timeout=30)
        try:
            if tunnel:
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                leftover = head.split(b"\r\n\r\n", 1)[1]
                if leftover:
                    upstream.sendall(leftover)
            else:
                upstream.sendall(head)
            _relay(client, upstream)
        finally:
            upstream.close()

    def env(self) -> dict:
        """The proxy variables, in the spellings tools read (Node needs NODE_USE_ENV_PROXY to honour them)."""
        out = {}
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            out[k] = out[k.lower()] = self.url
        out["NO_PROXY"] = out["no_proxy"] = NO_PROXY
        out["NODE_USE_ENV_PROXY"] = "1"
        return out

    def summary(self, kind: str = "") -> dict:
        """Every host reached (`live_domains`, with counts) and the ones beyond model and harness infrastructure
        (`live_web`): what the agent itself may have researched live."""
        with self._lock:
            hosts = dict(sorted(self.hosts.items()))
        own = HARNESS_HOSTS.get(kind, ())
        local = sum(n for h, n in hosts.items() if is_loopback(h))  # not the web: counted apart
        hosts = {h: n for h, n in hosts.items() if not is_loopback(h)}
        names = {h: n for h, n in hosts.items() if not is_web_host(h)}  # a broken command's words: not the web
        hosts = {h: n for h, n in hosts.items() if h not in names}
        beyond = [h for h in hosts if not is_infra(h) and not _under(h, own)]
        tools = [h for h in beyond if is_tool_lookup(h)]
        live = [h for h in beyond if h not in tools]
        return {
            "live_domains": hosts,
            "live_web": live,
            **({"tool_lookups": tools} if tools else {}),
            **({"loopback_requests": local} if local else {}),
            **({"not_web_hosts": names} if names else {}),
        }

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
