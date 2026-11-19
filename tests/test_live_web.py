"""The per-trial logging proxy: it relays untouched and records which hosts were reached."""

import socket
import socketserver
import threading
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from onchainbench import arms as A
from onchainbench import live_web as L
from onchainbench import stats


@pytest.fixture
def proxy():
    p = L.DomainLog()
    yield p
    p.stop()


@pytest.fixture
def echo():
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            while data := self.request.recv(1024):
                self.request.sendall(data)

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _client(proxy):
    return socket.create_connection(("127.0.0.1", int(proxy.url.rsplit(":", 1)[1])), timeout=5)


def test_a_connect_tunnel_is_relayed_untouched(proxy, echo):
    c = _client(proxy)
    c.sendall(f"CONNECT 127.0.0.1:{echo} HTTP/1.1\r\nHost: x\r\n\r\nearly".encode())  # bytes after the head too
    got = b""
    while b"early" not in got:
        got += c.recv(1024)
    assert got.startswith(b"HTTP/1.1 200")
    c.sendall(b"ping")
    assert c.recv(1024) == b"ping"
    c.close()
    assert proxy.hosts == {"127.0.0.1": 1}  # logged, but loopback is the trial's own services, not the web
    assert proxy.summary() == {"live_domains": {}, "live_web": [], "loopback_requests": 1}


def test_plain_http_is_forwarded(proxy):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"ok")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy.url}))
        assert opener.open(f"http://127.0.0.1:{srv.server_address[1]}/x", timeout=5).read() == b"ok"
    finally:
        srv.shutdown()
        srv.server_close()
    assert proxy.hosts["127.0.0.1"] == 1


def test_unreachable_hosts_are_logged_and_cut_requests_ignored(proxy, monkeypatch):
    c = _client(proxy)
    c.sendall(b"CONNECT no-such-host.invalid:443 HTTP/1.1\r\n\r\n")
    assert c.recv(1024) == b""  # the proxy closes: nothing to relay to
    c.close()
    c = _client(proxy)
    c.sendall(b"CONNECT half")  # the client goes away before its head ends
    c.close()
    monkeypatch.setattr(L, "RELAY_IDLE_SECONDS", 0.05)
    a, b = socket.socketpair()
    L._relay(a, b)  # an idle tunnel ends
    a.close()
    b.close()
    deadline = threading.Event()
    deadline.wait(0.3)
    assert proxy.summary()["live_domains"] == {"no-such-host.invalid": 1}


def test_request_lines_env_and_summary(proxy):
    assert L._target("CONNECT [::1]:8443 HTTP/1.1", []) == ("::1", 8443, True)
    assert L._target("GET http://a.org:8080/p HTTP/1.1", []) == ("a.org", 8080, False)
    assert L._target("GET / HTTP/1.1", ["Host: b.org"]) == ("b.org", 80, False)
    env = proxy.env()
    assert env["HTTPS_PROXY"] == env["http_proxy"] == proxy.url and "127.0.0.1" in env["NO_PROXY"]
    assert env["NODE_USE_ENV_PROXY"] == "1"
    proxy.hosts.update({"openrouter.ai": 3, "api.anthropic.com": 1, "github.com": 2, "defillama.com": 1})
    assert proxy.summary("codex")["live_web"] == ["defillama.com"]  # Codex's own update check isn't the agent's
    assert proxy.summary("hermes")["live_web"] == ["defillama.com", "github.com"]
    assert L.is_infra("OpenRouter.ai.") and not L.is_infra("notopenrouter.ai")


def test_arms_pass_the_proxy_on_and_mark_the_trial(monkeypatch, tmp_path):
    t = types.SimpleNamespace(data={}, ws=tmp_path / "ws", chain="gnosis")
    assert A.proxy_env(None) == {} and A.proxy_env(t) == {} and "proxy_applied" not in t.data
    t.data["proxy_env"] = {"HTTPS_PROXY": "http://127.0.0.1:1"}
    assert A._env_with_key({"K": "v"}, t)["HTTPS_PROXY"] == "http://127.0.0.1:1" and t.data["proxy_applied"]
    # the Python CLIs get it for their agent's shell alone (--shell-env, the environment config): their own model
    # calls stay off the log, as they were in the bench's process
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    assert "HTTPS_PROXY" not in A._env_with_key({"K": "v"}, t, proxy=False)


def test_the_summary_counts_live_web_trials():
    def r(**kw):
        return {"arm": "a", "task": "t", "chain": "gnosis", "block": "C", "passed": True, "seconds": 1, **kw}

    s = stats.arm_summary(
        [
            r(domains_logged=True, live_web=["defillama.com"]),
            r(domains_logged=True, live_web=[]),
            r(domains_logged=False),
        ]
    )["live_web"]
    want = {"logged_share": 0.667, "trials_reaching_live_web": 1, "hosts": {"defillama.com": 1}}
    assert s == {**want, "trials_with_tool_lookups": 0}


def test_a_broken_commands_words_arent_logged_as_web_hosts(proxy):
    """open-run-1 (codex c-token-diligence): a mis-quoted curl argument sent "role", "was" and "vqr" through the
    proxy as hosts; they aren't the web, so they are kept apart, never in live_domains or live_web."""
    proxy.hosts.update({"openrouter.ai": 2, "role": 1, "was": 1, "vqr": 1, "8.8.8.8": 1, "defillama.com": 1})
    s = proxy.summary("codex")
    assert s["live_web"] == ["8.8.8.8", "defillama.com"] and s["not_web_hosts"] == {"role": 1, "vqr": 1, "was": 1}
    empty = L.DomainLog(refuse_local=False)
    assert "role" not in s["live_domains"] and "not_web_hosts" not in empty.summary()
    empty.stop()
    assert L.is_web_host("Api.Example.COM.") and L.is_web_host("[2001:db8::1]") and not L.is_web_host("-a.b")
    assert not L.is_web_host("a..b") and not L.is_web_host("x y.com")
    assert L.is_web_host("my_host.example.com") and L.is_web_host("_dmarc.example.org")  # underscores in labels
    assert not L.is_web_host("my_host")  # still a single word
