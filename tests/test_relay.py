"""The fork RPC relay under the conditions a run puts it in: many concurrent requests on kept-alive connections, an
upstream that drops idle connections, a slow upstream, and large responses. A wrong answer here would silently corrupt
every trial on a fork, so each request must get exactly its own response."""

import concurrent.futures as cf
import http.client
import http.server
import json
import threading
import time

import pytest

from onchainbench.evm import BusyHTTPServer, UpstreamRelay


class Echo(http.server.BaseHTTPRequestHandler):
    """Answers each JSON-RPC request with its own id and params, optionally slowly, closing some connections."""

    protocol_version = "HTTP/1.1"
    delay = 0.0
    drop_every = 0  # close the connection after every n-th response, as upstreams do with idle keep-alives
    served = 0
    answered = 0  # responses written or failed: a slow answer may come after the relay gave up on it

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(type(self).delay)
        body = json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": req["params"]}).encode()
        type(self).served += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if type(self).drop_every and type(self).served % type(self).drop_every == 0:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)
        finally:
            type(self).answered += 1

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    Echo.delay, Echo.drop_every, Echo.served, Echo.answered = 0.0, 0, 0, 0
    # a BusyHTTPServer, as a real upstream doesn't print when its client hangs up: an answer written after the relay
    # timed out and closed the connection fails (on Windows always, WinError 10053) and is counted, not printed
    srv = BusyHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/rpc"
    srv.shutdown()
    srv.server_close()


def _client(url: str, n: int, tag: str, size: int = 1) -> list[bool]:
    """One kept-alive client connection (as Anvil keeps one), sending n requests; each must get its own answer back."""
    host, port = url.removeprefix("http://").split(":")
    c, ok = http.client.HTTPConnection(host, int(port), timeout=20), []
    for i in range(n):
        params = [f"{tag}-{i}", "x" * size]
        c.request("POST", "/", json.dumps({"jsonrpc": "2.0", "id": i, "method": "eth_getStorageAt", "params": params}))
        r = json.loads(c.getresponse().read())
        ok.append(r["id"] == i and r["result"] == params)
    c.close()
    return ok


def test_concurrent_kept_alive_clients_each_get_their_own_answers(upstream):
    relay = UpstreamRelay(upstream)
    # the standard library's backlog of 5 reset some of 16 clients connecting at once (flaky on Linux, always on macOS)
    assert relay.server.request_queue_size >= 64
    try:
        with cf.ThreadPoolExecutor(16) as ex:
            results = list(ex.map(lambda k: _client(relay.url, 25, f"c{k}"), range(16)))
        assert all(all(r) for r in results) and sum(map(len, results)) == 400
    finally:
        relay.stop()


def test_an_upstream_that_drops_connections_is_reconnected_transparently(upstream):
    Echo.drop_every = 3  # every third response closes the upstream connection
    relay = UpstreamRelay(upstream)
    try:
        assert all(_client(relay.url, 30, "drop"))
    finally:
        relay.stop()


def test_large_responses_pass_whole(upstream):
    relay = UpstreamRelay(upstream)
    try:
        assert all(_client(relay.url, 3, "big", size=2_000_000))  # e.g. a block with its transactions
    finally:
        relay.stop()


def test_a_slow_upstream_times_out_as_a_502_not_a_hang(upstream, capsys):
    Echo.delay = 1.0
    relay = UpstreamRelay(upstream, timeout=0.2)
    try:
        host, port = relay.url.removeprefix("http://").split(":")
        c = http.client.HTTPConnection(host, int(port), timeout=10)
        t0 = time.time()
        c.request("POST", "/", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_getStorageAt", "params": []}))
        r = c.getresponse()
        assert r.status == 502 and b"upstream unreachable" in r.read() and time.time() - t0 < 5
        # the upstream's late answers (one per attempt) go to connections the relay closed: they end here, quietly,
        # not in a later test's output
        deadline = time.time() + 5
        while Echo.answered < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert Echo.answered == 2 and "Traceback" not in capsys.readouterr().err
    finally:
        relay.stop()


def test_only_the_reads_a_fork_needs_are_forwarded(upstream):
    """The relay's localhost URL is on Anvil's command line: a send to the real chain, or any other method, through it
    is refused, a batch whole."""
    relay = UpstreamRelay(upstream)
    try:
        host, port = relay.url.removeprefix("http://").split(":")
        c = http.client.HTTPConnection(host, int(port), timeout=10)

        def post(body):
            c.request("POST", "/", body if isinstance(body, bytes) else json.dumps(body))
            return json.loads(c.getresponse().read())

        ok = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance", "params": ["0x1", "latest"]}
        send = {"jsonrpc": "2.0", "id": 2, "method": "eth_sendRawTransaction", "params": ["0x02"]}
        assert post(ok)["result"] == ["0x1", "latest"] and Echo.served == 1
        assert post(send) == {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": ANY_REFUSAL}}
        batch = post([ok, send, "junk"])
        assert [x["id"] for x in batch] == [1, 2, None] and all("error" in x for x in batch)
        assert post(b"not json")["error"]["code"] == -32700
        assert Echo.served == 1 and relay.refused == [
            "eth_sendRawTransaction",
            "eth_sendRawTransaction",
            "(not a call)",
            "(unparsable)",
        ]
    finally:
        relay.stop()


ANY_REFUSAL = "method not available through the fork relay"


def test_a_ws_fork_url_is_refused():
    from onchainbench.evm import Fork

    with pytest.raises(ValueError, match="must be http"):
        Fork("wss://node.example/key", 1)


def test_a_client_that_resets_is_counted_not_printed(capsys):
    # a fork Anvil stopped at its trial's end resets the keep-alive connections it held to the relay while their
    # threads wait for the next request: counted (peer_resets), with no traceback in the run's output. Any other
    # error in a handler is still printed.
    import socket
    import struct

    from onchainbench.evm import BusyHTTPServer

    class Boom(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            raise ValueError("a real handler bug")

        def log_message(self, *a):
            pass

    srv = BusyHTTPServer(("127.0.0.1", 0), Boom)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = socket.create_connection(srv.server_address)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # close with a reset
        c.close()
        deadline = time.time() + 5
        while srv.peer_resets < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert srv.peer_resets == 1
        assert "Traceback" not in capsys.readouterr().err
        conn = http.client.HTTPConnection(*srv.server_address, timeout=5)
        conn.request("GET", "/")
        with pytest.raises((http.client.RemoteDisconnected, ConnectionResetError)):
            conn.getresponse()
        time.sleep(0.2)
        err = capsys.readouterr().err
        assert "ValueError: a real handler bug" in err and srv.peer_resets == 1
    finally:
        srv.shutdown()
        srv.server_close()


PEER_ABORTS = [
    ConnectionAbortedError(10053, "An established connection was aborted by the software in your host machine"),
    ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host"),
    BrokenPipeError(32, "Broken pipe"),
]


def test_a_peer_abort_is_counted_not_printed_on_any_os(capsys):
    # what Windows raises (WinError 10053, 10054) when a client hangs up, raised here on every OS: counted, quiet;
    # any other error is still printed
    srv = BusyHTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    try:
        for e in [*PEER_ABORTS, ValueError("a real handler bug")]:
            try:
                raise e
            except (OSError, ValueError):
                srv.handle_error(None, ("127.0.0.1", 1))
        err = capsys.readouterr().err
        assert srv.peer_resets == 3 and err.count("Traceback") == 1 and "ValueError: a real handler bug" in err
    finally:
        srv.server_close()


@pytest.mark.parametrize("abort", PEER_ABORTS, ids=lambda e: type(e).__name__)
def test_a_relay_client_that_aborts_before_its_answer_ends_quietly(abort, capsys):
    import io

    class Aborted(io.BytesIO):
        def write(self, b):
            raise abort

    relay = UpstreamRelay("http://127.0.0.1:9/rpc")  # a refused method: the upstream is never reached
    try:
        h = relay.server.RequestHandlerClass.__new__(relay.server.RequestHandlerClass)
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_sendRawTransaction", "params": ["0x02"]}).encode()
        h.rfile, h.wfile, h.headers = io.BytesIO(body), Aborted(), {"Content-Length": str(len(body))}
        h.request_version, h.requestline, h.command = "HTTP/1.1", "POST / HTTP/1.1", "POST"
        h.do_POST()  # the answer's write fails as the client is gone: no exception, no traceback
        assert relay.refused == ["eth_sendRawTransaction"] and capsys.readouterr().err == ""
    finally:
        relay.stop()
