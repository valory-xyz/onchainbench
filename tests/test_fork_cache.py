"""Anvil's fork RPC cache: a stable relay URL per chain, and the handoff to and from <chain>/<block>/storage.json."""

import json
import os
import socket
import time
import urllib.request

import pytest
from eth_utils import keccak

from onchainbench import fork_util
from onchainbench.evm import UpstreamRelay, free_port
from onchainbench.fork_util import ForkCache, relay_port


def test_relay_ports_are_fixed_per_chain_and_distinct(monkeypatch):
    from onchainbench.forks import FORK_CHAINS

    ports = {relay_port(c.chain_id) for c in FORK_CHAINS.values()}
    assert len(ports) == len(FORK_CHAINS) and relay_port(100) == 29100
    # below every ephemeral range (Linux starts at 32768), so no outgoing connection takes a chain's port
    assert all(1024 < p < 32768 for p in ports)
    monkeypatch.setenv("ONCHAINBENCH_RELAY_PORT_BASE", "50000")
    assert relay_port(100) == 50100


def _no_fixed_port(forks, relays) -> str:  # pragma: no cover - only when a fork got no fixed port
    """Why a fork's relay didn't get the chain's port, in full (pytest.fail doesn't shorten it): its bind error (the
    socket that held the port and its process), this process and xdist worker, and the relays this process holds."""
    lines = [f"this process: pid {os.getpid()}, xdist {os.environ.get('PYTEST_XDIST_WORKER')}"]
    lines += [f"relay {f.relay.url}: {f.relay.bind_error}" for f in forks]
    lines.append(f"relays held here: { {p: (r.upstream, n) for p, (r, n) in relays.items()} }")
    return "\n".join(lines)


def test_the_fork_files_share_one_xdist_worker(request):
    """Every file that starts forks runs on one worker (conftest's FORK_FILES group): a chain's relay binds its fixed
    port once per process, and a second worker's relay held it, so a fork there started cold. The group is in the id
    xdist schedules by only when conftest marked the test before xdist read the marks, and only on a pytest whose
    nodeid xdist can rewrite (pytest < 9, pyproject.toml)."""
    if os.environ.get("PYTEST_XDIST_WORKER") and request.config.getvalue("loadgroup"):
        assert request.node.nodeid.endswith("@forks"), request.node.nodeid
    from tests.conftest import FORK_FILES

    assert "test_fork_cache.py" in FORK_FILES and "test_run_forks.py" in FORK_FILES


@pytest.mark.skipif(not os.path.isfile("/proc/sys/net/ipv4/ip_local_port_range"), reason="Linux's port range")
def test_relay_ports_are_outside_this_machines_ephemeral_range():
    """An outgoing connection takes its local port from this range; a relay port inside it can be taken (a cold
    fork)."""
    from pathlib import Path

    from onchainbench.forks import FORK_CHAINS

    low, high = map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
    ports = sorted(relay_port(c.chain_id) for c in FORK_CHAINS.values())
    assert all(p < low or p > high for p in ports), (low, high, ports)


def test_a_relay_that_cant_bind_its_port_says_what_holds_it(tmp_path):
    from onchainbench import evm

    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        r = UpstreamRelay("http://127.0.0.1:1", port=port)
        try:
            assert not r.fixed and "Error" in r.bind_error and "; held by " in r.bind_error
            if os.path.isfile("/proc/net/tcp"):  # Linux: the listener that holds it is named
                assert "LISTEN" in r.bind_error
        finally:
            r.stop()
    finally:
        held.close()
    free = UpstreamRelay("http://127.0.0.1:1", port=free_port())
    try:
        assert free.fixed and free.bind_error is None
    finally:
        free.stop()
    # Linux's socket table, as /proc/net/tcp lists it: local and remote address:port in hex, the state, the uid
    table = tmp_path / "tcp"
    table.write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        f"   0: 0100007F:{29100:04X} 0100007F:{40000:04X} 01 00000000:00000000 00:00000000 00000000  1001 0 1\n"
        f"   1: 0100007F:{29100:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1001 0 2\n"
        f"   2: 0100007F:{8545:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1001 0 3\n"
        # lines it can't read are skipped, not raised: a diagnostic mustn't fail the relay's start
        "   3: 0100007F:zz 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1001 0 4\n"
        f"   4: 0100007F:{29100:04X} 00000000 0A 00000000:00000000 00:00000000 00000000  1001 0 5\n"
    )
    assert evm.port_holders(29100, (str(table), str(tmp_path / "missing"))) == [
        "ESTABLISHED to port 40000 (uid 1001)",
        "LISTEN to port 0 (uid 1001)",
    ]


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_a_sockets_owner_is_named(tmp_path):
    from onchainbench import evm

    proc = tmp_path / "proc"
    (proc / "42" / "fd").mkdir(parents=True)
    (proc / "42" / "fd" / "3").symlink_to("socket:[777]")
    (proc / "42" / "cmdline").write_bytes(b"anvil\0--port\0" + b"0\0")
    (proc / "7" / "fd").mkdir(parents=True)  # a process without the socket, and one this user can't look into
    (proc / "self").mkdir()
    assert evm._socket_owner("777", proc) == "pid 42: anvil --port 0"
    assert evm._socket_owner("778", proc) is None


def test_cache_names_follow_foundry(tmp_path):
    url = "http://127.0.0.1:46100"
    c = ForkCache(100, 41000000, url, root=tmp_path)
    assert c.dir == tmp_path / "xdai" / "41000000"  # Foundry's directory for Gnosis
    assert c.hashed.name == f"storage-{keccak(url.encode()).hex()}.json"  # Foundry >= 1.9: keccak of the fork URL
    assert c.stable.name == "storage.json"  # older Foundry, and what CI saves and merges
    assert ForkCache(4242, 7, url, root=tmp_path).dir == tmp_path / "4242" / "7"
    assert fork_util.rpc_cache_root().parts[-3:] == (".foundry", "cache", "rpc")


def _cache(accounts=(), slots=(), hashes=(), **meta):
    """A fork cache as Anvil writes it: meta, accounts, storage per account, block hashes by number."""
    storage: dict = {}
    for a, s, v in slots:
        storage.setdefault(a, {})[s] = v
    return {
        "meta": {"block_env": {"number": "0x10"}, **meta},
        "accounts": {a: {"balance": "0x1", "nonce": 0} for a in accounts},
        "storage": storage,
        "block_hashes": {hex(n): h for n, h in hashes},
    }


def _anvil_exits(c, data, compressed=False):
    """Anvil writes its whole cache as it exits."""

    def stop():
        fork_util.write_cache(c.hashed, data, compressed)
        os.utime(c.hashed, (time.time() + 5, time.time() + 5))

    return stop


def test_the_handoff_warms_and_keeps_the_cache(tmp_path):
    c = ForkCache(100, 16, "http://127.0.0.1:46100", root=tmp_path)
    assert c.warm() is False and c.dir.is_dir()  # nothing cached yet
    c.stable.write_text(json.dumps(_cache(["0xa"])))
    assert c.warm() is True and json.loads(c.hashed.read_text()) == _cache(["0xa"])  # the saved one, where Anvil looks
    c.keep(_anvil_exits(c, _cache(["0xa", "0xb"])))
    assert json.loads(c.stable.read_text()) == _cache(["0xa", "0xb"])  # kept under the stable name
    c.keep(lambda: None)  # an Anvil that wrote nothing new: the stable file stays
    assert json.loads(c.stable.read_text()) == _cache(["0xa", "0xb"])
    os.utime(c.hashed, (time.time() - 100, time.time() - 100))
    c.stable.write_text(json.dumps(_cache(["0xc"])))  # a newer saved cache (CI merged the shards'): it replaces Anvil's
    c.warm()
    assert json.loads(c.hashed.read_text()) == _cache(["0xc"])
    assert not list(c.dir.glob("*.tmp"))


def test_the_union_keeps_both_forks_state_and_only_fetched_block_hashes():
    saved = _cache(["0xa"], [("0xt", "0x1", "0x11"), ("0xt", "0x2", "0x22")], [(16, "0xh16"), (17, "0xmined")])
    new = _cache(
        ["0xa", "0xb"],
        [("0xt", "0x2", "0x99"), ("0xt", "0x3", "0x33"), ("0xu", "0x1", "0x44")],
        [(15, "0xh15"), (18, "0xmined too")],
        source_id="0xnew",
    )
    new["accounts"]["0xa"] = {"balance": "0x5", "nonce": 3}
    u = fork_util.union_cache(saved, new, 16)
    assert set(u["accounts"]) == {"0xa", "0xb"}
    assert u["storage"] == {"0xt": {"0x1": "0x11", "0x2": "0x22", "0x3": "0x33"}, "0xu": {"0x1": "0x44"}}
    # the same slot or account can't differ at a pinned block; if it does (a bad upstream answer), the saved one stays
    assert u["accounts"]["0xa"] == {"balance": "0x1", "nonce": 0} and u["storage"]["0xt"]["0x2"] == "0x22"
    # blocks above the fork block are ones a fork mined itself (Foundry before 1.9 caches their hashes): never kept
    assert u["block_hashes"] == {hex(16): "0xh16", hex(15): "0xh15"}
    assert u["meta"]["source_id"] == "0xnew"  # meta is the newest Foundry's, which wrote it
    assert fork_util.union_cache({}, {"block_hashes": {"16": "0xd", "x": "0xbad"}, "other": 1}, 16) == {
        "accounts": {},
        "storage": {},
        "block_hashes": {"16": "0xd"},
        "other": 1,
    }
    assert fork_util.union_cache({"accounts": [], "storage": {"0xt": 3}}, {}, 16)["storage"] == {"0xt": {}}


def test_two_forks_of_a_block_stopping_together_both_end_up_saved(tmp_path):
    import threading

    c = ForkCache(100, 16, "http://127.0.0.1:46100", root=tmp_path)
    c.stable.parent.mkdir(parents=True)
    c.stable.write_text(json.dumps(_cache(["0xsaved"])))
    c.warm()
    forks = [ForkCache(100, 16, "http://127.0.0.1:46100", root=tmp_path) for _ in range(4)]
    exits = [_anvil_exits(f, _cache(["0xsaved", f"0x{i}"], [("0xt", hex(i), "0x1")])) for i, f in enumerate(forks)]
    threads = [threading.Thread(target=f.keep, args=(stop,)) for f, stop in zip(forks, exits, strict=True)]
    # each started from the same saved file, and each writes the same Anvil file as it stops
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    saved = json.loads(c.stable.read_text())
    assert set(saved["accounts"]) == {"0xsaved", "0x0", "0x1", "0x2", "0x3"}
    assert saved["storage"] == {"0xt": {hex(i): "0x1" for i in range(4)}}


def test_older_foundry_writing_the_stable_file_itself_still_ends_in_a_union(tmp_path):
    """Foundry before 1.9 writes <chain>/<block>/storage.json itself as it exits, over what other forks saved."""
    c = ForkCache(100, 16, "http://127.0.0.1:46100", root=tmp_path)
    c.stable.parent.mkdir(parents=True)
    c.stable.write_text(json.dumps(_cache(["0xother fork"])))  # saved by a fork that stopped while this one ran

    def old_anvil_exits():
        c.stable.write_text(json.dumps(_cache(["0xthis fork"])))

    c.keep(old_anvil_exits)
    assert set(json.loads(c.stable.read_text())["accounts"]) == {"0xother fork", "0xthis fork"}
    assert not c.hashed.exists()


def test_foundry_1_9s_compressed_cache_is_read_and_written_compressed(tmp_path):
    import zstandard

    c = ForkCache(100, 16, "http://127.0.0.1:46100", root=tmp_path)
    c.stable.parent.mkdir(parents=True)
    c.stable.write_text(json.dumps(_cache(["0xa"])))  # saved by an older Foundry, plain JSON
    c.keep(_anvil_exits(c, _cache(["0xb"]), compressed=True))
    raw = c.stable.read_bytes()
    assert raw.startswith(fork_util.ZSTD_MAGIC)  # as Foundry >= 1.9 reads it
    assert set(json.loads(zstandard.ZstdDecompressor().stream_reader(raw).read())["accounts"]) == {"0xa", "0xb"}
    assert fork_util.read_cache(c.stable) == (fork_util.union_cache(_cache(["0xa"]), _cache(["0xb"]), 16), True)
    c.keep(_anvil_exits(c, _cache(["0xc"])))  # back to plain JSON with the Foundry that wrote the newest file
    assert set(json.loads(c.stable.read_text())["accounts"]) == {"0xa", "0xb", "0xc"}


def test_a_half_written_or_corrupt_cache_never_replaces_the_saved_one(tmp_path):
    c = ForkCache(100, 16, "http://127.0.0.1:46100", root=tmp_path)
    c.stable.parent.mkdir(parents=True)
    good = _cache(["0xa"])
    c.stable.write_text(json.dumps(good))
    whole = json.dumps(_cache(["0xb"])).encode()
    import zstandard

    packed = zstandard.ZstdCompressor().compress(whole)
    corrupt = fork_util.ZSTD_MAGIC + b"\xff" * 20  # a zstd frame header zstandard rejects
    for bad in (whole[: len(whole) // 2], packed[: len(packed) // 2], corrupt, b"[1]", b""):

        def killed(bad=bad):  # Anvil killed while writing its cache
            c.hashed.write_bytes(bad)
            os.utime(c.hashed, (time.time() + 5, time.time() + 5))

        c.keep(killed)  # never raises into the trial
        assert json.loads(c.stable.read_text()) == good, bad[:20]
        with pytest.raises(ValueError):
            fork_util.read_cache(c.hashed)
    c.stable.write_bytes(b'{"accounts": {"0x')  # a saved file that can't be read: Anvil's state replaces it
    c.keep(_anvil_exits(c, _cache(["0xb"])))
    assert json.loads(c.stable.read_text()) == fork_util.union_cache({}, _cache(["0xb"]), 16)
    assert fork_util.merge_into(c.stable, tmp_path / "missing.json", 16) is False  # nothing to add
    c.stable.unlink()  # no saved cache, and Anvil killed while writing its own: nothing written
    c.keep(lambda: c.hashed.write_bytes(b'{"accounts": {"0x'))
    assert not c.stable.exists()


def test_a_failed_write_leaves_the_saved_cache_whole(tmp_path, monkeypatch):
    dst, src = tmp_path / "storage.json", tmp_path / "anvil.json"
    dst.write_text(json.dumps(_cache(["0xa"])))
    src.write_text(json.dumps(_cache(["0xb"])))

    def full_disk(self, data):
        with self.open("wb") as f:
            f.write(data[:3])  # part of it, then the disk is full
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fork_util.Path, "write_bytes", full_disk)
    assert fork_util.merge_into(dst, src, 16) is False
    assert json.loads(dst.read_text()) == _cache(["0xa"]) and sorted(p.name for p in tmp_path.iterdir()) == [
        "anvil.json",
        "storage.json",
    ]


def test_the_handoff_is_locked_against_concurrent_forks(tmp_path):
    import threading

    c = ForkCache(100, 1, "http://127.0.0.1:46100", root=tmp_path)
    inside, order = threading.Event(), []

    def slow_exit():
        inside.set()
        time.sleep(0.3)
        order.append("first stopped")

    t = threading.Thread(target=c.keep, args=(slow_exit,))
    t.start()
    inside.wait(5)
    ForkCache(100, 1, "http://127.0.0.1:46100", root=tmp_path).warm()  # waits for the other fork's exit
    order.append("second warmed")
    t.join()
    assert order == ["first stopped", "second warmed"]


def _rpc(url):
    req = urllib.request.Request(url, json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_chainId"}).encode())
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_one_relay_per_chain_in_a_process():
    import http.server
    import threading

    from onchainbench.evm import release_relay, shared_relay

    class Up(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = b'{"jsonrpc": "2.0", "id": 1, "result": "0x64"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    up = http.server.HTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    port, upstream = free_port(), f"http://127.0.0.1:{up.server_address[1]}"
    a = shared_relay(upstream, port)
    b = shared_relay(upstream, port)  # a second fork of the chain: the same relay, not a second server on the port
    other = shared_relay("http://127.0.0.1:1", port)  # another upstream can't take the port: a free one, cold
    try:
        assert a is b and a.fixed and a.url == f"http://127.0.0.1:{port}"
        assert not other.fixed and other.url != a.url
        assert _rpc(a.url)["result"] == "0x64"
        release_relay(b)
        assert _rpc(a.url)["result"] == "0x64"  # one fork stopping leaves the chain's relay serving the others
    finally:
        release_relay(a)
        release_relay(other)
        up.shutdown()
    with pytest.raises(OSError):  # stopped with its last fork
        _rpc(f"http://127.0.0.1:{port}")
    plain = socket.socket()  # the port held by another process: a free port, and no cache handoff
    plain.bind(("127.0.0.1", 0))
    plain.listen()
    try:
        c = shared_relay("http://127.0.0.1:1", plain.getsockname()[1])
        assert not c.fixed and c.url != f"http://127.0.0.1:{plain.getsockname()[1]}"
        release_relay(c)
    finally:
        plain.close()
    d = UpstreamRelay("http://127.0.0.1:1")
    assert not d.fixed
    d.stop()


@pytest.mark.fork
def test_a_second_fork_of_the_same_block_starts_warm():
    from onchainbench.evm import Fork
    from onchainbench.forks import FORK_CHAINS

    c = FORK_CHAINS["gnosis"]
    W = c.wrapped.address
    first = Fork(c.fork_url, c.block, c.chain_id)
    try:
        from onchainbench import evm

        if first.cache is None:  # why a fork got no fixed port
            pytest.fail(_no_fixed_port([first], evm._RELAYS))
        assert first.relay.url == f"http://127.0.0.1:{relay_port(c.chain_id)}"
        first.code(W)  # something fetched upstream, kept in the cache when the fork stops
    finally:
        first.stop()
    assert first.cache.stable.is_file()  # saved under <chain>/<block>/storage.json
    second = Fork(c.fork_url, c.block, c.chain_id)
    try:
        assert second.warm and second.cache.hashed.is_file()
        before = second.relay.forwarded
        assert len(second.code(W)) > 2
        assert second.relay.forwarded == before  # read from the cache, not the upstream
    finally:
        second.stop()


@pytest.mark.fork
def test_concurrent_forks_of_a_block_keep_each_others_state():
    """Two forks running at once both start from the saved file; what each fetched is in it after both stop."""
    from onchainbench.evm import Fork
    from onchainbench.forks import FORK_CHAINS

    c = FORK_CHAINS["gnosis"]
    fresh = ["0x" + os.urandom(20).hex() for _ in range(2)]  # accounts no earlier run has fetched
    forks = [Fork(c.fork_url, c.block, c.chain_id) for _ in fresh]
    from onchainbench import evm

    if any(f.cache is None for f in forks):
        pytest.fail(_no_fixed_port(forks, evm._RELAYS))  # in full: the holder's pid, process and xdist worker
    try:
        for f, a in zip(forks, fresh, strict=True):
            f.rpc("eth_getBalance", a, "latest")  # fetched upstream by this fork only
    finally:
        for f in forks:
            f.stop()
    saved = fork_util.read_cache(forks[0].cache.stable)[0]
    assert set(fresh) <= set(saved["accounts"])  # not only the last fork's to stop
    third = Fork(c.fork_url, c.block, c.chain_id)
    try:
        before = third.relay.forwarded
        for a in fresh:
            third.rpc("eth_getBalance", a, "latest")
        assert third.relay.forwarded == before  # both read from the cache, not the upstream
    finally:
        third.stop()
