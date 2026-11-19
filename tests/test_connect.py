"""The bench's own Connect layer: client, policy, broker, `pearl` CLI, shell, sandbox, session.

Ported from olas-ocb-agent's tests when the bench took its own copy of this plumbing; `build` below opens the bench's
Connect session where the agent's tests opened the agent's runtime.
"""

import json
import os
import platform
import random
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from onchainbench.connect import pearl_cli
from onchainbench.connect.chains import native_usd
from onchainbench.connect.connect_client import ConnectClient, ConnectError, decode_tool_result, find_mcp_config
from onchainbench.connect.policy import Policy, SessionCaps, _int, summarize
from onchainbench.connect.sandbox import wrapper_for
from onchainbench.connect.session import _decode_symbol, mech_asset_lookup, open_session
from onchainbench.connect.shell import PersistentShell, _truncate
from onchainbench.connect.workspace import Workspace
from onchainbench.mock_connect.server import LLM_MECH, PRED_MECH, SAFE, MockConnect, MockState

mac = pytest.mark.skipif(platform.system() != "Darwin", reason="Seatbelt is macOS-only")

PWD = "pwd -W" if os.name == "nt" else "pwd"  # Git Bash's plain pwd prints /c/... on Windows

no_sandbox_on_windows = pytest.mark.skipif(
    os.name == "nt", reason="no OS sandbox on native Windows (WSL2 has bubblewrap)"
)


class ScriptedModel:  # the agent's tests passed a model; the bench's session has none
    def __init__(self, *_):
        pass


def _req(url, token, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def build(root, model=None, approval="ask", approver=None, sandbox="auto", caps=None, **_):
    return open_session(root, caps or SessionCaps(), approval=approval, approver=approver, sandbox=sandbox)


def test_find_mcp_config_and_from_mcp_json(connect_ws, mock, tmp_path):
    url, token = find_mcp_config(connect_ws)
    assert url == mock.url and token == mock.token
    assert ConnectClient.from_mcp_json(connect_ws).list_tools()
    with pytest.raises(FileNotFoundError):
        find_mcp_config(tmp_path / "nowhere")


def test_rpc_body_parsing_and_errors(mock):
    sse = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\ndata:\n'
    assert ConnectClient._parse_rpc_body({"Content-Type": "text/event-stream"}, sse) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"ok": True},
    }
    assert ConnectClient._parse_rpc_body({}, "  ") is None
    c = ConnectClient(mock.url, mock.token)
    with pytest.raises(ConnectError, match="method not found"):  # JSON-RPC error from the server
        c._rpc("no/such/method", {})
    with pytest.raises(ConnectError, match="HTTP 401"):
        ConnectClient(mock.url, "wrong").list_tools()
    with pytest.raises(ConnectError, match="unknown tool"):
        c.call_tool("does_not_exist", {})


def test_http_error_with_non_json_body(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 500, "x", {}, __import__("io").BytesIO(b"<html>oops"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ConnectError, match="HTTP 500: <html>oops"):
        ConnectClient("http://127.0.0.1:1/mcp/", "t").rest("GET", "/wallet")


def test_empty_rpc_response(mock, monkeypatch):
    c = ConnectClient(mock.url, mock.token)
    monkeypatch.setattr(c, "_http", lambda *a, **k: (200, {}, ""))
    with pytest.raises(ConnectError, match="empty response"):
        c._rpc("tools/list", {})


def test_decode_tool_result_variants():
    assert decode_tool_result({"structuredContent": {"a": 1}}) == {"a": 1}
    assert decode_tool_result({"content": [{"type": "text", "text": '{"b": 2}'}]}) == {"b": 2}
    assert decode_tool_result({"content": [{"type": "text", "text": "plain"}]}) == "plain"


# ---- broker ------------------------------------------------------------------------


def test_broker_routes(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        base, tok = rt.broker.url.removesuffix("/mcp/"), rt.broker.token
        assert rt.shell.hold == rt.broker.busy and not rt.broker.busy()  # nothing in flight
        assert _req(base + "/wallet", tok)[0] == 200
        assert _req(base + "/nope", tok)[0] == 404
        assert _req(base + "/wallet", tok, "POST", {})[0] == 404  # reads are GET-only
        code, body = _req(base + "/sign-message", tok, "POST", {"digest": "0x00"})
        assert code == 400 and "signing is off" in body["detail"]
        assert _req(rt.broker.url, tok, "POST", {"jsonrpc": "2.0", "id": 1, "method": "ping"})[1]["result"] == {}
        assert "error" in _req(rt.broker.url, tok, "POST", {"jsonrpc": "2.0", "id": 2, "method": "nope"})[1]
        rt.policy.caps.allow_sign_message = True
        code, body = _req(base + "/sign-message", tok, "POST", {"digest": "0x00"})
        assert code == 200 and body["signature"].startswith("0x")
        mock.stop()  # upstream gone: errors become refusals, not dropped connections
        assert _req(base + "/wallet", tok)[0] == 400
        assert _req(base + "/sign-message", tok, "POST", {"digest": "0x01"})[0] == 400
        r = _req(rt.broker.url, tok, "POST", {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})[1]
        assert r["result"]["isError"]
    finally:
        rt.close()


def test_runtime_reads_read_only_flags_from_connect(connect_ws, mock, monkeypatch):
    from onchainbench.mock_connect.server import READ_ONLY

    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    assert rt.policy.read_only == READ_ONLY  # taken from the mock's readOnlyHint flags
    rt.close()
    monkeypatch.setattr(ConnectClient, "list_tools", lambda self: (_ for _ in ()).throw(ConnectError("down")))
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    assert "wallet_info" in rt.policy.read_only  # fallback list kept
    rt.close()


def test_calls_of_a_killed_command_are_not_sent(connect_ws, mock):
    from onchainbench.connect.broker import KILLED

    rt = build(connect_ws, model=ScriptedModel([]), approval="ask", sandbox="none")
    try:

        def approve_after_kill(summary):
            rt.shell._kill()  # the shell times the command out while the user is deciding
            return True

        rt.policy.approver = approve_after_kill
        base, tok = rt.broker.url.removesuffix("/mcp/"), rt.broker.token
        tx = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 10**15}
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "safe_transaction", "arguments": tx},
        }
        r = _req(rt.broker.url, tok, "POST", body)[1]["result"]
        assert r["isError"] and r["content"][0]["text"] == KILLED
        code, out = _req(base + "/safe-transaction", tok, "POST", tx)
        assert code == 400 and out["detail"] == KILLED
        assert not [e for e in mock.s.ledger if e.get("kind") == "tx"]  # nothing reached Connect

    finally:
        rt.close()


def test_broker_reports_refusals(connect_ws, mock):
    """Policy refusals reach on_refusal (MCP and REST); a failing reporter never changes the agent's answer."""
    rt = build(connect_ws, model=ScriptedModel([]), approval="deny", sandbox="none")
    try:
        seen = []
        rt.broker.on_refusal = lambda tool, args, rule: seen.append((tool, args.get("target"), rule))
        base, tok = rt.broker.url.removesuffix("/mcp/"), rt.broker.token
        tx = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 1}
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "safe_transaction", "arguments": tx},
        }
        assert _req(rt.broker.url, tok, "POST", body)[1]["result"]["isError"]
        assert _req(base + "/safe-transaction", tok, "POST", tx)[0] == 400
        assert [s[1] for s in seen] == [tx["target"]] * 2 and all("disabled" in s[2] for s in seen)

        def boom(*a):
            raise RuntimeError("reporter down")

        rt.broker.on_refusal = boom
        assert _req(base + "/safe-transaction", tok, "POST", tx)[0] == 400
    finally:
        rt.close()


def test_broker_reports_tool_listings(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        lists = []

        def slow_report(tools):  # a slow reporter: reported after the answer, the listing would be missing below
            time.sleep(0.3)
            lists.append(tools)

        rt.broker.on_list = slow_report
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        assert _req(rt.broker.url, rt.broker.token, "POST", body)[1]["result"]["tools"]
        assert lists  # reported before the answer went out: no wait

        def boom(tools):
            raise RuntimeError("reporter down")

        rt.broker.on_list = boom
        assert _req(rt.broker.url, rt.broker.token, "POST", body)[1]["result"]["tools"]
    finally:
        rt.close()


def test_shell_exit_restart_interrupt_and_truncate(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=10, max_output=40)
    r = sh.run("exit 3")
    assert r["returncode"] == -1 and sh.proc is None
    assert sh.run("echo back")["output"].strip() == "back"  # restarted on demand
    sh.interrupt()
    assert "elided" in sh.run("python3 -c \"print('x'*200)\"")["output"]
    sh.close()
    assert _truncate("short", 10) == "short"


# ---- sandbox -------------------------------------------------------------------


def test_sandbox_modes_off_macos(tmp_path, monkeypatch):
    assert wrapper_for(tmp_path, mode="none") == ([], "no sandbox (requested)")
    monkeypatch.setattr("onchainbench.connect.sandbox.platform.system", lambda: "Linux")
    monkeypatch.setattr("onchainbench.connect.sandbox.shutil.which", lambda name: None)  # no bwrap installed
    with pytest.raises(RuntimeError):
        wrapper_for(tmp_path, mode="seatbelt")
    with pytest.raises(RuntimeError):
        wrapper_for(tmp_path, mode="bwrap")
    assert "install bubblewrap" in wrapper_for(tmp_path)[1]
    monkeypatch.setattr("onchainbench.connect.sandbox.platform.system", lambda: "Windows")
    assert "WSL2" in wrapper_for(tmp_path)[1]


@pytest.mark.skipif(os.name == "nt", reason="Windows paths can't hold the quote this checks; Seatbelt is macOS-only")
def test_seatbelt_profile_on_any_platform(tmp_path, monkeypatch):
    from onchainbench.connect.sandbox import seatbelt_profile

    secret, keys = tmp_path / 'my "odd".json', tmp_path / "keys"
    secret.write_text("x")
    keys.mkdir()
    p = seatbelt_profile(tmp_path, [secret, keys], [tmp_path / "w"])
    real = os.path.realpath(tmp_path)
    assert p.startswith("(version 1)(allow default)") and '(deny file-write* (subpath "/"))' in p
    assert f'(deny file-read* (literal "{real}/my \\"odd\\".json"))' in p  # quotes escaped
    assert f'(deny file-read* (subpath "{real}/keys"))' in p
    assert f'(allow file-write* (subpath "{real}")' in p
    assert p.rindex(f'(deny file-write* (literal "{real}"))') > p.index("(allow file-write*")  # parents can't move
    p = seatbelt_profile(tmp_path / "agent", [], [], deny_write=[tmp_path / "AGENTS.md", tmp_path / ".claude"])
    assert p.index(f'(deny file-write* (subpath "{real}/AGENTS.md"))') > p.index("(allow file-write*")  # wins
    assert f'(deny file-write* (subpath "{real}/.claude"))' in p  # not there yet: can't be created either
    # a protected root under a writable temp dir: denied after the allow, the workspace inside it allowed again
    p = seatbelt_profile(tmp_path / "ws" / "agent", [], [tmp_path], protect=[tmp_path / "ws"])
    deny_root = p.index(f'(deny file-write* (subpath "{real}/ws"))')
    assert p.index("(allow file-write*") < deny_root and f'(subpath "{real}")' in p
    assert p.index(f'(allow file-write* (subpath "{real}/ws/agent"))') > deny_root
    assert p.index(f'(deny file-write* (literal "{real}"))') > deny_root  # the dir above it can't be renamed
    p = seatbelt_profile(tmp_path / "outside", [], [], protect=[tmp_path / "ws"])
    assert f'(allow file-write* (subpath "{real}/outside"))' not in p
    monkeypatch.setattr("onchainbench.connect.sandbox.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "onchainbench.connect.sandbox.shutil.which", lambda n: "/usr/bin/sandbox-exec" if n == "sandbox-exec" else None
    )
    wrap, desc = wrapper_for(tmp_path)
    assert desc == "macOS Seatbelt" and wrap[:2] == ["sandbox-exec", "-p"]


def test_bwrap_args_on_linux(tmp_path, monkeypatch):
    from onchainbench.connect.sandbox import bwrap_args

    secret, keys = tmp_path / ".mcp.json", tmp_path / "keys"
    secret.write_text("x")
    keys.mkdir()
    args = bwrap_args(tmp_path, [secret, keys, tmp_path / "missing"], [tmp_path / "nope"])
    assert args[:4] == ["bwrap", "--ro-bind", "/", "/"] and args[-1] == "--"
    ws = os.path.realpath(tmp_path)

    def has(*seq):
        return any(args[i : i + len(seq)] == list(seq) for i in range(len(args)))

    assert has("--bind", ws, ws)
    assert has("--ro-bind", "/dev/null", os.path.realpath(secret))  # the file is covered
    assert has("--tmpfs", os.path.realpath(keys))  # the directory is covered (empty)
    assert args.index("--tmpfs") > args.index("--bind")  # covers come after the writable binds
    assert "missing" not in " ".join(args) and "nope" not in " ".join(args)
    brief = tmp_path / "AGENTS.md"
    brief.write_text("b")
    args = bwrap_args(tmp_path / "agent", [], [], deny_write=[brief, tmp_path / ".claude"])
    assert has("--ro-bind", os.path.realpath(brief), os.path.realpath(brief)) and ".claude" not in " ".join(args)
    top, ws_dir = tmp_path / "top", tmp_path / "top" / "ws" / "agent"
    ws_dir.mkdir(parents=True)
    (tmp_path / "secret").mkdir()
    args = bwrap_args(ws_dir, [tmp_path / "secret", tmp_path], [tmp_path], protect=[top / "ws", top / "ws"])
    t, w, sec = (os.path.realpath(x) for x in (top, ws_dir, tmp_path / "secret"))
    assert has("--bind", ws, ws) and has("--ro-bind", t, t) and has("--bind", w, w) and has("--ro-bind", sec, sec)
    assert args.index(t) < args.index(w) and args.count(t) == 2
    assert has("--tmpfs", sec, "--remount-ro", sec)
    tmpdir = tmp_path / "user-1"  # $TMPDIR inside a writable dir: bound on its own, anchors taken below it
    (tmpdir / "pe2-x" / "persistent_data" / "agent").mkdir(parents=True)
    monkeypatch.setattr("onchainbench.connect.sandbox.tempfile.gettempdir", lambda: str(tmpdir))
    root = tmpdir / "pe2-x" / "persistent_data"
    args = bwrap_args(root / "agent", [], [tmp_path], protect=[root])
    td, pe2 = os.path.realpath(tmpdir), os.path.realpath(tmpdir / "pe2-x")
    assert has("--bind", td, td) and has("--ro-bind", pe2, pe2) and not has("--ro-bind", td, td)
    assert args.index(ws) < args.index(td) < args.index(pe2)
    monkeypatch.setattr("onchainbench.connect.sandbox.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "onchainbench.connect.sandbox.shutil.which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None
    )
    wrap, desc = wrapper_for(tmp_path)
    assert desc == "Linux bubblewrap" and wrap[0] == "bwrap"


def test_the_users_package_caches_are_read_only_and_tools_get_their_own(tmp_path):
    from onchainbench.connect import sandbox
    from onchainbench.connect.workspace import STATE_DIR

    # an allow-list: no user cache is writable, whatever tool owns it; temp dirs are
    home = Path.home()
    assert not any(p.is_relative_to(home) for p in sandbox.DEFAULT_EXTRA_WRITE)
    p = sandbox.seatbelt_profile(tmp_path, [], sandbox.DEFAULT_EXTRA_WRITE)
    for cache in (home / ".cache", home / ".cache" / "pnpm", home / "Library" / "Caches"):
        assert sandbox._sb_str(os.path.realpath(cache)) not in p
    assert f"(subpath {sandbox._sb_str(os.path.realpath('/var/tmp'))})" in p
    args = sandbox.bwrap_args(tmp_path, [], sandbox.DEFAULT_EXTRA_WRITE)
    assert str(home / ".cache") not in args
    env = Workspace.open(tmp_path).install_bin()
    for k in ("XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "PRE_COMMIT_HOME", "npm_config_cache"):
        assert Path(env[k]).parent == tmp_path / STATE_DIR / "cache", k  # inside the workspace: writable


def test_policy_refusal_paths():
    tx = {"target": "0x" + "ab" * 20, "value": 10**15, "chain": "gnosis"}
    assert not Policy(mode="deny").check("safe_transaction", tx).allowed
    assert "signing is off" in Policy(mode="auto").check("sign_message", {"digest": "0x00"}).rule
    assert "no USD price" in Policy(mode="auto").check("safe_transaction", {**tx, "chain": "x"}).rule
    assert "max_payment" in Policy(mode="auto").check("mech_request", {"prompt": "q", "tool": "t"}).rule
    p = Policy(mode="auto", caps=SessionCaps(max_value_calls=1))
    assert p.check("safe_transaction", tx).allowed
    assert "value calls" in p.check("safe_transaction", tx).rule
    assert "unknown tool" in Policy(mode="auto").check("weird_tool", {}).rule
    assert not Policy(mode="ask").check("safe_transaction", tx).allowed  # no approver: declined


def test_int_and_summaries():
    assert _int(7) == 7 and _int("0x10") == 16 and _int("") == 0 and _int(" 5 ") == 5
    assert _int(None) == 0 and _int("0X10") == 16 and _int(True) == 1


def test_negative_and_junk_amounts_are_refused_never_counted():
    import pytest

    from onchainbench.connect.policy import NegativeAmount

    for v in (-1, "-5", "-0x10"):
        with pytest.raises(NegativeAmount):
            _int(v)
    for v in ("abc", 1.5, "0x"):
        with pytest.raises(ValueError):
            _int(v)
    # the repro: cap $0.10, a negative transfer used to lower the counter and let 1 xDAI through
    p = Policy(mode="auto", mech_asset_lookup=lambda c, m: (18, 1.0))
    tx = {"chain": "gnosis", "target": "0x" + "ab" * 20}
    assert not p.check("safe_transaction", {**tx, "value": 10**18}).allowed
    d = p.check("safe_transaction", {**tx, "value": -(10**20)})
    assert not d.allowed and d.rule.startswith("broker: rule 'negative-amount'")
    mech = {"chain": "gnosis", "prompt": "q", "tool": "t", "priority_mech": "0xm", "max_payment": "-1"}
    assert "'negative-amount'" in p.check("mech_request", mech).rule
    assert "'bad-amount'" in p.check("send_transaction", {**tx, "value": "ten"}).rule
    for v in (10**400, 2**256, hex(2**256)):  # past a uint256 (and the float conversion): refused, and logged
        d = p.check("safe_transaction", {**tx, "value": v})
        assert not d.allowed and "'bad-amount'" in d.rule and p.log[-1]["rule"] == d.rule
        assert "'bad-amount'" in p.check("mech_request", {**mech, "max_payment": v}).rule
    assert _int(2**256 - 1) == 2**256 - 1
    for v in ("1_000", "+5", "\u0665", "-", "0x-1", " 1 2 ", "9" * 5000):  # int() would take three of these
        with pytest.raises(ValueError) as e:
            _int(v)
        assert not isinstance(e.value, NegativeAmount)
    q = Policy(mode="auto", native_prices={"gnosis": -1.0})  # a negative price: refused, nothing counted
    d = q.check("safe_transaction", {**tx, "value": 10**15})
    assert not d.allowed and "'negative-amount'" in d.rule and (q.spent_native, q.value_calls) == (0, 0)
    assert "native cap" in p.check("safe_transaction", {**tx, "value": 2**256 - 1}).rule  # the largest: capped
    assert (p.spent_native, p.spent_mech, p.value_calls) == (0, 0, 0)
    assert not p.check("safe_transaction", {**tx, "value": 10**18}).allowed  # the cap still holds
    assert "agent EOA" in summarize("send_transaction", {"target": "0x1", "value": 1, "data": "0xa9059cbb" + "0" * 8})
    assert "(plain transfer)" in summarize("safe_transaction", {"to": "0x1"})
    # ERC-20 transfers and approvals name the recipient or spender and the amount (block E)
    rcpt = "ab" * 20
    tr = summarize(
        "safe_transaction",
        {"target": "0xt", "data": "0xa9059cbb" + rcpt.rjust(64, "0") + hex(3_500_000)[2:].rjust(64, "0")},
    )
    assert f"recipient 0x{rcpt}, amount 3500000" in tr
    ap = summarize("safe_transaction", {"target": "0xt", "data": "0x095EA7B3" + rcpt.rjust(64, "0") + "f" * 64})
    assert f"ERC-20 approve: spender 0x{rcpt}, amount {2**256 - 1}" in ap
    assert "pay mech" in summarize("mech_request", {"tool": "t", "prompt": "p"})
    assert "raw digest" in summarize("sign_message", {"digest": "0x1"})
    assert summarize("other", {"a": 1}) == "other({'a': 1})"


# ---- marketplace gate ------------------------------------------------------------


def test_read_only_tools_come_from_connect():
    from onchainbench.connect.policy import FALLBACK_READ_ONLY, read_only_from

    tools = [
        {"name": "wallet_info", "annotations": {"readOnlyHint": True}},
        {"name": "safe_transaction", "annotations": {"readOnlyHint": True}},  # mislabelled: still checked
        {"name": "new_view"},
    ]
    assert read_only_from(tools) == {"wallet_info"}
    assert read_only_from([{"name": "wallet_info"}]) is None  # no flags at all: keep the fallback
    p = Policy(mode="auto", read_only={"wallet_info", "safe_transaction"})
    tx = {"target": "0x" + "ab" * 20, "value": 10**15, "chain": "gnosis"}
    assert p.check("safe_transaction", tx).allowed and p.value_calls == 1  # counted as a value call
    assert "unknown tool" in p.check("mech_tools", {}).rule  # not flagged here: needs asking
    assert Policy().read_only == FALLBACK_READ_ONLY


@no_sandbox_on_windows
def test_seatbelt_denied_file_cant_be_renamed_and_read(tmp_path):

    secret = tmp_path / ".mcp.json"
    secret.write_text("SECRET")
    (tmp_path / "agent").mkdir()
    wrap, _ = wrapper_for(tmp_path, deny_read=[secret])
    for cmd in (
        "cat ../.mcp.json",
        "mv ../.mcp.json ../x; cat ../x",
        f"mv {tmp_path} {tmp_path}-moved; cat {tmp_path}-moved/.mcp.json",
    ):
        r = subprocess.run(
            [*wrap, "bash", "-c", cmd], cwd=tmp_path / "agent", capture_output=True, text=True, check=False
        )
        assert "SECRET" not in r.stdout, cmd
    assert secret.read_text() == "SECRET"  # still in place
    r = subprocess.run(
        [*wrap, "bash", "-c", "echo ok > note && cat note"],
        cwd=tmp_path / "agent",
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.stdout.strip() == "ok"  # normal writes in the workspace still work


def test_shell_deadline_waits_while_the_broker_holds(tmp_path):
    sh = PersistentShell(str(tmp_path), timeout=1)
    # each hold moves the deadline 1 s on: three of them give 4 s, room for a slow runner (Windows) to finish the 2.5 s
    held = iter([True, True, True, False, False])
    sh.hold = lambda: next(held, False)
    r = sh.run("sleep 2.5; echo done")  # longer than the 1 s deadline, but the broker held it
    assert r["output"].strip() == "done" and not r["timed_out"]
    assert sh.run("sleep 3")["timed_out"]  # nothing held: the deadline applies
    sh.hold, sh.max_hold = (lambda: True), 1.0  # held forever (a background loop of broker calls)
    assert sh.run("sleep 5")["timed_out"]  # still ends: the hold is capped
    sh.close()


MECHS = {"0xnative": (18, 1.0), "0xusdc": (6, 1.0), "0xodd": (18, None)}  # (decimals, USD) per mech


def _call(rng: random.Random, rid: str | None):
    if rng.random() < 0.5:
        value = rng.choice([10**14, 10**15, 5 * 10**16, 10**17])  # xDAI at $1
        return "safe_transaction", {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": value, "request_id": rid}
    mech = rng.choice(list(MECHS))
    pay = rng.choice([10**4, 10**5, 5 * 10**5]) if mech == "0xusdc" else rng.choice([10**15, 10**16, 10**17])
    return "mech_request", {
        "chain": "gnosis",
        "prompt": "q",
        "tool": "t",
        "max_payment": pay,
        "priority_mech": mech,
        "request_id": rid,
    }


def run(seed: int, n_calls: int = 40) -> dict:
    rng = random.Random(seed)
    mode = rng.choice(["auto", "ask"])
    asked = []

    def approver(summary):
        asked.append(summary)
        return rng.random() < 0.8

    p = Policy(mode=mode, caps=SessionCaps(), approver=approver, mech_asset_lookup=lambda c, m: MECHS.get(m))
    history: list[tuple[str, dict, bool]] = []
    out = {"violations": [], "replays": 0, "changed_retries": 0, "allowed": 0}
    for _ in range(n_calls):
        r = rng.random()
        if history and r < 0.25:  # replay an earlier call exactly
            tool, args, was_allowed = rng.choice(history)
            before = (p.spent_native, p.spent_mech, p.value_calls, len(asked))
            d = p.check(tool, dict(args))
            if was_allowed and args.get("request_id"):
                out["replays"] += 1
                if not d.allowed or (p.spent_native, p.spent_mech, p.value_calls, len(asked)) != before:
                    out["violations"].append("an approved retry changed counters or asked again")
        elif history and r < 0.4:  # same id, changed args
            tool, args, _ = rng.choice(history)
            if not args.get("request_id"):
                continue
            new_tool, new_args = _call(rng, args["request_id"])
            if new_args == args:
                continue
            out["changed_retries"] += 1
            n_log = len(p.log)
            d = p.check(new_tool, new_args)
            if len(p.log) != n_log + 1:
                out["violations"].append("a changed-args retry was not checked")
            history.append((new_tool, new_args, d.allowed))
        else:
            rid = f"r{rng.randrange(1000)}" if rng.random() < 0.7 else None
            tool, args = _call(rng, rid)
            d = p.check(tool, args)
            out["allowed"] += d.allowed
            history.append((tool, args, d.allowed))
        c = p.caps
        if p.spent_native > c.max_native_usd + 1e-9 or p.spent_mech > c.max_mech_usd + 1e-9:
            out["violations"].append(f"cap exceeded: native {p.spent_native}, mech {p.spent_mech}")
        if p.value_calls > c.max_value_calls:
            out["violations"].append("value-call cap exceeded")
    out["hit_cap"] = any("cap" in e["rule"] for e in p.log if not e["allowed"])
    return out


def test_caps_and_retries_hold_over_random_sequences():
    rs = [run(seed) for seed in range(2000)]
    assert [v for r in rs for v in r["violations"]] == []
    assert sum(r["replays"] for r in rs) > 1000 and sum(r["changed_retries"] for r in rs) > 1000  # both exercised
    assert sum(r["hit_cap"] for r in rs) > 1000  # the caps are reached, not just approached


def test_replaying_any_approved_args_set_is_free():
    p = Policy(mode="auto")
    a = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 10**15, "request_id": "r1"}
    b = {**a, "value": 2 * 10**15}
    p.check("safe_transaction", a)
    p.check("safe_transaction", b)
    calls = p.value_calls
    assert p.check("safe_transaction", a).allowed and p.value_calls == calls  # both sets were approved: free


def test_pearl_cli_commands(connect_ws, mock, monkeypatch, capsys):
    monkeypatch.chdir(connect_ws)  # the real .mcp.json here points straight at the mock
    assert pearl_cli.main([]) == 0 and "pearl tools" in capsys.readouterr().out
    assert pearl_cli.main(["tools"]) == 0 and "wallet_info" in capsys.readouterr().out
    assert pearl_cli.main(["help", "safe_transaction"]) == 0 and "parameters" in capsys.readouterr().out
    assert pearl_cli.main(["help", "nope"]) == 2
    assert pearl_cli.main(["wallet_info"]) == 0 and '"eoa"' in capsys.readouterr().out
    assert pearl_cli.main(["mech_result", "request_id=x"]) == 1  # tool error
    assert pearl_cli.main(["safe_transaction", "--chain"]) == 1  # flag without a value
    assert pearl_cli.main(["safe_transaction", "oops"]) == 1  # not key=value
    monkeypatch.chdir(connect_ws.parent.parent)
    assert pearl_cli.main(["tools"]) == 1  # no .mcp.json anywhere up the tree


# ---- the olas-ocb-agent REPL ------------------------------------------------------------------


def test_pearl_cli_json_args_hints_and_unknown_tool(connect_ws, mock, monkeypatch, capsys):
    monkeypatch.chdir(connect_ws)
    assert pearl_cli.main(["mech_tools", json.dumps({"chain": "gnosis"})]) == 0
    assert pearl_cli.main(["mech_request", "prompt=q", "tool=t", "mech=0x1"]) == 1
    assert "did you mean priority_mech" in capsys.readouterr().err
    assert pearl_cli.main(["mech_request", "prompt=q", "zzz=1", "tool=t"]) == 1  # unknown key with no hint
    assert pearl_cli.main(["safe_transaction", "value=1"]) == 1
    assert "needs: chain, target" in capsys.readouterr().err
    assert pearl_cli.main(["no_such_tool"]) == 2


def test_broker_get_requires_token(connect_ws, mock):

    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(rt.broker.url.removesuffix("/mcp/") + "/wallet", timeout=5)
        assert e.value.code == 401
    finally:
        rt.close()


def test_shell_state_persists_and_stdin_is_closed(tmp_path):
    (tmp_path / "sub").mkdir()
    sh = PersistentShell(str(tmp_path), timeout=10)
    assert sh.run("cd sub && export FOO=bar")["returncode"] == 0
    r = sh.run(f"{PWD}; echo $FOO")
    out = r["output"].splitlines()
    assert [os.path.normpath(out[0]), *out[1:]] == [str((tmp_path / "sub").resolve()), "bar"]
    assert sh.run("cat")["returncode"] == 0  # reads /dev/null, does not hang
    assert sh.run("false")["returncode"] == 1
    sh.close()


def test_the_shell_writes_no_script_where_the_agent_can_write(tmp_path):
    # the command goes in on stdin: nothing in the working dir to list (a listed script's name once ended the action)
    # or for a background job to swap for a link (the harness would write the script through it)
    sh = PersistentShell(str(tmp_path), timeout=10)
    r = sh.run("cd /tmp && cd - >/dev/null; ls -A; echo \"it's 'q'\"; echo after")
    assert r["returncode"] == 0 and r["output"].split("\n")[0] == "it's 'q'", r["output"]
    assert r["output"].strip().endswith("after") and list(tmp_path.iterdir()) == []
    r = sh.run("if then")  # a syntax error is the command's failure, and the shell lives on
    assert r["returncode"] != 0 and sh.run("echo ok")["output"].strip() == "ok"
    sh.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX sessions")
def test_close_kills_a_child_that_left_the_process_group(tmp_path):
    import sys
    import time

    sh = PersistentShell(str(tmp_path), timeout=10)
    code = "import os, time; os.setsid(); open('pid', 'w').write(str(os.getpid())); time.sleep(60)"
    sh.run(f"{sys.executable} -c {code!r} >/dev/null 2>&1 &")
    for _ in range(100):
        if (tmp_path / "pid").is_file() and (tmp_path / "pid").read_text():
            break
        time.sleep(0.05)
    pid = int((tmp_path / "pid").read_text())
    sh.close()
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the setsid child outlived the shell")


def test_shell_timeout_restarts_in_same_cwd(tmp_path):
    (tmp_path / "d").mkdir()
    sh = PersistentShell(str(tmp_path), timeout=10)
    sh.run("cd d")
    r = sh.run("sleep 5", timeout=1)
    assert r["timed_out"]
    assert os.path.normpath(sh.run(PWD)["output"].strip()) == str((tmp_path / "d").resolve())
    sh.close()


# ---- broker + pearl CLI + sandbox -----------------------------------------
@no_sandbox_on_windows
def test_sandbox_blocks_real_token_but_pearl_cli_works(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto")
    try:
        r = rt.shell.run("cat ../.mcp.json")  # Seatbelt refuses; bubblewrap shows an empty file
        assert mock.token not in r["output"] and (r["returncode"] != 0 or not r["output"].strip())
        assert mock.token not in rt.shell.run("cat .mcp.json")["output"]  # shim holds the broker token only
        r = rt.shell.run("pearl wallet_info")
        assert r["returncode"] == 0, r["output"]
        assert json.loads(r["output"])["chains"]["gnosis"]["safe"] == SAFE
        r = rt.shell.run("echo x > /Users/Shared/olas-ocb-agent-should-not-write.txt")
        assert r["returncode"] != 0
        # the workspace root is read-only to the agent (it sits under pytest's temp dir here, as the bench's does
        # under its own): the brief, the skills, the real .mcp.json, anything new, and renaming it or the dir above
        brief = (connect_ws / "AGENTS.md").read_text()
        skill = connect_ws / ".agents" / "skills" / "pearl-connect" / "SKILL.md"
        before = sorted(str(p.relative_to(connect_ws)) for p in connect_ws.rglob("*") if "agent" not in p.parts)
        tmp = connect_ws.parent
        for cmd in (
            "echo INJECTED >> ../AGENTS.md",
            "echo INJECTED >> ../.agents/skills/pearl-connect/SKILL.md",
            "echo x > ../CLAUDE.md",
            "mkdir -p ../.claude/skills",
            "echo x > ../GEMINI.md",
            "echo FAKE > ../.mcp.json",
            f"mv {connect_ws} {connect_ws}.old",
            f"mv {tmp} {tmp}.old",
            f"d=$(mktemp -d) && mv {connect_ws / 'agent'} $d/moved",
        ):
            assert rt.shell.run(cmd)["returncode"] != 0, cmd
        assert (connect_ws / "AGENTS.md").read_text() == brief and "INJECTED" not in skill.read_text()
        after = sorted(str(p.relative_to(connect_ws)) for p in connect_ws.rglob("*") if "agent" not in p.parts)
        assert after == before and mock.token in (connect_ws / ".mcp.json").read_text()
        assert (connect_ws / "agent").is_dir() and not (connect_ws / "agent").is_symlink()
        real = os.path.realpath(connect_ws)  # the instruction files are denied by name too, from open_session()
        if rt.sandbox_desc.startswith("macOS"):
            assert f'(deny file-write* (subpath "{real}/AGENTS.md"))' in rt.shell.wrapper[2]
        else:
            assert f"{real}/AGENTS.md" in rt.shell.wrapper
        r = rt.shell.run("echo ok > note && cat note && f=$(mktemp) && echo t > $f && cat $f")
        assert r["returncode"] == 0 and r["output"].split() == ["ok", "t"]  # its own dir and temp dir stay writable
    finally:
        rt.close()


def test_the_harness_never_writes_through_a_link_the_agent_planted(connect_ws, tmp_path):
    """The agent's directory is writable, so it can plant links where the harness writes on the next session."""
    from onchainbench.connect.workspace import STATE_DIR

    outside = tmp_path / "outside"
    outside.write_text("untouched")
    brief = (connect_ws / "AGENTS.md").read_text()
    build(connect_ws, model=ScriptedModel([]), sandbox="none").close()
    agent = connect_ws / "agent"
    (agent / ".mcp.json").unlink()
    (agent / ".mcp.json").symlink_to("../AGENTS.md")
    (agent / STATE_DIR / "bin" / "pearl").unlink()
    (agent / STATE_DIR / "bin" / "pearl").symlink_to(outside)
    (agent / ".gitignore").unlink()
    (agent / ".gitignore").symlink_to(tmp_path / "dangling")
    rt = build(connect_ws, model=ScriptedModel([]), sandbox="none")
    rt.close()
    assert (connect_ws / "AGENTS.md").read_text() == brief and outside.read_text() == "untouched"
    for name in (".mcp.json", f"{STATE_DIR}/bin/pearl", ".gitignore"):
        assert not (agent / name).is_symlink() and (agent / name).is_file(), name
    assert not (tmp_path / "dangling").exists() and os.access(agent / STATE_DIR / "bin" / "pearl", os.X_OK)
    shutil.rmtree(agent / STATE_DIR)
    (agent / STATE_DIR).symlink_to(tmp_path)
    with pytest.raises(RuntimeError, match="symlink"):
        build(connect_ws, model=ScriptedModel([]), sandbox="none")
    # the agent's directory itself swapped for a link (to anywhere, or to the root): the next session refuses it
    shutil.rmtree(agent)
    for target in (tmp_path, Path(".")):
        agent.symlink_to(target)
        with pytest.raises(RuntimeError, match="symlink"):
            build(connect_ws, model=ScriptedModel([]), sandbox="none")
        agent.unlink()
    (connect_ws / "real").mkdir()
    (connect_ws / "link").symlink_to("real")
    with pytest.raises(RuntimeError, match="symlink"):  # or a directory under a link
        Workspace.open(connect_ws, agent_subdir="link/agent")


def test_broker_policy_ask_auto_caps_and_idempotency(connect_ws, mock):
    answers = []
    rt = build(
        connect_ws,
        model=ScriptedModel([]),
        approval="ask",
        sandbox="none",
        approver=lambda s: answers.pop(0) if answers else False,
        caps=SessionCaps(max_native_usd=0.01, max_value_calls=5),
    )
    try:
        cli = ConnectClient(rt.broker.url, rt.broker.token)
        tx = {"to": "0x" + "ab" * 20, "value": 10**15, "request_id": "r1"}
        res = cli.call_tool_raw("safe_transaction", tx)
        assert res["isError"] and "declined" in res["content"][0]["text"]
        answers.append(True)
        assert not cli.call_tool_raw("safe_transaction", tx).get("isError")
        assert not cli.call_tool_raw("safe_transaction", tx).get("isError")  # retry, same id: no new prompt
        assert len([e for e in mock.s.ledger if e["kind"] == "tx"]) == 1
        rt.policy.mode = "auto"
        big = {"to": "0x" + "ab" * 20, "value": 10**16, "request_id": "r2"}
        res = cli.call_tool_raw("safe_transaction", big)
        assert res["isError"] and "native cap" in res["content"][0]["text"]
        assert not cli.call_tool_raw("wallet_info", {}).get("isError")  # reads never gated
        # signer_client.py's REST route is gated the same way
        with pytest.raises(Exception, match="native cap"):
            cli.rest("POST", "/safe-transaction", {**big, "chain": "gnosis"})
    finally:
        rt.close()


def test_broker_rejects_wrong_token(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        with pytest.raises(Exception, match="401"):
            ConnectClient(rt.broker.url, mock.token).list_tools()  # the upstream token is not the broker's
    finally:
        rt.close()


# ---- agent loop ------------------------------------------------------------
def test_pearl_cli_kv_args(connect_ws, mock):
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        r = rt.shell.run(f"pearl mech_tools priority_mech={LLM_MECH}")
        assert r["returncode"] == 0 and json.loads(r["output"])["tools"] == ["chat"]
        r = rt.shell.run("pearl help safe_transaction")
        assert "parameters" in r["output"]
        r = rt.shell.run("pearl nonsense_tool")
        assert r["returncode"] != 0
    finally:
        rt.close()


def test_pearl_cli_flag_styles_and_unknown_keys(connect_ws, mock):
    from onchainbench.connect.pearl_cli import parse_args

    assert parse_args(["--chain=gnosis", "--max-payment", "5", "tool=x"]) == {
        "chain": "gnosis",
        "max_payment": 5,
        "tool": "x",
    }
    rt = build(connect_ws, model=ScriptedModel([]), approval="auto", sandbox="none")
    try:
        r = rt.shell.run("pearl mech_request prompt=hi tool=prediction-online mech=0xabc max_payment=1")
        assert r["returncode"] == 1 and "did you mean priority_mech" in r["output"]
        r = rt.shell.run("pearl safe_transaction --target 0x" + "11" * 20 + " --value 1")
        assert r["returncode"] == 1 and "needs: chain" in r["output"]
    finally:
        rt.close()


def test_shell_deadline_and_clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    sh = PersistentShell(str(tmp_path), timeout=10)
    r = sh.run("while true; do echo tick; sleep 0.2; done", timeout=2)  # prints constantly: still ends at the deadline
    assert r["timed_out"]
    assert "sk-should-not-leak" not in sh.run("env")["output"]
    sh.close()


def test_retry_with_same_id_but_different_args_needs_approval():
    answers = [True, False]
    p = Policy(
        mode="ask", caps=SessionCaps(max_native_usd=1.0), approver=lambda s: answers.pop(0), default_chain="gnosis"
    )
    tx = {"target": "0x" + "ab" * 20, "value": 1, "request_id": "r", "chain": "gnosis"}
    assert p.check("safe_transaction", tx).allowed
    assert p.check("safe_transaction", dict(tx)).allowed  # true retry: no prompt
    assert not p.check(
        "safe_transaction", {**tx, "value": 10**17}
    ).allowed  # $0.10 on gnosis: asked, declined  # same id, bigger value: asked, declined


def test_mech_cap_binds_in_usd_on_gnosis():
    p = Policy(
        mode="auto",
        caps=SessionCaps(max_mech_usd=1.0),
        default_chain="gnosis",
        mech_asset_lookup=lambda chain, mech: (18, 1.0),
    )
    req = {"prompt": "q", "tool": "prediction-online", "max_payment": 6 * 10**17, "priority_mech": "0x1"}  # 0.6 xDAI
    assert p.check("mech_request", {**req, "request_id": "a"}).allowed
    d = p.check("mech_request", {**req, "request_id": "b"})  # 1.2 xDAI in total > $1
    assert not d.allowed and "mech payment cap" in d.rule


def test_mech_payment_asset_resolution_in_the_broker():
    req = {"prompt": "q", "tool": "t", "max_payment": 10**6, "chain": "ethereum"}
    assert "name the mech" in Policy(mode="auto").check("mech_request", req).rule  # always named
    assert "isn't known" in Policy(mode="auto").check("mech_request", {**req, "priority_mech": "0x1"}).rule
    p = Policy(mode="auto", mech_asset_lookup=lambda chain, mech: (6, 1.0))
    assert p.check("mech_request", {**req, "priority_mech": "0x1"}).allowed and p.spent_mech == pytest.approx(1.0)


# ---- runtime's token lookup ----------------------------------------------------------


def test_mech_asset_lookup_native_token_and_failures(monkeypatch):
    m = MockConnect(MockState(chain="base"))  # base: mechs paid in the mock USDC
    try:
        c = ConnectClient(m.url, m.token)
        chains = c.rest("GET", "/wallet")["chains"]
        look = mech_asset_lookup(c, chains, {})
        assert look("base", PRED_MECH) == (6, 1.0)  # USDC: decimals and symbol read via the mock RPC
        assert look("base", PRED_MECH.upper().replace("X", "x")) == (6, 1.0)  # cached, case-insensitive
        assert look("base", "0x" + "99" * 20) is None  # unknown mech: Connect refuses
        assert mech_asset_lookup(c, {}, {})("base", PRED_MECH) is None  # no RPC for the chain
        bad = {"base": {"rpc": m.url.removesuffix("/mcp/") + "/rpc"}}
        m.s.mechs[PRED_MECH]["price"] = 1
        monkeypatch.setattr(m, "rpc", lambda req: {"jsonrpc": "2.0", "id": 1, "error": {}})
        assert mech_asset_lookup(c, bad, {})("base", PRED_MECH) is None  # RPC can't answer
    finally:
        m.stop()
    g = MockConnect(MockState())  # gnosis: native xDAI
    try:
        c = ConnectClient(g.url, g.token)
        assert mech_asset_lookup(c, {}, {})("gnosis", PRED_MECH) == (18, 1.0)
    finally:
        g.stop()
    assert _decode_symbol("0x" + b"WXDAI".hex().ljust(64, "0")) == "WXDAI"  # bytes32-style symbol


def test_native_prices_and_a_session_without_wallet_or_hook(connect_ws, mock, monkeypatch):
    """A user price overrides the table; a session still opens when Connect's /wallet fails (no default chain)."""
    assert native_usd("gnosis", {"gnosis": 2.0}) == 2.0 and native_usd("nochain") is None

    def no_wallet(self, method, path, body=None):
        raise ConnectError("wallet down")

    monkeypatch.setattr(ConnectClient, "rest", no_wallet)
    cs = open_session(connect_ws, SessionCaps(), sandbox="none")  # and no on_broker hook
    try:
        assert cs.policy.default_chain is None and cs.shell.run("echo ok")["output"].strip() == "ok"
    finally:
        cs.close()


def test_windows_shell_is_git_bash_not_wsl_and_kills_the_tree(tmp_path, monkeypatch):
    from onchainbench.connect import shell as S

    monkeypatch.setattr(S, "WINDOWS", True)
    system32, git = tmp_path / "Windows" / "System32", tmp_path / "Git" / "bin"
    for d in (system32, git):  # System32's bash.exe is WSL's launcher: never the one to start
        d.mkdir(parents=True)
        for name in ("bash", "bash.exe"):
            (d / name).write_text("")
            (d / name).chmod(0o755)
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    monkeypatch.setenv("PATH", os.pathsep.join(["", str(system32), str(git)]))
    assert os.path.dirname(S._bash()) == str(git)
    monkeypatch.setenv("PATH", str(system32))
    with pytest.raises(RuntimeError, match="Git Bash"):
        S._bash()
    calls = []
    monkeypatch.setattr(S.subprocess, "run", lambda argv, **kw: calls.append(argv))
    S._kill_tree(123)
    assert calls == [["taskkill", "/F", "/T", "/PID", "123"]]
