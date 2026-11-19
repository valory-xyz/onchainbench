"""Agent isolation (onchainbench/isolation.py, onchainbench/agent_exec.py): the launch wrapper, the rules, the ACLs and
the session around an arm, with the privileged commands mocked; then, as root on Linux with the agent users present,
the real thing: the agent user refused on an Anvil-like port and let through on a Connect-like one, and the cheater
refused rather than caught."""

import json
import os
import platform
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from onchainbench import agent_exec as X
from onchainbench import isolation as I

# isolation is Linux-only (users, iptables, setuid): these tests use POSIX calls (grp, getuid, SIGKILL) that Windows
# lacks. Linux runs them all and gates coverage; Windows still imports the modules above, as the bench does.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="agent isolation is Linux-only (POSIX calls)")

SLOT = I.Slot(0, "ocb-agent-0", 4242, 4243, (4243, 4300))


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(I, "_state", {"pool": None, "private": None})
    yield
    subprocess.Popen = I._ORIG_POPEN  # every test leaves Popen as it found it
    I._local.agent = None


@pytest.fixture
def helper_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(I, "run_helper", lambda spec, timeout=60: calls.append(spec))
    return calls


def trial_dir():
    return Path(tempfile.mkdtemp(prefix="pe2-test-"))


# ---- the root helper's pieces ----------------------------------------------------------------------
def test_chain_rules_allow_only_the_given_ports_then_refuse():
    text = X.chain_rules("OCB-AGENT-3", [8080, "9090", 8080])
    lines = text.splitlines()
    assert lines[0] == "*filter" and lines[1] == ":OCB-AGENT-3 - [0:0]" and lines[-1] == "COMMIT"
    assert "-A OCB-AGENT-3 -p tcp --dport 8080 -j ACCEPT" in lines and text.count("--dport 8080") == 1
    assert "-A OCB-AGENT-3 -p tcp --dport 9090 -j ACCEPT" in lines
    assert "-A OCB-AGENT-3 -p udp --dport 53 -j ACCEPT" in lines  # DNS
    assert lines[-3:-1] == ["-A OCB-AGENT-3 -p tcp -j REJECT --reject-with tcp-reset", "-A OCB-AGENT-3 -j REJECT"]
    assert X.jump_rule(7, "C") == ["OUTPUT", "-o", "lo", "-m", "owner", "--uid-owner", "7", "-j", "C"]
    assert ":OCB-AGENT-3-R" not in text  # no uid: ports only
    with pytest.raises(ValueError):
        X.chain_rules("C", [70000])


def test_chain_rules_let_the_slot_reach_its_own_listeners_and_no_one_elses():
    """A CLI's local gateway (Hermes's, on an ephemeral port) must be reachable by the slot that started it: a new
    connection to an unlisted port is marked, and only a reply from the slot's own uid completes it."""
    lines = X.chain_rules("OCB-AGENT-3", [8080], uid=4242).splitlines()
    c, r = "-A OCB-AGENT-3", "-A OCB-AGENT-3-R"
    mark, denied = "0x4000000/0xff000000", "0x84000000/0xff000000"  # slot 3: index + 1 in the top byte
    assert lines[1] == ":OCB-AGENT-3 - [0:0]"
    assert lines[2] == f"{c} -p tcp -m connmark --mark {denied} -j REJECT --reject-with tcp-reset"  # first: refused
    allowed = lines.index(f"{c} -p tcp --dport 8080 -j ACCEPT")
    marking = lines.index(f"{c} -p tcp --syn -m conntrack --ctstate NEW -j CONNMARK --set-xmark {mark}")
    assert allowed < marking  # a listed harness port is never marked: its replies aren't the slot's
    assert lines[marking + 1] == f"{c} -p tcp --syn -m conntrack --ctstate NEW -j ACCEPT"
    assert lines[marking + 2 : marking + 4] == [f"{c} -p tcp -j REJECT --reject-with tcp-reset", f"{c} -j REJECT"]
    reply = lines[lines.index(":OCB-AGENT-3-R - [0:0]") + 1 : -1]
    answered = "0x800000/0x800000"
    assert reply == [
        f"{r} -p tcp --tcp-flags RST RST -j RETURN",
        f"{r} -m connmark --mark {answered} -j RETURN",  # the handshake passed: the rest of the connection
        # only the SYN-ACK is judged by its owner (smoke-matrix-4: the kernel ACKs data on a connection the listener
        # hasn't accepted yet from a socket no process holds, which matched no uid and cost the connection)
        f"{r} -p tcp --tcp-flags SYN,ACK SYN,ACK -m owner --uid-owner 4242 -j CONNMARK --set-xmark {answered}",
        f"{r} -m connmark --mark {answered} -j RETURN",
        f"{r} -j CONNMARK --set-xmark {denied}",  # anyone else's (the harness's): denied, and dropped
        f"{r} -j DROP",
    ]
    assert X.ANSWERED & (X.MARK_MASK | X.DENIED) == 0  # its own bit: the slot's mark and the denial are unchanged
    # the preflight's own-listener probe speaks first and accepts late, as a busy gateway does
    assert 'c.sendall(b"ping"); time.sleep(0.5)' in I.OWN_PROBE and "c.recv(4)" in I.OWN_PROBE
    assert X.reply_jump_rule("OCB-AGENT-3") == [
        *("OUTPUT", "-o", "lo", "-p", "tcp", "-m", "connmark", "--mark", "0x4000000/0x7f000000"),
        *("-m", "conntrack", "--ctdir", "REPLY", "-j", "OCB-AGENT-3-R"),
    ]
    assert X.slot_mark("OCB-AGENT-63") == 64 << 24 and 64 << 24 & X.MARK_MASK == 64 << 24  # the last slot fits


@pytest.fixture
def agent_users(monkeypatch):
    import grp
    import pwd

    users = {4242: "ocb-agent-0", 4250: "ocb-agent-1", 0: "root", 1000: "runner", 4260: "ocb-agent-x"}

    def getpwuid(uid):
        if uid not in users:
            raise KeyError(uid)
        return types.SimpleNamespace(pw_name=users[uid], pw_uid=uid, pw_gid=uid + 1)

    monkeypatch.setattr(pwd, "getpwuid", getpwuid)
    monkeypatch.setattr(grp, "getgrnam", lambda n: types.SimpleNamespace(gr_gid=4300))
    return users


def test_the_helper_acts_for_agent_users_only(agent_users, monkeypatch):
    assert X.slot_user(4242) == ("ocb-agent-0", 4243, {4243, 4300})
    for uid in (0, 1000, 4260, 99):
        with pytest.raises(X.Refused):
            X.slot_user(uid)
    import grp

    def no_group(n):
        raise KeyError(n)

    monkeypatch.setattr(grp, "getgrnam", no_group)
    assert X.slot_user(4242)[2] == {4243}


def test_firewall_restores_both_families_and_adds_the_jump_once(agent_users, monkeypatch):
    runs = []

    def run(cmd, **kw):
        runs.append((cmd, kw.get("input")))
        return types.SimpleNamespace(returncode=1 if "-C" in cmd and "ip6tables" in cmd[0] else 0)

    monkeypatch.setattr(X.subprocess, "run", run)
    monkeypatch.setattr(X.shutil, "which", lambda n, path=None: f"/usr/sbin/{n}")
    X.firewall({"chain": "OCB-AGENT-0", "uid": 4242, "ports": [5000]})
    cmds = [c for c, _ in runs]
    assert cmds[0][:3] == ["/usr/sbin/iptables-restore", "-w", "--noflush"] and "--dport 5000" in runs[0][1]
    assert ":OCB-AGENT-0-R - [0:0]" in runs[0][1] and "--uid-owner 4242 -j CONNMARK" in runs[0][1]  # own listeners
    assert cmds[1][2] == "-C" and cmds[2][2] == "-C"  # v4: both jumps exist already
    assert cmds[1][-1] == "OCB-AGENT-0" and cmds[2][-1] == "OCB-AGENT-0-R"
    assert cmds[3][0] == "/usr/sbin/ip6tables-restore" and cmds[4][2] == "-C"
    assert cmds[5][:5] == ["/usr/sbin/ip6tables", "-w", "-I", "OUTPUT", "1"] and cmds[5][-1] == "OCB-AGENT-0"
    assert cmds[6][2] == "-C" and cmds[7][:5] == ["/usr/sbin/ip6tables", "-w", "-I", "OUTPUT", "1"]
    assert cmds[7][-1] == "OCB-AGENT-0-R" and len(cmds) == 8
    for bad in ({"chain": "OUTPUT", "uid": 4242}, {"chain": "OCB-AGENT-0", "uid": 0}):
        with pytest.raises(X.Refused):
            X.firewall(bad)
    monkeypatch.setattr(X.shutil, "which", lambda n, path=None: None)
    with pytest.raises(RuntimeError):
        X._tool("iptables")


def test_kill_signals_only_agent_owned_pids(agent_users, monkeypatch):
    owners = {10: 4242, 11: 1000, 12: None, 13: 4242}
    sent = []

    def kill(pid, sig):
        if pid == 13:
            raise ProcessLookupError
        sent.append((pid, sig))

    monkeypatch.setattr(X, "proc_uid", owners.get)
    monkeypatch.setattr(X.os, "kill", kill)
    X.kill({"pids": [10, 11, 12, 13], "uids": [4242], "sig": 15})
    assert sent == [(10, 15)]
    with pytest.raises(X.Refused):
        X.kill({"pids": [1], "uids": [0]})


def test_proc_uid_reads_the_real_uid(monkeypatch, tmp_path):
    f = tmp_path / "status"
    f.write_text("Name:\tx\nUid:\t1001\t1001\t1001\t1001\n")
    real_open = open
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: real_open(f if str(p).startswith("/proc/") else p, *a))
    assert X.proc_uid(5) == 1001
    f.write_text("Name:\tx\n")
    assert X.proc_uid(5) is None
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: (_ for _ in ()).throw(OSError()))
    assert X.proc_uid(5) is None


def test_killall_kills_as_the_slot_until_nothing_is_left(agent_users, monkeypatch):
    forks, done = [], []
    monkeypatch.setattr(X.os, "fork", lambda: forks.append(1) or 77)
    monkeypatch.setattr(X.os, "waitpid", lambda pid, f: done.append(pid))
    X.killall({"uid": 4242})
    assert forks and done == [77]
    with pytest.raises(X.Refused):
        X.killall({"uid": 1000})
    ids, kills = [], [None, None, ProcessLookupError()]

    def kill(pid, sig):
        assert pid == -1 and sig == signal.SIGKILL
        if (e := kills.pop(0)) is not None:
            raise e

    for name in ("setgroups", "setgid", "setuid"):
        monkeypatch.setattr(X.os, name, lambda v, n=name: ids.append((n, v)))
    monkeypatch.setattr(X.os, "kill", kill)
    monkeypatch.setattr(X.time, "sleep", lambda s: None)
    monkeypatch.setattr(X, "live_pids", lambda uid: [1])
    assert X._kill_own(4242, 4243) == 0 and ids == [("setgroups", []), ("setgid", 4243), ("setuid", 4242)]
    monkeypatch.setattr(X.os, "kill", lambda pid, sig: None)
    assert X._kill_own(4242, 4243, rounds=3) == 1
    monkeypatch.setattr(X, "live_pids", lambda uid: [os.getpid()])  # only itself (and zombies) left
    assert X._kill_own(4242, 4243, rounds=3) == 0


def test_live_pids_leaves_out_zombies_and_other_users(monkeypatch, tmp_path):
    procs = {"10": ("S", 4242), "11": ("Z", 4242), "12": ("R", 1000), "13": None, "14": ("S", 4242)}
    for pid, v in procs.items():
        if v:
            (tmp_path / f"{pid}.stat").write_text(f"{pid} (a (b) c) {v[0]} 1 2" if pid != "14" else f"{pid} broken")
            (tmp_path / f"{pid}.status").write_text(f"Uid:\t{v[1]}\t0\t0\t0\n")
    real_open = open

    def fake_open(p, *a, **k):
        if str(p).startswith("/proc/"):
            _, _, pid, kind = str(p).split("/")
            return real_open(tmp_path / f"{pid}.{kind}", *a)
        return real_open(p, *a, **k)

    monkeypatch.setattr(X.os, "listdir", lambda d: [*procs, "self"])
    monkeypatch.setattr("builtins.open", fake_open)
    assert X.live_pids(4242) == [10]


def test_reclaim_hands_back_the_slots_files_without_following_links(agent_users, monkeypatch, tmp_path):
    root = trial_dir()
    (root / "d").mkdir()
    (root / "d" / "f").write_text("x")
    (root / "link").symlink_to("/etc/passwd")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    (tmp / "agent-dir").mkdir()
    (tmp / "agent-dir" / "g").write_text("y")
    (tmp / "agent-file").write_text("z")
    os.mkfifo(tmp / "agent-fifo")
    (tmp / "agent-link").symlink_to("/etc/passwd")
    me = os.getuid()
    monkeypatch.setattr(X, "slot_user", lambda uid: ("ocb-agent-0", 1, set()))
    chowns = []
    real_stat = os.stat

    def fake_stat(p, *a, **k):  # everything looks like the slot's
        st = real_stat(p, *a, **k)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 4242, *st[5:]))

    monkeypatch.setattr(X.os, "stat", fake_stat)
    monkeypatch.setattr(X.os, "fstat", fake_stat)
    monkeypatch.setattr(X.os, "chown", lambda p, u, g, **k: chowns.append((p, u, k.get("follow_symlinks"))))
    monkeypatch.setattr(X.os, "fchown", lambda fd, u, g: chowns.append(("fd", u, None)))
    monkeypatch.setattr(X, "TMP_DIRS", (str(tmp), str(tmp_path / "missing")))
    try:
        X.reclaim({"uid": 4242, "to_uid": me, "to_gid": os.getgid(), "roots": [str(root)]})
    finally:
        monkeypatch.undo()
        import shutil

        shutil.rmtree(root)
    names = {c[0] for c in chowns}
    assert {"d", "f", "link"} <= names and all(c[1] == me for c in chowns)
    assert all(c[2] is False for c in chowns if c[0] != "fd")  # never through a link
    assert str(tmp / "agent-link") in names and str(tmp / "agent-fifo") in names
    assert stat.S_IMODE(os.stat(tmp / "agent-dir").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(tmp / "agent-file").st_mode) == 0o600
    monkeypatch.setattr(X, "slot_user", lambda uid: ("ocb-agent-0", 1, set()))
    with pytest.raises(X.Refused, match="caller"):  # back to the caller only
        X.reclaim({"uid": 4242, "to_uid": me + 1, "to_gid": 0, "roots": []})
    with pytest.raises(X.Refused):  # a trial directory directly in a temp dir only
        X.reclaim({"uid": 4242, "to_uid": me, "to_gid": 0, "roots": ["/etc"]})


def test_reclaim_moves_the_slots_shared_tmp_entries_into_its_trial(agent_users, monkeypatch, tmp_path):
    """The slot's /tmp entries become its trial's (the run keeps them by trial): no end-of-run sweep has to guess."""
    root = trial_dir()
    collect = Path(tempfile.mkdtemp(prefix=".shared-tmp-", dir=root))
    tmp = Path(tempfile.mkdtemp(dir=root.parent, prefix="ocb-test-shared-"))
    (tmp / "helper.py").write_text("x")
    (tmp / "work").mkdir()
    (tmp / "theirs").write_text("another uid's")
    me = os.getuid()
    real_stat = os.stat

    def fake_stat(p, *a, **k):
        st = real_stat(p, *a, **k)
        uid = 1 if str(p).endswith("theirs") else 4242
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, uid, *st[5:]))

    monkeypatch.setattr(X, "slot_user", lambda uid: ("ocb-agent-0", 1, set()))
    monkeypatch.setattr(X.os, "stat", fake_stat)
    monkeypatch.setattr(X.os, "chown", lambda *a, **k: None)
    monkeypatch.setattr(X.os, "fchown", lambda *a: None)
    monkeypatch.setattr(X, "TMP_DIRS", (str(tmp),))
    spec = {"uid": 4242, "to_uid": me, "to_gid": os.getgid(), "roots": [str(root)]}
    try:
        X.reclaim({**spec, "collect_to": str(collect)})
        assert sorted(p.name for p in collect.iterdir()) == ["helper.py", "work"]
        assert sorted(p.name for p in tmp.iterdir()) == ["theirs"]
        for bad in ("/tmp", str(tmp), str(collect / "nope")):  # only a directory of the caller's in the trial dir
            with pytest.raises((X.Refused, OSError)):
                X.reclaim({**spec, "collect_to": bad})
    finally:
        monkeypatch.undo()
        import shutil

        shutil.rmtree(root)
        shutil.rmtree(tmp)


def test_reclaim_skips_what_vanishes_or_resists(agent_users, monkeypatch, tmp_path):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    (tmp / "gone").write_text("x")
    (tmp / "other").write_text("x")
    monkeypatch.setattr(X, "slot_user", lambda uid: ("ocb-agent-0", 1, set()))
    monkeypatch.setattr(X, "TMP_DIRS", (str(tmp),))

    real_stat = os.stat
    owners = {"other": 1, "x": 4242}

    def fake_stat(p, *a, **k):
        name = os.path.basename(str(p))
        if name == "gone":
            raise FileNotFoundError(p)
        st = real_stat(tmp / "other", *a, **k)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, owners[name], *st[5:]))

    monkeypatch.setattr(X.os, "listdir", lambda d: ["gone", "other", "x"])
    monkeypatch.setattr(X.os, "stat", fake_stat)
    X.reclaim({"uid": 4242, "to_uid": os.getuid(), "to_gid": 0})  # "x" doesn't exist: its open fails, skipped
    monkeypatch.setattr(X.os, "stat", real_stat)
    assert stat.S_IMODE(os.stat(tmp / "other").st_mode) != 0o600

    # inside a tree, an entry that can't be stat'ed is skipped
    def stat_fails_for_f(p, *a, **k):
        if p == "f":
            raise OSError(p)
        return real_stat(p, *a, **k)

    monkeypatch.setattr(X.os, "stat", stat_fails_for_f)
    d = tmp_path / "t"
    d.mkdir()
    (d / "f").write_text("x")
    fd = os.open(d, os.O_RDONLY)
    X._chown_tree(fd, 4242, 0, 0)
    os.close(fd)


def test_exec_drops_to_the_slot_and_execs(agent_users, monkeypatch):
    calls = []
    for name in ("setgroups", "setgid", "setuid", "umask"):
        monkeypatch.setattr(X.os, name, lambda v, n=name: calls.append((n, v)))
    monkeypatch.setattr(X.os, "getuid", lambda: 4242)
    monkeypatch.setattr(X.os, "geteuid", lambda: 4242)
    monkeypatch.setattr(X.os, "getppid", lambda: 50)
    monkeypatch.setattr(X, "die_with_parent", lambda: calls.append(("pdeathsig", 9)))
    monkeypatch.setattr(X.os, "execvpe", lambda f, a, e: calls.append(("exec", (f, a, e))))
    spec = {"uid": 4242, "gid": 4243, "groups": [4243, 4300], "umask": 0o077, "argv": ["bash", "-c", "x"], "env": {}}
    X.run_exec(spec)
    assert [c[0] for c in calls] == ["setgroups", "setgid", "setuid", "umask", "pdeathsig", "exec"]
    assert calls[0][1] == [4243, 4300] and calls[-1][1] == ("bash", ["bash", "-c", "x"], {})
    for bad in ({"gid": 0}, {"groups": [27]}, {"argv": []}, {"argv": [1]}, {"uid": 1000}):
        with pytest.raises(X.Refused):
            X.run_exec(spec | bad)
    monkeypatch.setattr(X.os, "getppid", lambda: 1)
    with pytest.raises(SystemExit):
        X.run_exec(spec)
    monkeypatch.setattr(X.os, "getppid", lambda: 50)

    def missing(*a):
        raise FileNotFoundError(2, "No such file")

    monkeypatch.setattr(X.os, "execvpe", missing)
    with pytest.raises(SystemExit) as e:
        X.run_exec(spec)
    assert e.value.code == 127


def test_die_with_parent_sets_pdeathsig(monkeypatch):
    seen = []
    monkeypatch.setattr(X.ctypes, "CDLL", lambda *a, **k: types.SimpleNamespace(prctl=lambda *a: seen.append(a)))
    X.die_with_parent()
    assert seen == [(X.PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)]


def test_helper_main_reads_and_deletes_the_spec(monkeypatch, tmp_path, capsys):
    got = []
    monkeypatch.setitem(X.OPS, "kill", got.append)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"op": "kill", "pids": []}))
    assert X.main([str(p)]) == 0 and not p.exists() and got == [{"op": "kill", "pids": []}]
    p.write_text(json.dumps({"op": "rm -rf"}))
    assert X.main([str(p)]) == 2 and "unknown op" in capsys.readouterr().err

    def refuse(spec):
        raise X.Refused("no")

    monkeypatch.setitem(X.OPS, "kill", refuse)
    p.write_text(json.dumps({"op": "kill"}))
    assert X.main([str(p)]) == 3


def test_caller_uid_and_under_tmp(monkeypatch):
    monkeypatch.setenv("SUDO_UID", "1001")
    assert X.caller_uid() == 1001
    monkeypatch.delenv("SUDO_UID")
    assert X.caller_uid() == os.getuid()
    d = trial_dir()
    try:
        assert X.under_tmp(str(d)) == os.path.realpath(d)
        for bad in (tempfile.gettempdir(), "/etc", str(d / "x" / "y")):
            with pytest.raises(X.Refused):
                X.under_tmp(bad)
    finally:
        d.rmdir()


# ---- slots, the pool and the helper's invocation ---------------------------------------------------
def test_requested_and_slots(monkeypatch):
    import grp
    import pwd

    for v, want in (("1", True), ("true", True), ("", False), ("0", False)):
        monkeypatch.setenv(I.FLAG, v)
        assert I.requested() is want
    users = {"ocb-agent-0": 5000, "ocb-agent-1": 5001, "ocb-agent-3": 5003}

    def getpwnam(n):
        if n not in users:
            raise KeyError(n)
        return types.SimpleNamespace(pw_name=n, pw_uid=users[n], pw_gid=users[n])

    monkeypatch.setattr(pwd, "getpwnam", getpwnam)
    monkeypatch.setattr(grp, "getgrnam", lambda n: types.SimpleNamespace(gr_gid=6000))
    slots = I.find_slots()
    assert [s.name for s in slots] == ["ocb-agent-0", "ocb-agent-1"] and slots[1].groups == (5001, 6000)
    assert slots[1].chain == "OCB-AGENT-1"

    def no_group(n):
        raise KeyError(n)

    monkeypatch.setattr(grp, "getgrnam", no_group)
    assert I.find_slots()[0].groups == (5000,)
    p = I.pool()
    assert I.pool() is p and len(p.slots) == 2
    s = p.acquire()
    p.release(s)
    with pytest.raises(RuntimeError):
        I.Pool([]).acquire()


def test_helper_argv_and_run_helper(monkeypatch):
    monkeypatch.setattr(I.os, "geteuid", lambda: 1000)
    path = I.write_spec({"op": "kill"})
    assert json.loads(path.read_text()) == {"op": "kill"}
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700 and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert I.helper_argv(path)[:3] == ["sudo", "-n", sys.executable] and I.helper_argv(path)[-2:] == [
        str(I.HELPER),
        str(path),
    ]
    monkeypatch.setattr(I.os, "geteuid", lambda: 0)
    assert I.helper_argv(path)[0] == sys.executable
    path.unlink()
    seen = []

    def run(argv, **kw):
        seen.append((argv, kw["env"], I.current()))
        return types.SimpleNamespace(returncode=len(seen) - 1, stderr="boom", stdout="")

    monkeypatch.setattr(I.subprocess, "run", run)
    agent = I.Agent(SLOT, Path("/nonexistent"))
    with agent.active():
        I.run_helper({"op": "kill"})  # inside an agent's session: still the harness's
    assert seen[0][2] is None and set(seen[0][1]) == {"PATH"}
    assert not list(I.private_dir().iterdir())  # the spec never outlives the call
    with pytest.raises(RuntimeError, match="boom"):
        I.run_helper({"op": "kill"})


# ---- launching -----------------------------------------------------------------------------------
def test_launch_builds_the_helper_call_with_the_env_in_the_spec(monkeypatch):
    monkeypatch.setattr(I.os, "geteuid", lambda: 1000)
    agent = I.Agent(SLOT, Path("/tmp/pe2-x"))
    argv, kw, spec = agent.launch(["claude", "-p", "hi"], {"env": {"K": "secret"}, "cwd": "/w"})
    s = json.loads(spec.read_text())
    assert argv[:2] == ["sudo", "-n"] and argv[-1] == str(spec) and "secret" not in " ".join(argv)
    assert s["argv"] == ["claude", "-p", "hi"] and s["uid"] == 4242 and s["groups"] == [4243, 4300]
    assert s["env"] == {"K": "secret", "HOME": "/tmp/pe2-x/home", "USER": "ocb-agent-0", "LOGNAME": "ocb-agent-0"}
    assert set(kw["env"]) == {"PATH"} and kw["cwd"] == "/w"
    _, _, spec = agent.launch("echo $X", {"shell": True})
    s = json.loads(spec.read_text())
    assert s["argv"] == ["/bin/sh", "-c", "echo $X"] and s["env"]["PATH"] == os.environ["PATH"]
    _, _, spec = agent.launch(["x", "-v"], {"executable": "/usr/bin/real", "env": {b"B": b"v"}})
    s = json.loads(spec.read_text())
    assert s["argv"] == ["/usr/bin/real", "-v"] and s["env"]["B"] == "v"
    _, _, spec = agent.launch("ls", {"shell": True, "executable": "/bin/bash"})
    assert json.loads(spec.read_text())["argv"] == ["/bin/bash", "-c", "ls"]
    _, _, spec = agent.launch(Path("/bin/true"), {})
    assert json.loads(spec.read_text())["argv"] == ["/bin/true"]
    with pytest.raises(ValueError):
        agent.launch(["x"], {"pass_fds": (5,)})


def test_popen_goes_through_the_agent_only_inside_its_context(monkeypatch):
    agent = I.Agent(SLOT, Path("/nonexistent"))
    launched, signalled = [], []

    def launch(args, kw):
        launched.append(args)
        spec = I.write_spec({})
        return [sys.executable, "-c", "import time; time.sleep(30)"], kw, spec

    monkeypatch.setattr(agent, "launch", launch)
    monkeypatch.setattr(agent, "signal_tree", lambda pid, sig: signalled.append(sig))
    I.install()
    I.install()
    assert subprocess.Popen is I.AgentPopen
    assert subprocess.run(["true"], check=False).returncode == 0 and not launched  # no agent: the harness's
    with agent.active():
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(["claude"], timeout=0.5, check=False)  # the timeout kills the tree
        with I.harness():
            assert I.current() is None
        assert I.current() is agent
        p = subprocess.Popen(["claude"], -1, None)  # positional arguments too
        p.terminate()
        p.wait()
    assert launched == [["claude"], ["claude"]] and signalled == [signal.SIGKILL, signal.SIGTERM]
    p.send_signal(signal.SIGTERM)  # exited: nothing to signal
    assert signalled == [signal.SIGKILL, signal.SIGTERM]

    def failing(args, kw):
        spec = I.write_spec({})
        failing.spec = spec
        return ["/nonexistent/binary"], kw, spec

    monkeypatch.setattr(agent, "launch", failing)
    with agent.active(), pytest.raises(FileNotFoundError):
        subprocess.Popen(["x"])
    assert not failing.spec.exists()


def test_kill_pids_and_signal_tree(monkeypatch, helper_calls):
    monkeypatch.delenv(I.FLAG, raising=False)
    I.kill_pids([1, 2])
    assert helper_calls == []
    monkeypatch.setenv(I.FLAG, "1")
    monkeypatch.setattr(I, "_state", {"pool": I.Pool([SLOT]), "private": None})
    I.kill_pids([])
    I.kill_pids([5, "6"])
    assert helper_calls == [{"op": "kill", "pids": [5, 6], "uids": [4242], "sig": 9}]
    from onchainbench.connect import shell

    agent = I.Agent(SLOT, Path("/nonexistent"))
    monkeypatch.setattr(shell, "_descendants", lambda pid: [pid + 1] if pid else [])
    agent.signal_tree(10, 15)
    agent.signal_tree(0, 15)  # no children: nothing to do
    assert helper_calls[-1] == {"op": "kill", "pids": [11], "uids": [4242], "sig": 15} and len(helper_calls) == 2


def test_shell_kills_what_it_may_not_signal_through_the_helper(monkeypatch):
    from onchainbench.connect import shell

    denied = []
    monkeypatch.setattr(shell, "_descendants", lambda pid: [pid + 1, pid + 2])

    def kill(pid, sig):
        if pid == 101:
            raise PermissionError
        if pid == 102:
            raise ProcessLookupError

    monkeypatch.setattr(shell.os, "kill", kill)
    monkeypatch.setattr(shell.os, "killpg", kill)
    monkeypatch.setattr(I, "kill_pids", denied.extend)
    shell._kill_tree(100)
    assert denied == [101]


# ---- ACLs ----------------------------------------------------------------------------------------
def test_acl_encoding_and_entries():
    acl = I.acl_from_mode(0o750)
    assert acl == [(1, 7, I.ACL_UNDEFINED), (4, 5, I.ACL_UNDEFINED), (0x20, 0, I.ACL_UNDEFINED)]
    raw = I.acl_encode(acl)
    assert raw[:4] == b"\x02\x00\x00\x00" and I.acl_decode(raw) == acl
    with pytest.raises(ValueError):
        I.acl_decode(b"\x01\x00\x00\x00")
    granted = I.acl_set_user(acl, 4242, 6)
    assert (I.ACL_USER, 6, 4242) in granted and (I.ACL_MASK, 7, I.ACL_UNDEFINED) in granted  # mask: r-x | rw-
    assert I.acl_set_user(granted, 4242, None) == acl  # removed again, mask and all
    d = I.default_acl([4242, 1000, 4242])
    assert [e for e in d if e[0] == I.ACL_USER] == [(I.ACL_USER, 7, 1000), (I.ACL_USER, 7, 4242)]
    assert (I.ACL_OTHER, 0, I.ACL_UNDEFINED) in d and (I.ACL_MASK, 7, I.ACL_UNDEFINED) in d


@pytest.fixture
def xattrs(monkeypatch):
    store = {}

    def ino(fd):
        return os.fstat(fd).st_ino

    def getx(fd, name):
        if (ino(fd), name) not in store:
            raise OSError(61, "No data")
        return store[(ino(fd), name)]

    monkeypatch.setattr(I.os, "getxattr", getx, raising=False)
    monkeypatch.setattr(I.os, "setxattr", lambda fd, n, v: store.__setitem__((ino(fd), n), v), raising=False)
    return store


def test_grant_gives_the_slot_what_the_harness_owns_and_skips_links(xattrs, tmp_path):
    root = tmp_path / "trial"
    (root / "ws").mkdir(parents=True)
    (root / "ws" / ".mcp.json").write_text("{}")
    os.chmod(root / "ws" / ".mcp.json", 0o600)
    (root / "ws" / "pearl").write_text("#!/bin/sh")
    os.chmod(root / "ws" / "pearl", 0o755)
    secret = tmp_path / "secret"
    secret.write_text("token")
    (root / "ws" / "link").symlink_to(secret)
    os.mkfifo(root / "ws" / "fifo")
    assert I.grant(root, 4242) == 4  # trial, ws, .mcp.json, pearl
    by_ino = {ino: I.acl_decode(v) for (ino, n), v in xattrs.items() if n == I.ACCESS}
    mcp = by_ino[os.stat(root / "ws" / ".mcp.json").st_ino]
    assert (I.ACL_USER, 6, 4242) in mcp and (I.ACL_GROUP_OBJ, 0, I.ACL_UNDEFINED) in mcp  # no group exposure
    assert (I.ACL_USER, 7, 4242) in by_ino[os.stat(root / "ws" / "pearl").st_ino]
    assert os.stat(secret).st_ino not in by_ino  # the planted link was not followed
    assert (os.stat(root).st_ino, I.DEFAULT) in xattrs
    n = len(xattrs)
    I.grant(root, 4242)  # idempotent: the access ACLs stay as they are
    assert len(xattrs) == n
    I.revoke(root, 4242)
    assert (I.ACL_USER, 7, 4242) not in I.acl_decode(xattrs[(os.stat(root).st_ino, I.ACCESS)])


def test_grant_leaves_other_owners_and_vanishing_files(xattrs, tmp_path, monkeypatch):
    root = tmp_path / "trial"
    root.mkdir()
    (root / "a").write_text("x")
    (root / "b").write_text("x")
    real_stat, real_open = os.stat, os.open

    def fake_stat(p, *a, **k):
        st = real_stat(p, *a, **k)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 4242, *st[5:])) if p == "a" else st

    monkeypatch.setattr(I.os, "stat", fake_stat)
    monkeypatch.setattr(
        I.os, "open", lambda p, *a, **k: (_ for _ in ()).throw(OSError()) if p == "b" else real_open(p, *a, **k)
    )
    assert I.grant(root, 4242) == 1  # only the directory: "a" is the agent's, "b" vanished


# ---- the agent ------------------------------------------------------------------------------------
def fake_trial(root: Path, web=True):
    fork = types.SimpleNamespace(url="http://127.0.0.1:8545", relay=types.SimpleNamespace(url="http://127.0.0.1:8546"))
    fc = types.SimpleNamespace(
        url="http://127.0.0.1:7000/mcp/", s=types.SimpleNamespace(rpc_url="http://127.0.0.1:7000/rpc")
    )
    return types.SimpleNamespace(
        ws=root / "ws",
        fork=fork,
        fc=fc,
        web=types.SimpleNamespace(base="http://127.0.0.1:7100") if web else None,
        data={"proxy_env": {"HTTP_PROXY": "http://127.0.0.1:7200"}},
    )


def test_the_trials_ports_and_never_the_forks(helper_calls, tmp_path):
    agent = I.Agent(SLOT, tmp_path)
    agent.allow_trial(fake_trial(tmp_path))
    assert agent.ports == {7000, 7100, 7200} and agent.forbidden == {8545, 8546}
    assert helper_calls == []  # nothing synced before the first sync
    agent.sync()
    agent.allow("http://127.0.0.1:7300/mcp/")  # a broker started mid-session: synced at once
    agent.allow("http://127.0.0.1:7300/mcp/")
    agent.allow("https://openrouter.ai/api")  # not local: no rule needed
    agent.allow(None)
    assert [c["ports"] for c in helper_calls] == [[7000, 7100, 7200], [7000, 7100, 7200, 7300]]
    assert helper_calls[0] == {"op": "firewall", "uid": 4242, "chain": "OCB-AGENT-0", "ports": [7000, 7100, 7200]}
    with pytest.raises(RuntimeError, match="fork"):
        agent.allow("http://localhost:8545")
    agent2 = I.Agent(SLOT, tmp_path)
    agent2.allow_trial(types.SimpleNamespace(fc=types.SimpleNamespace(), web=None, data={}))
    assert agent2.ports == set() and agent2.forbidden == set()
    assert I.url_port("https://x") == 443 and I.url_port("http://x") == 80 and I.url_port("") is None
    assert I.is_local("http://[::1]:5/") and I.is_local("http://127.0.0.2:5/") and not I.is_local(None)


def test_finish_runs_every_step_and_raises_the_first_failure(helper_calls, monkeypatch, tmp_path):
    revoked = []
    monkeypatch.setattr(I, "revoke", lambda root, uid: revoked.append(uid))
    agent = I.Agent(SLOT, tmp_path)
    agent.finish()
    assert [c["op"] for c in helper_calls] == ["killall", "reclaim", "firewall"] and revoked == [4242]
    assert helper_calls[1]["roots"] == [str(tmp_path)] and helper_calls[2]["ports"] == []

    def boom(root, uid):
        raise OSError("acl")

    monkeypatch.setattr(I, "revoke", boom)
    with pytest.raises(OSError, match="acl"):
        agent.finish()
    assert [c["op"] for c in helper_calls[3:]] == ["killall", "reclaim", "firewall"]


class Inner:
    def __init__(self):
        self.stats, self.oracle, self.log = "stats", "oracle", []

    def send(self, text):
        self.log.append(("send", I.current()))
        return "answer"

    def crash_and_resume(self):
        self.log.append(("crash", I.current()))

    def total_calls(self):
        return 3

    def transcript(self):
        return ([], [])

    def close(self):
        self.log.append(("close", I.current()))
        return "closed"


def test_isolated_session_runs_the_arm_inside_its_agent(helper_calls, monkeypatch, xattrs):
    root = trial_dir()
    (root / "ws").mkdir()
    monkeypatch.setattr(I, "_state", {"pool": I.Pool([SLOT]), "private": None})
    monkeypatch.setattr(I, "revoke", lambda root, uid: None)
    t = fake_trial(root)
    made = []
    inner = Inner()

    def make(sdir):
        made.append(I.current())
        assert sdir is None  # no sessions dir asked for: none given
        (root / "ws" / "shim.json").write_text("{}")  # what the session writes is granted before any command runs
        return inner

    s = I.isolated(t, make)
    agent = t.data["agent_user"]
    assert made == [agent] and subprocess.Popen is I.AgentPopen
    assert (root / "home" / ".gitconfig").read_text() == I.GITCONFIG
    assert (os.stat(root / "ws" / "shim.json").st_ino, I.ACCESS) in xattrs
    assert s.send("hi") == "answer" and s.stats == "stats" and s.oracle == "oracle"
    s.crash_and_resume()
    assert s.transcript() == ([], [])
    with pytest.raises(AttributeError):
        _ = s.missing
    del Inner.transcript  # a session without a transcript (the scripted arms'): the runner's lookup gets its default
    try:
        assert getattr(s, "transcript", lambda: None)() is None
    finally:
        Inner.transcript = lambda self: ([], [])
    assert s.close() == "closed"
    assert inner.log == [("send", agent), ("crash", agent), ("close", agent)]
    ops = [c["op"] for c in helper_calls]
    assert ops == ["firewall", "handover", "killall", "reclaim", "firewall"]  # the home handed over once
    assert helper_calls[1]["dirs"] == [os.path.realpath(root / "home")]  # not the workspace
    assert I.pool().acquire(timeout=1) is SLOT  # released
    I.pool().release(SLOT)
    with pytest.raises(AttributeError):
        I.IsolatedSession.__getattr__(s, "inner")

    def broken(sdir):
        raise RuntimeError("arm failed")

    with pytest.raises(RuntimeError, match="arm failed"):
        I.isolated(t, broken)
    assert I.pool().acquire(timeout=1) is SLOT  # released on failure too
    import shutil

    shutil.rmtree(root)


def test_any_arms_sessions_dir_moves_into_the_trial_and_back(helper_calls, monkeypatch, xattrs):
    """smoke-matrix-2: an arm's sandbox denied its sessions dir (results/<run>/sessions) by mounting over it, which
    the slot user couldn't reach (results/ is closed to agents): every command failed. The rule for every arm: a
    sessions dir it is given is a harness-only one in its trial, moved to the run's at close."""
    root = trial_dir()
    (root / "ws").mkdir()
    run_sessions = Path(tempfile.mkdtemp()) / "results" / "r" / "sessions"
    monkeypatch.setattr(I, "_state", {"pool": I.Pool([SLOT]), "private": None})
    monkeypatch.setattr(I, "revoke", lambda root, uid: None)
    removed = []
    monkeypatch.setattr(I.os, "removexattr", lambda fd, n: removed.append(n), raising=False)
    given = []

    class Arm(Inner):
        def close(self):
            (given[0] / "abc.jsonl").write_text("{}")  # the arm keeps its session file where it was told
            return types.SimpleNamespace(extra={"session": str(given[0] / "abc.jsonl")})

    def make(sdir):
        given.append(sdir)
        return Arm()

    s = I.isolated(fake_trial(root), make, run_sessions)
    own = given[0]
    assert own == root / "sessions" and stat.S_IMODE(os.stat(own).st_mode) == 0o700
    assert removed == [I.ACCESS, I.DEFAULT]  # nothing inherited from the trial dir's default ACL
    assert not [k for k in xattrs if k[0] == os.stat(own).st_ino]  # never granted to the slot
    handed = next(c for c in helper_calls if c["op"] == "handover")["dirs"]
    assert os.path.realpath(own) not in handed  # nor handed over
    st = s.close()
    assert (run_sessions / "abc.jsonl").exists() and not list(own.iterdir())
    assert st.extra["session"] == str(run_sessions / "abc.jsonl")  # where the run (and inspect) finds it
    import shutil

    shutil.rmtree(root)
    shutil.rmtree(run_sessions.parents[2])


# ---- sticky folders for OpenClaw's temp-root check ------------------------------------------------------------------
def test_make_sticky_keeps_the_mode_and_skips_links_and_other_owners(tmp_path, monkeypatch):
    top = tmp_path / "top"
    (top / "a" / "b").mkdir(parents=True)
    os.chmod(top / "a", 0o770)  # how a slot's ACL mask shows in the group bits
    outside = tmp_path / "outside"
    outside.mkdir()
    (top / "link").symlink_to(outside)
    assert I.make_sticky(top) == 1  # top alone
    assert stat.S_IMODE(os.stat(top / "a").st_mode) == 0o770
    assert I.make_sticky(top, tree=True) == 2  # a and b; top already is
    assert stat.S_IMODE(os.stat(top / "a").st_mode) == 0o1770  # sticky, the rest of the mode as it was
    assert not os.stat(outside).st_mode & stat.S_ISVTX  # the link was not followed
    assert I.make_sticky(top, tree=True) == 0  # idempotent
    (top / "c").mkdir()
    monkeypatch.setattr(I.os, "geteuid", lambda: os.getuid() + 1)  # not the harness's: left alone
    assert I.make_sticky(top / "c") == 0
    monkeypatch.delattr(I.os, "O_DIRECTORY")  # Windows: no modes
    assert I.make_sticky(top / "c", tree=True) == 0
    assert not os.stat(top / "c").st_mode & stat.S_ISVTX


def test_sticky_trial_is_a_no_op_outside_an_agents_session():
    assert I.sticky_trial() is False


def _arm_in_trial(kind, monkeypatch):
    """An arm's session opened under isolation (the root helper mocked, so nothing changes owner): its trial dir."""
    from onchainbench import arms as A

    root = trial_dir()
    (root / "ws").mkdir()
    monkeypatch.setattr(I, "_state", {"pool": I.Pool([SLOT]), "private": None})
    monkeypatch.setattr(I, "revoke", lambda root, uid: None)
    monkeypatch.delenv("OPENROUTER_PROVISIONING_KEY", raising=False)
    t = fake_trial(root)
    mcp = {"pearl-connect": {"url": "http://127.0.0.1:7000/mcp/", "headers": {"Authorization": "Bearer tok"}}}
    t.fc.mcp_json = lambda: {"mcpServers": mcp}
    make = {
        "openclaw": lambda sdir: A.OpenClawSession(t, "openclaw", "openrouter/m", 30),
        "codex": lambda sdir: A.CodexSession(t, "openrouter/m"),
        "hermes": lambda sdir: A.HermesSession(t, "hermes", "openrouter/m", 30),
    }[kind]
    s = I.isolated(t, make)
    return root, s


def _dir_modes(root: Path) -> dict[str, int]:
    return {os.path.relpath(d, root): stat.S_IMODE(os.lstat(d).st_mode) for d, _, _ in os.walk(root)}


@pytest.mark.parametrize("kind", ["openclaw", "codex", "hermes"])
def test_only_openclaws_trial_folders_are_sticky(kind, helper_calls, monkeypatch, xattrs):
    """smoke-011: OpenClaw 2026.11.5 refused to start in 6 of 7 trials ("No safe OpenClaw temp root is available"):
    under isolation the trial dir and its temp dir show the slot's ACL mask as group-writable, and its check wants
    such an ancestor sticky. Its trial dir and the folders handed to the slot are sticky; every other arm's layout is
    as it was (no sticky bit anywhere, the trial dir mkdtemp's 0700)."""
    root, s = _arm_in_trial(kind, monkeypatch)
    try:
        modes = _dir_modes(root)
        handed = next(c for c in helper_calls if c["op"] == "handover")["dirs"]
        if kind == "openclaw":
            sticky = {".", "home", "tmp", "openclaw-state", "openclaw-state/state", "openclaw-state/home"}
            assert sticky <= {d for d, m in modes.items() if m & stat.S_ISVTX}
            assert not modes["ws"] & stat.S_ISVTX  # the workspace is the harness's, unchanged
            assert os.path.realpath(root / "tmp") in handed  # its TMPDIR exists, sticky, before its first turn
        else:
            assert not [d for d, m in modes.items() if m & stat.S_ISVTX]
            assert modes["."] == 0o700
        assert s.inner.stats is not None
    finally:
        s.close()
        import shutil

        shutil.rmtree(root)


OPENCLAW_DIST = Path(__file__).resolve().parents[1] / "vendor" / "node_modules" / "openclaw" / "dist"
# OpenClaw's own temp-root choice (its tmp-openclaw-dir module), with its shared /tmp/openclaw out of reach, as it is
# for an agent user when another uid made it: the trial's TMPDIR is what it falls back to
TMP_ROOT_JS = """
import { readdirSync } from "node:fs";
import path from "node:path";
const [dist, unusable] = process.argv.slice(1);
const mod = readdirSync(dist).find((f) => f.startsWith("tmp-openclaw-dir-") && f.endsWith(".mjs"));
const m = await import(path.join(dist, mod));
try { console.log("ROOT " + m.n({ preferredDir: unusable })); } catch (e) { console.log("REFUSED " + e.message); }
"""


@pytest.mark.skipif(
    not (OPENCLAW_DIST.is_dir() and __import__("shutil").which("node")),
    reason="needs the vendored OpenClaw (scripts/install-arms.sh openclaw) and node",
)
@pytest.mark.parametrize("kind", ["openclaw", "codex"])
def test_openclaws_own_temp_root_check_accepts_its_trial_layout(kind, helper_calls, monkeypatch, xattrs):
    """OpenClaw's own check on the layout a session leaves, with every directory group-writable as the slot's ACL
    mask makes it under isolation: OpenClaw's layout passes; the same mask on a layout without the sticky bit (any
    other arm's) is the refusal smoke-011 saw."""
    root, s = _arm_in_trial(kind, monkeypatch)
    try:
        for d in _dir_modes(root):
            p = root / d
            os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) | 0o070)
        (root / "file").write_text("")  # no directory can be made under a file, even by root
        r = subprocess.run(
            ["node", "--input-type=module", "-e", TMP_ROOT_JS, str(OPENCLAW_DIST), str(root / "file" / "openclaw")],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": os.environ["PATH"], "TMPDIR": str(root / "tmp"), "HOME": str(root / "home")},
            check=False,
        )
        if kind == "openclaw":
            assert r.stdout.startswith(f"ROOT {root / 'tmp'}{os.sep}"), r.stdout + r.stderr
        else:
            assert r.stdout.startswith("REFUSED No safe OpenClaw temp root is available"), r.stdout + r.stderr
    finally:
        s.close()
        import shutil

        shutil.rmtree(root)


def test_hand_over_files_stats_without_a_session_path(tmp_path):
    own, dest = tmp_path / "own", tmp_path / "dest"
    own.mkdir()
    (own / "x").write_text("1")
    assert I.hand_over_sessions(own, dest, None) == ["x"]
    stats = types.SimpleNamespace(extra={"session": "/elsewhere/s.jsonl"})
    I.hand_over_sessions(own, dest, stats)
    assert stats.extra["session"] == "/elsewhere/s.jsonl"  # not one of the trial's: untouched


def test_the_arms_folders_are_handed_over_but_not_the_workspace(helper_calls, tmp_path):
    root = tmp_path / "trial"
    for d in ("ws", "home", "hermes-home", "tmp", ".shared-tmp-1", "sessions"):
        (root / d).mkdir(parents=True)
    (root / "file").write_text("x")
    (root / "link").symlink_to(root / "home")
    agent = I.Agent(SLOT, root, keep=[root / "ws"], private=[root / "sessions"])
    assert agent.hand_over() == ["hermes-home", "home", "tmp"]
    assert helper_calls == [
        {"op": "handover", "uid": 4242, "dirs": [os.path.realpath(root / d) for d in ("hermes-home", "home", "tmp")]}
    ]
    (root / "codex-home").mkdir()  # a folder made later (a turn's): handed over then, the others not again
    assert agent.hand_over() == ["codex-home"] and agent.hand_over() == [] and len(helper_calls) == 2


def test_the_helper_hands_over_only_folders_of_a_trial_dir(agent_users, monkeypatch):
    root = trial_dir()
    (root / "hermes-home" / "sub").mkdir(parents=True)
    (root / "hermes-home" / "config.yaml").write_text("x")
    chowns = []
    monkeypatch.setattr(X, "slot_user", lambda uid: ("ocb-agent-0", 4243, set()))
    monkeypatch.setattr(X.os, "fchown", lambda fd, u, g: chowns.append(("top", u, g)))
    monkeypatch.setattr(X.os, "chown", lambda p, u, g, **k: chowns.append((p, u, k.get("follow_symlinks"))))
    try:
        X.handover({"uid": 4242, "dirs": [str(root / "hermes-home")]})
        assert chowns[0] == ("top", 4242, 4243) and {c[0] for c in chowns[1:]} == {"sub", "config.yaml"}
        assert all(c[2] is False for c in chowns[1:])  # never through a link
        for bad in (str(root), "/etc", str(root / "hermes-home" / "sub")):  # a trial dir's own folders only
            with pytest.raises(X.Refused):
                X.handover({"uid": 4242, "dirs": [bad]})
    finally:
        monkeypatch.undo()
        import shutil

        shutil.rmtree(root)


def test_grant_leaves_the_harnesses_private_dirs_alone(xattrs, tmp_path):
    root = tmp_path / "trial"
    (root / "sessions").mkdir(parents=True)
    (root / "sessions" / "s.jsonl").write_text("{}")
    (root / "home").mkdir()
    I.grant(root, 4242, skip=[root / "sessions"])
    inos = {k[0] for k in xattrs}
    assert os.stat(root / "home").st_ino in inos
    assert not {os.stat(root / "sessions").st_ino, os.stat(root / "sessions" / "s.jsonl").st_ino} & inos
    acl = I.acl_decode(xattrs[(os.stat(root / "home").st_ino, I.ACCESS)])
    assert (I.ACL_USER, 7, os.geteuid()) in acl  # the harness keeps its access once the folder is the slot's


def test_a_file_written_for_the_arm_mid_turn_is_readable_by_the_slot(xattrs, tmp_path, monkeypatch):
    """smoke-matrix-2: an arm that opens its Connect session inside its turn wrote its stand-in .mcp.json (0600,
    the harness's) after the turn's grant, and the chmod had emptied the slot's mask: every `pearl` call from
    its directory failed. The rule for every arm: what the harness writes for it is shared as it is written."""
    from onchainbench.connect.workspace import write_file

    root = trial_dir()
    agent = I.Agent(SLOT, root)
    shim = root / "agent" / ".mcp.json"
    shim.parent.mkdir()
    write_file(shim, "{}", 0o600)  # no agent's session: nothing to share
    assert (os.stat(shim).st_ino, I.ACCESS) not in xattrs
    with agent.active():
        write_file(shim, "{}", 0o600)
    acl = I.acl_decode(xattrs[(os.stat(shim).st_ino, I.ACCESS)])
    assert (I.ACL_USER, 6, 4242) in acl and (I.ACL_MASK, 6, I.ACL_UNDEFINED) in acl
    assert stat.S_IMODE(os.stat(shim).st_mode) == 0o600  # still nobody else's
    with agent.active():
        fd = os.open(root, os.O_RDONLY)  # not a regular file
        assert I.share_fd(fd) is False
        os.close(fd)
        other = Path(tempfile.mkdtemp()) / "f"  # outside the trial (Linux tells by /proc/self/fd)
        other.write_text("x")
        fd = os.open(other, os.O_RDONLY)
        assert I.share_fd(fd) is (not os.path.isdir("/proc/self/fd"))
        os.close(fd)
        real = os.fstat
        monkeypatch.setattr(I.os, "fstat", lambda f: os.stat_result((*real(f)[:4], 1, *real(f)[5:])))
        fd = os.open(shim, os.O_RDONLY)
        assert I.share_fd(fd) is False  # not the harness's
        os.close(fd)
    import shutil

    shutil.rmtree(root)


def test_agent_processes_get_an_allowlisted_environment(monkeypatch):
    monkeypatch.setenv("ARM", "codex:openrouter/x")  # CI's: names the model
    monkeypatch.setenv("ACTIONS_ORCHESTRATION_ID", "abc")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-main")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setattr(I.os, "geteuid", lambda: 1000)
    agent = I.Agent(SLOT, Path("/tmp/pe2-x"))
    env = {**os.environ, "CODEX_HOME": "/tmp/pe2-x/codex-home", "TMPDIR": "/tmp/pe2-x/tmp"}
    _, _, spec = agent.launch(["codex"], {"env": env})
    got = json.loads(spec.read_text())["env"]
    assert "ARM" not in got and "ACTIONS_ORCHESTRATION_ID" not in got and "OPENROUTER_API_KEY" not in got
    assert got["PATH"] == os.environ["PATH"] and got["LC_ALL"] == "C.UTF-8" and got["CODEX_HOME"].endswith("codex-home")
    with agent.active():
        from onchainbench import arms as A

        A._env_with_key({"OPENROUTER_API_KEY": "sk-main"})  # the arm passes the model key on purpose
    _, _, spec = agent.launch(["codex"], {"env": env})
    assert json.loads(spec.read_text())["env"]["OPENROUTER_API_KEY"] == "sk-main"
    I.declare(["X"])  # outside an agent's session: nothing
    assert "X" not in agent.declared
    assert I.env_allowed("HTTPS_PROXY") and I.env_allowed("NODE_USE_ENV_PROXY") and not I.env_allowed("GITHUB_TOKEN")


def test_brokers_join_the_allowlist(helper_calls, tmp_path):
    agent = I.Agent(SLOT, tmp_path)
    t = types.SimpleNamespace(data={"agent_user": agent})
    I.allow_broker(t, types.SimpleNamespace(url="http://127.0.0.1:7400/mcp/"))
    I.allow_broker(types.SimpleNamespace(data={}), types.SimpleNamespace(url="http://127.0.0.1:7500/mcp/"))
    I.allow_broker(types.SimpleNamespace(), None)
    assert agent.ports == {7400}


def test_open_session_isolates_when_requested(monkeypatch):
    from onchainbench import arms as A

    seen, dirs = [], []
    monkeypatch.setattr(A, "_open_session", lambda *a: dirs.append(a[-1]) or "session")

    def isolated(t, make, sessions_dir):
        seen.append((t, sessions_dir))
        return "isolated", make(Path("/trial/sessions"))

    monkeypatch.setattr(I, "isolated", isolated)
    monkeypatch.delenv(I.FLAG, raising=False)
    assert A.open_session("null", "t", [], 5, sessions_dir=Path("/run/sessions")) == "session" and seen == []
    monkeypatch.setenv(I.FLAG, "1")
    assert A.open_session("null", "t", [], 5, sessions_dir=Path("/run/sessions")) == ("isolated", "session")
    assert seen == [("t", Path("/run/sessions"))]
    assert dirs == [Path("/run/sessions"), Path("/trial/sessions")]  # the arm gets the trial's under isolation


def test_log_broker_refusals_opens_the_brokers_port(tmp_path, helper_calls):
    from onchainbench import arms as A

    agent = I.Agent(SLOT, tmp_path)
    t = types.SimpleNamespace(data={"agent_user": agent}, fc=None)
    broker = types.SimpleNamespace(url="http://127.0.0.1:7600/mcp/", token="tok")
    A.log_broker_refusals(t, broker)
    assert agent.ports == {7600} and t.data["secrets"] == ["tok"]


def test_stop_home_processes_tolerates_an_agents_processes(monkeypatch, tmp_path):
    from onchainbench import arms as A

    class Ps:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def communicate(self):
            return (f"99999 hermes HERMES_HOME={tmp_path} x\n", None)

    monkeypatch.setattr(A.subprocess, "Popen", Ps)

    def kill(pid, sig):
        raise PermissionError

    monkeypatch.setattr(A.os, "kill", kill)
    assert A.stop_home_processes(tmp_path) == [99999]


# ---- the live-web proxy -----------------------------------------------------------------------------
def test_the_proxy_refuses_local_targets_under_isolation(monkeypatch):
    from onchainbench import live_web as L

    monkeypatch.setenv(I.FLAG, "1")
    p = L.DomainLog()
    try:
        assert p.refuse_local
        c = socket.create_connection(("127.0.0.1", int(p.url.rsplit(":", 1)[1])), timeout=5)
        c.sendall(b"CONNECT 127.0.0.1:8545 HTTP/1.1\r\nHost: x\r\n\r\n")
        assert c.recv(100).startswith(b"HTTP/1.1 403")
        c.close()
        assert p.refused == {"127.0.0.1": 1}
    finally:
        p.stop()
    monkeypatch.delenv(I.FLAG)
    q = L.DomainLog()
    assert not q.refuse_local
    q.stop()
    assert L.is_local_ip("127.0.0.5") and L.is_local_ip("::1") and L.is_local_ip("::ffff:127.0.0.1")
    assert L.is_local_ip("0.0.0.0") and L.is_local_ip("fe80::1%eth0") and not L.is_local_ip("1.1.1.1")
    monkeypatch.setattr(L.socket, "getaddrinfo", lambda h, p, **k: [(0, 0, 0, "", ("93.184.216.34", p))])
    assert L.public_address("example.com", 443) == ("93.184.216.34", 443)
    monkeypatch.setattr(L.socket, "getaddrinfo", lambda h, p, **k: [(0, 0, 0, "", ("127.0.0.1", p))])
    assert L.public_address("localtest.me", 80) is None  # a public name for a local address
    monkeypatch.setattr(L.socket, "getaddrinfo", lambda h, p, **k: [])
    assert L.public_address("x", 80) is None

    def fail(*a, **k):
        raise OSError

    monkeypatch.setattr(L.socket, "getaddrinfo", fail)
    assert L.public_address("nx.invalid", 80) is None


def test_the_proxy_relays_public_targets_to_the_checked_address(monkeypatch):
    from onchainbench import live_web as L

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    p = L.DomainLog(refuse_local=True)
    monkeypatch.setattr(L, "public_address", lambda h, pt: ("127.0.0.1", port))  # as if it were public
    try:
        c = socket.create_connection(("127.0.0.1", int(p.url.rsplit(":", 1)[1])), timeout=5)
        c.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: x\r\n\r\n")
        assert c.recv(100).startswith(b"HTTP/1.1 200")
        c.close()
    finally:
        p.stop()
        srv.close()


# ---- the preflight --------------------------------------------------------------------------------
def test_problems_off_linux_and_missing_pieces(monkeypatch):
    monkeypatch.setattr(I.platform, "system", lambda: "Darwin")
    assert "needs Linux" in I.problems()[0]
    monkeypatch.setattr(I.platform, "system", lambda: "Linux")
    monkeypatch.setattr(I, "_state", {"pool": I.Pool([]), "private": None})
    monkeypatch.setattr(I.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(I.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1))
    monkeypatch.setattr(I.shutil, "which", lambda n, path=None: None)
    found = I.problems(2)
    assert any("0 agent users for 2 workers" in p for p in found) and any("sudo" in p for p in found)
    assert any("ip6tables not found" in p for p in found)
    monkeypatch.setattr(I.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(I.shutil, "which", lambda n, path=None: "/usr/sbin/" + n)
    assert I.problems(0) == []  # nothing missing, but no slots: nothing to probe


def test_problems_probes_the_slots(monkeypatch):
    s1 = I.Slot(1, "ocb-agent-1", 4250, 4251, (4251,))
    monkeypatch.setattr(I.platform, "system", lambda: "Linux")
    monkeypatch.setattr(I, "_state", {"pool": I.Pool([SLOT, s1]), "private": None})
    monkeypatch.setattr(I.os, "geteuid", lambda: 0)
    monkeypatch.setattr(I.shutil, "which", lambda n, path=None: "/usr/bin/" + n)
    monkeypatch.setattr(I, "probe", lambda s: [f"probe {s.name}"])
    monkeypatch.setattr(I, "_baseline", lambda s: [f"baseline {s.name}"])
    monkeypatch.setattr(I.Agent, "prepare", lambda self: None)
    monkeypatch.setattr(I.Agent, "finish", lambda self: None)
    monkeypatch.setattr(I, "run_as", lambda agent, argv: "denied")
    assert I.problems(2) == ["probe ocb-agent-0", "baseline ocb-agent-1", "bubblewrap fails as ocb-agent-0: denied"]
    monkeypatch.setattr(I, "run_as", lambda agent, argv: "ok\n")
    assert I.problems(2)[-1] == "baseline ocb-agent-1"


def test_baseline_and_probe_report_failures(monkeypatch, xattrs):
    def fail(spec, timeout=60):
        raise RuntimeError("no iptables")

    monkeypatch.setattr(I, "run_helper", fail)
    assert I._baseline(SLOT) == ["ocb-agent-0: no iptables"]
    assert "RuntimeError: no iptables" in I.probe(SLOT)[0]
    monkeypatch.setattr(I, "run_helper", lambda spec, timeout=60: None)
    monkeypatch.setattr(I.Agent, "prepare", lambda self: None)
    # the second port should have been refused, the own listener opened, the harness's processes and sudo refused,
    # and the shim read back
    answers = iter(["open\n", "open\n", "refused\n", "signalled\n", "sudo\n", "[rc 1] cat: Permission denied"])
    monkeypatch.setattr(I, "run_as", lambda agent, argv: next(answers))
    found = I.probe(SLOT)
    assert len(found) == 5 and "expected refused" in found[0] and "can't read a shim" in found[4]
    assert found[1] == "ocb-agent-0 on its own listener: refused, expected open"
    assert found[2] == "ocb-agent-0 on signalling the harness: signalled, expected refused"
    assert found[3] == "ocb-agent-0 on sudo: sudo, expected refused"
    assert I._baseline(SLOT) == []
    answers = iter(['{"probe": 1}\n'])
    a = I.Agent(SLOT, Path(tempfile.mkdtemp()))
    assert I.shim_readable(a) == []  # written with the slot's read, inside its context


def test_run_as_reports_failures(monkeypatch):
    agent = I.Agent(SLOT, Path(tempfile.gettempdir()))
    monkeypatch.setattr(agent, "launch", lambda args, kw: (args, kw, I.write_spec({})))
    assert I.run_as(agent, ["sh", "-c", "echo hi"]) == "hi\n"
    assert I.run_as(agent, ["sh", "-c", "echo err >&2; exit 3"]).startswith("[rc 3] err")


def test_read_text_reads_an_arms_file_as_its_agent(monkeypatch, tmp_path):
    """smoke-3arms-2: OpenClaw's export, written 0600 by the slot in a folder it owns, was unreadable to the harness
    (every transcript empty). Read as the agent, through the same launcher as the arm's processes."""
    f = tmp_path / "events.jsonl"
    f.write_text("line\n")
    assert I.read_text(f) == "line\n"  # no agent: directly
    agent = I.Agent(SLOT, tmp_path)
    launched = []
    monkeypatch.setattr(agent, "launch", lambda args, kw: launched.append(list(args)) or (args, kw, I.write_spec({})))
    I.install()
    with agent.active():
        assert I.read_text(f) == "line\n"
        with pytest.raises(OSError, match="as the agent: cat"):
            I.read_text(tmp_path / "missing")
    assert launched[0] == ["cat", "--", str(f)]


def test_main_runs_the_preflight_then_the_commands(monkeypatch, capsys):
    monkeypatch.setattr(I, "_state", {"pool": I.Pool([SLOT]), "private": None})
    monkeypatch.setattr(I, "problems", lambda workers: [])
    monkeypatch.setattr(I.Agent, "prepare", lambda self: None)
    monkeypatch.setattr(I.Agent, "finish", lambda self: None)
    monkeypatch.setattr(I, "run_as", lambda agent, argv: "[rc 1] nope" if argv[0] == "bad" else "v1\n")
    assert I.main(["--run", "claude --version"]) == 0 and "ready, 1 agent users" in capsys.readouterr().out
    assert I.main(["--run", "bad"]) == 1 and "bad failed" in capsys.readouterr().out
    monkeypatch.setattr(I, "problems", lambda workers: [f"{workers} short"])
    assert I.main(["--workers", "4"]) == 1 and "4 short" in capsys.readouterr().out


def test_run_refuses_an_isolated_run_that_cant_be(monkeypatch, capsys):
    from onchainbench import run

    monkeypatch.setenv(I.FLAG, "1")
    monkeypatch.setattr(I, "problems", lambda workers: ["no agent users"])
    assert run.main(["--arms", "null", "--tasks", "a-wrap-unwrap", "--workers", "1"]) == 2
    assert "refusing an isolated run: no agent users" in capsys.readouterr().out


# ---- the real thing: root on Linux, with the agent users ---------------------------------------------
def _have_agent_users() -> bool:
    try:
        return bool(I.find_slots())
    except Exception:  # noqa: BLE001 - no pwd (Windows)
        return False


_privileged = os.name == "posix" and (os.geteuid() == 0 or bool(os.environ.get("ONCHAINBENCH_TEST_SUDO")))
real = pytest.mark.skipif(  # ONCHAINBENCH_TEST_SUDO=1: as a user with passwordless sudo, the way the run works
    platform.system() != "Linux" or not _privileged or not _have_agent_users(),
    reason="needs root on Linux (or ONCHAINBENCH_TEST_SUDO=1) and the agent users (sudo scripts/agent-users.sh 2)",
)


@real
def test_real_the_agent_is_refused_on_anvils_port_and_let_through_to_connects():
    assert I.probe(I.find_slots()[0]) == []


@real
def test_real_two_slots_cant_reach_each_others_ports_or_files():
    s0, s1 = I.find_slots()[:2]
    connect0, connect1 = I._listener(), I._listener()
    r0, r1 = trial_dir(), trial_dir()
    a0 = I.Agent(s0, r0, ports={connect0.getsockname()[1]})
    a1 = I.Agent(s1, r1, ports={connect1.getsockname()[1]})
    (r1 / "secret").write_text("trial 1's")
    try:
        for a in (a0, a1):
            a.prepare()
            a.sync()
        p0, p1 = connect0.getsockname()[1], connect1.getsockname()[1]
        assert I.run_as(a0, ["bash", "-c", I.PROBE.format(port=p0)]).strip() == "open"
        assert I.run_as(a0, ["bash", "-c", I.PROBE.format(port=p1)]).strip() == "refused"  # the other trial's
        assert "[rc " in I.run_as(a0, ["cat", str(r1 / "secret")])
        out = I.run_as(
            a0, ["bash", "-c", f"id -un; echo $HOME; touch {r0}/made; setsid sleep 300 >/dev/null 2>&1 & echo started"]
        )
        assert out.split() == [s0.name, str(r0 / "home"), "started"]
        assert (r0 / "made").stat().st_uid == s0.uid
        I.run_as(a0, ["bash", "-c", f"echo helper > /tmp/ocb-test-helper-{s0.uid}"])  # a literal /tmp path
    finally:
        for a in (a0, a1):
            a.finish()
        connect0.close()
        connect1.close()
    assert (r0 / "made").stat().st_uid == os.geteuid()  # handed back
    assert X.live_pids(s0.uid) == []  # the detached sleep went with the session
    kept = list(r0.glob(f".shared-tmp-*/ocb-test-helper-{s0.uid}"))  # moved into its own trial, out of /tmp
    assert len(kept) == 1 and kept[0].stat().st_uid == os.geteuid()
    assert not Path(f"/tmp/ocb-test-helper-{s0.uid}").exists()
    import shutil

    shutil.rmtree(r0)
    shutil.rmtree(r1)


@real
def test_real_a_slot_reaches_its_own_listener_and_no_one_elses():
    """smoke-3arms-2: Hermes's CLI couldn't reach the gateway it had started on an ephemeral loopback port. A slot's
    own listener, on any port, takes its connections; the harness's ports and another slot's listener still refuse."""
    s0, s1 = I.find_slots()[:2]
    harness = I._listener()
    r0, r1 = trial_dir(), trial_dir()
    a0, a1 = I.Agent(s0, r0), I.Agent(s1, r1)
    serve = 'python3 -c \'import socket, time; s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(4); '
    serve += f'open("{r0}/port", "w").write(str(s.getsockname()[1])); time.sleep(60)\''
    try:
        for a in (a0, a1):
            a.prepare()
            a.sync()
        assert I.run_as(a0, ["bash", "-c", I.OWN_PROBE]).strip() == "open"
        I.run_as(a0, ["bash", "-c", f"setsid {serve} >/dev/null 2>&1 & sleep 2"])
        port = int((r0 / "port").read_text())
        assert I.run_as(a0, ["bash", "-c", I.PROBE.format(port=port)]).strip() == "open"  # a helper's server
        assert I.run_as(a1, ["bash", "-c", I.PROBE.format(port=port)]).strip() == "refused"  # another slot's
        hp = harness.getsockname()[1]
        assert I.run_as(a0, ["bash", "-c", I.PROBE.format(port=hp)]).strip() == "refused"  # the harness's
        harness.settimeout(0.5)
        with pytest.raises(OSError):  # and the harness never got the connection
            harness.accept()
    finally:
        for a in (a0, a1):
            a.finish()
        harness.close()


@real
def test_real_the_cheaters_direct_anvil_calls_are_refused_not_just_caught(connect_ws, mock):
    """The cheater's calls (cheat_steps, direct_cheats) are curl commands its scripted shell runs: a subprocess, so
    under isolation they run as the trial's agent user. Its Anvil calls must not arrive at all; its Connect calls must.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from onchainbench import arms as A
    from onchainbench.scripted import Script, ScriptedSession

    hits = []

    class Anvil(BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    anvil = ThreadingHTTPServer(("127.0.0.1", 0), Anvil)
    import threading

    threading.Thread(target=anvil.serve_forever, daemon=True).start()
    anvil_url = f"http://127.0.0.1:{anvil.server_address[1]}"
    root = trial_dir()
    ws = root / "ws"
    import shutil

    shutil.copytree(connect_ws, ws)
    os.chmod(root, 0o700)
    t = types.SimpleNamespace(
        ws=ws, fork=types.SimpleNamespace(url=anvil_url), fc=mock, web=None, data={}, safe="0x" + "5a" * 20
    )
    cheat = A._rpc_curl("anvil_setBalance", [t.safe, "0x1"], anvil_url)
    honest = A._rpc_curl("eth_chainId", [], mock.s.rpc_url)
    steps = [f"{cheat}; echo rc=$?", f"{honest} >/dev/null; echo rc=$?", "whoami", "done"]
    outs = []
    s = I.isolated(t, lambda _sessions_dir=None: ScriptedSession(ws, Script(steps), 10, A.EVAL_CAPS))  # make(sdir)
    try:
        real_run = s.inner.cs.shell.run
        s.inner.cs.shell.run = lambda c: outs.append(real_run(c)) or outs[-1]
        assert s.send("go") == "done"
    finally:
        s.close()
        anvil.shutdown()
        shutil.rmtree(root)
    assert hits == []  # refused at the firewall: nothing reached "Anvil", so there is nothing to catch
    assert outs[0]["output"].strip().endswith("rc=7")  # curl: couldn't connect
    assert outs[1]["output"].strip().endswith("rc=0")  # Connect's port is the trial's own
    assert outs[2]["output"].strip() == I.find_slots()[0].name


@real
def test_real_a_sandbox_denying_a_closed_results_path_fails_and_the_trials_own_works():
    """smoke-matrix-2 (olas-ocb-agent, but any arm whose sandbox denies the sessions dir it is given): bwrap, as the
    slot user, can't mount over a path under a results dir closed to agents. A harness-only dir in the trial works."""
    import shutil

    from onchainbench.connect.sandbox import wrapper_for

    if not shutil.which("bwrap"):
        pytest.skip("no bubblewrap")
    s0 = I.find_slots()[0]
    results = Path(tempfile.mkdtemp(prefix="ocb-test-results-"))  # closed: 0700, no ACL for the agents
    (results / "sessions").mkdir()
    root = trial_dir()
    (root / "ws").mkdir()
    a = I.Agent(s0, root, keep=[root / "ws"])
    try:
        a.prepare()
        own = a.private_dir("sessions")
        (own / "s.jsonl").write_text("the harness's")
        a.grant()
        for deny, ok in ((results / "sessions", False), (own, True)):
            wrapper, _ = wrapper_for(root / "ws", deny_read=[deny], mode="bwrap")
            out = I.run_as(a, [*wrapper, "bash", "-c", "echo ran"])
            assert (out.strip() == "ran") is ok, out
        assert "[rc " in I.run_as(a, ["ls", str(own)])  # reachable for the mount, not readable
    finally:
        a.finish()
        shutil.rmtree(root)
        shutil.rmtree(results)


@real
def test_real_the_arms_folders_are_the_slots_while_it_runs():
    """smoke-matrix-2: Hermes refused a home it didn't own (`unsafe_control_path`). Every folder the harness makes in
    the trial for an arm is the slot's while the session runs, and the harness's again after."""
    import shutil

    s0 = I.find_slots()[0]
    root = trial_dir()
    (root / "ws").mkdir()
    (root / "hermes-home").mkdir()
    (root / "hermes-home" / "config.yaml").write_text("model: m\n")
    a = I.Agent(s0, root, keep=[root / "ws"])
    try:
        a.prepare()
        a.hand_over()
        check = f"[ -O {root}/hermes-home ] && echo owned; [ -O {root}/ws ] || echo ws-not-owned; cat {root}/hermes-home/config.yaml"
        assert I.run_as(a, ["bash", "-c", check]).split() == ["owned", "ws-not-owned", "model:", "m"]
        (root / "hermes-home" / "config.yaml").write_text("model: n\n")  # the harness still writes it
        assert (root / "hermes-home").stat().st_uid == s0.uid
    finally:
        a.finish()
    assert (root / "hermes-home").stat().st_uid == os.geteuid()  # reclaimed
    assert (root / "hermes-home" / "config.yaml").read_text() == "model: n\n"
    shutil.rmtree(root)


@real
def test_real_a_0600_file_written_mid_session_in_a_handed_folder_is_the_slots_to_read():
    """OpenClaw's config (0600, the key in it) is written at the first send, in `openclaw-state/`, which the slot owns
    by then: written through write_file (share_fd), the slot reads it; a plain write and chmod it couldn't."""
    import shutil

    from onchainbench.connect.workspace import write_file

    s0 = I.find_slots()[0]
    root = trial_dir()
    (root / "ws").mkdir()
    state = root / "openclaw-state"
    state.mkdir()
    a = I.Agent(s0, root, keep=[root / "ws"])
    try:
        a.prepare()
        a.hand_over()
        with a.active():  # inside the arm's send
            write_file(state / "run.json", '{"key": "k"}', 0o600)
            (state / "plain.json").write_text("{}")
            (state / "plain.json").chmod(0o600)
        assert I.run_as(a, ["cat", str(state / "run.json")]) == '{"key": "k"}'
        assert "[rc " in I.run_as(a, ["cat", str(state / "plain.json")])  # what the arm did before
    finally:
        a.finish()
    shutil.rmtree(root)


@real
def test_real_a_timeout_kills_the_agents_tree():
    s0 = I.find_slots()[0]
    a = I.Agent(s0, trial_dir())
    a.prepare()
    try:
        I.install()
        with a.active(), pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(["bash", "-c", "sleep 300 & sleep 300"], timeout=2, check=False)
        import time

        time.sleep(0.5)
        assert X.live_pids(s0.uid) == []  # the background sleep too, not only the process run started
    finally:
        a.finish()


def test_python_m_runs_the_shared_module_not_a_second_copy(monkeypatch):
    """`python -m onchainbench.isolation` must run the onchainbench.isolation every other module imports: a second
    copy has its own thread-local, so share_fd (called from connect.workspace) never sees the preflight's agent."""
    import runpy

    seen = []
    monkeypatch.setattr(I, "main", lambda argv=None: seen.append(I) or 0)
    with pytest.raises(SystemExit) as e:
        runpy.run_module("onchainbench.isolation", run_name="__main__")
    assert e.value.code == 0 and seen == [I]


def test_a_fork_is_always_started_as_the_harness(monkeypatch, tmp_path):
    """An agent's shell must never be able to signal the bench's Anvil: under isolation the agent's uid can't signal
    the harness's (the preflight checks it), so Anvil must never be started as an agent, even inside a turn."""
    from onchainbench import evm

    agent = I.Agent(SLOT, tmp_path)
    started = []

    def popen(cmd, **kw):
        started.append(I.current())
        raise RuntimeError("stop here")

    monkeypatch.setattr(evm.subprocess, "Popen", popen)
    monkeypatch.setattr(evm, "shared_relay", lambda url, port: types.SimpleNamespace(url="http://r", fixed=False))
    with agent.active(), pytest.raises(RuntimeError, match="stop here"):
        evm.Fork("http://upstream", 1, chain_id=100, log=tmp_path / "a.log")
    assert started == [None]  # inside the agent's turn, yet started with no agent active


@real
def test_real_an_agent_cant_signal_the_harness_or_sudo():
    s0 = I.find_slots()[0]
    a0 = I.Agent(s0, trial_dir())
    try:
        a0.prepare()
        assert I.run_as(a0, ["bash", "-c", I.SIGNAL_PROBE.format(pid=os.getpid())]).strip() == "refused"
        assert I.run_as(a0, ["bash", "-c", I.SUDO_PROBE]).strip() == "refused"
    finally:
        a0.finish()
