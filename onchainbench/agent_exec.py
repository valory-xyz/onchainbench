"""The root side of agent isolation (`onchainbench.isolation`): one short-lived privileged step per call.

The harness runs it as `sudo -n <python> -I agent_exec.py <spec.json>` (directly when it is root already). The spec
is a JSON file in the harness's private directory (mode 0600, so agents can't read it, and nothing secret goes on a
command line `ps` shows); it is deleted as soon as it is read. Standard library only: it runs as root.

Operations (`op`):
  exec      drop to the slot's uid, gid and groups, set the umask and the environment, and exec argv. The process
            dies with its parent (sudo), so killing the process the harness started takes it down too.
  firewall  (re)build the slot's iptables and ip6tables chain atomically: loopback traffic from the slot's uid may
            reach only the given ports (plus DNS and replies) and listeners of its own uid (a CLI's local gateway);
            everything else on loopback is refused.
  kill      send a signal to the given pids, only those owned by one of the given uids (a pid may have been reused).
  killall   kill every process of the slot's uid (as that uid: kill(-1)), until none is left.
  handover  give the slot what the harness owns in the arm's folders of a trial dir (its home, a CLI's home or
            state folder): some CLIs refuse a home they don't own.
  reclaim   hand back to the harness what the slot's uid owns under the trial's directory, and its top-level
            entries in the shared temp dirs (made private, and moved into the trial's `collect_to` directory when
            on the same file system), without following links. Run after killall.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

PR_SET_PDEATHSIG = 1
TMP_DIRS = ("/tmp", "/var/tmp", "/dev/shm")
USER_PREFIX = "ocb-agent-"  # the slot users (isolation.USER_PREFIX); the helper acts for no other uid
GROUP = "ocb-agents"
CHAIN_RE = re.compile(r"OCB-AGENT-[0-9]{1,2}")


class Refused(Exception):
    """A spec outside what the helper may do: it acts only for the slot users, on their own things."""


def slot_user(uid) -> tuple[str, int, set[int]]:
    """(name, primary gid, allowed groups) of an agent user, or Refused: root never acts for any other uid."""
    import grp
    import pwd

    try:
        pw = pwd.getpwuid(int(uid))
    except KeyError:
        raise Refused(f"uid {uid} is no user") from None
    if not pw.pw_name.startswith(USER_PREFIX) or not pw.pw_name[len(USER_PREFIX) :].isdigit() or pw.pw_uid == 0:
        raise Refused(f"uid {uid} ({pw.pw_name}) is not an agent user")
    allowed = {pw.pw_gid}
    with contextlib.suppress(KeyError):
        allowed.add(grp.getgrnam(GROUP).gr_gid)
    return pw.pw_name, pw.pw_gid, allowed


def caller_uid() -> int:
    """Whom files go back to: the user who ran sudo (or root's caller when the harness is root)."""
    return int(os.environ.get("SUDO_UID") or os.getuid())


def under_tmp(path: str) -> str:
    """A trial directory lives directly in a temp dir (tempfile.mkdtemp); the helper touches nothing elsewhere."""
    real = os.path.realpath(path)
    parent = os.path.dirname(real)
    tmps = {os.path.realpath(d) for d in (*TMP_DIRS, tempfile.gettempdir())}
    if parent not in tmps or real == parent:
        raise Refused(f"{path} is not a trial directory in a temp dir")
    return real


MARK_MASK = 0x7F000000  # the connmark bits naming the slot whose agent opened a connection to one of its own listeners
DENIED = 0x80000000  # the bit set once that connection turned out to reach someone else's listener
ANSWERED = 0x00800000  # the bit set once the slot's own listener answered the handshake (its SYN-ACK)


def slot_mark(chain: str) -> int:
    """The connmark of the slot's own-listener connections: its index + 1, in MARK_MASK's bits."""
    return (int(chain.rsplit("-", 1)[1]) + 1) << 24


def reply_chain(chain: str) -> str:
    return f"{chain}-R"


def chain_rules(chain: str, ports: list[int], uid: int | None = None) -> str:
    """The slot's chains for iptables-restore --noflush: declaring a chain flushes it, and the whole table commits
    at once, so there is never a moment without the final refusal.

    With `uid`, the slot may also connect to its own listeners on loopback (a CLI's local gateway on an ephemeral
    port, a helper process's server): a new connection to a port not on the list is let through but marked, and its
    first reply must come from a socket of the slot's uid (the reply chain: the SYN-ACK carries the listening
    socket's owner). A reply from anyone else's (the harness's, Anvil's, another slot's) is dropped and the
    connection marked denied; the agent's retransmitted SYN is then reset, so it sees a refusal within about a
    second, and the other side never accepts a connection. Nothing but the handshake reaches it.

    Only the SYN-ACK is judged by its owner; once it passed, the connection is marked answered and every later
    reply goes through. The kernel sends some replies from a socket no process holds yet: a connection the
    listener hasn't accept()ed has no owner, so the ACK of data the client sends first (an HTTP or WebSocket
    request to a busy server) matched no uid, was dropped, and the connection was reset (Hermes's gateway, in
    smoke-matrix-4: "did not receive a valid HTTP response"). The same holds for the replies after the server
    closes its end."""
    lines = [f":{chain} - [0:0]"]
    if uid is not None:
        mask = MARK_MASK | DENIED
        mark, denied = f"{slot_mark(chain):#x}/{mask:#x}", f"{slot_mark(chain) | DENIED:#x}/{mask:#x}"
        lines.append(f"-A {chain} -p tcp -m connmark --mark {denied} -j REJECT --reject-with tcp-reset")
    lines += [
        f"-A {chain} -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT",  # replies, e.g. to a harness that connects
        f"-A {chain} -p udp --dport 53 -j ACCEPT",  # the local resolver (systemd-resolved on 127.0.0.53)
        f"-A {chain} -p tcp --dport 53 -j ACCEPT",
    ]
    lines += [f"-A {chain} -p tcp --dport {p} -j ACCEPT" for p in sorted({_port(p) for p in ports})]
    if uid is not None:  # any other port: only if the slot's own socket answers (the reply chain)
        syn = f"-A {chain} -p tcp --syn -m conntrack --ctstate NEW"
        lines += [f"{syn} -j CONNMARK --set-xmark {mark}", f"{syn} -j ACCEPT"]
    lines += [f"-A {chain} -p tcp -j REJECT --reject-with tcp-reset", f"-A {chain} -j REJECT"]
    if uid is not None:
        r, answered = reply_chain(chain), f"{ANSWERED:#x}/{ANSWERED:#x}"
        own_syn_ack = f"-A {r} -p tcp --tcp-flags SYN,ACK SYN,ACK -m owner --uid-owner {int(uid)}"
        lines += [
            f":{r} - [0:0]",
            f"-A {r} -p tcp --tcp-flags RST RST -j RETURN",  # a refusal (our own reset to the agent's SYN) gets through
            f"-A {r} -m connmark --mark {answered} -j RETURN",  # the handshake passed: the rest of the connection
            f"{own_syn_ack} -j CONNMARK --set-xmark {answered}",  # the slot's own listener answered the handshake
            f"-A {r} -m connmark --mark {answered} -j RETURN",
            f"-A {r} -j CONNMARK --set-xmark {denied}",
            f"-A {r} -j DROP",
        ]
    return "*filter\n" + "\n".join(lines) + "\nCOMMIT\n"


def jump_rule(uid: int, chain: str) -> list[str]:
    """OUTPUT's jump for the slot: every packet the uid sends through loopback, which carries all traffic to the
    machine's own addresses (127.0.0.0/8, ::1 and the interfaces' own)."""
    return ["OUTPUT", "-o", "lo", "-m", "owner", "--uid-owner", str(uid), "-j", chain]


def reply_jump_rule(chain: str) -> list[str]:
    """OUTPUT's jump for the replies on the slot's own-listener connections, whoever sends them."""
    mark = f"{slot_mark(chain):#x}/{MARK_MASK:#x}"
    return [
        *("OUTPUT", "-o", "lo", "-p", "tcp", "-m", "connmark", "--mark", mark),
        *("-m", "conntrack", "--ctdir", "REPLY", "-j", reply_chain(chain)),
    ]


def _port(p) -> int:
    p = int(p)
    if not 0 < p < 65536:
        raise ValueError(f"bad port {p}")
    return p


def _tool(name: str) -> str:
    path = shutil.which(name) or shutil.which(name, path="/usr/sbin:/sbin:/usr/bin:/bin")
    if path is None:
        raise RuntimeError(f"{name} not found")
    return path


def firewall(spec: dict) -> None:
    chain, uid = spec["chain"], int(spec["uid"])
    slot_user(uid)
    if not CHAIN_RE.fullmatch(chain):
        raise Refused(f"bad chain name {chain!r}")
    text = chain_rules(chain, spec.get("ports") or [], uid)
    for tables in ("iptables", "ip6tables"):
        subprocess.run([_tool(tables + "-restore"), "-w", "--noflush"], input=text, text=True, check=True)
        for jump in (jump_rule(uid, chain), reply_jump_rule(chain)):
            if subprocess.run([_tool(tables), "-w", "-C", *jump], capture_output=True, check=False).returncode:
                subprocess.run([_tool(tables), "-w", "-I", jump[0], "1", *jump[1:]], check=True)


def proc_uid(pid: int) -> int | None:
    """The real uid of a live process, from /proc."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Uid:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def kill(spec: dict) -> None:
    uids, sig = {int(u) for u in spec["uids"]}, int(spec.get("sig", signal.SIGKILL))
    for u in uids:
        slot_user(u)
    for pid in spec["pids"]:
        if proc_uid(int(pid)) in uids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid), sig)


def killall(spec: dict) -> None:
    uid = int(spec["uid"])
    _, gid, _ = slot_user(uid)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child: covered through _kill_own
        os._exit(_kill_own(uid, gid))
    os.waitpid(pid, 0)


def _kill_own(uid: int, gid: int, rounds: int = 200) -> int:
    """As the slot's uid, kill(-1) reaches exactly that uid's processes (not this one); repeat until none is left,
    so a process forking meanwhile goes too."""
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    me = os.getpid()
    for _ in range(rounds):
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            return 0
        time.sleep(0.01)
        if not [p for p in live_pids(uid) if p != me]:  # only zombies left: their parents reap them
            return 0
    return 1


def live_pids(uid: int) -> list[int]:
    """The uid's processes that aren't zombies, from /proc."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        if state != "Z" and proc_uid(int(name)) == uid:
            out.append(int(name))
    return out


def _chown_tree(top_fd: int, uid: int, to_uid: int, to_gid: int) -> None:
    """Chown what uid owns below an open directory, without following links (fwalk opens every directory with
    O_NOFOLLOW relative to its parent)."""
    for _dirpath, dirs, files, dfd in os.fwalk(".", dir_fd=top_fd, follow_symlinks=False):
        for name in dirs + files:
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                if st.st_uid == uid:
                    os.chown(name, to_uid, to_gid, dir_fd=dfd, follow_symlinks=False)
            except OSError:
                continue


def reclaim(spec: dict) -> None:
    uid = int(spec["uid"])
    slot_user(uid)
    to_uid, to_gid = int(spec["to_uid"]), int(spec["to_gid"])
    if to_uid != caller_uid():
        raise Refused("files go back to the caller only")
    roots = [under_tmp(r) for r in spec.get("roots") or []]
    collect = _collect_to(spec, roots, to_uid)
    for root in roots:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if os.fstat(fd).st_uid == uid:
                os.fchown(fd, to_uid, to_gid)
            _chown_tree(fd, uid, to_uid, to_gid)
        finally:
            os.close(fd)
    for tmp in TMP_DIRS:
        try:
            names = os.listdir(tmp)
        except OSError:
            continue
        for name in names:
            path = os.path.join(tmp, name)
            try:
                st = os.stat(path, follow_symlinks=False)
            except OSError:
                continue
            if st.st_uid != uid:
                continue
            try:
                if stat.S_ISDIR(st.st_mode):
                    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        _chown_tree(fd, uid, to_uid, to_gid)
                        os.fchown(fd, to_uid, to_gid)
                        os.fchmod(fd, 0o700)  # the next trial on any slot can't look inside
                    finally:
                        os.close(fd)
                elif stat.S_ISREG(st.st_mode):
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    try:
                        os.fchown(fd, to_uid, to_gid)
                        os.fchmod(fd, 0o600)
                    finally:
                        os.close(fd)
                else:
                    os.chown(path, to_uid, to_gid, follow_symlinks=False)
                if collect is not None and st.st_dev == collect[1]:  # the trial's now: it keeps them, by trial
                    os.rename(path, os.path.join(collect[0], name))
            except OSError:
                continue


def handover(spec: dict) -> None:
    """Give the slot what the caller owns in the given folders of a trial dir, links not followed (the reverse of
    reclaim). Only folders directly in a trial dir, never the trial dir itself."""
    uid = int(spec["uid"])
    _, gid, _ = slot_user(uid)
    me = caller_uid()
    for d in spec["dirs"]:
        real = os.path.realpath(d)
        under_tmp(os.path.dirname(real))  # its parent is a trial dir directly in a temp dir
        fd = os.open(real, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if os.fstat(fd).st_uid == me:
                os.fchown(fd, uid, gid)
            _chown_tree(fd, me, uid, gid)
        finally:
            os.close(fd)


def _collect_to(spec: dict, roots: list[str], to_uid: int) -> tuple[str, int] | None:
    """(path, device) of the directory that receives the slot's shared-temp entries: one the caller made directly in
    one of the trial dirs being reclaimed, or None."""
    c = spec.get("collect_to")
    if not c:
        return None
    real = os.path.realpath(c)
    st = os.lstat(real)
    if os.path.dirname(real) not in roots or not stat.S_ISDIR(st.st_mode) or st.st_uid != to_uid:
        raise Refused(f"{c} is not the caller's directory in a trial dir")
    return real, st.st_dev


def die_with_parent() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


def run_exec(spec: dict) -> None:
    _, gid, allowed = slot_user(spec["uid"])
    groups = [int(g) for g in spec["groups"]]
    if int(spec["gid"]) != gid or not set(groups) <= allowed:
        raise Refused("an agent runs with its own group and the agents' group only")
    if not spec["argv"] or not all(isinstance(a, str) for a in spec["argv"]):
        raise Refused("argv must be a non-empty list of strings")
    os.setgroups(groups)
    os.setgid(gid)
    os.setuid(int(spec["uid"]))
    if os.getuid() != int(spec["uid"]) or os.geteuid() != int(spec["uid"]):  # pragma: no cover - the kernel's job
        raise SystemExit("could not drop privileges")
    os.umask(int(spec.get("umask", 0o077)))
    die_with_parent()  # after setuid: a credential change clears it
    if os.getppid() == 1:  # the parent is already gone
        raise SystemExit(1)
    argv, env = spec["argv"], spec["env"]
    try:
        os.execvpe(argv[0], argv, env)
    except OSError as e:
        print(f"agent_exec: {argv[0]}: {e}", file=sys.stderr)
        raise SystemExit(127) from None


OPS = {
    "exec": run_exec,
    "firewall": firewall,
    "kill": kill,
    "killall": killall,
    "handover": handover,
    "reclaim": reclaim,
}


def main(argv: list[str] | None = None) -> int:
    path = (argv if argv is not None else sys.argv[1:])[0]
    with open(path) as f:
        spec = json.load(f)
    os.unlink(path)
    op = OPS.get(spec.get("op"))
    if op is None:
        print(f"agent_exec: unknown op {spec.get('op')!r}", file=sys.stderr)
        return 2
    try:
        op(spec)
    except Refused as e:
        print(f"agent_exec: refused: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
