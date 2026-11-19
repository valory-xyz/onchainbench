"""Agent isolation: every process an arm starts runs as an unprivileged user of its own, locked out of the harness.

Opt-in with ONCHAINBENCH_AGENT_ISOLATION=1 (the GitHub run sets it on its Linux runners, after
`scripts/agent-users.sh` created the users). Without it nothing changes, and the integrity check (Anvil's log against
the harness's own admin calls) stays the backstop everywhere.

- **Users.** One user per concurrent slot, `ocb-agent-0`, `ocb-agent-1`, ... (all in the group `ocb-agents`); a trial
  holds one slot from session open to close, so two trials never share a uid.
- **Network.** A per-slot iptables and ip6tables chain on OUTPUT's loopback traffic (`-o lo` carries everything to
  the machine's own addresses: 127.0.0.0/8, ::1 and the interfaces'): the slot's uid may reach only its own trial's
  agent-facing ports (Connect's MCP, REST and RPC port, the brokers in front of it, the web corpus, the live-web
  logging proxy) plus DNS, and the listeners its own uid opens (a CLI's local gateway, a helper process's server, on
  whatever port: the owner of the socket that answers the handshake decides, `agent_exec.chain_rules`). Anvil, its
  upstream relay, other trials' ports and the harness's internals are refused. Outbound traffic to other hosts is
  unchanged. Between trials the chain refuses everything else on loopback (and the slot has no processes left to
  listen).
- **Processes.** While a session's method runs, `subprocess.Popen` started from that thread goes through
  `agent_exec.py` as root (sudo) and drops to the slot's uid: every contestant's CLI (and so everything it starts:
  its shell, its sandbox, olas-ocb-agent's own broker) and the scripted arms' shell. The environment travels in a
  0600 file, not on a command line. Timeouts kill the whole tree through the root helper; closing the session kills
  every process of the uid.
- **Files.** The trial's directory (workspace, the arms' homes and state, its temp dir) gets an ACL for the slot's
  uid; HOME is a directory inside it. Files the agent creates there carry a default ACL for the harness. At close
  the uid's files are handed back to the harness, and the slot loses access to the directory. Everything else of
  the harness's (checkouts, results, other trials) is outside the agent's reach, given the permissions
  `scripts/agent-users.sh` and the run workflow set.

No contestant code runs as the harness: only the scripted controls run in the bench's process. A broker the bench
starts for an arm (OpenClaw's, the scripted arms') is the harness's and joins the trial's
allowlist (`allow_broker`); olas-ocb-agent's own broker runs in its CLI's process as the slot's uid, on a localhost
port it picks, which its shell reaches through the rule for the slot's own listeners (`agent_exec.chain_rules`), the
same that lets a CLI reach its local gateway. Raw-key arms are handed an RPC URL by design; it is
Connect's own port (its method-filtered `/rpc`), so it stays reachable and the integrity check remains their control.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import platform
import queue
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from onchainbench import redact

FLAG = "ONCHAINBENCH_AGENT_ISOLATION"
USER_PREFIX = "ocb-agent-"
GROUP = "ocb-agents"
MAX_SLOTS = 64
HELPER = Path(__file__).with_name("agent_exec.py")
AGENT_UMASK = 0o077  # outside the trial's directory (its default ACL decides there): private to the slot
ACQUIRE_SECONDS = 3600

_ORIG_POPEN = subprocess.Popen
_local = threading.local()


def requested() -> bool:
    return os.environ.get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


# ---- slots ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    index: int
    name: str
    uid: int
    gid: int
    groups: tuple[int, ...]

    @property
    def chain(self) -> str:
        return f"OCB-AGENT-{self.index}"


def find_slots(prefix: str = USER_PREFIX) -> list[Slot]:
    """The agent users, ocb-agent-0 upwards until the first missing one."""
    import grp
    import pwd

    try:
        shared = [grp.getgrnam(GROUP).gr_gid]
    except KeyError:
        shared = []
    out = []
    for i in range(MAX_SLOTS):
        try:
            pw = pwd.getpwnam(f"{prefix}{i}")
        except KeyError:
            break
        out.append(Slot(i, pw.pw_name, pw.pw_uid, pw.pw_gid, tuple(dict.fromkeys([pw.pw_gid, *shared]))))
    return out


class Pool:
    def __init__(self, slots: list[Slot]):
        self.slots = slots
        self._q: queue.Queue = queue.Queue()
        for s in slots:
            self._q.put(s)

    def acquire(self, timeout: float = ACQUIRE_SECONDS) -> Slot:
        if not self.slots:
            raise RuntimeError(f"{FLAG} is set but there are no agent users ({USER_PREFIX}0...)")
        return self._q.get(timeout=timeout)

    def release(self, slot: Slot) -> None:
        self._q.put(slot)


_state: dict = {"pool": None, "private": None}  # this process's slot pool and spec directory
_pool_lock = threading.Lock()


def pool() -> Pool:
    with _pool_lock:
        if _state["pool"] is None:
            _state["pool"] = Pool(find_slots())
        return _state["pool"]


# ---- the root helper -----------------------------------------------------------------------------
def private_dir() -> Path:
    """The harness's own directory for helper specs (0700): agents can't read what is in them."""
    d = _state["private"]
    if d is None or not d.is_dir():
        d = _state["private"] = Path(tempfile.mkdtemp(prefix="ocb-iso-"))
        os.chmod(d, 0o700)
    return d


def write_spec(spec: dict) -> Path:
    fd, name = tempfile.mkstemp(dir=private_dir(), prefix="spec-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(spec, f)
    return Path(name)


def helper_argv(spec_path: Path) -> list[str]:
    root = [] if os.geteuid() == 0 else ["sudo", "-n"]
    return [*root, sys.executable, "-I", str(HELPER), str(spec_path)]


def run_helper(spec: dict, timeout: float = 60) -> None:
    """One privileged step (agent_exec.py), never as an agent: harness plumbing, even when called mid-session."""
    path = write_spec(spec)
    try:
        with harness():
            r = subprocess.run(
                helper_argv(path),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env={"PATH": os.environ.get("PATH", os.defpath)},
            )
    finally:
        path.unlink(missing_ok=True)
    if r.returncode:
        raise RuntimeError(f"agent_exec {spec['op']} failed ({r.returncode}): {(r.stderr or r.stdout)[-500:]}")


# ---- launching processes as the agent -------------------------------------------------------------
_POPEN_PARAMS = list(inspect.signature(_ORIG_POPEN.__init__).parameters)[2:]  # after self and args


class TurnClosed(RuntimeError):
    """A process start refused: the turn that asked for it was ended by the runner, or the trial's agent user is
    finished. A turn the runner abandons keeps running in its daemon thread; it can't act any more."""


class TurnProcs(list):
    """The processes a turn started (watchdog); once `closed` (the runner ended the turn), the turn's thread may
    start no more, with or without isolation."""

    closed = False


class AgentPopen(_ORIG_POPEN):
    """subprocess.Popen, which starts the process as the thread's current agent when one is active, and refuses to
    start one for a turn the runner ended (TurnClosed)."""

    def __init__(self, args, *pos, **kw):
        kw.update(zip(_POPEN_PARAMS, pos, strict=False))
        if getattr(getattr(_local, "procs", None), "closed", False):
            raise TurnClosed("the runner ended this turn: no process starts from it")
        agent = getattr(_local, "agent", None)
        self._ocb_agent, spec = agent, None
        if agent is not None:
            args, kw, spec = agent.launch(args, kw)
        try:
            super().__init__(args, **kw)
        except BaseException:
            if spec is not None:
                spec.unlink(missing_ok=True)
            raise
        tracked = getattr(_local, "procs", None)
        if tracked is not None:  # a turn under the runner's wall clock (watchdog): its processes, to kill at the end
            tracked.append(self)

    def send_signal(self, sig):
        agent = getattr(self, "_ocb_agent", None)
        if agent is not None and self.returncode is None:
            agent.signal_tree(self.pid, sig)  # what runs below sudo is the agent's: only root may signal it
        super().send_signal(sig)


def install() -> None:
    """Route subprocess.Popen through AgentPopen (idempotent). Outside an agent's context it is the original."""
    subprocess.Popen = AgentPopen


def current():
    return getattr(_local, "agent", None)


def read_text(path: Path | str, timeout: float = 120) -> str:
    """A file an arm's processes wrote, read on the arm's side: as the thread's agent user when one is active (what
    a CLI writes in a folder the slot owns is often 0600, so the harness's ACL entry grants nothing), else directly.
    Raises OSError when it can't be read, either way."""
    if current() is None:
        return Path(path).read_text()
    r = subprocess.run(["cat", "--", str(path)], capture_output=True, text=True, timeout=timeout, check=False)
    if r.returncode:
        raise OSError(f"as the agent: cat {path}: {r.stderr.strip()[-300:]}")
    return r.stdout


@contextlib.contextmanager
def track(procs: list):
    """Collect the Popen objects this thread creates (a turn's: watchdog), with or without isolation."""
    prev = getattr(_local, "procs", None)
    _local.procs = procs
    try:
        yield procs
    finally:
        _local.procs = prev


@contextlib.contextmanager
def harness():
    """The harness's own processes (the root helper, `ps`), even inside an agent's session."""
    prev = getattr(_local, "agent", None)
    _local.agent = None
    try:
        yield
    finally:
        _local.agent = prev


def kill_pids(pids: list[int], sig: int = 9) -> None:
    """Kill processes the harness may not signal itself (an agent's, across the uid boundary); only pids owned by an
    agent user are touched. A no-op without isolation."""
    if not pids or not requested():
        return
    uids = [s.uid for s in pool().slots]
    with contextlib.suppress(Exception):  # best effort: closing the session kills the uid's processes anyway
        run_helper({"op": "kill", "pids": [int(p) for p in pids], "uids": uids, "sig": int(sig)})


# ---- ACLs (the harness, as the owner of the trial's files) -------------------------------------------
ACL_USER_OBJ, ACL_USER, ACL_GROUP_OBJ, ACL_GROUP, ACL_MASK, ACL_OTHER = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
ACL_UNDEFINED = 0xFFFFFFFF
ACCESS, DEFAULT = "system.posix_acl_access", "system.posix_acl_default"


def acl_decode(raw: bytes) -> list[tuple[int, int, int]]:
    (version,) = struct.unpack_from("<I", raw)
    if version != 2:
        raise ValueError(f"ACL version {version}")
    return [struct.unpack_from("<HHI", raw, 4 + 8 * i) for i in range((len(raw) - 4) // 8)]


def acl_encode(entries: list[tuple[int, int, int]]) -> bytes:
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *e) for e in sorted(entries))


def acl_from_mode(mode: int) -> list[tuple[int, int, int]]:
    return [
        (ACL_USER_OBJ, (mode >> 6) & 7, ACL_UNDEFINED),
        (ACL_GROUP_OBJ, (mode >> 3) & 7, ACL_UNDEFINED),
        (ACL_OTHER, mode & 7, ACL_UNDEFINED),
    ]


def acl_set_user(entries: list, uid: int, perm: int | None) -> list:
    """entries with the named-user entry for uid set to perm (None: removed) and the mask recomputed."""
    out = [e for e in entries if not (e[0] == ACL_USER and e[2] == uid) and e[0] != ACL_MASK]
    if perm is not None:
        out.append((ACL_USER, perm, uid))
    group_class = [e[1] for e in out if e[0] in (ACL_USER, ACL_GROUP_OBJ, ACL_GROUP)]
    if any(e[0] in (ACL_USER, ACL_GROUP) for e in out):
        mask = 0
        for p in group_class:
            mask |= p
        out.append((ACL_MASK, mask, ACL_UNDEFINED))
    return sorted(out)


def default_acl(uids: list[int]) -> list:
    """For the trial's directories: what is created in them is the owner's and these users' (the slot's and the
    harness's), nobody else's."""
    entries = [(ACL_USER_OBJ, 7, ACL_UNDEFINED), (ACL_GROUP_OBJ, 0, ACL_UNDEFINED), (ACL_OTHER, 0, ACL_UNDEFINED)]
    entries += [(ACL_USER, 7, u) for u in dict.fromkeys(uids)]
    entries.append((ACL_MASK, 7, ACL_UNDEFINED))
    return sorted(entries)


def _get_acl(fd: int, st: os.stat_result) -> list:
    try:
        return acl_decode(os.getxattr(fd, ACCESS))
    except OSError:  # none yet (ENODATA): the mode's
        return acl_from_mode(st.st_mode)


def _grant_fd(fd: int, st: os.stat_result, uid: int, me: int) -> None:
    is_dir = stat.S_ISDIR(st.st_mode)
    perm = 7 if is_dir or st.st_mode & 0o111 else 6
    acl = _get_acl(fd, st)
    # the harness's own entry too: it keeps its access once a folder is handed over to the slot (Agent.hand_over)
    new = acl_set_user(acl_set_user(acl, uid, perm), me, perm)
    if new != acl:
        os.setxattr(fd, ACCESS, acl_encode(new))
    if is_dir:
        os.setxattr(fd, DEFAULT, acl_encode(default_acl([uid, me])))


# What every agent process may inherit from the runner's environment, the same for every arm: the shell's basics
# (connect/shell.py's allowlist), locales, certificates and the proxy settings. Anything else the runner has (CI
# variables that name the arm or the model, ACTIONS_*, keys) stays out, unless the arm passes it on purpose: a
# value it set itself, or a variable it declared (declare(), e.g. the model key its CLI calls the model with).
AGENT_ENV_EXTRA = frozenset({"SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "NODE_USE_ENV_PROXY", "ALL_PROXY", "all_proxy"})


def env_allowed(name: str) -> bool:
    from onchainbench.connect.shell import ENV_ALLOW

    return name in ENV_ALLOW or name in AGENT_ENV_EXTRA or name.startswith("LC_")


def agent_environ(env: dict, declared: set) -> dict:
    """env for an agent process: allowlisted names, declared ones, and the ones the arm set to a value of its own."""
    return {k: v for k, v in env.items() if env_allowed(k) or k in declared or os.environ.get(k) != v}


def declare(names) -> None:
    """Variables the arm passes its process on purpose (its model key, its home): kept even when their value is the
    runner's own. A no-op outside an agent's session."""
    agent = current()
    if agent is not None:
        agent.declared.update(names)


def sticky_trial() -> bool:
    """For an arm whose harness refuses a temp dir under a writable ancestor that isn't sticky (OpenClaw: "No safe
    OpenClaw temp root is available"), called as its session opens: the trial dir, and from now on the folders handed
    to the slot (its temp dir, its state folder), get the sticky bit, like /tmp. Under isolation the slot's ACL entry
    makes them show as group-writable (the ACL mask is the mode's group bits), which that check reads as shared. The
    handed folders become the slot's own, so the sticky bit costs it nothing there; in the trial dir only the owner of
    an entry may remove or rename it, so the harness's folders (the workspace) stay where they are. Every other arm's
    layout is unchanged. A no-op outside an agent's session (no ACLs: the dirs are 0700)."""
    agent = current()
    if agent is None:
        return False
    agent.sticky = True
    make_sticky(agent.root)
    return True


def share_fd(fd: int) -> bool:
    """A file the harness just wrote for the arm, while an agent's session runs (a shim, a script): the slot's ACL
    entry, right away, not at the next grant(): a chmod (mode 0600) empties the mask that gives the slot its read,
    and the arm may read the file before its next turn (a shim written inside its turn). The same
    for every arm; outside an agent's context, or for a file outside its trial, nothing happens."""
    agent = current()
    if agent is None:
        return False
    st = os.fstat(fd)
    real = os.path.realpath(f"/proc/self/fd/{fd}") if os.path.isdir("/proc/self/fd") else None
    if st.st_uid != os.geteuid() or not stat.S_ISREG(st.st_mode):
        return False
    if real is not None and not real.startswith(os.path.realpath(agent.root) + os.sep):
        return False
    _grant_fd(fd, st, agent.slot.uid, os.geteuid())
    return True


def grant(root: Path, uid: int, skip: list | tuple = ()) -> int:
    """Give uid access to everything the harness owns under root: an ACL entry on each directory and regular file,
    and on directories a default ACL, so what either side creates stays reachable by the other. Every entry is
    opened without following links, relative to its parent: a link the agent planted is skipped, never followed.
    Directories in skip (the harness's own, e.g. a sessions dir) and what is in them are left alone. Returns how many
    entries changed hands."""
    me, n = os.geteuid(), 0
    skipped = {os.path.realpath(p) for p in skip}
    for dirpath, dirs, files, dfd in os.fwalk(root, follow_symlinks=False):
        dirs[:] = [d for d in dirs if os.path.realpath(os.path.join(dirpath, d)) not in skipped]  # not descended
        st = os.fstat(dfd)
        if st.st_uid == me:
            _grant_fd(dfd, st, uid, me)
            n += 1
        for name in files:
            try:
                if os.stat(name, dir_fd=dfd, follow_symlinks=False).st_uid != me:
                    continue
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY, dir_fd=dfd)
            except OSError:
                continue
            try:
                st = os.fstat(fd)
                if stat.S_ISREG(st.st_mode) and st.st_uid == me:
                    _grant_fd(fd, st, uid, me)
                    n += 1
            finally:
                os.close(fd)
    return n


def make_sticky(top: Path, tree: bool = False) -> int:
    """Set the sticky bit on top (with `tree`, on every directory below it too) where the harness owns it, its mode
    otherwise unchanged (the group bits are the ACL mask). Links are never followed. Returns how many directories
    changed (none on Windows, which has no modes to set). Used for the arms that ask for a sticky trial dir
    (`sticky_trial`)."""
    if not hasattr(os, "O_DIRECTORY"):
        return 0
    me, n = os.geteuid(), 0

    def one(fd: int) -> None:
        nonlocal n
        st = os.fstat(fd)
        if st.st_uid == me and not st.st_mode & stat.S_ISVTX:
            os.fchmod(fd, stat.S_IMODE(st.st_mode) | stat.S_ISVTX)
            n += 1

    if not tree:
        fd = os.open(top, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            one(fd)
        finally:
            os.close(fd)
        return n
    for _, _, _, dfd in os.fwalk(top, follow_symlinks=False):
        one(dfd)
    return n


def revoke(root: Path, uid: int) -> None:
    """Take uid's entry off root itself: with no way into the directory, nothing below it is reachable."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        os.setxattr(fd, ACCESS, acl_encode(acl_set_user(_get_acl(fd, st), uid, None)))
    finally:
        os.close(fd)


# ---- one trial's agent --------------------------------------------------------------------------
def url_port(url: str | None) -> int | None:
    if not url:
        return None
    u = urllib.parse.urlsplit(url)
    return u.port or (443 if u.scheme == "https" else 80)


def is_local(url: str | None) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "") if url else ""
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0") or host.startswith("127.")


GITCONFIG = "[safe]\n\tdirectory = *\n"  # the trial's files are the harness's: git would call them dubious


@dataclass
class Agent:
    slot: Slot
    root: Path  # the trial's directory
    ports: set = field(default_factory=set)
    forbidden: set = field(default_factory=set)  # Anvil's and its relay's: never allowed, whoever asks
    private: list = field(default_factory=list)  # directories in the trial that grant() leaves alone
    keep: list = field(default_factory=list)  # directories in the trial that stay the harness's (the workspace)
    handed: set = field(default_factory=set)  # the trial's folders the slot owns now
    declared: set = field(default_factory=set)  # variables the arm passes its process on purpose (declare())
    finished: bool = False  # finish() ran: the slot is the harness's again, no process may start as it
    sticky: bool = False  # the arm asked for sticky folders (sticky_trial): the trial dir and what is handed over

    @property
    def home(self) -> Path:
        return self.root / "home"

    # -- network
    def allow(self, url_or_port) -> None:
        port = url_or_port if isinstance(url_or_port, int) else url_port(url_or_port)
        if isinstance(url_or_port, str) and not is_local(url_or_port):
            return  # not on this machine: the loopback rules don't apply
        if port is None or port in self.ports:
            return
        if port in self.forbidden:
            raise RuntimeError(f"refusing to open port {port} to the agent: it is the fork's")
        self.ports.add(port)
        if getattr(self, "_synced", False):
            self.sync()

    def allow_trial(self, t) -> None:
        """The trial's agent-facing services the harness hands every arm."""
        fork = getattr(t, "fork", None)
        for u in (getattr(fork, "url", None), getattr(getattr(fork, "relay", None), "url", None)):
            if (p := url_port(u)) is not None:
                self.forbidden.add(p)
        fc = t.fc
        self.allow(getattr(fc, "url", None))  # Connect: MCP, REST and the /rpc the raw-key arms use
        self.allow(getattr(getattr(fc, "s", None), "rpc_url", None))
        if t.web is not None:
            self.allow(t.web.base)
        self.allow((t.data.get("proxy_env") or {}).get("HTTP_PROXY"))

    def sync(self) -> None:
        run_helper({"op": "firewall", "uid": self.slot.uid, "chain": self.slot.chain, "ports": sorted(self.ports)})
        self._synced = True

    # -- files
    def prepare(self) -> None:
        self.home.mkdir(exist_ok=True)
        (self.home / ".gitconfig").write_text(GITCONFIG)
        self.grant()

    def grant(self) -> None:
        grant(self.root, self.slot.uid, skip=self.private)

    def hand_over(self) -> list[str]:
        """Give the slot ownership of the folders the harness made in the trial for the arm (its home, its temp dir,
        a CLI's home or state folder): some CLIs refuse a home they don't own. The same rule for every arm: every
        directory directly in the trial dir but the workspace and the harness's private ones. The harness keeps its
        access through its ACL entry; close hands them back (reclaim)."""
        exclude = {os.path.realpath(p) for p in [*self.keep, *self.private]}
        new = []
        for p in sorted(self.root.iterdir()):
            real = os.path.realpath(p)
            if p.name.startswith(".") or p.is_symlink() or not p.is_dir() or real in exclude or real in self.handed:
                continue
            new.append(real)
        if new:
            for d in new if self.sticky else ():  # while the harness still owns them (sticky_trial)
                make_sticky(Path(d), tree=True)
            run_helper({"op": "handover", "uid": self.slot.uid, "dirs": new})
            self.handed.update(new)
        return [os.path.basename(p) for p in new]

    def private_dir(self, name: str) -> Path:
        """A directory in the trial that stays the harness's alone (no ACL for the slot, none inherited, never
        granted): the agent user can reach the path, e.g. for its sandbox to mount over it, but not look inside."""
        d = self.root / name
        d.mkdir(mode=0o700)
        fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for attr in (ACCESS, DEFAULT):  # what it inherited from the trial dir's default ACL
                with contextlib.suppress(OSError):
                    os.removexattr(fd, attr)
            os.fchmod(fd, 0o700)
        finally:
            os.close(fd)
        self.private.append(d)
        return d

    # -- processes
    @contextlib.contextmanager
    def active(self):
        prev = getattr(_local, "agent", None)
        _local.agent = self
        try:
            yield self
        finally:
            _local.agent = prev

    def launch(self, args, kw: dict) -> tuple[list[str], dict, Path]:
        """Popen's arguments rewritten to start the process through the root helper as this slot. Refused once the
        slot's agent is finished: a turn the runner abandoned still runs in its thread with this agent active, and
        the slot may already belong to the next trial."""
        if self.finished:
            raise TurnClosed(f"the trial's agent user {self.slot.name} is finished: no process starts as it")
        shell, executable = kw.pop("shell", False), kw.pop("executable", None)
        if kw.get("pass_fds"):
            raise ValueError("an isolated agent process can't inherit file descriptors (sudo closes them)")
        argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
        argv = [os.fsdecode(a) for a in argv]
        if shell:
            argv = [os.fsdecode(executable) if executable else "/bin/sh", "-c", *argv]
        elif executable:
            argv = [os.fsdecode(executable), *argv[1:]]
        env = kw.get("env")
        env = dict(os.environ if env is None else {os.fsdecode(k): os.fsdecode(v) for k, v in env.items()})
        env = agent_environ(env, self.declared)
        env |= {"HOME": str(self.home), "USER": self.slot.name, "LOGNAME": self.slot.name}
        s = self.slot
        spec = write_spec(
            {
                "op": "exec",
                "uid": s.uid,
                "gid": s.gid,
                "groups": list(s.groups),
                "umask": AGENT_UMASK,
                "env": env,
                "argv": argv,
            }
        )
        kw["env"] = {"PATH": os.environ.get("PATH", os.defpath)}  # sudo's own; the agent's is in the spec
        return helper_argv(spec), kw, spec

    def signal_tree(self, pid: int, sig: int) -> None:
        from onchainbench.connect.shell import _descendants

        if kids := _descendants(pid):
            with contextlib.suppress(Exception):
                run_helper({"op": "kill", "pids": kids, "uids": [self.slot.uid], "sig": int(sig)})

    def finish(self) -> None:
        """Kill everything the slot runs, hand its files back, lock the slot out of the trial and loopback. From here
        on nothing launches as it (`launch`), before the kill, so nothing new starts while the old is killed."""
        self.finished = True
        s, problems = self.slot, []

        def reclaim():
            # what the slot left in the shared temp dirs moves into its own trial (the run keeps it by trial): no
            # other run, nor this run's end-of-run sweep, has to guess whose it is
            collect = tempfile.mkdtemp(prefix=".shared-tmp-", dir=self.root)
            spec = {"uid": s.uid, "to_uid": os.geteuid(), "to_gid": os.getegid(), "roots": [str(self.root)]}
            run_helper({"op": "reclaim", **spec, "collect_to": collect})

        steps = (
            lambda: run_helper({"op": "killall", "uid": s.uid, "gid": s.gid}),
            reclaim,
            lambda: revoke(self.root, s.uid),
            lambda: run_helper({"op": "firewall", "uid": s.uid, "chain": s.chain, "ports": []}),
        )
        for step in steps:
            try:
                step()
            except Exception as e:  # noqa: BLE001 - every step runs; the first failure is raised after
                problems.append(e)
        if problems:
            raise problems[0]


class IsolatedSession:
    """An arm's session whose processes run as the trial's agent user (see the module docstring)."""

    def __init__(self, inner, agent: Agent, release):
        self.inner, self.agent, self._release = inner, agent, release
        self.sessions = None  # (the trial's sessions dir, the run's), when the arm keeps session files
        self.secrets = list  # the trial's secrets, which the hand-over redacts from those files

    def __getattr__(self, name):  # stats, oracle, backend_spent ...: the arm's own
        if name in ("inner", "agent", "_release", "sessions", "secrets", "_finished"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def send(self, text: str) -> str:
        self.agent.grant()  # files the harness wrote since (a turn's event, a resumed session's shim)
        self.agent.hand_over()  # folders it made since
        with self.agent.active():
            return self.inner.send(text)

    def crash_and_resume(self) -> None:
        with self.agent.active():
            self.inner.crash_and_resume()

    # no transcript() here: __getattr__ forwards it, so a session without one (the scripted arms') still has none,
    # and the runner's optional lookup (run_trial_session) gets its default instead of an AttributeError

    def kill_agent_processes(self) -> None:
        """Every process of the trial's agent user, now (the runner's wall clock on a turn: watchdog)."""
        s = self.agent.slot
        run_helper({"op": "killall", "uid": s.uid, "gid": s.gid})

    def close(self):
        stats = None
        try:
            with self.agent.active():
                stats = self.inner.close()
            return stats
        finally:
            self._finish(stats)

    def abandon(self) -> None:
        """The arm's close never returned (watchdog): give the user back without it, as close would."""
        self._finish(None)

    def _finish(self, stats) -> None:
        with _finish_lock:
            if getattr(self, "_finished", False):
                return  # a late close after abandon(), or the other way round: once only
            self._finished = True
        try:
            self.agent.finish()
        finally:
            self._release()
            if self.sessions is not None:  # the session files, where the run keeps them (and inspect finds them)
                hand_over_sessions(*self.sessions, stats, self.secrets())


_finish_lock = threading.Lock()


def isolated(t, make, sessions_dir: Path | None = None):
    """Open an arm's session (`make(sessions_dir)`) with its processes as an agent user of its own; see the module
    docstring. A run's sessions_dir (under results/, closed to the agents) is replaced by a harness-only directory in
    the trial, where the arm's close copies its session files (olas-ocb-agent's: its CLI writes them in a folder of
    its own in the trial). They move to sessions_dir, redacted, once the agent user is finished."""
    install()
    p = pool()
    slot = p.acquire()
    agent = Agent(slot, Path(t.ws).parent, keep=[Path(t.ws)])
    own = None
    try:
        t.data["agent_user"] = agent
        agent.allow_trial(t)
        agent.prepare()
        if sessions_dir is not None:
            own = agent.private_dir("sessions")
        agent.sync()
        with agent.active():
            inner = make(own)
        agent.grant()  # what the session wrote (shims, configs) before any agent command ran
        agent.hand_over()  # and the arm's folders, which some CLIs must own
    except BaseException:
        try:
            agent.finish()
        finally:
            p.release(slot)
        raise
    s = IsolatedSession(inner, agent, lambda: p.release(slot))
    s.sessions = (own, Path(sessions_dir)) if own is not None else None
    s.secrets = lambda: redact.trial_secrets(t)  # at close: the arm registers its keys when it opens
    return s


def hand_over_sessions(own: Path, dest: Path, stats, secrets=()) -> list[str]:
    """Move a session's files from its trial's sessions dir to the run's, redacted (onchainbench.redact, with the
    trial's `secrets`), and point its stats at their new place."""
    moved = []
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted(own.iterdir()):
        shutil.move(str(p), str(dest / p.name))
        redact.tree(dest / p.name, secrets)
        moved.append(p.name)
    extra = getattr(stats, "extra", None)
    if isinstance(extra, dict) and isinstance(extra.get("session"), str):
        path = Path(extra["session"])
        if path.parent == own:
            extra["session"] = str(dest / path.name)
    return moved


def allow_broker(t, broker) -> None:
    """A broker the bench started for an arm's agent: its port joins the trial's allowlist. (olas-ocb-agent's own
    broker is the slot's listener: see the module docstring.)"""
    agent = (getattr(t, "data", None) or {}).get("agent_user")
    if agent is not None:
        agent.allow(getattr(broker, "url", None))


# ---- preflight ---------------------------------------------------------------------------------
PROBE = "exec 3<>/dev/tcp/127.0.0.1/{port} && echo open || echo refused"
SIGNAL_PROBE = "kill -0 {pid} 2>/dev/null && echo signalled || echo refused"  # a harness pid: another uid's
SUDO_PROBE = "sudo -n true 2>/dev/null && echo sudo || echo refused"
# the slot's own listener on a port it picks, then a connection to it: what a CLI's local gateway does. The client
# speaks first and the server accepts late, as a busy gateway does with a WebSocket request: the kernel acknowledges
# that request from a socket no process holds yet, which must not cost the connection (smoke-matrix-4's Hermes).
# The answer must come back through it.
OWN_PROBE = (
    'python3 -c \'import socket, time; s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1); '
    'c = socket.create_connection(s.getsockname(), timeout=10); c.sendall(b"ping"); time.sleep(0.5); '
    'a = s.accept()[0]; a.sendall(a.recv(4)); print("open" if c.recv(4) == b"ping" else "refused")\' '
    "2>/dev/null || echo refused"
)


def _listener():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    return s


def probe(slot: Slot) -> list[str]:
    """As the slot's user, with only one of two harness ports allowed: the allowed one must open, the other refuse,
    a listener the slot opens itself must take its own connection, and the harness's processes must refuse its
    signals (and sudo the slot)."""
    problems = []
    a, b = _listener(), _listener()
    with tempfile.TemporaryDirectory(prefix="ocb-probe-") as d:
        agent = Agent(slot, Path(d))
        agent.ports = {a.getsockname()[1]}
        try:
            agent.prepare()
            agent.sync()
            pa, pb = a.getsockname()[1], b.getsockname()[1]
            checks = [(PROBE.format(port=pa), "open", f"port {pa}"), (PROBE.format(port=pb), "refused", f"port {pb}")]
            checks.append((OWN_PROBE, "open", "its own listener"))
            # the harness's processes (Anvil, the relays, the brokers: the harness's uid) can't be signalled, and the
            # slot can't become root: an agent's `pkill -f "anvil --fork-url"` (open-run-1) matches but kills nothing
            checks.append((SIGNAL_PROBE.format(pid=os.getpid()), "refused", "signalling the harness"))
            checks.append((SUDO_PROBE, "refused", "sudo"))
            for cmd, want, what in checks:
                got = run_as(agent, ["bash", "-c", cmd])
                if got.strip() != want:
                    problems.append(f"{slot.name} on {what}: {got.strip() or 'no answer'}, expected {want}")
            problems += shim_readable(agent)
        except Exception as e:  # noqa: BLE001 - reported as a problem
            problems.append(f"{slot.name}: {type(e).__name__}: {e}")
        finally:
            with contextlib.suppress(Exception):
                agent.finish()
            a.close()
            b.close()
    return problems


def shim_readable(agent: Agent) -> list[str]:
    """The stand-in .mcp.json a Connect session writes into the agent's directory mid-turn (mode 0600, the harness's)
    must be readable by the slot, as `pearl` reads it from there."""
    from onchainbench.connect.workspace import write_file

    d = agent.root / "agent"
    d.mkdir(exist_ok=True)
    with agent.active():  # as inside a turn
        write_file(d / ".mcp.json", '{"probe": 1}', 0o600)
    got = run_as(agent, ["cat", str(d / ".mcp.json")])
    return [] if got.strip() == '{"probe": 1}' else [f"{agent.slot.name} can't read a shim: {got.strip()[-200:]}"]


def run_as(agent: Agent, argv: list[str], timeout: float = 60) -> str:
    install()
    with agent.active():
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, cwd=agent.root)
    return r.stdout + (f"[rc {r.returncode}] {r.stderr[-300:]}" if r.returncode else "")


def problems(workers: int = 1) -> list[str]:
    """Why isolation can't run here as requested (empty: it can)."""
    if platform.system() != "Linux":
        return [f"{FLAG} needs Linux (iptables owner match); unset it to run unisolated"]
    out = []
    slots = pool().slots
    if len(slots) < workers:
        out.append(f"{len(slots)} agent users for {workers} workers: run scripts/agent-users.sh {workers}")
    if os.geteuid() != 0 and subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        out.append("passwordless sudo is needed for the root helper")
    for tool in ("iptables", "ip6tables"):
        if not (shutil.which(tool) or shutil.which(tool, path="/usr/sbin:/sbin")):
            out.append(f"{tool} not found")
    if out or not slots:
        return out
    for s in slots:
        out += probe(s) if s is slots[0] else _baseline(s)
    if shutil.which("bwrap"):  # the Linux sandboxes (the bench's, olas-ocb-agent's, Codex's) must start as the agent
        with tempfile.TemporaryDirectory(prefix="ocb-probe-") as d:
            agent = Agent(slots[0], Path(d))
            agent.prepare()
            got = run_as(agent, ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-pid", "--", "echo", "ok"])
            if got.strip() != "ok":
                out.append(f"bubblewrap fails as {slots[0].name}: {got.strip()[-300:]}")
            with contextlib.suppress(Exception):
                agent.finish()
    return out


def _baseline(slot: Slot) -> list[str]:
    try:
        run_helper({"op": "firewall", "uid": slot.uid, "chain": slot.chain, "ports": []})
    except Exception as e:  # noqa: BLE001 - reported as a problem
        return [f"{slot.name}: {e}"]
    return []


def main(argv: list[str] | None = None) -> int:
    """`python -m onchainbench.isolation [--workers N] [--run CMD]...`: the preflight, then each CMD as the first
    agent user (e.g. an arm's `claude --version`), failing on any problem."""
    import argparse
    import shlex

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run", action="append", default=[])
    a = ap.parse_args(argv)
    found = problems(a.workers)
    if not found:
        with tempfile.TemporaryDirectory(prefix="ocb-probe-") as d:
            agent = Agent(pool().slots[0], Path(d))
            agent.prepare()
            for cmd in a.run:
                got = run_as(agent, shlex.split(cmd))
                print(f"$ {cmd}\n{got.rstrip()}")
                if "[rc " in got:
                    found.append(f"as {agent.slot.name}: {cmd} failed")
            with contextlib.suppress(Exception):
                agent.finish()
    for p in found:
        print("isolation:", p)
    print("agent isolation: " + ("NOT READY" if found else f"ready, {len(pool().slots)} agent users"))
    return 1 if found else 0


if __name__ == "__main__":
    # `python -m` runs this file as __main__, a second copy of the module beside the onchainbench.isolation that
    # every other module imports: its own thread-local, so current() there never sees the agent this copy
    # activates (share_fd would grant nothing in the preflight). Run the one copy everybody shares.
    from onchainbench import isolation as _shared  # noqa: PLW0406 (on purpose: the shared copy)

    sys.exit(_shared.main())
