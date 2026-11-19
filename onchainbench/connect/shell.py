"""A persistent bash session: `cd`, exported variables and activated venvs survive between actions.

Running every action in a fresh subprocess keeps a harness simple
but makes chat sessions awkward ("cd into the repo" is forgotten next turn).
Here one bash process lives for the session. Each action is written to a file
and sourced with stdin from /dev/null, so a command that reads stdin can't eat
the end marker. On timeout the process group is killed and bash restarted in
the last known cwd.

On native Windows the shell is Git Bash: a bare "bash" there can start WSL's launcher instead (Windows
searches System32 before PATH), there are no process groups (taskkill /T kills the tree), and the cwd is
read back as a Windows path so bash can be restarted in it.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

WINDOWS = os.name == "nt"

ENV_ALLOW = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "TERM",
    "TMPDIR",
    "TZ",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    # what Windows programs need to start (Git Bash and anything it runs)
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
}


class PersistentShell:
    def __init__(
        self,
        cwd: str,
        env: dict | None = None,
        wrapper: list[str] | None = None,
        timeout: int = 120,
        max_output: int = 10000,
    ):
        self.cwd = str(Path(cwd).resolve())
        # an allowlisted environment, not the harness's: API keys and tokens stay out of the agent's shell
        self.env = {k: v for k, v in os.environ.items() if k in ENV_ALLOW or k.startswith("LC_")} | (env or {})
        self.wrapper = wrapper or []
        self.timeout = timeout
        self.hold = lambda: False  # true while the broker handles one of this shell's calls (set by the runtime)
        self.max_hold = 600.0  # the broker's call deadline (policy.CALL_DEADLINE_S, set by the runtime)
        self.on_kill = lambda: None  # tells the broker the running command was killed (set by the runtime)
        self.max_output = max_output
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [*self.wrapper, _bash(), "--noprofile", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.cwd,
            env=self.env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self._q = queue.Queue()
        q, out = self._q, self.proc.stdout

        def pump():
            for line in out:
                q.put(line)
            q.put(None)

        threading.Thread(target=pump, daemon=True).start()

    def run(self, command: str, timeout: int | None = None) -> dict:
        """Run one action. Returns {output, returncode, timed_out}."""
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        tag = uuid.uuid4().hex
        marker = f"__OCB_{tag}__"
        # the command goes in on stdin, quoted for eval in this shell (so cd and exports carry over): no script file
        # on a path the agent can write, which a background job of its could swap for a link
        self.proc.stdin.write(
            f"eval {_q(command)} < /dev/null\n__ocb_rc=$?\nprintf '\\n{marker} %s %s\\n' \"$__ocb_rc\" {_PWD}\n"
        )
        self.proc.stdin.flush()
        lines: list[str] = []
        deadline = time.monotonic() + (timeout or self.timeout)  # a deadline from the start, not an idle timeout
        hard = deadline + self.max_hold  # holds can't stretch it forever (e.g. a background loop of calls)
        while True:
            try:
                line = self._q.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                if self.hold() and time.monotonic() < hard:  # waiting on the broker (an approval, the gate)
                    deadline = time.monotonic() + 1.0
                    continue
                note = self._kill()  # the broker's word on calls already sent: their request ids, to retry
                out = "".join(lines) + (f"\n[{note}]" if note else "")
                return {"output": _truncate(out, self.max_output), "returncode": -1, "timed_out": True}
            if line is None:  # shell exited (e.g. the action ran `exit`)
                self.proc = None
                return {"output": _truncate("".join(lines), self.max_output), "returncode": -1, "timed_out": False}
            if line.startswith(marker + " "):
                _, rc, cwd = line.rstrip("\n").split(" ", 2)
                self.cwd = os.path.normpath(cwd) if WINDOWS else cwd
                out = "".join(lines)
                if out.endswith("\n"):
                    out = out[:-1]  # the newline printf put before the marker
                return {"output": _truncate(out, self.max_output), "returncode": int(rc), "timed_out": False}
            lines.append(line)

    def interrupt(self) -> None:
        """Ctrl-C: kill the running action, keep a fresh shell in the same cwd."""
        self._kill()

    def _kill(self) -> str | None:
        note = self.on_kill()
        if self.proc and self.proc.poll() is None:
            _kill_tree(self.proc.pid)
        self.proc = None
        self._start()
        return note

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            _kill_tree(self.proc.pid)


_PWD = '"$(pwd -W)"' if WINDOWS else '"$PWD"'  # Git Bash's $PWD is /c/...; pwd -W gives C:/...


def _bash() -> str:
    """The bash to start: on Windows, the first bash.exe on PATH outside the Windows directory (Git Bash)."""
    if not WINDOWS:
        return "bash"
    windir = os.path.normcase(os.environ.get("SYSTEMROOT", r"C:\Windows"))
    for d in os.environ.get("PATH", "").split(os.pathsep):
        exe = shutil.which("bash", path=d) if d else None
        if exe and not os.path.normcase(os.path.abspath(exe)).startswith(windir):
            return exe
    raise RuntimeError("no bash on PATH: install Git for Windows (Git Bash), or run in WSL2")


def _kill_tree(pid: int) -> None:
    """Kill the shell and everything it started: its process group, and every descendant, including those that
    left the group with setsid (on Linux the sandbox's PID namespace also takes them down with the shell)."""
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
        return
    procs = _descendants(pid)
    denied = []
    for p in [pid, *procs]:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(p, signal.SIGKILL)  # p's own group, if it leads one (a setsid child does)
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            denied.append(p)
    if denied:  # an isolated agent's processes, under another uid: only the root helper may kill them
        from onchainbench import isolation

        isolation.kill_pids(denied)


def _descendants(pid: int) -> list[int]:
    """Every process below pid, from one `ps` snapshot (children by parent pid)."""
    from onchainbench import isolation

    try:
        with isolation.harness():  # the harness's view of the process table, not an isolated agent's
            out = subprocess.run(
                ["ps", "-A", "-o", "pid=", "-o", "ppid="], check=False, capture_output=True, text=True, timeout=10
            )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no ps: the process group still goes
        return []
    children: dict[int, list[int]] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            found.append(c)
            todo.append(c)
    return found


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    half = n // 2
    return s[:half] + f"\n[... {len(s) - n} characters elided; narrow the command (head, grep, tail) ...]\n" + s[-half:]
