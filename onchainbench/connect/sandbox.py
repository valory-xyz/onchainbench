"""OS sandbox for the agent's shell.

macOS: Seatbelt (sandbox-exec), the same mechanism Claude Code uses there.
  - reads are denied for key material and for the real connect token files;
  - writes are allowed only in the working directory (which holds the tools' own caches) and temp dirs, and denied
    for a few paths under them that the harness reads as instructions (deny_write);
  - network stays open (the agent needs the web and the broker).
Linux (and Windows through WSL2): bubblewrap (`bwrap`), the mechanism Claude Code uses there.
  - the filesystem is mounted read-only, with the workspace (and the tools' caches in it) and temp dirs writable;
  - denied files are covered by /dev/null (reading them fails) and denied directories by an empty read-only tmpfs;
    a covered path is a mount point, so it can't be renamed or replaced either;
  - the network stays shared; the shell gets its own PID namespace and session and dies with the harness.
Elsewhere (native Windows): no sandbox, with a warning. The broker's policy still applies.

The sandbox is what turns "never read .mcp.json" from an instruction into a fact.
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path

HOME = Path.home()

DEFAULT_DENY_READ = [
    HOME / ".operate" / "keys",
    HOME / ".operate" / "wallets",
    HOME / ".operate" / "wallet_recovery",
    HOME / ".operate" / "user.json",
    HOME / ".ssh",
    HOME / ".aws",
    HOME / ".gnupg",
    HOME / ".config" / "gh",
    HOME / ".netrc",
    HOME / "Library" / "Keychains",
]

# What the shell may write besides its own directory: an allow-list of temp dirs. The user's caches (~/.cache,
# ~/Library/Caches) are not on it: code outside the sandbox runs from them later (uv hard-links installed packages
# from its cache, pip installs cached wheels, pre-commit runs its hook envs, pnpm, yarn, go and node-gyp reuse what
# they find), so a write there outlives the session. The agent's tools get one cache of their own in its directory
# instead (the workspace's install_bin: XDG_CACHE_HOME, which most tools follow, plus the variables of those that
# don't, such as npm's), the same for every arm.
DEFAULT_EXTRA_WRITE = [
    Path("/private/tmp"),  # macOS
    Path("/private/var/folders"),  # macOS
    Path("/tmp"),  # Linux (on macOS a link to /private/tmp)
    Path("/var/tmp"),
]


def _sb_str(p: Path | str) -> str:
    return '"' + str(p).replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(
    workspace: Path,
    deny_read: list[Path],
    extra_write: list[Path],
    deny_write: list[Path] | None = None,
    protect: list[Path] | None = None,
) -> str:
    rules = ["(version 1)", "(allow default)"]
    for raw in deny_read:
        p = Path(os.path.realpath(raw))
        kind = "subpath" if p.is_dir() else "literal"
        rules.append(f"(deny file-read* ({kind} {_sb_str(p)}))")
    rules.append('(deny file-write* (subpath "/"))')
    allow = [f"(subpath {_sb_str(os.path.realpath(workspace))})"]
    allow += [f"(subpath {_sb_str(os.path.realpath(p))})" for p in extra_write]
    allow += ['(literal "/dev/null")', '(literal "/dev/tty")', '(subpath "/dev/fd")', '(regex #"^/dev/ttys[0-9]+$")']
    rules.append("(allow file-write* " + " ".join(allow) + ")")
    ws = Path(os.path.realpath(workspace))
    for raw in protect or []:  # read-only even under a temp dir, with the dirs above it; the workspace stays writable
        p = Path(os.path.realpath(raw))
        rules.append(f"(deny file-write* (subpath {_sb_str(p)}))")
        rules += [f"(deny file-write* (literal {_sb_str(a)}))" for a in p.parents]
        if ws.is_relative_to(p):
            rules.append(f"(allow file-write* (subpath {_sb_str(ws)}))")
    for raw in deny_write or []:  # a file or a directory, existing or not: subpath covers both
        rules.append(f"(deny file-write* (subpath {_sb_str(os.path.realpath(raw))}))")
    # the workspace directory itself (not what's in it) can't be removed, renamed or replaced, e.g. by a symlink
    # that would make the next session's writable root somewhere else
    rules.append(f"(deny file-write* (literal {_sb_str(ws)}))")
    # Later rules win: a denied path, and every directory above it, can't be renamed or replaced, so a
    # literal-path read denial can't be dodged with `mv .mcp.json x && cat x`.
    for raw in deny_read:
        p = Path(os.path.realpath(raw))
        rules.append(f"(deny file-write* ({'subpath' if p.is_dir() else 'literal'} {_sb_str(p)}))")
        rules += [f"(deny file-write* (literal {_sb_str(a)}))" for a in p.parents]
    return "".join(rules)


def _anchor(p: Path, writable: list[Path]) -> Path | None:
    """The directory right below the innermost writable bind that holds p: bound read-only, it can't be renamed, and
    nothing under it can be created, replaced or moved (bind mounts stack; a mount point can't be renamed)."""
    for w in sorted(writable, key=lambda w: len(w.parts), reverse=True):
        if p != w and p.is_relative_to(w):
            return w / p.relative_to(w).parts[0]
    return None


def bwrap_args(
    workspace: Path,
    deny_read: list[Path],
    extra_write: list[Path],
    deny_write: list[Path] | None = None,
    protect: list[Path] | None = None,
) -> list[str]:
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    writable = [Path(os.path.realpath(raw)) for raw in extra_write]
    writable = [p for p in writable if p.exists()]
    # the temp dir tools use, when it is a directory inside a writable one ($TMPDIR=/tmp/user-1): bound on its own,
    # so it can't be renamed and an anchor is taken below it, never above it (that would make it read-only)
    tmp = Path(os.path.realpath(tempfile.gettempdir()))
    if tmp not in writable and any(tmp.is_relative_to(w) for w in writable):
        writable.append(tmp)
    for p in sorted(writable, key=lambda w: len(w.parts)):  # outer binds first: inner ones stack on top
        args += ["--bind", str(p), str(p)]
    # what lies under a writable temp dir and must not change (the protected root, denied paths and their parents):
    # its topmost directory there goes read-only; the workspace is bound writable again on top
    anchors = []
    for raw in [*(protect or []), *deny_read, *(deny_write or [])]:
        a = _anchor(Path(os.path.realpath(raw)), writable)
        if a is not None and a.exists() and a not in anchors:
            anchors.append(a)
            args += ["--ro-bind", str(a), str(a)]
    ws = Path(os.path.realpath(workspace))
    if ws.exists():
        args += ["--bind", str(ws), str(ws)]
    for raw in deny_write or []:  # read-only over the writable binds (a path that doesn't exist can't be covered)
        p = Path(os.path.realpath(raw))
        if p.exists():
            args += ["--ro-bind", str(p), str(p)]
    for raw in deny_read:  # after the binds: later mounts cover earlier ones
        p = Path(os.path.realpath(raw))
        if p.is_dir():
            args += ["--tmpfs", str(p), "--remount-ro", str(p)]  # empty and read-only: nothing can be written there
        elif p.exists():
            args += ["--ro-bind", "/dev/null", str(p)]
    return [*args, "--unshare-pid", "--new-session", "--die-with-parent", "--"]


def wrapper_for(
    workspace: Path,
    deny_read: list[Path] | None = None,
    extra_write: list[Path] | None = None,
    mode: str = "auto",
    deny_write: list[Path] | None = None,
    protect: list[Path] | None = None,
) -> tuple[list[str], str]:
    """Return (argv prefix, description). mode: auto | none | seatbelt | bwrap.

    workspace is the one directory the shell may write (besides temp dirs and caches); protect are directories
    (the Connect workspace root around it) that stay read-only, with everything under them but the workspace and
    the directories above them, even under a temp dir; deny_write are paths it may not write even where it
    otherwise could; deny_read are paths it may neither read nor write."""
    deny = [*DEFAULT_DENY_READ, *(deny_read or [])]
    write = [*DEFAULT_EXTRA_WRITE, *(extra_write or [])]
    deny_write = list(deny_write or [])
    if mode == "none":
        return [], "no sandbox (requested)"
    if mode in ("auto", "seatbelt") and platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", seatbelt_profile(workspace, deny, write, deny_write, protect)], "macOS Seatbelt"
    if mode in ("auto", "bwrap") and platform.system() == "Linux" and shutil.which("bwrap"):
        return bwrap_args(workspace, deny, write, deny_write, protect), "Linux bubblewrap"
    if mode in ("seatbelt", "bwrap"):
        raise RuntimeError(f"{mode} sandbox requested but it is not available on this machine")
    hint = " (install bubblewrap)" if platform.system() == "Linux" else " (on Windows, run the benchmark in WSL2)"
    return [], f"NO SANDBOX{hint}: the broker policy is the only guard"
