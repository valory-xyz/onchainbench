"""The trial workspace as the harness sees it: where the agent's shell starts, what it must not read, and the `pearl`
command on its PATH. The benchmark's own copy: no agent's prompt, skills or trust store, only Connect's
plumbing, so the scripted arms and every non-agent harness run without any contestant's code."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

AGENT_SUBDIR = "agent"  # the agent's working dir inside a Connect workspace; checkers read the agent's files here
STATE_DIR = ".harness"  # the harness's own files in the agent dir (bin, tmp)
# The cache variables the agent's shell gets, each a directory of its own under STATE_DIR/cache: XDG_CACHE_HOME, which
# most tools follow; uv, pip and pre-commit, which don't on macOS; npm's (~/.npm otherwise)
CACHE_VARS = ("XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "PRE_COMMIT_HOME", "npm_config_cache")
# the dir that makes this package importable, whatever the package is called
PKG_ROOT = Path(__file__).resolve().parents[len(__package__.split("."))]

INSTRUCTION_PATHS = ("AGENTS.md", "CLAUDE.md", ".agents", ".claude")  # under the workspace root


def safe_dir(base: Path, *parts: str) -> Path:
    """base/parts, created as real directories: a symlink planted on the way (the agent can write base) is refused,
    so the harness never writes through it."""
    d = base
    for part in parts:
        d = d / part
        if d.is_symlink():
            raise RuntimeError(f"{d} is a symlink: refusing to write the harness's files through it")
        d.mkdir(exist_ok=True)
    return d


def write_file(path: Path, text: str, mode: int = 0o600) -> None:
    """Write path as a new file, in the same directory then renamed over it: a symlink at path is replaced, never
    followed, and the mode is set on the new file before it gets its name."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    if hasattr(os, "fchmod"):
        os.fchmod(fd, mode)
    else:  # pragma: no cover - Windows (Python < 3.13)
        os.chmod(tmp, mode)
    from onchainbench import isolation

    isolation.share_fd(fd)  # under isolation, the chmod emptied the slot's ACL mask: the agent must read what it gets
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


@dataclass
class Workspace:
    root: Path  # the trial's Connect persistent_data
    agent_dir: Path  # where the agent's shell starts; holds the shim .mcp.json
    connect: bool  # root has a real connect .mcp.json
    deny_read: list = field(default_factory=list)
    # what the agent may not write even where it may write: the brief and the skills, read as instructions
    deny_write: list = field(default_factory=list)
    bin_dir: Path | None = None

    @classmethod
    def open(cls, root: Path, agent_subdir: str = AGENT_SUBDIR) -> Workspace:
        root = root.resolve()
        connect = (root / ".mcp.json").is_file()
        agent_dir = root / agent_subdir if connect else root
        if connect and (agent_dir.is_symlink() or agent_dir.resolve() != agent_dir):
            # the agent can write its directory: a link in its place would move the next session's writable root
            raise RuntimeError(f"{agent_dir} is a symlink (or under one): refusing to use it as the agent's directory")
        agent_dir.mkdir(parents=True, exist_ok=True)
        deny = [root / ".mcp.json", root / ".codex" / "config.toml"] if connect else []
        # the workspace brief and skills are the next session's instructions: never the agent's to rewrite
        protect = [root / n for n in INSTRUCTION_PATHS] if connect else []
        return cls(root=root, agent_dir=agent_dir, connect=connect, deny_read=deny, deny_write=protect)

    def upstream_mcp(self) -> tuple[str, str]:
        """The real connect URL and token: read by the harness, never by the agent."""
        from .connect_client import find_mcp_config

        return find_mcp_config(self.root)

    def install_shim(self, shim: dict) -> None:
        write_file(self.agent_dir / ".mcp.json", json.dumps(shim, indent=2), 0o600)
        gi = self.agent_dir / ".gitignore"
        if gi.is_symlink() or not gi.exists():
            write_file(gi, f".mcp.json\n{STATE_DIR}/\n", 0o644)

    def install_bin(self) -> dict:
        """Put `pearl` on the agent's PATH; return env additions for the shell."""
        self.bin_dir = safe_dir(self.agent_dir, STATE_DIR, "bin")
        cache = safe_dir(self.agent_dir, STATE_DIR, "cache")
        write_file(
            self.bin_dir / "pearl",
            f'#!/bin/sh\nPYTHONPATH="{PKG_ROOT}" exec "{sys.executable}" -m {__package__}.pearl_cli "$@"\n',
            0o755,
        )
        return {
            "PATH": f"{self.bin_dir}:{os.environ.get('PATH', '')}",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "PIP_NO_INPUT": "1",
            # caches of its own: the user's are read-only in the sandbox (connect/sandbox.py's allow-list). Most tools
            # follow XDG_CACHE_HOME (pnpm, yarn, go, huggingface, pip and uv on Linux); the rest by their own variable
            **{k: str(cache / k.lower()) for k in CACHE_VARS},
        }
