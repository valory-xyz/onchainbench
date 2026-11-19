"""The auditable-surface metric per arm (METHODOLOGY R5.5): how much trusted code a user must
trust, and how much can talk to the model.

    python -m onchainbench.surface        # prints the table (measured where the arm is installed here)

Columns:
- trusted_loc: lines of the harness's own code (Python for olas-ocb-agent; the shipped JS bundle for npm arms).
  Bundles include their vendored dependencies, so they overstate first-party code: read them as "code
  you'd have to audit", not "code the vendor wrote";
- deps: installed distributions or packages it pulls in;
- tools_in_prompt: tool schemas the model sees in an eval run;
- third_party: components from outside the harness vendor in the agent's path to the wallet;
- token_readable_at: places a key or token is readable by the agent's own shell in the eval setup.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the onchainbench repo root
VENDOR = Path(__file__).resolve().parents[1] / "vendor"  # the repo root's vendor/


def loc(paths: list[Path], suffixes=(".py", ".js", ".mjs", ".cjs", ".ts")) -> int:
    n = 0
    for base in paths:
        files = (
            [base]
            if base.is_file()
            else [
                p for p in base.rglob("*") if p.suffix in suffixes and "node_modules" not in p.parts[len(base.parts) :]
            ]
        )
        for f in files:
            try:
                n += sum(1 for line in f.read_text(errors="ignore").splitlines() if line.strip())
            except OSError:
                continue
    return n


def pip_closure(name: str) -> int:
    seen, todo = set(), [name]
    while todo:
        n = todo.pop().lower().replace("_", "-")
        if n in seen:
            continue
        try:
            reqs = metadata.requires(n) or []
        except metadata.PackageNotFoundError:
            continue
        seen.add(n)
        for r in reqs:
            if "extra ==" in r:
                continue
            todo.append(
                r.split(";")[0]
                .split("[")[0]
                .split(" ")[0]
                .split(">")[0]
                .split("<")[0]
                .split("=")[0]
                .split("!")[0]
                .split("~")[0]
                .strip()
            )
    return len(seen) - 1


def npm_deps(pkg_dir: Path) -> int | None:
    lock = VENDOR / "package-lock.json"
    if not lock.is_file():
        return None
    pk = json.loads(lock.read_text()).get("packages", {})
    name = "node_modules/" + pkg_dir.relative_to(VENDOR / "node_modules").as_posix()
    todo, seen = [name], set()
    while todo:
        n = todo.pop()
        if n in seen or n not in pk:
            continue
        seen.add(n)
        for d in pk[n].get("dependencies") or {}:
            todo.append(f"node_modules/{d}")
    return len(seen) - 1


def agent_dir() -> Path:
    from onchainbench.freeze import agent_dir as frozen_agent_dir

    return frozen_agent_dir()


def table() -> list[dict]:
    try:
        agent = agent_dir()
    except FileNotFoundError:  # a public checkout: the agent isn't installed, so it isn't measured
        agent = None
    rows = [
        {
            "arm": "olas-ocb-agent",
            "trusted_loc": loc([agent]) if agent else None,
            "deps": pip_closure("litellm") + 1 if agent else None,
            "tools_in_prompt": 1,
            "third_party": "LiteLLM (model client)",
            "token_readable_at": "none (Seatbelt/bwrap deny .mcp.json and keys; the shell gets the broker's capped token)",  # noqa: E501 - a literal command or config line
        }
    ]
    for arm, pkg, tools, third, token in [
        (
            "codex",
            "@openai/codex",
            "shell, apply_patch + 13 Connect MCP tools",
            "Connect MCP",
            "CODEX_HOME config is outside the sandbox's writable roots but readable",
        ),
        (
            "openclaw",
            "openclaw",
            "coding profile (~20) + 13 Connect MCP tools",
            "Connect MCP via the bench's broker",
            "run.json (the broker's capped token) and the environment: the shell is unsandboxed",
        ),
        (
            "evm-mcp-server",
            "@mcpdotdirect/evm-mcp-server",
            "22 EVM tools",
            "viem, public chain registry",
            "the raw private key in the MCP server's environment",
        ),
    ]:
        d = VENDOR / "node_modules" / pkg
        rows.append(
            {
                "arm": arm,
                "trusted_loc": loc([d]) if d.is_dir() else None,
                "deps": npm_deps(d) if d.is_dir() else None,
                "tools_in_prompt": tools,
                "third_party": third,
                "token_readable_at": token,
            }
        )
    rows.append(
        {
            "arm": "claude-code",
            "trusted_loc": None,
            "deps": None,
            "tools_in_prompt": "Bash, Read, Write, Edit, Glob, Grep + 13 Connect MCP tools",
            "third_party": "Connect MCP",
            "token_readable_at": "workspace .mcp.json (Connect's own token) unless a sandbox is configured",
        }
    )
    return rows


if __name__ == "__main__":  # pragma: no cover
    json.dump(table(), sys.stdout, indent=1)
    print()
