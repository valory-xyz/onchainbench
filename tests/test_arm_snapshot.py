"""What every model arm's CLI is given, pinned: on a fake trial, one turn and the close of each arm kind, with every
subprocess the arm starts recorded (its command line, its environment's keys and non-secret values, its cwd), the
config files it writes, the secrets it registers, and the runner's step-limit and cap-enforcement strings. A change
to any of them shows up here as a diff of tests/fixtures/arm_snapshot.json.

Regenerate the snapshot (only for an intended change): ONCHAINBENCH_UPDATE_ARM_SNAPSHOT=1 pytest
tests/test_arm_snapshot.py."""

import json
import os
import re
import subprocess
import types
from pathlib import Path

import pytest

from onchainbench import arms as A
from onchainbench import pricing
from onchainbench import run as R
from onchainbench.pricing import SECRET_ENV

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the snapshot's paths are POSIX paths")

SNAPSHOT = Path(__file__).parent / "fixtures" / "arm_snapshot.json"
UPDATE = bool(os.environ.get("ONCHAINBENCH_UPDATE_ARM_SNAPSHOT"))  # read before the test replaces the environment
MODEL = "openrouter/vendor/m"
ARMS = [
    "claude-code",
    f"claude-code:{MODEL}",
    f"codex:{MODEL}",
    f"hermes:{MODEL}",
    f"hermes-raw:{MODEL}",
    f"openclaw:{MODEL}",
    f"openclaw-raw:{MODEL}",
    f"olas-ocb-agent:{MODEL}",
    f"olas-ocb-agent-mech:{MODEL}",
]
ENV = {  # the runner's whole environment for the test: fake keys, nothing of the machine's
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/runner",
    "LITELLM_MODE": "PRODUCTION",
    "OPENROUTER_API_KEY": "sk-or-fake",
    "ANTHROPIC_API_KEY": "sk-ant-fake",
    "FORK_RPC_GNOSIS": "https://rpc.example/v2/aaaaaaaaaaaaaaaaaaaaaaaa",
}
PRICE = {"input": 1e-7, "output": 2e-7, "cache_read": 5e-8, "cache_write": 1e-7}


def _fake_t(root: Path):
    ws = root / "ws"
    ws.mkdir(parents=True)
    (ws / ".mcp.json").write_text("{}")
    (ws / "AGENTS.md").write_text("the brief\n")
    st = types.SimpleNamespace(
        eoa_key="0xk",
        fc=types.SimpleNamespace(chain_id=100),
        rpc_url="http://127.0.0.1:8545",
        mech_amount=lambda hundredths: hundredths * 10,
        llm_backend=None,
        prepaid=0,
    )
    fc = types.SimpleNamespace(
        mcp_json=lambda: {
            "mcpServers": {
                "pearl-connect": {"url": "http://127.0.0.1:7000/mcp/", "headers": {"Authorization": "Bearer ctok"}}
            }
        },
        s=st,
    )
    data = {"proxy_env": {"HTTPS_PROXY": "http://127.0.0.1:3128", "HTTP_PROXY": "http://127.0.0.1:3128"}}
    return types.SimpleNamespace(ws=ws, fc=fc, data=data, chain="gnosis")


def _norm(v, root: Path):
    if isinstance(v, dict):
        return {k: _norm(x, root) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_norm(x, root) for x in v]
    if isinstance(v, str):
        v = v.replace(str(root.resolve()), "<tmp>").replace(str(root), "<tmp>")
        return re.sub(r"ocb-[0-9a-f]{32}", "ocb-<uuid>", v)
    return v


def _env(env: dict, root: Path) -> dict:
    return {k: ("<secret>" if v and SECRET_ENV.search(k) else _norm(v, root)) for k, v in sorted(env.items())}


def _secret_free(text: str) -> str:
    for v in ("sk-or-fake", "sk-ant-fake", "ctok", "btok"):
        text = text.replace(v, "<secret>")
    return text


def record(arm: str, root: Path, monkeypatch) -> dict:
    calls = []

    def run(cmd, **kw):
        calls.append(
            {
                "via": "run",
                "argv": _norm([str(c) for c in cmd], root),
                "cwd": _norm(str(kw.get("cwd", "")), root),
                "timeout": None if kw.get("timeout") is None else round(kw["timeout"]),  # Hermes's: what's left
                "env": _env(kw["env"], root) if kw.get("env") is not None else None,
            }
        )
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    class Popen:
        def __init__(self, cmd, **kw):
            calls.append({"via": "Popen", "argv": [str(c) for c in cmd]})

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def communicate(self, *a, **k):
            return "", ""

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", Popen)
    t = _fake_t(root)
    session = A.open_session(arm, t, [], A.STEP_LIMIT)
    session.send("do the task")
    session.close()
    files = {}
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root).as_posix()
        if f.is_file() and not rel.startswith("ws/") and f.suffix in (".toml", ".yaml", ".json", ".md"):
            files[rel] = _norm(_secret_free(f.read_text()), root)
    return {
        "calls": calls,
        "files": files,
        "named_secrets": sorted(t.data.get("named_secrets", {})),
        "step_limit": R.step_limit_enforcement(arm),
        "cap": R.cap_enforcement(arm),
    }


def test_every_arm_s_command_line_and_environment_match_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(ENV))
    monkeypatch.setattr(pricing, "_PRICES", {MODEL: PRICE})
    monkeypatch.setattr(A, "_find_bin", lambda kind: f"/opt/bin/{A.CLI_BINS[kind][0]}")
    monkeypatch.setattr("onchainbench.raw_key.ensure_patched", lambda: Path("/opt/evm-mcp/index.js"))
    monkeypatch.setattr("onchainbench.raw_key.broker_for", lambda t: ("http://127.0.0.1:7100/mcp/", "btok"))
    got = {arm: record(arm, tmp_path / str(i), monkeypatch) for i, arm in enumerate(ARMS)}
    assert all(c["argv"] for r in got.values() for c in r["calls"])
    if UPDATE:
        SNAPSHOT.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    assert got == json.loads(SNAPSHOT.read_text())
