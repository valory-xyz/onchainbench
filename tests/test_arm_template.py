"""The harness template in examples/arm-template: a session class around a stand-in CLI that meets the CLI
contract. It runs the CLI once per turn, resumes it, reads its JSON into stats and a transcript, and maps its errors.
A contributor's copy starts from code that works."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from onchainbench import arms as A
from onchainbench import failures as F

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stand-in CLI is a script with a shebang")
DIR = Path(__file__).resolve().parents[1] / "examples" / "arm-template"


def load():
    spec = importlib.util.spec_from_file_location("my_harness_arm", DIR / "my_harness_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trial(tmp_path):
    ws = tmp_path / "pe2-x" / "ws"
    ws.mkdir(parents=True)
    return types.SimpleNamespace(ws=ws, data={})


def stub(tmp_path, body):
    p = tmp_path / "cli"
    p.write_text(f"#!{sys.executable}\n{body}")
    p.chmod(0o755)
    return str(p)


@pytest.fixture
def cli(tmp_path):
    """The stand-in, run by this test's Python (its shebang names whatever python3 is on PATH)."""
    return stub(tmp_path, (DIR / "my-harness").read_text())


def test_the_template_runs_resumes_and_records_a_transcript(tmp_path, cli, monkeypatch):
    monkeypatch.delenv("OPENROUTER_PROVISIONING_KEY", raising=False)
    m = load()
    s = m.MyHarnessSession(trial(tmp_path), "openrouter/m", 500, cli)
    assert isinstance(s, A.Session) and s.send("abc") == "cba" and s.send("hello") == "olleh"
    assert s.stats.model_calls == 2 and s.stats.prompt_tokens == 8 and s.stats.completion_tokens == 8
    events, raw = s.transcript()
    assert [(e["turn"], e["kind"]) for e in events] == [(0, "user"), (0, "assistant"), (1, "user"), (1, "assistant")]
    assert len(raw) == 2 and len(list((s.t.ws.parent / "my-harness-state").iterdir())) == 1  # one resumed session
    s.stats.cost_usd = A.TRIAL_CAP_USD
    assert s.send("more") == A.CAP_REACHED and s.close() is s.stats


def test_the_template_maps_the_cli_s_errors(tmp_path, monkeypatch):
    m = load()
    t = trial(tmp_path)
    s = m.MyHarnessSession(t, "m", 500, stub(tmp_path, 'import sys; sys.stderr.write("RateLimitError"); sys.exit(1)\n'))
    assert s.send("x").kind == F.PROVIDER_ERROR
    out = '{"reply": "", "session_id": "s", "error": "harness"}'
    s = m.MyHarnessSession(t, "m", 500, stub(tmp_path, f"print('{out}')\n"))
    assert s.send("x").kind == F.HARNESS_ERROR


def test_the_template_has_the_bench_s_one_wall_clock(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(A, "TURN_SECONDS", 1)
    s = m.MyHarnessSession(trial(tmp_path), "m", 500, stub(tmp_path, "import time; time.sleep(10)\n"))
    assert s.send("slow").kind == F.TIMEOUT
    assert [e["kind"] for e in s.transcript()[0]] == ["user", "error"]
