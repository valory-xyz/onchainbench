"""Onboarding polish. The end-of-run table (and `stats --table`), the validation verdict
of a scripted-arms run, progress-line cuts on a word boundary, `publish --help` once, and no LiteLLM banners."""

import json
import re

import pytest
from test_stop_and_wiring import TASK, rec, runner  # noqa: F401 - runner: the fixture

from onchainbench import llm
from onchainbench import publish as P
from onchainbench import run as R
from onchainbench import stats as S

BANNERS = ("Provider List", "Give Feedback / Get Help")
MODEL = "olas-ocb-agent:openrouter/deepseek/deepseek-v4.1-flash-1017"


def arm_stats(pass_mean, ci=None, infra=0.0, cost=0.0):
    return {
        "pass_mean": pass_mean,
        "ci95": ci,
        "infra_rate": infra,
        "infra_high": infra > S.INFRA_FLAG,
        "cost_usd": cost,
    }


def test_table_ranks_the_ranked_arms_and_marks_controls():
    summary = {
        "arms": {
            "oracle": arm_stats(1.0),
            MODEL: arm_stats(0.62, [0.512, 0.731], infra=0.02, cost=1.23456),
            "codex:openrouter/z/w": arm_stats(0.3, None, infra=0.25, cost=2.0),
            "null": arm_stats(0.0),
            # an older summary's reference baseline: not in the ranking, so no rank, whatever it is marked
            "mini-swe-agent-pearl:openrouter/x/y": arm_stats(0.4, [0.2, 0.6], cost=0.5) | {"reference": True},
        },
        # the summary ranks every arm, the scripted ones too: the table leaves those unranked
        "ranking": ["oracle", MODEL, "codex:openrouter/z/w", "null"],
    }
    lines = S.table(summary).splitlines()
    assert lines[0].split() == ["arm", "rank", "pass", "95%", "interval", "infra", "cost"]
    rows = [line.split() for line in lines[1:]]
    assert rows[0] == [MODEL, "1", "0.620", "[0.512,", "0.731]", "2.0%", "$1.2346"]
    assert rows[1] == ["codex:openrouter/z/w", "2", "0.300", "none", "25.0%", "(high)", "$2.0000"]
    assert [r[:2] for r in rows[2:]] == [
        ["oracle", "control"],
        ["null", "control"],
        ["mini-swe-agent-pearl:openrouter/x/y", S.NO_VALUE],
    ]
    assert len({line.index(" ", len(MODEL)) for line in lines}) == 1  # one column grid
    assert all(line == line.rstrip() for line in lines)
    assert S.table({"arms": {}}) == "arm  rank  pass  95% interval  infra  cost"


def test_stats_cli_prints_the_table_the_json_or_the_noise_baseline(tmp_path, capsys):
    p = tmp_path / "trials.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [rec(), rec(arm="null", passed=False)]) + "\n\n")
    assert S.main([str(p), "--table"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("arm") and "control" in out and "{" not in out
    assert S.main([str(p)]) == 0
    assert json.loads(capsys.readouterr().out)["arms"]["oracle"]["pass_mean"] == 1.0
    assert S.main(["--noise", str(p), str(p)]) == 0
    assert "{" in capsys.readouterr().out


def test_clip_cuts_on_a_word_boundary_with_an_ellipsis():
    assert R.clip("short", 90) == "short"
    reason = "integrity: state changed outside Connect: balance of 0xabc on gnosis, nonce of the Safe, and more besides"
    cut = R.clip(reason, 90)
    assert len(cut) <= 90 and cut.endswith("…") and reason.startswith(cut[:-1])
    assert reason[len(cut) - 1] in " ,"  # ended where a word ended
    assert R.clip("a" * 100, 10) == "a" * 9 + "…"  # no space to cut on (an arm spec): cut mid-word, still marked
    assert R.clip(MODEL, R.ARM_WIDTH) == MODEL[:39] + "…"
    assert R.clip(None, 90) == "None" and R.clip("abc", 0) == "" and R.clip("abc", 1) == "…"  # odd input: no raise


@pytest.mark.parametrize("missing", ["pass_mean", "ci95", "infra_rate", "cost_usd"])
def test_table_shows_a_dash_for_a_missing_field(missing):
    s = arm_stats(0.5, [0.4, 0.6], infra=0.01, cost=1.0)
    del s[missing]
    row = re.split(r"\s{2,}", S.table({"arms": {MODEL: s}, "ranking": [MODEL]}).splitlines()[1])
    cells = {"pass_mean": 2, "ci95": 3, "infra_rate": 4, "cost_usd": 5}
    assert row[cells[missing]] == S.NO_VALUE and row.count(S.NO_VALUE) == 1


def test_table_never_raises_on_odd_summaries():
    odd = {
        "hermes:m": {"connect_tools_listed_share": 0},  # a stub: no field at all, not ranked
        "codex:m": {"pass_mean": "x", "ci95": [0.1], "infra_rate": None, "cost_usd": None},  # not numbers
        "openclaw:m": None,  # not a summary at all
        "hermes-raw:m": {"pass_mean": None, "ci95": 5, "infra_rate": "1", "infra_high": True, "cost_usd": "$"},
    }
    lines = S.table({"arms": odd, "ranking": ["codex:m", "gone"]}).splitlines()
    assert [line.split()[:2] for line in lines[1:]] == [
        ["codex:m", "1"],
        ["hermes:m", "—"],
        ["openclaw:m", "—"],
        ["hermes-raw:m", "—"],
    ]
    assert all(line.split()[2:] == ["—"] * 4 for line in lines[1:])
    assert S.table({}) == S.table({"arms": None, "ranking": None})


def test_table_of_an_arm_with_no_scored_trial():
    infra = rec(arm=MODEL, passed=False, infra=True, reason="URLError: Connection refused", cost_usd=None)
    summary = S.summarize([rec(), infra])  # a real summary: the arm's only trial was lost to infra
    assert summary["arms"][MODEL]["tasks"] == 0
    assert S.table(summary).splitlines()[1].split() == [MODEL, "1", "0.000", "none", "100.0%", "(high)", "$0.0000"]


def test_validation_verdict_only_for_scripted_runs():
    ok = [rec(), rec(arm="null", passed=False, reason="nothing happened")]
    assert R.validation_verdict(ok) == "validation: valid (2 trials came out as the scripted arms must)"
    bad = R.validation_verdict([rec(), rec(arm="null")])
    assert bad.splitlines() == ["validation: 1 problems", f"  null {TASK}: passed"]
    assert R.validation_verdict([*ok, rec(arm=MODEL)]) is None  # a model arm ran: its FAILs are results
    assert R.validation_verdict([]) is None
    assert R.validation_verdict([{"task": TASK}]) is None  # no arm: not a scripted run
    odd = R.validation_verdict([{"arm": "oracle"}])  # a record validate_run can't read: said, not raised
    assert odd.startswith("validation: not checked here (KeyError") and "scripts/validate_run.py" in odd


def test_validation_verdict_needs_the_scripts_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "VALIDATE_RUN", tmp_path / "missing.py")  # an installed package: no scripts/
    assert R.validation_verdict([rec()]) is None


def test_a_scripted_run_ends_with_the_table_the_json_and_the_verdict(runner, monkeypatch, capsys):  # noqa: F811
    long = "the null arm did nothing at all, so the wrapped balance never moved and the check found no transfer here"

    def trial(arm, task, k, envs, seed, sessions, tag):
        return rec(arm=arm, attempt=k) if arm == "oracle" else rec(arm=arm, attempt=k, passed=False, reason=long)

    monkeypatch.setattr(R, "trial", trial)
    assert R.main(["--arms", "oracle,null", "--tasks", TASK, "--run", "t", "--no-judges"]) == 0
    out = capsys.readouterr().out
    fail = next(line for line in out.splitlines() if " FAIL " in line)
    assert fail.endswith("…") and long not in fail and fail.rsplit("  ", 1)[1][:-1] in long
    head = out.index("arm  ")
    assert out.index("oracle", head) < out.index("{", head)  # the table before the JSON
    assert out.rstrip().endswith("validation: valid (2 trials came out as the scripted arms must)")


def test_publish_help_prints_the_usage_once_and_exits_zero(capsys):
    assert P.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert out.count("Held-out runs never publish") == 1 and out.count("held-out") == 3
    assert out.count("python -m onchainbench.publish <run>") == 1
    assert P.main(["-h"]) == 0 and capsys.readouterr().out == out
    assert P.main([]) == 2 and capsys.readouterr().out == out  # no run: the usage, as a refusal


def test_litellm_banners_are_off_and_errors_still_raise(monkeypatch, capsys):
    import litellm

    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    with pytest.raises(Exception) as e:  # an unknown provider: LiteLLM refuses before any request, nothing is paid
        llm.complete("no-such-provider-model", [{"role": "user", "content": "hi"}], max_tokens=1)
    assert "LLM Provider NOT provided" in str(e.value)  # the error itself still reaches the caller
    out = capsys.readouterr()
    assert not any(b in out.out + out.err for b in BANNERS)
    assert litellm.suppress_debug_info is True


@pytest.mark.parametrize("entry", ["pricing", "mech_backend"])
def test_every_litellm_entry_point_turns_the_banners_off(entry, monkeypatch, tmp_path):
    """Every use of LiteLLM left in the bench's process: pricing, and the chat mech's model for olas-ocb-agent-mech
    (every contestant's own LiteLLM runs in its CLI's process)."""
    import litellm

    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    if entry == "pricing":
        from onchainbench import pricing

        pricing.register_price("openrouter/unknown")
    else:
        from onchainbench import ocb_cli

        ocb_cli.mech_backend("openrouter/m", [0.0], [])  # set up, no call made
    assert litellm.suppress_debug_info is True
