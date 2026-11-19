"""The bench's wiring found by the open-run-1 transcript audit: one policy constructor for every bench broker, no
caps on any arm, spend counted only for calls that were sent, partial streams kept on a timeout, and the records."""

from __future__ import annotations

import json
import re
import types

import pytest

from onchainbench import arms as A
from onchainbench import raw_key
from onchainbench.connect.broker import Broker, rest_error
from onchainbench.connect.connect_client import ConnectClient, ConnectError
from onchainbench.connect.policy import UNCAPPED, Decision, Policy, SessionCaps, not_sent
from onchainbench.connect.session import make_policy, start_broker
from onchainbench.mock_connect.server import LOST_RESPONSE, PRED_MECH, SAFE, MockConnect, MockState

TX = {"chain": "gnosis", "target": "0x" + "ab" * 20, "value": 10**18, "data": "0x"}


@pytest.fixture
def mockc():
    m = MockConnect(MockState())
    yield m
    m.stop()


# ---- one constructor: the CLI arms' broker knows mechs and the wallet's chain ---------------------------------------
def test_make_policy_reads_the_wallets_chain_and_the_mechs_payment_asset(mockc):
    p = make_policy(ConnectClient(mockc.url, mockc.token), SessionCaps(max_mech_usd=5.0))
    assert p.default_chain == "gnosis" and p.mech_asset_lookup("gnosis", PRED_MECH) == (18, 1.0)
    assert "wallet_info" in p.read_only
    req = {"prompt": "q", "tool": "prediction-online", "max_payment": 10**16, "priority_mech": PRED_MECH}
    assert p.check("mech_request", req).allowed  # open-run-1: "the payment asset of mech ... isn't known"


def test_the_cli_arms_broker_uses_the_shared_constructor(mockc, tmp_path, monkeypatch):
    built = []
    real = make_policy

    def spy(upstream, caps, approval="auto", approver=None):
        built.append(caps)
        return real(upstream, caps, approval, approver)

    monkeypatch.setattr("onchainbench.connect.session.make_policy", spy)
    mcp = {"mcpServers": {"pearl-connect": {"url": mockc.url, "headers": {"Authorization": f"Bearer {mockc.token}"}}}}
    fc = types.SimpleNamespace(mcp_json=lambda: mcp, _log=lambda e: None, s=types.SimpleNamespace(rug_pull=None))
    fc.poisoned_listing = lambda: {}
    t = types.SimpleNamespace(fc=fc, chain="gnosis", data={}, ws=tmp_path)
    url, token = raw_key.broker_for(t)
    try:
        assert built == [A.EVAL_CAPS] and token != mockc.token
        c = ConnectClient(url, token)
        out = c.call_tool("mech_request", {"prompt": "q", "tool": "prediction-online", "priority_mech": PRED_MECH})
        assert out  # no max_payment, no cap: the broker passes it to Connect, as for an arm holding the token
        assert mockc.token not in (tmp_path / ".mcp.json").read_text()  # the real token stays hidden
    finally:
        for fn in t.data["cleanup"]:
            fn()


# ---- uncapped: the broker passes every call on, Connect's rules decide ------------------------------------------------
def test_uncapped_policy_refuses_nothing_but_still_asks_the_user():
    p = Policy(mode="auto", caps=UNCAPPED)
    big = {**TX, "value": 10**30, "chain": "nochain"}  # no price known, a huge value: Connect decides
    assert p.check("safe_transaction", big).allowed
    assert p.check("sign_message", {"digest": "0x" + "00" * 32}).allowed
    assert p.check("mech_request", {"prompt": "q"}).allowed  # no mech, no max_payment
    assert p.check("some_new_tool", {}).allowed
    assert p.check("safe_transaction", {**TX, "value": "-1"}).allowed  # Connect validates amounts itself
    assert p.value_calls == 5 and p.spent_native == 0.0
    asked = Policy(mode="ask", caps=UNCAPPED, approver=lambda s: False)
    assert asked.check("safe_transaction", TX).rule == "broker: the user declined this call"
    assert not Policy(mode="deny", caps=UNCAPPED).check("safe_transaction", TX).allowed


def test_a_policy_never_changes_a_shared_caps_object():
    p = Policy(mode="auto", caps=UNCAPPED)
    p.caps.allow_sign_message = False
    assert UNCAPPED.allow_sign_message and A.EVAL_CAPS is UNCAPPED and A.EVAL_CAPS.uncapped
    assert not SessionCaps().uncapped and not SessionCaps(None, None, None).uncapped


# ---- count only what was sent ---------------------------------------------------------------------------------------
def test_release_gives_back_a_calls_count_once():
    p = Policy(mode="auto", caps=SessionCaps(max_native_usd=100.0, max_value_calls=2))
    d = p.check("safe_transaction", {**TX, "request_id": "r1"})
    assert d.allowed and p.value_calls == 1 and p.spent_native > 0
    p.release(d)
    p.release(d)  # once only
    assert p.value_calls == 0 and p.spent_native == pytest.approx(0.0) and not p.approved
    p.release(Decision(True))  # nothing counted: nothing to give back
    assert p.value_calls == 0
    assert p.check("safe_transaction", {**TX, "request_id": "r1"}).counted is not None  # checked and counted again


def test_not_sent_matches_connects_refusals_before_sending_only():
    assert not_sent("intent: spend exceeds the declared amount. Nothing was sent.")
    assert not_sent("guardrail: rule 'whitelist' (0xab is not on the operator's whitelist)")
    assert not not_sent(LOST_RESPONSE)
    assert not not_sent("transaction 0x12 reverted: boom") and not not_sent(None)
    # a contract's revert reason is its own text: one that reads like the gate is still a mined (sent) call
    assert not not_sent("transaction 0xab12 reverted: Nothing was sent.")
    assert not not_sent("transaction 0xab12 reverted: guardrail: rule 'x'")
    assert not not_sent("the gate said: Nothing was sent. but more followed")  # only Connect's suffix, at the end


def test_a_mined_revert_whose_reason_reads_like_the_gate_stays_counted(mockc, monkeypatch):
    b, c = _value_broker(mockc, SessionCaps(max_native_usd=100.0))
    try:
        reverted = {"content": [{"type": "text", "text": "transaction 0xab reverted: Nothing was sent."}]}
        monkeypatch.setattr(b.upstream, "call_tool_raw", lambda name, args: {**reverted, "isError": True})
        assert c.call_tool_raw("safe_transaction", {**TX, "request_id": "rv"})["isError"]
        assert b.policy.value_calls == 1  # sent and mined: counted
    finally:
        b.stop()


def test_a_lost_response_reads_the_same_on_every_transport_and_gives_no_advice(mockc):
    """The lost-response fault once told MCP callers to retry with the same request_id and REST callers nothing:
    a hint only some arms got. Every path now carries the same words, and they state the fact only."""
    b, viab = _value_broker(mockc, SessionCaps(max_native_usd=100.0))
    direct = ConnectClient(mockc.url, mockc.token)
    seen = {}
    try:
        for name, client in (("direct", direct), ("broker", viab)):
            mockc.s.drop_first_response, mockc.s.dropped = {"safe_transaction"}, set()
            res = client.call_tool_raw("safe_transaction", {**TX, "value": 1, "request_id": f"{name}-mcp"})
            assert res["isError"]
            seen[f"{name} mcp"] = res["content"][0]["text"]
            mockc.s.dropped = set()
            with pytest.raises(ConnectError) as e:
                client.rest("POST", "/safe-transaction", {**TX, "value": 1, "request_id": f"{name}-rest"})
            seen[f"{name} rest"] = str(e.value)
            assert (e.value.status, e.value.detail) == (504, LOST_RESPONSE)
        assert b.policy.value_calls == 2  # the two through the broker: each may have landed, so both stay counted
    finally:
        b.stop()
    # the body is word for word the same; REST adds only its status line, as it does for every error. The broker
    # passes Connect's 504 on: a 400 of its own would read as a rejected request, the opposite of "may have"
    assert seen["direct mcp"] == seen["broker mcp"] == LOST_RESPONSE
    assert seen["direct rest"] == seen["broker rest"] == f"HTTP 504: {LOST_RESPONSE}"
    assert not not_sent(LOST_RESPONSE)
    # neutral: the fact, with no advice a path could lack
    assert not re.search(
        r"\b(?:retry|again|resend|request_id|same|should|check|timed? ?out|landed)\b", LOST_RESPONSE, re.I
    )
    assert "lost" in LOST_RESPONSE and "may or may not" in LOST_RESPONSE


def test_the_broker_passes_connects_status_on_and_keeps_400_for_its_own_refusals(mockc):
    """Connect's HTTP status and detail reach a REST caller unchanged; the broker's own answers (a policy refusal,
    an unreachable Connect, a crash) stay 400 with the broker's text."""
    assert rest_error(ConnectError("HTTP 504: x", status=504, detail="x")) == (504, {"detail": "x"})
    assert rest_error(ConnectError("connect server unreachable: refused")) == (
        400,
        {"detail": "connect server unreachable: refused"},
    )
    assert rest_error(ValueError("bad")) == (400, {"detail": "broker: could not handle the request: ValueError: bad"})
    b, c = _value_broker(mockc, SessionCaps(max_native_usd=100.0, max_value_calls=0))
    try:
        with pytest.raises(ConnectError) as e:
            c.rest("POST", "/safe-transaction", {**TX, "value": 1, "request_id": "over"})
        assert e.value.status == 400 and b.policy.value_calls == 0  # the broker's own refusal: never sent
        b.upstream.rest = lambda *a, **k: (_ for _ in ()).throw(ConnectError("HTTP 503: busy", 503, "busy"))
        with pytest.raises(ConnectError) as e:
            c.rest("GET", "/wallet")
        assert (e.value.status, e.value.detail, str(e.value)) == (503, "busy", "HTTP 503: busy")
    finally:
        b.stop()


def _value_broker(mockc, caps):
    up = ConnectClient(mockc.url, mockc.token)
    b = Broker(up, make_policy(up, caps)).start()
    return b, ConnectClient(b.url, b.token)


def test_a_call_connect_refused_unsent_isnt_counted_over_mcp_and_rest(mockc):
    """open-run-1 (openclaw x-mode-wrap): a gate refusal counted its $60 and the corrected send hit the cap."""
    b, c = _value_broker(mockc, SessionCaps(max_native_usd=100.0, max_value_calls=1))
    try:
        mockc.s.guardrail_mode = "whitelist"
        res = c.call_tool_raw("safe_transaction", {**TX, "request_id": "a"})
        assert res["isError"] and b.policy.value_calls == 0 and b.policy.spent_native == pytest.approx(0.0)
        with pytest.raises(Exception, match="guardrail"):
            c.rest("POST", "/safe-transaction", {**TX, "request_id": "b"})
        assert b.policy.value_calls == 0
        mockc.s.guardrail_mode = "unrestricted"
        assert not c.call_tool_raw("safe_transaction", {**TX, "request_id": "a"}).get("isError")
        assert b.policy.value_calls == 1  # the call that was sent counts
        assert "Bearer" not in json.dumps(b.policy.log)
    finally:
        b.stop()


def test_a_killed_calls_count_is_given_back(mockc):
    b, c = _value_broker(mockc, SessionCaps(max_native_usd=100.0))
    try:
        real = b.send_upstream
        b.send_upstream = lambda epoch, args, send: None  # the command was killed before it was sent
        assert c.call_tool_raw("safe_transaction", {**TX, "request_id": "k"})["isError"]
        with pytest.raises(Exception, match="stopped"):
            c.rest("POST", "/safe-transaction", {**TX, "request_id": "k2"})
        assert b.policy.value_calls == 0
        b.send_upstream = real
    finally:
        b.stop()


def test_start_broker_runs_on_the_shared_policy(mockc):
    b = start_broker(mockc.url, mockc.token, UNCAPPED)
    try:
        assert b.policy.default_chain == "gnosis" and b.policy.caps.uncapped
        assert ConnectClient(b.url, b.token).call_tool("wallet_info", {})["chains"]["gnosis"]["safe"] == SAFE
    finally:
        b.stop()


# ---- the records: Hermes's version, OpenClaw's transcripts --------------------------------------------------------
def test_hermes_from_its_tags_source_is_recorded_with_the_tags_version(monkeypatch):
    out = "Hermes Agent vunknown (2026.9.24)\nInstall directory: /x/vendor/hermes-src/hermes-agent-0.23.2\n"
    monkeypatch.setattr(A, "cli_bin", lambda kind: "hermes")
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=out, stderr=""))
    assert A.harness_version("hermes") == "Hermes Agent v0.23.2 (2026.9.24)"
    assert A._named_version("Hermes Agent vunknown (x)", "no directory line") == "Hermes Agent vunknown (x)"
    assert A._named_version("codex-cli 0.170.0", out) == "codex-cli 0.170.0"


OC_SRC = (  # an earlier build's three spots (2026.11.2), as they read in its export-trajectory-<hash>.mjs
    'import { t as sanitizeDiagnosticPayload } from "./payload-redaction-N9xicL2K.mjs";\n'
    "async function exportTrajectoryBundle(params) {\n"
    "\tconst events = rawEvents.map((event) => redactTrajectoryExportValue(event, redaction));\n"
    "\tfiles.push(trajectoryJsonlFile('events.jsonl', events));\n"
    "\tconst redactedFiles = files.map(redactTrajectoryBundleFileContent);\n}\n"
)
OC_SRC_MINIFIED = (  # the pinned build's (2026.11.5): the same spots with minified local names
    'import { t as sanitizeDiagnosticPayload } from "./payload-redaction-DKjh6Xhl.mjs";\n'
    "async function exportTrajectoryBundle(e) {\n"
    "\tconst v = _.map((e) => redactTrajectoryExportValue(e, n));\n"
    '\tw.push(trajectoryJsonlFile("events.jsonl", v));\n'
    "\tconst D = w.map(redactTrajectoryBundleFileContent);\n}\n"
)


def test_openclaw_exports_are_patched_to_raw_events_only_in_the_vendored_build(tmp_path, monkeypatch):
    out = A.patch_openclaw_export(OC_SRC)
    assert A.OPENCLAW_EXPORT_MARK in out and A.patch_openclaw_export(out) == out  # idempotent
    assert f'const events = process.env.{A.OPENCLAW_RAW_EXPORT} === "1" ? rawEvents : rawEvents.map(' in out
    assert "(event) => redactTrajectoryExportValue(event, redaction));" in out
    assert "const redactedFiles = files.map((file) => process.env." in out
    assert 'file.path === "events.jsonl" ? file : redactTrajectoryBundleFileContent(file));' in out
    assert 'import { t as sanitizeDiagnosticPayloadOwn } from "./payload-redaction-N9xicL2K.mjs";' in out
    assert '=== "1" ? value : sanitizeDiagnosticPayloadOwn(value)' in out
    assert A.patch_openclaw_export("a changed build") is None
    assert A.patch_openclaw_export(OC_SRC.split("\n", 1)[1]) is None  # no transcript-side import: a changed build
    twice = OC_SRC.replace("}\n", "\tconst again = more.map((x) => redactTrajectoryExportValue(x, r));\n}\n")
    assert A.patch_openclaw_export(twice) is None  # a second events spot: a changed build, not a guess
    # the minified build: its own local names kept in the switch
    mini = A.patch_openclaw_export(OC_SRC_MINIFIED)
    assert f'{A.OPENCLAW_EXPORT_MARK} const v = process.env.{A.OPENCLAW_RAW_EXPORT} === "1" ? _ : _.map((e) =>' in mini
    assert "const D = w.map((file) => process.env." in mini and "redactTrajectoryBundleFileContent(file));" in mini
    assert 'import { t as sanitizeDiagnosticPayloadOwn } from "./payload-redaction-DKjh6Xhl.mjs";' in mini
    assert A.patch_openclaw_export(mini) == mini
    vendor = tmp_path / "vendor"
    dist = vendor / "node_modules" / "openclaw" / "dist"
    dist.mkdir(parents=True)
    (vendor / "node_modules" / ".bin").mkdir()
    binp = vendor / "node_modules" / ".bin" / "openclaw"
    binp.write_text("")
    monkeypatch.setattr(A, "VENDOR", vendor)
    monkeypatch.setattr(A, "_NODE_BIN", vendor / "node_modules" / ".bin")
    with pytest.raises(RuntimeError, match="found 0 OpenClaw export modules"):
        A.ensure_openclaw_raw_export(str(binp))
    mod = dist / "export-trajectory-AbC.mjs"
    mod.write_text(OC_SRC)
    mod.chmod(0o644)
    mode = mod.stat().st_mode & 0o777  # what the platform keeps of it (Windows: read-only or not)
    (dist / "export-trajectory-other.mjs").write_text("an unrelated chunk")
    got = A.ensure_openclaw_raw_export(str(binp))
    import hashlib

    assert got == {"node_modules/openclaw/dist/export-trajectory-AbC.mjs": hashlib.sha256(mod.read_bytes()).hexdigest()}
    assert A.OPENCLAW_EXPORT_MARK in mod.read_text() and mod.stat().st_mode & 0o777 == mode
    assert not list(dist.glob(".*.tmp"))  # written atomically: no temp file left
    mtime = mod.stat().st_mtime_ns
    assert A.ensure_openclaw_raw_export(str(binp)) == got and mod.stat().st_mtime_ns == mtime  # once: unchanged
    with pytest.raises(RuntimeError, match="isn't the vendored OpenClaw"):  # the machine's own: never changed
        A.ensure_openclaw_raw_export("/usr/local/bin/openclaw")
    mod.write_text("async function exportTrajectoryBundle() {} // changed build")
    with pytest.raises(RuntimeError, match="differs from the pinned build"):
        A.ensure_openclaw_raw_export(str(binp))


def test_the_preflight_patches_openclaw_once_and_refuses_a_build_it_cant_patch(monkeypatch):
    from onchainbench import run as R

    calls = []
    monkeypatch.setattr(A, "cli_bin", lambda kind: "oc")
    monkeypatch.setattr(A.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(A, "ensure_openclaw_raw_export", lambda binp: calls.append(binp) or {"m.mjs": "abc"})
    assert A.harness_problem("openclaw") is None and A.harness_problem("codex") is None and calls == ["oc"]
    assert R.harness_patches(["openclaw:m", "codex:m"]) == {"openclaw:m": {"m.mjs": "abc"}}
    stamp = R.stamp("openclaw:m", {"harness_patches": {"openclaw:m": {"m.mjs": "abc"}}}, None)
    assert stamp["harness_patch"] == {"m.mjs": "abc"} and "harness_patch" not in R.stamp("codex:m", {}, None)

    def cant(binp):
        raise RuntimeError("OpenClaw's export-trajectory-X.mjs differs from the pinned build's")

    monkeypatch.setattr(A, "ensure_openclaw_raw_export", cant)
    assert "differs from the pinned build" in A.harness_problem("openclaw-raw")  # the run is refused


def test_openclaws_export_runs_raw_and_the_saved_transcript_is_redacted_by_the_bench(tmp_path, monkeypatch):
    from onchainbench import redact

    secret = "sk-or-v1-" + "a" * 40
    ev = {"source": "transcript", "type": "assistant.message"}
    ev["data"] = {"message": {"content": [{"type": "toolCall", "arguments": {"value": "1500000", "key": secret}}]}}
    out = tmp_path / "out"
    out.mkdir()
    (out / "events.jsonl").write_text(json.dumps(ev) + "\n")
    seen = {}

    def run(cmd, **k):
        seen["env"] = k.get("env") or {}
        return types.SimpleNamespace(stdout=json.dumps({"outputDir": str(out)}), stderr="")

    monkeypatch.setattr(A.subprocess, "run", run)
    monkeypatch.setattr(A, "ensure_openclaw_raw_export", lambda binp: pytest.fail("patched per export"))
    s = A.OpenClawSession.__new__(A.OpenClawSession)
    s.bin, s.sid, s.dir, s.exports, s.exported, s.env = "openclaw", "ocb-x", tmp_path, 0, set(), {"A": "1"}
    s.stats, s.fmt = A.Stats(), "openclaw"
    s._init_transcript()
    s._export_transcript("x", "{}")
    assert seen["env"][A.OPENCLAW_RAW_EXPORT] == "1"
    saved = redact.text("\n".join(s.raw), [secret])
    assert "1500000" in saved and secret not in saved  # amounts kept, secrets gone
