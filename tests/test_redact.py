"""smoke-matrix-5: an agent that read its own .mcp.json put Connect's bearer token in a saved transcript. What the
run keeps (transcripts, session files, trial records) is redacted, for every arm; an old run's files are redacted
by `python -m onchainbench.redact`."""

import base64
import json
import secrets
import types

import pytest

from onchainbench import isolation as I
from onchainbench import redact
from onchainbench import run as R
from onchainbench import transcript as TR

R_ = redact.REDACTED
TOKEN = secrets.token_urlsafe(16)  # Connect's token, as the mock server makes it
OR_KEY = "orkey" + secrets.token_hex(12)  # a registered secret without a known shape: matched by value only
SK_KEY = "sk-or-v1-" + secrets.token_hex(24)  # an unregistered key of a known shape
MCP_JSON = json.dumps(
    {
        "mcpServers": {
            "pearl-connect": {"url": "http://127.0.0.1:7000/mcp/", "headers": {"Authorization": f"Bearer {TOKEN}"}}
        }
    },
    indent=2,
)


def lines(*objs) -> str:
    return "".join(json.dumps(o) + "\n" for o in objs)


def codex_raw() -> str:  # a CLI shell arm: codex's command_execution items
    return lines(
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat .mcp.json",
                "exit_code": 0,
                "aggregated_output": MCP_JSON,
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "env",
                "exit_code": 0,
                "aggregated_output": f"KEY={OR_KEY}\nOTHER={SK_KEY}",
            },
        },
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Safe 0x5afe holds 50 USDC.e."}},
    )


def hermes_raw() -> str:  # an MCP arm: Hermes's tool events
    return lines(
        {"type": "tool_use", "name": "terminal", "input": {"command": "head -c 500 .mcp.json"}},
        {"type": "tool_result", "name": "terminal", "output": MCP_JSON[:500]},
        {"type": "tool_use", "name": "pearl-connect.wallet_info", "input": {}},
        {
            "type": "tool_result",
            "name": "pearl-connect.wallet_info",
            "output": f"safe 0x5afe; echo {base64.b64encode(OR_KEY.encode()).decode()}",
        },
        {"type": "text", "text": "Done: the Safe is 0x5afe."},
    )


def text_protocol_raw() -> (
    str
):  # a text-protocol shell arm (olas-ocb-agent's chat shape): bash blocks and <returncode> outputs
    return json.dumps(
        [
            {"role": "assistant", "content": "```bash\ncat .mcp.json\n```"},
            {"role": "user", "content": f"<returncode>0</returncode>\n{MCP_JSON}\n{OR_KEY}"},
            {"role": "assistant", "content": "Done: the Safe is 0x5afe."},
        ]
    )


# fixed ids: the raw output holds this module's random secrets, so ids made from it would differ between xdist
# workers, which then refuse to run ("Different tests were collected")
@pytest.mark.parametrize(
    "fmt, raw",
    [("codex", codex_raw()), ("hermes", hermes_raw()), ("chat", text_protocol_raw())],
    ids=["codex", "hermes", "chat"],
)
def test_a_saved_transcript_is_redacted_and_nothing_else_changes(tmp_path, fmt, raw):
    events = TR.PARSERS[fmt](raw, 0)
    seen = json.loads(json.dumps(events))
    data = {}
    r = {"_transcript": (events, [raw])}
    R.keep_transcript(r, "arm__e-ambiguous-recipient__0", tmp_path, data, [TOKEN, OR_KEY])
    path = tmp_path / "transcripts" / "arm__e-ambiguous-recipient__0.jsonl"
    assert r["transcript"] == str(path)
    saved = path.read_text() + (path.parent / f"{path.stem}.raw.jsonl").read_text()
    for s in (TOKEN, OR_KEY, SK_KEY, base64.b64encode(OR_KEY.encode()).decode()):
        assert s not in saved
    assert f"Bearer {R_}" in saved and "0x5afe" in saved and ".mcp.json" in saved
    # the events the checks read are what the agent saw, untouched
    assert data["transcript_events"] == seen and TOKEN in json.dumps(seen)

    # and the saved events are the same ones, each secret replaced
    def scrub(s: str) -> str:
        for v in (TOKEN, OR_KEY, base64.b64encode(OR_KEY.encode()).decode()):
            s = s.replace(v, R_)
        return s.replace(SK_KEY, R_)

    written = [json.loads(x) for x in path.read_text().splitlines()]
    assert written == [{k: scrub(v) if isinstance(v, str) else v for k, v in e.items()} for e in seen]


def test_the_patterns_and_values():
    assert redact.text(f"Authorization: Bearer {TOKEN}") == f"Authorization: Bearer {R_}"
    assert redact.text(f"bearer   {TOKEN}xyz==") == f"bearer   {R_}"
    assert redact.text(f"key {SK_KEY} and sk-ant-{'a' * 30}") == f"key {R_} and {R_}"
    # short or prose-like words after Bearer, and "sk-" inside a word, stay
    assert redact.text("a Bearer token; Bearer $PEARL_CONNECT_TOKEN; task-0123456789abcdefgh") == (
        "a Bearer token; Bearer $PEARL_CONNECT_TOKEN; task-0123456789abcdefgh"
    )
    assert redact.text(f"x {TOKEN.upper()} y", [TOKEN]) == f"x {R_} y"  # case-insensitive, like the leak check
    assert redact.text(f"0x{'ab' * 32}", ["0x" + "ab" * 32]) == R_
    assert redact.text("short abc", ["abc", "", None]) == "short abc"  # too short to be a credential
    obj = {"a": [f"Bearer {TOKEN}", 3, None], "t": ("x", OR_KEY), OR_KEY: 1}
    out = redact.data(obj, [OR_KEY])
    assert out == {"a": [f"Bearer {R_}", 3, None], "t": ("x", R_), R_: 1} and obj["t"][1] == OR_KEY


def test_the_trials_secrets():
    fc = types.SimpleNamespace(token=TOKEN, s=types.SimpleNamespace(eoa_key="0x" + "cd" * 32))
    t = types.SimpleNamespace(fc=fc, data={"secrets": ["brokertoken-0123456789"], "named_secrets": {"the key": OR_KEY}})
    assert set(redact.trial_secrets(t)) == {TOKEN, "0x" + "cd" * 32, "brokertoken-0123456789", OR_KEY}
    # before Connect started (or a fake trial): only what the data holds
    assert redact.trial_secrets(types.SimpleNamespace(fc=None, data={"named_secrets": {"k": OR_KEY}})) == [OR_KEY]
    assert redact.trial_secrets(None) == []


def test_session_files_are_redacted_where_the_run_keeps_them(tmp_path):
    """An isolated arm's session files at hand-over, and an unisolated arm's session file in the run's sessions dir."""
    own, dest = tmp_path / "own", tmp_path / "sessions"
    (own / "sub").mkdir(parents=True)
    (own / "s.jsonl").write_text(lines({"role": "tool", "content": MCP_JSON}))
    (own / "sub" / "t.txt").write_text(f"key {OR_KEY}")
    # not text (a session database): masked to the same length, so every offset after it stays where it was
    blob = b"\xff\xfe\x00" + OR_KEY.encode() + b"\x00Bearer " + TOKEN.encode() + b"\x00\xff"
    (own / "blob.bin").write_bytes(blob)
    assert I.hand_over_sessions(own, dest, None, [OR_KEY]) == ["blob.bin", "s.jsonl", "sub"]
    assert TOKEN not in (dest / "s.jsonl").read_text() and f"Bearer {R_}" in (dest / "s.jsonl").read_text()
    assert (dest / "sub" / "t.txt").read_text() == f"key {R_}"
    stars = b"*" * len(OR_KEY) + b"\x00Bearer " + b"*" * len(TOKEN)
    assert (dest / "blob.bin").read_bytes() == b"\xff\xfe\x00" + stars + b"\x00\xff" and len(stars) + 5 == len(blob)
    unreadable = tmp_path / "gone"
    assert redact.file(unreadable) is False  # a file that can't be read is left, not raised
    # unisolated: the session file the stats point at, in the run's sessions dir; one elsewhere is not the run's
    (dest / "p.jsonl").write_text(f"{OR_KEY} Bearer {TOKEN}")
    elsewhere = tmp_path / "p.jsonl"
    elsewhere.write_text(OR_KEY)
    R.keep_transcript({"session": str(dest / "p.jsonl")}, "x", dest, secrets=[OR_KEY])
    R.keep_transcript({"session": str(elsewhere)}, "x", dest, secrets=[OR_KEY])
    R.keep_transcript({"session": str(dest / "p.jsonl")}, "x", None, secrets=[OR_KEY])  # no sessions dir: nothing
    assert (dest / "p.jsonl").read_text() == f"{R_} Bearer {R_}" and elsewhere.read_text() == OR_KEY
    assert redact.tree(tmp_path / "missing") == [] and redact.tree(elsewhere, [OR_KEY]) == [elsewhere]


def test_the_cli_redacts_an_existing_run(tmp_path, capsys):
    """open-run-1 was recorded before redaction: no secret is known by value, so every bearer token any file shows is
    collected and redacted everywhere, a bare copy included, with the key shapes."""
    run = tmp_path / "results" / "open-run-1"
    tr = run / "sessions" / "transcripts"
    tr.mkdir(parents=True)
    other = secrets.token_urlsafe(24)  # a broker's token
    (tr / "codex__e__0.jsonl").write_text(lines({"kind": "tool_result", "text": MCP_JSON}))
    (tr / "codex__e__0.raw.jsonl").write_text(json.dumps(codex_raw()) + "\n")
    (tr / "hermes__f__0.jsonl").write_text(lines({"kind": "tool_result", "text": f"TOKEN={TOKEN} {other}"}))
    (run / "sessions" / "s.jsonl").write_text(f'{{"h": "Bearer {other}"}}\n')
    (run / "trials.jsonl").write_text(json.dumps({"answer": f"token is {TOKEN}", "passed": True}) + "\n")
    clean = json.dumps({"passed": True, "reason": "right 10, wrong 0"}) + "\n"
    (run / "summary.json").write_text(clean)
    png = b"\x89PNG\xff" + TOKEN.encode()  # a binary file: the token collected elsewhere is masked in it too
    (run / "plot.png").write_bytes(png)
    (run / "notes.txt").write_text(f"model key {OR_KEY}")  # no known shape, no Bearer: found only by value
    keys = tmp_path / "keys.txt"
    keys.write_text(f"# the run's keys\n\n  {OR_KEY}  \n")
    assert redact.main([str(tmp_path / "results"), "--secrets-file", str(keys)]) == 0
    out = capsys.readouterr().out
    assert "redacted 7 file(s)" in out and "summary.json" not in out
    assert (run / "plot.png").read_bytes() == b"\x89PNG\xff" + b"*" * len(TOKEN)
    assert (run / "notes.txt").read_text() == f"model key {R_}"
    for p in run.rglob("*"):
        if p.is_file() and p.suffix != ".png":
            text = p.read_text()
            assert TOKEN not in text and other not in text and SK_KEY not in text and OR_KEY not in text
    assert (tr / "hermes__f__0.jsonl").read_text() == lines({"kind": "tool_result", "text": f"TOKEN={R_} {R_}"})
    assert json.loads((run / "trials.jsonl").read_text()) == {"answer": f"token is {R_}", "passed": True}
    assert (run / "summary.json").read_text() == clean  # nothing else changes
    json.loads((tr / "codex__e__0.raw.jsonl").read_text())  # still valid JSON
    assert redact.run(tmp_path / "results") == []  # a second pass finds nothing left
    assert redact.main([str(tmp_path / "nope")]) == 2


PEM_BODY = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nBKcwggSjAgEAAoIBAQC7\n"


@pytest.mark.parametrize(
    "key",
    [
        "ghp_" + "A1b2" * 9,
        "gho_" + "z" * 36,
        "ghs_" + "9" * 40,
        "github_pat_11ABCDEFG0" + "x_Y" * 20,
        "AKIA" + "IOSFODNN7EXAMPLE",
        "ASIA" + "Q" * 16,
        "xoxb-" + "1234567890-abcdefghij",
        "xoxp-" + "a1" * 12,
        "sk-ant-" + "b" * 30,
        "-----BEGIN PRIVATE KEY-----\n" + PEM_BODY + "-----END PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\\n" + PEM_BODY.replace("\n", "\\n") + "-----END RSA PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----\n" + PEM_BODY,  # cut short: no END line
    ],
    ids=["ghp", "gho", "ghs", "github_pat", "akia", "asia", "xoxb", "xoxp", "sk-ant", "pem", "pem-json", "pem-cut"],
)
def test_common_key_shapes_are_redacted_for_every_arm(key):
    s = f"before {key} after."
    assert redact.text(s).replace("\n", "") == f"before {R_} after."  # a cut block's last newline may stay
    m = redact.masked(s)
    assert len(m) == len(s) and "PRIVATE KEY" not in m and key[8:24] not in m and m.endswith(" after.")


def test_key_shapes_leave_look_alikes_alone():
    plain = (
        "ghp_short, AKIAshort, xoxz-1234567890ab, github_pat_x, a 0x" + "ab" * 32 + " hash, "
        "-----BEGIN PUBLIC KEY----- and -----BEGIN CERTIFICATE-----, a PEM header in prose: -----BEGIN PRIVATE KEY----- "
        "is how it starts"
    )
    out = redact.text(plain)
    assert out.startswith(plain.split("a PEM header")[0]) and out.endswith(" is how it starts")
