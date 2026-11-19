"""Secrets out of what a run keeps: transcripts, session files and trial records, the same for every arm.

An agent may read its own `.mcp.json` (it is the agent's config), so Connect's bearer token can end up in a tool's
output, and from there in a saved transcript. What the agent sees during the trial is never changed: only the saved
copies are redacted, when the runner writes them (run.keep_transcript, isolation.hand_over_sessions, the trial's
record). Three things are replaced with `[redacted]`:
- the trial's own secrets, matched by value (metrics.secrets: Connect's token, the signing key, the brokers' tokens,
  the model and provisioning keys an arm registered), with the encodings the leak check looks for;
- `Bearer <token>` wherever it appears, with or without `Authorization:`;
- common key shapes (`KEY_SHAPES`): `sk-...` (OpenAI, OpenRouter's `sk-or-...`, Anthropic's `sk-ant-...`), GitHub's
  `ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_` and `github_pat_`, AWS's `AKIA`/`ASIA` key ids, Slack's `xox[abpr]-`, and PEM
  private key blocks. The same rule for every arm (OpenClaw's own redaction is off in its exports).

A run recorded before this has no secrets to match by value: `python -m onchainbench.redact <results dir>` rewrites
its files with the patterns, and first collects every bearer token the files show (Connect's token is
`secrets.token_urlsafe(16)`, a brokers' `token_urlsafe(24)`), so a copy of one printed without `Bearer` goes too.
A key with no known shape printed without `Bearer` is found only by value: `--secrets-file` gives the run's keys.
Files that aren't UTF-8 text (a session database) are masked to the same length instead, so they stay readable.
"""

from __future__ import annotations

import argparse
import base64
import re
from collections.abc import Iterable
from pathlib import Path

REDACTED = "[redacted]"
SECRET_MIN = 8  # a value this short isn't a credential, and matching it would redact ordinary text
TOKEN = r"[A-Za-z0-9._~+/-]{16,}=*"  # a bearer token: Connect's and the brokers' are urlsafe base64 (22, 32 chars)
BEARER = re.compile(rf"(\bBearer\s+)({TOKEN})", re.I)
KEY_SHAPES = re.compile(
    "|".join(
        (
            r"\bsk-[A-Za-z0-9_-]{16,}",  # OpenAI, OpenRouter's sk-or-..., Anthropic's sk-ant-...
            r"\bgh[pousr]_[A-Za-z0-9]{36,}",  # GitHub tokens: personal, OAuth, user-to-server, server, refresh
            r"\bgithub_pat_[A-Za-z0-9_]{22,}",  # GitHub fine-grained personal access tokens
            r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",  # AWS access key ids (long-term, temporary)
            r"\bxox[abpr]-[A-Za-z0-9-]{10,}",  # Slack tokens
            # a PEM private key block, to its END line; one cut short (truncated output) through its base64 lines
            r"(?s:-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
            r"(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY-----|(?:(?:\s|\\n|\\r)*[A-Za-z0-9+/=]{16,})*))",
        )
    )
)


def forms(secret: str) -> set[str]:
    """A secret as written, without 0x, and base64- or hex-encoded (the copies metrics.token_leaks counts)."""
    raw, b = secret.removeprefix("0x"), secret.encode()
    out = {secret, raw, base64.b64encode(b).decode(), base64.urlsafe_b64encode(b).decode().rstrip("="), b.hex()}
    return {f for f in out if len(f) >= SECRET_MIN}


def text(s: str, secrets: Iterable[str] = ()) -> str:
    values = {f for v in secrets if v and len(v) >= SECRET_MIN for f in forms(v)}
    for v in sorted(values, key=len, reverse=True):  # longest first: a form inside another goes with it
        s = re.sub(re.escape(v), REDACTED, s, flags=re.I)
    s = BEARER.sub(lambda m: m.group(1) + REDACTED, s)
    return KEY_SHAPES.sub(REDACTED, s)


def data(obj, secrets: Iterable[str] = ()):
    """A copy of a JSON-shaped value with every string redacted (keys too); the original is left as it was."""
    secrets = list(secrets)
    if isinstance(obj, str):
        return text(obj, secrets)
    if isinstance(obj, dict):
        return {data(k, secrets) if isinstance(k, str) else k: data(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [data(v, secrets) for v in obj]
    if isinstance(obj, tuple):
        return tuple(data(v, secrets) for v in obj)
    return obj


def masked(s: str, secrets: Iterable[str] = ()) -> str:
    """`text`'s redaction with each match overwritten by as many `*` as it has characters: for a binary file (a
    session database), where a shorter replacement would move every offset after it and corrupt the file."""
    values = {f for v in secrets if v and len(v) >= SECRET_MIN for f in forms(v)}
    for v in sorted(values, key=len, reverse=True):
        s = re.sub(re.escape(v), lambda m: "*" * len(m.group(0)), s, flags=re.I)
    s = BEARER.sub(lambda m: m.group(1) + "*" * len(m.group(2)), s)
    return KEY_SHAPES.sub(lambda m: "*" * len(m.group(0)), s)


def _read(path: Path) -> tuple[str, bool] | None:
    """A file's text and whether it is UTF-8; a binary file is read byte for byte (latin-1), None if unreadable."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("latin-1"), False


def file(path: Path, secrets: Iterable[str] = ()) -> bool:
    """Redact a file in place; True if it changed. Text is redacted with `[redacted]`; a file that isn't UTF-8 text
    is masked to the same length, byte for byte (`masked`), so its structure is kept."""
    if (got := _read(path)) is None:
        return False
    old, is_text = got
    new = text(old, secrets) if is_text else masked(old, secrets)
    if new != old:
        path.write_bytes(new.encode("utf-8" if is_text else "latin-1"))
    return new != old


def tree(path: Path, secrets: Iterable[str] = ()) -> list[Path]:
    """A file, or every file under a directory; the ones that changed."""
    secrets = list(secrets)
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path] if path.is_file() else []
    return [p for p in files if file(p, secrets)]


def trial_secrets(t) -> list[str]:
    """Every secret of a trial (metrics.secrets), or none when it failed before it had any."""
    if not hasattr(getattr(t, "fc", None), "token"):
        d = getattr(t, "data", None) or {}
        return [*d.get("secrets", []), *d.get("named_secrets", {}).values()]
    from onchainbench.metrics import secrets

    return list(secrets(t).values())


def bearer_tokens(s: str) -> set[str]:
    return {m.group(2) for m in BEARER.finditer(s)}


def run(root: Path, known: Iterable[str] = ()) -> list[Path]:
    """An existing results dir (a run, or a folder of runs): every bearer token any of its files shows is collected
    first, then every file is redacted with those tokens, the `known` secrets (the run's keys, where the person
    redacting has them: a key with no known shape printed without "Bearer" is found only by value) and the patterns.
    Returns the files that changed."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    found = set(known)
    for p in files:
        if (got := _read(p)) is not None:
            found |= bearer_tokens(got[0])
    return [p for p in files if file(p, found)]


def secrets_file(path: Path) -> list[str]:
    """One secret per line (blank lines and `#` comments skipped): a file, so no key goes on a command line."""
    lines = (s.strip() for s in path.read_text(encoding="utf-8").splitlines())
    return [s for s in lines if s and not s.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m onchainbench.redact", description="Redact secrets from a run's saved files, in place."
    )
    ap.add_argument("dirs", nargs="+", type=Path, help="a results dir (results/<run>) or a folder of them")
    ap.add_argument(
        "--secrets-file",
        type=Path,
        help="the run's keys, one per line, redacted by value too (for a key with no known shape)",
    )
    a = ap.parse_args(argv)
    for d in a.dirs:
        if not d.is_dir():
            print(f"{d}: not a directory")
            return 2
    known = secrets_file(a.secrets_file) if a.secrets_file else []
    for d in a.dirs:
        changed = run(d, known)
        print(f"{d}: redacted {len(changed)} file(s)")
        for p in changed:
            print(f"  {p.relative_to(d)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
