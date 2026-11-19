"""install-arms.sh's downloads fail on an HTTP error and retry, and are never piped straight into tar: an error or
rate-limit page piped into tar only said "not in gzip format" (a CI shard's Hermes install)."""

import re
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent / "scripts" / "install-arms.sh").read_text()


def test_every_download_fails_on_http_errors_and_retries():
    calls = [line for line in SCRIPT.splitlines() if re.match(r"\s*(?:\S+=\S+\s+)*curl\s", line)]  # curl commands
    assert calls
    for line in calls:
        assert "-f" in line.split() or "-fsSL" in line, line
        assert "--retry" in line, line


def test_no_download_is_piped_into_tar():
    assert not re.search(r"curl[^\n|]*\|\s*tar", SCRIPT)
