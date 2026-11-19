#!/bin/sh
# The full local gate: lint, format, the tests, and the validation runs (on every open task, and the held-out
# stand-in's, the oracle passes, null and cheater fail), with 100% line coverage of the package over the tests and
# the validation runs combined.
# Run from anywhere:
#   scripts/check.sh               the whole gate
#   scripts/check.sh --tests-only  lint and tests, coverage kept in .coverage but not gated: CI's check job
#                                  (validate-open runs the validation; the coverage job gates both)
set -e
cd "$(dirname "$0")/.."
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --dist loadgroup --durations=30 --cov-fail-under=0
[ "$1" = "--tests-only" ] && exit 0
.venv/bin/python -m coverage run --append -m onchainbench.run --arms oracle,null,cheater --split open --workers 4 --run check-open
.venv/bin/python scripts/validate_run.py results/check-open/trials.jsonl --split open
export ONCHAINBENCH_HELDOUT="$PWD/tests/fixtures/heldout/onchainbench_heldout"  # the tests' one-task stand-in
.venv/bin/python -m coverage run --append -m onchainbench.run --arms oracle,null,cheater --split test --run check-test
.venv/bin/python scripts/validate_run.py results/check-test/trials.jsonl --split test
.venv/bin/python -m coverage report
