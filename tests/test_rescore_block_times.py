"""The re-score dates each ledger transaction by its mined block, from the run's Anvil logs (`block_timestamp`)."""

import json
from datetime import UTC, datetime
from email.utils import format_datetime

from onchainbench import rescore as R

T0 = 1794241200  # Mon, 9 Nov 2026 16:20:00 UTC


def h(n: int) -> str:
    return "0x" + f"{n:064x}"


def mined(tx: int, block: int, ts: int, created: bool = False) -> str:
    """One mined transaction as Anvil logs it."""
    when = format_datetime(datetime.fromtimestamp(ts, UTC)).replace("-0000", "+0000")
    return (
        f"eth_sendTransaction\neth_getTransactionReceipt\n\n    Transaction: {h(tx)}\n"
        + ("    Contract created: 0xF2D805C083cc50c097644164E12BFCF66CCE7B14\n" if created else "")
        + f"    Gas used: 21000\n\n    Block Number: {block}\n    Block Hash: {h(block)}\n"
        f'    Block Time: "{when}"\n\neth_getTransactionReceipt\n'
    )


def sim(tx: int, block: int, ts: int) -> str:
    """A simulation: mined in a snapshot, then reverted."""
    return "evm_snapshot\n" + mined(tx, block, ts) + "debug_traceTransaction\nevm_revert\n"


def rec(task, attempt, sends, chain="gnosis", seq=1, slot="gnosis#0@a", failed=()):
    ledger = [{"kind": "simulate", "to": "0x1"}]
    ledger += [{"kind": "tx", "tx_hash": h(s), "status": "success"} for s in sends]
    ledger += [{"kind": "tx", "tx_hash": h(s), "status": "reverted"} for s in failed]
    ledger += [{"kind": "refusal", "rule": "x"}]
    return {
        "arm": "codex:m",
        "task": task,
        "attempt": attempt,
        "chain": chain,
        "fork_slot": slot,
        "fork_generation": 0,
        "fork_seq": seq,
        "ledger": ledger,
        "passed": True,
        "reason": "ok",
        "block": "A",
        "family": "wrap",
        "split": "open",
        "infra": False,
        "integrity": [],
        "metrics": {},
        "seconds": 1.0,
        "cost_usd": 0.0,
        "model_calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "turns": 1,
        "steps_done": 1,
        "steps_total": 1,
    }


def logs(tmp_path, **files):
    d = tmp_path / "shard-0"
    d.mkdir(parents=True)
    for name, text in files.items():
        (d / f"anvil-{name}.log").write_text("evm_revert\nevm_snapshot\n" + text)
    return d


def test_the_anvil_log_s_mined_blocks_and_their_times():
    text = mined(1, 101, T0, created=True) + sim(2, 102, T0 + 2) + mined(2, 102, T0 + 4)
    assert R.mined(text) == [(h(1), 101, T0), (h(2), 102, T0 + 2), (h(2), 102, T0 + 4)]
    # a fall in the block number starts the next trial; a simulation's revert reuses the number, never lower
    spans = R.trial_spans(R.mined(text + mined(9, 101, T0 + 60) + mined(3, 102, T0 + 62)))
    assert [[b[0] for b in s] for s in spans] == [[h(1), h(2), h(2)], [h(9), h(3)]]


def test_each_send_gets_the_time_of_the_mining_that_stayed(tmp_path):
    fork_a = (
        mined(100, 101, T0)  # the harness's setup
        + sim(1, 102, T0 + 1)
        + mined(1, 102, T0 + 3)  # the send itself, after its simulation
        + mined(2, 103, T0 + 9)
        + mined(5, 104, T0 + 10)  # a send that reverted on chain is mined too
        + mined(100, 101, T0 + 60)  # the next trial: same setup, the fork reverted to its base
        + mined(3, 102, T0 + 61)
    )
    d = logs(tmp_path, **{"gnosis-1": fork_a, "polygon-1": mined(1, 7, T0 + 99)})
    recs = [
        rec("a", 0, [1, 2], failed=[5]),
        rec("b", 0, [3], seq=2),
        rec("c", 0, [4]),  # mined in no log: undated
        rec("d", 0, []),  # sent nothing
        rec("e", 0, [2, 1]),  # the ledger's order isn't the log's: undated
    ]
    times = R.block_times(recs, R.anvil_spans([d]))
    assert times[0][h(1)] == T0 + 3 and times[0][h(2)] == T0 + 9 and times[0][h(5)] == T0 + 10
    assert times[1][h(3)] == T0 + 61 and times[2:] == [None, None, None]
    out = R.with_block_times(recs[0], times[0])
    txs = [e for e in out["ledger"] if e["kind"] == "tx"]
    assert [e["block_timestamp"] for e in txs] == [T0 + 3, T0 + 9, T0 + 10]
    assert all("block_timestamp" not in e for e in out["ledger"] if e["kind"] != "tx")
    assert R.with_block_times(recs[2], None) is recs[2]


def test_trials_with_the_same_sends_take_their_fork_s_spans_in_the_order_it_ran_them(tmp_path):
    # two attempts with the same parameters send the same transaction: told apart by the order on their fork
    fork = mined(100, 101, T0) + mined(1, 102, T0 + 1) + mined(100, 101, T0 + 100) + mined(1, 102, T0 + 101)
    d = logs(tmp_path, **{"gnosis-1": fork})
    first, second = rec("a", 1, [1], seq=1), rec("a", 0, [1], seq=2)
    t = R.block_times([second, first], R.anvil_spans([d]))
    assert t == [{h(100): T0 + 100, h(1): T0 + 101}, {h(100): T0, h(1): T0 + 1}]
    # on different forks, or one trial for two spans: undated
    other = rec("a", 0, [1], seq=2, slot="gnosis#1@a")
    assert R.block_times([first, other], R.anvil_spans([d])) == [None, None]
    assert R.block_times([first], R.anvil_spans([d])) == [None]
    # a span another trial claims alone is set aside first
    d2 = logs(
        tmp_path / "b",
        **{"gnosis-1": fork + mined(100, 101, T0 + 200) + mined(1, 102, T0 + 201) + mined(9, 103, T0 + 202)},
    )
    claims = rec("z", 0, [1, 9], seq=3)
    t = R.block_times([first, second, claims], R.anvil_spans([d2]))
    assert t[0][h(1)] == T0 + 1 and t[1][h(1)] == T0 + 101 and t[2][h(9)] == T0 + 202


def test_the_re_score_writes_block_timestamp_from_the_logs_it_is_given(tmp_path, capsys):
    run = tmp_path / "run"
    (run / "sessions" / "transcripts").mkdir(parents=True)
    (run / "trials.jsonl").write_text(json.dumps(rec("a-wrap-unwrap", 0, [1])) + "\n")
    d = logs(tmp_path, **{"gnosis-1": mined(1, 102, T0 + 3)})
    assert R.main([str(run), str(tmp_path / "out"), "--anvil-logs", str(d)]) == 0
    out = json.loads((tmp_path / "out" / "trials.jsonl").read_text("utf-8"))
    assert [e.get("block_timestamp") for e in out["ledger"] if e["kind"] == "tx"] == [T0 + 3]
    assert "mined block times on every send of 1 of 1 trials that sent" in capsys.readouterr().out
    R.main([str(run), str(tmp_path / "plain")])  # without logs: nothing attached
    out = json.loads((tmp_path / "plain" / "trials.jsonl").read_text("utf-8"))
    assert all("block_timestamp" not in e for e in out["ledger"])


def test_a_hash_mined_at_two_block_numbers_in_a_span_is_left_undated(tmp_path):
    # tx 1 stays at block 102, and the same hash is mined again at 103 within the span: which is the ledger's?
    fork = mined(100, 101, T0) + mined(1, 102, T0 + 1) + mined(2, 103, T0 + 5) + mined(1, 103, T0 + 9)
    d = logs(tmp_path, **{"gnosis-1": fork})
    (times,) = R.block_times([rec("a", 0, [2])], R.anvil_spans([d]))
    assert h(1) not in times and times[h(2)] == T0 + 5
    out = R.with_block_times(rec("a", 0, [1, 2], failed=[]), times)
    assert [e.get("block_timestamp") for e in out["ledger"] if e["kind"] == "tx"] == [None, T0 + 5]


def test_trials_in_one_merged_span_take_their_own_stretch_unless_they_overlap(tmp_path):
    # a trial that mined one block leaves no fall before the next: one span holds both trials
    fork = mined(100, 101, T0) + mined(1, 102, T0 + 1) + mined(2, 102, T0 + 60) + mined(3, 103, T0 + 61)
    d = logs(tmp_path, **{"gnosis-1": fork})
    a, b = rec("a", 0, [1]), rec("b", 0, [2, 3], seq=2)
    assert R.block_times([b, a], R.anvil_spans([d])) == [{h(2): T0 + 60, h(3): T0 + 61}, {h(1): T0 + 1}]
    # a's hash mined again after b's began: their stretches overlap, so neither is dated
    fork = mined(100, 101, T0) + mined(1, 102, T0 + 1) + mined(2, 102, T0 + 60) + mined(1, 102, T0 + 62)
    d = logs(tmp_path / "overlap", **{"gnosis-1": fork + mined(3, 103, T0 + 63)})
    assert R.block_times([a, b], R.anvil_spans([d])) == [None, None]
    # on two forks: not one fork's order, so neither is dated
    d = logs(tmp_path / "forks", **{"gnosis-1": mined(100, 101, T0) + mined(1, 102, T0 + 1) + mined(2, 102, T0 + 60)})
    other = rec("b", 0, [2], seq=2, slot="gnosis#1@a")
    assert R.block_times([a, other], R.anvil_spans([d])) == [None, None]
