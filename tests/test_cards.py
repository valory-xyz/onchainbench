"""Open-track task cards: every open task, never a held-out one, leak-checked."""

import dataclasses
import json

import pytest

from onchainbench import cards as C
from onchainbench.tasks import all_tasks
from onchainbench.tasks.base import Result, Task


def open_tasks():
    return [t for t in all_tasks() if t.split == "open"]


def test_every_dev_task_gets_a_card_with_its_check():
    cs = C.cards(open_tasks())
    assert len(cs) == len({t.variant_of or t.id for t in open_tasks()}) and all(c["check"].get("source") for c in cs)
    wrap = next(c for c in cs if c["id"] == "a-wrap-unwrap")
    assert wrap["check"]["path"].startswith("onchainbench/tasks/block_a.py#L") and wrap["instruction_example"]
    twin = next(c for c in cs if c.get("attack"))
    assert twin["twin_of"] and isinstance(twin["threat"], dict)
    assert any(c["instruction_example"] is None for c in cs)  # names a contract its setup deploys


def test_labels_and_odd_checks():
    def labelled(t):
        return [("asked the mech", True), ("staked", False), ("asked the mech", True), (1, 2)]

    assert C.checks_of(labelled) == ["asked the mech", "staked"]
    assert C.checks_of(len) == [] and C.check_of(len) == {"function": "len", "labels": []}  # no source


def test_variants_share_a_card_and_heldout_tasks_are_refused():
    base = next(t for t in open_tasks() if t.id == "a-wrap-unwrap")
    v = [dataclasses.replace(base, id=f"a-wrap-unwrap~v{i}", variant_of="a-wrap-unwrap") for i in range(2)]
    assert [c["id"] for c in C.cards(v)] == ["a-wrap-unwrap~v0"]
    held = next(t for t in all_tasks() if t.split == "test")
    with pytest.raises(C.CardError, match="only open-set tasks get cards"):
        C.cards([held])


def test_export_writes_and_refuses_leaks(tmp_path, monkeypatch, capsys):
    out = tmp_path / "cards.jsonl"
    assert C.export(open_tasks()[:3], out) == 3 and len(out.read_text().splitlines()) == 3
    leaky = Task("a-leaky", "A", "f", "gnosis", "mentions " + "t2-f-" + "secret-twin", lambda t: Result(True))
    with pytest.raises(C.CardError, match="give the held-out set away"):
        C.export([leaky], tmp_path / "x.jsonl")
    monkeypatch.setattr("onchainbench.tasks.heldout_dir", lambda: None)  # no held-out set: shapes still checked
    with pytest.raises(C.CardError):
        C.export([leaky], tmp_path / "x.jsonl")
    monkeypatch.chdir(tmp_path)
    assert C.main([]) == 0 and f"wrote {len(C.cards(open_tasks()))} task cards" in capsys.readouterr().out
    assert json.loads((tmp_path / "task-cards.jsonl").read_text().splitlines()[0])["id"]
    monkeypatch.setattr(C, "export", lambda tasks, dest: (_ for _ in ()).throw(C.CardError("leak")))
    assert C.main(["--out", str(out)]) == 1 and "refusing to export: leak" in capsys.readouterr().out
