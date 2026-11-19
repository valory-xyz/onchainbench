"""The held-out id shape the leak check looks for without the held-out set: every authoring round's prefix, while no
open task id ever matches it."""

from onchainbench import leaks
from onchainbench.tasks import all_tasks


def _shape(*parts: str) -> str:
    return "-".join(parts)  # built here, so the leak check over this file finds no held-out shape


def test_every_round_prefix_is_a_held_out_shape():
    for text in (
        _shape("t", "c", "anything"),
        _shape("t2", "g", "anything"),
        _shape("t4", "g", "anything"),
        _shape("t5", "f", "md", "anything"),
        _shape("t12", "a", "anything"),
    ):
        assert leaks.ID_PATTERN.search(f"see {text} here"), text
    assert not leaks.ID_PATTERN.search(_shape("md", "anything"))  # no class abbreviations: the public code names none


def test_no_open_task_id_has_a_held_out_shape():
    open_ids = [t.id for t in all_tasks() if t.split != "test"]
    assert open_ids and not [i for i in open_ids if leaks.ID_PATTERN.fullmatch(i)]
