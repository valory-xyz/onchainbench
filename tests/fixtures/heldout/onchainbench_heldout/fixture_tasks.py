"""One held-out-shaped task, made from an open task, so CI without the private set still runs the held-out machinery."""

import copy

from onchainbench.tasks.block_a import TASKS as _OPEN

_t = copy.copy(next(t for t in _OPEN if t.id == "a-pay-invoice"))
_t.id, _t.family = "t-fixture-pay", "fixture"
TASKS = [_t]
