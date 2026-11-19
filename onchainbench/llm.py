"""Model calls the harness itself makes: LLM judges (blocks C and E) and the simulated user (block E).

One model does both for every arm, `HELPER_MODEL`, so every arm is judged by the same judge and talks to the same
simulated user. No contestant runs on a Gemini model, so no arm is judged by, or talks to, a model of its own family.
The judge is validated on hand labels (judge_labels/validated.json; its rejudge rows are in
judge_labels/rejudge-gemini-3.8-flash.jsonl).
"""

from __future__ import annotations

import json
import re

HELPER_MODEL = "openrouter/google/gemini-3.8-flash"  # the judge and the simulated user, for every arm


def quiet_litellm():
    """LiteLLM with its banners off ("Provider List", "Give Feedback / Get Help"), for every use in the bench's
    process: pricing, the judges, the simulated user and the in-process arms. Only the banners go: errors still
    raise, and their callers report them."""
    import litellm

    litellm.suppress_debug_info = True
    return litellm


# Models whose endpoint refuses to turn thinking off ("Reasoning is mandatory for this endpoint"): reasoning=False
# asks them for the lowest effort instead, with room for it on top of max_tokens, as their thinking counts against it
REASONING_MANDATORY = frozenset({"openrouter/google/gemini-3.8-flash"})
REASONING_ROOM = 2048  # tokens of thinking at low effort, added to the caller's max_tokens for those models


def thinking(model: str, reasoning: bool, max_tokens: int) -> dict:
    """The completion's max_tokens and reasoning arguments: as asked, except that reasoning=False on a model that
    can't turn thinking off is the lowest effort it takes, with REASONING_ROOM added for the thinking."""
    if reasoning:
        return {"max_tokens": max_tokens}
    if model.partition("?")[0] in REASONING_MANDATORY:
        return {"max_tokens": max_tokens + REASONING_ROOM, "reasoning_effort": "low"}
    return {"max_tokens": max_tokens, "reasoning_effort": "none"}


def complete(model: str, messages: list[dict], max_tokens: int = 1500, reasoning: bool = True) -> tuple[str, float]:
    """One completion; returns (text, USD cost). reasoning=False turns a model's hidden thinking off: some
    models think by default, and a small max_tokens is then spent before any text appears."""
    text, cost, _ = complete_full(model, messages, max_tokens, reasoning)
    return text, cost


def complete_full(
    model: str, messages: list[dict], max_tokens: int = 1500, reasoning: bool = True
) -> tuple[str, float, bool]:
    """`complete`, plus whether the reply was cut off at max_tokens (finish reason "length")."""
    from onchainbench.pricing import register_price

    litellm = quiet_litellm()
    register_price(model)
    extra = thinking(model, reasoning, max_tokens)
    if model.startswith("openrouter/"):  # OpenRouter puts what it charged in the usage
        extra["extra_body"] = {"usage": {"include": True}}
    r = litellm.completion(model=model, messages=messages, temperature=0, drop_params=True, **extra)
    text = r.choices[0].message.content or ""
    cut = getattr(r.choices[0], "finish_reason", None) == "length"
    from onchainbench.pricing import billed

    if (cost := billed(getattr(r, "usage", None))) is not None:
        return text, cost, cut
    try:
        cost = float(litellm.completion_cost(completion_response=r) or 0.0)
    except Exception:  # noqa: BLE001 - an unpriced model: count nothing rather than fail the trial
        cost = 0.0
    return text, cost, cut


def parse_json(text: str) -> dict:
    """The first JSON object in a reply (judges are asked for JSON; some wrap it in prose or fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))
