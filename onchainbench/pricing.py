"""Model prices and the secret-name filter, shared by the runner and the arms.

A trial's cost is what the provider billed where the arm can see it (OpenRouter's `usage.cost`), and otherwise an
estimate: tokens times the price of the provider the arm is pinned to (`?provider=` in the model spec), with cached
input at that provider's cache-read price and input written to the cache at its cache-write price (Anthropic and
OpenAI charge more for a write than for plain input). Without a pin, the model's OpenRouter list price is used.
Every run saves the prices it used in its config.json (`prices`).
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)  # env names an agent's shell must not read
FORK_ENV = re.compile(r"^FORK_RPC", re.I)  # the forks' upstream RPCs: a paid archive node's URL carries its key


def fork_urls(environ=None) -> dict[str, str]:
    """The upstream RPC URLs configured for the forks (FORK_RPC_<CHAIN>), by variable name."""
    env = os.environ if environ is None else environ
    return {k: v for k, v in env.items() if FORK_ENV.match(k) and v}


def secret_env(environ=None) -> set[str]:
    """The variables an agent's shell must not inherit: secret-named ones, the forks' upstream RPCs, and any other
    variable holding one of those URLs."""
    env = os.environ if environ is None else environ
    urls = set(fork_urls(env).values())
    return {k for k, v in env.items() if SECRET_ENV.search(k) or FORK_ENV.match(k) or any(u in v for u in urls)}


OR_API = "https://openrouter.ai/api/v1"
_PRICES: dict = {}  # openrouter/<id> -> {"input", "output", "cache_read", "cache_write"} list prices, USD per token
_ENDPOINTS: dict = {}  # <id> -> [{"provider", "tag", "input", "output", "cache_read", "cache_write"}]
KEYS = ("input", "output", "cache_read", "cache_write")


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _entry(p: dict) -> dict:
    pin = float(p.get("prompt") or 0)
    read, write = p.get("input_cache_read"), p.get("input_cache_write")
    return {
        "input": pin,
        "output": float(p.get("completion") or 0),
        "cache_read": float(read) if read else pin,
        "cache_write": float(write) if write else pin,  # no write price: a write costs plain input
    }


def openrouter_prices() -> dict:
    """List prices for openrouter/<id> models (LiteLLM doesn't know the newest ids)."""
    if not _PRICES:
        try:
            for m in _get(f"{OR_API}/models")["data"]:
                _PRICES["openrouter/" + m["id"]] = _entry(m.get("pricing") or {})
        except OSError:
            pass
    return _PRICES


def endpoints(model_id: str) -> list[dict]:
    """The providers serving an OpenRouter model, with their prices (`/models/<id>/endpoints`)."""
    if model_id not in _ENDPOINTS:
        try:
            data = _get(f"{OR_API}/models/{urllib.parse.quote(model_id)}/endpoints")["data"]
            _ENDPOINTS[model_id] = [
                {"provider": e.get("provider_name", ""), "tag": e.get("tag", ""), **_entry(e.get("pricing") or {})}
                for e in data.get("endpoints") or []
            ]
        except (OSError, KeyError):
            _ENDPOINTS[model_id] = []
    return _ENDPOINTS[model_id]


def parse_spec(spec: str) -> tuple[str, dict]:
    """'openrouter/x?provider=fireworks' -> ('openrouter/x', {'provider': 'fireworks'})."""
    name, _, q = spec.partition("?")
    opts = {}
    for part in filter(None, q.split("&")):
        k, _, v = part.partition("=")
        opts[k] = {"1": True, "true": True, "0": False, "false": False}.get(v.lower(), v) if v else True
    return name, opts


def price(spec: str) -> dict | None:
    """USD per token for a model spec (`openrouter/<id>?provider=<slug>`): the pinned provider's price (the endpoint
    whose tag is the slug, else the first of that provider's), else the list price. None if unknown."""
    model, _, q = spec.partition("?")
    if not model.startswith("openrouter/"):
        return None
    pin = urllib.parse.parse_qs(q).get("provider", [""])[0].lower()
    if pin:
        eps = endpoints(model.removeprefix("openrouter/"))
        match = [e for e in eps if e["tag"].lower() == pin] or [
            e for e in eps if e["provider"].lower() == pin or e["tag"].lower().startswith(pin + "/")
        ]
        if match:
            e = match[0]
            return {k: e[k] for k in KEYS} | {"source": f"endpoint {e['tag']}"}
    if model in openrouter_prices():
        return {**openrouter_prices()[model], "source": "list"}
    return None


def cost_from_tokens(spec: str, prompt: int, completion: int, cached: int = 0, cache_write: int = 0) -> float:
    """An estimate: plain input, cached input and input written to the cache (both part of `prompt`), and output,
    at the spec's price. Prices saved before cache writes were priced have no `cache_write`: plain input then."""
    p = price(spec) or dict.fromkeys(KEYS, 0.0)
    write = p.get("cache_write", p["input"])
    return (
        (prompt - cached - cache_write) * p["input"]
        + cached * p["cache_read"]
        + cache_write * write
        + completion * p["output"]
    )


def price_entry(model: str) -> dict | None:
    """The spec's price as a LiteLLM model_cost entry (USD per token, cache reads and writes too), or None if it
    has none: what the bench registers in its own process, and hands a CLI arm that runs LiteLLM in its own
    (olas-ocb-agent's `--model-price`)."""
    if (p := price(model)) is None:
        return None
    return {
        "input_cost_per_token": p["input"],
        "output_cost_per_token": p["output"],
        "cache_read_input_token_cost": p["cache_read"],
        "cache_creation_input_token_cost": p.get("cache_write", p["input"]),
        "litellm_provider": "openrouter",
    }


def price_registry(model: str) -> dict | None:
    """{name: entry} for LiteLLM's register_model, under both names (responses may report the model id without the
    provider prefix), or None without a price."""
    if (entry := price_entry(model)) is None:
        return None
    name = model.partition("?")[0]
    return {name: entry, name.removeprefix("openrouter/"): entry}


def register_price(model: str) -> bool:
    """Give LiteLLM the spec's price (with its cache-read price) in the bench's process: the judge's and the
    simulated user's estimates use it. True if the model has a price (registered here, or already known to
    LiteLLM)."""
    from onchainbench.llm import quiet_litellm

    litellm = quiet_litellm()
    if (reg := price_registry(model)) is not None:
        litellm.register_model(reg)
        return True
    return model.partition("?")[0] in litellm.model_cost


def billed(usage) -> float | None:
    """What OpenRouter charged, from a response's usage (`usage.cost`), or None."""
    if usage is None:
        return None
    c = usage.get("cost") if isinstance(usage, dict) else getattr(usage, "cost", None)
    if c is None and not isinstance(usage, dict):
        c = (getattr(usage, "model_extra", None) or {}).get("cost")
    return float(c) if c is not None else None
