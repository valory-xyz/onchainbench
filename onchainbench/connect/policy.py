"""What the broker lets the agent do without asking, what it asks about, and what it refuses.

This sits *in front of* Connect's own guardrail, never instead of it: a call
the broker allows still has to pass the connect server's rules. The broker's
job is the part Connect can't know about: this session, this user, right now.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from ..interim.readonly_fallback import FALLBACK_READ_ONLY  # until Connect flags its read-only tools; its flags win
from .chains import native_usd

# One deadline for a broker call: an unanswered approval is declined, and the marketplace gate
# gives up, when it passes; the agent's shell command waits for the broker meanwhile (shell.py `hold`).
CALL_DEADLINE_S = 600

# Calls that move funds or produce a signature someone could use to move funds.
VALUE_TOOLS = {"safe_transaction", "send_transaction", "mech_request", "sign_message"}
REST_VALUE_ROUTES = {
    "/safe-transaction": "safe_transaction",
    "/sign-message": "sign_message",
}  # signer_client.py's routes
REST_READ_ROUTES = {"/wallet"}


@dataclass
class Decision:
    allowed: bool
    rule: str = ""  # named rule when refused, Connect-style
    counted: tuple | None = None  # (native USD, mech USD, approval key) an allowed call added: released if not sent


# What Connect answers for a call it refused before sending anything: the intent gate's and the guardrail's refusals.
# Such a call spent nothing, so the policy gives back what it counted for it (Policy.release). A call that may have
# landed (a lost response, a mined revert) stays counted. Only Connect's own formats: the gate ends its refusal with
# NOT_SENT_SUFFIX, the guardrail starts with GUARDRAIL_PREFIX. A mined revert ("transaction 0x... reverted: <reason>")
# carries the contract's own text, which could say anything, so it is never read for them.
NOT_SENT_SUFFIX = ". Nothing was sent."
GUARDRAIL_PREFIX = "guardrail: rule "
MINED = "transaction 0x"


def not_sent(error_text) -> bool:
    t = str(error_text or "").strip()
    if t.startswith(MINED):
        return False
    return t.startswith(GUARDRAIL_PREFIX) or t.endswith(NOT_SENT_SUFFIX)


@dataclass
class SessionCaps:
    """Per-session ceilings in USD, converted per chain (chains.py, or user prices). None = no cap."""

    max_native_usd: float | None = 0.10  # value sent from the safe (or EOA)
    max_mech_usd: float | None = 1.00  # sum of declared max_payment across mech requests
    max_value_calls: int | None = 20
    allow_sign_message: bool = False  # raw digests are opaque: off unless the user turns it on

    @property
    def uncapped(self) -> bool:
        """No ceiling and nothing off: the broker passes every call on, and Connect's own rules decide."""
        return (
            self.max_native_usd is None
            and self.max_mech_usd is None
            and self.max_value_calls is None
            and self.allow_sign_message
        )


# No caps at all: what the benchmark gives every arm (the same conditions as an arm holding Connect's token).
UNCAPPED = SessionCaps(max_native_usd=None, max_mech_usd=None, max_value_calls=None, allow_sign_message=True)


@dataclass
class Policy:
    """mode: 'ask' (default: prompt for every value call), 'auto' (allow within caps), 'deny'."""

    mode: str = "ask"
    caps: SessionCaps = field(default_factory=SessionCaps)
    approver: object = None  # callable(summary: str) -> bool, used in 'ask' mode
    spent_native: float = 0.0  # USD
    spent_mech: float = 0.0  # USD
    default_chain: str | None = None  # for calls that omit `chain` (set by the runtime from wallet_info)
    native_prices: dict = field(default_factory=dict)  # user overrides: chain -> native USD price (any EVM chain)
    # (chain, mech address) -> (decimals, USD price) of that mech's payment asset, or None (set by the runtime)
    mech_asset_lookup: Callable | None = None
    read_only: set = field(default_factory=lambda: set(FALLBACK_READ_ONLY))  # never includes VALUE_TOOLS
    value_calls: int = 0
    approved: set = field(default_factory=set)  # (request_id, hash of the arguments) of every approved call
    log: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self.caps = replace(self.caps)  # the session's own copy: a shared one (UNCAPPED) is never changed through it

    def check(self, tool: str, args: dict) -> Decision:
        with self._lock:
            d = self._check(tool, args)
            self.log.append({"tool": tool, "args": args, "allowed": d.allowed, "rule": d.rule})
            return d

    def _check(self, tool: str, args: dict) -> Decision:
        if tool in self.read_only and tool not in VALUE_TOOLS:
            return Decision(True)
        rid = args.get("request_id")
        if rid and (rid, _args_hash(tool, args)) in self.approved:
            return Decision(True)  # idempotent retry of something already approved: no new spend
        if self.mode == "deny":
            return Decision(False, "broker: value calls disabled for this session (approval mode 'deny')")
        if self.caps.uncapped:  # nothing to count against: only the user's approval in 'ask' mode
            return self._allow(tool, args, rid, 0.0, 0.0)
        if tool == "sign_message" and not self.caps.allow_sign_message:
            return Decision(False, "broker: raw digest signing is off for this session")
        chain = args.get("chain") or self.default_chain
        try:  # a negative amount would lower the counters and lift the caps, so it is refused, never counted
            wei = _int(args.get("value")) if tool in ("safe_transaction", "send_transaction") else 0
            pay = _int(args.get("max_payment")) if tool == "mech_request" else 0
        except NegativeAmount:
            return Decision(False, "broker: rule 'negative-amount': value and max_payment can't be negative")
        except ValueError:
            return Decision(
                False, "broker: rule 'bad-amount': value and max_payment must be integers (or 0x hex) below 2**256"
            )
        native = 0.0
        if wei:
            price = native_usd(chain, self.native_prices)
            if price is None:
                return Decision(
                    False,
                    f"broker: no USD price for {chain!r}'s native token, so value can't be capped "
                    f"(the user can set one: --native-usd {chain}=<price>)",
                )
            native = wei / 1e18 * price
        if tool == "mech_request" and args.get("max_payment") is None:
            # Connect's default cap is 0.1 of the payment asset, whose units differ per chain;
            # without an explicit cap the session total can't be counted, so ask for one
            return Decision(
                False,
                "broker: set max_payment explicitly (base units of the mech's payment "
                "asset; see mech_tools max_delivery_rate)",
            )
        mech = 0.0
        if tool == "mech_request":
            if not args.get("priority_mech"):  # its payment asset (and so the USD cap) depends on the mech
                return Decision(
                    False, "broker: name the mech (priority_mech; see mech_tools) so its price can be capped"
                )
            asset = self.mech_asset_lookup(chain, args["priority_mech"]) if self.mech_asset_lookup else None
            if asset is None or asset[1] is None:
                return Decision(
                    False,
                    f"broker: the payment asset of mech {args['priority_mech']} on {chain!r} isn't known "
                    "(not native or a USD stablecoin), so its price can't be capped",
                )
            mech = pay / 10 ** asset[0] * asset[1]
        if native < 0 or mech < 0:  # a negative price (a user's --native-usd, a mech asset): never lower the counters
            return Decision(False, "broker: rule 'negative-amount': the call would count a negative USD amount")
        c = self.caps
        if c.max_value_calls is not None and self.value_calls + 1 > c.max_value_calls:
            return Decision(False, f"broker: session limit of {c.max_value_calls} value calls reached")
        if c.max_native_usd is not None and self.spent_native + native > c.max_native_usd + 1e-12:
            return Decision(False, f"broker: session native cap ${c.max_native_usd:.2f} would be exceeded")
        if c.max_mech_usd is not None and self.spent_mech + mech > c.max_mech_usd + 1e-12:
            return Decision(False, f"broker: session mech payment cap ${c.max_mech_usd:.2f} would be exceeded")
        if tool not in VALUE_TOOLS and self.mode != "ask":
            return Decision(False, f"broker: unknown tool '{tool}' is not allowed without asking")
        return self._allow(tool, args, rid, native, mech)

    def _allow(self, tool: str, args: dict, rid, native: float, mech: float) -> Decision:
        if self.mode == "ask" and not (self.approver and self.approver(summarize(tool, args))):
            return Decision(False, "broker: the user declined this call")
        self.value_calls += 1
        self.spent_native += native
        self.spent_mech += mech
        key = (rid, _args_hash(tool, args)) if rid else None
        if key:
            self.approved.add(key)
        return Decision(True, counted=(native, mech, key))

    def release(self, d: Decision) -> None:
        """Give back what an allowed call counted, once it is known that it was never sent (killed before sending,
        or refused by Connect before sending: `not_sent`). A retry of it is then checked, and counted, again."""
        if d.counted is None:
            return
        with self._lock:
            native, mech, key = d.counted
            d.counted = None
            self.value_calls -= 1
            self.spent_native -= native
            self.spent_mech -= mech
            self.approved.discard(key)


def _args_hash(tool: str, args: dict) -> str:
    """A retry is the same call only if its arguments match what was approved."""
    return hashlib.sha256(json.dumps([tool, args], sort_keys=True, default=str).encode()).hexdigest()


class NegativeAmount(ValueError):
    """A negative value or max_payment: refused, since counting it would lower the session's spend."""


_AMOUNT = re.compile(r"(?P<neg>-)?(?:0[xX](?P<hex>[0-9a-fA-F]+)|(?P<dec>[0-9]+))", re.ASCII)


def _int(v) -> int:
    """A non-negative amount below 2**256 (a uint256) from an int, a decimal string or a 0x hex string; None and ""
    are 0.

    Raises NegativeAmount for a negative amount, ValueError for anything else: not an integer, or too large for
    a uint256 (which would also overflow the float conversion to USD).
    """
    if v is None:
        return 0
    if isinstance(v, int):
        n = int(v)
    else:
        s = str(v).strip() or "0"
        m = _AMOUNT.fullmatch(s)
        if m is None:  # ASCII only: no "1_000", "+5" or other scripts' digits, which int() would take
            raise ValueError(f"not an amount: {s[:40]!r}")
        n = (int(m["hex"], 16) if m["hex"] else int(m["dec"])) * (-1 if m["neg"] else 1)
    if n < 0:
        raise NegativeAmount(n)
    if n >= 2**256:
        raise ValueError("amount too large for a uint256")
    return n


def summarize(tool: str, args: dict) -> str:
    if tool in ("safe_transaction", "send_transaction"):
        who = "safe" if tool == "safe_transaction" else "agent EOA"
        data = str(args.get("data", "0x"))
        sel = data[:10] if len(data) >= 10 else "(plain transfer)"
        out = (
            f"{who} call on {args.get('chain', 'default chain')}: to {args.get('target') or args.get('to')} "
            f"value {args.get('value', 0)} wei, selector {sel}, {max(0, (len(data) - 2) // 2)} bytes of calldata"
        )
        erc20 = {"0xa9059cbb": "ERC-20 transfer", "0x095ea7b3": "ERC-20 approve"}.get(sel.lower())
        if erc20 and len(data) >= 138:  # say who gets what: a user can't approve what they can't read
            party = "0x" + data[34:74].lower()
            role = "recipient" if erc20.endswith("transfer") else "spender"
            out += f" ({erc20}: {role} {party}, amount {int(data[74:138], 16)} base units of the token at the target)"
        return out
    if tool == "mech_request":
        return (
            f"pay mech {args.get('priority_mech', '(marketplace picks)')} for tool {args.get('tool')}, "
            f"max_payment {args.get('max_payment', 'default')}: {str(args.get('prompt', ''))[:120]!r}"
        )
    if tool == "sign_message":
        return f"sign raw digest {args.get('digest')}"
    return f"{tool}({args})"


def read_only_from(tools: list[dict]) -> set | None:
    """The tools Connect flags read-only (MCP `readOnlyHint`); None if it flags none (fall back)."""
    flagged = {t["name"] for t in tools if (t.get("annotations") or {}).get("readOnlyHint")}
    return (flagged - VALUE_TOOLS) or None
