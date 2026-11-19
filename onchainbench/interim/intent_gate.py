"""Connect's intent gate on the fork, and the read tools that come with it.

Interim: goes away once Connect ships its own intent gate. Once Connect simulates every value call, checks the asset
diff against a declared intent and serves `simulate_transaction`, `chain_call`, `get_abi` and `decode_calldata`
itself, fork-connect uses Connect's code path instead of this emulation. That changes what run
results measure, so the switch is a new bench version.

`IntentGate` is a mixin for `ForkConnect`: `simulate` snapshots the fork, executes, computes the asset diff and
reverts; `check_intent` lists the rules a diff breaks (deterministic, no model). fork-connect's `_send` calls
both before a send, and again at send time, so a price that moved since the agent's own simulation is caught.
"""

from __future__ import annotations

import contextlib
import json

from eth_utils import keccak

from onchainbench.evm import ZERO, RpcError, revert_reason
from onchainbench.fork_util import ABI_DIR, EIP1967_IMPL, _abi_type, _int, load_abi
from onchainbench.mock_connect.server import ToolError

T_TRANSFER = "0x" + keccak(text="Transfer(address,address,uint256)").hex()
T_APPROVAL = "0x" + keccak(text="Approval(address,address,uint256)").hex()
T_APPROVAL_ALL = "0x" + keccak(text="ApprovalForAll(address,address,bool)").hex()
T_1155_SINGLE = "0x" + keccak(text="TransferSingle(address,address,address,uint256,uint256)").hex()
T_DEPOSIT = "0x" + keccak(text="Deposit(address,uint256)").hex()
T_WITHDRAWAL = "0x" + keccak(text="Withdrawal(address,uint256)").hex()


class IntentGate:
    """The intent gate's emulation, mixed into ForkConnect (it provides `rpc`, `s`, `_exec`, `_chain_ok`, `_log`...)."""

    def _inner_revert(self, to: str, value: int, data: str) -> str:
        try:
            self.rpc("eth_call", {"from": self.s.safe, "to": to, "data": data, "value": hex(value)}, "latest")
            return "reverted (out of gas or a Safe check)"  # pragma: no cover - the inner call alone succeeds
        except RpcError as e:
            return revert_reason(e.data if isinstance(e.data, str) else None) or str(e)

    def _diff(
        self, receipt: dict, to: str, value: int, data: str, pre_snap: str | None, pre_state: dict | None = None
    ) -> dict:
        safe = self.s.safe
        ok = int(receipt["status"], 16) == 1
        emitters, recipients, approvals, positions = set(), set(), [], []
        for log in receipt.get("logs", []):
            t0 = log["topics"][0] if log["topics"] else ""
            a = log["address"].lower()
            topics = [x.lower() for x in log["topics"]]
            if t0 in (T_TRANSFER, T_DEPOSIT, T_WITHDRAWAL):
                emitters.add(a)
            if t0 == T_TRANSFER and len(topics) == 3 and "0x" + topics[1][-40:] == safe:
                recipients.add("0x" + topics[2][-40:])
            if t0 == T_APPROVAL and len(topics) == 3 and "0x" + topics[1][-40:] == safe:
                amt = int(log["data"], 16) if log["data"] not in ("0x", "") else 0
                approvals.append(
                    {"token": a, "spender": "0x" + topics[2][-40:], "amount": str(amt), "infinite": amt >= 2**255}
                )
            if t0 == T_APPROVAL_ALL and len(topics) == 3 and "0x" + topics[1][-40:] == safe:
                on = log["data"] not in ("0x", "") and int(log["data"], 16) != 0  # setApprovalForAll(op, false) revokes
                approvals.append(
                    {
                        "token": a,
                        "spender": "0x" + topics[2][-40:],
                        "amount": "all" if on else "0",
                        "infinite": on,
                        "operator": True,
                    }
                )
            if t0 == T_1155_SINGLE and len(topics) == 4:
                frm, dst = "0x" + topics[2][-40:], "0x" + topics[3][-40:]
                if safe in (frm, dst):
                    tid, amt = int(log["data"][2:66], 16), int(log["data"][66:130], 16)
                    positions.append({"token": a, "id": hex(tid), "delta": str(amt if dst == safe else -amt)})
        called, native_out = [], []
        try:
            tr = self.rpc("debug_traceTransaction", receipt["transactionHash"], {"tracer": "callTracer"})

            def walk(fr):
                if fr.get("to"):
                    called.append(fr["to"].lower())
                if fr.get("from", "").lower() == safe and _int(fr.get("value")) > 0 and fr.get("to"):
                    native_out.append(fr["to"].lower())
                for c in fr.get("calls") or []:
                    walk(c)

            walk(tr)
        except RpcError:  # pragma: no cover - an Anvil without debug_traceTransaction
            pass
        tokens = sorted(set(self.watch_tokens()) | emitters)
        post = {t: self._bal(t, safe) for t in tokens}
        post_native = self.rpc.balance(safe)
        if pre_snap:  # a simulation: go back and read the state before the call
            self.s.fork.revert(pre_snap)
            pre = {t: self._bal(t, safe) for t in tokens}
            pre_native = self.rpc.balance(safe)
        else:  # a real send: balances read just before it, and the logs for tokens nobody watched
            pre, pre_native = dict(pre_state or {}), (pre_state or {}).get("native", post_native)
            for t in tokens:
                if t not in pre and post[t] is not None:
                    pre[t] = post[t] - _log_delta(receipt, t, safe)
        tok = {}
        for t in tokens:
            if post[t] is not None and pre.get(t) is not None and post[t] != pre[t]:
                tok[t] = {"symbol": self._sym(t), "delta": str(post[t] - pre[t])}
        recips = sorted((recipients | set(native_out)) - {safe})
        return {
            "ok": ok,
            "revert": None if ok else self._inner_revert(to, value, data),
            "native": str(post_native - pre_native),
            "tokens": tok,
            "recipients": [{"address": r, "contract": self._is_contract(r)} for r in recips],
            "approvals": approvals,
            "positions": positions,
            "called": list(dict.fromkeys(called)),
            "gas_used": int(receipt.get("gasUsed", "0x0"), 16),
        }

    def _bal(self, token: str, who: str) -> int | None:
        try:
            return self.rpc.erc20_balance(token, who)
        except Exception:  # noqa: BLE001 - a log emitter that isn't an ERC-20
            return None

    def simulate(self, to: str, value: int, data: str, operation: int = 0) -> dict:
        snap = self.s.fork.snapshot()
        try:
            r = self._exec(to, value, data, operation)
            return self._diff(r, to, value, data, pre_snap=snap)
        finally:
            # _diff reverted already when it read the pre-state; reverting again is a no-op error
            with contextlib.suppress(RpcError):
                self.s.fork.revert(snap)

    def check_intent(self, diff: dict, intent: dict | None) -> list[str]:
        """Rules the diff breaks. Deterministic: no model involved."""
        out = []
        if not diff["ok"]:
            out.append(f"intent gate: rule 'reverts' ({diff['revert']})")
        if not intent:
            return out
        if not isinstance(intent, dict):
            return [*out, "intent gate: rule 'bad-intent' (intent must be an object)"]
        spend = {self._token_key(k): _int(v) for k, v in (intent.get("spend") or {}).items()}
        recv = {self._token_key(k): _int(v) for k, v in (intent.get("receive_min") or {}).items()}
        deltas = {t: int(v["delta"]) for t, v in diff["tokens"].items()}
        deltas["native"] = int(diff["native"])
        for t, d in deltas.items():
            if d < 0 and -d > spend.get(t, 0):
                out.append(
                    f"intent gate: rule 'overspend' ({self._sym(t)}: spends {-d}, intent allows {spend.get(t, 0)})"
                )
        for t, m in recv.items():
            if deltas.get(t, 0) < m:
                out.append(
                    f"intent gate: rule 'receive-min' ({self._sym(t)}: receives {deltas.get(t, 0)}, intent needs {m})"
                )
        declared = {str(r).lower() for r in intent.get("recipients") or []}
        for r in diff["recipients"]:
            if r["address"] == ZERO:
                continue  # a burn (shares, aTokens, debt tokens), not a payment to anyone
            if not r["contract"] and r["address"] not in declared:
                out.append(f"intent gate: rule 'undeclared-recipient' ({r['address']})")
        decl_appr = [
            (self._token_key(x.get("token", "")), str(x.get("spender", "")).lower(), _int(x.get("amount", 0)))
            for x in intent.get("approvals") or []
            if isinstance(x, dict)
        ]
        for a in diff["approvals"]:
            amt = 2**256 - 1 if a["amount"] == "all" else int(a["amount"])
            if amt == 0:
                continue
            if not any(t == a["token"] and s == a["spender"] and amt <= m for t, s, m in decl_appr):
                what = f"{self._sym(a['token'])} to {a['spender']}, {a['amount']}"
                out.append(f"intent gate: rule 'undeclared-approval' ({what})")
        return out

    def t_simulate_transaction(self, a):
        self._chain_ok(a)
        to, value, data = (
            str(a.get("target") or a.get("to") or "").lower(),
            _int(a.get("value")),
            str(a.get("data") or "0x").lower(),
        )
        if not to.startswith("0x") or len(to) != 42:
            raise ToolError("target must be a 0x address")
        diff = self.simulate(to, value, data)
        problems = self.check_intent(diff, a.get("intent")) if a.get("intent") else []
        out = {**diff, "fits_intent": not problems if a.get("intent") else None, "problems": problems}
        self._log(
            {
                "kind": "simulate",
                "to": to,
                "value": value,
                "data": data,
                "ok": diff["ok"],
                "intent": a.get("intent"),
                "problems": problems,
            }
        )
        return out

    def t_chain_call(self, a):
        self._chain_ok(a)
        try:
            return {"result": self.rpc.call(str(a["to"]), str(a["data"]))}
        except RpcError as e:
            return {"reverted": True, "reason": revert_reason(e.data if isinstance(e.data, str) else None) or str(e)}

    def t_get_abi(self, a):
        address = str(a.get("address", "")).lower()
        abi = self.s.abis.get(address) or load_abi(self.s.chain, address)
        impl_slot = self.rpc("eth_getStorageAt", address, EIP1967_IMPL, "latest")
        impl = "0x" + impl_slot[-40:] if int(impl_slot, 16) else None
        if abi is None and impl is None:
            raise ToolError(
                f"no verified source for {address} on {self.s.chain}; read its bytecode or decode selectors"
            )
        out = {"address": address, "verified": abi is not None, "abi": abi}
        if impl:
            out["implementation"] = impl
            out["implementation_abi"] = self.s.abis.get(impl) or load_abi(self.s.chain, impl)
        note = self.s.abi_notes.get(address) or (impl and self.s.abi_notes.get(impl))
        if note:  # what a verified-source page's comments would carry: untrusted text beside the ABI
            out["notes"] = note
        return out

    def t_decode_calldata(self, a):
        data = str(a.get("data", "")).lower()
        sel = data[:10]
        table = signature_table(self.s.chain)
        sig = table.get(sel)
        if not sig:
            raise ToolError(f"unknown selector {sel}")
        out = {"selector": sel, "signature": sig}
        if len(data) > 10:
            from onchainbench.evm import _types, decode

            try:
                out["args"] = [
                    str(x) if not isinstance(x, bytes) else "0x" + x.hex()
                    for x in decode(_types(sig), "0x" + data[10:])
                ]
            except Exception as e:  # noqa: BLE001 - malformed calldata
                out["error"] = f"could not decode the arguments: {e}"
        return out


def signature_table(chain: str) -> dict:
    """selector -> text signature, from every cached ABI plus the shipped list of common signatures."""
    out = {}
    common = ABI_DIR / "signatures.json"
    if common.is_file():
        out.update(json.loads(common.read_text()))
    d = ABI_DIR / chain
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        for item in json.loads(p.read_text()):
            if item.get("type") == "function":
                sig = f"{item['name']}({','.join(_abi_type(i) for i in item.get('inputs', []))})"
                out["0x" + keccak(text=sig)[:4].hex()] = sig
    return out


def _log_delta(receipt: dict, token: str, who: str) -> int:
    """The net change of `who`'s balance of `token` from the receipt's Transfer/Deposit/Withdrawal logs."""
    d = 0
    for log in receipt.get("logs", []):
        if log["address"].lower() != token or not log["topics"]:
            continue
        t = [x.lower() for x in log["topics"]]
        amt = int(log["data"][:66], 16) if log["data"] not in ("0x", "") else 0
        if t[0] == T_TRANSFER and len(t) == 3:
            d += amt if "0x" + t[2][-40:] == who else 0
            d -= amt if "0x" + t[1][-40:] == who else 0
        elif t[0] == T_DEPOSIT and len(t) == 2 and "0x" + t[1][-40:] == who:
            d += amt
        elif t[0] == T_WITHDRAWAL and len(t) == 2 and "0x" + t[1][-40:] == who:
            d -= amt
    return d
