"""A Connect session for one arm in one trial: workspace, broker, policy, sandboxed shell.

The benchmark's own wiring, so the scripted arms (oracle, null, cheater, obeyer) run without any contestant's code,
and the CLI arms behind the bench's broker (OpenClaw) get it from here. Agents that bring their own broker
(olas-ocb-agent) open their own sessions.
"""

from __future__ import annotations

import contextlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .broker import Broker
from .chains import STABLE_SYMBOLS, native_usd
from .connect_client import ConnectClient, ConnectError
from .policy import CALL_DEADLINE_S, Policy, SessionCaps, read_only_from
from .sandbox import wrapper_for
from .shell import PersistentShell
from .workspace import Workspace


@dataclass
class ConnectSession:
    workspace: Workspace
    shell: PersistentShell
    broker: Broker
    policy: Policy
    sandbox_desc: str

    def close(self) -> None:
        self.shell.close()
        self.broker.stop()


def open_session(
    root: Path,
    caps: SessionCaps,
    approval: str = "auto",
    approver=None,
    sandbox: str = "auto",
    timeout: int = 120,
    on_broker=None,
) -> ConnectSession:
    """`on_broker(broker)` runs once the broker holds its upstream session and before the shell starts (the eval logs
    refusals and hides the real token there)."""
    ws = Workspace.open(Path(root))
    env = ws.install_bin()
    broker = start_broker(*ws.upstream_mcp(), caps, approval, approver)
    policy = broker.policy
    ws.install_shim(broker.shim_mcp_json())
    if on_broker is not None:
        on_broker(broker)
    # the shell may write its own directory only, not the workspace root that holds the brief and the skills (nor
    # rename it or add files to it, even when the root sits under a temp dir)
    wrapper, desc = wrapper_for(
        ws.agent_dir,
        deny_read=ws.deny_read,
        deny_write=ws.deny_write,
        protect=[ws.root] if ws.connect else [],
        mode=sandbox,
    )
    shell = PersistentShell(str(ws.agent_dir), env=env, wrapper=wrapper, timeout=timeout)
    shell.hold, shell.max_hold = broker.busy, CALL_DEADLINE_S
    shell.on_kill = broker.command_killed  # its calls not yet sent are refused
    return ConnectSession(workspace=ws, shell=shell, broker=broker, policy=policy, sandbox_desc=desc)


def make_policy(upstream: ConnectClient, caps: SessionCaps, approval: str = "auto", approver=None) -> Policy:
    """The session policy for a broker in front of `upstream`, set up from what Connect reports: the chain that calls
    omitting `chain` act on (`/wallet`), each mech's payment asset, and Connect's read-only tools. The one constructor
    every bench broker uses (a Connect session's, and the CLI arms' in raw_key.broker_for), so they can't drift."""
    policy = Policy(mode=approval, caps=caps, approver=approver)
    chains = {}
    try:
        chains = upstream.rest("GET", "/wallet").get("chains") or {}
        policy.default_chain = next((c for c, e in chains.items() if e.get("actionable")), None)
    except ConnectError:
        policy.default_chain = None
    policy.mech_asset_lookup = mech_asset_lookup(upstream, chains, policy.native_prices)
    with contextlib.suppress(ConnectError):  # read-only tools as Connect declares them
        policy.read_only = read_only_from(upstream.list_tools()) or policy.read_only
    return policy


def start_broker(url: str, token: str, caps: SessionCaps, approval: str = "auto", approver=None) -> Broker:
    """A running broker in front of the Connect at `url` (its real `token`), with `make_policy`'s policy."""
    upstream = ConnectClient(url, token)
    policy = make_policy(upstream, caps, approval, approver)
    return Broker(ConnectClient(url, token), policy).start()


def mech_asset_lookup(upstream: ConnectClient, chains: dict, prices: dict):
    """(chain, mech) -> (decimals, USD price) of the mech's payment asset, read through Connect and the chain.
    Stablecoins (by symbol) count at $1; anything else is unknown. Cached per (chain, mech); failures give None."""
    cache: dict = {}

    def resolve(chain, mech):
        try:
            info = upstream.call_tool("mech_tools", {"chain": chain, "priority_mech": mech})
        except ConnectError:
            return None
        token = str(info.get("payment_token") or "")
        if token == "native":
            return 18, native_usd(chain, prices)
        rpc = (chains.get(chain) or {}).get("rpc")
        if not token.startswith("0x") or not rpc:
            return None
        try:
            decimals = int(_eth_call(rpc, token, "0x313ce567"), 16)  # decimals()
            symbol = _decode_symbol(_eth_call(rpc, token, "0x95d89b41"))  # symbol()
        except (OSError, ValueError, KeyError):
            return None
        return decimals, (1.0 if symbol.upper() in STABLE_SYMBOLS else None)

    def lookup(chain: str | None, mech: str):
        key = (chain, mech.lower())
        if key not in cache:
            cache[key] = resolve(chain, mech)
        return cache[key]

    return lookup


def _eth_call(rpc: str, to: str, data: str) -> str:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [{"to": to, "data": data}, "latest"]})
    headers = {"Content-Type": "application/json", "User-Agent": "onchainbench"}
    req = urllib.request.Request(rpc, data=body.encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:  # nosec B310 - the chain RPC Connect reports
        return json.loads(r.read())["result"]


def _decode_symbol(hexdata: str) -> str:
    """An ABI-encoded string, or a bytes32 symbol (older tokens)."""
    raw = bytes.fromhex(hexdata[2:])
    if len(raw) >= 96:
        n = int.from_bytes(raw[32:64], "big")
        return raw[64 : 64 + n].decode(errors="replace")
    return raw.rstrip(b"\x00").decode(errors="replace")
