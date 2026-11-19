"""A team Safe the agent's Safe co-owns, for tasks that administer a multisig.

Connect refuses a call from the agent's Safe to itself, so the agent's own Safe can't be the one it administers. A
task deploys a second Safe 1.4.1 from the canonical singleton and factory (on every fork), owned by the agent's Safe,
two people whose keys the task creates and funds, and a co-signer contract (`contracts/src/TeamSafe.sol`), with
threshold 2. The agent reaches the threshold on its own: it has the co-signer approve a transaction's hash
(`cosign`), then calls `execTransaction` with two pre-validated signatures (v=1), one for its own Safe as the caller
and one for the co-signer's approved hash, in ascending owner order.

The co-signer's policy is about a transaction's shape only (a plain call from the Safe to itself, no value, no gas
refund): it never names the admin call a task wants. The two people's keys are real, so a task can also leave
transactions signed with ECDSA that anyone could execute.

The ABIs in `contracts/external/` are the verified ones of Safe's SafeL2 1.4.1 singleton and of its allowance
module, as Sourcify serves them: a task adds them to `get_abi` for the addresses it uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from eth_account import Account
from eth_utils import keccak

from onchainbench.contracts import OUT, abi, deploy
from onchainbench.evm import ZERO, RpcError, calldata, checksum, revert_reason
from onchainbench.fork_connect import random_key
from onchainbench.forks import SAFE_141, owner_signature
from onchainbench.metrics import SAFE_FALLBACK_SLOT as HANDLER_SLOT
from onchainbench.metrics import SAFE_GUARD_SLOT as GUARD_SLOT

SENTINEL = "0x" + "00" * 19 + "01"  # the head and the end of a Safe's linked lists of owners and of modules
EXEC = "execTransaction(address,uint256,bytes,uint8,uint256,uint256,uint256,address,address,bytes)"
TX_HASH = "getTransactionHash(address,uint256,bytes,uint8,uint256,uint256,uint256,address,address,uint256)"
COSIGN = "cosign(address,address,uint256,bytes,uint8,uint256,uint256,uint256,address,address)"
SETUP = "setup(address[],uint256,address,bytes,address,address,uint256,address)"
PROXY_CREATION = "0x" + keccak(text="ProxyCreation(address,address)").hex()
ALLOWANCE_MODULE = "0xcfbfac74c26f8647cbdb8c5caf80bb5b32e43134"  # Safe's allowance module 0.1.0, on the forks used
KEY_GAS = 10**17  # native each person's key starts with
READ_TRIES = 3  # reads of a call's outcome before a transport error is raised as the bench's


def external_abi(name: str) -> list:
    return json.loads((OUT.parent / "external" / f"{name}.json").read_text())


def _raw(data: str) -> bytes:
    return bytes.fromhex(data.removeprefix("0x"))


@dataclass
class Team:
    """A team Safe and its four owners; `keys` holds the two people's private keys, by address."""

    safe: str
    cosigner: str
    alice: str
    mallory: str
    agent: str
    keys: dict = field(default_factory=dict, repr=False)


# ---- reads ----------------------------------------------------------------------------------------------------------
def owners(t, safe: str) -> list[str]:
    """The Safe's owners, in its linked list's order (what a previous-owner pointer is read from)."""
    return [a.lower() for a in t.rpc.view(safe, "getOwners()", returns=["address[]"])]


def threshold(t, safe: str) -> int:
    return t.rpc.view(safe, "getThreshold()", returns=["uint256"])


def nonce(t, safe: str) -> int:
    return t.rpc.view(safe, "nonce()", returns=["uint256"])


def modules(t, safe: str) -> list[str]:
    """The Safe's enabled modules, in its linked list's order (the newest first)."""
    page, _ = t.rpc.view(safe, "getModulesPaginated(address,uint256)", SENTINEL, 50, returns=["address[]", "address"])
    return [a.lower() for a in page]


def guard_and_handler(t, safe: str) -> tuple[str, str]:
    """The Safe's transaction guard and its fallback handler, from their storage slots (the Safe has no view for
    either): a guard can block or allow what the owners sign, and a fallback handler answers calls in the Safe's
    name."""
    return tuple("0x" + t.rpc("eth_getStorageAt", safe, slot, "latest")[-40:] for slot in (GUARD_SLOT, HANDLER_SLOT))


def before(items: list[str], item: str) -> str:
    """What points at `item` in a Safe's linked list: the entry before it, or the sentinel for the first."""
    i = [a.lower() for a in items].index(item.lower())
    return checksum(items[i - 1] if i else SENTINEL)


def call_revert(t, frm: str, to: str, data: str) -> str | None:
    """Why `frm`'s call would revert now, or None if it wouldn't. A transport error is retried, then raised: a
    dropped connection says nothing about the chain."""
    last: Exception | None = None
    for _ in range(READ_TRIES):
        try:
            t.rpc("eth_call", {"from": frm, "to": to, "data": data}, "latest")
            return None
        except RpcError as e:
            if isinstance(e.data, str):
                return revert_reason(e.data) or str(e)
            last = e
        except OSError as e:
            last = e
    raise RuntimeError(f"read failed {READ_TRIES} times: {last}")


# ---- calldata and signatures ----------------------------------------------------------------------------------------
def tx_hash(t, safe: str, to: str, data: str = "0x", value: int = 0, at: int | None = None) -> str:
    """The Safe's hash of a plain call with no gas refund, at nonce `at` (default: its current nonce)."""
    n = nonce(t, safe) if at is None else at
    h = t.rpc.view(safe, TX_HASH, checksum(to), value, _raw(data), 0, 0, 0, 0, ZERO, ZERO, n, returns=["bytes32"])
    return "0x" + h.hex()


def ecdsa(key: str, digest: str) -> bytes:
    """A key's signature over a Safe transaction hash (r, s, v with v 27 or 28), as `execTransaction` reads it."""
    sign = Account._sign_hash if hasattr(Account, "_sign_hash") else Account.unsafe_sign_hash
    return bytes(sign(_raw(digest), key).signature)


def pack(by_owner: dict[str, bytes]) -> bytes:
    """Signatures as `execTransaction` wants them: one after another, in ascending order of the owners' addresses."""
    return b"".join(by_owner[o] for o in sorted(by_owner, key=lambda a: int(a, 16)))


def approved_by(*who: str) -> bytes:
    """Pre-validated signatures (v=1) for owners that send the transaction themselves or have approved its hash."""
    return pack({o: owner_signature(o.lower()) for o in who})


def exec_data(to: str, data: str = "0x", signatures: bytes = b"", value: int = 0) -> str:
    return calldata(EXEC, checksum(to), value, _raw(data), 0, 0, 0, 0, ZERO, ZERO, signatures)


def cosign_data(safe: str, to: str, data: str = "0x", value: int = 0, operation: int = 0, gas_price: int = 0) -> str:
    return calldata(COSIGN, checksum(safe), checksum(to), value, _raw(data), operation, 0, 0, gas_price, ZERO, ZERO)


# ---- setup: the harness acting as the team --------------------------------------------------------------------------
def deploy_team(t, order: list[int], salt: int = 1) -> Team:
    """A Safe 1.4.1 owned by the agent's Safe, two new funded keys and a new co-signer, threshold 2. `order` is the
    owners' order in the Safe's list, as indices into (agent, first person, second person, co-signer): a task draws
    it, so the previous-owner pointer differs between attempts. The Safe's and the co-signer's ABIs are added to
    `get_abi`."""
    (alice, alice_key), (mallory, mallory_key) = random_key(), random_key()
    for a in (alice, mallory):
        t.fork.set_balance(a, KEY_GAS)
    cosigner = deploy(t, "PolicyCoSigner")
    agent = t.safe.lower()
    listed = [(agent, alice, mallory, cosigner)[i] for i in order]
    init = calldata(SETUP, [checksum(o) for o in listed], 2, ZERO, b"", checksum(SAFE_141["fallback"]), ZERO, 0, ZERO)
    create = calldata("createProxyWithNonce(address,bytes,uint256)", checksum(SAFE_141["singleton"]), _raw(init), salt)
    r = t.fc.harness_tx(t.fork.accounts[3], SAFE_141["factory"], create)
    safe = next("0x" + log["topics"][1][-40:] for log in r["logs"] if log["topics"][0] == PROXY_CREATION).lower()
    if owners(t, safe) != listed or threshold(t, safe) != 2:
        raise RuntimeError(f"premise broken: the team Safe's owners are {owners(t, safe)}, expected {listed}")
    t.fc.s.abis[safe] = external_abi("SafeL2_1_4_1")
    t.fc.s.abis[cosigner] = abi("PolicyCoSigner")
    return Team(safe, cosigner, alice, mallory, agent, {alice: alice_key, mallory: mallory_key})


def signed(t, team: Team, to: str, data: str = "0x", value: int = 0, at: int | None = None) -> bytes:
    """The two people's ECDSA signatures over a team Safe transaction, packed for `execTransaction`."""
    h = tx_hash(t, team.safe, to, data, value, at)
    return pack({a: ecdsa(k, h) for a, k in team.keys.items()})


def exec_signed(t, team: Team, to: str, data: str = "0x", value: int = 0) -> dict:
    """The team Safe makes a call on the two people's signatures; a harness account sends it."""
    sigs = signed(t, team, to, data, value)
    return t.fc.harness_tx(t.fork.accounts[3], team.safe, exec_data(to, data, sigs, value))


def exec_cosigned(t, team: Team, owner: str, data: str) -> dict:
    """One owner key changes the team Safe through the co-signer, as the agent can: `cosign`, then `execTransaction`
    with its own and the co-signer's pre-validated signatures."""
    t.fc.harness_tx(owner, team.cosigner, cosign_data(team.safe, team.safe, data))
    return t.fc.harness_tx(owner, team.safe, exec_data(team.safe, data, approved_by(owner, team.cosigner)))
