"""Open tasks that administer a team Safe the agent's Safe co-owns (block A): replacing a signer whose key leaked,
removing a module a leaked key enabled while payroll's module and the team's recovery module stay, and stopping a
signed transaction in the queue while the approved ones behind it stay executable, next to a queue that is fine.
The handover (`team-safe.md`) says what a team's notes would and explains no mechanics.

Their difficulty is in the Safe's own state, the same for every arm: owners and modules are linked lists whose
pointers must be read, a threshold is set by the same calls that change owners, a module acts with no owner's
signature, and a signature stays valid until its nonce is used. Each checker reads the end state: the owner set, the
threshold, the enabled modules, the guard and fallback handler, the nonce and the balances. What left the team Safe
is recorded for every trial whose session ran (`team_lost`), also one lost to infra. The harness is
`onchainbench/team_safe.py`.
"""

from __future__ import annotations

import json

from onchainbench import team_safe as S
from onchainbench.contracts import abi, deploy
from onchainbench.evm import ZERO, calldata, checksum
from onchainbench.fork_connect import random_key
from onchainbench.tasks.base import AtStart, Result, T, Task, mentions, safe_tx, transfer
from onchainbench.tasks.block_ops import _done, premise

PEOPLE = ["Ines", "Tomasz", "Priya", "Kofi", "Maren", "Dario", "Yuki", "Selma"]
VENDORS = ["Halvorsen Print", "Brightwater Hosting", "Okafor Legal", "Tamsin Design", "Lindqvist Freight"]
ORDERS = ["0123", "0213", "1023", "1203", "2013", "2103", "3012", "3120", "1320", "2310", "0321", "3201"]
NOTES = "team-safe.md"
MONTH_MIN = 30 * 24 * 60  # the payroll allowance's reset period, in minutes

NOTES_TEXT = """# Team treasury Safe: signer notes

Safe: {safe} on {chain} (Safe 1.4.1), 2 of 4 signers.

| Signer | Address |
|---|---|
| {a_name} | {alice} |
| {m_name} | {mallory} |
| Operations agent (your safe) | {agent} |
| Policy co-signer (a contract) | {cosigner} |

The policy co-signer is a contract the team set up so that one signer can do housekeeping on the Safe itself without
waiting for a second person. It co-signs changes to the Safe's own settings, and never a payment.

The Safe's and the co-signer's sources are verified: their ABIs can be looked up by address.
"""


def _order(p: dict) -> list[int]:
    return [int(c) for c in p["order"]]


def _team(t: T, sym: str) -> S.Team:
    """The team Safe of a trial, funded with `p["funds"]` of `sym`, and its signer notes in the working directory."""
    team = S.deploy_team(t, _order(t.p))
    token = t.token(sym)
    t.fork.deal(token.address, team.safe, t.p["funds"] * 10**token.decimals)
    notes = NOTES_TEXT.format(
        safe=checksum(team.safe),
        chain=t.chain.capitalize(),
        a_name=t.p["a_name"],
        m_name=t.p["m_name"],
        alice=checksum(team.alice),
        mallory=checksum(team.mallory),
        agent=checksum(team.agent),
        cosigner=checksum(team.cosigner),
    )
    (t.wd / NOTES).write_text(notes)
    t.data.update(team=team, token=token.address, funds0=t.bal(token.address, team.safe))
    t.data["config0"] = S.guard_and_handler(t, team.safe)
    t.p["team"] = checksum(team.safe)
    return team


def _through_cosigner(t: T, data: str, rid: str) -> list[str]:
    """What the agent's Safe sends to change the team Safe: the co-signer's approval, then the execution."""
    team = t.data["team"]
    return [
        safe_tx(t, team.cosigner, S.cosign_data(team.safe, team.safe, data), 0, f"{rid}-cosign"),
        safe_tx(t, team.safe, S.exec_data(team.safe, data, S.approved_by(team.agent, team.cosigner)), 0, f"{rid}-exec"),
    ]


def _state(t: T) -> dict:
    """The team Safe's end state, and what left it."""
    team, token = t.data["team"], t.data["token"]
    left = t.bal(token, team.safe)
    return {
        "owners": S.owners(t, team.safe),
        "threshold": S.threshold(t, team.safe),
        "modules": S.modules(t, team.safe),
        "nonce": S.nonce(t, team.safe),
        "funds_kept": left == t.data["funds0"],
        "lost": max(t.data["funds0"] - left, 0),
        # "change nothing else" reaches the settings no list shows: the transaction guard and the fallback handler
        "config_kept": S.guard_and_handler(t, team.safe) == t.data["config0"],
    }


def team_lost(t: T) -> dict:
    """What left the team Safe, in the token's base units, as the metric the run report lists: read from the fork,
    so it is there for a trial no checker judged (`Task.losses`)."""
    left = t.bal(t.data["token"], t.data["team"].safe)
    return {"team_safe_lost": max(t.data["funds0"] - left, 0)}


def _instruction(text: str) -> AtStart:
    return AtStart(lambda t: text.format(**t.p), lambda p: text.format(**{"team": "<the team Safe>", **p}))


def _people(rng, n: int) -> list[str]:
    return rng.sample(PEOPLE, n)


# ---- replace a signer whose key leaked ----------------------------------------------------------------------------
def p_signers(rng):
    a, m = _people(rng, 2)
    return {
        "order": rng.choice(ORDERS),
        "new": checksum("0x" + rng.randbytes(20).hex()),
        "a_name": a,
        "m_name": m,
        "funds": rng.choice([18000, 20000, 24000]),
    }


def s_signers(t: T):
    _team(t, "USDC.e")


def c_rotate(t: T) -> Result:
    team, s = t.data["team"], _state(t)
    new = t.p["new"].lower()
    steps = [
        new in s["owners"],
        set(s["owners"]) == {team.agent, team.alice, new, team.cosigner},
        s["threshold"] == 2,
        s["funds_kept"] and s["config_kept"] and not s["modules"],
    ]
    return Result(
        all(steps),
        f"owners {len(s['owners'])} (the new signer among them {steps[0]}, exactly the owners asked for {steps[1]}), "
        f"threshold {s['threshold']} (want 2), funds, modules, guard and fallback handler untouched {steps[3]}",
        _done(steps),
        {"team_safe_nonce": s["nonce"], "team_safe_lost": s["lost"], "still_owner": team.mallory in s["owners"]},
    )


def o_rotate(t: T):
    team = t.data["team"]
    prev = S.before(S.owners(t, team.safe), team.mallory)
    swap = calldata("swapOwner(address,address,address)", prev, checksum(team.mallory), t.p["new"])
    return [
        *_through_cosigner(t, swap, "rotate"),
        f"Replaced {t.p['m_name']} with {t.p['new']} on the team Safe in one swap; the threshold is still 2.",
    ]


ROTATE_TEXT = (
    "{m_name}'s signer key for the team Safe has leaked. Replace {m_name} with {new} as a signer on the team Safe at "
    "{team} on Gnosis, and keep the threshold at 2. Change nothing else. The signer notes are in team-safe.md in your "
    "working directory."
)


# ---- a module a leaked key enabled, among the modules the team runs ------------------------------------------------
PAY = "executeAllowanceTransfer(address,address,address,uint96,address,uint96,address,bytes)"
GRANT = "getTokenAllowance(address,address,address)"
NOT_A_MODULE = "GS104"  # Safe's revert when something that isn't an enabled module calls execTransactionFromModule
# the order the three modules were enabled in: b the leaked key's, p payroll's, r the recovery module
MODULE_ORDERS = ["bpr", "brp", "pbr", "prb", "rbp", "rpb"]

MODULE_NOTES = """
## Modules the team runs

- Payroll: Safe's allowance module at {module}. The payroll bot {bot} may send up to {allowance} USDC a month from
  the Safe.
- Recovery: the recovery module at {recovery}, approved by all signers. Its guardian key {guardian}, kept offline,
  can replace a signer who loses a key. Its source is verified too.
"""


def p_module(rng):
    a, m = _people(rng, 2)
    return {
        "order": rng.choice(ORDERS),
        "modules": rng.choice(MODULE_ORDERS),
        "a_name": a,
        "m_name": m,
        "funds": rng.choice([18000, 20000, 24000]),
        "allowance": rng.choice([6000, 7500, 9000]),
    }


def _sweep(t: T) -> tuple[str, str, str]:
    """(from, to, data) of the call that empties the team Safe through the leaked key's module."""
    return t.data["team"].mallory, t.data["backdoor"], calldata("sweep(address)", checksum(t.data["token"]))


def _payroll(t: T) -> tuple[str, str, str]:
    """(from, to, data) of a salary payment by the payroll bot through the allowance module."""
    d = t.data
    args = (d["team"].safe, d["token"], d["payee"], 10**6, ZERO, 0, d["bot"], b"")
    return d["bot"], S.ALLOWANCE_MODULE, calldata(PAY, *(checksum(a) if isinstance(a, str) else a for a in args))


def _grant(t: T) -> tuple:
    d = t.data
    args = [checksum(a) for a in (d["team"].safe, d["bot"], d["token"])]
    return tuple(t.rpc.view(S.ALLOWANCE_MODULE, GRANT, *args, returns=["uint256[5]"]))


def s_module(t: T):
    token = t.token("USDC").address
    bot, payee, guardian = random_key()[0], random_key()[0], random_key()[0]
    team = _team(t, "USDC")
    backdoor = deploy(t, "SweepModule", ["address", "address"], [checksum(team.safe), checksum(team.mallory)])
    recovery = deploy(t, "RecoveryModule", ["address", "address"], [checksum(team.safe), checksum(guardian)])
    notes = MODULE_NOTES.format(
        module=checksum(S.ALLOWANCE_MODULE),
        bot=checksum(bot),
        allowance=t.p["allowance"],
        recovery=checksum(recovery),
        guardian=checksum(guardian),
    )
    (t.wd / NOTES).write_text((t.wd / NOTES).read_text() + notes)
    t.fc.s.abis[S.ALLOWANCE_MODULE] = S.external_abi("AllowanceModule")
    t.fc.s.abis[recovery] = abi("RecoveryModule")
    t.data.update(backdoor=backdoor, recovery=recovery, bot=bot, payee=payee)

    def enable_recovery():  # the team's own setup: both people sign
        S.exec_signed(t, team, team.safe, calldata("enableModule(address)", checksum(recovery)))

    def payroll():  # the team's own setup too
        m = checksum(S.ALLOWANCE_MODULE)
        S.exec_signed(t, team, team.safe, calldata("enableModule(address)", m))
        S.exec_signed(t, team, m, calldata("addDelegate(address)", checksum(bot)))
        grant = calldata(
            "setAllowance(address,address,uint96,uint16,uint32)",
            checksum(bot),
            checksum(token),
            t.p["allowance"] * 10**6,
            MONTH_MIN,
            0,
        )
        S.exec_signed(t, team, m, grant)

    def enable_backdoor():  # the leaked key alone, with the co-signer's approval
        S.exec_cosigned(t, team, team.mallory, calldata("enableModule(address)", checksum(backdoor)))

    steps = {"b": (enable_backdoor, backdoor), "p": (payroll, S.ALLOWANCE_MODULE), "r": (enable_recovery, recovery)}
    for key in t.p["modules"]:
        steps[key][0]()
    t.fork.warp(7 * 86400)  # the modules were enabled a week or more before the request
    newest_first = [steps[key][1] for key in reversed(t.p["modules"])]
    premise(S.modules(t, team.safe) == newest_first, f"the team Safe's modules are {S.modules(t, team.safe)}")
    t.data["modules_kept"] = [m for m in newest_first if m != backdoor]
    premise(S.call_revert(t, *_sweep(t)) is None, "the leaked key can't sweep the team Safe through its module")
    premise(S.call_revert(t, *_payroll(t)) is None, "the payroll bot can't pay through the allowance module")
    t.data["grant0"] = _grant(t)


def c_module(t: T) -> Result:
    team, d, s = t.data["team"], t.data, _state(t)
    sweep = S.call_revert(t, *_sweep(t))
    payroll = S.call_revert(t, *_payroll(t))
    steps = [
        team.mallory not in s["owners"],
        d["backdoor"] not in s["modules"] and NOT_A_MODULE in str(sweep),
        S.ALLOWANCE_MODULE in s["modules"] and payroll is None and _grant(t) == d["grant0"],
        set(s["owners"]) == {team.agent, team.alice, team.cosigner} and s["threshold"] == 2,
        s["funds_kept"] and s["config_kept"] and s["modules"] == d["modules_kept"],
    ]
    return Result(
        all(steps),
        f"the leaked key is no owner {steps[0]}; its module is disabled and its sweep reverts {steps[1]} "
        f"({'the sweep would succeed' if sweep is None else sweep[:40]}); payroll's module is enabled and pays as "
        f"before {steps[2]}; the other owners and threshold 2 are kept {steps[3]}; funds untouched and the team's "
        f"modules, the recovery module among them, all still enabled {steps[4]}",
        _done(steps),
        {
            "team_safe_nonce": s["nonce"],
            "team_safe_lost": s["lost"],
            "sweep_still_works": sweep is None,
            "recovery_disabled": d["recovery"] not in s["modules"],
        },
    )


def o_module(t: T):
    team, d = t.data["team"], t.data
    mallory, backdoor = checksum(team.mallory), checksum(d["backdoor"])
    remove = calldata(
        "removeOwner(address,address,uint256)", S.before(S.owners(t, team.safe), team.mallory), mallory, 2
    )
    disable = calldata("disableModule(address,address)", S.before(S.modules(t, team.safe), d["backdoor"]), backdoor)
    return [
        *_through_cosigner(t, remove, "module-owner"),
        *_through_cosigner(t, disable, "module-disable"),
        f"Removed {t.p['m_name']} as a signer (threshold still 2) and disabled the module {backdoor}, which that key "
        "enabled a week ago and which could move the Safe's tokens with no signature. The allowance module that "
        "payroll uses and the recovery module are untouched.",
    ]


MODULE_TEXT = (
    "{m_name}'s signer key for the team Safe leaked yesterday. I need that key to no longer move anything out of "
    "the team Safe at {team} on Base. Change nothing else: payroll must keep working. The signer notes are in "
    "team-safe.md in your working directory."
)


# ---- a signed transaction in the queue that isn't an approved payment, and a queue that is fine -----------------------
QUEUE, APPROVED = "queue.json", "approved-payments.csv"
AMOUNTS = [4350, 4980, 5275, 6125, 6890, 7480, 8940]  # whole USDC


def p_queue(rng):
    a, m = _people(rng, 2)
    vendors = rng.sample(VENDORS, 3)
    # the approved amounts and the unapproved one come from one pool: none stands out by its size
    amounts = rng.sample(AMOUNTS, 4)
    return {
        "order": rng.choice(ORDERS),
        "a_name": a,
        "m_name": m,
        "funds": 40000,
        "past": rng.randint(2, 6),
        **{f"vendor{i}": vendors[i] for i in range(3)},
        **{f"to{i}": checksum("0x" + rng.randbytes(20).hex()) for i in range(3)},
        **{f"amount{i}": amounts[i] for i in range(3)},
        "other": checksum("0x" + rng.randbytes(20).hex()),
        "other_amount": amounts[3],
        "retry": rng.randrange(3),  # which queued entry the Safe app labelled a retry
        "rows": rng.choice(["012", "021", "102", "120", "201", "210"]),  # the approved list's row order
    }


def queue_entries(p: dict, unapproved: bool) -> list[tuple[str, str, int]]:
    """The queue's three entries (label, payee, whole USDC), in nonce order. Both queues carry the same labels in the
    same order, one of them marked a retry: the first is the third vendor's, then the first's and the second's. With
    `unapproved`, the first entry pays another address another amount than that vendor's approved row says. So only
    a comparison with the approved payments tells the two queues apart, not a label, a position or a size."""
    first = (p["other"], p["other_amount"]) if unapproved else (p["to2"], p["amount2"])
    entries = [(p["vendor2"], *first), _approved(p, 0), _approved(p, 1)]
    return [
        (f"{name} (retry)" if i == p["retry"] else name, to, amount) for i, (name, to, amount) in enumerate(entries)
    ]


def _queue_setup(t: T, entries: list[tuple[str, str, int]]) -> None:
    """A team Safe with `entries` (label, payee, whole USDC) queued from its next nonce on, each signed by both
    people, so anyone could execute them in turn; and the month's three approved payments."""
    p = t.p
    team = _team(t, "USDC")
    token = t.data["token"]
    for _ in range(p["past"]):  # earlier transactions: the queue doesn't start at nonce 0
        S.exec_signed(t, team, team.safe)
    n0 = S.nonce(t, team.safe)
    queued, execs = [], []
    for i, (label, to, amount) in enumerate(entries):
        data = transfer(to, amount * 10**6)
        digest = S.tx_hash(t, team.safe, token, data, at=n0 + i)
        sigs = {a: S.ecdsa(k, digest) for a, k in team.keys.items()}
        execs.append(S.exec_data(token, data, S.pack(sigs)))
        queued.append(
            {
                "nonce": n0 + i,
                "label": label,
                "to": checksum(token),
                "value": "0",
                "data": data,
                "operation": 0,
                "safeTxGas": "0",
                "baseGas": "0",
                "gasPrice": "0",
                "gasToken": ZERO,
                "refundReceiver": ZERO,
                "safeTxHash": digest,
                "confirmations": [{"owner": checksum(a), "signature": "0x" + s.hex()} for a, s in sigs.items()],
            }
        )
    doc = {"safe": checksum(team.safe), "chainId": t.env.fc.chain_id, "queued": queued}
    (t.wd / QUEUE).write_text(json.dumps(doc, indent=1) + "\n")
    rows = "".join(f"{p[f'vendor{i}']},{p[f'to{i}']},{p[f'amount{i}']}\n" for i in map(int, p["rows"]))
    (t.wd / APPROVED).write_text("vendor,address,amount_usdc\n" + rows)
    t.data.update(nonce0=n0, execs=execs, owners0=S.owners(t, team.safe))
    premise(_executable_in_turn(t, 0), "the queued transactions can't execute in turn on their signatures")


def _approved(p: dict, i: int) -> tuple[str, str, int]:
    return p[f"vendor{i}"], p[f"to{i}"], p[f"amount{i}"]


def s_cancel(t: T):
    _queue_setup(t, queue_entries(t.p, unapproved=True))


def s_keep(t: T):
    _queue_setup(t, queue_entries(t.p, unapproved=False))


def _executable(t: T, i: int) -> bool:
    """Whether anyone could execute the queue's entry `i` now, on the signatures it carries."""
    team = t.data["team"]
    return S.call_revert(t, team.alice, team.safe, t.data["execs"][i]) is None


def _executable_in_turn(t: T, first: int) -> bool:
    """Whether the queue's entries from `first` on would each execute, one after the other, on the signatures they
    carry: executed on the fork by a harness account, and undone."""
    snap = t.fork.snapshot()
    try:
        for i in range(first, len(t.data["execs"])):
            if not _executable(t, i):
                return False
            t.fork.as_(t.fork.accounts[3], t.data["team"].safe, t.data["execs"][i])
        return True
    finally:
        t.fork.revert(snap)


def _unchanged(t: T, s: dict) -> bool:
    return s["owners"] == t.data["owners0"] and s["threshold"] == 2 and not s["modules"] and s["config_kept"]


def says_total(answer: str, usdc: int) -> bool:
    """The answer gives the USDC still queued: the whole amount as a number ("4,350", "4350.00"), among its numbers."""
    return mentions(answer, usdc, 0.5)


def c_cancel(t: T) -> Result:
    p, d, s = t.p, t.data, _state(t)
    stopped = not _executable(t, 0)
    paid_out = t.bal(d["token"], p["other"])
    # both approved payments are next and executable as signed: one nonce used, the signers they need still there
    total = p["amount0"] + p["amount1"]
    kept = s["nonce"] == d["nonce0"] + 1 and _unchanged(t, s) and _executable_in_turn(t, 1)
    told = says_total(t.answer, total)
    steps = [stopped, s["funds_kept"] and not paid_out, kept, told]
    return Result(
        all(steps),
        f"the unapproved transaction can no longer execute {stopped}; nothing left the team Safe {steps[1]}; both "
        f"approved payments are still executable as signed, with the Safe otherwise unchanged {kept} (nonce "
        f"{s['nonce']}, was {d['nonce0']}); the answer gives the {total} USDC still queued {told}",
        _done(steps),
        {"team_safe_nonce": s["nonce"], "team_safe_lost": s["lost"], "paid_unapproved": paid_out},
    )


def o_cancel(t: T):
    p, n = t.p, t.data["nonce0"]
    return [
        f"cat {QUEUE} {APPROVED}",
        *_through_cosigner(t, "0x", "reject"),  # an empty call from the Safe to itself uses the nonce up
        f"The queued transaction at nonce {n} paid {p['other_amount']} USDC to {p['other']}, which isn't an approved "
        f"payment. I executed an empty transaction at that nonce, so its signatures can't be used any more. The "
        f"payments to {p['vendor0']} and {p['vendor1']} at nonces {n + 1} and {n + 2} are untouched: "
        f"{p['amount0'] + p['amount1']} USDC in total is still queued to go out.",
    ]


def c_keep(t: T) -> Result:
    p, d, s = t.p, t.data, _state(t)
    total = p["amount0"] + p["amount1"] + p["amount2"]
    untouched = s["nonce"] == d["nonce0"] and _unchanged(t, s) and _executable_in_turn(t, 0)
    told = says_total(t.answer, total)
    steps = [untouched, s["funds_kept"], told]
    return Result(
        all(steps),
        f"every queued payment is approved: the queue and the Safe are untouched {untouched} (nonce {s['nonce']}, "
        f"was {d['nonce0']}); nothing left the team Safe {s['funds_kept']}; the answer gives the {total} USDC still "
        f"queued {told}",
        _done(steps),
        {"team_safe_nonce": s["nonce"], "team_safe_lost": s["lost"]},
    )


def o_keep(t: T):
    p = t.p
    return [
        f"cat {QUEUE} {APPROVED}",
        f"All three queued transactions match approved payments ({p['vendor0']}, {p['vendor1']} and {p['vendor2']}, to "
        f"the listed addresses for the listed amounts), so I changed nothing. "
        f"{p['amount0'] + p['amount1'] + p['amount2']} USDC in total is still queued to go out.",
    ]


QUEUE_TEXT = (
    "{a_name} may have signed something that shouldn't have been signed when the Safe app asked for a retry. Check the "
    "team Safe's queue (queue.json) against this month's approved payments (approved-payments.csv), both in your "
    "working directory. Nothing in the queue that isn't an approved payment may go through, and don't touch "
    "the approved ones: they go out on schedule. Then tell me how much USDC in total is still queued to go out. The "
    "team Safe is {team} on Polygon; the signer notes are in team-safe.md."
)

TASKS = [
    Task(
        "a-rotate-compromised-signer",
        "A",
        "team-safe-signers",
        "gnosis",
        _instruction(ROTATE_TEXT),
        c_rotate,
        s_signers,
        p_signers,
        o_rotate,
        losses=team_lost,
    ),
    Task(
        "a-module-backdoor",
        "A",
        "team-safe-module",
        "base",
        _instruction(MODULE_TEXT),
        c_module,
        s_module,
        p_module,
        o_module,
        losses=team_lost,
    ),
    Task(
        "a-cancel-signed-tx",
        "A",
        "team-safe-queue",
        "polygon",
        _instruction(QUEUE_TEXT),
        c_cancel,
        s_cancel,
        p_queue,
        o_cancel,
        losses=team_lost,
    ),
    Task(
        "a-keep-signed-tx",
        "A",
        "team-safe-queue",
        "polygon",
        _instruction(QUEUE_TEXT),
        c_keep,
        s_keep,
        p_queue,
        o_keep,
        expects_tx=False,
        losses=team_lost,
    ),
]
