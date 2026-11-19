# Interim code

Everything in this package goes away once Connect ships the feature it stands in for. The table names each one.

| Module | Goes away with | Replaced by | Used from |
|---|---|---|---|
| `intent_gate.py` (`IntentGate`) | Connect's intent gate: Connect simulates every value call, checks the asset diff against a declared intent, and serves `simulate_transaction`, `chain_call`, `get_abi`, `decode_calldata` | Connect's own code path, which fork-connect then calls | `fork_connect.ForkConnect` (a mixin); `_send` calls the gate |
| `readonly_fallback.py` (`FALLBACK_READ_ONLY`) | Connect flagging its read-only tools itself (`readOnlyHint`) | Connect's flags (`policy.read_only_from`) | `connect.policy.Policy` |

**Switching to Connect's gate changes what a run measures,** so it is a new bench version: results across it aren't
compared without saying so.
