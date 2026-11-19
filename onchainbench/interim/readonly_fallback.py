"""Connect's read-only tools, until Connect flags them itself.

Interim: goes away once Connect flags its read-only tools itself, with MCP's `readOnlyHint`; the policy already
prefers Connect's own flags as soon as its tool list carries any.
"""

FALLBACK_READ_ONLY = {"wallet_info", "mech_tools", "mech_result", "transaction_status", "preflight_transaction"}
