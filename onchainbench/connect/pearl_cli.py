"""`pearl`: Connect's MCP tools as a shell command, for bash-only agents.

    pearl tools                         list tools
    pearl help <tool>                   a tool's description and parameters
    pearl <tool> '{"json": "args"}'     call with JSON arguments
    pearl <tool> key=value ...          call with key=value (or --key value) arguments; values parsed as JSON
                                        when they can be. Unknown or missing parameters are refused.

Finds the server through the nearest .mcp.json (cwd upwards), like signer_client.py.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .connect_client import ConnectClient, ConnectError


def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_args(argv: list[str]) -> dict:
    """One JSON object, or key=value / --key=value / --key value pairs."""
    if len(argv) == 1 and argv[0].lstrip().startswith("{"):
        return json.loads(argv[0])
    out, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" not in a:
            if i + 1 >= len(argv):
                raise ValueError(f"{a} needs a value")
            out[a[2:].replace("-", "_")] = _value(argv[i + 1])
            i += 2
            continue
        if "=" not in a:
            raise ValueError(f"expected key=value or one JSON object, got {a!r}")
        k, v = a.split("=", 1)
        out[k.lstrip("-").replace("-", "_")] = _value(v)
        i += 1
    return out


def check_args(tool: dict, args: dict) -> None:
    """Refuse keys the tool doesn't take (a misspelt cap would otherwise be silently dropped)."""
    import difflib

    props = (tool.get("inputSchema") or {}).get("properties") or {}
    unknown = [k for k in args if k not in props]
    if unknown:

        def near(k):
            m = [p for p in props if k in p or p in k] or difflib.get_close_matches(k, props, 1)
            return f"{k} (did you mean {m[0]}?)" if m else k

        hints = [near(k) for k in unknown]
        raise ValueError(f"{tool['name']} does not take: {', '.join(hints)}. Parameters: {', '.join(props)}")
    missing = [k for k in (tool.get("inputSchema") or {}).get("required") or [] if k not in args]
    if missing:
        raise ValueError(f"{tool['name']} needs: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    try:
        client = ConnectClient.from_mcp_json()
        cmd, rest = argv[0], argv[1:]
        if cmd == "tools":
            for t in client.list_tools():
                first = (t.get("description") or "").strip().splitlines()
                print(f"{t['name']:24} {first[0] if first else ''}")
            return 0
        if cmd == "help":
            tools = {t["name"]: t for t in client.list_tools()}
            if not rest or rest[0] not in tools:
                print(f"unknown tool; available: {', '.join(sorted(tools))}", file=sys.stderr)
                return 2
            t = tools[rest[0]]
            print(t.get("description", ""))
            print("\nparameters:", json.dumps(t.get("inputSchema", {}), indent=2))
            return 0
        tools = {t["name"]: t for t in client.list_tools()}
        if cmd not in tools:
            print(f"error: unknown tool {cmd!r}; available: {', '.join(sorted(tools))}", file=sys.stderr)
            return 2
        args = parse_args(rest)
        check_args(tools[cmd], args)
        res = client.call_tool_raw(cmd, args)
        from .connect_client import decode_tool_result

        value = decode_tool_result(res)
        text = value if isinstance(value, str) else json.dumps(value, indent=2)
        if res.get("isError"):
            print(f"error: {text}", file=sys.stderr)
            return 1
        print(text)
        return 0
    except (ConnectError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
