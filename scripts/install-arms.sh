#!/bin/sh
# Install the external harnesses OnchainBench compares, pinned, into vendor/ (not committed), and check each one can
# load an MCP server: an arm that can't never sees Connect's tools (Codex; Hermes without its [mcp] extra).
#
#   scripts/install-arms.sh                      # everything
#   scripts/install-arms.sh --for "<arm spec>"   # only what that arm needs (the scripted arms: none)
#   scripts/install-arms.sh codex hermes         # named parts: codex openclaw hermes claude-code evm-mcp
#                                                # olas-ocb-agent
#   scripts/install-arms.sh --help               # this usage
#
# Every contestant is a CLI: olas-ocb-agent installs its `ocb-agent`.
# Run from anywhere. Needs node/npm (Node >= 24 for OpenClaw) for codex, openclaw, claude-code and evm-mcp; uv and curl
# for hermes; uv for olas-ocb-agent (and git with OCB_AGENT_REF).
set -e
ALL="codex openclaw hermes claude-code evm-mcp olas-ocb-agent"
usage() {
  echo "usage: scripts/install-arms.sh [--for '<arm spec>' | <part>...]"
  echo "  no arguments: every part ($ALL)"
  echo "  --for SPEC:   only what that arm needs (the scripted arms need nothing)"
  echo "  <part>...:    the named parts, from: $ALL"
  echo "  env OCB_AGENT_REF=<git ref>: install olas-ocb-agent from its git repo at that ref, not the pinned release"
}
case "$1" in -h | --help) usage; exit 0 ;; esac
cd "$(dirname "$0")/.."
V=vendor
# the bench's Python, the one the run uses (the active venv, else .venv), never a bare python on PATH
PY="${VIRTUAL_ENV:-$PWD/.venv}/bin/python"

parts_for() {  # the parts an arm spec needs: its harness, and its wallet adapter (|wallet=evm-mcp, or *-raw)
  kind=${1%%[:|]*}
  case "$kind" in
    codex) echo codex ;;
    openclaw) echo openclaw ;;
    hermes) echo hermes ;;
    claude-code) echo claude-code ;;
    olas-ocb-agent | olas-ocb-agent-mech) echo olas-ocb-agent ;;
    hermes-raw) echo hermes evm-mcp ;;
    openclaw-raw) echo openclaw evm-mcp ;;
  esac
  case "$1" in *"|wallet=evm-mcp"*) echo evm-mcp ;; esac
}

if [ "$1" = "--for" ]; then
  PARTS=$(parts_for "$2" | tr '\n' ' ')
elif [ $# -gt 0 ]; then
  for part in "$@"; do
    case " $ALL " in
      *" $part "*) ;;
      *) echo "unknown part: $part (the parts: $ALL)" >&2; usage >&2; exit 2 ;;
    esac
  done
  PARTS="$*"
else
  PARTS="$ALL"
fi
[ -n "$(echo $PARTS)" ] || { echo "nothing to install for this arm"; exit 0; }
mkdir -p "$V"

has() { case " $PARTS " in *" $1 "*) return 0 ;; esac; return 1; }

NPM=""
has codex && NPM="$NPM @openai/codex@0.170.0"
OPENCLAW_PKG=openclaw@2026.11.5
has openclaw && NPM="$NPM $OPENCLAW_PKG"
if has openclaw; then
  # npm 11 skips dependencies' install scripts unless package.json's allowScripts names them; OpenClaw's postinstall
  # (prune stale dist files, restore fs-safe's native addon, clear its lifecycle marker) must run here, as the
  # installer: left pending, the CLI finishes it on its first launch, which under isolation is an agent user's
  mkdir -p "$V"
  node -e 'const fs = require("fs"), f = process.argv[1]; const p = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : {};
    p.allowScripts = {...(p.allowScripts || {}), [process.argv[2]]: true}; fs.writeFileSync(f, JSON.stringify(p, null, 2) + "\n");' \
    "$V/package.json" "$OPENCLAW_PKG"
fi
has claude-code && NPM="$NPM @anthropic-ai/claude-code@2.1.327"
has evm-mcp && NPM="$NPM @mcpdotdirect/evm-mcp-server@2.0.4"
# shellcheck disable=SC2086 # the package list is meant to split
[ -n "$NPM" ] && npm install --prefix "$V" $NPM

if has hermes; then
  # Hermes v0.23.2 refuses wheel builds: install the tag's source editable, WITH its [mcp] extra, and its overrides
  HERMES=hermes-agent-0.23.2
  if [ ! -d "$V/hermes-src/$HERMES" ]; then
    mkdir -p "$V/hermes-src"
    # to a file first, failing on an HTTP error and retrying: an error or rate-limit page piped into tar only said
    # "not in gzip format"
    HERMES_TGZ="$V/hermes-src/$HERMES.tar.gz"
    curl -fsSL --retry 3 --retry-delay 5 --retry-all-errors \
      -o "$HERMES_TGZ" https://codeload.github.com/NousResearch/hermes-agent/tar.gz/refs/tags/v0.23.2 \
      || { echo "couldn't download Hermes v0.23.2 from GitHub (see curl's error above)" >&2; exit 1; }
    tar -xzf "$HERMES_TGZ" -C "$V/hermes-src" && rm -f "$HERMES_TGZ"
  fi
  [ -d "$V/hermes-venv" ] || uv venv "$V/hermes-venv" --python 3.14  # Hermes imports ruamel, a dependency only on 3.14+
  # uv's --override takes a file: Hermes's pins, written into vendor/ (not committed)
  printf '%s\n' 'pynacl>=1.6,<1.7' 'cryptography==50.0.2' 'pillow==13.0.0' > "$V/hermes-overrides.txt"
  VIRTUAL_ENV="$V/hermes-venv" uv pip install -e "$V/hermes-src/$HERMES[mcp]" \
    -r "$V/hermes-src/$HERMES/pyproject.toml" --override "$V/hermes-overrides.txt"
  "$V/hermes-venv/bin/python3" -c "import mcp.client.streamable_http, ruamel.yaml" \
    || { echo "Hermes can't load MCP servers or its config library: [mcp] extra or Python 3.14 missing"; exit 1; }
fi

if has codex; then
  CODEX_HOME=$(mktemp -d) && printf '[mcp_servers.check]\nurl = "http://127.0.0.1:1/mcp/"\ndefault_tools_approval_mode = "approve"\n' \
    > "$CODEX_HOME/config.toml" && CODEX_HOME="$CODEX_HOME" "$V/node_modules/.bin/codex" mcp list >/dev/null \
    || { echo "Codex rejects the MCP config the bench writes"; exit 1; }
fi
if has openclaw; then
  [ ! -e "$V/node_modules/openclaw/.openclaw-lifecycle-pending" ] || { echo "OpenClaw's postinstall didn't run"; exit 1; }
  "$V/node_modules/.bin/openclaw" --version >/dev/null || { echo "OpenClaw isn't runnable"; exit 1; }
  # `openclaw mcp probe` (no model call) through the bench's broker, as a trial wires it: Connect's tools must load
  "$PY" -c "import sys; from onchainbench.arms import openclaw_mcp_problem as p; m = p(); print(m or 'OpenClaw loads Connect tools'); sys.exit(bool(m))" \
    || { echo "OpenClaw can't load Connect's tools through the broker"; exit 1; }
fi
if has claude-code; then
  "$V/node_modules/.bin/claude" --version >/dev/null || { echo "Claude Code isn't runnable"; exit 1; }
fi
# olas-ocb-agent's CLI `ocb-agent`, the pinned release from PyPI, into an environment of its own
# (vendor/ocb-agent-venv), like Hermes's. To move to another release, change this one line. OCB_AGENT_REF=<git ref>
# installs that ref from the agent's git repo instead (a published run's config.json records the agent's digest). One
# already installed in the bench's environment (a checkout, as the GitHub runs install it) is used instead, if it is
# the pinned version.
OCB_AGENT_VERSION=0.1.0
if [ -n "$OCB_AGENT_REF" ]; then
  OCB_AGENT="olas-ocb-agent[litellm,facilitator] @ git+https://github.com/valory-xyz/olas-ocb-agent@$OCB_AGENT_REF"
else
  OCB_AGENT="olas-ocb-agent[litellm,facilitator]==$OCB_AGENT_VERSION"
fi
if has olas-ocb-agent; then
  OCB_BIN="$(dirname "$PY")/ocb-agent"
  if [ ! -x "$OCB_BIN" ]; then
    [ -d "$V/ocb-agent-venv" ] || uv venv "$V/ocb-agent-venv" --python 3.12
    VIRTUAL_ENV="$V/ocb-agent-venv" uv pip install "$OCB_AGENT"
    OCB_BIN="$V/ocb-agent-venv/bin/ocb-agent"
  fi
  HAVE=$("$(dirname "$OCB_BIN")/python" -c "import importlib.metadata as m; print(m.version('olas-ocb-agent'))")
  [ -n "$OCB_AGENT_REF" ] || [ "$HAVE" = "$OCB_AGENT_VERSION" ] \
    || { echo "olas-ocb-agent is $HAVE at $OCB_BIN, not the pinned $OCB_AGENT_VERSION"; exit 1; }
  "$OCB_BIN" --help | grep -q -- "--turn-seconds" || { echo "ocb-agent lacks the headless flags (0.1.0+)"; exit 1; }
  # LiteLLM's retries (num_retries) import tenacity lazily: without it the first transient provider error ends the
  # turn as the harness's error
  "$(dirname "$OCB_BIN")/python" -c "import tenacity" \
    || { echo "olas-ocb-agent's environment lacks tenacity: LiteLLM's retries would fail (install its [litellm] extra)"; exit 1; }
fi
echo "installed and checked: $PARTS"
