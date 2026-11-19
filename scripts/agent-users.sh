#!/bin/sh
# Create the agent users for agent isolation (onchainbench/isolation.py) and let them read what the arms run, nothing
# more. Linux, as root (sudo). The run then sets ONCHAINBENCH_AGENT_ISOLATION=1.
#
#   sudo scripts/agent-users.sh <slots> [--read DIR]... [--private DIR]...
#
#   <slots>        how many agent users: one per concurrent trial (the run's --workers)
#   --read DIR     the agents may read and run what is in DIR (the bench's package, vendor/, .venv, the uv Pythons,
#                  a CLI arm's own install), and pass through every directory above it
#   --private DIR  nobody but its owner may enter DIR (checkouts and results the agents must not read)
#
# Users ocb-agent-0 ... ocb-agent-<slots-1>, each with a group of its own, all in the group ocb-agents; no home, no
# login shell. Grants are ACLs for the group ocb-agents, so they open nothing to other users.
set -e
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }
N=$1
case "$N" in '' | *[!0-9]*) echo "usage: $0 <slots> [--read DIR]... [--private DIR]..."; exit 2 ;; esac
shift
command -v setfacl >/dev/null || { echo "setfacl missing: install the acl package"; exit 1; }

getent group ocb-agents >/dev/null || groupadd --system ocb-agents
i=0
while [ "$i" -lt "$N" ]; do
  u="ocb-agent-$i"
  id -u "$u" >/dev/null 2>&1 \
    || useradd --system --user-group --groups ocb-agents --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin "$u"
  i=$((i + 1))
done

while [ $# -gt 0 ]; do
  case "$1" in
    --read)
      d=$(readlink -f "$2")
      [ -e "$d" ] || { echo "no such path: $2"; exit 1; }
      setfacl -R -m g:ocb-agents:rX "$d"
      [ -d "$d" ] && find "$d" -type d -exec setfacl -m d:g:ocb-agents:rX {} +  # what is added later, too
      p=$(dirname "$d")
      while [ "$p" != / ]; do  # through, not into: x on every directory above it
        setfacl -m g:ocb-agents:x "$p"
        p=$(dirname "$p")
      done
      ;;
    --private)
      mkdir -p "$2"
      chmod -R o-rwx "$2"
      ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift 2
done
echo "agent users: ocb-agent-0..$((N - 1))"
