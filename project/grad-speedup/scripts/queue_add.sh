#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="$(cd "$(dirname "$0")/.." && pwd)/queue"
QUEUE_FILE="$QUEUE_DIR/queue.txt"

usage() {
  cat <<USAGE
Usage: $(basename "$0") "<command>"

Appends a command to the grad-speedup queue file.
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mkdir -p "$QUEUE_DIR"
cmd="$*"
# Append as-is; caller is responsible for quoting.
echo "$cmd" >>"$QUEUE_FILE"

echo "queued: $cmd"
