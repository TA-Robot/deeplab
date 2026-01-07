#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="$(cd "$(dirname "$0")/.." && pwd)/queue"
QUEUE_FILE="$QUEUE_DIR/queue.txt"
LOCK_FILE="$QUEUE_DIR/queue.lock"
LOG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/runs/grad-speedup/_logs"
WATCH=true
SLEEP_SEC=30

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--once] [--watch] [--sleep SEC] [--queue FILE]

Runs queued commands sequentially. Commands are read from QUEUE_FILE, one per line.
Lines starting with # or empty lines are ignored.

Options:
  --once        Exit when the queue is empty (default: watch).
  --watch       Keep polling for new jobs (default).
  --sleep SEC   Sleep seconds between polls (default: 30).
  --queue FILE  Use a custom queue file (default: queue/queue.txt).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      WATCH=false
      shift
      ;;
    --watch)
      WATCH=true
      shift
      ;;
    --sleep)
      SLEEP_SEC="$2"
      shift 2
      ;;
    --queue)
      QUEUE_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
 done

mkdir -p "$QUEUE_DIR" "$LOG_DIR"
: >"$LOCK_FILE"

next_job() {
  # Prints "<line_no>:<command>" or empty if none.
  awk 'NF && $0 !~ /^#/ {print NR ":" $0; exit}' "$QUEUE_FILE" 2>/dev/null || true
}

dequeue_job() {
  local skip_line="$1"
  local tmp
  tmp="$(mktemp)"
  awk -v skip="$skip_line" 'NR!=skip {print}' "$QUEUE_FILE" >"$tmp"
  mv "$tmp" "$QUEUE_FILE"
}

extract_run_id() {
  local cmd="$1"
  local run_id
  run_id=$(python - "$cmd" <<'PY'
import shlex, sys
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    parts = shlex.split(cmd)
except ValueError:
    print("")
    sys.exit(0)
run_id = ""
for i, token in enumerate(parts):
    if token == "--run-id" and i + 1 < len(parts):
        run_id = parts[i + 1]
        break
print(run_id)
PY
)
  echo "$run_id"
}

while true; do
  job_line=""
  job_no=""
  job_cmd=""

  # Lock queue file to avoid concurrent edits
  exec 200>"$LOCK_FILE"
  flock 200
  job_line=$(next_job)
  if [[ -n "$job_line" ]]; then
    job_no="${job_line%%:*}"
    job_cmd="${job_line#*:}"
    dequeue_job "$job_no"
  fi
  flock -u 200

  if [[ -z "$job_cmd" ]]; then
    if [[ "$WATCH" == "true" ]]; then
      sleep "$SLEEP_SEC"
      continue
    fi
    exit 0
  fi

  run_id=$(extract_run_id "$job_cmd")
  timestamp=$(date +"%Y%m%d-%H%M%S")
  if [[ -n "$run_id" ]]; then
    log_file="$LOG_DIR/queue-${run_id}.log"
  else
    log_file="$LOG_DIR/queue-${timestamp}.log"
  fi

  echo "[queue] running: $job_cmd" | tee -a "$log_file"
  if bash -lc "$job_cmd" >>"$log_file" 2>&1; then
    echo "[queue] done: $job_cmd" | tee -a "$log_file"
  else
    exit_code=$?
    echo "[queue] failed (exit=$exit_code): $job_cmd" | tee -a "$log_file"
  fi
 done
