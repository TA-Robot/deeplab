#!/usr/bin/env bash
set -euo pipefail

config_dir="${1:-configs}"

if [[ ! -d "${config_dir}" ]]; then
  echo "Config dir not found: ${config_dir}" >&2
  exit 1
fi

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-$(date +%H%M%S)}"

for cfg in "${config_dir}"/*.json; do
  name="$(basename "${cfg}" .json)"
  run_id="${RUN_DATE}-grad-speedup-cifar10-${name}-${RUN_TAG}"
  echo "[${run_id}] ${cfg}"
  python scripts/run_cifar10.py --config "${cfg}" --run-id "${run_id}"
  echo
 done
