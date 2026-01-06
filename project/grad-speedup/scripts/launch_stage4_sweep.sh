#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
cd "${project_root}"

RUNS_DIR="${RUNS_DIR:-runs/grad-speedup}"
LOG_DIR="${LOG_DIR:-${RUNS_DIR}/logs}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
CLIP_VALUES="${CLIP_VALUES:-none,ggnc-global,ggnc-layerwise}"
OUTER_VALUES="${OUTER_VALUES:-none,anderson}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_RUNNING="${SKIP_RUNNING:-1}"
STAGE4_CMD_TEMPLATE="${STAGE4_CMD_TEMPLATE:-}"

STEP_RULE=""
DIRECTION=""

usage() {
    cat <<'EOF'
Usage: launch_stage4_sweep.sh --step-rule <value> --direction <value>

Required env:
  STAGE4_CMD_TEMPLATE must include placeholders:
    {step_rule}, {direction}, {clip}, {outer}, {run_id}

Example:
  STAGE4_CMD_TEMPLATE="python scripts/run_cifar10.py --model small-cnn --epochs 1 --device cuda:0 \
    --step-rule {step_rule} --direction {direction} --clip-mode {clip} \
    --anderson-memory {outer} --run-id {run_id}" \
  OUTER_VALUES="0,3" \
  bash project/grad-speedup/scripts/launch_stage4_sweep.sh --step-rule l0l1 --direction soap
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step-rule)
            STEP_RULE="$2"
            shift 2
            ;;
        --direction)
            DIRECTION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${STEP_RULE}" || -z "${DIRECTION}" ]]; then
    echo "Both --step-rule and --direction are required." >&2
    usage >&2
    exit 1
fi

if [[ -z "${STAGE4_CMD_TEMPLATE}" ]]; then
    echo "STAGE4_CMD_TEMPLATE is required." >&2
    usage >&2
    exit 1
fi

required_tokens=("{step_rule}" "{direction}" "{clip}" "{outer}" "{run_id}")
for token in "${required_tokens[@]}"; do
    if [[ "${STAGE4_CMD_TEMPLATE}" != *"${token}"* ]]; then
        echo "STAGE4_CMD_TEMPLATE must include ${token}." >&2
        exit 1
    fi
done

mkdir -p "${LOG_DIR}"

is_running() {
    local run_id="$1"
    pgrep -f " --run-id ${run_id}" >/dev/null 2>&1
}

active=0
run_cmd() {
    local cmd="$1"
    bash -lc "${cmd}" &
    ((active+=1))
    if (( MAX_PARALLEL > 0 && active >= MAX_PARALLEL )); then
        wait -n
        ((active-=1))
    fi
}

for clip in $(echo "${CLIP_VALUES}" | tr "," " "); do
    for outer in $(echo "${OUTER_VALUES}" | tr "," " "); do
        run_id="${RUN_DATE}-grad-speedup-stage4-${STEP_RULE}-${DIRECTION}-${clip}-${outer}-smallcnn-gpu"
        run_dir="${RUNS_DIR}/${run_id}"
        if [[ "${SKIP_RUNNING}" == "1" ]] && is_running "${run_id}"; then
            echo "[$(date +%Y-%m-%dT%H:%M:%S)] skip ${run_id} (already running)"
            continue
        fi
        if [[ "${SKIP_EXISTING}" == "1" && -d "${run_dir}" ]]; then
            echo "[$(date +%Y-%m-%dT%H:%M:%S)] skip ${run_id} (run dir exists)"
            continue
        fi
        cmd="${STAGE4_CMD_TEMPLATE//\{step_rule\}/${STEP_RULE}}"
        cmd="${cmd//\{direction\}/${DIRECTION}}"
        cmd="${cmd//\{clip\}/${clip}}"
        cmd="${cmd//\{outer\}/${outer}}"
        cmd="${cmd//\{run_id\}/${run_id}}"
        echo "[$(date +%Y-%m-%dT%H:%M:%S)] launch ${run_id}"
        run_cmd "${cmd} > ${LOG_DIR}/${run_id}.out 2>&1"
    done
done

wait
