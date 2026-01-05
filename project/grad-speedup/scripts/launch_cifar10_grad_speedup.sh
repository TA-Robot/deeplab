#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-$(date +%H%M%S)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/grad-speedup}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-5}"
SEEDS="${SEEDS:-0,1,2}"
DATA_SEED="${DATA_SEED:-123}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_THREADS="${NUM_THREADS:-0}"
DATA_DIR="${DATA_DIR:-data}"
MODEL="${MODEL:-resnet18}"
DETERMINISTIC="${DETERMINISTIC:-1}"
DOWNLOAD="${DOWNLOAD:-0}"

SGD_LR="${SGD_LR:-0.1}"
SGD_MOMENTUM="${SGD_MOMENTUM:-0.9}"
ADAM_LR="${ADAM_LR:-1e-3}"

common_args=(
    --model "${MODEL}"
    --output-root "${OUTPUT_ROOT}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --device "${DEVICE}"
    --seeds "${SEEDS}"
    --data-seed "${DATA_SEED}"
    --num-workers "${NUM_WORKERS}"
    --num-threads "${NUM_THREADS}"
    --data-dir "${DATA_DIR}"
)

if [[ "${DETERMINISTIC}" == "1" ]]; then
    common_args+=(--deterministic)
fi

if [[ "${DOWNLOAD}" == "1" ]]; then
    common_args+=(--download)
fi

run_variant() {
    local variant="$1"
    shift
    local run_id="${RUN_DATE}-grad-speedup-cifar10-${variant}-${RUN_TAG}"
    echo "[${run_id}]"
    python scripts/run_cifar10.py "${common_args[@]}" --run-id "${run_id}" "$@"
}

run_variant "baseline-sgd" --optimizer sgd --lr "${SGD_LR}" --momentum "${SGD_MOMENTUM}"
run_variant "baseline-adam" --optimizer adam --lr "${ADAM_LR}"
