#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

DATA_DIR="${DATA_DIR:-data}"
RUNS_DIR="${RUNS_DIR:-runs}"
LOG_DIR="${LOG_DIR:-${RUNS_DIR}/logs}"
DATASETS="${DATASETS:-mnist,fashion-mnist,cifar10}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-1,2,3}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_THREADS="${NUM_THREADS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_RUNNING="${SKIP_RUNNING:-1}"
OBL_PROFILE="${OBL_PROFILE:-full}"

mkdir -p "${LOG_DIR}"

if [[ "${DEVICE}" == cuda* ]]; then
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

is_running() {
    local run_id="$1"
    pgrep -f "run_mnist_experiment.py .* --run-id ${run_id}" >/dev/null 2>&1
}

download_dataset() {
    local name="$1"
    python - <<'PY' "${name}" "${DATA_DIR}"
import sys
from torchvision import datasets

name = sys.argv[1]
data_dir = sys.argv[2]
registry = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}
cls = registry.get(name)
if cls is None:
    raise SystemExit(f"Unsupported dataset: {name}")
cls(data_dir, train=True, download=True)
cls(data_dir, train=False, download=True)
PY
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

download_pids=()
download_names=()
for dataset in $(echo "${DATASETS}" | tr "," " "); do
    download_log="${LOG_DIR}/${RUN_DATE}-download-${dataset}.out"
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] download start ${dataset} -> ${download_log}"
    download_dataset "${dataset}" > "${download_log}" 2>&1 &
    download_pids+=("$!")
    download_names+=("${dataset}")
done

for i in "${!download_pids[@]}"; do
    pid="${download_pids[$i]}"
    dataset="${download_names[$i]}"
    if wait "${pid}"; then
        echo "[$(date +%Y-%m-%dT%H:%M:%S)] download done ${dataset}"
    else
        echo "[$(date +%Y-%m-%dT%H:%M:%S)] download failed ${dataset}" >&2
    fi
done

for dataset in $(echo "${DATASETS}" | tr "," " "); do
    for model in mlp cnn mlp-obl cnn-obl; do
        run_id="${RUN_DATE}-ros-alth-gpu-${dataset}-${model}"
        run_dir="${RUNS_DIR}/${run_id}"
        if [[ "${SKIP_RUNNING}" == "1" ]] && is_running "${run_id}"; then
            echo "[$(date +%Y-%m-%dT%H:%M:%S)] skip ${run_id} (already running)"
            continue
        fi
        if [[ "${SKIP_EXISTING}" == "1" && -d "${run_dir}" ]]; then
            echo "[$(date +%Y-%m-%dT%H:%M:%S)] skip ${run_id} (run dir exists)"
            continue
        fi
        base_cmd="python run_mnist_experiment.py --model ${model} --dataset ${dataset} --seeds ${SEEDS} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --num-threads ${NUM_THREADS} --deterministic --device ${DEVICE} --run-id ${run_id}"
        if [[ "${model}" == *"-obl" ]]; then
            base_cmd="${base_cmd} --obl-profile ${OBL_PROFILE} --gamma 0.1 --beta-init 0.01"
        fi
        run_cmd "${base_cmd} > ${LOG_DIR}/${run_id}.out 2>&1"
    done
done

wait
