#!/usr/bin/env bash
set -euo pipefail

WAIT_RUN="project/runs/grad-speedup/20260106-grad-speedup-step-l0l1-soap-resnet18-maxsteps14000-seeds012/summary.json"
LOG_DIR="project/runs/grad-speedup/_logs"
mkdir -p "${LOG_DIR}"

if [ -f "${WAIT_RUN}" ]; then
  while [ ! -f "${WAIT_RUN}" ]; do
    sleep 120
  done
else
  echo "WAIT_RUN not found; proceeding without wait: ${WAIT_RUN}" >> "${LOG_DIR}/20260106-grad-speedup-soap-fast-chain.log"
fi

python -u project/grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-baseline-short-resnet18-maxsteps2000-seed0 \
  --model resnet18 \
  --epochs 50 \
  --max-steps 2000 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --seed 0 \
  --data-seed 123 \
  --val-size 5000 \
  --num-workers 4 \
  --device cuda:0 \
  --log-interval-steps 100 \
  --eval-interval-epochs 0 \
  --eval-interval-steps 1000 \
  --target-acc 0.85 \
  --early-stop max \
  --warmup-steps 50 \
  --measure-steps 200 \
  > "${LOG_DIR}/20260106-grad-speedup-baseline-short-resnet18-maxsteps2000-seed0.log" 2>&1

python -u project/grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-soap-fast-f10-resnet18-maxsteps2000-seed0 \
  --model resnet18 \
  --epochs 50 \
  --max-steps 2000 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --seed 0 \
  --data-seed 123 \
  --val-size 5000 \
  --num-workers 4 \
  --device cuda:0 \
  --log-interval-steps 100 \
  --eval-interval-epochs 0 \
  --eval-interval-steps 1000 \
  --target-acc 0.85 \
  --early-stop max \
  --warmup-steps 50 \
  --measure-steps 200 \
  --direction soap \
  --direction-update-every 10 \
  > "${LOG_DIR}/20260106-grad-speedup-soap-fast-f10-resnet18-maxsteps2000-seed0.log" 2>&1

python -u project/grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-soap-fast-f50-resnet18-maxsteps2000-seed0 \
  --model resnet18 \
  --epochs 50 \
  --max-steps 2000 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --seed 0 \
  --data-seed 123 \
  --val-size 5000 \
  --num-workers 4 \
  --device cuda:0 \
  --log-interval-steps 100 \
  --eval-interval-epochs 0 \
  --eval-interval-steps 1000 \
  --target-acc 0.85 \
  --early-stop max \
  --warmup-steps 50 \
  --measure-steps 200 \
  --direction soap \
  --direction-update-every 50 \
  > "${LOG_DIR}/20260106-grad-speedup-soap-fast-f50-resnet18-maxsteps2000-seed0.log" 2>&1
