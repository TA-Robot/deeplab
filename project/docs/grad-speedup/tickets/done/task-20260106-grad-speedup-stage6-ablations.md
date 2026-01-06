# Task Ticket: Stage6 Ablations (ResNet18, 20 epochs, seeds 0/1/2)

Status: superseded (2026-01-06)
- Reason: migration to step-based control (max_steps=14,000, eval_interval_steps=1000); epoch-based ablations should not proceed.

## 1) Meta（識別情報）

- **ticket_id**: `task-20260106-grad-speedup-stage6-ablations`
- **role/agent**: `implementer-stage6-ablations`
- **owner（manager）**: `PM`
- **created_at**: `2026-01-06`
- **priority**: `P0`
- **timebox**: `GPU run: ~3-5h wall; report updates 30m`
- **workspace_scope**: `project/grad-speedup/`
- **related**
  - issue: `grad-speedup Stage6 ablations`
  - pr/branch: `n/a`
  - commits: `n/a`

---

## 2) Goal / Desired Outcome（目的・達成状態）

Stage6 ablations are executed on ResNet18 with the same training budget as the 20‑epoch promotion runs. We need multi‑seed runs (0/1/2) for three variants (l0l1-only, soap-only, l0l1+soap w/o anderson) so we can attribute gains and compare against baseline and the winner.

### What success looks like（成功の見え方）

- Ablation runs complete without errors and produce `summary.json` + per‑seed summaries under `project/runs/grad-speedup/`.
- Each run has a stable run ID that matches the naming scheme below.
- PM can compare cost‑to‑target and accuracy vs baseline and winner using the dashboard or scripts.

---

## 3) Background / Context（背景・前提・現状）

### Why now（なぜ今やるか）

Stage6 promotion runs are underway. We need ablations in the same 20‑epoch budget before deciding whether the Stage4 winner’s gains come from step‑rule, direction, or Anderson.

### Current behavior（現状のふるまい）

- Baseline 20‑epoch run is complete.
- Winner (l0l1 + soap + anderson) seed‑0 is complete; seeds 1/2 are running now.

### Constraints already known（既知の制約）

- Single GPU; do not run concurrent GPU jobs.
- Keep configs aligned with Stage6 baseline (batch size, lr, epochs, targets, etc.).

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- Run the following ablations (ResNet18, 20 epochs, seeds 0/1/2):
  1. l0l1‑only (direction=none, anderson off)
  2. soap‑only (step_rule=none, direction=soap, momentum=0.9)
  3. l0l1 + soap (direction=soap, anderson off)
- Save logs under `project/runs/grad-speedup/_logs/`.
- Ensure each run produces `summary.json` in the run root and `seed-*/summary.json`.

### Out of scope（やらないこと）

- No hyper‑parameter tuning beyond the fixed settings.
- No dashboard/report generation.

### Impacted areas（影響範囲）

- `project/grad-speedup/scripts/run_cifar10.py`
- `project/runs/grad-speedup/` (new run directories)

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- Use the run IDs exactly:
  - `20260106-grad-speedup-stage6-l0l1-only-resnet18-ep20-seeds012`
  - `20260106-grad-speedup-stage6-soap-only-resnet18-ep20-seeds012`
  - `20260106-grad-speedup-stage6-l0l1-soap-resnet18-ep20-seeds012`
- Use seeds `0,1,2`, epochs `20`, batch size `128`, lr `0.1`, weight decay `0.0005`.
- Match Stage6 baseline targets (`0.85,0.9,0.92,0.94`) and logging intervals.

### Should（できれば）

- Note GPU time and any anomalies in a brief comment to the manager.

### Nice（余裕があれば）

- Extract epoch20 test accuracy mean/std for each ablation.

---

## 6) Acceptance Criteria（完了条件：客観・機械的・再現可能）

- [ ] **Behavior**: All three runs finish without error and have run roots in `project/runs/grad-speedup/`.
- [ ] **Artifacts**: `summary.json` exists for each run root and for each seed.
- [ ] **Logs**: Log files exist under `project/runs/grad-speedup/_logs/`.
- [ ] **Docs**: Provide run IDs and any anomalies to the manager (no report required).

---

## 7) Implementation Notes（実装方針・設計メモ）

### Suggested approach（推奨アプローチ）

- Reuse Stage6 baseline settings; only change step_rule/direction/anderson.
- Soap‑only should keep SGD momentum at 0.9; l0l1 variants use momentum 0.0.

### Files / Modules to touch（触る可能性のある箇所）

- `project/grad-speedup/scripts/run_cifar10.py`
- `project/runs/grad-speedup/`

---

## 8) Commands（実行・検証コマンド）

```bash
cd project

# l0l1-only
python -u grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-stage6-l0l1-only-resnet18-ep20-seeds012 \
  --model resnet18 --epochs 20 --batch-size 128 --lr 0.1 --optimizer sgd \
  --momentum 0.0 --weight-decay 0.0005 --seeds 0,1,2 --data-seed 123 --val-size 5000 \
  --num-workers 4 --device cuda:0 --log-interval-steps 100 --eval-interval-epochs 1 \
  --target-acc 0.85,0.9,0.92,0.94 --early-stop max --warmup-steps 50 --measure-steps 200 \
  --step-rule l0l1 --step-l0 1.0 --step-l1 0.1 --direction none --clip-mode none --sparsity none

# soap-only
python -u grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-stage6-soap-only-resnet18-ep20-seeds012 \
  --model resnet18 --epochs 20 --batch-size 128 --lr 0.1 --optimizer sgd \
  --momentum 0.9 --weight-decay 0.0005 --seeds 0,1,2 --data-seed 123 --val-size 5000 \
  --num-workers 4 --device cuda:0 --log-interval-steps 100 --eval-interval-epochs 1 \
  --target-acc 0.85,0.9,0.92,0.94 --early-stop max --warmup-steps 50 --measure-steps 200 \
  --step-rule none --direction soap --clip-mode none --sparsity none

# l0l1 + soap (no anderson)
python -u grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-stage6-l0l1-soap-resnet18-ep20-seeds012 \
  --model resnet18 --epochs 20 --batch-size 128 --lr 0.1 --optimizer sgd \
  --momentum 0.0 --weight-decay 0.0005 --seeds 0,1,2 --data-seed 123 --val-size 5000 \
  --num-workers 4 --device cuda:0 --log-interval-steps 100 --eval-interval-epochs 1 \
  --target-acc 0.85,0.9,0.92,0.94 --early-stop max --warmup-steps 50 --measure-steps 200 \
  --step-rule l0l1 --step-l0 1.0 --step-l1 0.1 --direction soap \
  --anderson-memory 0 --anderson-interval 0 --clip-mode none --sparsity none
```

---

## 9) Deliverables（成果物：何を出して終わるか）

- **Run directories** under `project/runs/grad-speedup/` for the three ablations.
- **Logs** in `project/runs/grad-speedup/_logs/` with the same run IDs.
- **Notes for manager**: any anomalies and approximate runtime.

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

### Risks（リスク）

- GPU contention if jobs overlap.
- Partial runs if timeouts occur.

### Edge cases（境界条件）

- Seed directories might already exist if a run is accidentally restarted; avoid overwriting.

---

## 11) Open Questions（未確定事項：質問しないと進めない点）

- None.

---

## 12) Constraints / Guardrails（運用・安全上の制約）

- **Allowed paths**: `project/grad-speedup/` only.
- **Dependency changes**: Not allowed.
- **Dangerous operations**: No destructive commands.

---

## 13) Reporting（進捗報告の粒度・フォーマット）

### Cadence（頻度）

- On start and after each run completes.

### Format（書き方）

- **What I changed**: commands executed + run IDs
- **Evidence**: existence of summary.json + log file path
- **Next**: next run to execute
- **Blockers**: if any
