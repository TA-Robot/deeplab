# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

## 1) Meta（識別情報）

- **ticket_id**: `task-gn-layerwise-impl`
- **role/agent**: `implementer-task-gn-layerwise`
- **owner（manager）**: `pm`
- **created_at**: `2026-01-06`
- **priority**: `P1`
- **timebox**: `6-8h (prototype) / 2h (design)`
- **workspace_scope**: `project/grad-speedup/`
- **related**
  - issue: `N/A`
  - pr/branch: `N/A`
  - commits: `N/A`

---

## 2) Goal / Desired Outcome（目的・達成状態）

Layerwise Gauss–Newton (GN) を **方向モジュール**として実装し、CIFAR-10/ResNet18 の小スケール実験で **壁時計コストが現実的**な範囲に入ることを確認できる状態にする。

成功の見え方:
- CLI で `--direction gn-layerwise` を指定できる。
- `direction_update_every` で GN の更新頻度を制御できる（SOAP と同様に頻度がメインのノブ）。
- 1D パラメータや巨大層の扱いが明示され、計算が暴走しない。
- 既存の SOAP/SGD 等の挙動に影響を与えない。

---

## 3) Background / Context（背景・前提・現状）

- temp の議論では Full GN は上限材料で、Layerwise GN が実装可能性/計算コストの観点で優先とされている。
- 現状の direction 実装は SOAP / Shampoo / Sophia / Muon に対応しているが GN はない。
- 計算コストが爆発しやすいので、**Layerwise / block-diag 近似**前提で設計する。

---

## 4) Scope（スコープ設計）

### In scope
- 新しい direction: `gn-layerwise`
- GN の更新頻度を `direction_update_every` で制御
- 1D / bias / batchnorm などの扱いを明示（skip or cheap path）
- 速度計測（step_time, precond_update_time）を既存ログと同じフォーマットで出す

### Out of scope
- Full GN 実装
- 分散 GN / 大バッチ専用チューニング
- SOAP の再実装

---

## 5) Requirements（要件）

### Must
- `--direction gn-layerwise` が通る
- 既存方向モジュールの挙動が変わらない
- `direction_update_every` で頻度を落とせる

### Should
- 1Dパラメータはスキップまたは軽量処理
- 計算量が大きい層を保護する guardrail（dim上限など）

### Nice
- layerwise GN の対象層をフィルタで指定できる

---

## 6) Acceptance Criteria（完了条件）

- [ ] `python project/grad-speedup/scripts/run_cifar10.py --direction gn-layerwise ...` が実行できる
- [ ] `direction_update_every` が効く（更新頻度を落とすと step_time が改善する）
- [ ] 既存方向 (`soap`, `shampoo`, etc.) が壊れていない
- [ ] 実装方針を `project/docs/grad-speedup/gn-layerwise-design.md` に記録

---

## 7) Implementation Notes（実装方針）

Suggested approach:
- まずは **layerwise block-diagonal** 近似を実装（per-layer GN）
- loss Hessian の近似は cross-entropy の Fisher/softmax 近似を用いる（or diagonal approx）
- 大きな層はスキップ / 片側更新 / 低rank 近似で防御

Files to touch:
- `project/grad-speedup/src/train.py`
- `project/grad-speedup/src/modules.py`
- `project/grad-speedup/scripts/run_cifar10.py`
- `project/docs/grad-speedup/gn-layerwise-design.md`

---

## 8) Commands（実行・検証）

```bash
cd project/grad-speedup
python scripts/run_cifar10.py --direction gn-layerwise --max-steps 2000 --seed 0
```

---

## 9) Deliverables

- Code changes for `gn-layerwise` direction
- Design note in docs
- Small smoke run (optional) with run_id logged

---

## 10) Risks / Edge Cases

- GN update cost may explode on large layers
- Numerical instability in Hessian/Fisher estimation
- Accidental coupling with existing direction modules

---

## 11) Open Questions

- Which loss-specific GN approximation should we use first (Fisher vs exact Hessian)?
- Should we only apply GN to last layer for v1?

---

## 12) Constraints / Guardrails

- Keep changes minimal; no refactor of training loop
- No new dependencies unless approved

---

## 13) Reporting

- Update with: what changed, performance impact, open issues
