# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

---

## 1) Meta（識別情報）

- **ticket_id**: `task-20260108-muon-curvature-experiments`
- **role/agent**: `experimenter-task20260108-muon-curvature-experiments`
- **owner（manager）**: `codex`
- **created_at**: `2026-01-08`
- **priority**: `P2`
- **timebox**: `60min for plan + queue`
- **workspace_scope**: `project/grad-speedup/`
- **related**
  - depends_on: `task-20260108-muon-curvature-implementation`

---

## 2) Goal / Desired Outcome（目的・達成状態）

Muon改良版の効果を短時間で評価するための実験計画・キューを作成する。

### What success looks like（成功の見え方）

- `project/docs/grad-speedup/experiment-plan-YYYYMMDD-muon-curv.md` が作成される
- キューが作成され、muon-baseline vs muon-curv の比較ができる

---

## 3) Background / Context（背景・前提・現状）

- temp 由来の Muon 改良案を実装予定。
- まずは 2000-step の短い評価で “伸びる方向性” を見る。

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- muon baseline と muon-curv の比較計画
- 2〜3個の主要ハイパラ (beta2 / update_interval / ns_iters) の少数スイープ

### Out of scope（やらないこと）

- 14000step など長期実験
- TRAC/SuperLoRA の調整

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- baseline vs new の同条件比較
- 実験条件（lr/momentum/rank/T/ws/alphaなど）を明記

### Should（できれば）

- 3×2 程度の small sweep

---

## 6) Acceptance Criteria（完了条件）

- [ ] 実験計画ドキュメント作成
- [ ] queueファイルを作成し、run_idが割り当てられている

---

## 7) Implementation Notes（実装方針・設計メモ）

- 例: muon-curv の update_interval {1, 10}, beta2 {0.9, 0.99}, ns_iters {3,5}
- seed=0 でスクリーニング

---

## 8) Commands（実行・検証コマンド）

```bash
cd project
# queue_run.sh で実行
```

---

## 9) Deliverables（成果物）

- 計画書
- queueファイル

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

- muon-curv 実装が未完の状態で計画だけ先行する可能性

---

## 11) Open Questions（未確定事項）

- baseline は muon だけでよいか、adam/sgd も併記するか
