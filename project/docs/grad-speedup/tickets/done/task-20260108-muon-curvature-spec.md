# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

---

## 1) Meta（識別情報）

- **ticket_id**: `task-20260108-muon-curvature-spec`
- **role/agent**: `triage-task20260108-muon-curvature-spec`
- **owner（manager）**: `codex`
- **created_at**: `2026-01-08`
- **priority**: `P1`
- **timebox**: `60–90min for draft spec`
- **workspace_scope**: `project/grad-speedup/`
- **related**
  - issue: `muon-improvement from /workspace/deeplab/temp`
  - pr/branch: `TBD`
  - commits: `TBD`

---

## 2) Goal / Desired Outcome（目的・達成状態）

Muonの改良手法（temp記載の「反復で逆平方根を作る非対角プレコンディショニング＋Muon直交化」）を、
**実装可能な仕様に落とす**ことが目的。

### What success looks like（成功の見え方）

- `project/docs/grad-speedup/method-conformance.md` に
  - **ASGO** と **Muon-curvature / Curvature-Whitened Muon** のエントリが追加され、
  - 数式・ハイパラ・近似の制約が記載されている。
- `project/docs/grad-speedup/papers/` 配下に
  - **muon-improvement-notes.md** を新規作成し、tempの論点（NS反復、片側V、Muonとの関係）を要約する。
- 実装側へ渡す **具体的なAPI案** と **必要なログ** が明確になっている。

---

## 3) Background / Context（背景・前提・現状）

### Why now（なぜ今やるか）

- /workspace/deeplab/temp に Muon改良案（ASGO/NS反復を流用）が記載され、
  これを実装サイクルに載せる必要がある。

### Current behavior（現状のふるまい）

- 現在の `direction=muon` は直交化のみ（inverse sqrt の曲率補正なし）。
- ReLoRA系の精度停滞により「曲率補正×Muon」の検討が必要。

### Constraints already known（既知の制約）

- 既存の Muon 実装と共存させる。
- 追加メモリは層ごとの小次元 (min(m,n)) 行列に限定する。

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- tempの内容を **仕様メモ化**（数式・反復手順）
- ASGOとの関係、Muonとの極限一致条件の整理
- `method-conformance.md` への追加記載
- 新しい method notes ファイル作成
- 実装に必要な **引数/ログ/評価指標** の提案

### Out of scope（やらないこと）

- 実コード実装
- 実験キューの作成

### Impacted areas（影響範囲）

- `project/docs/grad-speedup/method-conformance.md`
- `project/docs/grad-speedup/papers/`

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- temp内容に沿った仕様整理（NS反復・片側V・white化・Muon直交化）
- 既存 Muon との違い / 極限一致の説明

### Should（できれば）

- ASGO論文との対応づけ（Algorithm番号など）
- 実装が困難な点・リスクの記載

### Nice（余裕があれば）

- 参考ハイパラ推奨値（beta2 / interval / ns_iters）

---

## 6) Acceptance Criteria（完了条件）

- [ ] method-conformance.md に新規エントリが追加されている
- [ ] muon-improvement-notes.md が新規作成されている
- [ ] 実装チケットへ渡す具体的パラメータ案が記載されている

---

## 7) Implementation Notes（実装方針・設計メモ）

- tempの「Curvature-Whitened Muon」設計をベースにする。
- 片側 (G^T G か G G^T) の小次元Vを使う点を重視。
- NS反復の係数は Muon の既存係数 or ASGO推奨の係数を調査。

---

## 8) Commands（実行・検証コマンド）

```bash
cd project
# docs only (no tests)
```

---

## 9) Deliverables（成果物）

- `project/docs/grad-speedup/method-conformance.md` 更新
- `project/docs/grad-speedup/papers/muon-improvement-notes.md` 新規

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

- 参照論文がローカルに存在しない可能性
- ASGOとMuonの「どこまで一致とみなすか」の解釈差

---

## 11) Open Questions（未確定事項）

- NS反復の係数は Muonの既存係数で良いか？ASGO係数を再現すべきか？
- Whitening後に Muonの直交化を必ず掛けるか、オプションにするか？
