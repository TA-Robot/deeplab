# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

---

## 1) Meta（識別情報）

- **ticket_id**: `task-20260108-muon-curvature-implementation`
- **role/agent**: `implementer-task20260108-muon-curvature-implementation`
- **owner（manager）**: `codex`
- **created_at**: `2026-01-08`
- **priority**: `P1`
- **timebox**: `180min for MVP implementation`
- **workspace_scope**: `project/grad-speedup/`
- **related**
  - issue: `Muon improvement from /workspace/deeplab/temp`
  - depends_on: `task-20260108-muon-curvature-spec`

---

## 2) Goal / Desired Outcome（目的・達成状態）

Muon改良版（非対角2次モーメントの inverse sqrt を NS 反復で計算し、Muon直交化に接続）を
**新しい direction モードとして追加**する。

### What success looks like（成功の見え方）

- `direction=muon-curv`（仮）を指定すると新手法が動く。
- 既存 `direction=muon` はそのまま動作。
- 追加パラメータ（beta2 / update_interval / ns_iters / eps / mode）が CLI から設定できる。
- ログに「curvature update time / apply time / ns iters」などが出る。

---

## 3) Background / Context（背景・前提・現状）

- temp で「ASGO/NS反復 + Muon直交化」の設計案が提示された。
- `project/grad-speedup/src/train.py` に Muon が実装済み。

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- 新しい `direction` の追加（例: `muon-curv`）
- 片側V（G^T G or G G^T）と NS 反復で V^{-1/2} を近似
- Whitening → Muon直交化 の流れ
- CLI引数追加（run_cifar10.py）
- ログ/summary項目を追加

### Out of scope（やらないこと）

- 実験キュー作成
- TRAC/SuperLoRA 変更

### Impacted areas（影響範囲）

- `project/grad-speedup/src/train.py`
- `project/grad-speedup/scripts/run_cifar10.py`
- `project/grad-speedup/src/modules.py`（必要なら）
- `project/docs/grad-speedup/method-conformance.md`（更新は別チケット）

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- `direction` に新モードを追加
- V のサイズは **min(m,n)** 次元を使いメモリを抑える
- NS反復の回数をパラメータ化
- update_interval を入れて毎step更新を回避可能にする

### Should（できれば）

- `muon` との挙動比較ができるようログを残す
- 失敗時に graceful に muon fallback できる

### Nice（余裕があれば）

- 簡易のユニットテスト（小行列で NaN が出ない）

---

## 6) Acceptance Criteria（完了条件）

- [ ] `direction=muon-curv` が CLI から指定可能
- [ ] 主要パラメータがログに記録される
- [ ] smoke run (`max_steps=50`) が完走する

---

## 7) Implementation Notes（実装方針・設計メモ）

### Suggested approach（推奨アプローチ）

1. `train.py` に **V行列のEMA** を追加（stateに保存）
2. `V^{-1/2}` を NS 反復で計算（Muon既存の係数を流用可）
3. Whitening: `M @ V^{-1/2}` or `V^{-1/2} @ M` (形状で選択)
4. Whitening後に Muonの `orthogonalize` を適用
5. 更新時に RMSスケール調整（Muon既存スケール関数を再利用）

### Guardrails（やり方の縛り）

- 既存 Muon 実装は壊さない
- 追加分は `direction == "muon-curv"` のみに閉じる

### Files / Modules to touch

- `project/grad-speedup/src/train.py`
- `project/grad-speedup/scripts/run_cifar10.py`
- `project/grad-speedup/README.md`（引数追加）

---

## 8) Commands（実行・検証コマンド）

```bash
cd project

# smoke run
python -u grad-speedup/scripts/run_cifar10.py \
  --run-id smoke-muon-curv \
  --model resnet18 --max-steps 50 --batch-size 128 --device cpu \
  --param-mode none --direction muon-curv
```

---

## 9) Deliverables（成果物）

- 新 direction 実装コード
- CLI引数追加
- README更新

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

- V が特異/数値不安定 → eps で回避
- update_interval > 1 のとき stale V で発散する可能性

---

## 11) Open Questions（未確定事項）

- whitening後に Muon直交化を必須にするか option にするか
- NS反復の係数は Muon既存の a,b,c を流用でよいか
