# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

---

## 1) Meta（識別情報）

- **ticket_id**: `task-20260107-grad-speedup-superlora-implementation`
- **role/agent**: `implementer-task20260107-superlora`
- **owner（manager）**: `pm-codex`
- **created_at**: `2026-01-07`
- **priority**: `P1`
- **timebox**: `120min で一次成果`
- **workspace_scope**: `project/`
- **related**
    - issue: `SuperLoRA integration request (paper notes available)`
  - pr/branch: `N/A`
  - commits: `N/A`

---

## 2) Goal / Desired Outcome（目的・達成状態）

SuperLoRA を ReLoRA 系の param-mode として使えるように実装する。
group/projection/shuffle を反映した forward を実装し、学習ループに統合する。

### What success looks like（成功の見え方）

- `--param-mode superlora` で学習が起動
- group/projection/shuffle が設定で制御できる
- merge/reset が ReLoRAController と整合

---

## 3) Background / Context（背景・前提・現状）

### Why now（なぜ今やるか）

- PM 指示で SuperLoRA を ReLoRA 系列として評価したい。

### Current behavior（現状のふるまい）

- `param-mode` は `none` / `relora` のみ。SuperLoRA 未実装。

### Constraints already known（既知の制約）

- 依存追加は原則不可。
- paper-accurate 仕様は `project/docs/grad-speedup/papers/superlora-notes.md` に従う。

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- SuperLoRA の layer wrapper (Linear/Conv) を実装
- group/projection/shuffle の設定を CLI 経由で制御
- `param-mode` に `superlora` を追加
- config/metrics に SuperLoRA 設定値を出力
- 簡単な smoke で forward が通る確認

### Out of scope（やらないこと）

- paper validation 実験・ハイパラ探索
- 大規模 refactor

### Impacted areas（影響範囲）

- `project/grad-speedup/src/relora.py` (または新規ファイル)
- `project/grad-speedup/src/train.py`
- `project/grad-speedup/scripts/run_cifar10.py`
- `project/grad-speedup/README.md`

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- SuperLoRA の forward/merge/reset が動作
- CLI 引数で group/projection/shuffle を指定できる
- 既存 relora と同じログ構造で出力

### Should（できれば）

- shuffle の切替（on/off, interval）を実装
- Conv2d への適用は最小構成で動作

### Nice（余裕があれば）

- small smoke test 用の run コマンドを docs に残す

---

## 6) Acceptance Criteria（完了条件：客観・機械的・再現可能）

- [ ] `--param-mode superlora` で `run_cifar10.py` が起動して 1 step 進む
- [ ] config.json に SuperLoRA ハイパラが記録される
- [ ] README に param-mode と主要引数の説明が追加される

---

## 7) Implementation Notes（実装方針・設計メモ）

### Suggested approach（推奨アプローチ）

- `project/docs/grad-speedup/papers/superlora-notes.md` に従って group/projection/shuffle を実装
- `ReLoRALayer` interface に合わせて `merge_into_base` を実装

### Guardrails（やり方の縛り）

- 依存追加禁止
- 既存 relora のコードを壊さない

### Files / Modules to touch（触る可能性のある箇所）

- `project/grad-speedup/src/relora.py`（or 新規 `relora_superlora.py`）
- `project/grad-speedup/src/train.py`
- `project/grad-speedup/scripts/run_cifar10.py`
- `project/grad-speedup/README.md`

---

## 8) Commands（実行・検証コマンド）

```bash
cd project/grad-speedup
python -m py_compile src/relora.py
python scripts/run_cifar10.py --help
# Optional smoke (tiny):
# python scripts/run_cifar10.py --run-id smoke-superlora --model resnet18 --max-steps 1 --param-mode superlora ...
```

---

## 9) Deliverables（成果物：何を出して終わるか）

- **Code changes**: SuperLoRA wrapper + CLI + config/log updates
- **Notes for manager**: 実装上の open questions

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

- group/projection の形状不整合
- shuffle を行う場合の deterministic 影響

---

## 11) Open Questions（未確定事項：質問しないと進めない点）

- projection の具体形（固定/学習）を paper に合わせて決める必要
- Conv2d の tensorization 方法

---

## 12) Constraints / Guardrails（運用・安全上の制約）

- **Allowed paths**: `project/grad-speedup/` のみ
- **Dependency changes**: 不可（必要なら manager に相談）

---

## 13) Reporting（進捗報告の粒度・フォーマット）

### Cadence（頻度）

- 節目ごと（実装開始 / 動作確認 / コミット）

### Format（書き方）

- **What I changed**: 変更ファイルと要点
- **Evidence**: smoke の結果
- **Next**: 次に必要な作業
- **Blockers**: 不明点

---

## Status update（2026-01-07）

- **What I changed**: SuperLoRA wrappers + controller (grouped LoRA + projection/shuffle) added in `project/grad-speedup/src/relora.py`, CLI/config wiring in `project/grad-speedup/scripts/run_cifar10.py`, README updated.
- **Evidence**: smoke run completed on small-cnn (CPU, max_steps=1). group_count=1 required for scope=all due to conv1 in_channels=3.
  - `python project/grad-speedup/scripts/run_cifar10.py --run-id smoke-superlora --model small-cnn --activation relu --epochs 1 --max-steps 1 --batch-size 4 --num-workers 0 --device cpu --param-mode superlora --relora-rank 2 --relora-alpha 1.0 --relora-merge-interval 1000 --relora-scope all --superlora-group 1 --superlora-projection fixed --no-superlora-shuffle --warmup-steps 0 --measure-steps 0 --log-interval-steps 1 --data-dir /workspace/deeplab/project/grad-speedup/data`
- **Next**: paper-accurate validation; verify projection/shuffle specifics vs notes.
- **Blockers**: None.
