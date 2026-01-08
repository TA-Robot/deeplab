# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

---

## 1) Meta（識別情報）

- **ticket_id**: `task-20260107-grad-speedup-trac-implementation`
- **role/agent**: `implementer-task20260107-trac`
- **owner（manager）**: `pm-codex`
- **created_at**: `2026-01-07`
- **priority**: `P1`
- **timebox**: `120min で一次成果`
- **workspace_scope**: `project/`
- **related**
    - issue: `TRAC integration request (paper notes available)`
  - pr/branch: `N/A`
  - commits: `N/A`

---

## 2) Goal / Desired Outcome（目的・達成状態）

TRAC (Tensor-Train LoRA with Across-layer shared Core) を ReLoRA 系の
param-mode として使えるように実装する。paper-accurate 前提で、
forward/merge/reset が動作し、学習ループに統合されている状態にする。

### What success looks like（成功の見え方）

- `--param-mode trac` で学習が起動し、LoRA 相当の delta が適用される
- merge/reset の動作が ReLoRAController と整合
- config/metrics に TRAC 固有の設定が記録される

---

## 3) Background / Context（背景・前提・現状）

### Why now（なぜ今やるか）

- PM 指示で TRAC を ReLoRA 系列として評価したい。

### Current behavior（現状のふるまい）

- `param-mode` は `none` / `relora` のみ。TRAC 未実装。

### Constraints already known（既知の制約）

- 依存追加は原則不可。
- 既存 ReLoRA の interface に沿って実装する。
- paper-accurate 仕様は `project/docs/grad-speedup/papers/trac-notes.md` に従う。

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- TRAC 用の layer wrapper (Linear/Conv) を実装
- TRAC を `ReLoRALayer` と同等の interface に揃える
- `param-mode` に `trac` を追加し、run_cifar10 の CLI で切替可能にする
- config/metrics に TRAC 設定値を出力
- 簡単な smoke（max_steps=1/10）で forward が通る確認

### Out of scope（やらないこと）

- paper validation 実験・ハイパラ探索
- 大規模 refactor

### Impacted areas（影響範囲）

- `project/grad-speedup/src/relora.py` (または新規ファイル)
- `project/grad-speedup/src/train.py`
- `project/grad-speedup/scripts/run_cifar10.py`
- `project/grad-speedup/README.md` (param-mode 追記)

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- TRAC の forward/merge/reset が動作
- CLI 引数で TRAC の主要ハイパラを指定可能
- 既存 relora と同じログ構造で出力

### Should（できれば）

- TRAC の shared core を層間で共有できる実装
- Conv2d への適用は最小構成で動作

### Nice（余裕があれば）

- small smoke test 用の run コマンドを docs に残す

---

## 6) Acceptance Criteria（完了条件：客観・機械的・再現可能）

- [ ] `--param-mode trac` で `run_cifar10.py` が起動して 1 step 進む
- [ ] config.json に TRAC ハイパラが記録される
- [ ] README に param-mode と主要引数の説明が追加される

---

## 7) Implementation Notes（実装方針・設計メモ）

### Suggested approach（推奨アプローチ）

- `project/docs/grad-speedup/papers/trac-notes.md` に従って TT 分解を実装
- `ReLoRALayer` interface に合わせて `merge_into_base` を実装
- shared core を module-level で保持し、同一 scope の層で共有できる設計

### Guardrails（やり方の縛り）

- 依存追加禁止
- 既存 relora のコードを壊さない

### Files / Modules to touch（触る可能性のある箇所）

- `project/grad-speedup/src/relora.py`（or 新規 `relora_trac.py`）
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
# python scripts/run_cifar10.py --run-id smoke-trac --model resnet18 --max-steps 1 --param-mode trac ...
```

---

## 9) Deliverables（成果物：何を出して終わるか）

- **Code changes**: TRAC wrapper + CLI + config/log updates
- **Notes for manager**: 実装上の open questions

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

- TT テンソル形状の不整合
- shared core の再初期化タイミング（merge/reset との衝突）

---

## 11) Open Questions（未確定事項：質問しないと進めない点）

- TT の rank/shape が CIFAR ResNet に適用可能か
- Conv2d のテンソル化の推奨があるか

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

- **What I changed**: TRAC wrappers + controller added in `project/grad-speedup/src/relora.py`, CLI/config wiring in `project/grad-speedup/scripts/run_cifar10.py`, README updated.
- **Evidence**: smoke run completed on small-cnn (CPU, max_steps=1).
  - `python project/grad-speedup/scripts/run_cifar10.py --run-id smoke-trac --model small-cnn --activation relu --epochs 1 --max-steps 1 --batch-size 4 --num-workers 0 --device cpu --param-mode trac --trac-rank 2 --trac-alpha 1.0 --trac-scope all --trac-merge-interval 1000 --warmup-steps 0 --measure-steps 0 --log-interval-steps 1 --data-dir /workspace/deeplab/project/grad-speedup/data`
- **Next**: paper-accurate validation; compare against TRAC notes.
- **Blockers**: None.
