# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

---

## 1) Meta（識別情報）

- **ticket_id**: `task-20260107-grad-speedup-trac-paper-spec`
- **role/agent**: `triage-task20260107-trac-paper`
- **owner（manager）**: `pm-codex`
- **created_at**: `2026-01-07`
- **priority**: `P1`
- **timebox**: `90min で一次成果`
- **workspace_scope**: `project/grad-speedup/`
- **related**
  - issue: `user request: integrate TRAC into ReLoRA`
  - pr/branch: `N/A`
  - commits: `N/A`

---

## 2) Goal / Desired Outcome（目的・達成状態）

TRAC (Tensor-Train LoRA with Across-layer shared Core) を paper-accurate に実装できるよう、
論文仕様の要点・式・ハイパラ・共有/固定ルールを整理し、実装者が迷わない状態にする。

### What success looks like（成功の見え方）

- TRAC の **更新式・TT分解の形・共有コアの扱い** が文書化されている
- 既存 ReLoRA 実装に **どう組み込むかの設計メモ** がある
- method-conformance / papers README が更新される

---

## 3) Background / Context（背景・前提・現状）

### Why now（なぜ今やるか）

- ユーザー要求で TRAC 論文を ReLoRA 系列に統合して比較したい。

### Current behavior（現状のふるまい）

- 現状は標準 LoRA / ReLoRA のみ。TT 分解や層間共有コアは未実装。

### Constraints already known（既知の制約）

- 追加依存は原則不可（必要なら manager に承認申請）。
- 既存インターフェース（ReLoRALayer/Controller）と整合が必要。

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- TRAC 論文の **定義・式・TT分解の具体形** を抽出
- 共有/固定コアの扱い、学習対象パラメータ、推奨ハイパラの整理
- CIFAR-10/ResNet での **Linear/Conv への適用方針** を提案
- `project/docs/grad-speedup/method-conformance.md` へ TRAC 追記
- `project/docs/grad-speedup/papers/README.md` に TRAC を追加
- 追加の設計メモを `project/docs/grad-speedup/papers/trac-notes.md` に作成

### Out of scope（やらないこと）

- 実装・実験はしない（implementer チケットで実施）

### Impacted areas（影響範囲）

- docs のみ（method-conformance / papers README / notes）

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- 論文の該当節・式番号を明示（ページ/セクション）
- 共有コア/凍結コアの扱いが実装に落とせる粒度で整理されている

### Should（できれば）

- 既存 ReLoRA への組込み案（param-mode / layer wrapper の案）
- Conv2d への適用での注意点

### Nice（余裕があれば）

- 既存 ReLoRA との計算コスト比較の見積り（概算）

---

## 6) Acceptance Criteria（完了条件：客観・機械的・再現可能）

- [ ] **Docs**: `project/docs/grad-speedup/method-conformance.md` に TRAC 追記済み
- [ ] **Docs**: `project/docs/grad-speedup/papers/README.md` に TRAC 追記済み
- [ ] **Docs**: `project/docs/grad-speedup/papers/trac-notes.md` が作成され、式・手順が記載

---

## 7) Implementation Notes（実装方針・設計メモ）

### Suggested approach（推奨アプローチ）

- TRAC 論文の Abstract / Method / Experiments を精読し、式とハイパラを抜粋
- 既存 ReLoRA (LoRA) の置換点（A,B）を TT で表す形を明示

### Files / Modules to touch（触る可能性のある箇所）

- `project/docs/grad-speedup/method-conformance.md`
- `project/docs/grad-speedup/papers/README.md`
- `project/docs/grad-speedup/papers/trac-notes.md`

---

## 8) Commands（実行・検証コマンド）

```bash
cd project
# docs only; no tests required
```

---

## 9) Deliverables（成果物：何を出して終わるか）

- **Docs**: TRAC notes + method-conformance + papers README
- **Notes for manager**: 実装上の open questions を列挙

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

- TT 分解のテンソル形状の解釈ミス
- 層間共有コアの「どの層で共有するか」の曖昧さ

---

## 11) Open Questions（未確定事項：質問しないと進めない点）

- TRAC の共有コアを **ResNet のどの層に共有**するのが妥当か
- Conv2d に直接適用する推奨があるか

---

## 12) Constraints / Guardrails（運用・安全上の制約）

- **Allowed paths**: `project/grad-speedup/` のみ
- **Dependency changes**: 不可（必要なら manager に相談）

---

## 13) Reporting（進捗報告の粒度・フォーマット）

### Cadence（頻度）

- 30〜45分ごと、または節目ごと

### Format（書き方）

- **What I changed**: 更新した docs
- **Evidence**: どの節・式を参照したか
- **Next**: 実装上の次の推奨
- **Blockers**: 不明点

