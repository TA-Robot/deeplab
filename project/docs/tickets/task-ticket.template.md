# Task Ticket Template（サブエージェント依頼用 / “これくらい書く” の標準）

> このファイルはテンプレです。**コピーして編集**してから、サブエージェント起動時に stdin で渡します。  
> 管理者が内容を改善する場合は、このテンプレ側に反映します（サブエージェントは原則更新しない）。
>
> 目標は「曖昧さを減らし、手戻りを減らし、並列化しても衝突しない」依頼文にすることです。  
> “短く済ませる” より **誤解の余地を潰す** ことを優先してください。

---

## 0) How to Use（運用手順）

1. `project/docs/tickets/task-ticket.template.md` を `project/docs/tickets/ready/` にコピーする
2. このチケットを **最後まで埋める**（未確定は “未確定” と明示する）
3. サブエージェント起動時に `cat ticket.md | codex-second-agent ... -` で渡す
4. 起動したら `running/` に移動し、完了したら `done/` に移動して履歴として残す（原則削除しない）

---

## 1) Meta（識別情報）

- **ticket_id**: `task-0001`
- **role/agent**: `implementer-task0001`（例: `triage-task0001` / `reviewer-task0001` / `newbie-task0001`）
- **owner（manager）**: `<<name>>`
- **created_at**: `YYYY-MM-DD`
- **priority**: `P0 | P1 | P2`
- **timebox**: `<<例: 90minで一次成果 / 30minで調査>>`
- **workspace_scope**: `project/<name>/`（原則ここだけ）
- **related**
  - issue: `<<link-or-id>>`
  - pr/branch: `<<link-or-branch>>`
  - commits: `<<hashes>>`

---

## 2) Goal / Desired Outcome（目的・達成状態）

このタスクのゴールを、**複数段落で**書いてください（1 行に収めない）。

### What success looks like（成功の見え方）

- 何がどう変わり、誰が何をできるようになるのか
- “ユーザー/運用者視点” の変化
- “品質面” の変化（テスト、ログ、事故対応性、パフォーマンス、セキュリティ）

### Example（例：このくらい書く）

> 現状、ユーザー登録 API が 400/500 を混同して返しており、クライアント側で誤った再試行が起きる。  
> このタスクでは、入力バリデーションエラーは常に 400（詳細はエラーコード）で返し、サーバ内部エラーは 500 に統一する。  
> さらに、ログに request_id と validation_error_code を残し、再発時に原因追跡できるようにする。  
> 完了時点で、該当 API のテストが追加され、`<<test-cmd>>` が通り、変更理由が `project/docs/decisions.md` に短く追記されている。

---

## 3) Background / Context（背景・前提・現状）

### Why now（なぜ今やるか）

- `<<障害/要望/期限/依存関係>>`

### Current behavior（現状のふるまい）

- 再現手順（可能なら最小手順）
- 現状の期待されない挙動（ログ/スクショ/レスポンス例があると強い）
- 関連する仕様/ドキュメント/コードへのリンク

### Constraints already known（既知の制約）

- `<<例: 互換性維持が必須 / 既存API変更不可 / データ移行不可>>`

---

## 4) Scope（スコープ設計：衝突防止の要）

### In scope（やること）

- `<<やること（粒度は細かめに）>>`
- `<<やること>>`

### Out of scope（やらないこと）

- `<<今回やらない。将来やるならどのタスクでやるかも>>`

### Impacted areas（影響範囲）

- 影響するモジュール/エンドポイント/画面/設定
- 影響しないことを明言（例: “認証フローは変更しない”）

---

## 5) Requirements（要件：Must/Should/Nice）

### Must（必須）

- `<<必須要件>>`
- `<<必須要件>>`

### Should（できれば）

- `<<望ましい要件>>`

### Nice（余裕があれば）

- `<<あれば嬉しい>>`

---

## 6) Acceptance Criteria（完了条件：客観・機械的・再現可能）

“主観” を避けて、**誰でも同じ結論になる条件**で書く。

- [ ] **Behavior**: `<<入力>>` のとき `<<出力>>`（エラー含む）がこうなる
- [ ] **Tests**: `<<test-cmd>>` が通る（新規テストが追加されている）
- [ ] **No regression**: 既存の `<<重要フロー>>` が壊れていない（確認手順も書く）
- [ ] **Logs/Observability**: `<<必要なログ/メトリクス>>` が追加され、追跡できる
- [ ] **Docs**: 変更理由/運用上の注意が `project/docs/<<...>>` に追記されている（必要な場合）

---

## 7) Implementation Notes（実装方針・設計メモ）

### Suggested approach（推奨アプローチ）

- `<<方針>>`
- `<<方針>>`

### Guardrails（やり方の縛り）

- **No big bang refactor**: 既存のふるまいを壊さず小さく進める
- **TDD preference**: 可能なら Red→Green→Refactor（既存改修なら characterization test から）

### TDD Plan（Red→Green→Refactor を運用に落とす）

このタスクを **どの“最小ステップ”で進めるか**を具体的に書く（曖昧なら manager に相談して止まる）。

- **Red（先に書くテスト）**:
  - `<<追加するテストの名前 / ケース>>`
  - `<<最初に落ちることを確認する観点>>`
- **Green（最小実装）**:
  - `<<最小の実装方針>>`
- **Refactor（ふるまいを変えずに整理）**:
  - `<<重複排除/命名/責務分離など>>`

### Files / Modules to touch（触る可能性のある箇所）

- `<<path>>`

---

## 8) Commands（実行・検証コマンド）

最低でも「再現」「テスト」「フォーマット/静的解析」が書かれている状態にする。

```bash
cd project

# reproduce (optional)
<<commands to reproduce>>

# test
<<test-cmd>>

# lint (optional)
<<lint-cmd>>
```

---

## 9) Deliverables（成果物：何を出して終わるか）

- **Code changes**:
  - `<<変更する/したファイルの候補>>`
- **Commits**:
  - `<<コミット方針。複数コミットOKなら分割方針>>`
- **Notes for manager**:
  - `<<取り込み時の注意点 / ロールバック観点>>`

---

## 10) Risks / Edge Cases（リスク・落とし穴・境界条件）

### Risks（リスク）

- `<<例: 互換性破壊 / パフォーマンス劣化 / セキュリティ>>`

### Edge cases（境界条件）

- `<<例: 空入力 / 巨大入力 / タイムゾーン / 競合 / リトライ>>`

---

## 11) Open Questions（未確定事項：質問しないと進めない点）

ここが空なのは稀です。**迷うなら書く**。

- `<<質問>>`
- `<<質問>>`

---

## 12) Constraints / Guardrails（運用・安全上の制約）

- **Allowed paths**: `project/<name>/` のみ（迷ったら manager に質問して止まる）
- **Dependency changes**: `<<可/不可。可なら承認手順>>`
- **Dangerous operations**: `<<例: 本番DB操作禁止 / destructive禁止>>`
- **Secrets**: `.env` / token をログに出さない、貼らない

---

## 13) Reporting（進捗報告の粒度・フォーマット）

### Cadence（頻度）

- `<<例: 10〜15分ごと>>` / `<<節目ごと（調査→結論、実装→テスト、コミット）>>`

### Format（書き方）

- **What I changed**: `<<何をした>>`
- **Evidence**: `<<テスト結果/ログ/差分>>`
- **Next**: `<<次にやる>>`
- **Blockers**: `<<詰まっている点>>`

---

## 14) Reviewer-specific（reviewer の場合だけ埋める）

### Review focus（重点）

- `<<例: 互換性 / 例外処理 / 競合 / パフォーマンス>>`

### Output format（出力形式）

- Must / Should / Nice
- Must には「根拠」と「修正案（最小）」を含める

---

## 15) Newbie-specific（newbie の場合だけ埋める：質問だけ）

### Rules（絶対ルール）

- コードは触らない（編集・コミットしない）
- “分からない” を前提に、前提/用語/運用を質問で掘る

### Questions to ask（質問）

- `<<質問>>`
- `<<質問>>`
