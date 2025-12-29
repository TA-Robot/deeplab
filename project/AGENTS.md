# project/AGENTS.md（サブエージェント用：project配下のみを対象にする指示テンプレ）

この `project/AGENTS.md` は、**サブエージェント（implementer/reviewer/triage）に渡す指示**です。  
目的は「サブエージェントが **project配下だけ**を見て作業する」運用を徹底することです。

> 注意: 技術的に“強制隔離”するものではなく、運用ルールとしてスコープを縛ります。  
> ただし、`codex-second-agent workspace init <path>` を必須にすることで「親リポジトリのworktreeを誤って作る事故」は防げます。

## codex-second-agent とは？（簡単に）

- `codex exec` を「セッションID自動保持」「agent別worktree」「ログ保存」付きで呼び出すラッパーです
- この基盤リポジトリでは `scripts/codex-second-agent` が実体です（PATHに無い場合はこれを直接実行）

---

## スコープ（最重要）

- **見てよい/触ってよい**: `project/` 配下のみ
- **見ない/触らない**:
  - リポジトリルート配下の `scripts/` / `.devcontainer/` / `docs/` / `README.md` / `AGENTS*.md` など
  - project外のファイルを参照してよいか迷ったら、必ず管理者へ質問して止まる

## project/docs（管理者が整備する前提）

- 管理者は `project/docs/` を整備します（runbook/decision log）
- サブエージェントは **参照してよい**（ただし更新が必要なら管理者に提案）

## 役割別の期待

- **implementer**
  - `project/` 配下のみで実装し、`project/` 配下のテスト/ビルドが通る状態まで持っていく
  - `git status -sb` を確認し、必要ならコミットしてブランチ先端を進める
- **reviewer**
  - `project/` 配下だけの差分レビューを行う（Must/Should/Nice）
- **triage**
  - `project/` 配下のログ/コードから原因切り分け。必要な追加情報があれば管理者に要求

## 実行コマンド（管理者が使う想定）

### implementer を project 固定で起動

```bash
# 最初に一度だけ、対象プロジェクト（git repo root）を保存する
codex-second-agent workspace init project/<name>

mkdir -p .codex-second-agent/nohup
cat <<'PROMPT' | nohup codex-second-agent --agent implementer --post-git-status - -- --cd project > .codex-second-agent/nohup/implementer.out 2>&1 &
あなたは implementer です。
- 作業対象は project/<name>/ 配下のみ
- 変更は project/ 配下のみに限定

要件:
- ...
制約:
- 依存追加は事前承認
完了条件:
- project/ 配下のテストが通る
PROMPT
echo "pid=$!"
```

### reviewer を project 固定で起動

```bash
# 対象プロジェクトを保存済みであること（未設定なら init する）
codex-second-agent workspace init project/<name>

mkdir -p .codex-second-agent/nohup
cat <<'PROMPT' | nohup codex-second-agent --agent reviewer - -- --cd project > .codex-second-agent/nohup/reviewer.out 2>&1 &
あなたは reviewer です。
- レビュー対象は project/ 配下の差分のみ
- 指摘は Must/Should/Nice に分ける
PROMPT
echo "pid=$!"
```

## ログ/スコープ確認（迷子防止）

```bash
codex-second-agent --agent implementer paths
codex-second-agent --agent implementer status --verbose
codex-second-agent --agent implementer doctor
```

`paths` の `effective_cd` が `.../project` になっていることを必ず確認してください。

## 実運用メモ（ハマりどころ）

- `nohup` の標準出力ファイルは **空のまま**になることがあります。進捗・成果物の回収は基本 **`transcript.jsonl` を見る**運用が安定します。
- reviewer は長引くことがあるので、バックグラウンド実行では `timeout` 付きにするのがおすすめです:

```bash
mkdir -p .codex-second-agent/nohup
cat <<'PROMPT' | nohup timeout 120s codex-second-agent --agent reviewer - > .codex-second-agent/nohup/reviewer.out 2>&1 &
対象コミット: <hash>
出力: Must/Should/Nice
PROMPT
echo "pid=$!"
```

## 補足（よくある誤解）

- **worktreeが project 配下に作られるわけではありません**  
  `codex-second-agent` は基本的に「workspace（git root）単位」で agent worktree を作ります。  
  `codex-second-agent workspace init project/<name>` を使うと、**project/<name> 側のGitをworkspaceとして扱う**ようになります。


