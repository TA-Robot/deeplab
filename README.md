# Cursor Dev Container

Cursor / VS Code 用の開発コンテナ環境。AI コーディングツール（Codex CLI、Gemini CLI）を統合した安全な開発環境を提供します。

## 特徴

- **Ubuntu 22.04** ベース
- **Node.js 22.x** プリインストール
- **AI ツール統合**: Codex CLI、Gemini CLI がすぐに使える
- **Docker-in-Docker**: コンテナ内でDockerを利用可能
- **ホスト設定の引き継ぎ**: SSH鍵、Git設定、認証情報を自動マウント

## クイックスタート

### 1. 前提条件

ホスト側に以下のディレクトリを作成しておく（存在しない場合）:

```bash
mkdir -p ~/.codex ~/.config/gemini
```

### 2. コンテナを起動

Cursor / VS Code でこのフォルダを開き、「Reopen in Container」を選択。

### 3. AI ツールを使う

```bash
# Codex CLI（通常モード）
codex "ファイルを整理して"

# Codex CLI（自動承認モード）
codex-auto "テストを書いて"

# Codex CLI（フルオートモード）
codex-full "リファクタリングして"

# Gemini CLI
gemini
```

## Cursorエージェント向け: Codex セカンドエージェント用ツール

このリポジトリには、`codex exec` を使って **非対話で呼べる“セカンドエージェント”口**として `codex-second-agent` を同梱しています。
セッションID(thread_id)は **ツール側が `~/.codex` 配下に保存**するため、CursorエージェントがIDを覚えておく必要がありません。

### 使い方（Cursorエージェント向け）

基本的な使い方は `AGENTS.md` を参照してください。

### 内部の挙動（実装の要点）

- `codex-second-agent` は内部で `codex exec --json ...` を実行します
- JSONLの `thread.started.thread_id` を抽出してセッションIDとして保存し、2回目以降は `codex exec resume <id> ...` で継続します
- 保存先はデフォルトで `<repo>/.codex-second-agent/<workspace_hash>/agents/<agent>/session_id` です（外部を汚さない方針）
  - `CODEX_SA_STATE_DIR` で保存先ルートを変更できます
- メインエージェント↔サブエージェントの会話ログは `<repo>/.codex-second-agent/<workspace_hash>/agents/<agent>/logs/` に保存されます
  - `events.jsonl`: `codex --json` の生イベント（JSONL）を追記
  - `transcript.jsonl`: 1リクエスト=1行で `agent` / `cd` / `prompt` / `response` をまとめたログ（JSONL）
  - `CODEX_SA_LOG_DIR` でログ保存先ディレクトリを変更できます
- エージェント用の作業ディレクトリ（git worktree）は次のいずれかに作成します
  - **デフォルト**: `~/.codex/cursor-second-agent/<workspace_hash>/worktrees/<agent>/`
  - `CODEX_SA_WORKTREES_MODE=workspace` または `--worktrees-in-workspace` を使う場合: `<repo>/.codex-worktrees/<agent>/`
  - `CODEX_SA_WORKTREES_DIR` を指定した場合: 指定パス配下
  - **非defaultエージェントは、未作成なら自動でworktreeを作成**してそこで実行します（`CODEX_SA_AUTO_WORKTREE=0` または `--no-auto-worktree` で無効化）

### 運用に効くコマンド（抜粋）

- `codex-second-agent paths`: state/log/worktree の実体パスと `effective_cd` を表示
- `codex-second-agent status --verbose`: session_id と各種パスをまとめて表示
- `codex-second-agent doctor`: 環境/パス/設定の簡易診断（トラブル切り分け）
- `codex-second-agent worktree remove <agent> [--keep-branch]`: worktree削除（必要なら `agent/<agent>` ブランチも整理）
- `codex-second-agent --post-git-status ...`（または `CODEX_SA_POST_GIT_STATUS=1`）: 実行後に未コミット変更を要約して検知

### 運用上の注意

- `codex-second-agent` は **常に `--dangerously-bypass-approvals-and-sandbox` と `--search` を有効化**します（運用方針として固定）
- `--dangerously-bypass-approvals-and-sandbox` は危険です。外部サンドボックスがある前提でのみ使用してください
- 使用モデルは **常に `gpt-5.2`** です（必要があれば `CODEX_SA_MODEL` で上書きできます）

## プリインストールツール

| カテゴリ | ツール |
|----------|--------|
| **ランタイム** | Node.js 22.x, Python 3 |
| **AI CLI** | @openai/codex, @google/gemini-cli |
| **開発ツール** | TypeScript, ESLint, Prettier |
| **ユーティリティ** | Git, GitHub CLI, ripgrep, jq, vim |
| **シェル** | Bash, Zsh (Oh My Zsh) |

## マウント設定

| ホスト | コンテナ | 用途 |
|--------|----------|------|
| `~/.ssh` | `/home/devuser/.ssh` | SSH鍵（Git操作） |
| `~/.gitconfig` | `/home/devuser/.gitconfig` | Git設定 |
| `~/.codex` | `/home/devuser/.codex` | Codex認証情報 |
| `~/.config/gemini` | `/home/devuser/.config/gemini` | Gemini設定 |

## 環境変数

ホスト側で以下の環境変数が設定されていれば、コンテナに引き継がれます:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

## エイリアス

| エイリアス | 展開 |
|------------|------|
| `codex-auto` | `codex --dangerously-bypass-approvals-and-sandbox` |
| `codex-full` | `codex --full-auto` |

## カスタマイズ

### 拡張機能の追加

`.devcontainer/devcontainer.json` の `customizations.vscode.extensions` に追加:

```json
"extensions": [
  "ms-python.python",
  "esbenp.prettier-vscode",
  // 追加したい拡張機能
]
```

### パッケージの追加

`.devcontainer/Dockerfile` の `npm install -g` セクションに追加:

```dockerfile
RUN npm install -g \
    typescript \
    ts-node \
    # 追加したいパッケージ
```

### コンテナ再ビルド

設定変更後は、コマンドパレットから「Dev Containers: Rebuild Container」を実行。

## トラブルシューティング

### マウントエラーが出る

ホスト側にディレクトリが存在しない可能性があります:

```bash
mkdir -p ~/.codex ~/.config/gemini
```

### 認証が効かない

1. ホスト側で先にログインしておく:
   ```bash
   codex  # 初回は認証フローが走る
   gemini # 初回は認証フローが走る
   ```
2. コンテナを再起動

### 権限エラー

ホストとコンテナのUID/GIDが異なる場合に発生することがあります。
Dockerfile の `USER_UID` を調整するか、ホスト側のファイル権限を確認してください。

## ライセンス

MIT


