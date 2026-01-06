# Decisions（重要な判断ログ）

管理者が「なぜそうしたか」を短く残すためのログです（箇条書きでOK）。

- 2025-12-29: ROS-ALTH 系ドキュメントを `project/docs/ros-alth/` に移動し、実験準備用ドキュメントを `project/docs/` に集約
  - 背景: ルート直下では散逸しやすく、実験運用の導線が不明確だった
  - 代替案: ルート直下に置き続ける / 別リポジトリに分離
  - 影響: 参照パスが変更されたため、今後は新パスを参照する

- 2025-12-29: 実験ログ保管先として `project/runs/` を新設し、VCS から除外
  - 背景: 大きなアーティファクトをコミットしない運用を明確化したかった
  - 代替案: ルート直下に置く / 都度パスを決める
  - 影響: 実験ログは `project/runs/` 配下に集約される

- 2025-12-29: 初回実験のベースラインを MLP と CNN の両方に設定し、PyTorch で実施
  - 背景: アーキテクチャ依存性を切り分け、比較可能な基準を2種類持ちたかった
  - 代替案: MLP のみ / CNN のみ / 先にタスクを増やす
  - 影響: 実験は4条件（MLP/CNN + OBL）で実行する

- 2025-12-29: OBL を多層化し、ROS-ALTH_02 の全系統を含むフル演算子ライブラリに拡張
  - 背景: ミニライブラリでは仮説検証が不十分だった
  - 代替案: 最小構成で継続 / 系統を段階的に追加
  - 影響: 実験コストが増加するため、CPU並列実行とガードレール監視を強化

- 2025-12-30: O(D^2) 系オペレータを除外した `fast` プロファイルを追加
  - 背景: GPU 実験で OBL が極端に重く、SoftSort/Sinkhorn/Attention がボトルネック疑い
  - 代替案: full のままチューニング / 演算子数だけ削減
  - 影響: `--obl-profile fast` で O(D^2) 演算を除外し、低ランク/グループ混合・permute blur・軽量活性を追加

- 2026-01-05: 勾配法高速化の新トラックを分離するため `project/docs/grad-speedup/` と `project/experiments/grad-speedup/` を新設
  - 背景: ROS-ALTH 系実験と目的/指標が異なるため、成果物と運用を分離したい
  - 代替案: 既存の docs/ と scripts/ に混在させる
  - 影響: 勾配法高速化の仕様/ノートは `project/docs/grad-speedup/`、実験設定は `project/experiments/grad-speedup/` に集約
  - 注記: 2026-01-05 の完全分離決定により `project/experiments/grad-speedup/` は廃止

- 2026-01-05: 勾配法高速化のコードを完全分離するため `project/grad-speedup/` を新設し、`project/experiments/grad-speedup/` を廃止
  - 背景: ROS-ALTH とコードや実験導線を混在させないため
  - 代替案: 既存の `project/src` を流用 / `project/experiments` 配下で継続
  - 影響: grad-speedup のコード/スクリプトは `project/grad-speedup/` に集約し、runs は `project/runs/grad-speedup/` を使用

- 2026-01-05: マルチエージェント運用のチケット置き場として `project/docs/tickets/ready/` を追加
  - 背景: 作業分担を明示し、履歴と引き継ぎを追跡しやすくしたい
  - 代替案: 一時ファイルとして個人管理 / 口頭共有
  - 影響: サブエージェント起動前のチケットは `project/docs/tickets/ready/` に置く

- 2026-01-05: grad-speedup のコード/ドキュメントを Git 追跡に含め、worktree から可視化
  - 背景: 未追跡ディレクトリは worktree に反映されず、サブエージェントが対象を参照できなかった
  - 代替案: 常に no-auto-worktree で実行 / 手動でファイル同期
  - 影響: grad-speedup の全ファイルをコミットし、worktree でも参照可能にする

- 2026-01-05: grad-speedup のチケット置き場を `project/docs/grad-speedup/tickets/` に分離
  - 背景: 既存実験と完全分離し、PM運用も混在させないため
  - 代替案: 共有の `project/docs/tickets/ready/` を継続利用
  - 影響: grad-speedup 用チケットは `project/docs/grad-speedup/tickets/ready/` に置く

- 2026-01-06: grad-speedup 実験を step ベース制御へ切り替え、eval_interval_steps=1000・max_steps=14,000 を既定に設定
  - 背景: epoch 指標より step 数の方が比較一貫性が高く、学習曲線の解析にも適するため
  - 代替案: epoch ベースを継続 / eval を epoch のみに固定
  - 影響: 既存の epoch ベース結果は再取得前提とし、以降の実験は step ベースで管理する

- YYYY-MM-DD: `<<判断>>`
  - 背景: `<<背景>>`
  - 代替案: `<<代替案>>`
  - 影響: `<<影響>>`
