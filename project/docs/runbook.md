# Runbook（管理者が整備する）

このドキュメントは、管理者が「運用が詰まらないように」最低限の手順を集約する場所です。

## セットアップ

- Python 仮想環境を作成し、CPU版 PyTorch を導入:

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

- GPU 環境の場合は CUDA 対応の PyTorch を導入してから `--device cuda:0` を指定:
  - 例: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  - 動作確認:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

- `--deterministic` を使う場合は cublas の設定が必要:
  - `export CUBLAS_WORKSPACE_CONFIG=:4096:8`

- データ格納先は `project/data/`（自動で作成されます）
- 実験ログは `project/runs/` 配下に保存されます

## 実行

### 1) MLP ベースライン

```bash
cd project
python run_mnist_experiment.py \
  --model mlp \
  --dataset mnist \
  --seeds 1,2,3 \
  --epochs 5 \
  --batch-size 128 \
  --num-threads 8 \
  --deterministic \
  --run-id 20251229-mlp-baseline
```

### 2) MLP + Operator Basis Layer (K=32)

```bash
cd project
python run_mnist_experiment.py \
  --model mlp-obl \
  --dataset mnist \
  --seeds 1,2,3 \
  --epochs 5 \
  --batch-size 128 \
  --num-threads 8 \
  --deterministic \
  --obl-profile full \
  --gamma 0.1 \
  --beta-init 0.01 \
  --run-id 20251229-mlp-obl
```

### 3) CNN ベースライン

```bash
cd project
python run_mnist_experiment.py \
  --model cnn \
  --dataset mnist \
  --seeds 1,2,3 \
  --epochs 5 \
  --batch-size 128 \
  --num-threads 8 \
  --deterministic \
  --run-id 20251229-cnn-baseline
```

### 4) CNN + Operator Basis Layer (K=32)

```bash
cd project
python run_mnist_experiment.py \
  --model cnn-obl \
  --dataset mnist \
  --seeds 1,2,3 \
  --epochs 5 \
  --batch-size 128 \
  --num-threads 8 \
  --deterministic \
  --obl-profile full \
  --gamma 0.1 \
  --beta-init 0.01 \
  --run-id 20251229-cnn-obl
```

補足:
- デフォルトの隠れ層構成:
  - MLP: 2 hidden layers (256, 256)
  - CNN: 2 fc layers (256, 128)
- `--beta-l1 1e-4` を付けると演算子寄与のスパース化を促進できます
- `--operator-dropout 0.1` は計算削減や正則化を試したい時に追加
- `--num-threads` は環境の CPU コア数に合わせて調整
- パラメータ数の差が大きい場合は `--hidden-dim` / `--cnn-hidden-dim` を調整して再実行
- 出力: `config.json`, `env.json`, `metrics.jsonl`, `summary.json` が `project/runs/<run-id>/` に生成
  - より細かい指定は `--mlp-hidden-dims` / `--cnn-fc-dims` を使う
  - OBL は `--obl-profile full` がデフォルト（`fast` は O(D^2) 演算を除外し、低ランク/グループ混合を追加）
  - ランダムプログラム数は `--obl-programs` で上書き可能
  - 正規化は `--obl-norm layernorm|rmsnorm`
  - 乱数固定は `--obl-seed <int>` を使用
  - 実装の詳細は `project/docs/obl-implementation.md` を参照
  - データセットは `--dataset mnist|fashion-mnist|cifar10` で切替
  - デフォルトはダウンロード無効。`--download` で取得可能
  - `project/data/` に手動配置する場合は `--data-dir` を指定

### バックグラウンド並列（例）

```bash
cd project
mkdir -p runs/logs
nohup python run_mnist_experiment.py --model mlp --seeds 1,2,3 --epochs 5 --batch-size 128 --num-threads 2 --deterministic --device cpu --run-id 20251229-mlp-baseline-par > runs/logs/20251229-mlp-baseline-par.out 2>&1 &
nohup python run_mnist_experiment.py --model mlp-obl --seeds 1,2,3 --epochs 5 --batch-size 128 --num-threads 2 --deterministic --device cpu --obl-profile full --gamma 0.1 --beta-init 0.01 --run-id 20251229-mlp-obl-par > runs/logs/20251229-mlp-obl-par.out 2>&1 &
nohup python run_mnist_experiment.py --model cnn --seeds 1,2,3 --epochs 5 --batch-size 128 --num-threads 2 --deterministic --device cpu --run-id 20251229-cnn-baseline-par > runs/logs/20251229-cnn-baseline-par.out 2>&1 &
nohup python run_mnist_experiment.py --model cnn-obl --seeds 1,2,3 --epochs 5 --batch-size 128 --num-threads 2 --deterministic --device cpu --obl-profile full --gamma 0.1 --beta-init 0.01 --run-id 20251229-cnn-obl-par > runs/logs/20251229-cnn-obl-par.out 2>&1 &
```

### GPU 並列バッチ（MNIST/FashionMNIST/CIFAR10）

```bash
cd project
mkdir -p runs/logs
nohup bash scripts/launch_gpu_experiments.sh > runs/logs/launch_gpu_experiments.out 2>&1 &
```

補足:
- 既定の並列数は `MAX_PARALLEL=2`。GPU 余力に合わせて調整する
- `DATASETS=mnist,fashion-mnist,cifar10` を変更して対象を絞り込める
- `DEVICE=cuda:0` / `NUM_WORKERS=4` / `BATCH_SIZE=128` などは環境変数で上書き可能
- `OBL_PROFILE=fast` で O(D^2) 演算を除外したプロファイルを使用

## テスト

- スモークテスト（1エポックで動作確認）:

```bash
cd project
python run_mnist_experiment.py --model mlp --epochs 1 --batch-size 64 --run-id smoke-mlp
```

## ダッシュボード

レポート用 JSON を生成して、`project/dashboard/` で表示します。

```bash
cd project
python scripts/build_report.py --runs-dir runs --output dashboard/data/report.json
python -m http.server 8000 --directory dashboard
```

ブラウザで `http://localhost:8000` を開きます。

## よくある詰まり

- PyTorch が入らない: CPU wheel を明示
  - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- `CUDA requested but not available`: `--device cpu` を指定する
