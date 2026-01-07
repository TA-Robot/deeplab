# Grad-Speedup 実験総括（2026-01-07）

対象: CIFAR-10 / ResNet-18（CIFAR用調整）における「勾配法高速化モジュール」の単体効果・併用効果の探索。

本レポートは **mean_step_time ではなく time-to-target（壁時計）** を主指標としてまとめる。

---

## 0. まず結論（要点）

- **追記（2026-01-07）: momentum が主要な交絡だった**
  - SGD+mom0.9 baseline は 14k で 0.85 到達が 1/3 seed と弱かったが、**SGD+mom0.0 baseline は 0.85 到達が 3/3 seed で time-to-target 105.1±9.2s**（後述）。
  - これにより、前回「最速候補」と見えていた `l0l1-only` は **mom=0.0 に揃えた baseline に負ける**ことが判明（`l0l1` の速さは交絡が大きい）。
- **現状の14k（max_steps=14,000）での 0.85 到達（time-to-target）最速は `l0l1-only`**（3/3 seed到達、平均 **145.7s**）。ただし **baselineと momentum が揃っていない**（`l0l1` 系は momentum=0.0）ため、速度差の純粋比較としては **要再実験**。
- **公平寄りの比較（baselineと同じSGD+momentum=0.9）だと `LinBreg(λ=2e-4)` が良い**（3/3 seed到達、平均 **188.3s**）。最終精度も baseline より高い。
- **`GGNC global (ρ=1.0)` は early target（2kの0.70/0.75）では有効**だが、14kでは 0.85 到達が 2/3 seed、かつ **最終精度が悪化するseedがある**（「一度0.85を踏んでも最終で下がる」ケースあり）。
- **`SOAP-only` / `l0l1+SOAP(+Anderson)` は step数は減っても壁時計で負け**（SOAP方向計算のオーバーヘッドが支配的）。今回の目的（wall-clock短縮）には不利。
- **Layerwise GN（paper-accurate）は計算コストが過大**で、実運用の探索からは除外。GN-lite（層サブサンプル＋更新間引き）も 2k 時点で baseline に time-to-target で届かず、優先度を下げる。

---

## 1. 実験セットアップ（今回まとめた範囲）

基準仕様は `project/docs/grad-speedup/cifar10-implementation-spec.md` を参照。

今回、結果をまとめた実験は大きく2系列:

### A) 2k スクリーニング（seed=0）

- `max_steps=2,000`
- `eval_interval_steps=200`（= time-to-target の分解能は200 step単位）
- target（test acc）: `[0.60, 0.65, 0.70, 0.75, 0.85]`
- 目的: **早期（低〜中閾値）で壁時計が改善する候補**を拾う

### B) 14k アブレーション（seed=0/1/2）

- `max_steps=14,000`
- `eval_interval_steps=1000`（= time-to-target の分解能が粗い点に注意）
- target（test acc）: `[0.85, 0.90, 0.92, 0.94]`
- 目的: **0.85到達の再現性と壁時計**、および最終精度の傾向を見る

---

## 2. 2k スクリーニング結果（seed=0, time-to-target）

### 2.1 単体（代表）

| 系列 | Run ID | t@0.70 (s) | t@0.75 (s) | 備考 |
|---|---|---:|---:|---|
| Baseline (SGD+mom0.9) | `20260106-grad-speedup-screen2000-baseline-sgd-seed0` | 40.74 | 40.74 | 0.70/0.75が2000stepで同時到達（分解能=200step） |
| GGNC global (ρ=1.0) | `20260106-grad-speedup-tune2000-ggnc-global-rho1.0-sgd-seed0` | **24.64** | **34.78** | **非Muon系で最速**（このseedでのscreening） |
| GGNC layerwise (ρ=0.2) | `20260106-grad-speedup-tune2000-ggnc-layerwise-rho0.2-sgd-seed0` | 28.68 | 59.62 | 0.70までは速いが0.75で失速 |
| LinBreg (λ=2e-4,u=1) | `20260106-grad-speedup-tune2000-linbreg-l0.0002-u1-sgd-seed0` | 31.72 | 40.72 | baselineと同程度（0.75） |
| Anderson (m=5,i=5) | `20260106-grad-speedup-tune2000-anderson-m5-i5-sgd-seed0` | 43.26 | — | baselineより遅い |
| Adaptive Backtracking (best) | `20260106-grad-speedup-tune2000-backtrack-c1e-4-r0.8-m4-sgd-seed0` | 45.36 | — | 0.75に届かず |
| EoSS（tune） | `20260106-grad-speedup-tune2000-eoss-*` | — | — | **0.70未到達**（現状） |

参考（今回深掘り対象外だが比較用）:

- Muon (SGD): `20260106-grad-speedup-screen2000-muon-sgd-seed0` → t@0.70=24.64s, t@0.75=33.00s
- Muon (Adam): `20260106-grad-speedup-screen2000-muon-adamw-seed0` → t@0.85=56.20s（optimizerが異なるため baseline(=SGD) と単純比較しない）

### 2.2 併用（2kで確認できた範囲）

| 系列 | Run ID | t@0.70 (s) | 備考 |
|---|---|---:|---|
| GGNC global + Anderson | `20260106-grad-speedup-combo2000-ggnc-global-rho1.0-anderson-m5-i5-d0.5-sgd-seed0` | 36.57 | **GGNC単体(24.64s)より悪化** |
| LinBreg + Anderson | `20260106-grad-speedup-combo2000-linbreg-l1.5e-3-anderson-m5-i5-d0.5-sgd-seed0` | 43.16 | 併用メリット見えず |
| GGNC global + LinBreg | `20260106-grad-speedup-combo2000-ggnc-global-rho1.0-linbreg-*` | — | 0.70未到達（今回のパラメータでは） |

---

## 3. 14k アブレーション結果（target=0.85, time-to-target）

注意:
- `eval_interval_steps=1000` のため、**到達時刻/stepは1000 step刻み**（真の到達より遅く見積もられる可能性）。
- time-to-target は「最初に閾値を踏んだ時点」なので、**最終精度が後で下がるケース**があり得る（実際に発生）。

### 3.1 0.85到達の壁時計（SGD系）

| 系列 | Run ID | 到達seed数 | time-to-target@0.85 (s) | steps-to-target@0.85 | 最終test acc（14k） |
|---|---|---:|---:|---:|---:|
| Baseline (SGD+mom0.9) | `20260106-grad-speedup-step-baseline-resnet18-maxsteps14000-seeds012` | 1/3 | 262.0（n=1） | 14000（n=1） | 0.8457±0.0189 |
| Baseline (SGD+mom0.0) | `20260107-grad-speedup-step-baseline-mom0-resnet18-maxsteps14000-seeds012` | **3/3** | **105.1±9.2** | 6667±577 | **0.8885±0.0092** |
| GGNC global (ρ=1.0) | `20260107-grad-speedup-step-ggnc-global-rho1.0-resnet18-maxsteps14000-seeds012` | 2/3 | 203.6±15.6（n=2） | 9500±707（n=2） | 0.8299±0.0133 |
| GGNC layerwise (ρ=0.5) | `20260107-grad-speedup-step-ggnc-layerwise-rho0.5-resnet18-maxsteps14000-seeds012` | 0/3 | — | — | 0.83前後（seed毎に0.815〜0.849） |
| LinBreg (λ=2e-4) | `20260107-grad-speedup-step-linbreg-l2e-4-resnet18-maxsteps14000-seeds012` | **3/3** | 188.3±26.2 | 8333±1528 | **0.8753±0.0105** |
| l0l1-only（mom=0.0） | `20260106-grad-speedup-step-l0l1-only-resnet18-maxsteps14000-seeds012` | **3/3** | **145.7±14.3** | 7667±577 | **0.8827±0.0108** |
| SOAP-only | `20260106-grad-speedup-step-soap-only-resnet18-maxsteps14000-seeds012` | 3/3 | 519.2±1.2 | 4000±0 | 0.8913±0.0038 |
| l0l1 + SOAP + Anderson（mom=0.0） | `20260106-grad-speedup-step-l0l1-soap-anderson-resnet18-maxsteps14000-seeds012` | 3/3 | 287.9±2.5 | 2000±0 | **0.9124±0.0017** |
| l0l1 + SOAP（mom=0.0） | `20260106-grad-speedup-step-l0l1-soap-resnet18-maxsteps14000-seeds012` | 2/3* | 612.6±96.1（n=2） | 4500±707（n=2） | 0.8959±0.0039 |

注: `l0l1+SOAP` の seed-2 は `seed-2/summary.json` が欠落しており、time-to-target は集計できていない（最終accは `metrics.jsonl` の最終test evalから取得）。

### 3.2 ここから読み取れること

- **baselineの改善は “momentum=0.0” だけで大きく出た**（schedule無し/固定LRの現設定では、mom0.9よりmom0.0の方が「到達stepが減って速い」）。
- したがって、現時点の主張を整理すると:
  - `l0l1-only` の “速さ” は、少なくともこの設定では **step則そのものより momentum 交絡の寄与が大きい**可能性が高い。
  - 今後の比較は **SGD+mom0.0 を基準**に置いた上で、各モジュールの純効果（併用含む）を見るのが妥当。
- **SOAP方向（`direction=soap`）は step数は減っても、壁時計で大敗**しやすい（`mean_step_time_sec` が ~0.13s と baselineの ~0.016s に対して約8倍）。
- **LinBreg は step が多少増えても壁時計で勝てている**（step当たりは遅いが、到達stepが減る）。
- **GGNC global は「到達の早さ」は出るが、最終精度が落ちるseedがある**ため、guardrail（最終精度 or 安定到達）をどう置くかが重要。
- **`l0l1-only` が最速**だが、baselineと momentum 設定が揃っていないため、次は「baseline mom=0.0」または「l0l1 mom=0.9」も走らせて **寄与分解**が必要。

---

## 4. Layerwise GN / GN-lite（方向系）の検討結果

### 4.1 Paper-accurate Layerwise GN

- `gn-layerwise-exact` のプロファイルでは、1 step の大部分が GN の更新計算（matvec/CG）に支配され、現状の予算感では探索対象として重すぎるため **park**。

### 4.2 GN-lite（層サブサンプル＋更新間引き）

目的: 「重すぎるGNを、軽量化して wall-clock で勝てる可能性があるか」を 2k で早期判定。

代表（seed=0）:

| 系列 | Run ID | t@0.60 (s) | t@0.70 (s) | 備考 |
|---|---|---:|---:|---|
| Baseline | `20260106-grad-speedup-screen2000-baseline-sgd-seed0` | 32.52 | 40.74 |  |
| GN-lite topk5 | `20260107-grad-speedup-gnlite-2000-topk5` | 129.34 | 186.97 | 大幅に遅い |
| GN-lite topk5 interval=5 | `20260107-grad-speedup-gnlite-2000-topk5-int5` | 69.87 | — | 0.70未到達 |
| GN-lite topk5 interval=20 | `20260107-grad-speedup-gnlite-2000-topk5-int20` | 52.84 | — | 0.70未到達 |

結論: **更新間引き（interval=20）でも baseline に time-to-target で勝てない**ため、優先度を下げる（当面は方向系の大規模探索はしない）。

---

## 5. 次アクション（実験計画へのフィードバック）

### 5.1 まず直すべき比較の前提（クリティカル）

- **baselineの再整備**:
  - 14kで 0.85 到達が 1/3 seed しかない → baselineが弱く、相対比較が不安定。
  - `momentum` が手法間で揃っていない（`l0l1` 系は 0.0）→ まず寄与分解する。
- **time-to-targetの分解能**:
  - 14k側の `eval_interval_steps=1000` は粗すぎる（到達時刻が量子化される）。
  - 少なくとも **序盤（例: 0〜2k）だけ200step間隔**、以降は500〜1000に落とす等の「2段階eval」を検討。

### 5.2 併用探索（“単体最強”ではなく掛け算を見る）

現状、2kで併用メリットが見えたケースはまだ無い（GGNC+Andersonは悪化）。

次の優先順（壁時計目的）:

1) `GGNC global` × `LinBreg`（パラメータ再設計して再挑戦）
2) `GGNC global` ×（step制御：Backtracking/EoSS は現状弱いので後回し）
3) `LinBreg` ×（幾何/クリップ系）: clippingが疎性に与える影響確認

※ SOAP/GN 系は、まず「wall-clockで勝てる計算量設計」を作れない限り、併用探索の母集団から外す方針。

---

## 6. 参照（run artifacts）

- 実験成果物: `project/runs/grad-speedup/<run_id>/`
- 主要runの一覧は `project/docs/experiment-log.md` を参照。
