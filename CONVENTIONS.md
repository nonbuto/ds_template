# CONVENTIONS — 規約リファレンス

> **この層（L1）の役割**: 迷ったら引く「辞書」。**決まりだけを書き、判断はしない。**
> 規範文（〜せよ／禁止／必須）は `CLAUDE.md`（L0）が SSoT。ここには置かない。
>
> オンデマンドで読む。毎ターン読む必要はないが、**下記のいずれかを実装/記録するときは必ず参照する**。

---

## ディレクトリ規約

| ディレクトリ | 用途 |
|---|---|
| `data/raw/` | 生データ（読み取り専用） |
| `data/processed/` | 加工済みデータ（pickle形式） |
| `data/output/submissions/` | 提出CSVのみ（`submission_path()` で生成） |
| `data/output/oof/` | OOF・test予測の `.npy` ファイル |
| `data/output/models/` | 学習済みモデルファイル |
| `data/output/params/` | best_params JSON |
| `data/output/plots/` | 可視化画像（`.png`）。Claudeが読んで対話に使う |
| `experiments/` | log.csv + MLflowアーティファクト |
| `scripts/` | 再利用可能なスクリプト（後述） |
| `experiments/runs/` | 実験ごとの1回限りスクリプト |

> `data/output/` 直下にファイルを置かない。役割別サブディレクトリを必ず使う。

---

## スクリプト構成

**`scripts/`（テンプレート本体・再利用可能）**

| ファイル | Stage | 役割 |
|---|---|---|
| `scripts/train.py` | 1・4 | CV学習の汎用骨格（モデル・特徴量をconfigで切り替え） |
| `scripts/feature_study.py` | 4 | 1列ΔCV計測（FE仮説の効果測定） |
| `scripts/optimize_hp.py` | 3・5 | Optuna HP探索 |
| `scripts/predict.py` | 全般 | OOF予測→提出ファイル生成 |
| `scripts/blend.py` | 6 | アンサンブル・ブレンド |
| `scripts/visualize.py` | 2 | EDA可視化→`data/output/plots/`に画像保存 |
| `scripts/feature_report.py` | 随時 | 特徴量重要度・ΔOOF棒グラフを画像生成 |

**`experiments/runs/`（コンペ固有・使い捨て）**

命名規約: `exp{NNN}_s{stage}_{内容}.py`

```
experiments/runs/
  exp001_s1_lgb_baseline.py       ← Stage 1: 最小ベースライン
  exp003_s3_hp_lgb_optuna.py      ← Stage 3: 作業用HP調整
  exp042_s4_fe_age_sq.py          ← Stage 4: 特徴量追加
  exp099_s5_hp_lgb_full.py        ← Stage 5: 本格HP最適化
  exp171_s6_lgb_cb_blend.py       ← Stage 6: アンサンブル
```

- `exp{NNN}`: `experiments/log.csv` の `experiment_id` と一致させる
- `s{stage}`: どのステージの実験かが一目で分かる
- `scripts/` のスクリプトを呼び出すラッパーとして書くことを推奨

## スクリプトの標準構成

```python
"""
スクリプトの説明

使い方:
    uv run python scripts/xxx.py --option value
"""

# 1. 標準ライブラリ
# 2. サードパーティ
# 3. ローカル（src.*）— パスは必ず src.config からインポート

from src.config import PROCESSED_DATA_DIR, TARGET_COL, RANDOM_STATE

# ──────────────────────────────────────────────
# TODO: コンペごとにここを変更する
# ──────────────────────────────────────────────
FEATURES: list[str] = []  # コンペ固有の特徴量リスト

# メイン処理
def main():
    ...

if __name__ == "__main__":
    main()
```

---

## コーディング規約

- パスは必ず `src.config` からインポート（ハードコード禁止）
- 乱数シードは `RANDOM_STATE`（`src.config`から）
- 特徴量名: snake_case・スペースなし
- `src/` 配下に型ヒントを付ける

---

## 可視化の規約

```python
import matplotlib
matplotlib.use("Agg")  # 非インタラクティブ（ファイル保存のみ）
import matplotlib.pyplot as plt
from src.config import PLOTS_DIR

fig, ax = plt.subplots(figsize=(10, 4))
# ... 描画 ...
fig.savefig(PLOTS_DIR / "eda_tenure_target_dist.png", dpi=120, bbox_inches="tight")
plt.close(fig)
```

- 画像は `data/output/plots/` に保存する（直接表示しない）
- Claude が `Read` ツールで読んで対話に使う
- ファイル名: `{prefix}_{変数名}_{テーマ}.png`

> この節が可視化規約の **SSoT**。CLAUDE.md・各スキルはここを参照するだけにする
> （v5 では 4 箇所に重複していた）。

---

## 提出ファイルの命名規約

提出CSVは必ず `submission_path()` ヘルパーで生成する:

```python
from src.config import submission_path
sub_path = submission_path(model="lgb_cb_blend", oof_score=0.91777, exp_id="171")
# → data/output/submissions/sub_171_lgb_cb_blend_0.91777_20260331_2347.csv
sub.to_csv(sub_path, index=False)
```

命名規約: `sub_{exp_id}_{model}_{oof_score:.5f}_{yyyymmdd_HHMM}.csv`

- `exp_id`: `experiments/log.csv` の `experiment_id` と紐付ける（省略可）
- `model`: ブレンド内容が分かる短い識別子（例: `lgb`, `lgb_cb_blend`, `greedy_ens`）
- `oof_score`: ファイル名だけで品質が分かるようにする
- タイムスタンプ: 同名ファイルの上書き防止と生成順の追跡

---

## 実験管理（log.csv）

`experiments/log.csv` の主要カラム:

| カラム | 記録タイミング | 説明 |
|---|---|---|
| `experiment_question` | `/ds-new-experiment` | この実験で何を明らかにしたいか |
| `success_criteria` | `/ds-new-experiment` | どんな結果なら成功か |
| `abort_criteria` | `/ds-new-experiment` | どんな結果なら中止するか |
| `cv_val_mean` / `oof_score` | 学習完了時 | OOFスコア |
| `submit_score` | `/ds-kaggle-submit` | LBスコア |
| `oof_lb_gap` | `/ds-kaggle-submit` | OOF tuned − LB（正=OOF過大評価、負=OOF過小評価）。乖離が大きい実験は汎化リスクあり |
| `learning` | `/ds-kaggle-submit` | この実験から何を学んだか |

> **ベスト実験の管理は SESSION.md のスコアテーブルで一元化する。** log.csv にベストフラグ列（`is_best` 等）を持たない
> — フラグ方式は過去コンペで 100 実験超のうちほぼ全行が未記入となり形骸化した。二重管理をやめ、SESSION.md の上書き更新に集約する。

---

## ExperimentTracker の使い方

すべての学習実行は `src.experiment` モジュールを通じて `experiments/log.csv` に記録する。

```python
from src.experiment import ExperimentTracker
from src.config import EXPERIMENT_NAME

tracker = ExperimentTracker(experiment_name=EXPERIMENT_NAME)
run_id = tracker.start_run(
    run_name="fold_training",
    description="実験の概略をここに記述",
    model="lgb",
    features="baseline_features",
)

# 学習ループ内で記録
tracker.log_fold_scores(fold, trn_score, val_score)

# 学習完了後
tracker.end_run(
    train_scores=train_scores,
    val_scores=val_scores,
    oof_score=oof_score,
    n_features=len(features),
)
```

## 実験名の命名規則

`{competition}_{model}_{feature_set}_{variant}`

例:
- `<competition-slug>_lgb_baseline`
- `<competition-slug>_lgb_fe_v2`
- `<competition-slug>_cb_optuna`
- `<competition-slug>_ensemble_lgb_cb`

---

## SESSION.md の構成と上限

SESSION.md は「今どこにいるか」を1画面で示すライブダッシュボード。
**アペンドではなく各セクションを上書き更新する**（蓄積禁止）。

**固定構成**（この順序を守り、追加セクションを作らない）:

| # | セクション | 上限・注意 |
|---|---|---|
| 1 | ファイルヘッダー | 最終更新日時 |
| 2 | 現在のステージ | 1〜2行。「次にやること」を1行目に |
| 3 | スコア状況 | **ベストのみ1テーブル**（新テーブル追加禁止） |
| 4 | 直近の実験 | **最大10件**。11件目以降は最古から削除 |
| 5 | 次にやること | 箇条書き最大5件 |
| 6 | 未解決の問い | ブロッカーのみ。解決済みは削除 |
| 7 | 重要な方針 | 実験を通じて確定した原則のみ |

**スコアテーブルの形式**（乖離列で「OOFは高いがLBで悪化」を早期検知する）:

| 指標 | OOF tuned | LB | OOF-LB乖離 | 実験ID |
|---|---|---|---|---|
| ベスト | 0.XXXXX | 0.XXXXX | ±0.XXXXX | expNNN |

**上限値の定義**: **ファイル全体は 80 行**、そのうち **「直近の実験」は最大 10 件**。
この 2 つは別物であり両立する（v5 では別々のファイルに書かれ非同期だった）。
`/ds-resume` が 80 行超過を検知したら、完了済みエントリを削除して収める。

**禁止パターン**: 「最後に完了したこと」を複数回追記する / 複数のスコアテーブルを並存させる /
過去セッションの履歴を蓄積する（git history に残るため SESSION.md には不要）。

---

## ブランチ管理

```
main              ← テンプレート本体（コンペ固有コード禁止）
comp/<competition> ← コンペ適用ブランチ（日々の実験コミットの置き場）
exp/<実験名>      ← 大きな方向転換のみ（下記基準参照）
template/fix-XXX  ← テンプレ改善ブランチ
```

**`exp/` ブランチを作る基準（すべての実験には不要）:**

| 作る | 作らない |
|---|---|
| 新しいアルゴリズムの追加（XGB, NN, RF など） | FEの1列追加 |
| 特徴量セットの大幅再設計（列数 ±20% 以上） | HPチューニング（Optuna） |
| CV戦略の変更（StratifiedKFold → GroupKFold など） | ブレンド重みの調整 |
| アーキテクチャ変更（Stacking の試験的導入） | 既存スクリプトのバグ修正 |

→ 上記に当てはまらない実験は `comp/<competition>` ブランチ上でコミットしてよい。

---

## コミット規約

**コミットのタイミング（3つのルール + 並行実行ルール）:**

1. **学習完了直後にコミットする** — OOFスコアが判明した直後 **5 分以内**。時間を置かない
2. **1実験 = 1コミット** — 複数の変更を一度のコミットにまとめない。何が効いたか追跡できなくなる
3. **`/ds-kaggle-submit` の前にコミット済みであること** — `git status` がcleanでなければ提出しない

**並行実行時の特例ルール（バックグラウンド実行時も厳守）:**

複数の実験をバックグラウンドで並行実行している場合でも、**各実験の OOF 判明ごとに個別 commit する**:

```
❌ NG パターン:
  exp_A 完了 → 待機 → exp_B 完了 → 待機 → exp_C 完了 → まとめて 1 commit

✅ OK パターン:
  exp_A 完了 → commit_A → exp_B 完了 → commit_B → exp_C 完了 → commit_C
```

待ち時間の活用:
- バックグラウンド実行中の「次の実験設計」は OK
- しかし完了した実験の commit は **絶対に後回しにしない**
- log.csv の更新も同じタイミングで（バッチ更新は禁止）

> **教訓 (過去事例)**: 7 実験を 1 コミットにまとめ、log.csv 更新を最終日に一括実施した結果、後追いで「どの変更が効いたか」が追跡困難になった

**実験番号の衝突防止:**

新しい実験番号を決める前に、必ず以下で既存ファイルを確認する:
```bash
ls experiments/runs/ | grep "^exp" | sort | tail -5
# log.csv の最大 experiment_id も確認
tail -3 experiments/log.csv | cut -d',' -f2
```
未コミットの実験スクリプトが `experiments/runs/` に存在する場合（`git status` で `??` 表示）、
それらの番号は使用済みとして扱い、それより大きい番号を使う。

**コミットメッセージの形式:**

```
feat(expNNN): <実験の目的を1文で>

OOF=<score>  model=<model>  features=<feature_set>
```

例:
```
feat(exp042): col_A×col_B の交互作用特徴量を追加

OOF=0.91688  model=lgb  features=fe_v7_interaction
```

- `expNNN` は `experiments/log.csv` の `experiment_id` と一致させる
- 本文行（2行目）は `tracker.end_run()` が自動提案する
- `feat` / `fix` / `refactor` を使い分ける（FE追加=feat, バグ修正=fix, リファクタ=refactor）
