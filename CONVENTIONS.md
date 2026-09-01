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
- ファイル名: **`{実験ID3桁}_{通し番号2桁}_{内容}.png`**（例: `042_01_importance_top30.png`）
  - EDA 段階など実験 ID が確定していない場合は `eda_{変数名}_{テーマ}.png` を使う
  - **数字プレフィックスを必ず先頭に置く**理由: `ls` のソート順が作成順と一致し、
    「どの実験に紐づく、いつ作られた図か」が後から追える。過去コンペでは
    `oof_dist_079_vs_080_xgb.png` / `loo_mlp_delta_oof_loss.png` のように
    命名がその場限りになり、時系列も紐付けも追えなくなった

**日本語フォント（必須の初期設定）**: uv 管理の venv には CJK フォントが入っていないため、
matplotlib で日本語ラベルを描くと文字化け（tofu 表示）する。可視化スクリプトの冒頭で
`src/utils/plot_style.py` の `setup_japanese_font()` を呼ぶ（`scripts/visualize.py` 等では実施済み）。

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

## OOF / test 予測ファイルの命名規約（派生アンサンブルの識別）

`data/output/oof/` に保存する `oof_{tag}.npy` / `test_{tag}.npy` は、
**「独立した基底モデルの予測」か「既存メンバーを結合した派生アンサンブルの出力」か**を
ファイル名だけで判別できる状態に保つ。

- **派生アンサンブル（ブレンド・スタッキング・rank average 等の出力）には必ず `_ens_` を含める**
  例: `oof_216_ens_stack_logit_pall.npy`, `oof_263_ens_p5_signed.npy`
- 独立した基底モデル（単一アーキテクチャの学習結果）には `_ens_` を含めない
  例: `oof_063_lgb_h012.npy`, `oof_144_realmlp_tuned_multiseed5.npy`

> **なぜ必須か**: プール自動探索（`discover_pool()` 等）は「派生アンサンブルを除外語の
> ブラックリストで弾く」実装になりがちだが、**新しい命名パターンを登録し忘れると
> 自分自身の出力を「独立メンバー」として再投入してしまう**（自己参照リーク）。
> 過去コンペでは同一コンペ内で 2 回発生し、2 回目は bagged OOF が異常値（seed 間 std が
> 通常の 17 倍）を示したことで偶然発覚した。異常値がレンジ内に収まっていれば
> 気づかず採用していた（→ `PLAYBOOK.md#教訓アーカイブ実測値つき` L-23）。
> `_ens_` マーカーを規約にすれば、除外側は 1 パターンの検査で済み、列挙漏れが構造的に消える。

**新しい派生アンサンブルのスクリプトを書いたら、プールへ投入する前に必ず確認する**:

```bash
python -c "import re; print(bool(re.search(r'_ens_', '<新しいファイル名>')))"  # True であること
```

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
| **単体ベスト** | 0.XXXXX | 0.XXXXX | ±0.XXXXX | expNNN |

**「単体ベスト」行は常設**（アンサンブルを組んでいる期間も維持する）。理由は 2 つ:
①実務デプロイでは 400 メンバーのスタックより単体モデルの方が扱いやすく、単体精度そのものに価値がある
②アンサンブルの伸びが止まった局面で「単体を鍛え直す」という選択肢を視野から外さないため（→ CLAUDE.md Stage 5）

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

---

## 付録A. 旧番号と恒久 ID の対応表

v5 までの `指針#N` は v6 で恒久 ID に置き換えた。**この表は v6 限りの移行用**（v7 で削除する）。
過去の SESSION.md・log.csv・FE_HYPOTHESES.md に残る旧番号を読むときに使う。

| 恒久 ID | 旧番号 | 主題 |
|---|---|---|
| `G-PURPOSE` | #1, #3 | 目的を先に言語化し、直前の学びに接続する |
| `G-CAUSAL` | #2 | FE は「なぜ効くか」を説明できるものだけ提案する |
| `G-STEPWISE` | #6, #15 | 1 実験 = 1 つの問い = 1 コミット |
| `G-PERSIST` | #4, #7, #8, #12, #14 | 「飽和」を宣言しない。問いを立て直す |
| `G-SOURCE` | #5, #10, #11 | モデルの多様性より情報源の多様性 |
| `G-FAIR` | #13, #22 | 棄却の前に比較条件を揃える |
| `G-DIAG` | #31 | CV 内部診断を常設の判断軸にする |
| `G-NOISE` | #17, #23 | その差は測れるのか（量子とノイズ床） |
| `G-OOF` | #18 | OOF は足切りに使い、1 位の選定には使わない |
| `G-TWOAXIS` | #21 | OOF 最大化と pub_oof_gap 最小化の二軸評価 |
| `G-OVERFIT` | #20, #24, #25 | 「OOF↑なのに LB↓」の 3 変奏 |
| `G-FULLCV` | #26 | 単一分割の兆候はフル CV で再確認する |
| `G-INFOCEIL` | #28 | 情報天井の判定と情報源への切り替え |
| `G-CEILING` | #19, #27 | 天井帯では「当てにいく」のをやめ「集約する」 |
| `G-CALIB-SUB` | #29 | 提出枠を判断基準の検証にも使う |
| `G-MECH` | #9, #16 | 規律は機械に守らせる |
| `G-BLOCKER` | #30 | 技術的ブロッカーを「回避＝解決」としない |

**なぜ番号をやめたか**: 指針を統合・並べ替えるたびに `#N` の参照が全ファイルで壊れ、v5 では
「ヘッダーが `#1-22` のまま」「`#20` と `#17` の取り違え」等のバグが 8 件蓄積していた。
恒久 ID なら統合しても参照が壊れず、未定義 ID は `scripts/doc_audit.py` の C3 が機械的に検出する。
