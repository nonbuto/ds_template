# PLAYBOOK — 実行レシピ集（局面参照用）

> **このファイルの位置づけ**
>
> `CLAUDE.md` は「毎ターン守る原則・判断基準（精神）」を持つ。
> この `PLAYBOOK.md` は「その局面に来たら読む実行レシピ（手順・コード・コマンド）」を持つ。
>
> - **判断に迷ったら** → `CLAUDE.md`（原則・AI指針）
> - **手順を実行するなら** → このファイル（該当セクションを Read してから着手）
>
> CLAUDE.md の各所から `→ PLAYBOOK.md#<セクション>` で参照される。
> ここに書かれた手順も CLAUDE.md の原則に従うこと（レシピは原則を上書きしない）。

---

## 目次

1. [合成データコンペ向けガイダンス](#合成データコンペ向けガイダンス)
2. [Kaggle Notebook 環境サポート](#kaggle-notebook-環境サポート)
3. [Kaggle GPU ワークフロー（CSV提出コンペ）](#kaggle-gpu-ワークフローcsv提出コンペ)
4. [Notebook提出コンペ向けフロー](#notebook提出コンペ向けフロー)
5. [データ読み込みパターン](#データ読み込みパターン)
6. [早期アーキテクチャサーベイの手順（Stage 1.5）](#早期アーキテクチャサーベイの手順stage-15)
7. [AV 診断（Adversarial Validation）](#av-診断adversarial-validation)
8. [アンサンブル探索の手順（Stage 6）](#アンサンブル探索の手順stage-6)
9. [アンサンブル棄却分析](#アンサンブル棄却分析)
10. [FE の採用・棄却判断（詳細）](#fe-の採用棄却判断詳細)
11. [Final 2 候補プールと Persona 投票](#final-2-候補プールとpersona-投票)
12. [天井帯での意思決定ツールキット](#天井帯での意思決定ツールキット)
13. [既知の落とし穴（ライブラリ別）](#既知の落とし穴ライブラリ別)
14. [教訓アーカイブ（実測値つき）](#教訓アーカイブ実測値つき)

---

## 合成データコンペ向けガイダンス

> **参照元**: CLAUDE.md `G-SOURCE`・「学習サイクル」。Kickoff で合成データと判明したら EDA 着手前に読む。

**なぜ外部シグナルが効くか（メカニズム）:**

合成データは元データを基に統計的に生成されるが、**ターゲットとの細かい相関関係は圧縮・平滑化**される傾向がある。
元データの統計量を特徴量として注入することで、合成プロセスで失われたシグナルを補完できる。

**最初に試すのは「元データを train に連結する」**（Playground で最も定石。上の 3 パターンより先）:

```python
orig = pd.read_csv("data/raw/original.csv")          # 生成元データセット
orig = orig[train.columns.drop(ID_COL, errors="ignore")]   # 列を揃える
aug = pd.concat([train, orig], ignore_index=True)
w = np.r_[np.ones(len(train)), np.full(len(orig), 0.5)]    # 元データは軽く
```

- **`sample_weight` で重みを分ける。** 合成データとは分布が違うので、同じ重みで混ぜると
  test（＝合成側）に合わない方向へ引っぱられる。0.3〜1.0 を 1 列 FE と同じ手順で計測する
- **元データは train 側にだけ足す。** OOF の評価は**元の train 行だけ**で行う
  （元データ行を検証に混ぜると、test と分布の違う行で測ることになりスコアの意味が変わる）
- **重複に注意**: 合成データが元データの行をそのまま含むことがある。
  `src/validation.py` の `validate_no_leakage` が train/test の同一行を検知する

**次に試す 3 パターン（元データの統計量を注入する）:**

| パターン | 内容 | 実装コスト |
|---|---|---|
| **カテゴリ別ターゲット率** | 元データの各カテゴリ列×ターゲット率をマッピング（外部ターゲットエンコーディング） | 低 |
| **数値分布特徴量** | 元データのターゲット群/非ターゲット群の分布との距離（z-score, percentile, Euclid距離） | 中 |
| **生成ルール逆算** | 元データで `DecisionTreeClassifier(max_depth=None)` を学習し、`tree.apply(X)`（葉ノードID）をfold内TargetEncoderでエンコード。元データでBA=1.0を達成できる場合は生成ルールを直接注入できる可能性がある | 中 |

**生成ルール逆算の注意点:**
- まず `DecisionTreeClassifier(max_depth=None).fit(orig_X, orig_y)` で元データに対して BA=1.0 を達成できるか確認する
- BA=1.0 を達成できない場合はこのパターンは無意味
- 達成できた場合でも、LGB が内部で同等の分割を既に学習している可能性がある（ΔOOF がゼロに近い）
- `tree.predict_proba()` の硬確率（0/1）を特徴量にするのは NG。必ず `tree.apply()`（葉ノードID）＋ fold内TargetEncoder を使うこと

**実装上の注意点:**

- カテゴリ別ターゲット率は1-wayで十分。2-way以上はtree splitsと冗長になりやすい
- percentile計算は `percentileofscore` をループすると O(N²) になる。大規模データでは `np.searchsorted` を使う:
  ```python
  # 高速版（O(N log N)）
  sorted_ref = np.sort(ref_vals)
  df["pct"] = np.searchsorted(sorted_ref, df[col].values, side="right") / len(sorted_ref)
  ```
- 元データが小規模（数千件）でも有効。合成データが数十万件あっても外部シグナルは機能する

**外部シグナルFE × アーキテクチャの相性（重要）:**

外部シグナルFEの効果はモデルの帰納バイアスに依存する。**主軸アーキテクチャ1つの結果だけで採否を確定しない**:

| アーキテクチャ | 外部シグナルFEとの相性 | 理由 |
|---|---|---|
| Tree系（LGB/XGB/CB） | 付加価値が薄れやすい | 非線形変換・分割を自力再現できるため、連続値シグナルの寄与が既存分割と重複しやすい |
| NN系（MLP系等） | 真価を発揮しやすい | 「良い座標系」（連続値の外部シグナル）が精度に直結する |

- Tree系で ΔOOF が閾値未満でも、**NN系が候補にあるなら同一FEを移植評価してから棄却を確定する**
- Tree系での外部シグナルFEは「OOF過小評価・LB浮上」の挙動を示すことがある（判断は CLAUDE.md `G-TWOAXIS` の二軸評価に従う）

> **教訓 (過去事例)**: 元データのクラス条件付き percentile 特徴量が Tree系では微小改善に留まったが、
> 同一特徴量を NN系に移植したところコンペ最大の LB 跳躍を生んだ。
> Tree系単独の結果で棄却していたらこの跳躍は生まれなかった。

---

## Kaggle Notebook 環境サポート

このテンプレートはローカル環境と Kaggle Notebook 環境の両方で動作するよう設計されている。
`src/config.py` が自動的に環境を検出し、パスを切り替える。

**環境検出の仕組み:**

```python
from src.config import IS_KAGGLE, RAW_DATA_DIR, OOF_DIR

# ローカル環境: IS_KAGGLE = False
#   RAW_DATA_DIR = <project_root>/data/raw/
#   OOF_DIR      = <project_root>/data/output/oof/

# Kaggle Notebook 環境: IS_KAGGLE = True
#   RAW_DATA_DIR = /kaggle/input/<competition>/   ← コンペスラッグで自動決定
#   OOF_DIR      = /kaggle/working/data/output/oof/
```

---

## Kaggle GPU ワークフロー（CSV提出コンペ）

> `.ipynb` 変換に marimo を使う。用途と禁止事項は `CONVENTIONS.md#marimo-の用途`
> （可視化 EDA には使わない —— Claude はレンダリング結果を認識できない）。

GPU を使う重い学習をKaggle Notebook で実行し、成果物（OOF .npy, submission.csv）をローカルに回収するフロー。

> ## ⚠️ 先に読む: Step 1（Dataset 同期）は **AI が実行するとブロックされる**
>
> `rsync` でプロジェクト全体を一時ディレクトリへ一括コピーする操作は、AI エージェントの実行環境で
> **「Data Exfiltration」のハードブロック**として拒否される（ユーザーが明示的に許可しても解除されない）。
> コマンドを言い換えても、同期先を作る `mkdir` だけでも再度ブロックされる。
>
> **→ 2 つの経路から選ぶ:**
>
> | 経路 | 手順 | 向いている場面 |
> |---|---|---|
> | **(A) 自己完結 Notebook**（推奨・AI 実行可） | Dataset 同期を行わず、必要な処理を**1 ファイルに閉じた**スクリプトとして書き、`.ipynb` 化して push する。アップロードは `.ipynb` + `kernel-metadata.json` の 2 ファイルのみ（`dataset_sources: []`） | 実験数が少ない／単発の GPU 実行 |
> | **(B) Dataset 同期**（下記 Step 1〜4） | **ユーザー自身のターミナルで `rsync` と `kaggle datasets` を実行してもらう**。AI は Step 2 以降（`.ipynb` 変換・push・回収）を担当する | 実験数が多く、コード重複が許容できない |
>
> **(A) のトレードオフ**: `src/config.py` や `scripts/train.py` の共通ロジックを都度複製することになり DRY に反する。
> 実験ごとに最新コードの同期が必要。
>
> **速度の注意**: Kaggle Notebook の **CPU 実行はローカルより遅いことがある**（実測で約 2〜3 倍）。
> CLAUDE.md の「30 分ルール」で Kaggle 実行を検討する際、**GPU を使わないなら速度改善は期待できない**。

**Step 1: テンプレートを Kaggle Dataset として同期する**（経路 B。**ユーザーのターミナルで実行**）

```bash
# ⚠️ --dir-mode zip は .kaggleignore を無視する。rsync で除外ファイルを管理すること
# ⚠️ 以下は AI では実行できない。ユーザー自身のターミナルで実行すること

# 同期先の一時ディレクトリを準備（<slug> はコンペスラッグに置換）
rsync -a --delete \
  --exclude='.git' --exclude='.venv' --exclude='data/' \
  --exclude='kaggle_nb/' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.DS_Store' \
  . /tmp/kaggle_dataset_<slug>/
cp dataset-metadata.json /tmp/kaggle_dataset_<slug>/

# 初回: Dataset を作成
kaggle datasets create -p /tmp/kaggle_dataset_<slug> --dir-mode zip

# 2回目以降: 変更を新バージョンとして push
kaggle datasets version -p /tmp/kaggle_dataset_<slug> -m "exp{NNN} 追加" --dir-mode zip
```

Dataset 名: `{your-username}/ds-template-{competition}` として登録される。
`dataset-metadata.json` をプロジェクトルートに置くこと（`id` と `title` を設定）。

> **注意**: `--dir-mode zip` は `.kaggleignore` を無視するため、rsync で一時ディレクトリに
> 必要なファイルだけコピーしてから push する。直接 `-p .` で push すると `.venv/` 等が
> 含まれてアップロードサイズが数百MBになる。

**Step 2: 実験スクリプトを .ipynb に変換する**

```bash
# 通常スクリプト → Kaggle Notebook 用 .ipynb
uv run python -m scripts.to_kaggle_nb experiments/runs/exp001_s1_lgb_baseline.py \
  --competition <competition-slug> \
  --dataset-name ds-template-<competition> \
  --gpu   # GPU を有効化する場合

# 生成先: kaggle_nb/exp001_s1_lgb_baseline.ipynb
#         kaggle_nb/kernel-metadata.json
```

**Step 3: Notebook を Kaggle に push して実行する**

```bash
# push（初回: Notebook を作成、2回目以降: 上書き更新して自動実行開始）
kaggle kernels push -p kaggle_nb/

# 実行状況を確認
kaggle kernels status {username}/exp001-s1-lgb-baseline
# → "status": "running" / "complete" / "error"
```

**Step 4: 成果物をローカルに回収する**

```bash
# 学習完了後、出力ファイルを取得
kaggle kernels output {username}/exp001-s1-lgb-baseline -p kaggle_nb/output/

# OOF .npy をローカルの data/output/oof/ に移動
mv kaggle_nb/output/data/output/oof/*.npy data/output/oof/
# submission CSV も同様
mv kaggle_nb/output/data/output/submissions/*.csv data/output/submissions/
```

**フロー全体:**

```
[ローカル] スクリプト編集 (.py)
    ↓ kaggle datasets version (Step 1)
[Kaggle]  Dataset に最新コードが反映される
    ↓ to_kaggle_nb.py (Step 2)  →  kaggle kernels push (Step 3)
[Kaggle]  GPU 環境で学習実行（最大12時間）
    ↓ kaggle kernels output (Step 4)
[ローカル] OOF .npy / submission.csv を回収 → commit → LB提出
```

**注意事項:**

- `/kaggle/working/` のみ書き込み可能（`/kaggle/input/` は読み取り専用）
- `/kaggle/working/` はセッション終了で消える → `kaggle kernels output` で即回収する
- GPU 利用時: LightGBM は `device = "gpu"`、PyTorch 系は `device = "cuda"`
- Internet access が必要な場合は Notebook 設定で有効化する
- **Kaggle API push 時のパス構造（UIと異なる）:**
  - Dataset:        `/kaggle/input/datasets/{user}/{dataset-name}/`（UI: `/kaggle/input/{dataset-name}/`）
  - Competition:    `/kaggle/input/competitions/{competition}/`（UI: `/kaggle/input/{competition}/`）
  - `scripts/to_kaggle_nb.py` の setup セルが両パターンを自動検出するため手動設定不要

---

## Notebook提出コンペ向けフロー

Notebook が直接 `/kaggle/working/submission.csv` を生成する必要があるコンペ向け。

**変換（--submission-mode を追加）:**

```bash
uv run python -m scripts.to_kaggle_nb experiments/runs/exp001_s1_lgb_baseline.py \
  --competition <competition-slug> \
  --dataset-name ds-template-<competition> \
  --submission-mode \
  --gpu
```

`--submission-mode` を付けると、末尾に `SUBMISSIONS_DIR` の最新 CSV を
`/kaggle/working/submission.csv` にコピーするセルが自動追加される。

**Notebook 提出フロー:**

```bash
# 1. push して実行
kaggle kernels push -p kaggle_nb/

# 2. 実行完了を待つ（Notebook提出コンペは実行完了が提出）
kaggle kernels status {username}/exp001-s1-lgb-baseline

# 3. 提出結果を確認（提出は kaggle competitions submit 不要）
kaggle competitions submissions -c <competition-slug> | head -3
```

---

## データ読み込みパターン

`src/config.py` の設定後は、環境を意識せずにデータを読める:

```python
import pandas as pd
from src.config import RAW_DATA_DIR

# ローカル: data/raw/train.csv
# Kaggle:  /kaggle/input/<competition>/train.csv
train = pd.read_csv(RAW_DATA_DIR / "train.csv")
test  = pd.read_csv(RAW_DATA_DIR / "test.csv")
```

ファイルが見つからない場合のフォールバックも `raw_data_path()` が処理する:

```python
from src.config import raw_data_path
train = pd.read_csv(raw_data_path("train.csv"))
```

---

## 早期アーキテクチャサーベイの手順（Stage 1.5）

> **参照元**: CLAUDE.md「作業ステージとゲート — Stage 1.5」「`G-TWOAXIS` / `G-FAIR`」。
> Stage 1（最小ベースライン）完了直後に実施する。FE探索を始める前に「主軸アーキテクチャ」を決定する。

```
目的: 「このデータに最も合うアーキテクチャ」を最小コストで特定する
実施タイミング: Stage 1 完了後・Stage 2（EDA）開始前
```

**実施手順:**

0. **上位解法のアーキテクチャ調査（前提入力）**: Stage 1.5 に入る前に `/ds-kaggle-research` のフェーズ0を実施し、上位カーネルの主軸アーキテクチャ分布を把握する。自前の思い込みで候補を絞らず、**上位で頻出するアーキテクチャを候補に必ず含める**。
   > **教訓 (過去事例)**: 上位で主流だったアーキテクチャを序盤に調べず、自前 GBDT に固執。終盤にようやく乗り換えて大きく改善したが、探索効率を損ねた。序盤調査があれば主軸を早期に正しく選べた。

1. **候補アーキテクチャの選定**: 最低3種を評価する（例: LightGBM / CatBoost / RealMLP / TabNet）。**上記の上位解法調査で頻出したアーキテクチャを優先的に含める**
2. **共通評価条件（公正比較のための必須条件）**:
   - 同一の特徴量セット（Stage 1 と同じ最小特徴量）
   - 同一の CV 戦略（fold 数・シード）
   - **HP**: Stage 3（作業用HP調整）完了前は **文献推奨デフォルト** を使う。Stage 3 完了後に作業用HP（Optuna 20-30試行）で再比較する
3. **記録項目**: 各アーキテクチャについて `OOF` と `pub_oof_gap` を記録する

   | アーキテクチャ | OOF | pub_oof_gap | 処理時間 | 採否 |
   |---|---|---|---|---|
   | LightGBM | 0.XXXX | -0.000XX | X min | 主軸候補 |
   | RealMLP | 0.XXXX | -0.000XX | X min | 副軸候補 |
   | … | … | … | … | … |

4. **主軸の決定**: OOF が最高 かつ pub_oof_gap が最小 のアーキテクチャを主軸とする。両者が競合する場合は **OOF を優先**（`G-TWOAXIS`）
5. **副軸の保持**: 主軸と 10% 以内の OOF 差のアーキテクチャは「Stage 6 アンサンブル候補」として記録しておく

**公正比較の注意点（過去事例の教訓）:**

- ❌ 「最適化済みモデル A（多数実験分の HP + FE）vs デフォルト HP の新アーキテクチャ」は **不公正比較**
- ❌ 特徴量セットを変えての比較は NG（アーキテクチャ差と FE 差が混在する）
- ✅ 「Stage 1 特徴量 × 同一デフォルト HP × 同一 CV」でまず比較し、Stage 3 後に作業用 HP で再比較する
- ✅ FE が完成した後に **再評価**する（Stage 4 完了後に全候補アーキテクチャへ同一 FE を移植）

> **教訓 (過去事例)**: 特定のアーキテクチャを主軸のままコンペの大半を費やし、別アーキテクチャを試したのが終盤だった。
> 早期サーベイで優れたアーキテクチャを特定できていれば、探索効率が大幅に改善した。

---

## AV 診断（Adversarial Validation）

> **参照元**: CLAUDE.md「作業ステージとゲート — Stage 4」。
> Stage 4 で特徴量追加が一段落した時点、および Stage 6 移行前に必ず実施する。

```python
# 簡易版: train+test 結合データで is_test を予測
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

X_av = pd.concat([X_train, X_test], ignore_index=True)
y_av = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
# 3-fold CV で AV-AUC を測定
```

**判定基準:**

| AV-AUC | 判定 | 対応 |
|---|---|---|
| < 0.55 | ✅ シフトなし | そのまま継続 |
| 0.55-0.65 | 🔶 軽度シフト | importance weight 試行価値あり |
| 0.65-0.80 | ⚠️ 中度シフト | 上位重要度特徴量を drop 検討 |
| > 0.80 | ❌ 強いシフト | drop 必須 or データ前処理の見直し |

> **教訓 (過去事例)**: BASE_FEATURES では AV-AUC=0.5（無問題）だったが、後追いで拡張した特徴量で AV-AUC=1.0（count 系特徴量が train/test 構造差を leakage していた）。最終日まで気づかなかった

**CV由来のtarget encoding列を含める場合の注意（s6e7で実証、2026-07-30）:**

`TargetEncoder(cv=k).fit_transform(train)` は内部 k-fold cross-fitting により、同一の生値でも fold ごとに異なる TE 値を生成する（train 側は多値化）。一方 `transform(test)` は全 train で学習した単一エンコーダを使うため、生値 1 つにつき TE 値も 1 つ（test 側は単値）。この非対称性だけで AV-AUC が容易に 1.0 に到達し、**真の分布シフトと誤認しやすい**（TE 自体の fold-safety には無関係——モデル予測へのリークではなく、AV 診断という手法側の限界）。

→ **対策**: raw 特徴量と CV 由来の TE/OOF 系特徴量を分離し、それぞれ個別に AV-AUC を測定する。TE 込みで高 AV-AUC が出ても、raw のみで軽度シフト以下なら「TE のアーティファクト」と判断してよい。逆に raw のみでも高 AV-AUC が出る場合は本物のシフトを疑う。

---

## アンサンブル探索の手順（Stage 6）

> **参照元**: CLAUDE.md「作業ステージとゲート — Stage 6」。特徴量・HP飽和を確認してから読む。

```
STEP 1【相関確認】← 必ず最初に実施
  追加候補モデルのOOFと既存モデル群のOOFの相関を計算する:
  ```python
  from src.utils.ensemble import correlation_check
  corr, skip = correlation_check(oof_existing, oof_candidate)
  ```
  → 相関 < 0.998: STEP 2 へ
  → 相関 ≥ 0.998: スキップ。ただし【棄却分析A】を実施してから次へ

STEP 2【Simple Blend】
  既存モデル群との最適重みブレンドを試す（`optimize_weights()` で重み探索）
  → OOFが改善: STEP 3 へ
  → OOFが改善しない: 【棄却分析B】を実施してから次へ

STEP 3【Greedy Hill Climbing】
  保有する全OOFファイルを対象に系統的に探索する（`greedy_ensemble()`）
  → 改善するモデルの組み合わせを特定する
  → 全モデルで改善なし: 【棄却分析C】を実施

STEP 4【Stacking】
  LGB/CB以外に予測パターンが異なるモデルが2種以上ある場合のみ検討する
  → 相関が高いモデル同士のStackingは効果がない（前提の再確認）

STEP 5【Pseudo-labeling】
  アンサンブルの多様性が飽和した場合に有効な代替戦略。
  test の高信頼度サンプルに疑似ラベルを付与し、train に追加して再学習する:
  ```python
  test_proba = <最良モデルの test 予測>
  mask = test_proba.max(axis=1) >= threshold  # 閾値: 0.95 を最初に試す
  pseudo_df = test[mask].copy()               # 疑似ラベルサンプル
  # 各 fold の train に pseudo を追加して学習。OOF は元 train のみで評価
  ```
  探索順序:
    1. threshold=0.95 で OOF 改善を確認
    2. 改善あれば threshold=0.99/0.90 も比較（OOF-LB 乖離に注意）
    3. 改善あれば Iterative（前回の test 予測を次回の pseudo 源泉に）を2回まで試す
       → 3回目以降はラベルノイズ増幅で悪化するケースが多い
  注意: OOF-LB 乖離が拡大する閾値は LB 悪化リスクあり。乖離を記録すること

### Pseudo 源泉の品質とリーク診断
  - pseudo 源泉の優先順位: train fold 内の高確信度サンプル > 自前 test 予測 > 外部公開予測
  - **外部予測を pseudo source に使う = モデル蒸留**（genuine pseudo-labeling ではない）
    → test 予測が外部予測と 99%+ 一致し、独立シグナルを失う。Highクラスのみなど部分的でも同様
  - **リーク診断テスト**（pseudo 採用前に必ず実施）:
    ```
    同一構成で比較:
      pseudo あり  → (OOF_A, LB_A)
      pseudo なし  → (OOF_B, LB_B)
    OOF_A > OOF_B かつ LB_A < LB_B → leakage 確定。pseudo 源泉を見直す
    ```
    → 外部予測由来 pseudo は高確率でこのパターンに該当する

STEP 6【自前マルチシード EoS voting】
  外部公開予測が存在しなくても使える row-level voting 手法。
  同一アーキテクチャを複数シードで学習し、シード間で予測が割れる「disagree 行」を
  別系統モデル（Aux）で解決する。

  **⚠️ 前提条件（適用前に必ず確認する）:**

  この手法は **「マルチシードのベースモデルが、その時点で最強の単一候補である」** ことを暗黙に仮定している。
  agree 行（通常 99% 前後）ではベースモデルの投票をそのまま採用するため、
  **既により強いブレンドが存在する場合、その 99% の行で強い判断を弱い判断に置き換えることになり構造的に改善余地がない**。

  ```
  適用OK : ベースモデル(multi-seed) が最良候補。Aux は disagree 行だけの tie-breaker
  適用NG : 既存ブレンド > ベースモデル単体  → EoS は既存ブレンドを劣化させるだけ
  ```

  **判定**: `ベースモデル単体の OOF` と `現在の最良ブレンドの OOF` を比較し、
  前者が下回るなら STEP 6 はスキップする。

  > **教訓 (過去事例)**: FT-Transformer 5 シードでEoSを実施したが、シード一致率 99.00% に対し
  > FT 単体の最良 OOF (0.95070) が 6-way ブレンド (0.95092) を下回っていたため、全変種が改善せず
  > （disagree→Aux は −0.00003、単純多数決は −0.00026）。事前にこの比較をしていればスキップできた。

  **設計パターン:**
  1. ベースモデルを N シード × K fold で学習（推奨: 5シード × 5fold）
  2. シードごとの test 予測を保存: `test_{exp_id}_seed{s}_proba.npy`（後続分析で再学習不要にする）
  3. 全シードが一致する「agree 行」はそのまま採用
  4. シード間で割れる「disagree 行」を Aux モデルで上書き

  **disagree 行の分類（5シードの場合）:**
  - **4:1 スプリット**（N-1 シードが同方向）: 決定論的ルール特徴量などが N-1 シードを同方向に引く系統誤りケース。LB への貢献が大きい
  - **3:2 スプリット**（真に曖昧）: 補正効果は小さいが加法的に寄与する

  **Aux モデルの選定基準:**
  - **OOF 精度の最低閾値**: ベースモデルの OOF との差が **-0.002 以内**であること
    → それ以上低い Aux は disagree 行の大半を誤って変更し LB を悪化させる
  - **系統的独立性**: ベースの系統誤りを引き起こす特徴量を「含まない」Aux を選ぶ
  - **変化行数で品質を判断**: 変化行が少ないほど Aux の選択精度が高い（量より質）

  **Union Aux による漸進的改善（収穫逓減に注意）:**
  - Aux A（N 行変更）+ Aux B のユニーク行（M 行）→ LB は M 行分だけ追加改善
  - M が小さくなるにつれて改善はゼロに収束する。**ユニーク行が 15 行未満の Aux 追加に提出枠を使うのは非推奨**

  **保存規約（再学習を防ぐための必須設計）:**
  ```python
  # 学習時: シードごとに test 予測を保存
  np.save(OOF_DIR / f"test_{exp_id}_seed{seed}_proba.npy", test_preds)
  # 分析時: 保存済み予測を読み込んで EoS バリアント生成（再学習不要）
  seed_preds = [np.load(OOF_DIR / f"test_{exp_id}_seed{s}_proba.npy") for s in seeds]
  ```

STEP 7【外部公開予測の Row-level Voting】
  Kaggle Datasets や Discussion に他の参加者が公開した高LB予測ファイルが存在する場合、
  それらを「外部モデル」として row-level voting に活用できる。

  **確認手順:**
  ```bash
  # 公開データセット一覧（コンペ名で検索）
  kaggle datasets list --search "<competition-id>" --sort-by voteCount
  # ダウンロード
  kaggle datasets download <author>/<dataset-slug> -p /tmp/external/
  ```

  **Row-level voting の基本パターン:**
  1. 自前モデル群の bias 修正済み予測 vs 外部予測 を行単位で多数決
  2. 不一致行のみを別系統モデル（Aux）で解決（EoS voting パターン）
  3. 外部予測を pseudo source として自前モデルを再学習

  **重要な注意点:**
  - **精度↔独立性のジレンマ**（原則は `CLAUDE.md` の `G-SOURCE`）: 外部予測を pseudo source にすると自前モデルが外部予測の蒸留になる
    → OOF は改善するが test 予測が外部予測と 99%+ 一致し、独立シグナルを失う
    → pseudo source に外部予測を使う場合は `agree_rate = (pred_test == ext_pred).mean()` を必ず計測する
  - **差分役割モデルの選び方**: voting で「差分を解決する役割（Aux）」には「最低限の精度」と「系統的独立性」の両方が必要
    → OOF が高い Aux モデル同士を比較した場合、より高精度な Aux は主モデルと同じ予測に収束しがちで差分を解決できない（Aux 精度↑→LB↓ の逆相関が生じる）

  **外部知見系の安定ピーク検知（Public LB 過適合を防ぐ 3 シグナル）** — 判定基準は `CLAUDE.md` の `G-OVERFIT`:

  外部 Row-level Voting は自前モデルの OOF 上限を超えられるが、Public LB を最大化し続けると
  Private LB で過適合が起きやすい。以下のシグナルで「安定ピーク」を見極める:

  1. **クラスター収束**: 補正セットが異なる実験 3 本以上が ±0.00002 の Public スコアに収束したら
     その帯域が安定ピーク。クラスターの最もシンプルな 1 本を最終選択候補として即座に確保する。
     → 1 本だけ跳ねた実験は Public テストサンプルへの偶然ヒットの可能性がある（確認前に手放さない）

  2. **限界改善の縮小**: 補正行数が増えているのに Public の増分が縮んでいるとき、
     1 行あたりの改善が Public テストの統計的雑音水準に達したサイン。
     目安: 多数決クラス補正の理論的最小単位を下回る改善は Private には出ない可能性が高い。

  3. **補正ルールの複雑化**: 単純な多数決閾値から複合条件フィルタへ移行するほど
     Public 過適合リスクが増加する。複雑さが増した実験が Public を超えても hedge は外さない。

  **自前系 vs 外部知見系の並行管理**（→ `G-CEILING` の Final 2 選定と接続）:
  - 自前系（自前モデルの補正）と外部知見系（外部 voting）は Private ピークが一致しない
  - 外部知見系は Public を高く引き上げられる一方、安定ピークを過ぎると Private が落ちる
  - OOF 変化なしで Public だけ上昇が続く状況は「外部知見系の安定ピーク超過」の典型サイン
  - 最終 2 本の選択は「外部知見系 Public 最高 1 本 ＋ 外部知見系安定ピーク 1 本」を基本とする
    （自前系と外部知見系を並行している場合は「外部知見系安定ピーク ＋ 自前系ベスト」も有効）

STEP 8【Blend of Blends - 構造的に異なる blend の consensus】
  Stage 6 で複数の異なる philosophy の blend が同等 LB に到達した時、
  それらの平均がさらに僅かな改善を提供することがある。

  **適用条件:**
  - 2 つ以上の blend が同等 Public LB を持つ
  - 各 blend が構造的に異なる（例: greedy HC vs equal weight）
  - 各 blend が overfit リスクの異なる profile を持つ
  - ⚠️ **親 blend が「同じモデルプール上の異なる重み付け」であること**（下記の除外条件を参照）

  **除外条件（BoB が逆効果になるケース）:**

  親 blend が**モデル構成自体を変えている**場合（ある成分モデルの重みが 0）は BoB を適用しない。
  平均すると、含まれるモデルの重みが**個別最適化された点から大きく引き離される**ため。

  ```
  例: 親A = 6-way (TabM 重み 0.179)
      親B = 3-way (TabM を含まない = 重み 0)
      BoB = 0.5A + 0.5B → TabM の重みが 0.0895 へ移動
      → 重み探索で観測された「OOF が平坦な領域」(例: TabM 0.137-0.229) の外に出る
  ```

  **適用前チェック**: `PLAYBOOK.md#天井帯での意思決定ツールキット` の手順 2（重み bagging）で
  各成分の重みの観測レンジを測っておき、**BoB 後の重みがそのレンジ内に収まるか**を確認する。
  レンジを外れるなら BoB は行わない。

  > **教訓 (過去事例)**: 同等 LB の 6-way と 3-way を 50/50 で BoB したところ、全バリアントが
  > 単独の親より **−0.00006 〜 −0.00013**（OOF ノイズ床超）悪化した。row-level 多数決でも改善せず。
  > 原因は上記の「重みが平坦領域の外へ出る」ことであり、事前に重みレンジを確認していれば回避できた。

  **実装パターン:**
  ```python
  # 構造的に異なる 2 つの blend を 50/50 で平均
  final = 0.5 * blend_greedy_hc.test_pred + 0.5 * blend_equal_weight.test_pred
  ```

  **メカニズム:**
  - Greedy HC は OOF ノイズに重みを最適化 → OOF overfit bias
  - Equal weight は variance reduction だが weighting suboptimal → variance bias
  - 2 つの異なる bias の consensus 効果で個別エラーが部分的に補完

  **期待改善:**
  - Public LB +0.00000〜+0.00002 (微小、`G-NOISE` のノイズ床近辺)
  - 統計的にはノイズ範囲内のことが多い → 必ず submit して確認

  **Private LB での挙動（重要な注意）:**
  - Public LB +1σ 改善が Private LB に **反映されないことが多い**
  - BoB の Private LB ≈ 親 blend の平均 になる場合が多い (50% 線形結合のため)
  - **BoB を Public LB ベストとして Final 1 に採用するのは `G-CEILING` / `G-OVERFIT` 違反のリスク**

  **Final 2 候補にする際の注意:**
  - BoB は親 blend を 50% 含むため、Final 2 で親 blend を hedge にすると **共倒れリスク**
  - 推奨構成（優先順）:
    1. **Final 2 = (親 blend A, 親 blend B)**: 純粋多様性、最も安全（推奨デフォルト）
    2. **Final 2 = (BoB, 別 family blend)**: BoB を採用するなら family が完全に独立な blend と組む
    3. ⚠️ **避けるべき: Final 2 = (BoB, 親 blend)** → 共倒れリスク高、25%/75% 不均衡で多様性低

  > **教訓**: BoB が Public LB を +1σ 改善したが、Private LB は親 blend と同等（差なし）の事例あり。Public 微改善 = Private 改善とは限らない

**STEP 6 への重要追記 — Multi-seed averaging のデフォルト化:**

  実証的に、tree モデル (LGB/XGB/CB) の multi-seed avg5 は **+0.00010-0.00020 OOF** の安定的改善を提供する。
  Stage 6 移行前の base model 構築時、production blend に投入する model は **multi-seed=5 を default** とする。

  ```python
  SEEDS = [42, 0, 1, 7, 2026]  # default の 5 seeds
  for seed in SEEDS:
      train_with_seed(seed) → save oof_{exp}_s{seed}.npy, test_{exp}_s{seed}.npy
  avg5_oof = np.mean(all_seed_oofs, axis=0)
  ```

  - n_ens 内蔵モデル (RealMLP) は internal ensemble で代替可
  - TabM は GPU 必須で multi-seed コスト高 → single seed で OK
  - **CB は特に multi-seed 効果が高い**（過去事例で Δ=+2σ レベルの OOF 改善）
```

---

## アンサンブル棄却分析

> **参照元**: CLAUDE.md「Stage 6」。「効かなかった」で終わらせないための次アクション表。

| 棄却パターン | なぜ効かなかったか | 次に試せること |
|---|---|---|
| **A: 相関 ≥ 0.998（同一予測）** | 同じ特徴量・同じCV分割・同じアルゴリズムファミリーは予測が収束する | ①異なるCV戦略（fold数・seed変更）②異なる特徴量サブセット③全く異なるアルゴリズム（NN・RF・XGB deep）を試す |
| **B: OOFは高いが blend で改善なし** | 既存モデルと同じエラーパターンを持つ（誤差の方向が同じ） | OOFスコアだけでなく**誤差の相関**を確認する（高OOFでも誤差が相関していれば多様性なし） |
| **C: Greedy HC で全モデル改善なし** | 保有モデル群の多様性が飽和している | ①FEに戻り新しいシグナルを探す ②Pseudo-labeling（STEP 5）を試す ③問題の性質上アンサンブルの伸びしろが小さい可能性 |
| **D: Stacking が Simple Blend を下回る** | ベースモデルの予測が相関しすぎてメタ学習できない | ①ベースモデルの多様性を高めてから再挑戦 ②メタ特徴量に生の特徴量を追加 |
| **E: Pseudo-labeling で OOF↑ LB↓** | leakage 確定。pseudo 源泉（外部予測・train高確信度）に問題がある | リーク診断テストで源泉を特定し、より独立な源泉（train fold内）に変更する |
| **F: EoS Aux で LB 改善なし** | Aux の OOF が最低閾値（差 -0.002）を下回っているか、系統的独立性がない | Aux OOF を確認し閾値以上の別候補を探す。変化行数が多い=精度不足のサイン |
| **G: 外部予測 voting で安定しない** | 外部予測の安定ピークを超えて補正ルールが複雑化している | 3 シグナルで安定ピークを確認し、シンプルな補正ルールの実験に戻る |

> **棄却は終わりではなく、次の探索方向を示すシグナル。**
> 各 STEP で「なぜ効かなかったか」を1文で記録してから次に進む。

---

## 小規模アンサンブル（Final 2 の 2 本目 / hedge 候補）を育てる手順

> **参照元**: CLAUDE.md「提出枠の管理方針 — 最終選択の2本ルール」、`G-CEILING`。
> 大規模プール（数百メンバー）が飽和した後でも、**5〜7 メンバーの小規模ブレンドには伸びしろが残ることがある**（→ `L-22`）。
> 「候補を足す」より先に「結合方式を疑う」のが要点。

**STEP 1【結合方式の確認】← 最優先。候補探しより先に実施する**

現在の小規模ブレンドが **非負 simplex 制約**（`optimize_weights()` の重み探索）で組まれていないか確認する。

| 制約 | 使える候補 | 弱いメンバーの扱い |
|---|---|---|
| 非負 simplex（`w∈[0,1], Σw=1`） | 単体性能が既存最弱メンバー以上のものだけ | **構造的に重み 0 になる**（引き算に使えない） |
| signed 係数（L2 ロジスティック回帰） | 単体性能が低くても可 | 負係数で「他メンバーの誤りを打ち消す補正項」として機能する |

→ simplex 制約なら、**まず signed 方式（`nested_stack` 等）への切り替えだけを試す**。
メンバー構成を変えずに結合方式だけ変えた場合の差分を測ること（変更を1つに保つ）。

**STEP 2【simplex で「重み 0」と棄却済みの候補を再評価する】**

signed 方式に切り替えたら、STEP 1 以前に「重み 0 だから無価値」と判断した候補群を**必ず再評価する**。
simplex での棄却は「価値がない」ではなく「**その結合方式では使えなかった**」を意味していた可能性がある。

**STEP 3【候補の包摂基準をスイープする】**

「手動で数個選ぶ」「無選別に全部入れる」の両極端を試したうえで、**単体性能の下限フィルタ**（例: 単体 AUC ≥ 閾値）で選ぶ中間解も比較する。実測では以下の順に良かった:

1. 手動選択の少数（最も弱い）
2. 関連候補を無選別に全投入
3. **単体性能の下限で足切りしつつ、情報源を広く取る**（最良）

→ 「量」と「質」にはバランス点がある。片方だけを最大化しない。

**STEP 4【各段階で必ず LB 検証する】**

天井帯では OOF 改善が LB で消えることが多い（→ `L-21`）。段階ごとに提出して、
**OOF の改善が LB でも確認できた時点の構成**を採用候補として記録する。

> **STEP 1 を飛ばして STEP 3 から始めない。** 実測では、プール全体（582 候補）を相関で
> スキャンして新規メンバーを探す試みは全滅した一方、結合方式の変更だけで LB +0.00017 が得られた。
> 候補探索は結合方式を確認した後に行う。

---

## Stage 4 / Stage 5 のゲート詳細

CLAUDE.md の作業ステージ表は「何を満たせば次へ進めるか」を 1 行で示す。
その 1 行に収まらない運用の詳細をここに置く。

### Stage 4（段階的 FE）のゲート詳細

- **1 列ずつ投入する**: `scripts/feature_study.py` で ΔOOF と feature importance (gain) を計測する。
  複数列の同時投入は `--allow-batch --batch-reason` の明示が必要
- **一括投入はスクリーニングにすぎない**。採用・棄却の判断は、必ず **LOO 分解で各列の寄与を
  分離してから**行う（一括の合計 ΔOOF では、効いた列と足を引っ張った列が打ち消し合う）
- 合成データコンペでは**外部シグナル FE を先に**検証する（→ `#合成データコンペ向けガイダンス`）
- **AV 診断**で train/test 分布シフトの有無を確認する（→ `#av-診断adversarial-validation`）
- **FE 確定後、全候補アーキテクチャに同一 FE を移植して再評価する**。
  主軸で効いた FE が他アーキで効くとは限らず、逆もある（→ `G-FAIR`）

### Stage 5（本格 HP 最適化）のゲート詳細

- study は SQLite に永続化され、同じ tag なら試行を積み増せる
  （→ `CONVENTIONS.md#optuna-study-の永続化と命名`。やり直すときは tag を変える）
- 特徴量セットが確定した状態で Optuna 100 試行以上。ΔOOF の改善が指標別閾値以内
  （`G-NOISE`。AUC なら ±0.0002 目安）で収束していること
- **FE 変更時の HP retune ルール**: Stage 4 以降に FE が **±20% 以上変動**した場合、または
  domain-specific な新特徴量を追加した場合は HP retune を再実行する。
  FE 変更で HP の最適点は確実に動く（過去事例では retune で +1σ の OOF 改善を実証）
- **到達した単体モデルの最良値を state/SESSION.md の「単体ベスト」行に記録する**。
  アンサンブルの部品ではなく、それ自体が到達点——実務デプロイでは単体精度に固有の価値がある

---

## FE の採用・棄却判断（詳細）

> **参照元**: CLAUDE.md「作業ステージとゲート — Stage 4」。ΔOOF だけで判断しないための詳細手順。

**ΔOOF だけで行わない（importance との併用）:**

ΔOOF（greedy な逐次追加）は、既存特徴量と相関が高い列の貢献を過小評価する。
`feature_study.py` で新列を追加したとき、ΔOOF が小さくても以下の手順で二重確認する:

```
1. ΔOOF を確認する（目安: +0.0003 以上 = 明確な改善）
2. 追加後モデルの feature importance (gain) を確認する
   → 新列の importance が BASE 既存列の中位以上なら「情報は持っているが既存列と重複」
   → 新列の importance が BASE 最下位を大幅に下回るなら「真に情報なし」
```

判断マトリクス:
| ΔOOF | Importance | 判断 |
|---|---|---|
| ≥ +0.0003 | — | ✅ 採用 |
| < +0.0003 | BASE 中位以上 | 🔶 保留: 既存列と競合。どの列と重複しているか分析する |
| < +0.0003 | BASE 最下位未満 | ❌ 棄却: 真に情報なし |
| マイナス | — | ❌ 棄却: ノイズ追加 |

> **「ΔOOF < 0.0003 → 即棄却」は誤り。importance が中位以上なら既存列の代替候補として記録する。**

**FE の有効性はアーキテクチャに依存する（LGB 棄却 ≠ 全アーキテクチャで棄却）:**

あるアーキテクチャで ΔOOF < 閾値だった特徴量が、別アーキテクチャでは有効なケースがある。
これは特徴量の表現力（線形 vs 非線形）とアーキテクチャの相性による。

```
棄却の意味を正しく解釈する:
  × 「この特徴量は無効」      ← 誤り
  ○ 「主軸アーキテクチャ（LGB等）ではこの FE が効かなかった」
```

**Stage 4 棄却記録への追記義務:**

state/FE_HYPOTHESES.md の棄却エントリには「棄却したアーキテクチャ」を必ず明記する:
```
- 棄却: LGB で ΔOOF=+0.00010（閾値未満）
- 未評価: RealMLP, CatBoost（別アーキテクチャでの効果は不明）
- 再試行条件: Stage 1.5 で RealMLP が主軸になった場合は再評価する
```

**Stage 4 → Stage 6 移行時のアーキテクチャ間 FE 移植:**

FE 確定後、Stage 1.5 で「副軸候補」にリストされた全アーキテクチャへ同一 FE セットを移植して再評価する。
LGB で棄却された FE でも、副軸アーキテクチャ（例: RealMLP）に対しては効果が異なる場合がある。

> **教訓 (過去事例)**: 主軸アーキテクチャで棄却した複数の特徴量が副軸アーキテクチャでは有効だったが、
> 「主軸棄却 = 不採用」と判断して移植せずに提出してしまった。アーキテクチャ乗り換え時は FE の棄却リストを再検討する。

**FE 仮説の棄却記録には「再試行条件」を必ず書く:**

state/FE_HYPOTHESES.md の棄却エントリには以下を記録する:
```
- 棄却理由: なぜ効かなかったか（メカニズムレベルで）
- 再試行条件: どう変えれば効く可能性があるか（「不明」も可）
```
改良版を実装する前に、「棄却理由」が「再試行条件」で本当に解決されるかを確認してから着手する。
（例: 硬確率→棄却理由「0/1ノイズ」→再試行条件「ソフトな連続値に変換」→改良案「leaf_id + TargetEncoder」）

**同一観察への複数表現バリエーションの事前枝刈り:**

同一の観察（例: 「特定セグメントに誤分類が集中」）に対して複数の数学的表現
（percentile / z-score / interaction / flag 等）を順番に試す場合、
**1つ目の棄却時点で以下を実施してから2つ目に着手する**:

1. **「既存特徴量がどう同じ情報を表現しているか」を具体的に特定する**
   → 新列と既存列の相関・importance の重複度を計測し、「どの既存列が同じシグナルを持つか」を1文で書く
2. 特定できた場合: 表現を変えても同じ天井に当たる可能性が高い。
   **2つ目以降を実装する前に**「情報源が同じなので表現替えでは突破できない見込みです」とユーザーへ提示する
3. それでも試す場合は「表現によって情報の取り出され方が変わる根拠」（アーキテクチャの帰納バイアス差等）を
   言語化してから着手する

> **教訓 (過去事例)**: 同一観察に percentile → z-score → interaction の3表現を順に実装したが、
> 全て同じ棄却理由（既存特徴量で汲み尽くされている）に到達し、CV 実行3回分（各1時間超）を消費した。
> 1つ目の棄却時に情報源の重複を特定していれば、2つ目以降は実装前に予見できた。

---

## Final 2 候補プールとPersona 投票

> **参照元**: CLAUDE.md「提出枠の管理方針 — 最終選択の2本ルール」「`G-OOF` / `G-CEILING` / `G-OVERFIT`」。最終日に読む。

**Step 0: コンペ戦略軸の再確認（最初に実施）:**

`state/COMPETITION.md` の「コンペ戦略軸」（`/ds-kickoff` Q7 で記録）を再掲する。
スコア期待値と戦略軸が対立する場合（例: 外部知見系が Public 最高だが戦略軸は「自前モデルの限界追求」）は
「スコア軸の推奨」と「戦略軸に沿った推奨」を両論併記し、**ユーザーが決定する**。
AI がスコア期待値だけで推奨を一本化しない。

**候補プール構築（Persona 投票の前に必須実施 - `G-CEILING`）:**

Public LB ベースだけのスクリーニングは Public 過適合候補を優先しがち。以下の和集合をプールに含める:

- **Public LB Top-10**: 標準的な選定基準
- **OOF Top-10**: Private LB の predictor として尊重（`G-OOF`）
- **重複除去で 10-15 個**: Persona 投票の対象

各候補のプロファイルを以下のテーブルで整理:

| 候補 | OOF rank | Public LB rank | OOF-Public gap | 分類 | 注目度 |
|---|---|---|---|---|---|
| sub_A | #1 | #1 | 標準 (例: +0.0007 for AUC) | Public + OOF 両 Top | 標準候補 |
| sub_B | #2 | #25 | 大 (例: +0.0010) | **OOF only Top** | ⭐ Private で勝つ可能性 |
| sub_C | #25 | #2 | 大 (例: +0.0004) | **Public only Top** | ⚠️ Public 過適合可能性 |
| sub_D | (例) BoB | #3 (Public 最高 +0.00001) | 標準 | Public 微改善 | ⚠️ ノイズ床近辺、要 #17 適用 |

**注目度の判断:**
- ⭐ OOF only Top: Public sampling で過小評価された真の高品質候補。Final 2 候補として **必ず検討対象に**
- ⚠️ Public only Top: OOF 平凡なのに Public 高 → Public test sample への過適合疑い。**hedge を必ず付ける**
- ⚠️ Public +1σ 改善: `G-NOISE` のノイズ床。「突破」と呼ばず、Private 確認まで保留扱い

**Persona チェックリスト（拡張プールに対して実施、最終日に必ず実施）:**

以下の 9 ペルソナの視点で Final 2 を評価し、多数決で選定する:

| Persona | 主張 |
|---|---|
| **Kaggle Grandmaster** (経験派) | "Public LB +0.00001 はノイズ。100回中70回はノイズ。**親 blend を取れ**" |
| **Statistical Theorist** (理論派) | "Public LB AUC 差 ±0.00005 以内は統計的区別不能。**Variance minimization で構造的に異なる 2 つ**" |
| **Risk Management** (守り派) | "共倒れ防止が最優先。**独立な 2 blend** を取れ" |
| **Pragmatic Engineer** (実践派) | "実証された Public 最高を **捨てるな**" |
| **Newcomer** (素朴視点) | "Blend of Blends は親の 50% 平均。**親をそのまま使えばいい**" |
| **Domain Expert** | "ドメイン的に最適な model を必ず 1 本入れる" |
| **ML Researcher** | "Bias 差が最大の **異なる philosophy のペア**を取る" |
| **External Reviewer** | "Family が同じ 2 つは hedge にならない" |
| **Behavioral Economist** | "**Hindsight bias / Loss aversion** を排除、データに基づけ" |

**投票ルール:** 多数派の意見に従う。同数の場合は **Risk Management の意見を優先**（shakedown 回避を最優先）。

**典型的 Final 2 構成パターン:**

| パターン | 1 本目 | 2 本目 | 適用条件 |
|---|---|---|---|
| **A. 親ペア** (推奨デフォルト) | Greedy HC blend | Equal weight blend | 両者 Public LB 同等の時 |
| **B. Public 最高 + 安定ピーク** | Public LB best | 外部 voting 安定ピーク代表 | 外部 voting 系列の時 |
| **C. 自前 + 外部** | 自前 best | 外部 best（安定ピーク代表） | 外部 voting 有効と確認後 |
| **D. Blend of Blends + 別 family** | Blend of Blends (Public 最高) | 別 family blend | BoB 親に含まれない blend がある時 |

**重要な警告:**

- **Blend of Blends を Final 2 に入れる場合の罠**: BoB は親 blend を 50% 含むため、Final 2 で親 blend を hedge にすると **共倒れリスク** (例: BoB + 親 A は実効重み 75% 親 A / 25% 親 B で多様性低)
- **Public 最高への過度な執着**: Public LB の微差（例: +0.00001）は Public test のサンプリングノイズ範囲内（#17 の閾値表を参照）
- **OOF-LB gap が一定なら**: OOF 同等 = Private LB 期待値も同等。Public LB 微差は誤差

**確保のタイミング:**
- 安定ピーク確認と同時に「2 本目候補」をメモ
- 終盤に判断すると Public 最高への執着で見逃しやすい
- **コンペ前日までに Final 2 候補を 3-4 個に絞り、最終日は ペルソナ投票のみ実施**

> **教訓 (過去事例)**: 9-persona vote で多数決により「親ペア (greedy HC + equal weight)」(パターン A) を選定。BoB は親 blend を 50% 含むため hedge 不適と判断し見送り → 結果的に Public LB 1σ 改善を放棄したが Private LB shakedown を回避

---

## 強制 brainstorm の発動条件と手順

CLAUDE.md `G-PERSIST` の発動条件に対応する手順。

| 発動条件 | 手順 |
|---|---|
| FE 棄却 3 連続 | ① Discussion / 上位 Notebook を調査（`/ds-kaggle-research`）② 棄却時は「なぜ効かなかったか」に加え「**まだ試していない情報次元は何か**」を必ず 1 文で記録 |
| 同一 LB ±0.00002 で 5 回以上提出 | ① **未試行情報次元を 5 個以上列挙**（うち**最低 2 個は「現データに存在しない変数」でなければならない**。既存変数の組み合わせだけで 5 枠を埋めるのは禁止）② `data/external/` の `保留` ファイルを再評価（→ `G-SOURCE`）③ 上位 Notebook 調査 |
| 情報天井を判定した後（→ `G-INFOCEIL`） | 「やめる判断」ではなく**問いの立て直し**を AI が主導する（下記 4 手順） |

情報天井後は**問いを立て直す 4 手順**を能動的に提示する → `PLAYBOOK.md#情報天井後の問いの立て直し`

---

## 情報天井後の問いの立て直し

CLAUDE.md `G-PERSIST` が求める 4 手順の中身。

1. **Stage 0（Kickoff）の前提を再検証** — データの素性・評価指標の性質・CV 設計の初期判断は今も正しいか
2. **棄却済み仮説を新しい観点で再評価** — 当時と条件（特徴量・HP・アーキテクチャ）が変わっていないか（→ `G-FAIR`）
3. **外部調査をやり直す** — Discussion / 上位解法を再度当たる
4. **評価指標そのものの性質を測り直す** — ノイズ床・量子・セグメント別の誤差分布（→ `G-NOISE`）

---

## 天井帯での意思決定ツールキット

> **いつ読むか**: 上位候補間の差が縮まり「どれを選べばいいか分からない」状態になったとき。
> 天井帯では OOF も LB もノイズに埋もれるため、**測定できる差と測定できない差を切り分ける**ことが最優先になる。
> 対応する原則は CLAUDE.md `G-NOISE` / `G-CEILING`-28。
>
> | 手順 | 目的 | 対応指針 |
> |---|---|---|
> | 1 | ノイズ床と量子を計算し、目標・観察の妥当性を検証する | #23 |
> | 2 | OOF 曲面の平坦性を診断する | #24 |
> | 3 | paired bootstrap で「その差は有意か」を検定する | #17 |
> | 4 | Final 2 の 2 本目を E[max] で定量選定する | #19 |
> | 5 | 新モデル/新構造の固有の寄与を中間条件で分離する | #25 |
> | **6** | **単一最良の選定をやめ「集約」に切り替える** | **#27** |
> | **7** | **情報天井を判定し、探索の方向を切り替える** | **#28** |
>
> **順序の目安**: 天井帯に入った疑いがあれば **7 →（天井確定なら）6** が最優先。
> 個別の判断（この差は有意か / この候補を 2 本目にすべきか）で 1・3・4 を使う。

### 手順 1: ノイズ床と量子を計算する（`G-NOISE`）

**2 つの用途があり、どちらでも必ず実行する:**

| 用途 | いつ | 目的 |
|---|---|---|
| **(A) 目標の妥当性検証** | 新しい目標を掲げる前 | 到達不可能な量を追うことを防ぐ |
| **(B) LB 観察の解釈** | LB のスコア重複・密集・段差を根拠に何かを主張する前 | 観測された現象が単なる**格子効果**で説明できないかを排除する |

> **(B) を怠った失敗例**: Private LB 上位帯で同一スコアが大量に重複しているのを見て
> 「共有された公開ノートブック由来」と推論したが、**自チームの提出でも全く別構成の 7 件が同一スコアに重複**していた。
> 正しい説明は格子の粗さ（最小クラス約 11,941 行 → 1 行で 0.0000289 動く → 実質 0.00002〜0.00003 間隔の格子）。
> **量子計算を 1 回するだけで防げた誤推論だった。**

```python
import numpy as np, pandas as pd

train = pd.read_csv(RAW_DATA_DIR / "train.csv"); test = pd.read_csv(RAW_DATA_DIR / "test.csv")
props = train[TARGET_COL].value_counts(normalize=True)
n_public = int(len(test) * 0.30)          # Public の想定比率（コンペページで確認）
n_classes = len(props)

# (a) 量子: 1行の予測を変えたときの指標変化（マクロ平均系: balanced_accuracy / macro-F1 等）
for cls, p in props.items():
    quantum = (1 / (n_public * p)) / n_classes
    print(f"{cls:12s} Public内n≈{n_public*p:7.0f}  1行あたり指標変化={quantum:.7f}")

# (b) Public のサンプリングノイズ = OOF実測ノイズ床 × √(OOF行数 / Public行数)
oof_noise = 0.00008                        # ← 手順3の paired bootstrap で実測した値を入れる
print(f"Publicノイズ床 ≈ ±{oof_noise * np.sqrt(len(train)/n_public):.5f}")
```

**判断**: 目標差が「少数派クラスの量子1個分未満」または「Publicノイズ床の1/3未満」なら、その目標は却下し
「Private 期待値の最大化」へ切り替える。

### 手順 2: OOF 曲面の平坦性を診断する（重み bagging）

ブレンド重みの探索結果が「意味のある最適点」なのか「平坦な曲面から偶然拾った1点」なのかを判別する。

```python
# 12個の独立シードで重み探索し、重みのばらつきとOOFのばらつきを比較する
all_w, all_s = [], []
for sd in range(42, 54):
    w, s = search_w(oof_stack, seed=sd)    # Dirichlet + 局所refine
    all_w.append(w); all_s.append(s)
W = np.array(all_w)

for j, name in enumerate(names):           # 重みのばらつき
    print(f"{name:<10} mean={W[:,j].mean():.3f} std={W[:,j].std():.3f} "
          f"min={W[:,j].min():.3f} max={W[:,j].max():.3f}")
print(f"OOFのばらつき: std={np.std(all_s):.6f}")   # ← これが小さいほど曲面が平坦
```

**解釈**:
- 重みの std が大きいのに **OOF の std がノイズ床より小さい** → 曲面は平坦。OOF argmax による重み選択は
  実質ノイズの選択であり、**その候補の好成績には運が含まれる**と認識する
- この場合、**bagged 重み（12シードの平均＝平坦領域の中心）**が任意の頂点より汎化的に頑健と期待できる。
  OOF は僅かに下がるが（ノイズ床内）、単一探索への依存を外せる

### 手順 3: paired bootstrap で「その差は有意か」を検定する（`G-NOISE`の実測版）

固定閾値（±0.0003 等）ではなく、**このデータセット・この規模での実際のノイズ床**を測る。

同一行に対する2候補の予測を比較するので、行を毎回リサンプルする素朴な実装は重い。
**(クラス数) × (A正誤 × B正誤 = 4) セルの多項分布**から直接リサンプルすると厳密かつ高速（ペア構造も保たれる）。

```python
from scipy.stats import norm

def paired_bootstrap(pred_A, pred_B, y, n_class=3, n_boot=20000, seed=42):
    """戻り値: Δ = BA_B - BA_A の bootstrap 分布"""
    rng = np.random.default_rng(seed)
    cell = y * 4 + (pred_A == y).astype(int) * 2 + (pred_B == y).astype(int)
    counts = np.bincount(cell, minlength=n_class * 4).astype(float)
    draws = rng.multinomial(len(y), counts / counts.sum(), size=n_boot).reshape(n_boot, n_class, 4)
    n_c = draws.sum(2)
    recall_A = (draws[:, :, 2] + draws[:, :, 3]) / n_c
    recall_B = (draws[:, :, 1] + draws[:, :, 3]) / n_c
    return recall_B.mean(1) - recall_A.mean(1)

d = paired_bootstrap(pred_A, pred_B, y)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"Δ={d.mean():+.5f}  95%CI=[{lo:+.5f},{hi:+.5f}]  "
      f"{'有意' if (lo > 0 or hi < 0) else '非有意(ゼロを含む)'}")
```

CI 半幅がそのデータセットの**ノイズ床の実測値**になる（手順1(b)に代入する）。

### 手順 4: Final 2 の 2 本目を E[max] で定量選定する（`G-CEILING`）

1 本目 A を固定し、各候補 B について期待利得を計算して比較する。

```python
N_PRIVATE = int(len(test) * 0.70)
SCALE = np.sqrt(len(train) / N_PRIVATE)     # OOF規模 → Private規模へ換算

for name, pred_B in candidates.items():
    d = paired_bootstrap(pred_A, pred_B, y)
    mu, sd_priv = d.mean(), d.std() * SCALE
    z = mu / sd_priv
    gain = sd_priv * norm.pdf(z) + mu * norm.cdf(z)      # E[max(0, Δ)]
    print(f"{name:<28} μ_Δ={mu:+.5f} σ_Δ={sd_priv:.5f} "
          f"B勝率={norm.cdf(z)*100:5.1f}% E[利得]={gain:+.6f}")
```

**判断**:
- E[利得] 最大の候補を 2 本目にする
- ただし**最大値がノイズ床未満なら、2 本目の選定に時間をかけない**。
  理論上限は「性能が完全互角（μ_Δ=0）」を仮定した `σ_Δ × φ(0) = σ_Δ × 0.399` なので、
  これを先に計算すれば**ヘッジ候補を新規に育てる価値があるか**を着手前に判定できる
- **OOF では測れないリスク**（例: test 側だけ生成方法が異なる構成）がある場合は、
  そのリスクを共有しない候補を 2 本目にすると定量評価に加えて質的ヘッジになる

### 手順 5: 新モデル/新構造の「固有の寄与」を中間条件で分離する（`G-OVERFIT`）

「新モデル C を足したら OOF が上がった」を、そのまま C の価値と解釈してはいけない。
**C の追加は同時に探索の自由度も増やしている**ため、過学習分が混入する。

```
(1) 現行構成                       ← ベースライン
(2) 構造だけ変更（新モデルは入れない） ← 過学習分の基準
(3) (2) + 新モデル C                ← 本命

C 固有の寄与 = (3) − (2)      ※ (3) − (1) ではない
```

> **実例**: セグメント別重み + 専用モデルで OOF +0.00004 と見えたが、
> 中間条件 (2)（セグメント別重みのみ）が +0.00002 を占めており、
> 専用モデル固有の寄与は **+0.00001**（ノイズ床の 1/8）だった。

### 手順 6: 天井帯に入ったら「単一最良の選定」をやめ「集約」に切り替える（`G-CEILING`）

**適用条件**（いずれか）: argmax 一致率 99% 超 / 上位候補の差がノイズ床未満 / OOF 順位と LB 順位が一致しない。

```
(i)   天井到達を判定           → 手順7（argmax 一致率 + セグメント別分解）
(ii)  OOF で足切り             → 上位 20 件程度に絞る（順位付けには使わない。#18 但し書き）
(iii) 集約する                 → (a) 重み探索の bagging  または  (b) 上位 N 構成の確率平均
(iv)  集約結果を Final 候補とする
```

**(a) 重み探索の bagging（s6e7 で Private 最高を記録した方法）**

```python
# 同じモデルプールに対し、seed だけ変えて独立に重み探索を複数回実行し、重みを平均する
ws = []
for seed in range(12):
    w, s = search_weights(oof_members, seed=seed)   # Dirichlet + 局所refine
    ws.append(w)
W = np.mean(ws, axis=0)          # ← 平坦領域の「重心」に着地する
W /= W.sum()

blend_oof  = (np.stack(oof_members, 2)  * W[None, None, :]).sum(2)
blend_test = (np.stack(test_members, 2) * W[None, None, :]).sum(2)
```

**(b) 上位 N 構成の確率平均**

```python
top_n = ceil_band.nlargest(N, "oof")["name"].tolist()   # N は 5〜10
blend = np.mean([test_pred[n] for n in top_n], axis=0)
```

> **s6e7 実測（天井帯 107 件の事後検証）**
>
> | 選び方 | Private |
> |---|---|
> | OOF 最高を 1 本選ぶ | **0.95003**（天井帯でも下位） |
> | Public 最高を 1 本選ぶ | 0.95048 |
> | 上位 3 件平均 | 0.95034 |
> | 上位 5 件平均 | 0.95043 |
> | 上位 10 件平均 | 0.95045 |
> | **重み bagging（12 seed 平均）** | **0.95060（実測の最高）** |
>
> way 数と Private も単調増加（2way 平均 0.95013 → 6way 平均 0.95050）。
> **少数構成が1本だけ高スコアを出しても、それは「当たり」であって期待値ではない。**

---

### 手順 7: 情報天井を判定する（`G-INFOCEIL`）

「これ以上モデルを増やしても無駄か」を、印象ではなく数値で判定する。

```python
# (1) 全モデルの argmax 一致率マトリクス
preds = {name: p.argmax(1) for name, p in models.items()}
names = list(preds)
agree = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        agree.loc[a, b] = (preds[a] == preds[b]).mean()
print(agree.round(4))
# → 非対角が軒並み 0.99 超なら「決定境界が収束」＝情報天井のサイン

# (2) セグメント別に分解し、全モデルが同じ帯に収束していないか確認する
#     （セグメント例: 重要変数の欠損数、クラス、確信度帯 など）
for name, p in preds.items():
    row = [balanced_accuracy_score(y[seg], p[seg]) for seg in segments]
    print(name, np.round(row, 5))
# → 各セグメントでモデル間のスプレッドが極めて狭い（例: 幅 0.001 未満）なら収束確定
```

**判定後にやること:**

| やめること | 切り替え先 |
|---|---|
| 新アーキテクチャの追加 | **情報源**の追加（外部データ / 未使用の変数 / 新しい観測軸） |
| 同じ特徴量セット上での FE の言い換え | 特徴量セット**自体**の情報量を増やす方向 |
| OOF 最高を狙う探索 | **集約**（手順6） |

> **s6e7 実測**: 12 モデル（木3種 / attention / MLP2種 / foundation model / 検索ベース / OvA / multi-seed）が、
> ゲート完全行 **0.97169〜0.97227**、欠損行 **0.88603〜0.88970** という極めて狭い帯に収束していた。
> 唯一の例外だった k-NN（検索ベース）は、欠損3個のセグメントで **0.33791（≒ランダム基準）** まで落ち、
> 「異質ではあるが質の低い異質さ」だった。**異質性そのものは価値を意味しない**（`G-OVERFIT`）。

---

### 候補プールの構築と集約（CLAUDE.md `G-CEILING` の詳細）

**候補プールの作り方**（規範は CLAUDE.md `G-CEILING`。ここは様式のみ）— 和集合を取ったら
重複を除いて 10-15 個に収め、各候補に次を併記した表を作る:

| 列 | 内容 |
|---|---|
| OOF rank / Public rank | それぞれの順位 |
| OOF-Public gap | 符号つき。正 = Public が浮いている |
| プロファイル分類 | Public+OOF Top ／ **OOF Top のみ（注目）** ／ **Public Top のみ（過適合の可能性）** |

9 Persona 投票はこの拡張プールに対して実施する。

> **【実測で判明した限界】プールを広げても「当てる」ことはできない。**
> 和集合戦略で構築したプールでも **Private 上位10件を 2 件しか捕捉できなかった**。プールの広さは的中率をほとんど改善しない。**プールを広げることより、広げたプールを「集約」する方が期待値が高い。**

**集約の手段（いずれか。実測では (a) が Private 最高を記録）**
- **(a) 重み探索の bagging** — ブレンド重み探索を**複数 seed で独立に実行し、得られた重みベクトルを平均**する。単一の最適化ランは「平坦な領域の中でたまたま辿り着いた 1 点」に過ぎず、平均すればその領域の重心に着地する
- **(b) 上位 N 構成の確率平均** — OOF で足切りした上位 5〜10 構成の予測確率を単純平均する
- 決して「OOF 最高の 1 本」を選ばない（→ `G-OOF` の但し書き）

**2 本目の価値は `max(A,B)` 構造で定量化できる（Persona 投票の前に計算する）**
Kaggle の最終 2 本は Private で**良い方が採用される**ため、2 本目 B の価値は `E[max(A,B)] − E[A] = σ_Δ·φ(z) + μ_Δ·Φ(z)`（`z = μ_Δ/σ_Δ`, `Δ = B − A`）で求まる。`μ_Δ` は B の性能不足（OOF 差から推定）、`σ_Δ` は 2 候補のスコア差のばらつき（paired bootstrap で実測。脱相関が大きいほど大）。

- **「多様性が高い」だけでは価値にならない。** `μ_Δ` が負に大きい候補は勝率が数 % に落ち、枠を捨てるのと同じ
- 最適な 2 本目は「**性能が互角のまま最も脱相関した候補**」であり、その両立点は上式で数値比較できる
- **E[利得] を計算してノイズ床未満なら、2 本目の選定に時間をかけない**（「明らかに劣るものを避ける」以上の意味がない）
- → **手順とコードは `PLAYBOOK.md#天井帯での意思決定ツールキット`**

## 既知の落とし穴（ライブラリ別）

> **参照元**: CLAUDE.md `G-BLOCKER`。該当ライブラリを使う前に確認する。
> 「回避できたから解決」とせず、遭遇したブロッカーをここへ追記していくこと。

### Kaggle Notebook P100 GPU（PyTorch全般・ライブラリ非依存）

| 症状 | 詳細 | 回避策 | 状態 |
|---|---|---|---|
| **`CUDA error: no kernel image is available for execution on the device`** | KaggleプリインストールのPyTorch(2.x系)はsm_70以降のみをターゲットにビルドされており、P100(Pascal世代, sm_60)向けのCUDAカーネルを含まない。GPU検出(`nvidia-smi`)で"P100"を確認した直後、モデル初期化時にクラッシュする。**pytabkit(RealMLP, exp138)・pytorch-tabnet(TabNet, exp172)の2ライブラリで独立に再現**——PyTorchを使うライブラリ全般に共通する環境要因であり、個別ライブラリのバグではない | Notebook冒頭で`nvidia-smi`によりGPU名を検出し、"P100"を含む場合のみ`pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118`でPascal世代をサポートする旧ビルドに差し替える（`torchvision`/`torchaudio`は道連れでuninstallし依存衝突を避ける）。T4/P100以外のGPU(A100等)では発生しないため、GPU名の条件分岐が必須 | 回避策確立・恒久対応（次回PyTorch系ライブラリをKaggle GPUで使う際は必ずこの節を先に確認する） |
| **`AttributeError: type object 'GuardSource' has no attribute 'LOCAL_NN_MODULE'`（再インストール後もP100非互換のまま）** | 上記のcu118再インストール自体は成功しているのに、`Tesla P100... is not compatible with the current PyTorch installation`警告が再度出た直後にtorch._dynamo内部でcrashする。原因は**`import torch`をpipの再インストール処理より前に書いていたこと**——Pythonの`sys.modules`に古い(sm_70+専用)torchが一度キャッシュされると、再インストール後に`import torch`を再度書いても**モジュールキャッシュから返されるだけで再読込されない**(exp182のSAINT自作実装で初回遭遇) | **torch・torch.nn・torch.nn.functional等、torchに依存するimportは必ず`nvidia-smi`検出→P100ならcu118再インストール、の一連の処理より後に書く**(TabNet/exp172は`pytorch_tabnet`のimportを再インストール後に置いていたため無事だったが、自前でtorchを直接importする実装では見落としやすい)。関数内で遅延importする方法もあるが、モジュール冒頭のimport順序を守るのが最もシンプル | 回避策確立・恒久対応 |

### pandas 3.0+（`astype(str)`のNaN保持、ライブラリ非依存・全般）

| 症状 | 詳細 | 回避策 | 状態 |
|---|---|---|---|
| **`series.astype(str)`が欠損値をNaNのまま保持し、`groupby`/`.map()`で欠損行が暗黙に脱落する** | pandas<3.0では`astype(str)`が欠損を文字列`"nan"`に変換し独自レベルとして扱っていたが、pandas>=3.0の新しいstr dtypeでは欠損値がそのままNaNとして残る。`groupby(dropna=True)`がデフォルトのため、その列でtarget/frequency encodingのマップを計算すると欠損行が集計から除外され、`apply`側の`.map()`もマッチせず`.fillna(全体平均)`にフォールバックする——**「欠損であること」自体が持つ情報がエンコーディングから静かに失われる**（外部カーネル発見、s6e8 exp252で自パイプラインでも実地確認: pandas==3.0.5でastype(str)後もNaN数が変化しないことを確認） | `.astype(str)`ではなく`.astype(object).fillna("__missing__").astype(str)`を使う（値そのものは変えず、NaN処理だけ修正）。念のため`groupby(...).size().sum() == len(df)`のようなassertで欠損行の脱落がないか検証する | 実地確認・回避策確立（本コンペでの実害はH-025参照、MCARに近い欠損だったためΔOOF+0.00001とごく小さかったが、**MNAR/MARの欠損があるデータセットでは重大な影響になりうる**ため次回コンペでは着手時に確認する） |

### LightGBM + OpenMP（macOS、プロセス強制終了後の再実行）

| 症状 | 詳細 | 回避策 | 状態 |
|---|---|---|---|
| **LightGBMを使う処理がPythonトレースバックなしにSIGSEGVで静かに落ちる** | 同一セッション内で他のPythonプロセスを`kill -9`で強制終了した後にLightGBMを呼ぶと、libompのスレッドプール生成(`__kmp_fork_call`/`__kmp_create_worker`)内でセグメンテーション違反が起きることがある。**Python例外ではなくネイティブクラッシュのため、ログには何も残らず「プロセスが理由もなく消える」形でしか観測できない**。原因特定にはmacOSのクラッシュレポート(`~/Library/Logs/DiagnosticReports/*.ips`、`exception.type: EXC_BAD_ACCESS`とフレーム内の`__kmp_`系シンボル)の確認が必須だった——症状だけではMPS/PyTorch起因と誤診しやすい(実際に1時間以上を浪費した) | 環境変数`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`を付けて実行する。ただしこの設定は他のOpenMP依存処理のスループットを落とすため、常用ではなくクラッシュ発生時の回避策として使う | 原因未特定(なぜkill -9後にlibompの状態が壊れるのかは不明)・回避策確立 |

> **診断の型**: 「Pythonプロセスが無言で消える」ときは、まず`~/Library/Logs/DiagnosticReports/`の
> 最新`.ips`を確認する。Pythonレベルのログだけを追っても原因に辿り着けない。

### pytabkit（RealMLP / TabM / RealTabR / Resnet_RTDL）

| 症状 | 詳細 | 回避策 | 状態 |
|---|---|---|---|
| **単一 train/val 分割での学習が系統的に不安定** | `StratifiedKFold.split()` ではなく `train_test_split()` 由来の単一分割（`X_val`/`y_val` を明示的に `.fit()` へ渡す形）を使うと、内部 val BA が **0.887〜0.889 に系統的に収束**し、5-fold 本番 OOF の 0.94-0.95 台から大きく乖離する。分割比率（5%/15%/20%）・`random_state` を変えても再現（4-5 回試行）。予測クラス分布が多数派へ偏る一貫したパターン | **必ず `StratifiedKFold.split()` 由来の分割で学習する**。全データ再学習など単一分割を要する処理は避け、5-fold CV 構造を保つ | 原因未特定 |
| **RealTabR がカテゴリカル列でクラッシュ** | 内部の `OrdinalEncoder` に 0 列が渡されて落ちる。最小合成データでも再現し、データ固有の問題ではない | 同系統の `Resnet_RTDL` で代替 | ライブラリ側バグ |

> 自前 PyTorch 実装（FT-Transformer）は同じ単一分割方式でも正常動作したため、
> **カスタム実装と pytabkit の実装差**が分水嶺と考えられるが、根本原因（内部の class-weight 計算や
> calibration_method のデフォルト挙動等）は未特定。

### XGBoost

| 症状 | 詳細 | 回避策 | 状態 |
|---|---|---|---|
| **特定構成で全データ再学習が失敗** | 生値 13 列 + TE 39 列の構成で全データ再学習すると内部 val BA=0.89-0.91（期待 0.95036）。早期停止の有無を問わず再現。データの健全性（NaN 数・値域）は元データと一致を確認済みで、データ破損ではない | 当該モデルを CV 版のまま使う | 原因未特定 |

### NN 全般（全データ再学習時）

| 症状 | 詳細 | 対策 |
|---|---|---|
| **内部ホールドアウトでの早期停止が不安定** | 全データ再学習で検証用に 5% だけ切り出すと、fold 検証（20%）より判定が不安定になり、単体 LB が **−0.00045** 劣化した実例あり | 全データ再学習時も**CV で実績のある検証比率を維持する**。「データを増やせば無条件に改善」は誤り |

### 記録の運用

- 新しいブロッカーに遭遇したら、**最小再現コードを `experiments/blockers/` に保存**し、この表に 1 行追記する
- 「状態」列は `原因未特定` / `ライブラリ側バグ` / `解決済み(PR/versionリンク)` を記入する

---

## Private 過適合候補と外部知見系の判定

CLAUDE.md `G-OVERFIT` の判定条件（L0 には結論だけを置く）。



---

## 教訓アーカイブ（実測値つき）

> **この章が教訓の唯一の本文（SSoT）。** `CLAUDE.md` の各指針からは `L-NN` の 1 行で参照される。
> コンペ識別子（`s6e7` 等）を書いてよいのは**この章だけ**。
>
> **なぜ数値を残すのか**: 数値は教訓の payload であり、規範を「守るべきもの」に変える。
> リファクタで最も静かに失われるため、`scripts/harness/doc_audit.py` の C4 が消失を機械検知する。

### L-01 CV 内部診断が最大の跳躍を生んだ（決定的）

*対応する指針: `G-DIAG`*

**教訓 (過去事例、決定的)**: 初回ベースラインで train=0.46598 / val=0.44485 / LB=0.43832。
OOF↔LB の 2 軸だけを見れば「LB 乖離 0.0065」で終わっていたが、ユーザーが
**「train/val 乖離 0.0211 も大きい」**と指摘したことで、CV 設計の問題でも単純な過学習でもなく
**「argmax 決定を伴う多クラス分類での校正不足」**という第三の仮説にたどり着き、
`class_weight="balanced"` の導入で **OOF 0.44485 → 0.81189 / LB 0.43832 → 0.81185** という
コンペ最大の跳躍につながった。**OOF↔LB の 2 軸だけでは、この打ち手には到達できなかった。**
**教訓 (s6e7、②の実例)**: 特徴量セット候補 3 つを比較した際、候補間の OOF 差は 0.0004〜0.0009 だったが
**fold 間の val std は 0.0012〜0.0013** と差より大きく、**単一 fold では順位が入れ替わりうる**と判明した。
std を見ずに OOF 差だけで採否を決めていれば、ノイズを実力と誤認していた。
**運用上の注意**: s6e7 では log.csv に列があるにもかかわらず、`cv_train_mean` の記入率は **28%**、
`cv_val_std` は **21%** にとどまった（使い捨てスクリプトが `ExperimentTracker` を経由しなかったため）。
**記録されない診断は存在しないのと同じ**。実験スクリプトは必ず `ExperimentTracker` 経由で記録すること。

### L-02 中核 FE を不公正比較で誤棄却しかけた（最重要）

*対応する指針: `G-FAIR`*

**教訓 (s6e7、最重要)**: **最終的に全モデルの土台となった中核 FE（13カラム全ての厳密値 target encoding、
52特徴量）を、AI は当初「棄却」していた。** 原因は 13 特徴量用の HP を 52 特徴量にそのまま流用した
不公正比較で、LB がベースを −0.00009 下回ったこと。ユーザーの「**判断が安直、せめて HP 調整してみれば**」
という指摘で HP を再調整したところ **OOF +0.00027 / LB +0.00018 で採用に転じた**。
この押し戻しが無ければ、解の水準は 1 段低いまま終わっていた。

**構造的な問題**: AI は「検証を尽くした」と判断する閾値が低く、**自分が出した結論を疑う方向の検証を
自発的に起動しにくい**。棄却を記録する瞬間が、この指針を最も強く適用すべきタイミングである。

### L-03 天井帯 107 件の事後検証 — 集約が期待値を上げる

*対応する指針: `G-CEILING`*

**教訓 (s6e7 実測)**: 天井帯 107 件を事後検証したところ、
- **OOF 最高を選ぶ → Private 0.95003**（天井帯でも下位）／ Public 最高を選ぶ → 0.95048 ／ 理論上の最良 → 0.95060
- **上位 N 件の平均を取るほど期待値が上がる**: N=1 → 0.95003、N=3 → 0.95034、N=5 → 0.95043、N=10 → 0.95045
- **Private 最高（0.95060）を記録したのは「12 シードの重み探索を平均した」bagging 構成**だった
- モデル数（way 数）と Private も単調増加（2way 平均 0.95013 → 6way 平均 0.95050）。**少数構成が1本だけ高スコアを出しても、それは当たりであって期待値ではない**

**集約は「保険」ではなく「期待値の最大化」である。** 選択の集中こそがリスクだった。

### L-04 12 モデルが同一の帯に収束（情報天井）

*対応する指針: `G-INFOCEIL`*

**教訓 (s6e7)**: 87 の FE 仮説の大半が「同じ 13 変数の表現を変える試み」（bin化 / エンコーディング / 交互作用 / TE）で、
10 アーキテクチャ（木3種 / attention / MLP2種 / foundation model / 検索ベース / OvA / multi-seed）も全て同一の特徴量セット・
同一の train.csv に対する学習だった。実測では **12 モデルがゲート完全行 0.97169〜0.97227、欠損行 0.88603〜0.88970 という
極めて狭い帯に収束**していた。argmax 一致率 99% 超が判明した時点（コンペ中盤）で方針転換すべきだったが、
その後も 4 アーキテクチャを追加し、いずれも寄与ゼロで終わった。

### L-05 到達不可能な目標と、スコア格子の誤推論

*対応する指針: `G-NOISE`*

**教訓 (過去事例)**: LB 0.95098 から「0.95100 台」を目標に据えたが、差 +0.00002 は
多数派クラスで 4.6 行分・少数派クラスでは **1 行未満**、Public ノイズ床 ±0.00022 の **1/11** だった。
計算した時点で「工学的な目標として成立しない」と判明し、残り時間の使い道を根本的に見直せた。
**教訓 (s6e7、LB 解釈での失敗)**: Private LB 上位帯で同一スコアが大量に重複していること
（0.95084 に6チーム、0.95081 に11チーム）を観測し、**「同一スコアの重複＝同一予測＝共有された公開ノートブック由来」と推論した。これは誤りだった。**
反証は自分たちのデータにあった——**自チームの 169 提出でも、2way/3way/5way/6way という全く別構成の 7 件が
Private=0.95047 に重複**していた。正しい説明は**スコア格子の粗さ**で、最小クラス（約 11,941 行）では
1 行の予測変化で 0.0000289 動くため、スコアは実質 0.00002〜0.00003 間隔の格子上にしか存在しない。
幅 0.00004 の帯に数十チームいれば重複は**必然**だった。

**本質**: この指針の量子計算を **1 回適用するだけで防げた誤推論**であり、**自分の道具を自分の分析に
向けていなかった**ことが問題。しかもこの誤推論は「上位は 1 ノートブックの共有にすぎない」という
**自分たちに有利な方向のバイアス**を含んでいた。**分析対象が数値実験ではなく観察・解釈になった瞬間に
検証基準が下がる**のは再発しやすいパターンとして警戒する。

### L-06 可視化の形骸化 — 努力目標→条件明示→機械的強制の3世代

*対応する指針: `G-MECH`*

**失敗の履歴（3 度目を防ぐために本文に残す）**
- **第1世代（努力目標）**: 「積極的に提案する」と書いた → 実験サイクルが高速化するほど省略され、100 実験超のコンペで**可視化が最初の3日間しか実施されなかった**
- **第2世代（発動条件の明示）**: 上記を受けて「必須発動条件」3 つを明文化した → **それでも s6e7 で同じ形で再発**。総プロット 31 枚のうち **23 枚（74%）が最初の 4 日間**に集中し、最後の 9 日間は 1 枚のみ。LB ベストを更新した時期（条件①該当）ですら可視化は 1 枚だった
- **第3世代（機械的強制・検知のみ）**: 「AI に守らせたい規律は、AI に守らせようとしてはいけない」。実行環境側（tracker / hook）で検知する方式に移行した
- **第4世代（機械的強制・実行ブロック）**: 第3世代でも**再発した**。検知して警告は出ていたのに、締切直前の時間的プレッシャー下で LB ベスト更新のたびに警告が出続けても一度も対応されなかった——「検知はできるが、対応するかは結局 AI の自発性に依存していた」。第4世代では `ExperimentTracker.start_run()` が**RuntimeError を送出して次の実験の開始自体を止める**。省略するには `skip_viz_check=True` または `DS_SKIP_VIZ_CHECK=1` の明示が必要（＝省略が意識的な選択として記録に残る）

**同じ観点で他の規律も点検すること** — 「1 実験 1 コミット」「1 列ずつの FE 投入」なども、AI の自己監査に依存していないかを確認し、依存しているなら機械的検知に置き換える。

### L-07 「使う」判定した外部データを締切前日まで放置

*対応する指針: `G-SOURCE`*

**教訓 (s6e7)**: `enhanced` 版の外部データは初日のインベントリで**「使う」判定**だったにもかかわらず、
検証したのは**締切前日**だった。原因は、別の外部データでの否定的結果（ゲート変数の回復不能性）を
**「外部データ探索そのものの終了」と過度に一般化**したこと。**実際の検証コストは数分**だった。
なお検証の結果、`enhanced` 版は**コンペとは別の生成プロセスのデータ**と判明した——
同一の `(student_id, timestamp)` を持つ共通 496 行で**ラベル一致率わずか 35%**。
ファイル名も変数構成も酷似していたが別物であり、**行レベル照合を最初にやれば数分で判定できた**。

### L-08 探索空間の拡大が OOF を過学習させ gap を反転させた

*対応する指針: `G-OVERFIT`*

**教訓 (過去事例)**: ブレンド重みを全体共通 6 個からセグメント別 13 個に増やしたところ、
**OOF は +0.00004 改善したのに LB は −0.00017 悪化**し、gap が +0.00006 → **−0.00015 と反転**した。
同様に 63 通り規模の網羅探索から選んだ高 OOF ブレンド群は、揃って gap が負だった。
「OOF が上がったから良い変更」は、探索空間を広げた場合には成立しない。
**教訓 (過去事例、s6e7)**: 同一データ上でセグメント別重み(4セグメント×12モデル=48パラメータ)を
最適化・評価すると OOF が +0.00013 改善したように見えたが、**tune-half で重みを決定し audit-half
（重み決定に一切使っていない独立データ）で評価すると −0.00021 の悪化に反転**した。決め手は
最小セグメント(n=248)で最も性能が低いモデルに最大の重みが割り当てられていたこと——
少数サンプルでは「たまたま良く見える組み合わせ」がいくらでも見つかり、本物のシグナルと
区別がつかない。**同一データでの最適化・評価だけで判断せず、必ず独立データ(tune/audit分割)
で検証してから採否を決める**。

### L-09 Final 2 の2本目の価値を E[max] で事前評価

*対応する指針: `G-CEILING`*

**教訓 (過去事例)**: 「構造的ヘッジを作るべきか」を議論する前に上式を計算したところ、最良の 2 本目でも
E[利得] は **+0.000021**（OOF ノイズ床の 1/4）、性能が完全互角の理想候補を仮定しても **+0.000052** で
やはりノイズ床未満だった。**ヘッジ候補を新規に育てる価値がない**ことが事前に判明し、残り時間を無駄にせずに済んだ。
併せて、最も異質な候補（不一致率が他の 2 倍）は性能不足のため**勝率 0.2%** と算出され、直感的な
「多様性は正義」が誤りであることも数値で確認できた。

### L-10 OOF 棄却で提出せず終わった実験が半数

*対応する指針: `G-CALIB-SUB`*

**教訓 (s6e7)**: 実験 339 件に対し提出 169 件で、**約半数が OOF 判断のみで終了**した。
一方、OOF の論理では「提出する価値なし」だった候補を、あえて提出した実験（exp320c）は
**OOF +0.00004 なのに LB −0.00017、gap が正から負へ反転**という結果を出し、
**`G-OVERFIT`（探索空間拡大の代償）の決定的証拠**になった。**この 1 提出が生んだ知見は、
OOF で棄却した数十実験の合計より価値があった**。日次上限 10 回に対し平均 6 回しか使っておらず、枠は余っていた。

### L-11 9-persona 投票と BoB の見送り

*対応する指針: `G-CEILING`*

**教訓 (過去事例)**: 9-persona vote で「親ペア (greedy HC + equal weight)」を選定。BoB は親 blend を 50% 含むため hedge 不適と判断し見送り → Public LB 1σ 改善を放棄したが Private LB shakedown を回避

### L-12 技術的ブロッカー4件を原因未特定のまま回避

*対応する指針: `G-BLOCKER`*

**教訓 (s6e7)**: 4 件のブロッカー（pytabkit の全データ再学習が内部 val 0.887 に収束 / RealTabR の
OrdinalEncoder クラッシュ / XGB rawfix の全データ再学習失敗 / FT-Transformer の内部ホールドアウト不安定）
を、いずれも **「原因未特定のまま回避」で処理**した。約 100 分の調査を費やしたケースもあるが、
知識としては何も残っていない。→ 既知の落とし穴は `PLAYBOOK.md#既知の落とし穴ライブラリ別` に転記した。

### L-14 情報天井の実証後も探索を続けて得た非スコア資産

*対応する指針: `G-PERSIST`*

**教訓 (s6e7)**: 情報天井を実証した後もユーザーの方針で探索を継続した結果、スコアは動かなかったが
**中核 FE の復活・`G-OVERFIT` の決定的証拠・外部データの決着・分析ミスの発見**といった、
スコアには表れない資産が確実に増えた。月次コンペのサイクルではこれらは翌月に複利で効く。
**反省すべきは「探索を続けたこと」ではなく、AI が「スタートに戻る」提案を自分から出せなかったこと。**

### L-15 brainstorm が既存変数の組み合わせに偏った

*対応する指針: `G-SOURCE`*

**教訓 (s6e7)**: brainstorm は実行されたが、列挙された「未試行の情報次元」が毎回**既存 13 変数の
別の組み合わせ**に偏り、外部データ由来の新情報源がリスト上位に来なかった。結果、87 仮説の大半が
「同じ情報源の表現を変える試み」に費やされた。**情報が増えない探索を何回繰り返しても情報は増えない**（→ #28）。

### L-16 新アーキテクチャを不公正条件で評価し誤判断

*対応する指針: `G-FAIR`*

**教訓 (過去事例)**: 新アーキテクチャを「FE 削減版 + デフォルト HP」で評価し「既存より劣る」と誤判断。
後に公正条件（同一 FE + 作業用 HP）で評価したところ、新アーキテクチャが主軸として有効と判明した。

### L-17 単一分割で見えた正の兆候がフル CV で消えた

*対応する指針: `G-FULLCV`*

**教訓 (過去事例)**: 単一 80/20 分割で「ブレンド寄与 +0.00006（初のプラス）」と見えた施策が、
同一手法のフル 5-fold OOF では **−0.00001** となり、ノイズだったと判明した。
`G-NOISE` の「Public LB のノイズ床」と同型の現象が、**OOF 内部の分割方法の違いでも起きる**。

### L-18 サブサンプル単一 fold で選んだ HP がフルスケールで発散した

*対応する指針: `G-FULLCV`*

大型 NN（TabM）の作業用 HP を **サブサンプル 15 万行 × 単一 fold** の Optuna で調整し（exp272）、
最良と判定された `n_blocks=4 / d_block=512` を採用した。ところがフル 5-fold へ移すと **1 fold が発散**
（val **0.633**）し、OOF は **0.896** とデフォルト HP の **0.95050** を大きく下回った（exp273）。

**メカニズム**: 単一分割での HP 選定には 2 つの独立したリスクがある。
①その分割への過学習（`G-FULLCV` の通常ケース）、②**スケールを戻したときの学習発散**。
②は「効果が消える」のではなく「壊れる」ため、平均値だけを見ていると原因を見誤る。
サブサンプルでは表面化しなかった大容量モデルの不安定性が、フルデータで顕在化した。

**恒久対応**: 不安定アーキの HP 探索は複数 fold 平均で評価し、`gradient_clipping` / lr 上限を
探索空間に制約として与え、発散 fold は大きくペナルティする。重いモデルは軽量アーキ（RealMLP 等）で
当たりを付けてから移植する。

### L-19 個別Δ≈0が13系統累積すると統計的に確定的な正の差になった

*対応する指針: `G-NOISE`, `G-CEILING`, `G-INFOCEIL`*

**教訓 (s6e8実測)**: 生成モデル3角度・10-fold木系・boosting variants・pseudo-labeling・
Residual-MLP・FLAML・AutoGluon等13系統を、単一seed比較のノイズ床(±0.00004)未満として
個々に「PALLへの寄与ゼロ」と判定してきた（G-INFOCEIL判定に至った根拠でもある）。
ところが127メンバー(旧プール)と348メンバー(全13系統込み)を**それぞれ12シードでbagging**
してから直接paired bootstrap比較すると、`mu_delta=+0.000051, sigma_delta=0.000010, z=+4.90,
95%CI=[+0.000031,+0.000072], P(delta>0)=1.0000` と**統計的に確定的な正の差**が出た。

**メカニズム**: 個別の棄却判定に使ったノイズ床(±0.00004)は**単一seed比較**用に較正された値。
baggingで seed 分散が `√12≈3.5倍` 縮むため、bagged構成同士の比較ではσ_deltaが0.00001まで
縮小し、単一seed比較では検出不能だった真の効果(各系統+0.00000〜+0.00004程度)が可視化される。
LOO分解の結果、**寄与の大部分はweakLGB(+0.00004)に集中**しており、10-fold木系(+0.00001)・
boosting variants(+0.00001)が僅かに追随、残り(生成モデル・NN多様化・AutoML系)はbagged比較でも
ほぼゼロと推定される（個別Δの単純合計+0.00005がbagged実測+0.00005と正確に一致）。
⚠️ **訂正**: 当初pseudo-labelingも寄与に含めていたが、`DERIVED_RE`(`exp160_stack_ablation.py`)が
"pseudo"を含むファイル名を自動発見から除外する設計のため、pseudo-labelingの生予測は
このbagged比較対象のプールに実際には1本も含まれていなかった（exp215の一回限りの手動追加
テストで観測した+0.00002は、その場限りの診断でありプールには反映されていない）。個別Δの
単純合計を検算に使う際は、**プール構成の実装(DERIVED_REの除外リスト)を必ず確認**すること。

**恒久対応**: 「情報天井」判定（G-INFOCEIL）は argmax一致率・セグメント別AUCという**別の診断軸**
であり無効化されない。ただし「新規メンバーの個別採否」を単一seed比較のノイズ床だけで機械的に
棄却すると、**符号が一貫して正の微小な真の効果を取りこぼす**。较正された正の符号を持つ系統
（本コンペではweakLGB系）は、個別Δがノイズ床未満でも**プールに残し量を増やす価値がある**。
「情報天井後は新アーキ追加禁止」を、符号確認済みの既存系統のスケールアップにまで拡大解釈しない。

> **追試(exp227-228, s6e8)**: 上記を受けてweakLGBを100本→313本にさらに拡大したが、
> bagged同士(exp216 vs 今回)のpaired bootstrapは `mu_delta=-0.000008, z=-1.15,
> 95%CI=[-0.000022,+0.000006]` で0を跨ぎ、有意差なしだった。**weakLGBの正の寄与は
> 35→100本の間で既に飽和しており、100→313本の追加分は新規情報を持たなかった。**
> 「符号が正の系統は増やせば増やすほど良い」という単純な外挿は誤りで、**各系統には
> 固有の飽和点がある**。「累積効果は本物」（本文）と「無限にスケールする」は別の主張であり、
> スケールアップ提案のたびに毎回bagged paired bootstrapで再検証すること。
>
> **追試2(exp229-230, s6e8) — DERIVED_RE除外の見落としが本物の改善を隠していた**:
> pseudo-labelingも同様にスケールアップ(1本→5本、LGB/XGB/CB×2seedで多様化)して
> 追試したところ、`mu_delta=+0.000041, z=+4.21, 95%CI=[+0.000022,+0.000060]`で
> **完全に正・統計的に有意な改善**だった。調査の過程で、`DERIVED_RE`が"pseudo"を含む
> ファイル名を自動発見から恒久的に除外する設計のため、**Final2 1本目(id=216)の348
> メンバープールにはpseudo-labeling系統が実は1本も含まれていなかった**と判明
> （exp215の一回限りの手動追加テストの結果はその場だけの診断で、後続のプールには
> 反映されていなかった）。この設計上の見落としを修正しただけで+0.00004の本物の改善が
> 得られた。**恒久対応**: `DERIVED_RE`のような「意図的な除外リスト」は、除外理由
> （循環依存・派生ファイルの二重カウント防止）が生きているかを系統ごとに再確認すること。
> 「あるはずのものが実はプールに入っていない」は個別Δの単純合計を検算に使うことで
> 検出できる（本ケースでも合計値の食い違いが発覚の糸口になった）。
>
> ⚠️ **LB検証(exp230のtest予測を提出)の結果、LB=0.97089**（id=216=0.97092より-0.00003、
> Publicノイズ床±0.00014内で有意差なし）。**OOFではz=+4.21・95%CIが完全に正という
> 強い統計的証拠があったにもかかわらず、LBでは改善どころかid=216よりわずかに下回った。**
> これは`G-OOF`の但し書き（天井帯ではOOF最高が必ずしもPrivateで勝つとは限らない、
> OOF差の約半分しかPrivate/Publicで実現しない）を実地で裏付ける新事例。**教訓**:
> bagged paired bootstrapで確認した「有意な差」は、その差自体の存在を否定しないが、
> 天井帯では**単一のLB観測点だけでFinal2候補を入れ替える根拠にはならない**——
> OOFの有意差とLBでの再現は別の主張であり、後者は独立した確認が必要。

### L-20 単体モデルの品質改善が、密なアンサンブルには伝播しなかった(検証過程でG-FAIR違反も発覚)

*対応する指針: `G-OVERFIT`, `G-CEILING`, `G-FAIR`*

**教訓 (s6e8実測)**: 外部カーネル(tomasa2氏)由来のFE(H-023: frequency encodingを
train foldのみでなくtrain+test結合でカウントするtransductive化)を実装したところ、
単体LGBでΔOOF=+0.00030・LB+0.00026、単体XGBでΔOOF=+0.00038と、**2アーキテクチャで
再現する本物の改善**を確認した(exp232-235)。

**1回目のPALL統合検証(exp234/236)で2つの問題があった(ユーザー指摘で発覚)**:
1. 1メンバーとして追加 → 完全希釈(`z=-0.13`)は妥当な検証だった
2. しかし「既存メンバーとの置換」検証(exp236)は、**単体seed版(064_lgb・080_xgb)
   だけを置換し、より影響力の大きいavg5版(065_lgb_h012_multiseed5・
   081_xgb_h012_multiseed5)を旧fq_のまま放置**していた。さらに**旧fq_向けに
   チューニングされたHPをそのまま流用**しており、`G-FAIR`の「新特徴量セットに
   旧HPを流用」パターンそのものだった。この不完全な検証は`mu_delta=-0.000003,
   z=-3.160`と**統計的に有意な悪化**という強い(誤った)シグナルを出した。

**修正後の再検証(exp237-241)**: ①現行HPのままavg5構築→②専用HP再調整
(Optuna25試行、Δ+0.00002〜+0.00005とごく僅かな改善に留まる=旧HPが既に近似的に
最適だった)→③再調整済みHPでavg5再構築(LGB avg5 Δ+0.00039, XGB avg5 Δ+0.00043、
現行HP版より更に改善)→④**4メンバー全て(単体+avg5)を正しく置換**して再検証。
結果は`mu_delta=-0.000001, z=-0.494, 95%CI=[-0.000003,+0.000002]`と**有意差なし**
（悪化ではない）に収束した。

**メカニズム**: 560+メンバーのL2 LogRegスタッカーは、既存メンバー(旧fq_版)との
組み合わせで既に局所最適な重みを学習済みである。個別モデルを「より高品質な版」に
差し替えても、相関構造がわずかに変化するだけでスタッカーの最適化前提が動き、
単体品質の向上分がそのまま伝播しない。ただし**不完全な検証(一部メンバーだけ置換・
HP未調整)は「悪化」という間違った結論を出しうる**——今回は「有意な悪化」から
「有意差なし」への訂正だったが、逆方向（見せかけの改善）も起こりうる。

**恒久対応**:
1. 「単体モデルのFE/HP改善がアンサンブルレベルの改善に自動的に変換される」という
   前提を置かない。改善が密なプールでも生きるかは、**1メンバー追加**と
   **既存メンバー置換**の両方で個別に検証する（追加は重複による希釈、置換は
   スタッカー再最適化の撹乱という異なる失敗モードを持つ）
2. **置換検証をするときは、対象アーキテクチャの「関連メンバー全て」（単体版・
   multi-seed avg版など）を洗い出してから置換する**。一部だけ置換すると
   「旧版と新版が混在した歪なプール」で測定することになり、結果の解釈を誤る
3. **G-FAIRは新機能をPALLに投入する前段階（単体モデルでのHP再調整）でも適用する**。
   借用HPでの結果を「決定的」と扱わず、専用HP再調整後に再検証してから最終判断する
4. 両方とも希釈/有意差なしの場合、パイプライン全体への大規模展開は費用対効果が
   低いと判断してよい（今回のケースでは展開を見送り確定）

> **追試(exp256-258, s6e8)**: H-023(transductive fq_)とH-026(sin/cos特徴量、
> tomasa2氏由来の別発見)を同時投入すると、単体LGBでΔOOF=**+0.00036**
> （H-023単独+0.00030・H-026単独+0.00018の部分的な相乗、検証全体で最大の
> 単体改善）を確認した。この**過去最大の単体改善**をPALLへ1メンバー追加した
> ところ、`mu_delta=+0.000000, z=+0.095, 95%CI=[-0.000002,+0.000002]`と
> **完全にゼロ**——H-023単独(z=-0.13)・H-026単独(z=+2.25)よりもさらに希釈
> された。**単体改善の大きさそのものは、密なアンサンブルへの伝播しやすさを
> 決めない**ことが最も明確な形で裏付けられた。すでに570以上のメンバーを持つ
> スタッカーにとって、新メンバーの価値は「単体品質」ではなく「既存メンバー群が
> 説明できていない残差との相関」で決まる——単体スコアがどれだけ良くても、
> 既存メンバーとの相関が高ければ実質的な情報追加はゼロになる。

> **⚠️訂正(exp267, s6e8)**: 上記の「完全にゼロ」は**単体seed・借用HP版
> （id=246, OOF=0.96823）を1メンバー追加**する形でのテストだった。ユーザーの
> 「再計算した方が良いのでは」という指摘を受け、その後avg5化→専用HP再調整→
> avg5再構築で伸びた**本番パイプライン最終版**（LGB id=251=0.96866, XGB
> id=253=0.96886）で、`064_lgb`/`065_lgb_h012_multiseed5`/`080_xgb`/
> `081_xgb_h012_multiseed5`の4メンバーを正しく置換（exp241と同型の手順）して
> 再検証したところ、`mu_delta=+0.000003, z=+2.458, 95%CI=[+0.000001,+0.000005]`
> と**95%CIが完全に正——統計的に有意な改善**を検出した。「完全にゼロ」という
> 結論は最適化前の弱い版でのテストによる誤りだったと判明。
> **G-FAIRは「置換テスト自体で使う新メンバーの版」にも適用される**——PALL統合
> 検証を単体seed・借用HPの初期版だけで済ませず、本番パイプライン最終版
> （avg5+専用HP再調整）に到達してから最終判断すること。ただし効果量
> （+0.000003 bagged OOF）はL-21のLB確認可能閾値の1/10未満で、実務上の結論
> （Final2への統合見送り）自体は変わらない——**「効果ゼロ」と「効果はLBで
> 測れないほど小さいが実在する」は異なる主張であり、後者が正確な表現**。
### L-21 「bagged paired bootstrapでOOF有意」が6回連続でLBに再現しなかった

*対応する指針: `G-OOF`, `G-NOISE`, `G-CEILING`*

**教訓 (s6e8実測)**: 天井帯(スコア密集帯)において、12シードbaggingしたPALL構成
同士のpaired bootstrapで統計的に有意なOOF差(z>2)を検出した候補を6回LB検証したが、
**6回とも有意な改善はLBで確認できなかった**（うち1回はノイズ域内のわずかな悪化、
1回は逆に「非有意」と判定した施策がLBでは方向一致した——下記#2参照）:

| # | 施策 | OOF paired bootstrap | LB(vs id=216基準=0.97092) |
|---|---|---|---|
| 1 | pseudo-labeling 5本スケールアップ(id=218) | z=+4.21, 95%CI完全に正 | 0.97089（-0.00003） |
| 2 | H-023(transductive fq_)PALLスワップ(id=231) | 単体では有意だがPALL swap自体はz=-0.49で非有意 | 0.97092（完全一致） |
| 3 | Cスイープ(C=1.0→0.03, id=236) | z=+3.52, 95%CI完全に正 | 0.97091（-0.00001） |
| 4 | pseudo-labeling 中間構成(id=219) | z=+5.65, 95%CI完全に正 | 0.97091（-0.00001） |
| 5 | H-023+H-026本番パイプライン最終版スワップ(id=254) | z=+2.46, 95%CI完全に正 | 0.97093（+0.00001、符号は一致） |
| 6 | #3(Cスイープ)+#5(swap)の複合(id=260) | z=+4.03、検証全体で最強の有意性 | 0.97092（完全一致） |

**メカニズム**: bagged paired bootstrapのσ_delta(0.000005〜0.00001)はPublic LBの
ノイズ床(±0.00014)より1桁以上小さい。そのため「OOFでは検出できるがLBの解像度
では見えない」領域の改善が構造的に存在する——G-OOFの「OOF差の約半分しかPrivateで
実現しない(回帰の傾き0.52)」という既存知見を、**6つの独立したメカニズムで定量的に
再確認**した形になる。**#6は特に重要**: 2つの独立した調整軸(メンバー品質向上・
正則化強度)がOOFでは加算的に効き、統計的有意性も単独より強化された(z=4.03)にも
かかわらず、LBでは効果が完全に消えた——「OOFでの有意性の強さ」と「LBでの再現
可能性」は別軸であり、複合適用による有意性の強化それ自体はLB再現を保証しない。

**恒久対応**: 天井帯でのOOF改善(+0.00001〜0.00003程度の小さい値)は、たとえ
paired bootstrapで統計的に有意でも、**単一のLB観測点で確認できなければFinal2の
構成を変更する根拠にしない**。複数のLB観測点で同じ方向の改善が積み重なった場合
のみ採用を検討する。逆に言えば、「bagged OOFで有意」は「その施策に真の効果がある
可能性が高い」ことの証拠にはなるが、「その効果がPrivateで実現する」ことの証拠には
ならない——両者を混同しない。

### L-22 小規模ブレンド(2本目)は結合方式そのものを変えると育つ——ただし「候補追加」ではなく「結合方式」が鍵

*対応する指針: `G-CEILING`, `G-OVERFIT`(b)*

**教訓 (s6e8実測)**: Final2 2本目(P5, 5メンバーの非負simplexブレンド)を育てる
試みが、L-20/L-21と対照的に**LBで確認できる改善**(OOF+0.00012→LB+0.00017)に
到達した。プロセス:
1. **「新規メンバー追加」は不発**(exp271, exp274): LGBを新規追加(P5には一度も
   含まれていなかった)しても、既存XGBとcorr=0.9994でweight=0.000。プール全体
   582候補を相関でスキャンしても、最も低相関の候補(corr=0.89)ですら単体性能が
   P5核の最弱メンバーを下回り、非負simplex制約では選ばれなかった——「相関が低い」
   だけでは不十分で、単体性能も既存最弱メンバー並みでなければならないという
   構造的制約が判明した
2. **「結合方式の変更」が鍵だった**(exp275): 同じ5メンバーを非負simplexではなく
   signed L2ロジスティック回帰(PALLと同じ結合方式)に変えても単体では効果なし
   (simplex OOF=0.96942 ≈ signed OOF=0.96942)。しかし**signed方式に切り替えた
   状態で**、simplexでは重み0だった低相関・低性能候補(単体AUC 0.955〜0.968、
   rank-average版PALLとweakLGBのrandom subspaceバリアント計4種)を投入したところ、
   OOF=0.96954(+0.00012)、**LB=0.97060(+0.00017、現行P5比)**と明確に改善した。
   4候補は全て負係数(-0.07〜-0.36)を獲得しており、単体では弱くても「既存メンバー
   群の誤差に対する補正方向」として機能した

**メカニズム**: 非負simplex制約(`w∈[0,1], Σw=1`)は「弱いメンバーを引き算に使う」
ことができない——単体性能が低い候補は必ず重み0になる。signed係数(正負とも許容)は
弱い候補を「他メンバーの誤りを打ち消す小さな補正項」として使えるため、同じ候補群
からでも全く異なる価値を引き出せる。PALL(1本目)が単体AUC=0.871まで単調寄与を
示していたのも同じメカニズムだが、P5(2本目)は元々simplex制約だったためこの経路が
封じられていた。

> **追試(exp276, id=263, s6e8)**: 手動選択した低相関候補4種(id=262)を、weakLGB
> 派生313件+生成モデル派生125件の整合性ゲート通過分421件**全て**に拡張したところ、
> OOF=0.96965(id=262から+0.00011)、**LB=0.97078(id=262から+0.00018)**とさらに
> 改善した。元exp145からの累計はLB+0.00035で**G-NOISEの2σ閾値(+0.0002)を明確に
> 超える確定的改善**——1本目id=216(LB=0.97092)まで-0.00014に接近した。ただし
> corr(vs id=216)は0.9985→0.9990へやや上昇(候補プールがPALLと重複するため)。
> **「新規メンバーを手動で数個選ぶ」より「関連候補を機械的に全投入してL2正則化に
> 選ばせる」方が一貫して良い結果を生んだ**——PALLが105→580+メンバーへ成長した
> のと同型の「量が正則化された結合器の質に転嫁する」機構が、独立に構築した2本目
> でも再現した。

**恒久対応**:
1. 小規模ブレンド(2本目・hedge候補)を「新規メンバー追加」で育てようとする前に、
   まず**結合方式がsigned係数を許容しているか**を確認する。simplex制約なら
   まずsigned方式への切り替えを試す方が投資対効果が高い
2. signed方式に切り替えたら、simplexで「重み0」と判定され棄却済みの候補群を
   **再評価する**——simplexでの棄却は「価値がない」ではなく「simplexでは使えない」
   だった可能性がある
3. L-20/L-21(密なPALLでは単体改善もLB確認も困難)と対照的に、**5〜7メンバーの
   小規模ブレンドは結合方式の選択次第でLB確認可能な改善余地が残っている**——
   天井帯でも「PALLは飽和・小規模ブレンドは結合方式で伸びる」という非対称性を
   意識する
### L-23 新しい派生アンサンブルの命名規則がDERIVED_REに一致せず自己参照混入した

*対応する指針: `G-MECH`, `G-OVERFIT`*

**教訓 (s6e8実測)**: `discover_pool()`は「既存メンバーの再構成(ブレンド・スタッキング
出力)」を`DERIVED_RE`という正規表現パターンで除外し、独立メンバーのみをアンサンブル
候補として扱う設計だった。しかしexp274-280で作った小規模blend派生(`263_p5_signed_
full_pool`等、`p5_`/`p6_`プレフィックス)とPALL派生結合(`260_pall_swap_csweep_
combined`等、`pall`を含む名前)は、既存のパターン(`blend|greedy|stacking|_stack_|
_5way|pseudo|exp1\d\d_|topn_avg|screen_to_sleep_replace`)のどれにも一致せず、
「PALLの全アーキテクチャ多様性×新特徴量コア」を検証する新しいPALL再構築
(exp280)で自己参照混入した。**症状は明確だった**: bagged OOF=0.97092(他の全
PALL派生候補の水準0.9698から異常に乖離)、seed間std=0.00017(通常の0.00001の
17倍)——既に高度に最適化された派生アンサンブル(`265_p5_broader_pool`等、
OOF=0.96981)を「独立な生の予測」として再度スタッキングしたことで生じた症状。

**より深刻な発見**: 同じ理由で`162_rank_average_pall`(PALLプール全体のrank-average)
・`163_stacking_lgb_pall`・`195_stack_calibrated_pall`・`196_greedy_hc_pall`
という**もっと古い派生ファイル**も、当初の`DERIVED_RE`に一度も一致しておらず、
**このセッション全体を通じて**(1本目id=216の構築時も含め)`discover_pool()`の
「独立メンバー」として混入し続けていた可能性がある。これらは単純な平均・軽量な
最適化のみで構成され、id=265のような完全最適化されたsigned stackerほど強力な
「チート特徴量」ではないため(実際、これまでのPALL系列の全結果でstdの異常値は
一度も見られていない)、実害は小さいと推測されるが、内部OOF推定にはわずかな
楽観バイアスが混入していた可能性を否定できない。**ただし1本目id=216を含む全ての
確定候補は実際にKaggle LBへ提出し実測値で確認済みであり、Public LBスコアという
「本物の答え」に基づく最終判断(Final2選定)自体はこの内部推定バイアスの影響を
受けない**——内部OOFでの細かい比較(swap判断・C-sweep等)の信頼性にのみ影響する。

**恒久対応**:
1. `DERIVED_RE`は「派生アンサンブルの命名慣習を網羅するブラックリスト」であり、
   **新しい派生アンサンブルを作るたびに、その名前が既存パターンに一致するか
   機械的に確認する**（`python -c "import re; print(bool(DERIVED_RE.search('新ファイル名')))"`
   を投入前に必ず実行する）
2. 派生アンサンブルの命名規則を統一する（例: 派生アンサンブルは必ず`_ens_`を
   含める等）ことで、ブラックリスト方式より取りこぼしに強いホワイトリスト方式
   ないし命名規約チェックに切り替える方が根本的な解決になる
3. 症状（bagged OOFの異常な高さ・seed間stdの異常な大きさ）を**プールの整合性
   チェックの一部として自動検知する**——今回は偶然OOF値が既存レンジから
   大きく外れていたため気づけたが、もし異常値がレンジ内に収まっていたら
   気づかずに採用していた可能性がある
4. `G-MECH`の原則通り、「除外し忘れないよう注意する」という指示ではなく、
   機械的な検知（既存メンバーとの相関が0.999999超の完全一致だけでなく、
   派生アンサンブルらしい高相関パターンの検知）をガードとして実装すべき

### L-24 学習と推論を分けたせいで「提出したくなった実験」を何度も回し直した

*対応する指針: `G-STEPWISE`, `G-MECH`*

**教訓 (s6e8実測)**: FE 実験の多くは「ΔOOF を見る」ことだけを目的に書かれたため、
`oof_*.npy` は保存しても `test_*.npy` を作らなかった。ところが後になって
「この構成を LB で確かめたい」「ブレンドのメンバーに入れたい」と判断が変わることが
繰り返し起き、そのたびに**まったく同じ学習をもう一度回した**。tree 系なら数分で済むが、
NN 系（RealMLP / mlp_embed）や multi-seed 構成では 1 回あたり数十分〜数時間かかり、
締切直前の最も貴重な時間帯にこの再実行が集中した。

**なぜ起きたか**: 「今は ΔOOF を測るだけ」という**その時点の目的**でスクリプトの
出力範囲を決めていた。しかし実験の価値は事後に変わる。test 予測は学習済みモデルが
メモリ上にある間なら**ほぼゼロコスト**（fold ごとに `predict` を 1 回足すだけ）なのに、
後から作ろうとすると学習コスト全額を払い直すことになる——この非対称性が見えていなかった。

**恒久対応**:
1. 学習した実験は同じ実行内で **OOF・test 予測・提出 CSV の 3 点を出し切る**。
   `src/utils/finalize.py` の `save_run_outputs()` を最後に 1 回呼ぶだけで揃う
2. `ExperimentTracker.end_run()` の**推論成果物ガード**が「OOF はあるのに test が無い」
   実験を機械検知して警告する（`G-MECH`: 注意ではなく観測可能な結果の側から守らせる）
3. 例外は ΔOOF スクリーニング専用（`scripts/feature_study.py` 等）だけに限る
4. 姉妹版の無駄として **multi-seed avg で基本 seed を再学習していた**問題がある。
   `src/utils/multiseed.py` の `run_multiseed()` は既存 seed 結果を再利用するので、
   avg5 の学習時間が 1/5 削減される（単体モデルの実験で基本 seed は既に回っているため）

### L-25 「規律は機械に守らせる」をテンプレート自身に適用していなかった

*対応する指針: `G-MECH`*

**教訓 (テンプレート点検で判明)**: `G-MECH` はテンプレートの旗印であり、可視化ガード・
診断記録ガード・`doc_audit` と 3 つの機構を持っていた。ところが**強制の入口は
PostToolUse hook 1 本だけ**で、Claude Code が提供する `SessionStart` / `PreToolUse` /
`Stop` / `PreCompact` は一度も使われていなかった。そして**実際に破られた規律は、
すべて機械化されていない側にあった**:

| 規律 | 機構 | 実績 |
|---|---|---|
| 可視化 | ブロッキング | 機能した |
| 提出前確認 | **無し**（テンプレート自ら「違反とみなす」と明記） | 毎回 AI の自己申告 |
| 1 実験 1 コミット | **無し** | 最も頻繁に破られた |
| 状態ファイルの更新 | **無し** | FEATURE_REPORT が 3 週間停滞 |

**設計として学んだこと**:

1. **規範を書いた場所と、それを守らせる機構の場所を対応表にして棚卸しする。**
   「守らせたい規律」を列挙し、各々に機構があるかを表で確認しないと、
   機構のある規律ばかり強化され、無い規律は放置されたままになる
2. **hook が検証できるのは「人間が承認したか」ではなく「提示された事実が実測か」。**
   提出ゲートは `permissionDecision: "ask"` で人間に承認を戻しつつ、
   数字（提出枠・締切・git 状態）は API と時計から取る。過去の事故はすべて
   「提示された数字が記憶であって実測でなかった」ことが原因だった
3. **ブロックしてよい場所を絞る。** 実績のある可視化ガードと、不可逆な提出だけ。
   Stop hook でブロックすると停止と再開のループを招く
4. **ガードは自分の変更にも牙を剥く。** 導入直後、提出ゲートは
   「ドキュメントに提出コマンドを書いた Bash」自身をブロックした（shlex による
   コマンド位置判定で解決）。上限行数チェックも自分の追記で ERROR になった。
   **このとき閾値やパターンを緩める誘惑に負けないこと** — ガードを無意味にする最短経路であり、
   正しい対応は「入れるなら出す」（L0 の退避ポリシー）を実行すること

### L-26 上限を「行数」で測っていたため、上限を守りながら中身が増えていた

*対応する指針: `G-MECH`*

**教訓 (テンプレート監査で判明)**: L0（CLAUDE.md）には「常時ロード 650 行」という上限があり、
`doc_audit` C1 が ERROR で強制していた。ところが**測っていたのは改行の数**だった。

ある改善作業の 1 日で、実測はこうなっていた:

| | 作業前 | 作業後 |
|---|---|---|
| 行数（C1 が見る値） | 650 | **648**（−2） |
| 文字数（実際のコンテキスト費用） | 30,214 | **32,112**（**+1,898**） |

**上限を守りながら中身は 6% 増えていた。** 原因は、上限に達するたびに
「2 つの箇条書きを 1 行に結合する」ことで行数だけ下げていたこと。
悪意も手抜きもなく、**規約（1 行入れるなら 1 行出す）に忠実に従った結果**そうなった。

**根本原因**: 規律の「宣言された単位」と「実際に効く単位」がずれていた。
コンテキストの費用はトークン（≒文字数）で決まるのに、測定は改行で行っていた。
同じ日に `session_brief` で「行数上限だけでは 150 字の 1 行を素通しする」と気づいて
文字数上限を足していたのに、**その教訓を CLAUDE.md 自身に適用していなかった**。

**恒久対応**:
1. C1 を**文字数**で測る（`ALWAYS_LOADED_BUDGET = 30_000`）。行数は参考表示に落とす
2. C11（README の自己申告値）も同じ単位に揃える。
   **単位を変えたら、その値を検査していたガードの追随を必ず確認する** —— 今回、
   README から該当記述を移した瞬間に C11 は「検査対象ゼロで合格」になっていた（2 度発生）
3. 退避ポリシーの文面を「1 行入れるなら 1 行出す」から「**入れた分の量を出す**」に改める

**一般化**: 規律を機械化するときは、**測る単位が「守らせたい実体」と一致しているか**を必ず問う。
一致していない指標は、遵守しているつもりの迂回を生む。しかもその迂回は自覚されない。

### L-27 上限は「自分で決めた数字」ではなく「外部が定める数字」で測る

*対応する指針: `G-MECH`*

**教訓 (テンプレート監査で判明)**: L0 の上限を「行数 650」→「文字数 30,000」→「27,000」と
**自分で決めた数字**で締めてきた。ところが **Claude Code 自身が CLAUDE.md に対して
15,000 字で警告を出す**（バイナリ内の定数、`Large CLAUDE.md file detected`）。
自分で設定した 27,000 は、**公式の推奨より 80% 緩い数字**だった。

しかも締め直すたびに「実測 + 数%」を上限にしていたため、**上限は常に現状の追認**になっていた。
現状を基準に上限を引く限り、どれだけ締めても「今の大きさは適正」という結論しか出ない。

**恒久対応**:
1. 上限に**外部の基準がないかを先に探す**。ツール本体・公式ドキュメント・リンタが
   閾値を持っていることがある（今回はバイナリの文字列検索で発見できた）
2. L0 を **憲法（3,000〜5,000 字）** に絞り、判断指針 17 件は `GUIDELINES.md` へ分離した。
   常時ロードは **26,105 → 3,243 字（−88%）**
3. 分離すると「索引にあるが本文が無い／本文はあるが索引に無い」が起きる。
   `doc_audit` に **C12（指針の索引と本文の一致）** を追加して機械検知する
4. 指針が常時ロードから降りた分、**決定地点で読ませる導線**を各スキルに置いた
   （8 スキルすべてが担当フェーズの `G-*` を明示）

**なぜ今それができたか**: v6 が 650 行を必要としたのは、規律を守らせる仕組みが文章しか
無かったから。v6.5 で **hook 6 種 × ガード 5 種**が入り、**強制力が散文から機械へ移った**。
散文が常時ロードから降りられるのは、機械が代わりに線を守っているからであって、
機械化なしに同じ削減をすれば、単に規律が消える。

### L-28 ガードもテストも「壊れたことを検知する側」を持っていなかった

*対応する指針: `G-MECH`*

**教訓 (テンプレート監査 第4ラウンドで判明)**: 43 件のテストが全部通る状態で、
実行時ガード 5 種を `return None` に潰しても**全件合格したまま**だった。
「ガードが動くこと」のテストはあっても「**ガードが発火すること**」のテストが 1 件も無い。
提出ゲートも同様で、テストは `&&` の形しか見ておらず、**8 パターン中 6 件が素通り**していた
（`uv run` 前置・`;` 区切り・改行・`nohup`・`time`・絶対パス）。
そのセッション自身が複数行コマンドを多用しており、**ゲートは実質機能していなかった**。

同じ構造がエージェント定義にもあった。テストは `glob("*.md")` の結果を parametrize していたので、
**エージェントを 1 つ消すと検査項目ごと消え、残った分が全部正しいので ✅ になる**。

**恒久対応**:
1. ガードには「発火する条件」と「黙る条件」の**両方**のテストを書く
2. **入力を列挙して回すテストは、入力が消えたことを検知できない**。
   期待する集合を別に持つか、参照する側（文書）から逆向きに検査する
3. 検知の見逃しと誤検知が非対称な場面（提出のような不可逆操作）では、
   **テストケースを検知側に厚く積む**。誤検知は確認が 1 回増えるだけ、
   見逃しは無確認の提出になる
4. 内部エラー時は**通す側ではなく確認を求める側**に倒す（fail open を作らない）

### L-29 「例外を出さずに結論だけが逆になる」欠陥は、動作確認では見つからない

*対応する指針: `G-DIAG` / `G-FAIR`*

**教訓 (テンプレート監査 第4ラウンドで判明)**: 実測で以下が同時に見つかった。
いずれも**エラーを出さず、それらしい数字を返す**。

| 欠陥 | 何が起きるか | 実測 |
|---|---|---|
| ΔOOF が改善方向に揃っていない | RMSE 系で**良い FE を棄却し悪い FE を採用** | 符号が反転 |
| 提出形式の既定が `label` | AUC でハードラベルを提出 | **AUC −0.074** |
| early stopping が検証 fold を監視 | OOF が構造的に楽観側へ寄る | **AUC +0.00467**（ノイズ床の 20 倍超） |
| `N_CLASSES = 3` の直書き | clone 直後に Stage 1 が動かない | 目的関数とクラス数が矛盾 |
| importance が split なのに表示は "gain" | `G-DIAG` 第3診断軸の解釈が狂う | 別の量を見ていた |
| `--resume` で `tr_score = val_score` | train−val 乖離が「乖離ゼロ」に化ける | 診断列が嘘になる |
| ΔOOF の比較相手が自分自身 | ΔOOF が常に ±0.00000 | 診断が常に「判別不能」 |
| FoldCache の tag が本数のみ | 列や HP を変えても**古い予測を再利用** | 条件が混ざる |

**恒久対応**:
1. **向きを持つ量（指標・gap・Δ）は、定義元で改善方向に揃える。**
   呼び出し側で符号を扱うと必ずどこかがずれる
2. **出力の形を決める分岐は 1 箇所に置く**（`shape_for_metric`）。
   同じ三項演算子が 6 箇所に写経されていた
3. **設定と実データが食い違ったらその場で止める**（`_resolve_n_classes`）
4. e2e スモークテストを常設する。個々の関数が正しくても**繋がっていない**ことがあり、
   上表の 3 件は単体テストでは見えなかった

### L-30 「ガードが発火するテスト」を、ソースの字面で書いてしまった

*対応する指針: `G-MECH`*

**教訓 (テンプレート監査 第5ラウンドで判明)**: L-28 で「ガードには発火する条件と黙る条件の
両方のテストを書く」と結論した。その直後に書いたテストは、**過去のパッチの字面**を
grep するものだった。

```python
assert 'importance_type="gain"' in src                       # ← XGBoost 分岐でも満たされる
assert "cross_val_predict" in body                           # ← 関数内の import 行で満たされる
assert "medians = train[NUMERIC_COLS].median()" in src       # ← test 側を見ていない
assert "covered" in src and "covered[val_idx] = True" in src  # ← 評価に使ったかは見ていない
```

字面テストは「**同じ diff を revert する**」ことしか検知しない。意味的に別の書き方で
同じ欠陥を入れると必ず通る。実測（変異注入 25 件）で **13 件がすり抜け**、
欠陥 3 件が同時に存在した状態で **98 件全件が緑**だった。

さらに悪い形が 2 つあった:

- **名前だけ見て振る舞いを見ていない**: `test_weight_bagging_is_available` は
  シグネチャに `n_seeds` があることしか見ておらず、`n_seeds` を完全に無視する実装でも通る。
  この関数は「bagging を実行できるようにする」ために作られたのに、
  **実行できない状態に戻っても落ちない**。
  `test_atomic_write_survives_reader` も、名前に反して**並行読み手を作っていなかった**。
- **一度もテストされていないファイル**: `predict.py` / 実験雛形 / `av_check.py` /
  `visualize.py` は変異させても全件緑。`start_run` に至っては
  **テストファイル全体に文字列が 1 度も出てこなかった** ——
  「警告ではなく実行を止める」という第4世代の対策が第3世代に戻っても誰も気づかない。

**恒久対応**:
1. **実装を呼んで出力を確かめる。** importance は gain と一致し split と一致しないことを
   値で見る。スタッキングは in-sample 予測と数値が異なることを見る。補完は実際に
   `preprocess` を走らせて test の値が train の中央値になっていることを見る
2. **「その機能が使えること」ではなく「使えなくなったら落ちること」を基準にテストを書く。**
   `n_seeds` は「引数がある」ではなく「変えると重みが変わる」を見る
3. **e2e にタスクを 1 つ足すのが最も効率が良い。** 回帰設定を追加しただけで、
   blend の回帰死・実験雛形の `AxisError`・CatBoost の `eval_metric` 残留が
   まとめて落ちた（どれも単体テストでは見えなかった）
4. **変異注入を定期的に回す。** 「テストが通ること」と「テストが守っていること」は別物で、
   後者は欠陥を入れて初めて測れる（→ `PLAYBOOK.md` のこの節）

### L-31 判断の床が桁で間違っていた —— 「有意なのに再現しない」の正体

*対応する指針: `G-NOISE` / `G-DIAG` / `G-TWOAXIS`*

**教訓 (テンプレート監査 第5ラウンドで判明)**: 採否を決める閾値を 3 箇所で持っていたが、
**どれも実測と桁で違っていた**。しかも 3 つは互いに矛盾していた。

| どこ | 使っていた床 | 実測すると | ずれ |
|---|---|---|---|
| `G-NOISE` の表（AUC, n_pos=5K） | ±0.0001 | 0.00319（Hanley-McNeil） | **32 倍過小** |
| `G-DIAG` の「fold 間 std 未満は測れていない」 | 0.01251 | 0.00124（fold 対応差の SE） | **10 倍過大** |
| `feature_study` の ΔOOF 閾値 | ±0.0003 | ΔOOF 自身の SD 0.0011 | 1/4 |

**同じ FE の採否判断に、40 倍違う 2 つの床（0.0003 と 0.0125）が同居していた。**

`G-NOISE` の表は典拠として Hanley-McNeil を挙げていたのに、その式に代入すると
32 倍の値が出る。±0.0001 は実際には**相関 0.999 のペア差**の値で、
それを単体スコアの床として掲げた上に「paired はさらに 5-10x 小さい」と重ねたため、
実効閾値が **10〜20 倍甘く**なっていた。

**これが L-21（bagged paired bootstrap で OOF 有意だった 6 件が、全部 LB に再現しなかった）
の直接の説明。** 有意判定の床が桁で低ければ、再現しないのは当然の帰結。
逆に `G-DIAG` の床は 10 倍高すぎて、実在する改善を「判別不能」と切り捨てていた
（L-19 で「個別 Δ≈0」と判定した 13 系統が累積すると確定的な正の差になった理由）。

**恒久対応**:
1. **床を表で持たず `src/noise.py` で計算する。** 用途で 2 種類に分ける ——
   別の観測点同士は `single_score_se()`、同じ行を予測した 2 本は `paired_se()`、
   同じ fold で測った 2 本は `fold_paired_se()`。取り違えると判断が桁で狂う
2. `feature_study` の固定閾値を廃止し、**その 2 本から実測した床**で判定する。
   実証: 合成データで**完全に無関係な列**の ΔOOF が +0.00081 になり、
   旧閾値（>0.0003）では「🔶 採用検討」と判定された。新しい床では z=+0.33 で「測れていない」
3. `log.csv` に `fold_val_scores` を残し、次の実験が対応差の SE を出せるようにする
4. **「測れていない」と「効果がない」を言い分ける。** 前者は床を下げれば測れる可能性があり、
   後者は測った上で差が無い。表示で混同すると、探索の打ち切り方を間違える

### L-32 「基準線を観測値の中央値で引く」ガードは、検知したい対象に盲目になる

*対応する指針: `G-MECH` / `G-TWOAXIS`*

**教訓 (同ラウンド)**: `pub_oof_gap` ガードは「全提出の gap の中央値 + 0.0005」を閾値にしていた。
モンテカルロ 20,000 回での実測:

| 条件 | 発火率（修正前） | 発火率（修正後） |
|---|---|---|
| 帰無（真の gap = 0） | 84〜97% | **0.3%** |
| 正当な一定オフセット（+0.004） | — | **0.3%** |
| **真に危険**（後半だけ gap が広がる） | **92.9%** | **92.3%** |

修正前は帰無条件 93.1% と真の危険 92.9% が**ほぼ同じ** —— この警告は情報を持っていなかった。
原因は 2 つ:

1. **基準線が同じデータの中央値**なので、検知したい系統差そのものが基準線に吸収される
2. 閾値 0.0005 が LB のノイズ床（実測 0.002 前後）より小さく、純粋なノイズで常に鳴る

**恒久対応**:
1. **基準線は前半で固定する**（初期の安定期の中央値）。見たいのは「後から広がったか」であって
   「今の水準」ではない。5-fold OOF と全学習相当の test 予測を比べる以上、
   gap には正当な系統オフセットが常に乗るが、それは基準線に含まれるので差分では消える
2. **閾値はノイズ床から決める**（`src/noise.py`）。固定値にすると、指標もデータ規模も違う
   次のコンペで意味を失う
3. **点ではなく水準で判定する**（直近窓の中央値）。1 点でも超えたら鳴らす形は、
   窓の件数だけ偽陽性が積み上がる（実測 20.7% → 0.3%）
4. ガードを作ったら、**帰無条件と「検知したい条件」の両方で発火率を測る。**
   両者が同じならそのガードは情報を持っていない

### L-33 実際に見ていた OOF↔Public gap こそが、最良の床の測定器だった

*対応する指針: `G-CALIB-SUB` / `G-NOISE`*

**教訓 (テンプレート監査 第5ラウンド + s6e8 実データ 165 提出の再分析)**:
コンペ中、テンプレートは train−val gap・fold 間 std を律儀に記録していたが、
実際に注視されていたのは **OOF と Public の gap** だった。この直感が正しかった。

`gap = LB − OOF` の**散らばり**は「OOF では説明できない LB の動き」そのもので、
行のブートストラップでは再現できないもの（CV 分割の引き直し・train/test の分布差・
Public の標本ゆらぎ）が**すべて含まれている**。合成では作れない数字。

s6e8 実測:

| OOF 帯 | n | gap 平均 | gap SD |
|---|---|---|---|
| 0.9400〜0.9650 | 22 | +0.00155 | 0.00066 |
| 0.9650〜0.9690 | 93 | +0.00121 | 0.00138 |
| **0.9690〜0.9699** | **47** | **+0.00112** | **0.00007** |

最終盤の床（2σ）= **0.00013**。ところが同じ帯の隣接実験の ΔOOF 中央値は **0.000010** で、
**床を超えたペアは 0%**。その間 8 日・32 提出で LB 更新はゼロだった。

**「飽和した」のではなく、検出可能な大きさの差をそもそも作れていなかった。**
当時この床を出していれば、「この方向はもう測れない」を数値で示せた。

当時の閾値との突き合わせ（実測床 0.00013 に対して）:

| 使っていた閾値 | 実測床との比 |
|---|---|
| `G-NOISE` の表「突破 2σ = +0.0002」 | 1.5 倍（**妥当だった**） |
| 同・脚注「paired は 5-10x 小」→ 0.00002〜0.00004 | 0.22 倍（甘すぎ） |
| `feature_study` の +0.0003 | 2.2 倍（やや厳しい） |
| `G-DIAG` の `cv_val_std` ≈ 0.01 | **80 倍**（桁違いに厳しい） |

**L-31 で「表が 32 倍過小」と書いたが、それは単体スコアの床としての話。**
実際の運用（対応比較）では見出し値はおおむね妥当で、真に外れていたのは
脚注（4 倍甘い）と `cv_val_std` 床（80 倍厳しい）だった。**問題は値の大小ではなく、
用途の違う 3 つの床が混在し、どれも自分の用途に合っていなかったこと。**

**恒久対応**:
1. `src.noise.empirical_lb_floor()` —— 直近 20 提出の gap SD から床を測る。
   古い提出を混ぜない（全体 SD 0.00106 に対し最終盤に絞ると 0.00007。混ぜると実態より甘くなる）
2. `feature_study` / `end_run` / 提出ゲートが**自動で併記**する。
   提出ゲートは「今回の ΔOOF は床の何倍か」を出し、床未満なら枠を使う価値を問い直させる
3. **床下探索の通知**（`_check_below_floor_guard`）—— 直近 8 実験がすべて床の下なら、
   ①seed / fold を増やして床を下げる ②情報源を変える ③集約に切り替える、を促す。
   **これは飽和の宣言ではなく測定の限界の通知**（`G-PERSIST` と矛盾しない）
4. 留意: 似た提出ばかりだと散らばりは小さく出る。**系統を変えたら床を測り直す**

### L-34 「機能を作った」と「その機能が使える状態になっている」は別

*対応する指針: `G-MECH` / `G-SOURCE` / `G-CEILING`*

**教訓 (テンプレート監査 第5ラウンド)**: 前ラウンドで `get_cv(seed=)` を追加し、
docstring に「分割の bagging をしたいときはここに別の seed を渡す」と書いた。
ところが **production の呼び出し 3 箇所すべてが引数なし**で、`multiseed` も
モデル seed しか振っていなかった。つまり**全実験が単一の分割に条件付いていた**。

同じ形が 3 つ同時にあった:

| 作ったもの | 使えなかった理由 |
|---|---|
| `get_cv(seed=)` | 呼び出し側が誰も渡していない |
| `optimize_weights(n_seeds=)` | 引数はあるが、テストは「引数がある」ことしか見ていなかった |
| RealMLP / TabM | `kaggle_nb/` のアドホック実装で、`run_cv` / `feature_study` の外にいた |

3 番目の影響がとくに大きい。**FE の 1 列 ΔOOF 計測が tree 系だけに対して行われる**ため、
特徴量セットが tree に偏って最適化される。上位解法は単体 NN か
「NN を最良単体としたスタック」が主流で、そこが主戦場なのに、
テンプレートの計測系がそこに届いていなかった。

分割を引き直すと何が見えるか（無関係な列を 4 分割で評価した実測）:

```
分割ごとの ΔOOF = +0.00081, +0.00491, +0.01481, -0.00008
床の内訳: 行 1σ=0.00243 / fold 1σ=0.00126 / 分割 1σ=0.00341
```

**最も見落とされやすい成分（分割）が実は最大だった。** 同じ無関係な列の Δ が
分割次第で 185 倍動く。単一分割では「たまたま良い」構成を選び続けることになる。

**恒久対応**:
1. 機能を足したら、**その機能を使う導線を同じコミットで通す**。
   引数を増やすだけでは「使えるようになった」とは言えない
2. テストは「引数がある」ではなく「**変えると結果が変わる**」を見る（→ L-30）
3. e2e に通す。`--split-seed` / NN 系は e2e に入れて初めて「使える」ことが保証される
4. 探索と本番で**別の学習コードを持たない**。`optimize_hp` は `TRAIN_FN` を経由させ、
   early stopping のプロトコルと目的関数のずれを構造的に不可能にした

### L-35 「レシピは書いてあるが実装が無い」定石は、毎回リークの余地を作る

*対応する指針: `G-MECH` / `G-FAIR`*

**教訓 (テンプレート監査 第5ラウンド)**: target encoding・pseudo-labeling・後処理は
`PLAYBOOK.md` に手順が書かれていたのに **`src/` に実装が無く、毎回手書き**だった。
前コンペでは中核 FE の 13 列が target encoding で、そのたびに手書きしていた。

**TE と pseudo のリークはエラーを出さない。** 学習時だけスコアが跳ね上がり、
LB で落ちる形で現れる。実測（合成データ・n=1500）:

| 対象 | 素朴な実装 | fold 外で計算 |
|---|---|---|
| **行ごとに一意な列**（情報ゼロのはず） | AUC = **1.00000**（完全なリーク） | AUC = 0.50000 |
| 本当に効く列 | AUC = 0.73309 | AUC = 0.72418（信号は残る） |

pseudo は別の形で壊れる。前コンペでは寄与を「ゼロ」と判定していたが、追試で
**プールへの取り込み自体が行われておらず 1 本も入っていなかった**と判明した
（修正後 z=+4.21）。**「効かなかった」の前に「実行されていたか」を見る。**

**恒久対応**:
1. `src/utils/encoders.py` —— fold 外 TE（平滑化つき、多クラスはクラスごとに 1 列）と
   count encoding。**train は fold 外、test は train 全体**という非対称を実装で固定する
2. `src/utils/pseudo.py` —— `make_fold_pseudo()` は**この fold の学習部分だけ**から
   擬似ラベルを作る。`describe_pseudo()` が採用件数を毎回表示し、
   0 件なら警告する（「実行されていたか」を機械が示す）
3. `src/utils/postprocess.py` —— 重複行の統一・rank 変換・範囲 clip。
   `apply_postprocess()` が**指標を見て適用してよいものだけ**を実行する。
   実測: 重複行の統一で AUC **+0.02025**、回帰の clip で RMSE 20.126 → 7.269
4. 合成データでは**元データを train に連結する**のが最優先の定石。
   `sample_weight` で重みを分け、**OOF の評価は元の train 行だけ**で行う

**なお rank 変換について、当初「logloss では必ず悪化する」と書いたが実測で誤りと判明した**
（較正の良い予測でも改善する場合がある）。正しくは「**AUC では結果が変わらないことが
保証される**が、値そのものを見る指標では予測が別物になり、その指標を最適化する操作ではない」。

### L-36 床は「1 回の判定」しか守らない —— 繰り返しと、測り方の省略

*対応する指針: `G-NOISE` / `G-FULLCV`*

**教訓 (テンプレート監査 第7ラウンド)**: L-31〜L-33 で床を実測に置き換えたが、
**床さえ正しければ判断が正しくなるわけではない**ことが 2 点で露呈した。

**① 床を「安く」測ると、最大成分が抜ける**

`feature_study --n-repeats` の既定は 1（＝分割を引き直さない）。実測（無関係な列）:

| 計測 | 行 1σ | fold 1σ | 分割 1σ | 採用する床 |
|---|---|---|---|---|
| 1 分割（既定） | 0.00243 | 0.00126 | — | 0.00243 |
| 4 分割 | 0.00243 | 0.00126 | **0.00341** | 0.00341（**40% 大きい**） |

**分割由来の分散が最大成分**なのに、既定ではそれを欠いた床で採否を決めていた。
かといって常に 3〜5 回引くと FE 1 列の計測時間が 3〜5 倍になり、
何十件も試す運用と両立しない。**用途で分けるのが正解**:

- **スクリーニング**（1 分割・既定）= 候補を絞る。床は**下限**だと明示し、
  「採用推奨」とは言わず「`--n-repeats 3` で測り直せ」と**次の行動を指定する**
- **採用判定**（3 分割以上）= 特徴量セットに入れる決定。分割由来の分散を床に含める

**② 判定を繰り返すと、床は偽陽性を止めない**

| 試行数 | 2σ での期待偽陽性 | 少なくとも 1 件出る確率 | 3σ での期待 |
|---|---|---|---|
| 10 | 0.2 件 | 20.6% | 0.01 件 |
| 50 | 1.1 件 | 68.4% | 0.07 件 |
| **87**（前コンペの FE 仮説数） | **2.0 件** | **86.6%** | 0.12 件 |

**87 件中 2 件が「採用推奨」に見えるのは、効果ゼロでも起きる**。
`feature_study` が毎回「これまで N 件を計測。効果ゼロでも 2σ で期待 X 件」を表示する。
機械的に補正（Bonferroni）はしない —— 本物の弱い改善まで落ちるので、
**表示して判断材料にする**（この設計は `G-NOISE` の「床を見せて判断させる」と同じ）。

**③ 行由来と分割由来は独立成分 —— 二乗和で合成する**

ここは**同じセッション内で 2 回判断を間違えた**箇所なので、経緯ごと残す。

1. 当初: 行・fold・分割の 3 つの `nanmax`（保守的だが原理的でない）
2. 第7ラウンド前半: 「分割は 1 段上の不確実性だから下位を含む」と考え、
   分割が揃えばそれだけを採る形に変更 —— **これが誤りだった**
3. 第7ラウンド後半（DS 指摘 + 検算）: **全分割が同じ行集合を使う**ので、
   行由来の誤差は全分割に共通に乗り、**分割間分散から相殺されて消える**。
   したがって分割を増やしても行由来の不確実性は減らない

モンテカルロ実測（真の効果ゼロ、σ_row=0.0024 / σ_split=0.0034、2σ 判定の偽陽性率）:

| 分割数 | 分割のみ（②の設計） | 二乗和（正しい設計） |
|---|---|---|
| 3 | 33.2% | 6.4% |
| 5 | 34.5% | 4.9% |
| 10 | 43.9% | 4.8% |
| 20 | **55.0%** | 4.8% |

**②の設計では、分割を増やすほど偽陽性が増える。** 「より丁寧に測ったつもり」が
逆方向に働く最悪の形だった。正しくは `hypot(max(行, fold), 分割)`。

**教訓**: 「上位の不確実性が下位を含む」が成り立つのは、**上位を引き直すたびに下位も
引き直される**場合だけ。ここでは分割を変えてもデータは同じなので成り立たない。

**恒久対応**: 判定ロジックを `build_verdict()` に切り出した。
main() に埋めたままだと**判定の正しさをソースの字面でしか検査できない**（L-30 の再発を、
このラウンドでも 1 回やった）。関数にすれば「同じ入力でモードが変われば結論が変わる」を
振る舞いで検証できる。

### L-37 過補正は過小補正と同じくらい危険 —— 「直した」の検証まで含めて 1 サイクル

*対応する指針: `G-NOISE` / `G-PERSIST` / `G-MECH`*

**教訓 (第7ラウンド・修正後の再検証)**: L-36 で「床が甘い」を直したが、
**再検証で真逆の失敗に振り切れていた**ことが判明した。

| 実装 | 偽陽性（μ=0, m=3） | 検出力（μ=+0.008, m=3） |
|---|---|---|
| 分割のみ × 2σ（L-36 以前） | **33.2%** | 57.8% |
| hypot × t(m−1)（L-36 の修正） | 0.0% | **5.7%** |
| hypot × Welch の有効自由度（正） | 5.5% | 62.7% |

原因は「合成した SE に**小さい方の自由度**を当てた」こと。支配的な `se_rows` は
400 回のブートストラップ由来で自由度は実質無限大なのに、`df = m−1` を全体に当てると
精度の高い成分まで不確かと見なす二重の罰になる。Welch–Satterthwaite の有効自由度を使う。

**推奨手順が `--n-repeats 3` なのに、その手順で AUC +0.008 の改善が
検出される確率が 5.7%** だった。`G-PERSIST`「実在する改善を体系的に切り捨てるな」の直撃。

**同じラウンドで見つかった「直したつもり」が 3 件**:

| 件 | 何が起きていたか |
|---|---|
| `pos_rate` を API に追加 | **本番の唯一の呼び出し元が渡していない**（床は半々仮定のまま） |
| `min_detectable_difference(df=)` を追加 | **`verdict()` に届いておらず**、毎回表示される診断は正規 2σ のまま |
| Optuna の categorical を tuple に | **要素がコンテナである限り警告は消えない**。実際に出続けていた |

**恒久対応**:
1. **修正を入れたら、同じ手法でもう一度測る。** 「偽陽性を下げた」だけでなく
   「検出力を落としていないか」を必ず対で測る（片方だけ見ると必ず振り切れる）
2. **API を足したら呼び出し元を同じコミットで直す**（L-34 の再発。3 件も出た）
3. **「警告が消えた」は目視でなく機械で確かめる** —— `warnings.simplefilter("error")` を
   テストに入れて、消えたことを固定する
4. 文書の警告だけに頼らない。旧 TE API は `is_fold_subset=True` の**明示的な自己申告**を
   必須にした（`G-MECH`: 呼び出し側の記憶に任せない）


### L-38 リークを塞ぐと学習データが減る —— 塞いだ分は取り返す

*対応する指針: `G-OOF` / `G-FAIR`*

**教訓 (第7ラウンド)**: 「検証 fold で early stopping すると OOF が +0.00467 楽観になる」
（L-29）を塞ぐため、学習 fold の内側から 15% を切り出す方式に変えた。
**それは正しかったが、副作用を測っていなかった。**

内側 15% を抜くと、最終モデルは全データの **0.8 × 0.85 = 68%** でしか学習していない。
本数が決まった後に**学習 fold 100%（80%）で本数固定の再学習**をすれば取り戻せる。

実測（合成データ・8 seed の**対応比較**・LightGBM）:

| 条件 | Δ(refit − inner) |
|---|---|
| n=8,000 | **+0.00122 ± 0.00049（z=+2.48）** |
| 内側の取り分 10% | +0.00128 |
| 内側の取り分 15% | +0.00202 |
| 内側の取り分 25% | +0.00261 |

**抜く量が増えるほど効果も増える**（機構的に整合）。学習時間は約 1.7 倍。

前コンペの「LB に現れる床」が 0.00013、Public 1 位と 645 位の差が 0.00024 だったことを
思えば、**+0.0012 はその帯では大きい**。既定を `inner_refit` にし、
速さが要るスクリーニングでは `--early-stopping inner` に落とせるようにした。

**あわせて直したもの**: 内側分割の seed が `RANDOM_STATE` 固定だったため、
multi-seed avg で seed を振っても **ES 用の 15% が毎回同じ行**だった。
seed 由来の多様性がその分だけ出ていなかったので、モデル seed に追従させた。

**教訓の形**: **リークを塞ぐ変更は、たいてい情報も減らす。**
塞いだ直後に「減った分をどう取り返すか」を必ず問う。
最初の測定（3 seed）では +0.00046 ± 0.00208 で「効果なし」に見えたが、
seed を 8 に増やすと z=+2.48 になった —— **床を測る道具を自分で持っているのに、
最初はそれを使わずに結論を出しかけた**（`G-NOISE` は自分の実験にも適用する）。
