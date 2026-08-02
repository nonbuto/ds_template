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

> **参照元**: CLAUDE.md「AIへの指針 #5」「学習サイクル」。Kickoff で合成データと判明したら EDA 着手前に読む。

**なぜ外部シグナルが効くか（メカニズム）:**

合成データは元データを基に統計的に生成されるが、**ターゲットとの細かい相関関係は圧縮・平滑化**される傾向がある。
元データの統計量を特徴量として注入することで、合成プロセスで失われたシグナルを補完できる。

**優先して試す3パターン（内部特徴量より先に試す価値がある）:**

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
- Tree系での外部シグナルFEは「OOF過小評価・LB浮上」の挙動を示すことがある（判断は CLAUDE.md 指針 #21 の二軸評価に従う）

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
uv run python scripts/to_kaggle_nb.py experiments/runs/exp001_s1_lgb_baseline.py \
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
uv run python scripts/to_kaggle_nb.py experiments/runs/exp001_s1_lgb_baseline.py \
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

> **参照元**: CLAUDE.md「作業ステージとゲート — Stage 1.5」「AI指針 #21・#22」。
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

4. **主軸の決定**: OOF が最高 かつ pub_oof_gap が最小 のアーキテクチャを主軸とする。両者が競合する場合は **OOF を優先**（AI 指針 #21）
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
  - **精度↔独立性のジレンマ**: 外部予測を pseudo source にすると自前モデルが外部予測の蒸留になる
    → OOF は改善するが test 予測が外部予測と 99%+ 一致し、独立シグナルを失う
    → pseudo source に外部予測を使う場合は `agree_rate = (pred_test == ext_pred).mean()` を必ず計測する
  - **差分役割モデルの選び方**: voting で「差分を解決する役割（Aux）」には「最低限の精度」と「系統的独立性」の両方が必要
    → OOF が高い Aux モデル同士を比較した場合、より高精度な Aux は主モデルと同じ予測に収束しがちで差分を解決できない（Aux 精度↑→LB↓ の逆相関が生じる）

  **外部知見系の安定ピーク検知（Public LB 過適合を防ぐ 3 シグナル）:**

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

  **自前系 vs 外部知見系の並行管理:**
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
  - Public LB +0.00000〜+0.00002 (微小、AI 指針 #17 のノイズ床近辺)
  - 統計的にはノイズ範囲内のことが多い → 必ず submit して確認

  **Private LB での挙動（重要な注意）:**
  - Public LB +1σ 改善が Private LB に **反映されないことが多い**
  - BoB の Private LB ≈ 親 blend の平均 になる場合が多い (50% 線形結合のため)
  - **BoB を Public LB ベストとして Final 1 に採用するのは AI 指針 #19/#20 違反のリスク**

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

FE_HYPOTHESES.md の棄却エントリには「棄却したアーキテクチャ」を必ず明記する:
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

FE_HYPOTHESES.md の棄却エントリには以下を記録する:
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

> **参照元**: CLAUDE.md「提出枠の管理方針 — 最終選択の2本ルール」「AI指針 #18・#19・#20」。最終日に読む。

**Step 0: コンペ戦略軸の再確認（最初に実施）:**

`COMPETITION.md` の「コンペ戦略軸」（`/ds-kickoff` Q7 で記録）を再掲する。
スコア期待値と戦略軸が対立する場合（例: 外部知見系が Public 最高だが戦略軸は「自前モデルの限界追求」）は
「スコア軸の推奨」と「戦略軸に沿った推奨」を両論併記し、**ユーザーが決定する**。
AI がスコア期待値だけで推奨を一本化しない。

**候補プール構築（Persona 投票の前に必須実施 - AI 指針 #19）:**

Public LB ベースだけのスクリーニングは Public 過適合候補を優先しがち。以下の和集合をプールに含める:

- **Public LB Top-10**: 標準的な選定基準
- **OOF Top-10**: Private LB の predictor として尊重（AI 指針 #18）
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
- ⚠️ Public +1σ 改善: AI 指針 #17 のノイズ床。「突破」と呼ばず、Private 確認まで保留扱い

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

## 天井帯での意思決定ツールキット

> **いつ読むか**: 上位候補間の差が縮まり「どれを選べばいいか分からない」状態になったとき。
> 天井帯では OOF も LB もノイズに埋もれるため、**測定できる差と測定できない差を切り分ける**ことが最優先になる。
> 対応する原則は CLAUDE.md 指針 #17 / #19 / #23-28。
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

### 手順 1: ノイズ床と量子を計算する（指針#23）

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

### 手順 3: paired bootstrap で「その差は有意か」を検定する（指針#17の実測版）

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

### 手順 4: Final 2 の 2 本目を E[max] で定量選定する（指針#19）

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

### 手順 5: 新モデル/新構造の「固有の寄与」を中間条件で分離する（指針#25）

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

### 手順 6: 天井帯に入ったら「単一最良の選定」をやめ「集約」に切り替える（指針#27）

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

### 手順 7: 情報天井を判定する（指針#28）

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
> 「異質ではあるが質の低い異質さ」だった。**異質性そのものは価値を意味しない**（指針#25）。

---

## 既知の落とし穴（ライブラリ別）

> **参照元**: CLAUDE.md 指針#30。該当ライブラリを使う前に確認する。
> 「回避できたから解決」とせず、遭遇したブロッカーをここへ追記していくこと。

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

## 教訓アーカイブ（実測値つき）

> **この章が教訓の唯一の本文（SSoT）。** `CLAUDE.md` の各指針からは `L-NN` の 1 行で参照される。
> コンペ識別子（`s6e7` 等）を書いてよいのは**この章だけ**。
>
> **なぜ数値を残すのか**: 数値は教訓の payload であり、規範を「守るべきもの」に変える。
> リファクタで最も静かに失われるため、`scripts/doc_audit.py` の C4 が消失を機械検知する。

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
- **第3世代（機械的強制）**: 「AI に守らせたい規律は、AI に守らせようとしてはいけない」。実行環境側（tracker / hook）で検知する方式に移行した

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
**指針#24（探索空間拡大の代償）の決定的証拠**になった。**この 1 提出が生んだ知見は、
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
**中核 FE の復活・指針#24 の決定的証拠・外部データの決着・分析ミスの発見**といった、
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
指針 #17 の「Public LB のノイズ床」と同型の現象が、**OOF 内部の分割方法の違いでも起きる**。
