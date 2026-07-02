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

**Step 1: テンプレートを Kaggle Dataset として同期する**

```bash
# ⚠️ --dir-mode zip は .kaggleignore を無視する。rsync で除外ファイルを管理すること

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

0. **上位解法のアーキテクチャ調査（前提入力）**: Stage 1.5 に入る前に `/kaggle-research` のフェーズ0を実施し、上位カーネルの主軸アーキテクチャ分布を把握する。自前の思い込みで候補を絞らず、**上位で頻出するアーキテクチャを候補に必ず含める**。
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

  **Pseudo 源泉の品質とリーク診断:**
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

---

## Final 2 候補プールとPersona 投票

> **参照元**: CLAUDE.md「提出枠の管理方針 — 最終選択の2本ルール」「AI指針 #18・#19・#20」。最終日に読む。

**Step 0: コンペ戦略軸の再確認（最初に実施）:**

`COMPETITION.md` の「コンペ戦略軸」（`/kickoff` Q7 で記録）を再掲する。
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
