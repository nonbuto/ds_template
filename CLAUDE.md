# DS Template — Kaggle Competition Workspace

---

## Part 1: テンプレートの設計思想（汎用）

> この節はKaggleに限らず、あらゆるデータ分析・モデリング作業に適用される。

### 共創の原則

**AIはユーザーのドメイン知識の代わりにはならない。**

AIが担うのは「構造化・分析・記録」であり、ユーザーが担うのは「文脈・判断・ドメイン知識」。
この役割分担を守ること。AIが先に答えを出すより、ユーザーの思考を引き出す問いを優先する。

### 学習サイクルの原則

すべての実験は独立したイベントではなく、学習サイクルの一部として扱う。

```
/resume ─────────── 新しいセッション開始時に必ず呼ぶ。SESSION.md + log.csv + FE_HYPOTHESES.md を読み「今どこにいるか」を1画面で復元する
    ↓ 現在地を確認したら
/kickoff ────────── 「そのデータが何者か」を文脈から理解する（コンペ参加直後に一度だけ）
    ↓ データ種別・外部データ有無・評価指標の特性を COMPETITION.md に記録
/new-experiment ─ 最小ベースライン実験を開始（前処理不要な数値カラムのみ・デフォルトHP）
    ↓
/kaggle-submit ── LBに提出してCV/LB相関を確立する（以降のOOF判断の基準点）
    ↓ CV/LB相関が確認できたら
/eda-visual ───── 「何を知りたいか」を先に言語化する（kickoff と基準点を持ち込む）
    ↓ 仮説の種を /fe-hypothesis に登録しながら進む
Optuna 軽量 ──── 作業用HP調整（20〜30試行）。FEのΔAUC計測ノイズを低減する目的
    ↓ 作業用HPが確定したら
/fe-hypothesis ── 「なぜ効くか」の因果連鎖を言語化してから実装する
    ↓ 必ず1列ずつ scripts/feature_study.py で投入・ΔAUCを計測。複数列の一括追加は禁止
/new-experiment ─ 「何を明らかにしたいか・成功基準・撤退基準」を先に記録する
    ↓ FEが収束したら（追加特徴量のΔAUC < ±0.0003 が続いたら）
Optuna フル ───── 本格HP最適化（100試行以上）。確定した特徴量セットで実施
    ↓
/kaggle-submit ── 「OOF/LBのギャップ」を解釈し「学び」を言語化する
    ↓ 次の実験の仮説を更新する
/new-experiment ─ 次のサイクルへ（アンサンブルへ移行 or FEに戻る）
```

**サイクルを回す際のルール:**
- 実験の「目的」は実行前に記録する。結果が出てから目的を決めない
- 「OOFが上がった」は成功ではない。「汎化性能が上がったか」が問い
- 棄却された仮説は失敗ではない。「なぜ効かなかったか」が次の仮説を賢くする

### 思考の外部化の原則

**「考えたこと」は記録しなければ存在しなかったのと同じ。**

| 記録すべきもの | どこに | スキル |
|---|---|---|
| 「何を知りたいか」「ドメイン知識」 | EDA_SUMMARY.md | `/eda-visual` |
| 各変数の特性・ΔOOF・採否 | FEATURE_REPORT.md | `/eda-visual` · `/fe-hypothesis` が記入を促す |
| 特徴量の仮説・因果・棄却理由 | FE_HYPOTHESES.md | `/fe-hypothesis` |
| 実験の目的・成功基準・撤退基準 | experiments/log.csv | `/new-experiment` |
| 実験から何を学んだか | experiments/log.csv | `/kaggle-submit` |
| テンプレートへの汎用的な気づき | TODO_TEMPLATE.md | `/template-update` |
| **現在地・次のアクション・未解決の問い** | **SESSION.md** | **`/new-experiment` · `/kaggle-submit` が自動更新** |

**SESSION.md の更新タイミング（自動）:**
- `/kickoff` 実行時 → Stage 0 完了・次のアクション（最小ベースライン）を記録
- `/eda-visual` 実行時 → Stage 2 完了・次のFE仮説リストを記録
- `/fe-hypothesis` 実行時（新規） → 仮説登録・次のアクション（実装→計測）を記録
- `/new-experiment` 実行時 → 実験開始・次のアクションを記録
- `/kaggle-submit` 実行時 → LBスコア・学び・次の方向性を記録

**新しいセッション開始時は必ず `/resume` を実行する。**

### AIへの指針

以下の場面では、**実行より前に問いかけを行うこと:**

1. **ユーザーが「〜をやってほしい」と言ったとき**
   → まず「それをすることで何を明らかにしたいですか？」を問う
   → 目的が明確になってから実装する

2. **FEの提案をするとき**
   → 「この特徴量が効く理由（因果連鎖）」を説明できるものだけ提案する
   → 「試してみる価値がある」だけでは不十分

3. **次のステップを提案するとき**
   → 直前の実験から「何が分かったか」を確認してから提案する
   → 学びを踏まえない提案はしない

4. **アンサンブル・別モデル追加を提案するとき**
   → FE_HYPOTHESES.md に仮説5件以上 かつ 特徴量飽和の確認後のみ提案する
   → それ以外では「まず特徴量の探索を深めましょう」と促す

5. **合成データコンペと判明したとき（EDA着手前）**
   → 「元データ（生成元）が入手可能か」を確認する
   → 入手可能なら、特徴量探索の**最初のステップ**として外部シグナル特徴量を試す
   → 詳細は下記「合成データコンペ向けガイダンス」を参照

### 合成データコンペ向けガイダンス

**なぜ外部シグナルが効くか（メカニズム）:**

合成データは元データを基に統計的に生成されるが、**ターゲットとの細かい相関関係は圧縮・平滑化**される傾向がある。
元データの統計量を特徴量として注入することで、合成プロセスで失われたシグナルを補完できる。

**優先して試す2パターン（内部特徴量より先に試す価値がある）:**

| パターン | 内容 | 実装コスト |
|---|---|---|
| **カテゴリ別ターゲット率** | 元データの各カテゴリ列×ターゲット率をマッピング（外部ターゲットエンコーディング） | 低 |
| **数値分布特徴量** | 元データのターゲット群/非ターゲット群の分布との距離（z-score, percentile, Euclid距離） | 中 |

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

## Part 2: このプロジェクトの設定

> この節はコンペ開始時に `src/config.py` と合わせて更新する。
> コンペ固有の詳細は `COMPETITION.md` に記録する（この節は環境・規約のみ）。

### コンペ概要

> `/kickoff` 実行後、`COMPETITION.md` を参照。

### 環境・ツール

- パッケージ管理: **uv のみ**（pip・conda 禁止）
- スクリプト実行: `uv run python scripts/<script>.py`
- スクリプト実行例: `uv run python scripts/train.py --model lgb`

**marimoは使用しない。** 可視化はスクリプトから画像ファイルとして出力し、Claudeが読んで対話する。
→ 理由: Claudeはmarimoのレンダリング結果を認識できず、可視化→対話のループが回らないため。

### ディレクトリ規約

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

### スクリプト構成

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
  exp042_s4_fe_tenure.py          ← Stage 4: 特徴量追加
  exp099_s5_hp_lgb_full.py        ← Stage 5: 本格HP最適化
  exp171_s6_lgb_cb_blend.py       ← Stage 6: アンサンブル
```

- `exp{NNN}`: `experiments/log.csv` の `experiment_id` と一致させる
- `s{stage}`: どのステージの実験かが一目で分かる
- `scripts/` のスクリプトを呼び出すラッパーとして書くことを推奨

### コーディング規約

- パスは必ず `src.config` からインポート（ハードコード禁止）
- 乱数シードは `RANDOM_STATE`（`src.config`から）
- 特徴量名: snake_case・スペースなし
- `src/` 配下に型ヒントを付ける

### 提出ファイルの命名規約

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

### 実験管理（log.csv）

`experiments/log.csv` の主要カラム:

| カラム | 記録タイミング | 説明 |
|---|---|---|
| `experiment_question` | `/new-experiment` | この実験で何を明らかにしたいか |
| `success_criteria` | `/new-experiment` | どんな結果なら成功か |
| `abort_criteria` | `/new-experiment` | どんな結果なら中止するか |
| `cv_val_mean` / `oof_score` | 学習完了時 | OOFスコア |
| `submit_score` | `/kaggle-submit` | LBスコア |
| `learning` | `/kaggle-submit` | この実験から何を学んだか |

### 作業ステージとゲート

| Stage | 目的 | 完了条件 | スキル・ツール |
|---|---|---|---|
| **0. Kickoff** | データの文脈理解 | `COMPETITION.md` にデータ種別・外部データ有無・評価指標特性・CV設計の初期判断を記録済み | `/kickoff` |
| **1. 最小ベースライン** | CV/LB相関の確立 | 前処理不要な数値カラムのみ・デフォルトHPでモデルを学習し、LBに提出してCV/LB相関を確認済み。以降すべての改善はこの基準点からのΔで判断する | `/new-experiment` + `/kaggle-submit` |
| **2. EDA** | 問いとFE仮説の種を獲得 | `/eda-visual` で「問い→発見→FE仮説の種」の対話完了。合成データの場合は元データとの分布比較も含む | `/eda-visual` |
| **3. 作業用HP調整** | FE計測の安定化 | Optuna 20〜30試行でFE実験中に使う「作業用HP」を確定済み。目的は完全最適化ではなくΔAUC計測のノイズ低減 | Optuna（軽量） |
| **4. 段階的FE** | 有効な特徴量の特定 | `FE_HYPOTHESES.md` に採用・棄却含む仮説5件以上、棄却理由が分類記録済み。**特徴量は必ず1列ずつ `scripts/feature_study.py` で投入**してΔAUCを計測済み。合成データの場合は外部シグナルFEを先に検証済み | `/fe-hypothesis` + `scripts/feature_study.py` |
| **5. 本格HP最適化** | 確定特徴量での性能最大化 | Stage4の特徴量セットが確定した状態でOptuna 100試行以上を実施済み。ΔAUCの改善が±0.0002以内で収束していること | Optuna（フルサーチ） |
| **6. アンサンブル** | モデル多様性の活用 | 特徴量・HP飽和を確認済み。下記「アンサンブル探索手順」に従い実施済み | `src/utils/ensemble.py` |

**アンサンブル探索の推奨順序（Stage 6 の標準手順）:**

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
```

**アンサンブル棄却分析（「効かなかった」で終わらせない）:**

| 棄却パターン | なぜ効かなかったか | 次に試せること |
|---|---|---|
| **A: 相関 ≥ 0.998（同一予測）** | 同じ特徴量・同じCV分割・同じアルゴリズムファミリーは予測が収束する | ①異なるCV戦略（fold数・seed変更）②異なる特徴量サブセット③全く異なるアルゴリズム（NN・RF・XGB deep）を試す |
| **B: OOFは高いが blend で改善なし** | 既存モデルと同じエラーパターンを持つ（誤差の方向が同じ） | OOFスコアだけでなく**誤差の相関**を確認する（高OOFでも誤差が相関していれば多様性なし） |
| **C: Greedy HC で全モデル改善なし** | 保有モデル群の多様性が飽和している | ①FEに戻り新しいシグナルを探す ②問題の性質上アンサンブルの伸びしろが小さい可能性 |
| **D: Stacking が Simple Blend を下回る** | ベースモデルの予測が相関しすぎてメタ学習できない | ①ベースモデルの多様性を高めてから再挑戦 ②メタ特徴量に生の特徴量を追加 |

> **棄却は終わりではなく、次の探索方向を示すシグナル。**
> 各 STEP で「なぜ効かなかったか」を1文で記録してから次に進む。

> **ステージを飛ばさない。**
> - Stage 1 を省くと CV/LB乖離に気づくのが遅れる
> - Stage 3 を省くと Stage 4 のΔAUC計測がノイズに埋もれる
> - Stage 4 で `scripts/feature_study.py` を使わず複数列を一度に追加すると、どの特徴量が効いたか分からなくなる
> - Stage 5 は Stage 4 完了後でないと最適HPが変わるため意味が薄い
> - Stage 6 の STEP 1（相関確認）を省くと、実装・学習コストをかけてから「重みゼロ」と判明する

### Kaggle提出ルール

- 提出は必ず `/kaggle-submit` スキル経由（直接CLIは禁止）
- 提出前: git working tree がcleanであること
- 提出後: `submit_score`・`lb_rank`・`learning` を log.csv に記録すること

### 提出枠の管理方針

**基本方針: 残り枠は使い切る。未使用の提出枠はゼロ価値。**

`/kaggle-submit` 実行のたびに以下を確認・提示する:
- 本日の使用済み回数 / 上限（通常5回）
- コンペ締め切りまでの残り日数
- 推定残り総提出枠（本日の残り + 残り日数 × 日次上限）
- `data/output/submissions/` 内の未提出候補ファイル一覧

> 「何をSubmitするか」は実験の進行状況と残り枠を見て毎回判断する。
> ステージごとの固定配分ではなく、「今この瞬間に最も価値のある1本」を選ぶ。

### ブランチ管理

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

### コミット規約

**コミットのタイミング（3つのルール）:**

1. **学習完了直後にコミットする** — OOFスコアが判明した直後。時間を置かない
2. **1実験 = 1コミット** — 複数の変更を一度のコミットにまとめない。何が効いたか追跡できなくなる
3. **`/kaggle-submit` の前にコミット済みであること** — `git status` がcleanでなければ提出しない

**コミットメッセージの形式:**

```
feat(expNNN): <実験の目的を1文で>

OOF=<score>  model=<model>  features=<feature_set>
```

例:
```
feat(exp042): tenure×MonthlyCharges の交互作用特徴量を追加

OOF=0.91688  model=lgb  features=fe_v7_interaction
```

- `expNNN` は `experiments/log.csv` の `experiment_id` と一致させる
- 本文行（2行目）は `tracker.end_run()` が自動提案する
- `feat` / `fix` / `refactor` を使い分ける（FE追加=feat, バグ修正=fix, リファクタ=refactor）

### テンプレート改善プロトコル

コンペ作業中に改善点を発見したら `/template-update <説明>` を実行する。

スキルが「汎用プロセス / 技術インフラ / コンペ固有」を峻別して記録先を判断する。
**コンペ固有の知見をそのままテンプレートに入れない**こと。

mainマージ前チェックリスト:
- [ ] コンペ名・ターゲット列のハードコードを `src/config.py` の変数に置換
- [ ] 回帰・分類の両方に対応（またはどちらか明記）
- [ ] 新依存関係を `pyproject.toml` に追加済み
- [ ] カスタマイズ箇所を `# TODO:` コメントで明示
