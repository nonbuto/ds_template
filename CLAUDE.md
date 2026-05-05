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
- `/kaggle-submit` 実行時 → LBスコア・OOF-LB乖離・**本日の提出数（例: 3/10）**・学び・次の方向性を記録

**SESSION.md の「現在の主要スコア」テーブルの形式:**

| 指標 | OOF tuned | LB | OOF-LB乖離 | 実験ID |
|---|---|---|---|---|
| ベスト | 0.XXXXX | 0.XXXXX | ±0.XXXXX | expNNN |

乖離列を常に記録することで「OOFは高いがLBで悪化」のパターンを早期に検知できる。

**新しいセッション開始時は必ず `/resume` を実行する。**

**SESSION.md の設計原則（蓄積禁止・上書き原則）:**

SESSION.md は「今どこにいるか」を1画面で示すライブダッシュボード。
アペンド（追記）ではなく、各セクションを必ず **上書き更新** する。

禁止パターン（AIが絶対にやってはいけないこと）:
- 「最後に完了したこと」セクションを複数回追記する（古いものは削除 → 1件だけ保持）
- 複数のスコアテーブルを並存させる（常に最新ベストのみ1テーブルを保持）
- 過去セッションの履歴を蓄積する（git history に残るので SESSION.md には不要）

SESSION.md の固定構成（このセクション順序を守り、追加セクションを作らない）:
1. **ファイルヘッダー** — 最終更新日時（`/new-experiment` または `/kaggle-submit` 実行のたびに更新）
2. **現在のステージ** — 1〜2行で現状を説明。「次にやること」を1行目に書く
3. **スコア状況** — ベストスコアのみ1テーブル。更新時は上書き（新テーブル追加禁止）
4. **直近の実験** — 最大10件。11件目以降は最古から削除（git log で追跡可能）
5. **次にやること** — 箇条書き最大5件
6. **未解決の問い** — ブロッカー・疑問点のみ。解決済みは削除
7. **重要な方針** — 実験を通じて確定した原則のみ

SESSION.md のオーバーフロー検知:
- **ファイルが 80 行を超えた場合**: 蓄積が起きているサイン。過去の完了済みエントリを削除して 80 行以内に収める
- `/resume` 実行時に行数を確認し、オーバーフローなら警告を出して整理する

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

6. **複数の実験候補が出たとき（Step by Step 原則）**
   → 「A・B・C を試しましょう」という提案は OK だが、ユーザーが選択する前に並行実装を開始してはならない
   → 各実験完了後、次に移る前に必ず①OOF報告 ②log.csv記録 ③git commit ④次の目的確認 を実施する
   → ユーザーが「続けて」「まとめて」と言った場合も、記録とコミットは各実験完了後に行う（省略禁止）
   → 実験実行中の「待ち時間」に次ステップの設計・実装を進めることも禁止。結果確認→ユーザーと判断→次の実験の順序を守る
   → **1実験 = 1コミット。この原則をAIが「効率化のため」にスキップすることは禁止**

7. **FEの棄却が続いたとき（飽和宣言禁止）**
   → 「飽和確定」「全方向探索済み」という断言を使わない。代わりに「現在試した方向では改善なし。別の角度を探しましょう」と表現する
   → **FE棄却3連続の時点で Kaggle Discussion・上位ノートブックの調査を行う**（「飽和」と宣言する前に他の参加者の知見を確認する）
   → Discussion 調査の手順:
     ```bash
     kaggle kernels list --competition <id> --sort-by voteCount --page-size 20
     kaggle kernels pull <author>/<slug> -p /tmp/kernels/
     kaggle competitions leaderboard -c <id> --show | head -20
     ```
   → FE棄却時は「なぜ効かなかったか」に加えて「**まだ試していない情報次元は何か**」を必ず1文で記録する
   → ステージの前後（Stage 6 → Stage 4 に戻るなど）は積極的に提案する。ステージは一方向に進むものではない

8. **探索継続姿勢（諦めない・PDCAを回し続ける）**
   → **「残り枠」「コンペ終盤」を理由に探索を縮小する判断をAI側からしない。枠の配分はユーザーが決める**
   → 棄却が続いても「棄却 = 今の角度が尽きた」であり「棄却 = 探索終了」ではない。常に「別の角度はないか」を能動的に複数提案し続ける
   → 3連続棄却後も次に試せる視点を提案する。提案が尽きた場合は「私が思いつく範囲では…。他に試したい方向はありますか？」と問いかける
   → 早期の収束宣言はユーザーの探索機会を奪う。**「まだ試せることがある」という前提で考え続けること**
   → PDCA サイクルの意識:
     - **P（計画）**: 実験前に「何を明らかにするか・成功基準・撤退基準」を言語化
     - **D（実行）**: 計画通りに実験し、結果を記録
     - **C（評価）**: OOF/LB・乖離・相関を多角的に分析。「なぜその結果になったか」をメカニズムレベルで解釈
     - **A（改善）**: 学びから「次の最も価値ある仮説」を導出してサイクルを回す
     → 棄却された仮説は「Aフェーズの情報」。蓄積するほど次の仮説が精緻になる

9. **可視化プロセスを省略しないこと（全ステージ共通）**
   → 可視化は「やれたらやる」ではなく「やらなければ次に進めない」チェックポイントとして扱う
   → 省略しがちな3つの場面:
     - **FE実装後・モデル投入前**（`/fe-hypothesis` フェーズ3）: 実装バグと特徴量の非効果を混同しないために必須
     - **学習完了後**（`/new-experiment` 完了時）: 特徴量重要度・OOF分布を確認し「何がスコアを動かしたか」を把握する
     - **提出後**（`/kaggle-submit` 後）: OOF-LB乖離が大きい場合は誤差分析（どのサンプルで外れたか）を行う
   → AI側から「可視化しましょうか？」と積極的に提案する。ユーザーが「省略」と言った場合のみスキップ可

### 合成データコンペ向けガイダンス

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
| `oof_lb_gap` | `/kaggle-submit` | OOF tuned − LB（正=OOF過大評価、負=OOF過小評価）。乖離が大きい実験は汎化リスクあり |
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
```

**アンサンブル棄却分析（「効かなかった」で終わらせない）:**

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

**FE 仮説の棄却記録には「再試行条件」を必ず書く:**

FE_HYPOTHESES.md の棄却エントリには以下を記録する:
```
- 棄却理由: なぜ効かなかったか（メカニズムレベルで）
- 再試行条件: どう変えれば効く可能性があるか（「不明」も可）
```
改良版を実装する前に、「棄却理由」が「再試行条件」で本当に解決されるかを確認してから着手する。
（例: 硬確率→棄却理由「0/1ノイズ」→再試行条件「ソフトな連続値に変換」→改良案「leaf_id + TargetEncoder」）

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

**⚠️ 提出前確認は AI の義務（省略禁止）:**

ユーザーが「提出します」「試します」と言った場合でも、**AIは必ず以下を表示してからコマンドを実行する:**

```
`sub_NNN_model_0.XXXXX_YYYYMMDD_HHMM.csv` を提出します。
本日 X/5 回目の提出になります。よろしいですか？
```

- 「提出します」= ユーザーの **意思表示**。AIの **実行許可** ではない
- 対象ファイル名・本日の提出回数・残り回数を明示してから「よろしいですか？」と確認する
- ユーザーの「OK」「はい」「どうぞ」を受けてから初めて実行する
- **この確認を省略した場合はテンプレート違反とみなす**

**`/kaggle-submit` スキルが実行不可の場合（`disable-model-invocation` エラー等）:**

スキル経由が不可でも、以下のチェックリストを手動で実施してから CLI 提出する:

```
□ git status がcleanか確認
□ 提出ファイルは submission_path() 生成のものか確認
□ 提出後: kaggle competitions submissions | head -3 でLBスコアを確認
□ log.csv の submit_score 列を更新
□ log.csv の oof_lb_gap 列を計算・更新（= oof_score - submit_score）
□ SESSION.md のスコアテーブルを更新（OOF-LB乖離列を必ず記入）
□ SESSION.md に「本日の提出数（N/10）」を記録
□ git commit でLB結果を記録
```

スキルが提供するフローをAIが代替する。チェックリストの省略は禁止。

### 提出枠の管理方針

**基本方針: 残り枠は使い切る。未使用の提出枠はゼロ価値。**

`/kaggle-submit` 実行のたびに以下を確認・提示する:
- 本日の使用済み回数 / 上限（通常5回）
- コンペ締め切りまでの残り日数
- 推定残り総提出枠（本日の残り + 残り日数 × 日次上限）
- `data/output/submissions/` 内の未提出候補ファイル一覧
- **OOF-LB 乖離**（今回提出の OOF tuned − LB）を SESSION.md のスコアテーブルに追記する

> 「何をSubmitするか」は実験の進行状況と残り枠を見て毎回判断する。
> ステージごとの固定配分ではなく、「今この瞬間に最も価値のある1本」を選ぶ。

**最終選択（Final Submission Selection）の 2 本ルール:**

Kaggle の最終選択 2 本は「両方 Public 最高を狙う」より「ヘッジ」を意識する。
Public LB は全テストの 30% 程度で評価されるため、Public 最高 ≠ Private 最高になりやすい。

| 枠 | 選び方 |
|---|---|
| **1 本目** | Public LB ベスト（最終段階で Public が最も高い実験） |
| **2 本目** | 外部知見系の「安定ピーク」実験（異なる補正セット複数が収束した帯域の代表。Public は 1 本目より低くてよい） |

- 2 本目を Public ベストより低い実験にするのは「過適合ヘッジ」であり弱腰ではない
- 外部知見系の安定ピークの見極め方は STEP 7「外部知見系の安定ピーク検知」を参照
- OOF 変化なしで Public だけ伸び続けた実験群の中では、最も早くピーク帯に達した実験が安定版候補
- **確保のタイミング**: 安定ピーク確認と同時に「2 本目候補」をメモしておく。終盤に判断すると Public 最高への執着で見逃しやすい

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

**TODO_TEMPLATE.md → CLAUDE.md 反映サイクル（重要）:**

TODO_TEMPLATE.md への記録は「改善の予約」に過ぎない。**記録したことを CLAUDE.md に反映して初めて完了**。

反映タイミング:
- コンペの区切り（LB新ベスト更新・ステージ移行・セッション開始時の `/resume`）
- TODO_TEMPLATE.md に TODO 項目が3件以上蓄積したとき
- ユーザーから「振り返りをしましょう」と指示されたとき

反映手順:
1. TODO_TEMPLATE.md の `TODO` 項目を読む
2. CLAUDE.md の対応セクションに実際に追記・修正する
3. TODO_TEMPLATE.md の `状態` を `DONE` に更新する
4. `IN PROGRESS` 項目も同様に処理する

**「TODO_TEMPLATE.mdに書いた」= 改善完了ではない。CLAUDE.mdが更新されるまで未完了。**

mainマージ前チェックリスト:
- [ ] コンペ名・ターゲット列のハードコードを `src/config.py` の変数に置換
- [ ] 回帰・分類の両方に対応（またはどちらか明記）
- [ ] 新依存関係を `pyproject.toml` に追加済み
- [ ] カスタマイズ箇所を `# TODO:` コメントで明示
