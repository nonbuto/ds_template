# DS Template v3 — Kaggle Competition Workspace

Claude Code と連携して動く Kaggle コンペ用データサイエンステンプレートです。
「実験の目的を先に言語化する」「1列ずつΔAUCを計測する」「学びをサイクルとして蓄積する」という
3つの原則を、スキルとスクリプトで仕組みとして強制します。

**実践コンペ**:
- v1: s6e3（Churn Prediction）
- v2: s6e4（Irrigation Need Prediction, rank 293/4316）
- **v3: s6e5 (Predicting F1 Pit Stops)** で開発・改良

---

## v3 で追加された主な改善

s6e5 実践を経て、AI 行動規範と Final 2 選定プロセスを大幅強化:

### AI 行動規範 #10-20（CLAUDE.md）
- **#10**: コンペ初日の外部データインベントリ義務化（使う/skip/保留 の 3 択判定）
- **#11**: ドメイン知識先行プロセス（EDA より前に当然影響変数を 5-10 個列挙）
- **#12**: LB プラトー検出時の強制 brainstorm（同 LB ±0.00002 で 5 回続いたら）
- **#13**: 早期却下の禁止（FE 却下前に 3 重チェック: 可視化/関連変数/相関-importance）
- **#14**: Final 2 早期決定の禁止（残り slot ≥ 2 では Final 議論禁止）
- **#15**: 1実験1コミットの厳守（並行実行時も例外なし、OOF 判明後 5 分以内）
- **#16**: 可視化の自発的提案（数値報告時に「グラフ生成しますか？」必須）
- **#17**: Public LB 微改善の懐疑主義（評価指標別ノイズ床テーブル: AUC/Logloss/RMSE/Acc-F1）
- **#18**: OOF を Public LB と同等以上に Private LB の predictor として尊重
- **#19**: Final 2 候補プール拡張（Public Top-10 ∪ OOF Top-10）
- **#20**: 新規 FE / 外部データの「Private 過適合候補」分類

### Stage 4-6 強化
- Stage 4: AV (Adversarial Validation) 診断を標準化
- Stage 5: FE 変更時の HP retune ルール（FE ±20% 変動で再実行）
- Stage 6: 新規 STEP 8「Blend of Blends - 構造的に異なる blend の consensus」
- Stage 6 STEP 6: Multi-seed averaging のデフォルト化（n_seeds=5）

### Final 2 Selection 完全改訂
- 9 Persona 投票プロトコル（Grandmaster/Theorist/Risk/Pragmatic/Newcomer/Domain/Researcher/Reviewer/Behavioral）
- 候補プール拡張ルール（Public 過適合候補の見落としを防ぐ）
- 典型 Final 2 構成パターン A-D 明文化

### Autonomous Skill Application
- スキル呼び出しが無くても skill のフェーズプロトコルに従う義務
- 場面別の autonomous 適用テーブル

### scripts/visualize.py 拡張
- `plot_feature_importance()`: LGB/XGB importance top N
- `plot_oof_distribution()`: 新旧 OOF 比較
- `plot_correlation_matrix()`: 複数モデル OOF 相関 heatmap
- `plot_lb_history()`: LB プラトー検出用

---

## 必須要件

- [uv](https://github.com/astral-sh/uv) — パッケージ・Python バージョン管理
- [Claude Code](https://claude.ai/claude-code) — AI ペアプログラミング（スキルの実行環境）
- Kaggle API（`~/.kaggle/kaggle.json` 設定済み）

---

## 新しいコンペを始める手順

### Step 1: リポジトリをクローンしてコンペブランチを作る

```bash
git clone https://github.com/nonbuto/ds_template.git my-competition
cd my-competition
git checkout -b comp/<competition-name>   # 例: comp/s6e5-tabular
uv sync
```

### Step 2: コンペ設定を更新する

`src/config.py` の以下を書き換える:

```python
COMPETITION   = "playground-series-s6e5"   # kaggle competitions list で確認
TARGET_COL    = "target"
PROBLEM_TYPE  = "binary_classification"    # "regression" / "multiclass"
EVAL_METRIC   = "auc"
CV_STRATEGY   = "StratifiedKFold"
N_FOLDS       = 5
```

### Step 3: データをダウンロードする

```bash
uv run kaggle competitions download -c <competition-name> -p data/raw/
```

### Step 4: Claude Code でキックオフする

```
/kickoff
```

→ COMPETITION.md にデータ種別・外部データ有無・CV設計の初期判断が記録される

---

## 学習サイクル（スキルとスクリプトの使い方）

```
/resume ─────── 毎セッション開始時に必ず呼ぶ
    ↓
/kickoff ─────── コンペ参加直後に1回だけ
    ↓ データ種別・合成データ判定・外部データ確認
/new-experiment ─ 最小ベースライン（数値列のみ・デフォルトHP）
    ↓
/kaggle-submit ── CV/LB相関を確立する（以降の改善判断の基準点）
    ↓
/eda-visual ───── 「何を知りたいか」を先に言語化してから可視化
    ↓ FE仮説の種を /fe-hypothesis に登録しながら進む
Optuna 軽量 ──── 作業用HP（20〜30試行）。ΔAUCのノイズを低減する目的
    ↓
/fe-hypothesis ── 「なぜ効くか」の因果を言語化 → 実装 → 可視化確認 → ΔAUCを計測
    ↓ 必ず1列ずつ feature_study.py で投入。複数列の一括追加は禁止
Optuna フル ───── 確定した特徴量セットで100試行以上
    ↓
/kaggle-submit ── OOF/LBギャップを解釈して学びを言語化
    ↓
/new-experiment ─ 次のサイクルへ（アンサンブルへ移行 or FEに戻る）
```

> FE棄却が3連続したら → `/kaggle-research` で上位者の知見を確認してから次の仮説を立てる

---

## スクリプトの実行

```bash
# Stage 2: EDA可視化（画像を data/output/plots/ に保存 → Claude が Read で読む）
uv run python scripts/visualize.py

# Stage 1・4: CV学習
uv run python scripts/train.py --model lgb

# Stage 3: 作業用HP（FE中のΔAUCノイズ低減）
uv run python scripts/optimize_hp.py --model lgb --n-trials 25 --tag working

# Stage 4: 1列ΔCV計測（FE仮説の効果測定）
uv run python scripts/feature_study.py --new-feature <feature_name>

# Stage 5: 本格HP最適化
uv run python scripts/optimize_hp.py --model lgb --n-trials 150 --tag full

# 提出ファイル生成
uv run python scripts/predict.py --exp-id 042 --model lgb --oof-score 0.91688

# Stage 6: アンサンブル（相関確認 → Simple Blend → Greedy HC）
uv run python scripts/blend.py --mode corr   --oofs lgb=oof_042.npy cb=oof_070.npy
uv run python scripts/blend.py --mode blend  --oofs lgb=oof_042.npy cb=oof_070.npy
uv run python scripts/blend.py --mode greedy --oofs lgb=oof_042.npy cb=oof_070.npy \
    --tests lgb=test_042.npy cb=test_070.npy

# 特徴量レポート（重要度・ΔOOF棒グラフ → Claude が Read で読む）
uv run python scripts/feature_report.py
```

---

## Claude Code スキル一覧

| スキル | タイミング | 役割 |
|---|---|---|
| `/resume` | **毎セッション開始時（必須）** | SESSION.md + log.csv + FE_HYPOTHESES.md を読み「今どこにいるか」を1画面で復元 |
| `/kickoff` | コンペ参加直後（1回のみ） | データ種別・外部データ有無・CV設計の初期判断を COMPETITION.md に記録 |
| `/new-experiment` | 実験開始前 | 目的・成功基準・撤退基準を言語化してからブランチとインフラを整備 |
| `/kaggle-submit` | 提出前後 | 提出前確認 → LBスコア取得 → OOF/LB乖離分析 → 学びを log.csv に記録 |
| `/eda-visual` | Stage 2 | 「問い→可視化→発見→FE仮説の種」の対話型EDA |
| `/fe-hypothesis` | Stage 4 | FE仮説の立案・実装後可視化確認・検証・棄却理由の構造化 |
| `/kaggle-research` | FE棄却3連続後・Stage 6 外部予測活用時 | Kaggle Discussion / Dataset / Kernel を CLI で系統的に調査 |
| `/template-update` | 随時 | テンプレート改善アイデアを TODO_TEMPLATE.md に記録 |

---

## ディレクトリ構成

```
├── CLAUDE.md              # Claude Code プロジェクト設定（AI の行動ルール・ステージ定義）
├── COMPETITION.md         # コンペ固有メモ（/kickoff が生成・更新）
├── FE_HYPOTHESES.md       # FE仮説の立案・検証・棄却記録（/fe-hypothesis が管理）
├── FEATURE_REPORT.md      # 特徴量の生きたレポート（EDA・FE段階を通じて記入）
├── EDA_SUMMARY.md         # EDA対話の発見まとめ（/eda-visual が生成）
├── SESSION.md             # セッション現在地・次のアクション（/resume で参照）
├── TODO_TEMPLATE.md       # テンプレート改善タスク（/template-update が追記）
│
├── scripts/               # 汎用骨格スクリプト（コンペ開始時に TODO を埋めて使う）
│   ├── train.py           # CV学習（LGB / CB / XGB 切り替え）
│   ├── feature_study.py   # 1列ΔOOF計測（Stage 4 FE仮説の効果測定）
│   ├── optimize_hp.py     # Optuna HP最適化（Stage 3: 軽量 / Stage 5: フル）
│   ├── predict.py         # OOF・test 予測 → 提出ファイル生成
│   ├── blend.py           # アンサンブル（相関確認 / 重み最適化 / Greedy HC）
│   ├── visualize.py       # EDA可視化 → data/output/plots/ に画像保存
│   └── feature_report.py  # 特徴量重要度・ΔOOF棒グラフを画像生成
│
├── experiments/
│   ├── log.csv            # 全実験サマリー（OOF・LB・oof_lb_gap・学びを記録）
│   └── runs/              # コンペ固有の1回限りスクリプト
│       └── exp{NNN}_s{stage}_{内容}.py
│
├── src/                   # 共通ライブラリ
│   ├── config.py          # パス・コンペ設定・命名規約（submission_path() など）
│   ├── experiment.py      # 実験トラッキング（log.csv 書き込み）
│   ├── validation.py      # データバリデーション
│   ├── hp_spaces.py       # Optuna サーチスペース定義
│   ├── feature_registry.py # 特徴量レジストリ
│   └── utils/
│       ├── ensemble.py    # correlation_check / optimize_weights / greedy_ensemble
│       └── logger.py      # ロガー
│
├── data/                  # ← Git 管理外（.gitignore で除外）
│   ├── raw/               # 生データ（読み取り専用）
│   ├── processed/         # 前処理済みデータ（.pkl）
│   └── output/
│       ├── submissions/   # 提出CSV（submission_path() で命名）
│       ├── oof/           # OOF・test予測（.npy）
│       ├── models/        # 学習済みモデル
│       ├── params/        # best_params JSON
│       └── plots/         # 可視化画像（Claude が Read で読んで対話に使う）
│
└── .claude/
    ├── skills/            # Claude Code スキル（上記スキル一覧）
    └── rules/             # コーディング規約
```

---

## 設計上の主要な判断

| 判断 | 理由 |
|---|---|
| **可視化は画像ファイルで保存** | Claude Code は marimo のレンダリングを認識できない。`data/output/plots/` に `.png` を保存し、Read ツールで読んで対話する |
| **FEは1列ずつ計測** | 複数列を一度に追加すると「どれが効いたか」が分からなくなる。`feature_study.py` で1列ずつΔAUCを計測する |
| **実験の目的を先に記録** | 結果が出てから目的を決めると合理化が起きる。`/new-experiment` で「何を明らかにするか」を先に log.csv に記録する |
| **SESSION.md は上書き原則** | 履歴を追記すると80行を超えて読めなくなる。各セクションは常に最新1件だけ上書きし、詳細は git log で追跡する |
| **OOF-LB乖離を常に記録** | 「OOFは高いがLBで悪化」のパターンを早期検知するため、`oof_lb_gap` を log.csv の標準カラムとして記録する |

---

## 技術スタック

| ツール | 用途 |
|---|---|
| `uv` | パッケージ管理（pip/conda 不使用） |
| `LightGBM` | デフォルトモデル |
| `XGBoost` / `CatBoost` | アンサンブル用追加モデル |
| `Optuna` | ハイパーパラメータ最適化 |
| `matplotlib` / `seaborn` | 可視化（非インタラクティブ・画像保存） |
| `SHAP` | 特徴量重要度の説明 |
| `scikit-learn` | CV / 前処理ユーティリティ |
