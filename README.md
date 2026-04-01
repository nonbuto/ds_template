# Data Science Template

Kaggle等のデータサイエンスコンペティション用の分析環境テンプレートです。
パッケージマネージャーとして `uv` を使用し、通常の `.py` スクリプトでデータ処理からモデル構築までを管理します。
可視化は `matplotlib` で画像ファイルとして出力し、Claude Code が読んで対話に使います。

## 必須要件

- [uv](https://github.com/astral-sh/uv) (依存関係とPythonバージョンの管理)

## 環境構築

```bash
git clone <this-repo>
cd ds_template
uv sync
```

## 新しいコンペを始める方法

1. **`src/config.py` のコンペティション設定を更新する:**
   ```python
   COMPETITION = "your-competition-name"
   TARGET_COL = "target"
   PROBLEM_TYPE = "binary_classification"  # "regression" / "multiclass"
   EVAL_METRIC = "auc"
   CV_STRATEGY = "StratifiedKFold"
   ```

2. **Kaggleデータをダウンロードする:**
   ```bash
   uv run kaggle competitions download -c <competition-name> -p data/raw/
   ```

3. **Claude Code でキックオフする:**
   ```
   /kickoff <コンペ名>
   ```

## スクリプトの実行

```bash
# 可視化（画像を data/output/plots/ に保存 → Claude が読んで対話）
uv run python scripts/visualize.py --theme overview
uv run python scripts/visualize.py --var tenure --theme target_dist

# Stage 1: 最小ベースライン学習
uv run python scripts/train.py --model lgb

# Stage 3/5: HP最適化
uv run python scripts/optimize_hp.py --model lgb --n-trials 25 --tag working   # Stage 3
uv run python scripts/optimize_hp.py --model lgb --n-trials 150 --tag full      # Stage 5

# Stage 4: 1列ΔCV計測（FE仮説の検証）
uv run python scripts/feature_study.py --new-feature tenure_monthly_ratio

# 提出ファイル生成
uv run python scripts/predict.py --test-npy data/output/oof/test_042_lgb.npy \
    --model lgb --oof-score 0.91688 --exp-id 042

# Stage 6: アンサンブル
uv run python scripts/blend.py --mode corr --oofs lgb=oof_042.npy cb=oof_070.npy
uv run python scripts/blend.py --mode greedy \
    --oofs lgb=oof_042.npy cb=oof_070.npy \
    --tests lgb=test_042.npy cb=test_070.npy

# 特徴量レポート生成（Claude が読んで対話に使う）
uv run python scripts/feature_report.py
```

## ディレクトリ構成

```
├── CLAUDE.md              # Claude Code 用プロジェクト設定
├── COMPETITION.md         # コンペ固有メモ
├── FEATURE_REPORT.md      # 特徴量の生きたレポート（EDA・FE段階を通じて手動記入）
├── FE_HYPOTHESES.md       # FE仮説の立案・検証・棄却記録（/fe-hypothesis が管理）
├── EDA_SUMMARY.md         # EDA対話の発見まとめ（/eda-visual が生成）
├── SESSION.md             # セッション現在地（/resume で参照）
├── TODO_TEMPLATE.md       # テンプレート改善タスク
├── scripts/               # 汎用骨格スクリプト（コンペ開始時にTODOを埋めて使う）
│   ├── train.py           # CV学習（LGB/CB/XGB）
│   ├── feature_study.py   # 1列ΔOOF計測（Stage 4）
│   ├── optimize_hp.py     # Optuna HP最適化（Stage 3/5）
│   ├── predict.py         # 提出ファイル生成
│   ├── blend.py           # アンサンブル（相関確認/重み最適化/Greedy HC）
│   ├── visualize.py       # EDA可視化 → data/output/plots/ に画像保存
│   └── feature_report.py  # 特徴量レポート画像生成
├── experiments/
│   ├── log.csv            # 実験サマリー（Excel/Numbers対応）
│   └── runs/              # 実験ごとの1回限りスクリプト
│       └── exp{NNN}_s{stage}_{内容}.py
├── src/                   # 共通ライブラリ
│   ├── config.py          # 設定（パス・コンペ設定・命名規約）
│   ├── experiment.py      # 実験トラッキング（log.csv書き込み）
│   ├── validation.py      # データバリデーション
│   ├── hp_spaces.py       # Optuna サーチスペース定義
│   ├── feature_registry.py # 特徴量レジストリ
│   └── utils/
│       ├── ensemble.py    # アンサンブルユーティリティ
│       └── logger.py      # ロガー
├── data/                  # データ（Git管理外）
│   ├── raw/               # 生データ（読み取り専用）
│   ├── processed/         # 前処理済みデータ（.pkl）
│   └── output/
│       ├── submissions/   # 提出CSV（submission_path()で命名）
│       ├── oof/           # OOF/test予測（.npy）
│       ├── models/        # 学習済みモデル
│       ├── params/        # best_params JSON
│       └── plots/         # 可視化画像（Claude が Read で読む）
└── .claude/
    ├── skills/            # Claude Code スキル
    └── rules/             # コーディング規約
```

## Claude Code スキル

| スキル | タイミング | 用途 |
|---|---|---|
| `/resume` | 毎セッション開始時 | 現在地を復元（SESSION.md + log.csv を一括読み込み） |
| `/kickoff` | コンペ参加直後（1回のみ） | データ種別・外部データ・CV設計の初期判断 |
| `/new-experiment` | 実験開始前 | 目的・成功基準・撤退基準を言語化してから実装 |
| `/kaggle-submit` | 提出前後 | 提出管理・OOF/LB比較・学びの言語化 |
| `/eda-visual` | Stage 2 | 問いを持ってEDA・可視化対話・FE仮説の種を獲得 |
| `/fe-hypothesis` | Stage 4 | FE仮説の立案・検証・棄却理由の構造化 |
| `/template-update` | 随時 | テンプレート改善アイデアを記録 |

## 技術スタック

| ツール | 用途 |
|---|---|
| `uv` | パッケージ管理 |
| `matplotlib` / `seaborn` | 可視化（画像ファイル出力） |
| `LightGBM` | デフォルトモデル |
| `XGBoost` / `CatBoost` | アンサンブル用追加モデル |
| `Optuna` | ハイパーパラメータ最適化 |
| `SHAP` | 特徴量重要度の説明 |
