from datetime import datetime
from pathlib import Path
import os

# ===== 環境検出（ローカル / Kaggle Notebook）=====
# Kaggle Notebook 環境では /kaggle/ が存在する
_KAGGLE_INPUT = Path("/kaggle/input")
IS_KAGGLE = _KAGGLE_INPUT.exists()

# プロジェクトのルートディレクトリ
if IS_KAGGLE:
    ROOT_DIR = Path("/kaggle/working")
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent

# 各種ディレクトリ
if IS_KAGGLE:
    # Kaggle Notebook: /kaggle/working/ 以下に出力する
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = _KAGGLE_INPUT          # Kaggle dataset は /kaggle/input/ 以下
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = DATA_DIR / "output"
    EXPERIMENTS_DIR = ROOT_DIR / "experiments"
else:
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = DATA_DIR / "output"
    EXPERIMENTS_DIR = ROOT_DIR / "experiments"

# output/ のサブディレクトリ（役割別に分離）
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"   # 提出CSVのみ
OOF_DIR = OUTPUT_DIR / "oof"                  # OOF / test .npy ファイル
MODELS_DIR = OUTPUT_DIR / "models"            # 学習済みモデルファイル
PARAMS_DIR = OUTPUT_DIR / "params"            # best_params JSON
PLOTS_DIR = OUTPUT_DIR / "plots"              # EDA・可視化画像（Claude が Read で読む）

# 実験スクリプトの保管先
RUNS_DIR = EXPERIMENTS_DIR / "runs"           # 実験ごとの1回限りスクリプト

# 乱数シードなどのグローバル定数
RANDOM_STATE = 42

# 必要なディレクトリが存在しない場合は作成（RAW_DATA_DIR は Kaggle 環境では作成しない）
_dirs_to_create = [PROCESSED_DATA_DIR, EXPERIMENTS_DIR,
                   SUBMISSIONS_DIR, OOF_DIR, MODELS_DIR, PARAMS_DIR, PLOTS_DIR, RUNS_DIR]
if not IS_KAGGLE:
    _dirs_to_create.append(RAW_DATA_DIR)
for _d in _dirs_to_create:
    _d.mkdir(parents=True, exist_ok=True)


def submission_path(model: str, oof_score: float, exp_id: str = "") -> Path:
    """提出ファイルパスを命名規約に従って生成する。

    規約: sub_{exp_id}_{model}_{oof_score:.5f}_{yyyymmdd_HHMM}.csv
    例:   sub_171_lgb_cb_blend_0.91777_20260331_2347.csv

    Args:
        model:     モデル・ブレンドを表す短い識別子（例: "lgb", "lgb_cb_blend"）
        oof_score: OOF AUCスコア（5桁で埋め込む）
        exp_id:    experiments/log.csv の experiment_id（空文字可）

    Returns:
        SUBMISSIONS_DIR 以下のファイルパス
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = f"sub_{exp_id}_" if exp_id else "sub_"
    name = f"{prefix}{model}_{oof_score:.5f}_{ts}.csv"
    return SUBMISSIONS_DIR / name

# ===== コンペティション設定（/kickoff スキルが更新する） =====
# TODO: /kickoff 実行時にここを更新する
COMPETITION = "your-competition-name"
TARGET_COL = "target"
PROBLEM_TYPE = "binary_classification"   # "regression" | "binary_classification" | "multiclass"
EVAL_METRIC = "auc"                       # "rmse" | "auc" | "logloss"
CV_STRATEGY = "StratifiedKFold"           # "KFold" | "StratifiedKFold" | "TimeSeriesSplit"
N_SPLITS = 5

# 実験トラッキング
MLFLOW_TRACKING_URI = str(EXPERIMENTS_DIR / "mlflow")
EXPERIMENT_NAME = f"{COMPETITION}_baseline"  # 実験ブランチごとに更新する

# Kaggle Notebook 環境情報（デバッグ用）
if IS_KAGGLE:
    _kaggle_datasets = list(_KAGGLE_INPUT.iterdir()) if _KAGGLE_INPUT.exists() else []
