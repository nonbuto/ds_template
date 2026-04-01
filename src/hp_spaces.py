"""
ハイパーパラメータサーチスペース定義モジュール

Optunaのサーチスペースを各モデルごとに定義する。
scripts/optimize_hp.py から呼び出し、PARAMS_DIR/best_params_{model}_{tag}.json に保存される。
scripts/train.py はこのJSONが存在すれば自動で読み込む（なければデフォルト値を使用）。

Stage 3（作業用HP, 20〜30試行）と Stage 5（本格HP, 100試行以上）の両方で使う。
"""

import json
from pathlib import Path
from typing import Any

import optuna

from src.config import PARAMS_DIR, RANDOM_STATE


# ===== LightGBM =====

LGB_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}


def lgb_space(trial: optuna.Trial) -> dict[str, Any]:
    """LightGBMのOptunaサーチスペース。"""
    return {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


# ===== XGBoost =====

XGB_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": RANDOM_STATE,
    "verbosity": 0,
    "device": "cpu",
}


def xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    """XGBoostのOptunaサーチスペース。"""
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "seed": RANDOM_STATE,
        "device": "cpu",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


# ===== CatBoost =====

CB_DEFAULT_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "verbose": False,
}


def cb_space(trial: optuna.Trial) -> dict[str, Any]:
    """CatBoostのOptunaサーチスペース。"""
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
    }


# ===== パラメータ保存・読み込み =====

def save_best_params(params: dict[str, Any], experiment_name: str) -> Path:
    """最適化されたパラメータをJSONファイルに保存する。

    Args:
        params: 保存するパラメータ辞書
        experiment_name: 実験名（ファイル名に使用）

    Returns:
        保存先のPathオブジェクト
    """
    path = PARAMS_DIR / f"best_params_{experiment_name}.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"💾 最適パラメータを保存: {path}")
    return path


def load_best_params(
    experiment_name: str,
    model: str = "lgb",
) -> dict[str, Any]:
    """保存済みの最適パラメータを読み込む。

    ファイルが存在しない場合はデフォルトパラメータを返す。

    Args:
        experiment_name: 実験名
        model: モデルタイプ ("lgb" | "xgb" | "cb")

    Returns:
        パラメータ辞書
    """
    path = PARAMS_DIR / f"best_params_{experiment_name}.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"📂 最適パラメータを読み込み: {path}")
        return params

    defaults = {"lgb": LGB_DEFAULT_PARAMS, "xgb": XGB_DEFAULT_PARAMS, "cb": CB_DEFAULT_PARAMS}
    default_params = defaults.get(model, LGB_DEFAULT_PARAMS)
    print(f"ℹ️  最適パラメータファイルなし。デフォルトパラメータを使用 ({model})")
    return default_params.copy()
