"""ハイパーパラメータ探索空間の定義。

`scripts/optimize_hp.py` から呼ばれ、最良 HP は `PARAMS_DIR/best_params_{model}_{tag}.json`
に保存される。`scripts/train.py --params <json>` で明示的に読み込む。
Stage 3（作業用 HP, 20〜30 試行）と Stage 5（本格 HP, 100 試行以上）の両方で使う。

**ここに置くのは「探索するハイパーパラメータ」だけ。**
目的関数・クラス数・評価指標・木の本数のような**タスクが決めるキーは置かない** ——
定義元は `scripts/train.build_params()` の 1 箇所。以前はここに `objective="binary"` /
`eval_metric="AUC"` が直書きされており、

- 二値前提のキーが多クラス・回帰コンペでも探索空間から入り込む
- CatBoost だけ `eval_metric="AUC"` が上書きされずに生き残り、
  **RMSE コンペでも AUC で early stopping していた**（例外は出ない）

という事故が起きていた。

パラメータ名は **sklearn ラッパーの名前に揃える**（`colsample_bytree` / `subsample` /
`random_state`）。ネイティブ名（`feature_fraction` / `bagging_fraction` / `seed`）と
混ぜると LightGBM がエイリアス衝突の警告を出し、どちらが効いたか分からなくなる。
"""

from typing import Any

import optuna

from src.config import RANDOM_STATE


def lgb_space(trial: optuna.Trial) -> dict[str, Any]:
    """LightGBM の探索空間（sklearn ラッパーの名前）。"""
    return {
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    """XGBoost の探索空間。"""
    return {
        "verbosity": 0,
        "random_state": RANDOM_STATE,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def cb_space(trial: optuna.Trial) -> dict[str, Any]:
    """CatBoost の探索空間。"""
    return {
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
    }


def nn_space(trial: optuna.Trial) -> dict[str, Any]:
    """pytabkit（RealMLP / TabM）の探索空間。

    tree 系と違い**エポック数が学習時間を直接決める**ので探索対象に入れる。
    `_nn_kind` はタスクではなくモデルの別（`build_params` が入れる）なのでここでは触らない。
    """
    return {
        "device": "cpu",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "n_epochs": trial.suggest_int("n_epochs", 32, 256, log=True),
        "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "hidden_sizes": trial.suggest_categorical(
            "hidden_sizes", [[256] * 3, [512] * 3, [256] * 4]),
        "p_drop": trial.suggest_float("p_drop", 0.0, 0.3),
    }
