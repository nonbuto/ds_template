"""評価指標と CV 分割器を `src/config.py` の設定から 1 箇所で決める。

**なぜこのモジュールがあるか**: 以前は `scripts/train.py` と `scripts/optimize_hp.py` が
**それぞれ独立に**評価指標を呼んでいた。`FEATURES` は train から import して共有していたのに、
指標だけ共有していなかった。片方の指標を変えてもう片方を変え忘れると、
**HP 最適化が学習とは別の指標を最適化し、誰も気づかない**（静かに壊れる）。

また `EVAL_METRIC` / `PROBLEM_TYPE` / `CV_STRATEGY` は `/ds-kickoff` が記録するのに
**コード側は一度も読んでいなかった**。設定と実装が乖離していた。

使い方:
    from src.metrics import get_metric, get_cv, needs_proba

    metric = get_metric()              # (y_true, y_pred) -> float。大きいほど良い向きに揃える
    cv = get_cv()                      # 設定に応じた分割器
    score = metric(y_val, pred)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.config import CV_STRATEGY, EVAL_METRIC, N_SPLITS, PROBLEM_TYPE, RANDOM_STATE

# 指標名 → (関数, 確率が必要か, 大きいほど良いか)
_METRICS: dict[str, tuple[Callable, bool, bool]] = {}


def _register() -> None:
    from sklearn import metrics as skm

    _METRICS.update({
        # 分類（確率を受け取る）
        "auc": (skm.roc_auc_score, True, True),
        "logloss": (skm.log_loss, True, False),
        # 分類（ラベルを受け取る）
        "accuracy": (skm.accuracy_score, False, True),
        "balanced_accuracy": (skm.balanced_accuracy_score, False, True),
        "f1": (skm.f1_score, False, True),
        # 回帰
        "rmse": (lambda y, p: float(np.sqrt(skm.mean_squared_error(y, p))), False, False),
        "mae": (skm.mean_absolute_error, False, False),
        "r2": (skm.r2_score, False, True),
    })


_register()


def needs_proba(metric_name: str | None = None) -> bool:
    """この指標は確率（predict_proba の出力）を必要とするか。"""
    name = (metric_name or EVAL_METRIC).lower()
    _assert_known(name)
    return _METRICS[name][1]


def greater_is_better(metric_name: str | None = None) -> bool:
    """大きいほど良い指標か（Optuna の direction や改善判定の符号に使う）。"""
    name = (metric_name or EVAL_METRIC).lower()
    _assert_known(name)
    return _METRICS[name][2]


def _assert_known(name: str) -> None:
    if name not in _METRICS:
        raise ValueError(
            f"EVAL_METRIC='{name}' は未対応です。src/metrics.py の _METRICS に追加するか、"
            f"次から選んでください: {sorted(_METRICS)}"
        )


def get_metric(metric_name: str | None = None) -> Callable[[np.ndarray, np.ndarray], float]:
    """設定された評価指標の関数を返す。

    返る関数は `(y_true, y_pred) -> float` で、**素の指標値**をそのまま返す
    （RMSE なら小さいほど良いまま）。改善の向きが要る場面では `greater_is_better()` を使う。

    **多クラス × 確率指標（auc）は `multi_class="ovr"` を自動で補う。**
    ここを呼び出し側に任せると、train.py と optimize_hp.py で扱いがずれる余地が残る
    （このモジュールを作った動機そのもの）。
    """
    name = (metric_name or EVAL_METRIC).lower()
    _assert_known(name)
    fn = _METRICS[name][0]

    def scorer(y_true, y_pred) -> float:
        pred = np.asarray(y_pred)
        # 多クラスの確率行列（2 列超）を渡された確率指標は、多クラス用の引数が要る
        if name == "auc" and pred.ndim == 2 and pred.shape[1] > 2:
            return float(fn(y_true, pred, multi_class="ovr", average="macro"))
        if name == "f1" and len(np.unique(y_true)) > 2:
            return float(fn(y_true, pred, average="macro"))
        return float(fn(y_true, pred))

    scorer.__name__ = f"metric_{name}"
    return scorer


def get_cv(n_splits: int | None = None, strategy: str | None = None, shuffle: bool = True):
    """設定された CV 分割器を返す。

    `StratifiedKFold` / `KFold` / `TimeSeriesSplit` / `GroupKFold` /
    `StratifiedGroupKFold` に対応する。**時系列とグループは `shuffle` を使わない**
    （順序やグループを壊すと leakage になる）。
    """
    from sklearn import model_selection as ms

    name = (strategy or CV_STRATEGY)
    k = n_splits or N_SPLITS
    if name == "StratifiedKFold":
        return ms.StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=RANDOM_STATE if shuffle else None)
    if name == "KFold":
        return ms.KFold(n_splits=k, shuffle=shuffle, random_state=RANDOM_STATE if shuffle else None)
    if name == "TimeSeriesSplit":
        return ms.TimeSeriesSplit(n_splits=k)
    if name == "GroupKFold":
        return ms.GroupKFold(n_splits=k)
    if name == "StratifiedGroupKFold":
        return ms.StratifiedGroupKFold(n_splits=k, shuffle=shuffle,
                                       random_state=RANDOM_STATE if shuffle else None)
    raise ValueError(
        f"CV_STRATEGY='{name}' は未対応です。次から選んでください: "
        "StratifiedKFold / KFold / TimeSeriesSplit / GroupKFold / StratifiedGroupKFold"
    )


def is_regression() -> bool:
    """回帰タスクか（`PROBLEM_TYPE` から判定する）。"""
    return PROBLEM_TYPE == "regression"


def describe() -> str:
    """現在の設定を 1 行で返す（実験開始時のログ用）。"""
    direction = "大きいほど良い" if greater_is_better() else "小さいほど良い"
    return (f"{PROBLEM_TYPE} / 指標={EVAL_METRIC}（{direction}）/ "
            f"CV={CV_STRATEGY} {N_SPLITS}-fold / seed={RANDOM_STATE}")
