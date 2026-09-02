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

from src.config import (CV_STRATEGY, EVAL_METRIC, GROUP_COL, N_SPLITS, PROBLEM_TYPE,
                        RANDOM_STATE)

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


def get_cv(n_splits: int | None = None, strategy: str | None = None,
           shuffle: bool = True, seed: int | None = None):
    """設定された CV 分割器を返す。

    `StratifiedKFold` / `KFold` / `TimeSeriesSplit` / `GroupKFold` /
    `StratifiedGroupKFold` に対応する。**時系列とグループは `shuffle` を使わない**
    （順序やグループを壊すと leakage になる）。

    `seed` は**分割そのものの seed**。省略すると `RANDOM_STATE` を使う。
    multi-seed 検証では「モデルの seed」だけを振って分割を固定するのが既定だが、
    **分割の引き直し**（`G-DIAG` の「fold 間 std より小さい差は測れていない」への対処）を
    したいときは、ここに別の seed を渡す。以前は引数が無く、分割の bagging ができなかった。
    """
    from sklearn import model_selection as ms

    name = (strategy or CV_STRATEGY)
    k = n_splits or N_SPLITS
    rs = RANDOM_STATE if seed is None else seed
    if name == "StratifiedKFold":
        return ms.StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=rs if shuffle else None)
    if name == "KFold":
        return ms.KFold(n_splits=k, shuffle=shuffle, random_state=rs if shuffle else None)
    if name == "TimeSeriesSplit":
        return ms.TimeSeriesSplit(n_splits=k)
    if name == "GroupKFold":
        return ms.GroupKFold(n_splits=k)
    if name == "StratifiedGroupKFold":
        return ms.StratifiedGroupKFold(n_splits=k, shuffle=shuffle,
                                       random_state=rs if shuffle else None)
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


def shape_for_metric(pred: np.ndarray, metric_name: str | None = None) -> np.ndarray:
    """モデルの出力を、評価指標が受け取れる形に整える。

    **なぜ関数にするか**: この三項演算子は `train.py` / `optimize_hp.py` /
    `feature_study.py` / 実験雛形など **6 箇所に写経されていた**。写経は必ずずれる ——
    片方だけ「二値なら陽性確率」を入れ忘れれば、`roc_auc_score` に (n,2) が渡って
    例外か誤ったスコアになる。このモジュールを作った動機（指標の定義元を 1 つにする）と同じ理由で、
    **形の整え方も 1 箇所**に置く。

    - 確率が要る指標 × 二値 (n,2) → 陽性列
    - 確率が要る指標 × 多クラス (n,k) → そのまま
    - ラベル指標 × 確率行列 → argmax
    - 回帰（1 次元） → そのまま
    """
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred
    if needs_proba(metric_name):
        return pred[:, 1] if pred.shape[1] == 2 else pred
    return np.argmax(pred, axis=1)


def n_classes(y: np.ndarray | None = None) -> int:
    """クラス数を返す。回帰では 1。

    `y` を渡せば**実データから数える**（設定と実データの食い違いを防ぐ）。
    渡さない場合は `PROBLEM_TYPE` から推定する —— `multiclass` は実データ無しでは
    決まらないので、その場合は `y` が必須。
    """
    if y is not None:
        return 1 if is_regression() else int(len(np.unique(np.asarray(y))))
    if is_regression():
        return 1
    if PROBLEM_TYPE == "binary_classification":
        return 2
    raise ValueError(
        "PROBLEM_TYPE='multiclass' のクラス数は実データからしか決まりません。"
        "n_classes(y) の形で y を渡してください。"
    )


def needs_groups(strategy: str | None = None) -> bool:
    """この CV 戦略は `groups` を要求するか。"""
    return (strategy or CV_STRATEGY) in ("GroupKFold", "StratifiedGroupKFold")


def get_groups(df, strategy: str | None = None):
    """`cv.split(X, y, groups=...)` に渡す groups を取り出す。要らない戦略では None。

    **なぜ関数にするか**: `GroupKFold` / `StratifiedGroupKFold` は `get_cv()` から返せても、
    呼び出し側が `groups` を渡さなければ `ValueError: The 'groups' parameter should not be None`
    で必ず落ちる。**設定として選べるのに一度も使えない**状態だった。
    グループ列は `src/config.py` の `GROUP_COL` が定義元。

        cv, groups = get_cv(), get_groups(train)
        for tr, va in cv.split(X, y, groups=groups):
            ...
    """
    if not needs_groups(strategy):
        return None
    if not GROUP_COL:
        raise ValueError(
            f"CV_STRATEGY='{strategy or CV_STRATEGY}' はグループ列を必要とします。"
            "src/config.py の GROUP_COL に列名を設定してください。"
        )
    if GROUP_COL not in df.columns:
        raise ValueError(f"GROUP_COL='{GROUP_COL}' がデータに見つかりません: {list(df.columns)[:10]}...")
    return df[GROUP_COL].to_numpy()
