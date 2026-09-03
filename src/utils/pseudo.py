"""pseudo-labeling を、リークを作らない形で行う。

**なぜこのモジュールがあるか**: `PLAYBOOK.md` にレシピは書かれていたが `src/` に実装が無く、
毎回手書きだった。pseudo-labeling は**手書きすると必ず同じ 2 つの穴**にはまる:

1. **fold をまたいだ汚染** —— pseudo ラベルを「全 train で学習したモデル」で作ると、
   その予測は検証 fold の情報を含む。それを学習に足せば OOF は必ず良く出る。
   **fold 内で完結させる**のが唯一の正解（`make_fold_pseudo`）。
2. **自己蒸留** —— 自分の予測を自分に教え込むだけで、新しい情報は増えない。
   確信度で絞ると「もともと正しく解けている行」だけが残るので、なおさら情報が増えない。
   **効くとすれば test の分布を学習側に持ち込む効果**なので、
   train/test に分布差があるかを先に確認する（`scripts/av_check.py`）。

前コンペでは pseudo の寄与を「ゼロ」と判定していたが、追試で
**プールへの取り込み自体が正規表現の除外で行われておらず、1 本も入っていなかった**
ことが判明した（修正後 z=+4.21 の改善）。**「効かなかった」の前に「実行されていたか」を見る。**

使い方:

    from src.utils.pseudo import make_fold_pseudo

    for tr_idx, va_idx in cv.split(X, y):
        X_aug, y_aug, w_aug = make_fold_pseudo(
            X.iloc[tr_idx], y.iloc[tr_idx], X_test, train_one_fold, threshold=0.95)
        model = fit(X_aug, y_aug, sample_weight=w_aug)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.95
DEFAULT_WEIGHT = 0.5


def _confidence_and_labels(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """予測から (確信度, ラベル) を取り出す。二値の陽性確率 1 次元にも対応する。"""
    proba = np.asarray(proba)
    if proba.ndim == 1:                      # 二値で陽性確率だけ渡された場合
        return np.maximum(proba, 1 - proba), (proba >= 0.5).astype(int)
    return proba.max(axis=1), proba.argmax(axis=1)


def select_confident(proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
                     max_n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """確信度が閾値を超えた行と、その擬似ラベルを返す。

    Args:
        max_n: 採用する上限件数。超える場合は**確信度の高い順**に絞る。
            確信度の高い行が大量に入ると、モデルが自分の予測を強化するだけになる。

    Returns:
        `(選ばれた行の真偽値配列, 選ばれた行のラベル)`
    """
    conf, labels = _confidence_and_labels(proba)
    mask = conf >= threshold
    if max_n is not None and mask.sum() > max_n:
        idx = np.flatnonzero(mask)
        keep = idx[np.argsort(-conf[idx])[:max_n]]
        mask = np.zeros(len(conf), dtype=bool)
        mask[keep] = True
    return mask, labels[mask]


def make_fold_pseudo(
    X_tr: pd.DataFrame,
    y_tr,
    X_test: pd.DataFrame,
    train_fn: Callable,
    params: dict | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    weight: float = DEFAULT_WEIGHT,
    max_ratio: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """**この fold の学習データだけ**から擬似ラベルを作り、学習セットに足す。

    `train_fn(X_tr, y_tr, X_val, y_val, params) -> (model, val_pred)` の形の関数を使う
    （`scripts/train.py` の `train_fold_*` と同じ形）。

    Args:
        X_tr / y_tr: **この fold の学習部分だけ**。検証 fold を渡してはいけない。
        X_test: 擬似ラベルを付ける対象（通常は test）。
        threshold: この確信度を超えた行だけ採用する。
        weight: 擬似ラベル行に与える sample_weight（本物より軽くする）。
        max_ratio: 擬似ラベル行数の上限（学習行数に対する比）。
            確信度の高い行ばかりが大量に入ると、モデルが自分の予測を強化するだけになる。

    Returns:
        `(拡張した X, 拡張した y, sample_weight)`

    Note:
        **検証 fold を一切見ない**ので、この関数を fold ループの中で呼ぶ限り OOF は汚れない。
        逆に「全 train で学習したモデルで pseudo を作る」実装は、
        必ず OOF を楽観側に寄せる（そして LB では再現しない）。
    """
    from src.metrics import is_regression

    if is_regression():
        # **回帰では「確信度」が定義できない。** 予測値をそのまま確信度として扱うと、
        # 値が大きい行ほど確信度が高いという無意味な基準で選び、しかも 0/1 の
        # 擬似ラベルを連続値ターゲットに混ぜることになる（例外は出ない）。実測:
        #   予測 [-3.2, 0.4, 120.0, 0.51] → 確信度 [4.2, 0.6, 120.0, 0.51] / ラベル [0,0,1,1]
        raise ValueError(
            "make_fold_pseudo は分類専用です（回帰では確信度が定義できません）。\n"
            "   回帰で擬似ラベルを使うなら、予測値そのものを擬似ターゲットにし、"
            "sample_weight で減衰させる別の設計が要ります。"
        )

    params = dict(params or {})
    # 擬似ラベルを作るためのモデルは、この fold の学習部分だけで作る
    model, _ = train_fn(X_tr, y_tr, X_tr, y_tr, params)
    proba = (model.predict_proba(X_test) if hasattr(model, "predict_proba")
             else model.predict(X_test))

    mask, labels = select_confident(proba, threshold, max_n=int(len(X_tr) * max_ratio))

    X_aug = pd.concat([X_tr, X_test.iloc[mask]], axis=0, ignore_index=True)
    y_aug = np.concatenate([np.asarray(y_tr), labels])
    w_aug = np.concatenate([np.ones(len(X_tr)), np.full(mask.sum(), weight)])
    return X_aug, y_aug, w_aug


def describe_pseudo(mask: np.ndarray, labels: np.ndarray, n_train: int) -> str:
    """採用した擬似ラベルの内訳を 1 行で返す。

    **「効かなかった」の前に「実行されていたか」を見るための表示。**
    前コンペでは 0 件しか入っていないのに「寄与ゼロ」と結論していた。
    """
    n = int(np.asarray(mask).sum())
    if n == 0:
        return "  ⚠️ pseudo: 採用 0 件（閾値が高すぎる可能性。効果を測る前にここを確認）"
    vals, counts = np.unique(labels, return_counts=True)
    dist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
    return f"  pseudo: {n:,} 件採用（学習行の {n / max(n_train, 1):.1%}）ラベル分布 {{{dist}}}"
