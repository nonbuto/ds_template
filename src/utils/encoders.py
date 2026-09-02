"""カテゴリのエンコーディング。**target encoding は fold の外で計算する。**

**なぜこのモジュールがあるか**: `PLAYBOOK.md` は target encoding を「中核 FE」として
扱っているのに（前コンペでは 13 列が TE だった）、**`src/` に実装が無く毎回手書き**だった。
手書きのたびにリークの余地が生まれる —— しかも TE のリークは**エラーを出さない**。
学習時だけスコアが跳ね上がり、LB で落ちる形で現れる。

リークの正体は単純で、「その行の target を、その行の特徴量に混ぜてしまう」こと。
極端な例として**行ごとに違う値を持つ列**（ID など）を素直に target encoding すると、
その列は target そのものになる。実測（合成データ・n=2000）:

    素朴な TE（全 train で集計）: OOF AUC = 1.00000   ← 完全なリーク
    fold 外で集計した TE       : OOF AUC = 0.49–0.51 ← 正しく「情報なし」

使い方:

    from src.utils.encoders import add_target_encoding

    train_te, test_te = add_target_encoding(train, test, ["city", "device"], y)
    # train_te は fold 外で、test_te は train 全体で計算される

平滑化（smoothing）は `(合計 + 事前平均 × m) / (件数 + m)`。
件数の少ないカテゴリを事前平均に寄せることで、**少数カテゴリが偶然の target を
そのまま覚えてしまう**のを抑える。`m` を大きくするほど保守的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SMOOTHING = 20.0


def _smoothed_map(keys: pd.Series, target: np.ndarray, prior: float,
                  smoothing: float) -> pd.Series:
    """カテゴリ → 平滑化した target 平均の対応表。"""
    df = pd.DataFrame({"k": keys.to_numpy(), "y": target})
    agg = df.groupby("k", dropna=False)["y"].agg(["sum", "count"])
    return (agg["sum"] + prior * smoothing) / (agg["count"] + smoothing)


def add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    columns: list[str],
    y,
    cv=None,
    groups=None,
    smoothing: float = DEFAULT_SMOOTHING,
    suffix: str = "_te",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """target encoding を **fold 外で** 計算して列を足す。

    Args:
        train / test: 元のデータフレーム（**破壊しない**。コピーを返す）。
        columns: エンコードするカテゴリ列。
        y: 目的変数（分類は 0/1 または 0..k-1、回帰は連続値）。
        cv: 分割器。省略すると `src.metrics.get_cv()`（学習と同じ分割）。
        groups: `GroupKFold` 系で使うグループ。省略すると `get_groups()`。
        smoothing: 平滑化の強さ。件数の少ないカテゴリを事前平均へ寄せる。

    Returns:
        `(train + TE 列, test + TE 列)`。test は `None` を渡せば `None` が返る。

    Note:
        **train 側は「その行が属さない fold の target だけ」で計算する。**
        test 側は train 全体で計算してよい（test の target は存在しないので漏れない）。
        多クラスでは**クラスごとに 1 列**作る（`{col}_te_c{クラス}`）。
    """
    from src.metrics import get_cv, get_groups, is_regression

    y = np.asarray(y)
    cv = cv if cv is not None else get_cv()
    if groups is None:
        try:
            groups = get_groups(train)
        except ValueError:
            groups = None

    classes = [] if is_regression() else sorted(np.unique(y).tolist())
    multiclass = len(classes) > 2

    out_train = train.copy()
    out_test = test.copy() if test is not None else None

    # 「クラス」ごとに 1 本のターゲット列を作る（二値・回帰は 1 本）
    targets: list[tuple[str, np.ndarray]] = (
        [(f"c{c}", (y == c).astype(float)) for c in classes] if multiclass
        else [("", y.astype(float))]
    )

    for col in columns:
        keys_tr = train[col].astype("object")
        for tag, tgt in targets:
            name = f"{col}{suffix}" + (f"_{tag}" if tag else "")
            prior = float(tgt.mean())
            oof = np.full(len(train), np.nan)
            for tr_idx, va_idx in cv.split(train, y, groups=groups):
                # **この fold の検証行は、自分の target を一切見ない**
                mapping = _smoothed_map(keys_tr.iloc[tr_idx], tgt[tr_idx], prior, smoothing)
                oof[va_idx] = keys_tr.iloc[va_idx].map(mapping).to_numpy(dtype=float)
            # どの fold でも現れなかったカテゴリ（未知値）は事前平均
            out_train[name] = np.where(np.isnan(oof), prior, oof)

            if out_test is not None:
                mapping_full = _smoothed_map(keys_tr, tgt, prior, smoothing)
                mapped = test[col].astype("object").map(mapping_full)
                out_test[name] = mapped.fillna(prior).to_numpy(dtype=float)

    return out_train, out_test


def add_count_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    columns: list[str],
    normalize: bool = True,
    suffix: str = "_count",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """出現回数エンコーディング。**target を使わないのでリークしない。**

    件数は train と test を合わせて数える（test の情報を使うが target ではないので、
    Kaggle のルール上も統計上も問題にならない。むしろ分布シフトに強くなる）。
    """
    out_train = train.copy()
    out_test = test.copy() if test is not None else None

    for col in columns:
        both = pd.concat([train[col].astype("object"),
                          test[col].astype("object") if test is not None else pd.Series(dtype=object)],
                         ignore_index=True)
        counts = both.value_counts(dropna=False)
        if normalize:
            counts = counts / len(both)
        name = f"{col}{suffix}"
        out_train[name] = train[col].astype("object").map(counts).fillna(0).to_numpy(dtype=float)
        if out_test is not None:
            out_test[name] = test[col].astype("object").map(counts).fillna(0).to_numpy(dtype=float)

    return out_train, out_test
