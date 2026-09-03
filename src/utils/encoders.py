"""カテゴリのエンコーディング。**target encoding は「使う fold の内側」で作る。**

**なぜこのモジュールがあるか**: `PLAYBOOK.md` は target encoding を「中核 FE」として
扱っているのに（前コンペでは 13 列が TE だった）、**`src/` に実装が無く毎回手書き**だった。
手書きのたびにリークの余地が生まれる —— しかも TE のリークは**エラーを出さない**。
学習時だけスコアが跳ね上がり、LB で落ちる形で現れる。

リークの正体は単純で、「その行の target を、その行の特徴量に混ぜてしまう」こと。
極端な例として**行ごとに違う値を持つ列**（ID など）を素直に target encoding すると、
その列は target そのものになる。実測（合成データ・n=2000）:

    素朴な TE（全 train で集計）: OOF AUC = 1.00000   ← 完全なリーク
    fold 外で集計した TE       : OOF AUC = 0.50000   ← 正しく「情報なし」

ただし**これだけでは足りない**。同じ「fold 外」でも、作る場所を間違えると漏れる:

**この関数は「モデルの fold ループの内側」で呼ぶ。** 前処理で 1 本作って使い回すと、
**入れ子のリーク**が残る —— 学習 fold A の行の TE は「A 以外の全 fold」で集計されており、
そこにモデルの**検証 fold B の target が入っている**。つまり学習に使う特徴量が
検証 fold の答えを含む。実測（効果ゼロのカテゴリ列・400 カテゴリ・8 seed）:

    前処理で 1 本作って使い回す : OOF AUC = 0.5194（z vs 0.5 = **+3.28**）
    fold ループの内側で作る     : OOF AUC = 0.5080（z = +1.77）

素朴な TE（全 train で集計）の 0.654 に比べれば桁違いに小さいが、
**FE の採否を決める床（0.001 前後）から見れば桁違いに大きい**。

正しい使い方 —— `add_target_encoding_in_fold()` を使う:

    for tr_idx, va_idx in cv.split(X, y):
        X_tr_te, X_va_te, X_test_te = add_target_encoding_in_fold(
            X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], X_test, ["city"])
        model.fit(X_tr_te, y[tr_idx])

低レベルの `add_target_encoding()` を直接使う場合も、**渡すのは「その fold の学習部分」**
であって train 全体ではない。

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
    """target encoding を計算して列を足す（**低レベル API**）。

    ⚠️ **`train` に渡すのは「その fold の学習部分」であって、train 全体ではない。**
    train 全体を渡して 1 本作り、それをモデルの CV で使い回すと**入れ子のリーク**になる
    （学習 fold の行の TE に、モデルの検証 fold の target が入る）。
    通常は `add_target_encoding_in_fold()` を使うこと —— そちらが正しい呼び方を強制する。

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
        **`train` 側は「その行が属さない fold の target だけ」で計算する。**
        `test` 側は `train` 全体で計算してよい（`test` の target は存在しないので漏れない）。
        多クラスでは**クラスごとに 1 列**作る（`{col}_te_c{クラス}`）。

        この「fold 外」は**渡された `train` の中での fold 外**でしかない。
        だから渡すものが「その fold の学習部分」でなければ意味を成さない。
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


def add_target_encoding_in_fold(
    X_tr: pd.DataFrame,
    y_tr,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    inner_cv=None,
    smoothing: float = DEFAULT_SMOOTHING,
    suffix: str = "_te",
):
    """**モデルの fold ループの内側**で target encoding を作る。これが正しい使い方。

    - 学習部分の行 → 学習部分の**内側 CV** で集計（自分の target を見ない）
    - 検証 fold と test → **学習部分の全体**で集計（検証 fold の target は使わない）

    こうすると、モデルが学習に使う特徴量に検証 fold の target が入る経路が無くなる。
    前処理で 1 本作って使い回すと、学習行の TE が「自分の fold 以外の全 fold」で
    作られるため、**検証 fold の target が学習側に漏れる**（モジュール冒頭の実測を参照）。

    Args:
        X_tr / y_tr: この fold の学習部分。
        X_val: この fold の検証部分。
        X_test: test（省略可）。
        columns: エンコードするカテゴリ列。
        inner_cv: 学習部分の内側で使う分割器。省略すると `get_cv()`。

    Returns:
        `(学習部分, 検証部分, test)` —— test を渡さなければ 3 つ目は None。
    """
    from src.metrics import get_cv

    if not columns:
        raise ValueError("columns を指定してください")

    inner_cv = inner_cv if inner_cv is not None else get_cv()
    X_tr = X_tr.reset_index(drop=True)
    y_tr = np.asarray(y_tr)

    # 学習部分は内側 CV で out-of-fold に、検証部分は学習部分の全体で
    tr_out, val_out = add_target_encoding(X_tr, X_val.reset_index(drop=True), columns, y_tr,
                                          cv=inner_cv, smoothing=smoothing, suffix=suffix)
    test_out = None
    if X_test is not None:
        _, test_out = add_target_encoding(X_tr, X_test.reset_index(drop=True), columns, y_tr,
                                          cv=inner_cv, smoothing=smoothing, suffix=suffix)
    return tr_out, val_out, test_out
