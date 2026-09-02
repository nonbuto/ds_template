"""提出直前の後処理。**モデルを変えずにスコアを動かせる数少ない手段。**

**なぜこのモジュールがあるか**: Playground では毎回いくつか点が動く定石なのに、
テンプレートには記述すら無かった。いずれも実装は数行だが、
**手書きすると「やったつもり」になりやすい**（適用したか、効いたかが記録に残らない）。

3 つとも**指標の性質に依存する**ので、効く場面と効かない場面をはっきりさせる:

| 手法 | 効く場面 | 効かない/害になる場面 |
|---|---|---|
| `unify_duplicates` | 同一特徴量の行が test に複数ある（合成データで頻出） | 重複がほぼ無いデータ |
| `rank_transform` | **AUC のみ**（順序しか見ないので結果が変わらないことが保証される） | 較正を見る指標（logloss 等）では**予測値が別物になる**。改善も悪化もしうるが、その指標を最適化する操作ではない |
| `clip_predictions` | RMSE・MAE で目標の取りうる範囲が既知 | 範囲を誤ると悪化。分類確率には不要 |

`apply_postprocess()` は指標を見て**適用してよいものだけ**を実行し、
何をしたかを 1 行で返す（記録に残す）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def unify_duplicates(preds: np.ndarray, features: pd.DataFrame,
                     how: str = "mean") -> tuple[np.ndarray, int]:
    """特徴量が完全に一致する行の予測を揃える。

    同じ入力に違う答えを返すのは、モデルの分散がそのまま誤差になっている状態。
    平均を取れば分散が減る分だけ期待損失が下がる（合成データでは重複が多く、効きやすい）。

    Returns:
        `(揃えた予測, 影響を受けた行数)`
    """
    preds = np.asarray(preds, dtype=float)
    keys = pd.util.hash_pandas_object(features, index=False)
    df = pd.DataFrame({"k": keys.to_numpy()})
    if preds.ndim == 1:
        df["p"] = preds
        agg = df.groupby("k")["p"].transform(how)
        out = agg.to_numpy()
    else:
        out = np.empty_like(preds)
        for j in range(preds.shape[1]):
            df["p"] = preds[:, j]
            out[:, j] = df.groupby("k")["p"].transform(how).to_numpy()
    dup = int((df.groupby("k")["k"].transform("size") > 1).sum())
    return out, dup


def rank_transform(preds: np.ndarray) -> np.ndarray:
    """予測を 0〜1 の順位に変換する。**AUC 専用。**

    順位しか残らないので、確率としての較正は失われる。
    **AUC では結果が変わらないことが保証される**（順序を保つ変換なので）のに対し、
    logloss のように値そのものを見る指標では**予測が別物になる**。
    実測では較正の良い予測でも logloss が動いた（改善する場合もある）が、
    いずれにせよ**その指標を最適化する操作ではない**ので、AUC 以外では使わない。
    `needs_proba()` だけでは判定できない（logloss も確率を要るが、順位変換は別問題）。

    複数モデルを混ぜる前に各予測を順位に揃えると、スケールの違いが消えて
    単純平均が使えるようになる（rank averaging）。
    """
    p = np.asarray(preds, dtype=float)
    if p.ndim == 1:
        return pd.Series(p).rank(method="average").to_numpy() / (len(p) + 1)
    out = np.empty_like(p)
    for j in range(p.shape[1]):
        out[:, j] = pd.Series(p[:, j]).rank(method="average").to_numpy() / (len(p) + 1)
    return out


def clip_predictions(preds: np.ndarray, lo: float | None = None,
                     hi: float | None = None, y_train=None) -> tuple[np.ndarray, int]:
    """予測を取りうる範囲に収める（回帰）。

    `y_train` を渡すとその min/max を範囲に使う。**学習データに存在しない値を
    予測しても当たらない**ので、範囲外は端に寄せる方が期待損失が下がる。
    ただし外挿が正しい問題（時系列のトレンド等）では害になるので、
    範囲の根拠を持てるときだけ使う。

    Returns:
        `(clip した予測, 変更された行数)`
    """
    p = np.asarray(preds, dtype=float)
    if y_train is not None:
        y = np.asarray(y_train, dtype=float)
        lo = float(y.min()) if lo is None else lo
        hi = float(y.max()) if hi is None else hi
    if lo is None and hi is None:
        return p, 0
    out = np.clip(p, lo, hi)
    return out, int((out != p).sum())


def apply_postprocess(
    preds: np.ndarray,
    features: pd.DataFrame | None = None,
    y_train=None,
    unify: bool = True,
    rank: bool = False,
    clip: bool = True,
) -> tuple[np.ndarray, str]:
    """指標に照らして**適用してよい後処理だけ**を実行し、内容を 1 行で返す。

    - `rank` は AUC のときだけ有効（他の指標では明示的に指定されても実行しない）
    - `clip` は回帰のときだけ有効
    - `unify` は指標に依らず安全（同じ入力に同じ答えを返すだけ）

    Returns:
        `(後処理した予測, 何をしたかの説明)`
    """
    from src.config import EVAL_METRIC
    from src.metrics import is_regression

    notes: list[str] = []
    out = np.asarray(preds, dtype=float)

    if unify and features is not None and len(features) == len(out):
        out, n_dup = unify_duplicates(out, features)
        notes.append(f"重複行の統一: {n_dup:,} 行" if n_dup else "重複行なし")

    if rank:
        if EVAL_METRIC.lower() == "auc":
            out = rank_transform(out)
            notes.append("rank 変換（AUC なので較正は不要）")
        else:
            notes.append(f"rank 変換はスキップ（EVAL_METRIC={EVAL_METRIC} は値そのものを見る指標）")

    if clip and is_regression():
        out, n_clip = clip_predictions(out, y_train=y_train)
        notes.append(f"範囲 clip: {n_clip:,} 行" if n_clip else "clip 対象なし")

    return out, " / ".join(notes) if notes else "後処理なし"
