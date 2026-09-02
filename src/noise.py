"""「その差は測れるのか」を、表の固定値ではなく**その場で計算して**答える。

**なぜこのモジュールがあるか**: `GUIDELINES.md` の `G-NOISE` は指標ごとのノイズ床を
固定値の表で持っていた。ところが実測すると **7〜32 倍過小**だった。

| n_pos=n_neg | Hanley-McNeil の SE | 表の値 |
|---|---|---|
| 5,000 | 0.00319 | ±0.0001（**32 倍**） |
| 50,000 | 0.00101 | ±0.0001（10 倍） |

表は「Hanley-McNeil 由来」と明記していたのに、その式に代入すると 0.0032 が出る。
±0.0001 は実際には**相関 0.999 のペア差**の値で、それを単体スコアの床として掲げ、
さらに脚注で「paired は 5-10x 小さい」と書いていたため、実効閾値が 10〜20 倍甘くなっていた。

**これが L-21（bagged paired bootstrap で OOF 有意だった 6 件が全部 LB に再現しなかった）を
説明する。** 有意判定の床が桁で低ければ、再現しないのは当然の帰結。

床は 2 種類あり、**用途がまったく違う**:

- `single_score_se()` —— **1 つのスコア**が標本のゆらぎでどれだけ動くか。
  「LB 0.9710 は本当に 0.9705 より良いのか」のように、**別の観測点**と比べるときに使う。
- `paired_se()` —— **同じ行を同じ分割で予測した 2 本**の差がどれだけ動くか。
  fold の難易度や行の当たり外れが差を取った時点で相殺するので、単体の床より 1〜2 桁小さい。
  FE の採否・モデルの優劣は**必ずこちら**で判定する。

使い方:

    from src.noise import paired_se, min_detectable_difference, describe_floor

    se = paired_se(y, oof_base, oof_new)          # 対応のあるブートストラップ
    if abs(delta) < 2 * se:
        ...                                        # 「測れていない」

`n` と AUC さえ分かれば計算だけで済む場面（提出前に Public の床を知りたい等）は
`single_score_se(metric_name="auc", n=..., score=...)` で解析式を使う。
"""

from __future__ import annotations

import numpy as np

from src.metrics import get_metric, needs_proba, shape_for_metric

DEFAULT_N_BOOT = 400
SIGMA_MULTIPLIER = 2.0          # 「突破」と呼ぶための倍率（2σ ≒ 95%）


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil による AUC の標準誤差（解析式）。

    ブートストラップ実測ともよく一致する（n=10,000 で式 0.00225 / 実測 0.00264）。
    """
    auc = float(np.clip(auc, 1e-6, 1 - 1e-6))
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def single_score_se(y_true=None, y_pred=None, *, metric_name: str | None = None,
                    n: int | None = None, score: float | None = None,
                    n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> float:
    """**1 つのスコア**の標本ゆらぎ（1σ）。

    予測を渡せばブートストラップで実測する。`metric_name` / `n` / `score` だけを渡した
    場合は解析式で近似する（AUC のみ。ほかは予測が要る）。

    別の観測点（LB と OOF、別コンペの水準）と比べるときの床。
    **同じ行を予測した 2 本の比較には使わない** —— そちらは `paired_se()`。
    """
    if y_true is None or y_pred is None:
        if metric_name is None or n is None or score is None:
            raise ValueError("予測を渡さない場合は metric_name / n / score が必要です")
        if metric_name.lower() != "auc":
            raise ValueError(f"解析式は auc のみ対応です（{metric_name} は予測を渡してください）")
        half = max(int(n) // 2, 1)
        return auc_se(score, half, half)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metric = get_metric(metric_name)
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, n_rows)
        if not needs_proba(metric_name) or len(np.unique(y_true[idx])) > 1:
            try:
                scores.append(metric(y_true[idx], shape_for_metric(y_pred[idx], metric_name)))
            except ValueError:
                continue
    return float(np.std(scores)) if scores else float("nan")


def paired_se(y_true, pred_a, pred_b, *, metric_name: str | None = None,
              n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> float:
    """**同じ行を予測した 2 本の差**の標準誤差（1σ）。

    行を復元抽出して「A のスコア − B のスコア」を作り直し、その標準偏差を返す。
    fold の難易度や行の当たり外れは差を取った時点で相殺するので、
    単体スコアの床より 1〜2 桁小さくなる（相関 0.999 のペアで実測 0.00011）。

    **FE の採否・モデルの優劣はこの床で判定する。** 単体の床を使うと、
    実在する改善を体系的に「測れていない」と切り捨てる。
    """
    y_true = np.asarray(y_true)
    a, b = np.asarray(pred_a), np.asarray(pred_b)
    metric = get_metric(metric_name)
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, n_rows)
        if needs_proba(metric_name) and len(np.unique(y_true[idx])) < 2:
            continue
        try:
            diffs.append(metric(y_true[idx], shape_for_metric(a[idx], metric_name))
                         - metric(y_true[idx], shape_for_metric(b[idx], metric_name)))
        except ValueError:
            continue
    return float(np.std(diffs)) if diffs else float("nan")


def fold_paired_se(scores_a, scores_b) -> float:
    """fold ごとのスコア対から、差の標準誤差を出す。

    **`cv_val_std`（fold 間ばらつき）を床に使ってはいけない。** それは
    「fold ごとの難易度の差」を主成分に含み、同じ fold で 2 つを比べれば相殺する成分。
    val std を床にすると、実在する改善を体系的に「判別不能」と切り捨てる
    （実測: val std 0.01251 に対し、正しい fold 対応差の SE は 0.00124 —— **10 倍**の差）。

    これが L-19（個別 Δ≈0 が 13 系統累積すると統計的に確定的な正の差になった）の直接の説明。
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    d = a - b
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan")
    return float(np.std(d, ddof=1) / np.sqrt(d.size))


def min_detectable_difference(se: float, sigma: float = SIGMA_MULTIPLIER) -> float:
    """「突破」と呼んでよい最小の差（既定 2σ）。"""
    return float(sigma * se)


def verdict(delta: float, se: float, sigma: float = SIGMA_MULTIPLIER) -> str:
    """効果量と床から、1 行の判定文を作る。

    **「測れていない」と「効果がない」を混同しない。** 前者は床を下げれば測れる可能性があり、
    後者は測った上で差が無い。表示ではこの 2 つを言い分ける。
    """
    if not np.isfinite(se) or se <= 0:
        return "床を推定できませんでした（n が小さい / 指標が計算できない）"
    z = delta / se
    floor = min_detectable_difference(se, sigma)
    if abs(delta) < floor:
        return (f"⬜ 測れていない（|Δ|={abs(delta):.5f} < {sigma:.0f}σ={floor:.5f}、z={z:+.2f}）"
                "—— seed / fold を増やすか集約へ")
    direction = "改善" if delta > 0 else "悪化"
    return f"{'✅' if delta > 0 else '❌'} {direction}（Δ={delta:+.5f}、{sigma:.0f}σ={floor:.5f}、z={z:+.2f}）"


def describe_floor(y_true, pred_a, pred_b=None, *, metric_name: str | None = None) -> str:
    """床の実測を 1 行で返す（実験ログ・提出前の確認用）。"""
    if pred_b is None:
        se = single_score_se(y_true, pred_a, metric_name=metric_name)
        return (f"単体スコアの床: 1σ={se:.5f} / {SIGMA_MULTIPLIER:.0f}σ="
                f"{min_detectable_difference(se):.5f}（n={len(y_true):,}）")
    se = paired_se(y_true, pred_a, pred_b, metric_name=metric_name)
    corr = float(np.corrcoef(np.asarray(pred_a).ravel(), np.asarray(pred_b).ravel())[0, 1])
    return (f"対応差の床: 1σ={se:.5f} / {SIGMA_MULTIPLIER:.0f}σ="
            f"{min_detectable_difference(se):.5f}（予測相関 {corr:.4f}）")
