"""「その差は測れるのか」を、表の固定値ではなく**その場で計算して**答える。

**なぜこのモジュールがあるか**: `GUIDELINES.md` の `G-NOISE` は指標ごとのノイズ床を
固定値の表で持っていた。ところが実測すると **7〜32 倍過小**だった。

| n_pos=n_neg | Hanley-McNeil の SE | 表の値 |
|---|---|---|
| 5,000 | 0.00319 | ±0.0001（**32 倍**） |
| 50,000 | 0.00101 | ±0.0001（10 倍） |

表は「Hanley-McNeil 由来」と明記していたのに、その式に代入すると 0.0032 が出る。
±0.0001 は実際には**相関 0.999 のペア差**の値で、それを単体スコアの床として掲げていた。

**ただし実データ（前回コンペ 165 提出）で突き合わせると、ずれ方は一様ではなかった**（`EmpiricalFloor` 参照）:

| 当時の閾値 | 実測床（0.00013）との比 |
|---|---|
| 表の見出し「突破 2σ = +0.0002」 | 1.5 倍（**妥当だった**） |
| 脚注「paired は 5-10x 小さい」→ 0.00002〜0.00004 | 0.22 倍（甘すぎ） |
| `feature_study` の +0.0003 | 2.2 倍（やや厳しい） |
| `G-DIAG` の `cv_val_std` ≈ 0.01 | **80 倍**（桁違いに厳しい） |

つまり問題は「全部が甘かった」ことではなく、**用途の違う 3 つの床が混在し、
そのどれもが自分の用途に合っていなかった**こと。とくに `cv_val_std` を床にしたことで、
実在する改善が体系的に「判別不能」に落とされていた。

床は 3 種類あり、**用途がまったく違う**:

- `single_score_se()` —— **1 つのスコア**が標本のゆらぎでどれだけ動くか。
  「LB 0.9710 は本当に 0.9705 より良いのか」のように、**別の観測点**と比べるときに使う。
- `paired_se()` —— **同じ行を同じ分割で予測した 2 本**の差がどれだけ動くか。
  fold の難易度や行の当たり外れが差を取った時点で相殺するので、単体の床より 1〜2 桁小さい。
  FE の採否・モデルの優劣は**必ずこちら**で判定する。
- `empirical_lb_floor()` —— **提出実績から測る「LB に現れるための床」**。
  行の再抽出では再現できないもの（分割の引き直し・分布差・Public の標本ゆらぎ）を
  すべて含むので、**「提出する価値があるか」の判断はこれが現実の壁**になる。

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
ZERO_FLOOR_EPS = 1e-7           # これ未満の SE は「潰れている」とみなし判定を出さない


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
    # **床がゼロ同然のときに「測れた」と言わない。** 記録の丸めや fold 数の不足で
    # 対応差がすべて同一になると SE が 0 近くに潰れ、z が発散して
    # 「z=+68 で改善」のような無意味な断定が出る（実際に出した）。
    if se < ZERO_FLOOR_EPS:
        return (f"床が算出できません（SE={se:.2e} ≈ 0）。fold 数を増やすか、"
                "記録の精度を確認してください")
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


# ──────────────────────────────────────────────────────────
# 実績から測る床 —— ブートストラップが取りこぼす分を含む
# ──────────────────────────────────────────────────────────

EMPIRICAL_WINDOW = 20          # 床を測る直近の提出数
EMPIRICAL_MIN_N = 8            # これ未満では床を出さない


class EmpiricalFloor:
    """`log.csv` の (OOF, LB) 実績から測った「LB に現れるための床」。

    `gap = LB − OOF` の**散らばり**が、OOF では説明できない LB の動きの大きさ。
    平均のオフセットは 5-fold OOF と全学習相当の test 予測の差で、これは無害
    （毎回同じだけ乗るので、差を取れば消える）。**閾値の情報を持つのは SD の方。**

    ブートストラップの床（`paired_se`）より優れている点:
    **実測の対象が本物**なので、行の再抽出では再現できないもの——
    CV 分割を引き直したときのばらつき、train/test の分布差、Public の標本ゆらぎ——が
    すべて含まれる。合成では作れない数字。

    前回コンペ最終盤（OOF 上位帯・n=47）での実測:

        LB = OOF + 0.00112 ± 0.00007  →  床（2σ）= 0.00013
        この帯の隣接実験の ΔOOF 中央値 = 0.000010
        床を超えた隣接ペア = **0%**

    「8 日間・32 提出で LB 更新ゼロ」は運ではなく、**検出可能な大きさの差を
    そもそも作れていなかった**という測定結果だった。

    留意: 似た提出ばかりだと散らばりは小さく出る。系統を変えたら測り直すこと。
    """

    def __init__(self, sd: float, n: int, oof_lo: float, oof_hi: float, offset: float):
        self.sd = sd
        self.n = n
        self.oof_range = (oof_lo, oof_hi)
        self.offset = offset

    @property
    def floor(self) -> float:
        """LB に現れることを期待してよい最小の ΔOOF（2σ）。"""
        return min_detectable_difference(self.sd)

    def ratio(self, delta: float) -> float:
        """その ΔOOF は床の何倍か。1 未満なら LB に出ない公算が大きい。"""
        return abs(delta) / self.floor if self.floor > 0 else float("inf")

    def __str__(self) -> str:
        return (f"LB 反映の床: {self.floor:.5f}（直近 {self.n} 提出の gap SD={self.sd:.5f}、"
                f"OOF {self.oof_range[0]:.5f}〜{self.oof_range[1]:.5f}、"
                f"オフセット {self.offset:+.5f}）")


def empirical_lb_floor(log_rows=None, window: int = EMPIRICAL_WINDOW,
                       min_n: int = EMPIRICAL_MIN_N) -> "EmpiricalFloor | None":
    """`log.csv` の実績から「LB に現れるための床」を測る。データ不足なら None。

    直近 `window` 件の (OOF, LB) が揃った提出だけを見る。**古い提出を混ぜない**のは、
    序盤の大きく違う構成を含めると散らばりが実態より大きく出るため
    （実測: 全体 SD 0.00106 に対し、最終盤に絞ると 0.00007）。
    """
    import csv as _csv

    if log_rows is None:
        from src.config import EXPERIMENTS_DIR

        path = EXPERIMENTS_DIR / "log.csv"
        if not path.exists():
            return None
        try:
            with open(path, newline="") as f:
                log_rows = list(_csv.DictReader(f))
        except OSError:
            return None

    pairs = []
    for row in log_rows:
        try:
            oof = float((row.get("oof_score") or "").strip())
            lb = float((row.get("submit_score") or "").strip())
        except (TypeError, ValueError):
            continue
        pairs.append((oof, lb))

    recent = pairs[-window:]
    if len(recent) < min_n:
        return None

    gaps = np.array([lb - oof for oof, lb in recent])
    oofs = np.array([oof for oof, _ in recent])
    sd = float(np.std(gaps, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return EmpiricalFloor(sd=sd, n=len(recent), oof_lo=float(oofs.min()),
                          oof_hi=float(oofs.max()), offset=float(np.mean(gaps)))


def expected_false_positives(n_tests: int, sigma: float = SIGMA_MULTIPLIER) -> float:
    """効果ゼロの候補だけを `n_tests` 件試したとき、閾値を超える期待件数。

    **床は 1 回の判定を守るもので、判定を繰り返すことは守らない。**
    2σ（片側 ≒ 2.3%）で 87 件試せば、**効果ゼロでも期待 2.0 件**が「採用推奨」に見える。
    前コンペの FE 仮説数がちょうど 87 件だったので、これは机上の話ではない。

        87 件を 2σ で判定 → 期待 2.0 件（少なくとも 1 件出る確率 86.6%）
        87 件を 3σ で判定 → 期待 0.12 件（同 11.1%）

    使い道は「何件かは偶然だと分かった上で読む」ことで、機械的に補正することではない
    （Bonferroni で締めると、本物の弱い改善まで落ちる）。**表示して判断材料にする。**
    """
    from math import erf, sqrt

    one_sided = 0.5 * (1.0 - erf(sigma / sqrt(2.0)))
    return float(n_tests * one_sided)


def multiple_testing_note(n_tests: int) -> str:
    """多重比較の注意を 1 行で返す（`feature_study` が毎回表示する）。"""
    if n_tests < 5:
        return ""
    return (f"多重比較: これまで {n_tests} 件を計測。**効果ゼロでも** "
            f"2σ で期待 {expected_false_positives(n_tests, 2.0):.1f} 件 / "
            f"3σ で期待 {expected_false_positives(n_tests, 3.0):.2f} 件が閾値を超えます")
