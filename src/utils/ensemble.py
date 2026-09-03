"""複数モデルの予測を結合する。**使う順番は 1 本道**（選択肢を並べない）。

    1. `correlation_check()` —— 候補と既存の OOF 相関を見る。**高すぎるなら追加しない**
    2. `hillclimb()`         —— 構成を決める（Caruana: 復元あり + サブセット bagging）
    3. `signed_stack()`      —— 結合方式を上げる（符号制約なし線形。**弱い候補を引き算に使える**）

`optimize_weights()` は simplex（非負・合計 1）の重み探索で、
**hillclimb と役割が重なる**。天井帯で重み bagging（`n_seeds`）をしたいときだけ使う。
`greedy_ensemble()` は非復元・等重みの旧実装で、過去の実験を再現する目的でのみ残している
（新規は `hillclimb` を使う）。

**なぜ 1 本道にしたか**: 以前は 9 個の API があり、うち 4 個は誰からも呼ばれていなかった
（`simple_average` / `rank_average` / `stacking_blend` / `load_predictions`）。
選択肢が多いこと自体がコスト —— 次のコンペで「重みを最適化したい」と思った人が、
**どれを使うべきか判断できない**。順路を 1 つ示し、残りは理由つきで位置づける。

削除したものの代替:
    simple_average    → `np.average(preds, axis=0, weights=w)` で足りる
    rank_average      → `src.utils.postprocess.rank_transform()` で順位に揃えてから結合
    stacking_blend    → `signed_stack()`（符号制約なし・正則化を fold 外で選ぶ・回帰も可）
    load_predictions  → `np.load` / `pd.read_csv` を直接呼ぶ
"""

import numpy as np


def correlation_check(
    oof_existing: np.ndarray,
    oof_candidate: np.ndarray,
    threshold: float = 0.998,
) -> tuple[float, bool]:
    """追加候補モデルと既存モデル群の予測相関を確認する。

    相関が threshold 以上の場合、アンサンブルに追加しても重みがゼロになる可能性が高い。
    Stage 6 の STEP 1 として、実装・学習コストをかける前に必ず実行する。

    Args:
        oof_existing: 既存モデルの OOF 予測値（複数モデルの場合は平均を渡す）
        oof_candidate: 追加候補モデルの OOF 予測値
        threshold: スキップ判定の相関閾値（デフォルト 0.998）

    Returns:
        (相関係数, スキップ推奨かどうか) のタプル

    Example:
        corr, skip = correlation_check(oof_lgb, oof_cb)
        if skip:
            print("追加しても重みゼロの可能性が高い。スキップを推奨。")
    """
    a, b = np.asarray(oof_existing, dtype=float), np.asarray(oof_candidate, dtype=float)
    # **未予測行（NaN）を黙って通さない。** `np.corrcoef` は NaN を伝播し、
    # `nan < threshold` は False なので **「✅ 追加を検討可」と答えてしまう**（実測）。
    # TimeSeriesSplit の OOF には未予測行が NaN で入る（`train.py` の covered）。
    both = np.isfinite(a) & np.isfinite(b)
    if not both.all():
        if both.sum() < 2:
            raise ValueError(
                f"比較できる行がありません（有限値 {int(both.sum())} 行）。OOF を確認してください")
        print(f"  ℹ️ 未予測行 {int((~both).sum()):,} 行を除いて相関を計算します（TimeSeriesSplit 等）")
        a, b = a[both], b[both]
    # 定数列（分散ゼロ）は相関が定義できない。numpy に 0 除算させず、先に弾く
    if a.std() == 0 or b.std() == 0:
        raise ValueError(
            "相関が計算できません: 予測が定数（分散ゼロ）です。"
            "学習が失敗しているか、間違ったファイルを渡していないか確認してください"
        )
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        raise ValueError("相関が計算できませんでした。予測を確認してください")
    skip = corr >= threshold
    status = "⚠️  スキップ推奨" if skip else "✅ 追加を検討可"
    print(f"OOF相関: {corr:.4f}  ({status}, 閾値={threshold})")
    return corr, skip


def optimize_weights(
    oofs: np.ndarray,
    y: np.ndarray,
    metric_fn,
    method: str = "nelder-mead",
    n_seeds: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """複数モデルの最適ブレンド重みを探索する。

    Args:
        oofs: shape=(n_samples, n_models) の OOF 予測値行列
        y: 正解ラベル
        metric_fn: スコア計算関数（高いほど良い, 例: roc_auc_score）
        method: "nelder-mead" または "differential-evolution"
        n_seeds: **独立に探索して重みを平均する回数**（bagging）。
            2 以上にすると、初期点をランダムに振った探索を n_seeds 回まわして
            得られた重みベクトルを平均する。
        seed: 初期点の生成に使う乱数 seed。

    Returns:
        (最適重みの配列, 最適スコア) のタプル

    Note:
        `G-CEILING` の集約戦略 (a)。天井帯では**単一の最適化ランは「平坦な領域の中で
        たまたま辿り着いた 1 点」に過ぎない**。過去コンペの Private 最高は 12 シードの
        重み bagging であり、OOF 最高の 1 本を選んだ場合は天井帯でも下位だった（L-03）。
        以前はこの関数に seed も複数開始点も無く、**その戦略が実行できなかった**。
    """
    from scipy.optimize import differential_evolution, minimize

    n_models = oofs.shape[1]

    def neg_score(w_raw: np.ndarray) -> float:
        w = np.clip(w_raw, 0, 1)
        w = w / (w.sum() + 1e-8)
        return -metric_fn(y, oofs @ w)

    def _one(x0: np.ndarray, run_seed: int) -> np.ndarray:
        if method == "nelder-mead":
            result = minimize(neg_score, x0=x0, method="Nelder-Mead")
        else:
            result = differential_evolution(
                neg_score, [(0, 1)] * n_models, seed=run_seed, maxiter=500, tol=1e-8, workers=1,
            )
        w = np.clip(result.x, 0, 1)
        total = w.sum()
        return w / total if total > 0 else np.ones(n_models) / n_models

    rng = np.random.default_rng(seed)
    starts = [np.ones(n_models) / n_models]                      # 1 本目は等重み
    starts += [rng.dirichlet(np.ones(n_models)) for _ in range(max(0, n_seeds - 1))]

    weights = [_one(x0, seed + i) for i, x0 in enumerate(starts)]
    w_opt = np.mean(weights, axis=0)
    w_opt = w_opt / w_opt.sum()

    if n_seeds > 1:
        spread = float(np.std(weights, axis=0).max())
        print(f"  重み bagging: {n_seeds} 本の平均（重みのばらつき最大 {spread:.4f}）")

    best_score = metric_fn(y, oofs @ w_opt)
    return w_opt, best_score


def greedy_ensemble(
    oofs: dict[str, np.ndarray],
    tests: dict[str, np.ndarray],
    y: np.ndarray,
    metric_fn,
    higher_is_better: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Greedy Hill Climbing によるアンサンブル探索。

    保有する全 OOF ファイルを対象に、追加するたびに最もスコアが改善する
    モデルを貪欲に選択する。新規学習不要・計算コストゼロ。

    Args:
        oofs: {"モデル名": oof_array} の辞書
        tests: {"モデル名": test_array} の辞書
        y: 正解ラベル
        metric_fn: スコア計算関数（例: roc_auc_score）
        higher_is_better: スコアが高いほど良いか

    Returns:
        (選択モデル名リスト, アンサンブルOOF, アンサンブルtest予測, 最終スコア)

    Example:
        from sklearn.metrics import roc_auc_score
        selected, ens_oof, ens_test, score = greedy_ensemble(oofs, tests, y, roc_auc_score)
    """
    sign = 1 if higher_is_better else -1
    model_names = list(oofs.keys())

    # 単体スコアを計算して初期モデルを選択
    single_scores = {n: metric_fn(y, oofs[n]) for n in model_names}
    best_start = max(model_names, key=lambda n: sign * single_scores[n])

    selected = [best_start]
    ensemble_oof = oofs[best_start].copy()
    current_score = single_scores[best_start]

    print(f"Start: {best_start}  score={current_score:.5f}")
    for n, s in sorted(single_scores.items(), key=lambda x: -sign * x[1]):
        print(f"  {n:35s}: {s:.5f}")

    # Greedy 追加ループ
    print("\nGreedy Hill Climbing ...")
    for _ in range(len(model_names) - 1):
        best_gain = 0.0
        best_next = None
        n_sel = len(selected)
        for name in model_names:
            if name in selected:
                continue
            trial = (n_sel * ensemble_oof + oofs[name]) / (n_sel + 1)
            gain = sign * (metric_fn(y, trial) - current_score)
            if gain > best_gain:
                best_gain = gain
                best_next = name

        if best_next is None:
            print("  改善なし → 探索終了")
            break

        selected.append(best_next)
        n_sel = len(selected)
        ensemble_oof = ((n_sel - 1) * ensemble_oof + oofs[best_next]) / n_sel
        current_score = metric_fn(y, ensemble_oof)
        print(f"  +{best_next:35s}  score={current_score:.5f}  (Δ={sign * best_gain:+.5f})")

    # テスト予測を選択モデルで均一平均。**test 予測を渡さない使い方（OOF だけで
    # 構成を探索する）でも落ちないようにする** —— 以前は `tests[n]` が KeyError になり、
    # 探索が全部終わった最後の 1 行で例外になって結果ごと失われていた。
    missing = [n for n in selected if n not in tests]
    if missing:
        if tests:
            print(f"  ⚠️ test 予測が無いモデルがあるため test 側は作りません: {missing}")
        ensemble_test = None
    else:
        ensemble_test = np.mean([tests[n] for n in selected], axis=0)

    print(f"\n選択モデル ({len(selected)}件): {selected}")
    print(f"Greedy Ensemble score: {current_score:.5f}")
    return selected, ensemble_oof, ensemble_test, current_score


def hillclimb(
    oofs: dict[str, np.ndarray],
    y: np.ndarray,
    metric_fn,
    n_iter: int = 100,
    n_bags: int = 20,
    bag_frac: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict[str, float], np.ndarray, float]:
    """Caruana の ensemble selection（**復元ありの前進選択 + サブセット bagging**）。

    `greedy_ensemble()` との違いは 3 点で、いずれも Playground の定石:

    1. **復元あり** —— 同じモデルを何度でも選べる。選ばれた回数がそのまま重みになるので、
       等重み平均しか作れない非復元版より表現力が高い。
    2. **サブセット bagging** —— 毎回モデルの一部（`bag_frac`）だけを候補にして
       選択を繰り返し、得られた重みを平均する。**選択の過学習**を抑えるための仕掛けで、
       同一 OOF 上で数百候補から選ぶ場面（`G-OVERFIT`(a)）では必須。
    3. **改善が止まっても続ける** —— 1 回の非改善で打ち切らず `n_iter` まで回す。

    Args:
        oofs: `{名前: OOF 予測}`。1 次元（二値の陽性確率 / 回帰の予測値）。
        metric_fn: **大きいほど良い**向きに揃えた指標（`blend.py` の `_ascending_metric()`）。
        n_iter: 1 bag あたりの選択回数。
        n_bags: bag の数。1 にすると単一パスの貪欲選択になる。
        bag_frac: 各 bag で候補にするモデルの割合。

    Returns:
        `(重み辞書, アンサンブル OOF, スコア)`。重みは合計 1。
    """
    names = list(oofs)
    if not names:
        raise ValueError("oofs が空です")
    mat = np.column_stack([np.asarray(oofs[n], dtype=float) for n in names])
    rng = np.random.default_rng(seed)
    n_pick = max(2, int(round(len(names) * bag_frac)))

    counts_total = np.zeros(len(names))
    for bag in range(max(1, n_bags)):
        cand = rng.choice(len(names), size=min(n_pick, len(names)), replace=False)
        counts = np.zeros(len(names))
        current = None
        for step in range(n_iter):
            best_j, best_score = None, -np.inf
            for j in cand:
                trial = (mat[:, j] if current is None
                         else (current * counts.sum() + mat[:, j]) / (counts.sum() + 1))
                score = metric_fn(y, trial)
                if score > best_score:
                    best_j, best_score = j, score
            if best_j is None:
                break
            current = (mat[:, best_j] if current is None
                       else (current * counts.sum() + mat[:, best_j]) / (counts.sum() + 1))
            counts[best_j] += 1
        if counts.sum() > 0:
            counts_total += counts / counts.sum()

    weights = counts_total / counts_total.sum()
    ens = mat @ weights
    score = metric_fn(y, ens)
    if verbose:
        used = {names[i]: round(float(w), 4) for i, w in enumerate(weights) if w > 0}
        print(f"  hillclimb: {len(used)}/{len(names)} モデル採用（{n_bags} bag 平均）"
              f"  score={score:.5f}")
    return {names[i]: float(w) for i, w in enumerate(weights)}, ens, float(score)


def signed_stack(
    oofs: dict[str, np.ndarray],
    tests: dict[str, np.ndarray] | None,
    y: np.ndarray,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    seed: int = 42,
    verbose: bool = True,
):
    """**符号制約なし**の線形スタッキング（分類は L2 ロジスティック、回帰は Ridge）。

    `optimize_weights()` は非負かつ合計 1（simplex）に制約するので、
    **弱いメンバーを「引き算」に使う経路が構造的にふさがれている**。
    前コンペで終盤に唯一 LB で確認できた改善（OOF +0.00012 → LB +0.00017、
    累計 +0.00035 で 2σ 超え）は、まさに simplex から符号制約なしへ結合方式を
    変えたことによるものだった。**その手法がテンプレートに無かった。**

    メタ予測は `cross_val_predict` で fold 外から作る（in-sample だと必ず楽観的に出る）。
    正則化の強さは fold 外スコアで選ぶ。

    Returns:
        `(係数辞書, OOF 予測, test 予測 or None, スコア)`
    """
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.metrics import get_cv, get_metric, greater_is_better, is_regression

    names = list(oofs)
    X = np.column_stack([np.asarray(oofs[n], dtype=float) for n in names])
    metric = get_metric()
    sign = 1.0 if greater_is_better() else -1.0

    def _make(alpha: float):
        if is_regression():
            return make_pipeline(StandardScaler(), RidgeCV(alphas=[alpha]))
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0 / alpha, max_iter=2000,
                                                random_state=seed))

    best = None
    for alpha in alphas:
        pipe = _make(alpha)
        if is_regression():
            pred = cross_val_predict(pipe, X, y, cv=get_cv())
        else:
            pred = cross_val_predict(pipe, X, y, cv=get_cv(), method="predict_proba")[:, 1]
        score = sign * metric(y, pred)
        if best is None or score > best[0]:
            best = (score, alpha, pred)

    _, alpha, oof_pred = best
    final = _make(alpha).fit(X, y)
    est = final[-1]
    coefs = np.ravel(getattr(est, "coef_", np.zeros(len(names))))

    test_pred = None
    if tests and all(n in tests for n in names):
        X_test = np.column_stack([np.asarray(tests[n], dtype=float) for n in names])
        test_pred = (final.predict(X_test) if is_regression()
                     else final.predict_proba(X_test)[:, 1])

    score = metric(y, oof_pred)
    if verbose:
        neg = sum(1 for c in coefs if c < 0)
        print(f"  signed_stack: alpha={alpha}（fold 外で選択）  score={score:.5f}"
              f"  負の係数 {neg}/{len(coefs)} 本 ← simplex では表現できない部分")
    return {names[i]: float(c) for i, c in enumerate(coefs)}, oof_pred, test_pred, float(score)
