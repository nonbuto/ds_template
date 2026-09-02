"""
アンサンブルユーティリティモジュール

複数モデルの予測値をブレンドするためのヘルパー関数群。
05_predict.py から呼び出す。
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def simple_average(
    predictions: list[np.ndarray],
    weights: Optional[list[float]] = None,
) -> np.ndarray:
    """単純平均（重み付き平均）でアンサンブルする。

    Args:
        predictions: 各モデルの予測値リスト（各要素はshape=(n_samples,)のndarray）
        weights: 重みリスト（Noneの場合は等重み）

    Returns:
        アンサンブル後の予測値 (shape=(n_samples,))

    Example:
        blended = simple_average([lgb_preds, cb_preds], weights=[0.6, 0.4])
    """
    preds_array = np.column_stack(predictions)  # shape: (n_samples, n_models)
    if weights is None:
        return preds_array.mean(axis=1)
    w = np.array(weights, dtype=float)
    w = w / w.sum()  # 正規化
    return (preds_array * w).sum(axis=1)


def rank_average(
    predictions: list[np.ndarray],
    weights: Optional[list[float]] = None,
) -> np.ndarray:
    """ランク平均でアンサンブルする。

    各モデルの予測値をランクに変換してから平均する。
    スケールの異なるモデルを組み合わせる際に有効。

    Args:
        predictions: 各モデルの予測値リスト
        weights: 重みリスト（Noneの場合は等重み）

    Returns:
        ランク平均後の予測値 (shape=(n_samples,), 0〜1に正規化)
    """
    n_samples = len(predictions[0])
    ranked = []
    for pred in predictions:
        ranks = pd.Series(pred).rank(method="average").values
        ranked.append(ranks / (n_samples + 1))  # 0〜1に正規化

    return simple_average(ranked, weights=weights)


def stacking_blend(
    oof_predictions: np.ndarray,
    test_predictions: np.ndarray,
    y_train: np.ndarray,
    meta_features_train: Optional[np.ndarray] = None,
    meta_features_test: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """スタッキング（メタ学習）でアンサンブルする。

    OOF予測値をメタ特徴量としてロジスティック回帰でブレンドする。

    Args:
        oof_predictions: shape=(n_samples, n_models) のOOF予測値
        test_predictions: shape=(n_test_samples, n_models) のテスト予測値
        y_train: 訓練データの正解ラベル
        meta_features_train: 追加メタ特徴量（任意, shape=(n_samples, n_features)）
        meta_features_test: テスト用追加メタ特徴量（任意）

    Returns:
        (train_meta_preds, test_meta_preds) のタプル。
        **train 側は out-of-fold で作る**（下記）。

    Note:
        以前は「全行で学習したメタモデルで同じ全行を予測」していた。これは in-sample 予測なので、
        返り値をそのままスタッキングの OOF として評価すると**必ず楽観的に出る**。
        メタモデルは重みを 2〜3 個決めるだけなので過学習は軽いが、
        `G-OVERFIT` が言うとおり天井帯では「OOF↑なのに LB↓」がこの経路でも起こる。
        ここで確保した OOF がブレンド候補の比較に使われるので、素性を揃える意味は大きい。
        メタ予測は `cross_val_predict` で fold 外から作り、test 用のモデルだけ全行で学習する。
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline

    from src.metrics import get_cv

    X_meta_train = oof_predictions
    X_meta_test = test_predictions

    if meta_features_train is not None:
        X_meta_train = np.hstack([X_meta_train, meta_features_train])
        X_meta_test = np.hstack([X_meta_test, meta_features_test])

    def _pipe():
        # スケーラも fold 内で fit する（全行で fit すると test 側の統計が漏れる）
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=1000, random_state=42))

    train_preds = cross_val_predict(_pipe(), X_meta_train, y_train,
                                    cv=get_cv(), method="predict_proba")[:, 1]

    final = _pipe().fit(X_meta_train, y_train)
    test_preds = final.predict_proba(X_meta_test)[:, 1]

    print(f"📐 スタッキング完了: {oof_predictions.shape[1]}モデルをメタ学習"
          f"（train 側は out-of-fold）")
    return train_preds, test_preds


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
    corr = float(np.corrcoef(oof_existing, oof_candidate)[0, 1])
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


def load_predictions(pred_files: list[Path]) -> list[np.ndarray]:
    """CSVまたはnpyファイルから予測値を読み込む。

    Args:
        pred_files: 予測値ファイルのパスリスト（CSV or npy）

    Returns:
        予測値のリスト
    """
    predictions = []
    for path in pred_files:
        if path.suffix == ".npy":
            pred = np.load(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            pred_col = df.columns[-1]  # 最後のカラムを予測値として使用
            pred = df[pred_col].values
        else:
            raise ValueError(f"非対応のファイル形式: {path.suffix}")
        predictions.append(pred)
        print(f"  ✅ {path.name}: shape={pred.shape}")
    return predictions
