"""実験スクリプトの雛形。**これをコピーして `exp{NNN}_s{stage}_{内容}.py` にする。**

    cp experiments/runs/_TEMPLATE_exp000_s0_example.py \\
       experiments/runs/exp042_s4_fe_age_ratio.py

**なぜ雛形があるか**: 実験スクリプトをゼロから書くと `ExperimentTracker` を経由し忘れ、
`cv_train_mean` / `cv_val_std` が log.csv に残らない。すると `G-DIAG`（CV 内部診断を
常設の判断軸にする）が空洞化する —— 過去コンペで**記入率が 28% / 21% まで落ちた**実績がある。
指標のハードコードや「学習だけして推論を省く」も同じ経路で起きる。
**最初から作法を満たした状態から始めれば、忘れようがない。**

この雛形が満たしている作法:

- `ExperimentTracker` の `start_run` / `log_fold_scores` / `end_run(feature_names=...)`
- 指標と CV は `src.metrics` からのみ取る（ハードコード禁止。`train.py` / `optimize_hp.py` と同一）
- `FoldCache` で中断した学習を `--resume` で再開できる
- `save_run_outputs()` で **学習 → OOF + test 予測 → 提出 CSV を 1 回の実行で出し切る**

実行:
    uv run python -m experiments.runs.exp042_s4_fe_age_ratio --model lgb
    uv run python -m experiments.runs.exp042_s4_fe_age_ratio --model lgb --resume
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import N_SPLITS, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COL
from src.experiment import ExperimentTracker
from src.metrics import (describe, get_cv, get_groups, get_metric, is_regression,
                         n_classes, shape_for_metric)
from src.utils.finalize import save_run_outputs
from src.utils.foldcache import FoldCache

# ──────────────────────────────────────────────
# TODO: この実験の内容に合わせて変更する
# ──────────────────────────────────────────────
DESCRIPTION = "（この実験で何を明らかにするか。/ds-new-experiment の Q1 の答え）"
FEATURES: list[str] = []          # この実験で使う特徴量
MODEL_NAME = "lgb"


def train_one_fold(X_tr, y_tr, X_val, params):
    """1 fold を学習して `(model, val 予測)` を返す。

    TODO: 使うモデルに差し替える。分類は確率、回帰は予測値を返せばよい ——
    指標に合う形への整形は `src.metrics.shape_for_metric()` が引き受ける
    （**この整形を自分で書かない**。同じ三項演算子が 6 箇所に写経され、
    片方だけ直る事故が起きた。定義元は 1 つ）。
    """
    import lightgbm as lgb

    Est = lgb.LGBMRegressor if is_regression() else lgb.LGBMClassifier
    model = Est(**params)
    model.fit(X_tr, y_tr)
    return model, _predict(model, X_val)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--resume", action="store_true",
                        help="fold 単位のキャッシュを使い、中断した学習を途中から再開する")
    args = parser.parse_args()

    assert FEATURES, "FEATURES が空です。この実験で使う特徴量を埋めてください。"

    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")
    X, X_test = train[FEATURES], test[FEATURES]
    # 回帰に LabelEncoder を通すと**全ての値が別クラスになる**ので分岐する
    if is_regression():
        y = pd.Series(train[TARGET_COL].to_numpy(dtype=float), index=train.index)
    else:
        y = pd.Series(LabelEncoder().fit_transform(train[TARGET_COL]), index=train.index)

    params = {"random_state": RANDOM_STATE, "verbose": -1}   # TODO: best_params を読むなら差し替える

    # 目的・成功基準・撤退基準は /ds-new-experiment が log.csv の予約行に記録済み
    tracker = ExperimentTracker(experiment_name=f"{args.model}_{len(FEATURES)}features",
                                model=args.model, features=f"{len(FEATURES)}features")
    exp_id = tracker.start_run(description=DESCRIPTION)
    print(f"評価設定: {describe()}")

    cv, metric = get_cv(), get_metric()
    groups = get_groups(train)              # GroupKFold 系のみ。他は None
    n_cls = n_classes(y)
    oof = np.zeros(len(train) if is_regression() else (len(train), n_cls))
    test_pred = np.zeros(len(test) if is_regression() else (len(test), n_cls))
    # TimeSeriesSplit は先頭行をどの fold の検証にも入れない。ゼロのまま採点しないための印
    covered = np.zeros(len(train), dtype=bool)
    train_scores, val_scores = [], []

    cache = FoldCache(tag=f"{args.model}_{len(FEATURES)}f", seed=RANDOM_STATE,
                      n_splits=N_SPLITS, enabled=args.resume,
                      signature={"features": FEATURES, "params": params})
    if args.resume:
        print(cache.report())

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups)):
        cached = cache.load(fold)
        if cached is not None:
            val_pred, fold_test = cached
            model = None
        else:
            model, val_pred = train_one_fold(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], params)
            fold_test = _predict(model, X_test)
            cache.save(fold, val_pred, fold_test)

        oof[va_idx] = val_pred
        covered[va_idx] = True
        test_pred += fold_test / N_SPLITS

        val_score = metric(y.iloc[va_idx], shape_for_metric(val_pred))
        # キャッシュ再利用時は train を再計算できない。**val と同じ値を入れない** ——
        # それは `cv_train_mean` に嘘を書くのと同じで、`--resume` のたびに
        # train−val 乖離が「乖離ゼロ」に化ける（`G-DIAG` の第1診断軸が死ぬ）
        tr_score = (float("nan") if model is None
                    else metric(y.iloc[tr_idx], shape_for_metric(_predict(model, X.iloc[tr_idx]))))
        train_scores.append(tr_score)
        val_scores.append(val_score)
        tracker.log_fold_scores(fold, tr_score, val_score)   # ← G-DIAG の診断列はここで残る
        print(f"Fold {fold}: train={tr_score:.5f}  val={val_score:.5f}"
              f"{'  （キャッシュ再利用）' if model is None else ''}")

    oof_score = metric(y[covered], shape_for_metric(oof[covered]))
    if not covered.all():
        print(f"  ℹ️ OOF は予測された {covered.sum()} / {len(covered)} 行で評価しました")
        oof = oof.astype(float)
        oof[~covered] = np.nan

    # 学習 → OOF + test 予測 → 提出 CSV を 1 回で出し切る（G-STEPWISE）
    save_run_outputs(exp_id=exp_id, model=args.model, oof=oof, test=test_pred, oof_score=oof_score)

    tracker.end_run(train_scores=train_scores, val_scores=val_scores,
                    oof_score=oof_score, feature_names=FEATURES)
    tracker.save_oof_analysis(oof, y.to_numpy())


def _predict(model, X):
    """分類なら確率、回帰なら値（`scripts/train.py` と同じ扱い）。"""
    return model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)


if __name__ == "__main__":
    main()
