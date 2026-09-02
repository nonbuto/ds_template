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
from src.metrics import describe, get_cv, get_metric, needs_proba
from src.utils.finalize import save_run_outputs
from src.utils.foldcache import FoldCache

# ──────────────────────────────────────────────
# TODO: この実験の内容に合わせて変更する
# ──────────────────────────────────────────────
DESCRIPTION = "（この実験で何を明らかにするか。/ds-new-experiment の Q1 の答え）"
FEATURES: list[str] = []          # この実験で使う特徴量
MODEL_NAME = "lgb"


def _shape(p: np.ndarray) -> np.ndarray:
    """指標が確率を要るかどうかで、渡すものを変える（二値は陽性確率、ラベル指標は argmax）。"""
    return (p[:, 1] if needs_proba() and p.shape[1] == 2
            else (p if needs_proba() else np.argmax(p, axis=1)))


def train_one_fold(X_tr, y_tr, X_val, params):
    """1 fold を学習して `(model, val 予測)` を返す。

    TODO: 使うモデルに差し替える。**確率が要る指標（AUC 等）なら `predict_proba` を返す。**
    """
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_val)


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
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(train[TARGET_COL]), index=train.index)

    params = {"random_state": RANDOM_STATE, "verbose": -1}   # TODO: best_params を読むなら差し替える

    # 目的・成功基準・撤退基準は /ds-new-experiment が log.csv の予約行に記録済み
    tracker = ExperimentTracker(experiment_name=f"{args.model}_{len(FEATURES)}features",
                                model=args.model, features=f"{len(FEATURES)}features")
    exp_id = tracker.start_run(description=DESCRIPTION)
    print(f"評価設定: {describe()}")

    cv, metric = get_cv(), get_metric()
    n_classes = int(y.nunique())
    oof = np.zeros((len(train), n_classes))
    test_pred = np.zeros((len(test), n_classes))
    train_scores, val_scores = [], []

    cache = FoldCache(tag=f"{args.model}_{len(FEATURES)}f", seed=RANDOM_STATE,
                      n_splits=N_SPLITS, enabled=args.resume)
    if args.resume:
        print(cache.report())

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        cached = cache.load(fold)
        if cached is not None:
            val_pred, fold_test = cached
            model = None
        else:
            model, val_pred = train_one_fold(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], params)
            fold_test = model.predict_proba(X_test)
            cache.save(fold, val_pred, fold_test)

        oof[va_idx] = val_pred
        test_pred += fold_test / N_SPLITS

        val_score = metric(y.iloc[va_idx], _shape(val_pred))
        tr_score = val_score if model is None else metric(y.iloc[tr_idx],
                                                          _shape(model.predict_proba(X.iloc[tr_idx])))
        train_scores.append(tr_score)
        val_scores.append(val_score)
        tracker.log_fold_scores(fold, tr_score, val_score)   # ← G-DIAG の診断列はここで残る
        print(f"Fold {fold}: train={tr_score:.5f}  val={val_score:.5f}"
              f"{'  （キャッシュ再利用）' if model is None else ''}")

    oof_score = metric(y, _shape(oof))

    # 学習 → OOF + test 予測 → 提出 CSV を 1 回で出し切る（G-STEPWISE）
    save_run_outputs(exp_id=exp_id, model=args.model, oof=oof, test=test_pred, oof_score=oof_score)

    tracker.end_run(train_scores=train_scores, val_scores=val_scores,
                    oof_score=oof_score, feature_names=FEATURES)
    tracker.save_oof_analysis(oof, y.to_numpy())


if __name__ == "__main__":
    main()
