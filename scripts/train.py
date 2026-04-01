"""
CV学習スクリプト（汎用骨格）

StratifiedKFold CV でモデルを学習し、OOF予測・テスト予測・特徴量重要度を保存する。
コンペ開始時に TODO 箇所を埋めて使う。

使い方:
    uv run python scripts/train.py
    uv run python scripts/train.py --model lgb
    uv run python scripts/train.py --model cb --params data/output/params/best_params_cb.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    PROCESSED_DATA_DIR, OOF_DIR, MODELS_DIR, PARAMS_DIR, PLOTS_DIR,
    RANDOM_STATE, N_SPLITS, TARGET_COL, EXPERIMENT_NAME,
)
from src.experiment import ExperimentTracker

# ──────────────────────────────────────────────
# TODO: コンペごとにここを変更する
# ──────────────────────────────────────────────

# 使用する特徴量リスト（src/config.py に定数として定義することを推奨）
# TODO: FEATURES = [...]
FEATURES: list[str] = []   # 空のまま実行するとエラーになる

# モデルごとのデフォルトパラメータ
DEFAULT_PARAMS: dict = {
    "lgb": {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    },
    "cb": {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
    },
    "xgb": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    },
}

# ──────────────────────────────────────────────
# 学習関数
# ──────────────────────────────────────────────

def train_fold_lgb(X_tr, y_tr, X_val, y_val, params: dict):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model, model.predict_proba(X_val)[:, 1]


def train_fold_cb(X_tr, y_tr, X_val, y_val, params: dict):
    from catboost import CatBoostClassifier, Pool
    cat_features = [c for c in X_tr.columns if X_tr[c].dtype == "object"]
    model = CatBoostClassifier(**params)
    model.fit(
        Pool(X_tr, y_tr, cat_features=cat_features),
        eval_set=Pool(X_val, y_val, cat_features=cat_features),
        early_stopping_rounds=50,
    )
    return model, model.predict_proba(X_val)[:, 1]


def train_fold_xgb(X_tr, y_tr, X_val, y_val, params: dict):
    import xgboost as xgb
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100,
    )
    return model, model.predict_proba(X_val)[:, 1]


TRAIN_FN = {"lgb": train_fold_lgb, "cb": train_fold_cb, "xgb": train_fold_xgb}


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgb", choices=["lgb", "cb", "xgb"])
    parser.add_argument("--params", type=str, default="",
                        help="best_params JSON ファイルパス（省略時はデフォルトパラメータを使用）")
    args = parser.parse_args()

    assert FEATURES, "FEATURES リストが空です。scripts/train.py の TODO を埋めてください。"

    # データ読み込み
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")
    X, y = train[FEATURES], train[TARGET_COL]
    X_test = test[FEATURES]

    # パラメータ読み込み
    params = DEFAULT_PARAMS[args.model].copy()
    if args.params:
        with open(args.params) as f:
            params.update(json.load(f))

    # 実験トラッキング
    tracker = ExperimentTracker(
        experiment_name=EXPERIMENT_NAME,
        model=args.model,
        features=f"{len(FEATURES)}features",
    )
    tracker.start_run(description=f"{args.model} CV学習")
    tracker.log_params(params)

    # CV学習
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    importances = []
    train_scores, val_scores = [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model, val_pred = TRAIN_FN[args.model](X_tr, y_tr, X_val, y_val, params)
        oof_preds[val_idx] = val_pred

        # テスト予測（フォールド平均）
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

        # スコア計算
        from sklearn.metrics import roc_auc_score
        tr_score = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
        val_score = roc_auc_score(y_val, val_pred)
        train_scores.append(tr_score)
        val_scores.append(val_score)
        tracker.log_fold_scores(fold, tr_score, val_score)

        # 特徴量重要度
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

        print(f"Fold {fold}: train={tr_score:.5f}  val={val_score:.5f}")

    # OOFスコア
    from sklearn.metrics import roc_auc_score
    oof_score = roc_auc_score(y, oof_preds)

    # 保存
    exp_id = tracker._experiment_id or "000"
    np.save(OOF_DIR / f"oof_{exp_id}_{args.model}.npy", oof_preds)
    np.save(OOF_DIR / f"test_{exp_id}_{args.model}.npy", test_preds)

    if importances:
        imp_df = pd.DataFrame({"feature": FEATURES, "importance": np.mean(importances, axis=0)})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df.to_csv(PLOTS_DIR / f"feature_importance_{exp_id}.csv", index=False)

    tracker.end_run(train_scores=train_scores, val_scores=val_scores,
                    oof_score=oof_score, n_features=len(FEATURES))


if __name__ == "__main__":
    main()
