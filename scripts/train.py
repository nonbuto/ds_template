"""
CV学習スクリプト（汎用骨格）

CV でモデルを学習し、OOF予測・テスト予測・特徴量重要度を保存する。
**評価指標と CV 分割器は `src/metrics.py` が `src/config.py` の設定から決める**
（`scripts/optimize_hp.py` と同じ定義元を使うため、指標がずれることがない）。
コンペ開始時に TODO 箇所を埋めて使う。

binary_classification / multiclass の両方に対応する（`N_CLASSES` で切り替える）。

使い方:
    uv run python -m scripts.train
    uv run python -m scripts.train --model lgb
    uv run python -m scripts.train --model cb --params data/output/params/best_params_cb.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from src.metrics import get_cv, get_metric, needs_proba, describe as describe_setup
from sklearn.preprocessing import LabelEncoder

from src.config import (
    PROCESSED_DATA_DIR, PLOTS_DIR,
    RANDOM_STATE, N_SPLITS, TARGET_COL, EXPERIMENT_NAME,
)
from src.experiment import ExperimentTracker
from src.utils.finalize import save_run_outputs
from src.utils.foldcache import FoldCache

# ──────────────────────────────────────────────
# TODO: コンペごとにここを変更する
# ──────────────────────────────────────────────

# 使用する特徴量リスト（空のまま実行するとエラーになる。コンペごとに埋める）
FEATURES: list[str] = []

# クラス数。二値分類なら 2、多クラスならクラス数を入れる（回帰では使わない）
# ※ DEFAULT_PARAMS の objective / metric もこの値に合わせて調整すること
N_CLASSES = 3

# モデルごとのデフォルトパラメータ（multiclass）
DEFAULT_PARAMS: dict = {
    "lgb": {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "metric": "multi_logloss",
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
    "lgb_balanced": {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "metric": "multi_logloss",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbose": -1,
    },
    "cb": {
        "loss_function": "MultiClass",
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
    },
    "xgb": {
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "eval_metric": "mlogloss",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    },
}

# ──────────────────────────────────────────────
# 学習関数（いずれも predict_proba (N, N_CLASSES) を返す）
# ──────────────────────────────────────────────

def train_fold_lgb(X_tr, y_tr, X_val, y_val, params: dict):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model, model.predict_proba(X_val)


def train_fold_cb(X_tr, y_tr, X_val, y_val, params: dict):
    from catboost import CatBoostClassifier, Pool
    cat_features = [c for c in X_tr.columns if X_tr[c].dtype in ("object", "category")]
    model = CatBoostClassifier(**params)
    model.fit(
        Pool(X_tr, y_tr, cat_features=cat_features),
        eval_set=Pool(X_val, y_val, cat_features=cat_features),
        early_stopping_rounds=50,
    )
    return model, model.predict_proba(X_val)


def train_fold_xgb(X_tr, y_tr, X_val, y_val, params: dict):
    import xgboost as xgb
    model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model, model.predict_proba(X_val)


TRAIN_FN = {"lgb": train_fold_lgb, "lgb_balanced": train_fold_lgb, "cb": train_fold_cb, "xgb": train_fold_xgb}


def run_cv(model_name: str, params: dict, seed: int, features: list[str] = None):
    """1回分のCV学習を実行し、スコアとOOF/test予測を返す（log.csv記録なし）。

    multi-seed検証など、複数回まとめて呼び出して自前で集計したい場合に使う。
    単発のCV学習で log.csv に記録したい場合は main() を使う。

    Returns:
        dict: train_scores, val_scores, oof_score, oof_preds, test_preds
    """
    features = features or FEATURES
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")
    X, y_raw = train[features], train[TARGET_COL]
    X_test = test[features]

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)

    seeded_params = {**params, "random_state": seed} if model_name != "cb" else {**params, "random_seed": seed}

    cv = get_cv()          # 分割器は CV_STRATEGY から。seed は RANDOM_STATE に従う
    metric = get_metric()
    oof_preds = np.zeros((len(train), N_CLASSES))
    test_preds = np.zeros((len(test), N_CLASSES))
    train_scores, val_scores = [], []
    importances = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model, val_pred = TRAIN_FN[model_name](X_tr, y_tr, X_val, y_val, seeded_params)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test) / N_SPLITS

        val_score = metric(y_val, val_pred[:, 1] if needs_proba() and val_pred.shape[1] == 2
                           else (val_pred if needs_proba() else np.argmax(val_pred, axis=1)))
        if needs_proba():
            tr_proba = model.predict_proba(X_tr)
            tr_score = metric(y_tr, tr_proba[:, 1] if tr_proba.shape[1] == 2 else tr_proba)
        else:
            tr_score = metric(y_tr, model.predict(X_tr))
        train_scores.append(tr_score)
        val_scores.append(val_score)

        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

    oof_score = metric(y, oof_preds[:, 1] if needs_proba() and oof_preds.shape[1] == 2
                       else (oof_preds if needs_proba() else np.argmax(oof_preds, axis=1)))

    importance_df = None
    if importances:
        importance_df = pd.DataFrame(
            {"feature": features, "importance": np.mean(importances, axis=0)}
        ).sort_values("importance", ascending=False)

    return {
        "train_scores": train_scores,
        "val_scores": val_scores,
        "oof_score": oof_score,
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "importance_df": importance_df,
    }


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgb", choices=["lgb", "lgb_balanced", "cb", "xgb"])
    parser.add_argument("--params", type=str, default="",
                        help="best_params JSON ファイルパス（省略時はデフォルトパラメータを使用）")
    parser.add_argument("--resume", action="store_true",
                        help="fold 単位のキャッシュを使い、中断した学習を途中から再開する")
    args = parser.parse_args()

    assert FEATURES, "FEATURES リストが空です。scripts/train.py の TODO を埋めてください。"

    # データ読み込み
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")
    X, y_raw = train[FEATURES], train[TARGET_COL]
    X_test = test[FEATURES]

    # ターゲットのラベルエンコード（文字列クラス → 0..N_CLASSES-1）
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    print(f"クラス対応: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"評価設定: {describe_setup()}")

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
    tracker.start_run(description=f"{args.model} CV学習（数値7カラムベースライン）")
    tracker.log_params(params)

    # CV学習
    cv = get_cv()          # 分割器は src.config の CV_STRATEGY から決まる
    metric = get_metric()  # 指標は EVAL_METRIC から決まる（optimize_hp と共有）
    oof_preds = np.zeros((len(train), N_CLASSES))
    test_preds = np.zeros((len(test), N_CLASSES))
    importances = []
    train_scores, val_scores = [], []

    # fold 単位のチェックポイント（--resume 時のみ有効）。
    # 中断・クラッシュで既に終わった fold の計算が失われるのを防ぐ（CONVENTIONS の実行規約）。
    cache = FoldCache(tag=f"{args.model}_{len(FEATURES)}f", seed=RANDOM_STATE,
                      n_splits=N_SPLITS, enabled=args.resume)
    if args.resume:
        print(cache.report())

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        cached = cache.load(fold)
        if cached is not None:
            val_pred, fold_test_pred = cached
            model = None
        else:
            model, val_pred = TRAIN_FN[args.model](X_tr, y_tr, X_val, y_val, params)
            fold_test_pred = model.predict_proba(X_test)
            cache.save(fold, val_pred, fold_test_pred)

        oof_preds[val_idx] = val_pred
        test_preds += fold_test_pred / N_SPLITS   # テスト予測（フォールド平均）

        # スコア計算（balanced_accuracy: argmaxクラスで評価）
        # キャッシュ再利用時は train スコアを再計算できないため val と同値を入れる
        # 指標が確率を要るか（AUC/logloss）ラベルで良いか（accuracy 等）で渡すものを変える
        val_score = metric(y_val, val_pred[:, 1] if needs_proba() and val_pred.shape[1] == 2
                           else (val_pred if needs_proba() else np.argmax(val_pred, axis=1)))
        if model is None:
            tr_score = val_score        # キャッシュ再利用時は train を再計算できない
        elif needs_proba():
            tr_proba = model.predict_proba(X_tr)
            tr_score = metric(y_tr, tr_proba[:, 1] if tr_proba.shape[1] == 2 else tr_proba)
        else:
            tr_score = metric(y_tr, model.predict(X_tr))
        train_scores.append(tr_score)
        val_scores.append(val_score)
        tracker.log_fold_scores(fold, tr_score, val_score)

        # 特徴量重要度
        if model is not None and hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

        mark = "（キャッシュ再利用）" if model is None else ""
        print(f"Fold {fold}: train={tr_score:.5f}  val={val_score:.5f} {mark}")

    # OOFスコア
    oof_score = metric(y, oof_preds[:, 1] if needs_proba() and oof_preds.shape[1] == 2
                       else (oof_preds if needs_proba() else np.argmax(oof_preds, axis=1)))

    # 保存: 学習 → OOF + test 予測 → 提出ファイルを 1 回で出し切る（CLAUDE.md `G-STEPWISE`）。
    # 学習だけして推論を省くと、提出したくなった時点で同じ学習をやり直すことになる。
    exp_id = tracker._experiment_id or "000"
    save_run_outputs(exp_id=exp_id, model=args.model, oof=oof_preds,
                     test=test_preds, oof_score=oof_score)

    if importances:
        imp_df = pd.DataFrame({"feature": FEATURES, "importance": np.mean(importances, axis=0)})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df.to_csv(PLOTS_DIR / f"feature_importance_{exp_id}.csv", index=False)

    # feature_names を渡すと params/features_{exp_id}.json に特徴量セットが残り、
    # 「今どれがベースか」を機械可読に追える（`feature_report --sync` が読む）
    tracker.end_run(train_scores=train_scores, val_scores=val_scores,
                    oof_score=oof_score, feature_names=FEATURES)

    # 誤差分析（`G-MECH` が必須と定める可視化局面③。問題種別に応じて内容が変わる）
    tracker.save_oof_analysis(oof_preds, y.to_numpy())


if __name__ == "__main__":
    main()
