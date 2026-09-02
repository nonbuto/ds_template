"""
Optuna ハイパーパラメータ最適化スクリプト

Stage 3（作業用HP、20〜30試行）と Stage 5（本格HP、100試行以上）の両方で使う。
--n-trials で試行数を指定して使い分ける。

特徴量セットは scripts/train.py の FEATURES をそのまま使う（二重管理を避ける）。

使い方:
    # Stage 3: 作業用HP（ノイズ低減目的、素早く終わらせる）
    uv run python -m scripts.optimize_hp --model lgb --n-trials 25 --tag working

    # Stage 5: 本格HP（確定特徴量セットで性能を最大化）
    uv run python -m scripts.optimize_hp --model lgb --n-trials 150 --tag full

結果:
    data/output/params/best_params_{model}_{tag}.json  ← 次回から --params で指定
"""

import argparse
import json

import numpy as np
import optuna
import pandas as pd
from src.metrics import get_cv, get_metric, greater_is_better, needs_proba
from sklearn.preprocessing import LabelEncoder

from src.config import PROCESSED_DATA_DIR, PARAMS_DIR, RANDOM_STATE, TARGET_COL
from src.hp_spaces import lgb_space, xgb_space, cb_space
from scripts.train import FEATURES, N_CLASSES

optuna.logging.set_verbosity(optuna.logging.WARNING)

# アーキテクチャ依存性検証（H-XGB-001）により、XGBoostはLGB向けbin化特徴量ではなく
# TODO: モデルごとに最適な特徴量セットが分かれた場合はここで管理する（未指定モデルは FEATURES を使う）。
#       例: {"xgb": [...生値版の特徴量...]}
MODEL_FEATURES: dict[str, list[str]] = {}

# lgb_space等（src/hp_spaces.py）は二値分類テンプレートのため、
# multiclass用の目的関数キーをここで上書きする（class_weight="balanced"はscripts/train.pyと統一）
# xgb/cbはclass_weight相当を持たないため重みなし学習+beta較正（Stage1.5で確立した方式）で評価する
MULTICLASS_OVERRIDES = {
    "lgb": {"objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
            "class_weight": "balanced", "n_estimators": 1000},
    "xgb": {"objective": "multi:softprob", "num_class": N_CLASSES, "eval_metric": "mlogloss", "n_estimators": 1000,
            "tree_method": "hist", "enable_categorical": True},
    "cb": {"loss_function": "MultiClass", "iterations": 1000},
}

HP_SPACE_FN = {"lgb": lgb_space, "cb": cb_space, "xgb": xgb_space}

BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]


def _best_calibrated_score(oof: np.ndarray, y: pd.Series, prior: np.ndarray) -> float:
    metric = get_metric()
    return max(
        metric(y, (oof / prior**b).argmax(1))
        for b in BETA_GRID
    )


def objective(trial, X: pd.DataFrame, y: pd.Series, model_type: str, prior: np.ndarray) -> float:
    params = HP_SPACE_FN[model_type](trial)
    params.update(MULTICLASS_OVERRIDES[model_type])
    cv = get_cv()   # train.py と同じ分割器（src.metrics が唯一の定義元）
    oof = np.zeros((len(y), N_CLASSES))

    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if model_type == "lgb":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        elif model_type == "cb":
            from catboost import CatBoostClassifier, Pool
            model = CatBoostClassifier(**params)
            model.fit(Pool(X_tr, y_tr, cat_features=cat_cols),
                      eval_set=Pool(X_val, y_val, cat_features=cat_cols),
                      early_stopping_rounds=50, verbose=0)
        elif model_type == "xgb":
            import xgboost as xgb
            model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof[val_idx] = model.predict_proba(X_val)

    if model_type == "lgb":
        metric = get_metric()
        return metric(y, oof[:, 1] if needs_proba() and oof.shape[1] == 2
                      else (oof if needs_proba() else np.argmax(oof, axis=1)))
    return _best_calibrated_score(oof, y, prior)


def main():
    parser = argparse.ArgumentParser(description="Optuna HP最適化")
    parser.add_argument("--model", type=str, default="lgb", choices=["lgb", "cb", "xgb"])
    parser.add_argument("--n-trials", type=int, default=25,
                        help="試行数（Stage 3: 20〜30 / Stage 5: 100以上）")
    parser.add_argument("--tag", type=str, default="working",
                        help="保存ファイルのタグ（working / full）")
    args = parser.parse_args()

    features = MODEL_FEATURES.get(args.model, FEATURES)
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    X, y_raw = train[features], train[TARGET_COL]

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    classes = le.classes_
    prior = pd.Series(y_raw).value_counts().reindex(classes).to_numpy() / len(y_raw)

    stage = "Stage 3（作業用）" if args.n_trials <= 40 else "Stage 5（本格）"
    print(f"\n{stage} HP最適化を開始します")
    print(f"  モデル: {args.model} / 試行数: {args.n_trials} / 特徴量数: {len(features)}")
    if args.model != "lgb":
        print(f"  評価: 重みなし学習 + beta較正（BETA_GRID={BETA_GRID}）")

    # study を SQLite に永続化する（インメモリだと TPE の探索状態が失われ、
    # 「あと N 試行だけ追加したい」ができず毎回ゼロからやり直しになる）
    study_dir = PARAMS_DIR / "optuna_studies"
    study_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"{args.model}_{args.tag}"
    storage = f"sqlite:///{study_dir / f'{study_name}.db'}"

    study = optuna.create_study(
        direction="maximize" if greater_is_better() else "minimize",   # 指標の向きに追従
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    done = len(study.trials)
    if done:
        print(f"  既存 study を再開: 完了済み {done} 試行 → 追加 {args.n_trials} 試行")
    study.optimize(lambda trial: objective(trial, X, y, args.model, prior),
                   n_trials=args.n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    # 保存
    out_path = PARAMS_DIR / f"best_params_{args.model}_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Optuna 最適化完了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 モデル     : {args.model}
 試行数     : {args.n_trials}  ({stage})
 Best OOF: {best_score:.5f}
 保存先     : {out_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
次のステップ:
  uv run python -m scripts.train --model {args.model}_balanced --params {out_path}
""")


if __name__ == "__main__":
    main()
