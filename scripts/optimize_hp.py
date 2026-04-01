"""
Optuna ハイパーパラメータ最適化スクリプト

Stage 3（作業用HP、20〜30試行）と Stage 5（本格HP、100試行以上）の両方で使う。
--n-trials で試行数を指定して使い分ける。

使い方:
    # Stage 3: 作業用HP（ノイズ低減目的、素早く終わらせる）
    uv run python scripts/optimize_hp.py --model lgb --n-trials 25 --tag working

    # Stage 5: 本格HP（確定特徴量セットで性能を最大化）
    uv run python scripts/optimize_hp.py --model lgb --n-trials 150 --tag full

結果:
    data/output/params/best_params_{model}_{tag}.json  ← 次回から --params で指定
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import PROCESSED_DATA_DIR, PARAMS_DIR, RANDOM_STATE, N_SPLITS, TARGET_COL
from src.hp_spaces import lgb_space, xgb_space, cb_space

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ──────────────────────────────────────────────
# TODO: 最適化に使う特徴量セットをここに定義する
# ──────────────────────────────────────────────
# Stage 3: ベース特徴量（最小限）
# Stage 5: 確定した全特徴量
FEATURES: list[str] = []   # 空のまま実行するとエラーになる

HP_SPACE_FN = {"lgb": lgb_space, "cb": cb_space, "xgb": xgb_space}


def objective(trial, X: pd.DataFrame, y: pd.Series, model_type: str) -> float:
    params = HP_SPACE_FN[model_type](trial)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if model_type == "lgb":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        elif model_type == "cb":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=50, verbose=0)
        elif model_type == "xgb":
            import xgboost as xgb
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      early_stopping_rounds=50, verbose=False)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof)


def main():
    parser = argparse.ArgumentParser(description="Optuna HP最適化")
    parser.add_argument("--model", type=str, default="lgb", choices=["lgb", "cb", "xgb"])
    parser.add_argument("--n-trials", type=int, default=25,
                        help="試行数（Stage 3: 20〜30 / Stage 5: 100以上）")
    parser.add_argument("--tag", type=str, default="working",
                        help="保存ファイルのタグ（working / full）")
    args = parser.parse_args()

    assert FEATURES, "FEATURES が空です。scripts/optimize_hp.py の TODO を埋めてください。"

    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    X, y = train[FEATURES], train[TARGET_COL]

    stage = "Stage 3（作業用）" if args.n_trials <= 40 else "Stage 5（本格）"
    print(f"\n{stage} HP最適化を開始します")
    print(f"  モデル: {args.model} / 試行数: {args.n_trials} / 特徴量数: {len(FEATURES)}")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X, y, args.model),
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
 Best OOF   : {best_score:.5f}
 保存先     : {out_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
次のステップ:
  scripts/train.py --model {args.model} --params {out_path}
""")


if __name__ == "__main__":
    main()
