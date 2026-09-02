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
from src.metrics import (get_cv, get_groups, get_metric, greater_is_better, needs_proba,
                         is_regression, n_classes, shape_for_metric)
from sklearn.preprocessing import LabelEncoder

from src.config import PROCESSED_DATA_DIR, PARAMS_DIR, RANDOM_STATE, TARGET_COL
from src.hp_spaces import lgb_space, xgb_space, cb_space
from scripts.train import FEATURES, _lgb_eval_kwargs

optuna.logging.set_verbosity(optuna.logging.WARNING)

# アーキテクチャ依存性検証（H-XGB-001）により、XGBoostはLGB向けbin化特徴量ではなく
# TODO: モデルごとに最適な特徴量セットが分かれた場合はここで管理する（未指定モデルは FEATURES を使う）。
#       例: {"xgb": [...生値版の特徴量...]}
MODEL_FEATURES: dict[str, list[str]] = {}

HP_SPACE_FN = {"lgb": lgb_space, "cb": cb_space, "xgb": xgb_space}

# beta 較正のグリッド。多クラス × ラベル指標（accuracy / balanced_accuracy / f1）でのみ意味を持つ。
# 予測確率を事前分布の beta 乗で割ることで、少数クラスの過小予測を補正する（`G-DIAG`）。
BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]


def _use_beta_calibration(n_cls: int) -> bool:
    """beta 較正を使う局面か。

    確率をそのまま評価する指標（AUC・logloss）や回帰では、argmax を挟む較正は
    **情報を捨てるだけ**。二値でも事前分布による再重み付けは閾値移動と等価で、
    ランキング指標には効かない。多クラス × ラベル指標に限る。
    """
    return (not is_regression()) and (not needs_proba()) and n_cls > 2


def _score_with_beta(oof: np.ndarray, y: pd.Series, prior: np.ndarray,
                     n_cls: int) -> tuple[float, float]:
    """スコアと、それを与えた beta を返す。

    以前の実装には 3 つの問題があった:

    1. `max(...)` で最良を選んでいた —— **小さいほど良い指標では最悪の beta を選ぶ**
    2. `.argmax(1)` を常に挟んでいた —— AUC / logloss では確率を潰してしまう
    3. **最良 beta を保存していなかった** —— HP は保存されるのに較正は再現できず、
       最適化時のスコアと推論時のスコアが一致しない（`G-FAIR` の「較正」項そのもの）
    4. lgb だけ較正を通さず、xgb / cb とは違う条件で比較していた（不公正比較）
    """
    metric = get_metric()
    if not _use_beta_calibration(n_cls):
        return metric(y, shape_for_metric(oof)), 1.0

    pick = max if greater_is_better() else min
    scored = [(metric(y, (oof / prior**b).argmax(1)), b) for b in BETA_GRID]
    return pick(scored, key=lambda t: t[0])


def build_search_params(trial, model_type: str, n_cls: int) -> dict:
    """探索空間の HP に、タスク種別が決めるキー（目的関数・クラス数）を重ねる。

    **以前は `MULTICLASS_OVERRIDES` という定数で多クラス前提のキーを無条件に上書きし、
    さらに lgb にだけ `class_weight="balanced"` を混ぜていた。** そのため
    ①二値・回帰コンペでは動かず、②lgb だけ重み付き・xgb/cb は重みなしという
    条件の揃わない比較になっていた —— テンプレート自身が `G-FAIR` 違反を作っていた。
    目的関数の定義元は `scripts/train.build_params()` 一箇所に統一する。
    """
    from scripts.train import build_params

    params = HP_SPACE_FN[model_type](trial)
    task = build_params(model_type, n_cls)
    for key in ("objective", "num_class", "metric", "loss_function", "eval_metric",
                "tree_method", "enable_categorical"):
        if key in task:
            params[key] = task[key]
    return params


def objective(trial, X: pd.DataFrame, y: pd.Series, model_type: str,
              prior: np.ndarray, n_cls: int, groups=None) -> float:
    params = build_search_params(trial, model_type, n_cls)
    cv = get_cv()   # train.py と同じ分割器（src.metrics が唯一の定義元）
    oof = np.zeros(len(y)) if is_regression() else np.zeros((len(y), n_cls))
    covered = np.zeros(len(y), dtype=bool)   # TimeSeriesSplit は先頭を一度も検証しない

    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if model_type == "lgb":
            import lightgbm as lgb
            Est = lgb.LGBMRegressor if is_regression() else lgb.LGBMClassifier
            model = Est(**params)
            model.fit(X_tr, y_tr, **_lgb_eval_kwargs(Est, X_val, y_val),
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        elif model_type == "cb":
            from catboost import CatBoostClassifier, CatBoostRegressor, Pool
            Est = CatBoostRegressor if is_regression() else CatBoostClassifier
            model = Est(**params)
            model.fit(Pool(X_tr, y_tr, cat_features=cat_cols),
                      eval_set=Pool(X_val, y_val, cat_features=cat_cols),
                      early_stopping_rounds=50, verbose=0)
        elif model_type == "xgb":
            import xgboost as xgb
            Est = xgb.XGBRegressor if is_regression() else xgb.XGBClassifier
            model = Est(**params, early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof[val_idx] = model.predict(X_val) if is_regression() else model.predict_proba(X_val)
        covered[val_idx] = True

    # **予測されなかった行を評価に混ぜない。** TimeSeriesSplit では先頭の行が
    # どの fold の検証にも入らず、ゼロのまま残る。それを正解と突き合わせると
    # 「全部クラス0と予測した」ことになり、スコアが実力と無関係に動く。
    score, beta = _score_with_beta(oof[covered], y[covered], prior, n_cls)
    trial.set_user_attr("beta", beta)
    trial.set_user_attr("n_scored_rows", int(covered.sum()))
    return score


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

    if is_regression():
        y = pd.Series(y_raw.to_numpy(dtype=float), index=y_raw.index)
        prior = np.array([1.0])
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
        prior = (pd.Series(y_raw).value_counts().reindex(le.classes_).to_numpy()
                 / len(y_raw))
    n_cls = n_classes(y)
    groups = get_groups(train)   # GroupKFold 系のみ。他は None

    stage = "Stage 3（作業用）" if args.n_trials <= 40 else "Stage 5（本格）"
    print(f"\n{stage} HP最適化を開始します")
    print(f"  モデル: {args.model} / 試行数: {args.n_trials} / 特徴量数: {len(features)}")
    if _use_beta_calibration(n_cls):
        print(f"  評価: beta 較正つき（BETA_GRID={BETA_GRID}）— 全モデル同条件")

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
    study.optimize(lambda trial: objective(trial, X, y, args.model, prior, n_cls, groups),
                   n_trials=args.n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value
    best_beta = study.best_trial.user_attrs.get("beta", 1.0)

    # 保存。**較正パラメータ（beta）も一緒に残す** —— HP だけ保存して beta を捨てると、
    # 最適化時のスコアを推論時に再現できない（`G-FAIR` の「較正」項）。
    out_path = PARAMS_DIR / f"best_params_{args.model}_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    if _use_beta_calibration(n_cls):
        calib_path = PARAMS_DIR / f"calibration_{args.model}_{args.tag}.json"
        with open(calib_path, "w") as f:
            json.dump({"beta": best_beta, "prior": prior.tolist()}, f, indent=2)
        print(f"  較正パラメータを保存しました: {calib_path.name}（beta={best_beta}）")

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
  uv run python -m scripts.train --model {args.model} --params {out_path}
""")


if __name__ == "__main__":
    main()
