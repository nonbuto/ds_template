"""
CV学習スクリプト（汎用骨格）

CV でモデルを学習し、OOF予測・テスト予測・特徴量重要度を保存する。
**評価指標と CV 分割器は `src/metrics.py` が `src/config.py` の設定から決める**
（`scripts/optimize_hp.py` と同じ定義元を使うため、指標がずれることがない）。
コンペ開始時に TODO 箇所を埋めて使う。

regression / binary_classification / multiclass に対応する
（`src/config.py` の `PROBLEM_TYPE` から目的関数・クラス数・出力の形が決まる）。

使い方:
    uv run python -m scripts.train
    uv run python -m scripts.train --model lgb
    uv run python -m scripts.train --model cb --params data/output/params/best_params_cb.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from src.metrics import (get_cv, get_groups, get_metric, n_classes, is_regression,
                         native_eval_metric,
                         shape_for_metric, describe as describe_setup)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    PROCESSED_DATA_DIR, PLOTS_DIR,
    RANDOM_STATE, N_SPLITS, TARGET_COL, EXPERIMENT_NAME, PROBLEM_TYPE, CV_STRATEGY,
    EARLY_STOPPING_ON, EARLY_STOPPING_INNER_SIZE,
)
from src.experiment import ExperimentTracker
from src.utils.finalize import save_run_outputs
from src.utils.foldcache import FoldCache

# ──────────────────────────────────────────────
# TODO: コンペごとにここを変更する
# ──────────────────────────────────────────────

# 使用する特徴量リスト（空のまま実行するとエラーになる。コンペごとに埋める）
FEATURES: list[str] = []

# クラス数は `PROBLEM_TYPE` から導く（回帰なら 1、二値なら 2）。
# **以前はここが `N_CLASSES = 3` の直書きで、config のデフォルト（binary_classification / auc）と
# 矛盾していた**。clone した直後に Stage 1 の最小ベースラインを回すと、
# `num_class=3` の目的関数に 2 クラスのラベルを渡して落ちる。設定から導けば矛盾しようがない。
# multiclass のときだけ実データが要るので、学習時に y から数え直す（_resolve_n_classes）。
try:
    N_CLASSES = n_classes()
except ValueError:
    N_CLASSES = 0          # multiclass — run_cv / main が y から確定させる


def _resolve_n_classes(y) -> int:
    """設定と実データを突き合わせてクラス数を決める。食い違いはその場で止める。"""
    actual = n_classes(y)
    if N_CLASSES and actual != N_CLASSES:
        raise ValueError(
            f"PROBLEM_TYPE='{PROBLEM_TYPE}' はクラス数 {N_CLASSES} を意味しますが、"
            f"データには {actual} クラスあります。src/config.py の PROBLEM_TYPE を見直してください。"
        )
    return actual


def build_params(model_name: str, n_cls: int) -> dict:
    """モデル別のデフォルト HP を、タスク種別に合わせて組み立てる。

    目的関数・評価指標はクラス数で変わる（二値 `binary` / 多クラス `multiclass` /
    回帰 `regression`）。dict のリテラルに書くと二値と多クラスで別の定数群を
    保守することになり、片方が必ず腐る。
    """
    is_reg = is_regression()
    common = {"n_estimators": 1000, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
              "random_state": RANDOM_STATE, "verbose": -1}

    if model_name in ("lgb", "lgb_balanced"):
        if is_reg:
            task = {"objective": "regression", "metric": "rmse"}
        elif n_cls == 2:
            task = {"objective": "binary", "metric": "binary_logloss"}
        else:
            task = {"objective": "multiclass", "num_class": n_cls, "metric": "multi_logloss"}
        params = {**common, **task, "num_leaves": 63}
        native = native_eval_metric("lgb", n_cls)
        if native:
            params["metric"] = native
        if model_name == "lgb_balanced":
            if is_reg:
                raise ValueError("lgb_balanced（class_weight）は回帰では使えません。--model lgb を使ってください")
            params["class_weight"] = "balanced"
        return params

    if model_name == "cb":
        loss = "RMSE" if is_reg else ("Logloss" if n_cls == 2 else "MultiClass")
        params = {"loss_function": loss, "iterations": 1000, "learning_rate": 0.05,
                  "depth": 6, "random_seed": RANDOM_STATE, "verbose": 0}
        # **CatBoost にも明示的に持たせる。** ここが空だと、探索空間側に直書きされた
        # `eval_metric="AUC"` が上書きされずに生き残り、RMSE コンペでも AUC で
        # early stopping していた（例外は出ず、best iteration だけが狂う）
        native = native_eval_metric("cb", n_cls)
        if native:
            params["eval_metric"] = native
        return params

    if model_name in ("realmlp", "tabm"):
        # 学習時間はエポック数で決まる。既定は「作業用」に振った軽めの設定で、
        # Stage 5 では optimize_hp が n_epochs も含めて探索する。
        return {"_nn_kind": model_name, "n_epochs": 64, "device": "cpu",
                "random_state": RANDOM_STATE, "verbosity": 0}

    if model_name == "xgb":
        if is_reg:
            task = {"objective": "reg:squarederror", "eval_metric": "rmse"}
        elif n_cls == 2:
            task = {"objective": "binary:logistic", "eval_metric": "logloss"}
        else:
            task = {"objective": "multi:softprob", "num_class": n_cls, "eval_metric": "mlogloss"}
        params = {**task, "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6,
                  "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                  "enable_categorical": True, "random_state": RANDOM_STATE, "verbosity": 0}
        native = native_eval_metric("xgb", n_cls)
        if native:
            params["eval_metric"] = native
        return params

    raise ValueError(f"未対応のモデルです: {model_name}")


# 後方互換: 既存コードが `DEFAULT_PARAMS[name]` を参照している箇所のための辞書ビュー。
# クラス数が実データ依存（multiclass）の場合は build_params() を直接呼ぶこと。
DEFAULT_PARAMS: dict = {
    name: build_params(name, N_CLASSES or 2)
    for name in ("lgb", "lgb_balanced", "cb", "xgb", "realmlp", "tabm")
    if not (is_regression() and name == "lgb_balanced")
}

# ──────────────────────────────────────────────
# 学習関数
# ──────────────────────────────────────────────
# 分類は `predict_proba (N, n_classes)`、回帰は 1 次元の予測を返す。
# 呼び出し側は `src.metrics.shape_for_metric()` で指標に合う形へ整えるので、
# ここでは**モデルの素の出力**を返せばよい。

def _has_proba(model) -> bool:
    """確率を出せるモデルか（回帰器は出せない）。"""
    return hasattr(model, "predict_proba")


def _predict(model, X):
    """分類なら確率、回帰なら値。"""
    return model.predict_proba(X) if _has_proba(model) else model.predict(X)


def _early_stopping_split(X_tr, y_tr, X_val, y_val, inner_seed: int = RANDOM_STATE):
    """early stopping の監視に使う `(X, y)` を返す。

    **既定は train fold の内側から切り出す。** 検証 fold（= OOF になる行）を
    early stopping の監視に使うと、木の本数がその fold に合わせて選ばれる。
    OOF はその選択を経た予測なので、**OOF が本来より良く出る**。
    `G-OOF` は「OOF を足切りに使う」ことを前提にしているので、
    OOF が構造的に楽観側へ寄ると、その前提が静かに崩れる。

    `EARLY_STOPPING_ON = "val"` にすれば従来どおり検証 fold を使う
    （Kaggle で広く使われる書き方。速く、この程度のリークは無視できるという判断もある）。
    """
    if EARLY_STOPPING_ON == "val":
        return X_val, y_val
    stratify = None if is_regression() else y_tr
    # **内側分割の seed はモデル seed に追従させる。** 固定にすると multi-seed avg で
    # seed を振っても ES 用の 15% が毎回同じ行になり、seed 由来の多様性が出ない。
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr, y_tr, test_size=EARLY_STOPPING_INNER_SIZE,
        random_state=inner_seed, stratify=stratify)
    return (X_fit, y_fit), (X_es, y_es)


def _split_for_fit(X_tr, y_tr, X_val, y_val, params: dict | None = None):
    """`(学習に使う X, y, early stopping 用 X, y)` に展開する。"""
    seed = RANDOM_STATE
    if params:
        seed = int(params.get("random_state", params.get("random_seed", RANDOM_STATE)))
    a, b = _early_stopping_split(X_tr, y_tr, X_val, y_val, inner_seed=seed)
    if EARLY_STOPPING_ON == "val":
        return X_tr, y_tr, a, b
    (X_fit, y_fit), (X_es, y_es) = a, b
    return X_fit, y_fit, X_es, y_es


def _best_iteration(model) -> int | None:
    """early stopping が選んだ本数を、ライブラリ差を吸収して取り出す。"""
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(model, attr, None)
        if isinstance(v, (int, np.integer)) and v > 0:
            return int(v)
    getter = getattr(model, "get_best_iteration", None)
    if callable(getter):
        v = getter()
        if isinstance(v, (int, np.integer)) and v > 0:
            return int(v)
    return None


def _needs_refit() -> bool:
    """early stopping の後に、学習 fold 全体で本数固定の再学習をするか。"""
    return EARLY_STOPPING_ON == "inner_refit"


def _refit_on_full_fold(Est, params: dict, X_tr, y_tr, best_iter: int | None,
                        n_key: str, fit_kwargs: dict | None = None):
    """**学習 fold 100% で、本数を固定して学習し直す。**

    `EARLY_STOPPING_ON="inner"` は検証 fold を覗かない代わりに、
    学習 fold からさらに 15% を抜く。つまり**最終モデルは全データの 0.8 × 0.85 = 68%**
    でしか学習していない。本数が決まった後に 80% で学習し直せばその分を取り戻せる。

    実測（合成データ・8 seed の対応比較・LightGBM）:

        Δ(refit − inner) = **+0.00122 ± 0.00049（z=+2.48）**
        内側の取り分 10% → +0.00128 / 15% → +0.00202 / 25% → +0.00261
        （抜く量が増えるほど効果も増える＝機構的に整合）
        学習時間は約 1.7 倍

    前コンペの「LB に現れる床」が 0.00013、Public 1 位と 645 位の差が 0.00024
    だったことを思えば、+0.0012 はその帯では大きい。
    """
    if best_iter is None:
        return None
    refit_params = dict(params)
    refit_params[n_key] = best_iter
    model = Est(**refit_params)
    model.fit(X_tr, y_tr, **(fit_kwargs or {}))
    return model


def _lgb_eval_kwargs(Est, X_es, y_es) -> dict:
    """LightGBM の検証データ引数を、インストール済みバージョンに合わせて作る。

    4.7 で `eval_set` が非推奨になり、使うたびに `LGBMDeprecationWarning` が出る。
    警告を放置すると本当に見るべき警告が埋もれるので、ここで吸収する
    （`pyproject.toml` は 4.6 以上を許すため、両方に対応する）。
    """
    import inspect

    if "eval_X" in inspect.signature(Est.fit).parameters:
        return {"eval_X": X_es, "eval_y": y_es}
    return {"eval_set": [(X_es, y_es)]}


def train_fold_lgb(X_tr, y_tr, X_val, y_val, params: dict):
    import lightgbm as lgb
    Est = lgb.LGBMRegressor if is_regression() else lgb.LGBMClassifier
    X_fit, y_fit, X_es, y_es = _split_for_fit(X_tr, y_tr, X_val, y_val, params)
    model = Est(**params)
    # LightGBM 4.7 で `eval_set` は非推奨（`eval_X` / `eval_y` へ移行）。
    # 4.6 以前も動くよう、シグネチャを見てから渡し方を決める。
    model.fit(X_fit, y_fit, **_lgb_eval_kwargs(Est, X_es, y_es),
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    if _needs_refit():
        refit = _refit_on_full_fold(Est, params, X_tr, y_tr,
                                    _best_iteration(model), "n_estimators")
        model = refit or model
    return model, _predict(model, X_val)


def train_fold_cb(X_tr, y_tr, X_val, y_val, params: dict):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    cat_features = [c for c in X_tr.columns if X_tr[c].dtype in ("object", "category")]
    Est = CatBoostRegressor if is_regression() else CatBoostClassifier
    X_fit, y_fit, X_es, y_es = _split_for_fit(X_tr, y_tr, X_val, y_val, params)
    model = Est(**params)
    model.fit(
        Pool(X_fit, y_fit, cat_features=cat_features),
        eval_set=Pool(X_es, y_es, cat_features=cat_features),
        early_stopping_rounds=50,
    )
    if _needs_refit():
        best = _best_iteration(model)
        if best:
            refit = Est(**{**params, "iterations": best})
            refit.fit(Pool(X_tr, y_tr, cat_features=cat_features))
            model = refit
    return model, _predict(model, X_val)


def train_fold_nn(X_tr, y_tr, X_val, y_val, params: dict):
    """pytabkit の RealMLP / TabM。**tree 系と同じ入口で扱えるようにする。**

    **なぜ第一級にするか**: 前コンペで使った RealMLP / TabM はすべて `kaggle_nb/` の
    アドホック実装で、`run_cv` / `feature_study` / `optimize_hp` の恩恵を受けていなかった。
    結果、**FE の 1 列 ΔOOF 計測が tree 系だけに対して行われ、特徴量セットが
    tree に偏って最適化される**（NN に効く FE は測られない）。
    上位解法は単体 NN か「NN を最良単体としたスタック」が主流で、ここが主戦場になる。

    early stopping は pytabkit が内部で行うので `_split_for_fit` の検証データを
    `val_idxs` として渡す（tree 系と同じ「検証 fold を覗かない」プロトコル）。

    **`inner_refit` は tree 系のみ。** NN は「本数を固定して学習し直す」に相当する
    操作が単純でない（エポック数を固定しても最適点が同じとは限らない）ので適用しない。
    """
    from pytabkit import (RealMLP_TD_Classifier, RealMLP_TD_Regressor,
                          TabM_D_Classifier, TabM_D_Regressor)

    # **呼び出し側の dict を壊さない。** `run_cv` は fold ループの外で params を 1 個作り、
    # 全 fold に同じ dict を渡す。ここで `pop` すると fold 1 以降は `_nn_kind` を失い、
    # **`--model tabm` が fold0 だけ TabM・残りは RealMLP** になる（実測）。
    # 例外も警告も出ず、log.csv には `tabm` と記録される。
    kind = params.get("_nn_kind", "realmlp")
    params = {k: v for k, v in params.items() if not k.startswith("_")}
    if kind == "tabm":
        Est = TabM_D_Regressor if is_regression() else TabM_D_Classifier
    else:
        Est = RealMLP_TD_Regressor if is_regression() else RealMLP_TD_Classifier

    X_fit, y_fit, X_es, y_es = _split_for_fit(X_tr, y_tr, X_val, y_val, params)
    # pytabkit は 1 つの行列と検証行のインデックスを受け取る形
    X_all = pd.concat([X_fit, X_es], axis=0, ignore_index=True)
    y_all = np.concatenate([np.asarray(y_fit), np.asarray(y_es)])
    val_idxs = np.arange(len(X_fit), len(X_all))

    model = Est(**params)
    model.fit(X_all, y_all, val_idxs=val_idxs)
    return model, _predict(model, X_val)


def train_fold_xgb(X_tr, y_tr, X_val, y_val, params: dict):
    import xgboost as xgb
    Est = xgb.XGBRegressor if is_regression() else xgb.XGBClassifier
    X_fit, y_fit, X_es, y_es = _split_for_fit(X_tr, y_tr, X_val, y_val, params)
    model = Est(**params, early_stopping_rounds=50)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        verbose=100,
    )
    if _needs_refit():
        # XGBoost は best_iteration が 0 始まりなので +1 して本数にする
        best = _best_iteration(model)
        refit = _refit_on_full_fold(Est, params, X_tr, y_tr,
                                    (best + 1) if best else None, "n_estimators",
                                    fit_kwargs={"verbose": False})
        model = refit or model
    return model, _predict(model, X_val)


TRAIN_FN = {"lgb": train_fold_lgb, "lgb_balanced": train_fold_lgb,
            "cb": train_fold_cb, "xgb": train_fold_xgb,
            "realmlp": train_fold_nn, "tabm": train_fold_nn}
MODEL_CHOICES = sorted(TRAIN_FN)



def extract_importance(model, features: list[str]) -> "np.ndarray | None":
    """特徴量重要度を**gain ベース**で取り出す。

    `sklearn` ラッパーの `feature_importances_` は、LightGBM では既定が
    `importance_type="split"`（分岐に使われた**回数**）。回数は「よく使われたか」は示すが
    「どれだけ損失を減らしたか」は示さない。カーディナリティの高い列が上位に来やすく、
    **文書・グラフの軸ラベルはどちらも "gain" と書いていた**ので、読む側は別物を見ていた。
    `G-DIAG` の第3診断軸（新特徴量が実際に効いたか）はこの値で判断するため、定義を揃える。
    """
    booster = getattr(model, "booster_", None)                  # LightGBM
    if booster is not None:
        return np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    if hasattr(model, "get_booster"):                           # XGBoost
        score = model.get_booster().get_score(importance_type="gain")
        return np.array([score.get(f, 0.0) for f in features], dtype=float)
    if hasattr(model, "get_feature_importance"):                # CatBoost
        # CatBoost の既定は PredictionValuesChange（gain 相当の寄与度）
        return np.asarray(model.get_feature_importance(), dtype=float)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    return None


def run_cv(model_name: str, params: dict, seed: int, features: list[str] = None,
           split_seed: int | None = None, n_splits: int | None = None):
    """1回分のCV学習を実行し、スコアとOOF/test予測を返す（log.csv記録なし）。

    multi-seed検証など、複数回まとめて呼び出して自前で集計したい場合に使う。
    単発のCV学習で log.csv に記録したい場合は main() を使う。

    Args:
        seed: **モデルの** seed（初期化・サンプリング）。
        split_seed: **分割の** seed。省略すると `RANDOM_STATE` 固定。
            モデル seed だけを振っても、**分割は毎回同じ**なので
            「この分割の上でたまたま良い」構成を選び続けることになる。
            前コンペで「OOF 有意なのに LB に再現しない」が 6 回続いた構造的な原因
            （行のブートストラップは分割由来の分散を再現しない → `src/noise.py`）。
        n_splits: fold 数。省略すると `N_SPLITS`。

    Returns:
        dict: train_scores, val_scores, oof_score, oof_preds, test_preds, y_true, covered
    """
    features = features or FEATURES
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")
    X, y_raw = train[features], train[TARGET_COL]
    X_test = test[features]

    if is_regression():
        y = pd.Series(y_raw.to_numpy(dtype=float), index=y_raw.index)
    else:
        y = pd.Series(LabelEncoder().fit_transform(y_raw), index=y_raw.index)

    seeded_params = ({**params, "random_seed": seed} if model_name == "cb"
                     else {**params, "random_state": seed})

    k = n_splits or N_SPLITS
    cv = get_cv(n_splits=k, seed=split_seed)   # 分割器は CV_STRATEGY から
    groups = get_groups(train)   # GroupKFold 系のみ。他は None
    metric = get_metric()
    n_cls = _resolve_n_classes(y)
    shape = (len(train),) if is_regression() else (len(train), n_cls)
    test_shape = (len(test),) if is_regression() else (len(test), n_cls)
    oof_preds = np.zeros(shape)
    test_preds = np.zeros(test_shape)
    # TimeSeriesSplit では先頭の行がどの fold の検証にも入らない。予測ゼロのまま
    # 正解と突き合わせると「全部クラス0と予測した」ことになり、**スコアが実力と無関係に動く**。
    # どの行が実際に予測されたかを持っておき、評価と保存の両方でそれを尊重する。
    covered = np.zeros(len(train), dtype=bool)
    train_scores, val_scores = [], []
    importances = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model, val_pred = TRAIN_FN[model_name](X_tr, y_tr, X_val, y_val, seeded_params)
        oof_preds[val_idx] = val_pred
        covered[val_idx] = True
        test_preds += _predict(model, X_test) / k

        val_score = metric(y_val, shape_for_metric(val_pred))
        tr_pred = model.predict_proba(X_tr) if _has_proba(model) else model.predict(X_tr)
        tr_score = metric(y_tr, shape_for_metric(tr_pred))
        train_scores.append(tr_score)
        val_scores.append(val_score)

        imp = extract_importance(model, features)
        if imp is not None:
            importances.append(imp)

    oof_score = metric(y[covered], shape_for_metric(oof_preds[covered]))
    if not covered.all():
        print(f"  ℹ️ OOF は予測された {covered.sum()} / {len(covered)} 行で評価しました"
              f"（{CV_STRATEGY} は先頭を検証に使いません）")
        oof_preds = oof_preds.astype(float)
        oof_preds[~covered] = np.nan   # 保存側で「未予測」と「クラス0」を混同させない

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
        # 呼び出し側が床を実測できるように、正解と「どの行を予測したか」も返す
        # （`src/noise.py` の対応差ブートストラップに必要）
        "y_true": y.to_numpy(),
        "covered": covered,
        "importance_df": importance_df,
    }


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgb", choices=MODEL_CHOICES)
    parser.add_argument("--params", type=str, default="",
                        help="best_params JSON ファイルパス（省略時はデフォルトパラメータを使用）")
    parser.add_argument("--resume", action="store_true",
                        help="fold 単位のキャッシュを使い、中断した学習を途中から再開する")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS,
                        help="fold 数（既定は config の N_SPLITS）。10-fold にすると "
                             "各モデルが 90%% を学習に使え、床も下がる")
    parser.add_argument("--early-stopping", type=str, default=None,
                        choices=["inner_refit", "inner", "val"],
                        help="early stopping の扱い（既定は config の EARLY_STOPPING_ON）。"
                             "スクリーニングで速さが要るときだけ inner に落とす")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="**分割の** seed（既定は RANDOM_STATE 固定）。"
                             "モデル seed だけ振っても分割は同じままなので、"
                             "「この分割の上でたまたま良い」構成を選び続けることになる")
    args = parser.parse_args()

    assert FEATURES, "FEATURES リストが空です。scripts/train.py の TODO を埋めてください。"

    if args.early_stopping:
        # モジュール変数を差し替える（`_split_for_fit` / `_needs_refit` が参照する）
        global EARLY_STOPPING_ON
        EARLY_STOPPING_ON = args.early_stopping

    # データ読み込み
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")
    X, y_raw = train[FEATURES], train[TARGET_COL]
    X_test = test[FEATURES]

    # ターゲットの整形。分類はラベルエンコード（文字列クラス → 0..n_classes-1）、
    # 回帰はそのまま使う（LabelEncoder に連続値を通すと**全値がユニークな「クラス」になる**）。
    if is_regression():
        y = pd.Series(y_raw.to_numpy(dtype=float), index=y_raw.index)
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
        print(f"クラス対応: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"評価設定: {describe_setup()}")

    # パラメータ読み込み。クラス数が実データ依存（multiclass）なら y から組み立て直す
    params = build_params(args.model, _resolve_n_classes(y)).copy()
    if args.params:
        with open(args.params) as f:
            params.update(json.load(f))

    # 実験トラッキング
    tracker = ExperimentTracker(
        experiment_name=EXPERIMENT_NAME,
        model=args.model,
        features=f"{len(FEATURES)}features",
    )
    split_note = (f"{args.n_splits}-fold"
                  + (f" / split_seed={args.split_seed}" if args.split_seed is not None else ""))
    tracker.start_run(description=f"{args.model} CV学習（{len(FEATURES)}特徴量 / {split_note}）")
    tracker.log_params(params)

    # CV学習
    cv = get_cv(n_splits=args.n_splits, seed=args.split_seed)
    groups = get_groups(train)   # GroupKFold 系のみ。他は None
    metric = get_metric()  # 指標は EVAL_METRIC から決まる（optimize_hp と共有）
    n_cls = _resolve_n_classes(y)
    shape = (len(train),) if is_regression() else (len(train), n_cls)
    test_shape = (len(test),) if is_regression() else (len(test), n_cls)
    oof_preds = np.zeros(shape)
    test_preds = np.zeros(test_shape)
    covered = np.zeros(len(train), dtype=bool)   # run_cv と同じ理由（未予測行を評価に混ぜない）
    importances = []
    train_scores, val_scores = [], []

    # fold 単位のチェックポイント（--resume 時のみ有効）。
    # 中断・クラッシュで既に終わった fold の計算が失われるのを防ぐ（CONVENTIONS の実行規約）。
    # signature に特徴量と HP を渡す。tag だけだと「モデル名 + 特徴量の本数」しか
    # 区別せず、列を入れ替えても HP を変えても古い fold が再利用される。
    # 分割の条件もキャッシュのシグネチャに入れる（fold 数や分割 seed が変われば別物）
    cache = FoldCache(tag=f"{args.model}_{len(FEATURES)}f", seed=RANDOM_STATE,
                      n_splits=args.n_splits, enabled=args.resume,
                      signature={"features": FEATURES, "params": params,
                                 "n_splits": args.n_splits, "split_seed": args.split_seed})
    if args.resume:
        print(cache.report())

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        cached = cache.load(fold)
        if cached is not None:
            val_pred, fold_test_pred = cached
            model = None
        else:
            model, val_pred = TRAIN_FN[args.model](X_tr, y_tr, X_val, y_val, params)
            fold_test_pred = _predict(model, X_test)
            cache.save(fold, val_pred, fold_test_pred)

        oof_preds[val_idx] = val_pred
        covered[val_idx] = True
        test_preds += fold_test_pred / args.n_splits   # テスト予測（フォールド平均）

        val_score = metric(y_val, shape_for_metric(val_pred))
        # キャッシュ再利用時は train を再計算できない。**以前は val と同じ値を入れていた**が、
        # それは `cv_train_mean` に嘘を書くのと同じで、`G-DIAG` の第1診断軸
        # （train − val 乖離で過学習か校正不足かを切り分ける）が --resume のたびに
        # 「乖離ゼロ」に化けていた。測れないものは NaN にして、集計から除く。
        if model is None:
            tr_score = float("nan")
        else:
            tr_score = metric(y_tr, shape_for_metric(_predict(model, X_tr)))
        train_scores.append(tr_score)
        val_scores.append(val_score)
        tracker.log_fold_scores(fold, tr_score, val_score)

        # 特徴量重要度
        imp = extract_importance(model, FEATURES) if model is not None else None
        if imp is not None:
            importances.append(imp)

        mark = "（キャッシュ再利用）" if model is None else ""
        print(f"Fold {fold}: train={tr_score:.5f}  val={val_score:.5f} {mark}")

    # OOFスコア。予測されなかった行（TimeSeriesSplit の先頭）は評価から外す
    oof_score = metric(y[covered], shape_for_metric(oof_preds[covered]))
    if not covered.all():
        print(f"ℹ️ OOF は予測された {covered.sum()} / {len(covered)} 行で評価しました"
              f"（{CV_STRATEGY} は先頭を検証に使いません）")
        oof_preds = oof_preds.astype(float)
        oof_preds[~covered] = np.nan

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
