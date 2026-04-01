"""
1列ΔCV計測スクリプト（Stage 4: 段階的FE用）

ベースとなる特徴量セットに1列を追加し、OOFスコアの変化（ΔOOF）を計測する。
/fe-hypothesis で仮説を立案した後、このスクリプトで効果を測定する。

使い方:
    uv run python scripts/feature_study.py --new-feature tenure_monthly_ratio
    uv run python scripts/feature_study.py --new-feature tenure_monthly_ratio --model cb

結果の読み方:
    ΔOOF = new_oof - base_oof
    +0.0003 以上: 採用を強く推奨
    +0.0001〜+0.0003: 採用を検討（他のモデルでも確認）
    ±0.0001 以内: ノイズ範囲（採用不要）
    マイナス: 棄却
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import (
    PROCESSED_DATA_DIR, OOF_DIR, PARAMS_DIR,
    RANDOM_STATE, N_SPLITS, TARGET_COL,
)

# ──────────────────────────────────────────────
# TODO: ベース特徴量セットをここに定義する
# ──────────────────────────────────────────────
# 実験ごとに更新する。src/config.py の FEATURES 定数を使うことを推奨。
BASE_FEATURES: list[str] = []  # 空のまま実行するとエラーになる


def cv_score(X: pd.DataFrame, y: pd.Series, params: dict, model_type: str = "lgb") -> tuple[float, float]:
    """5-fold CV の OOF AUC を返す。(oof_score, std) のタプル。"""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if model_type == "lgb":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
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
        else:
            raise ValueError(f"未対応モデル: {model_type}")

        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    scores = [roc_auc_score(y.iloc[val_idx], oof[val_idx])
              for _, val_idx in StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                                random_state=RANDOM_STATE).split(X, y)]
    return float(roc_auc_score(y, oof)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser(description="1列追加のΔOOF計測")
    parser.add_argument("--new-feature", type=str, required=True, help="追加する特徴量の列名")
    parser.add_argument("--model", type=str, default="lgb", choices=["lgb", "cb", "xgb"])
    parser.add_argument("--params", type=str, default="",
                        help="作業用HPのJSONファイルパス（Stage 3 で確定したもの）")
    args = parser.parse_args()

    assert BASE_FEATURES, "BASE_FEATURES が空です。scripts/feature_study.py の TODO を埋めてください。"

    # データ読み込み
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    y = train[TARGET_COL]

    # パラメータ（作業用HP）
    params_path = args.params or str(PARAMS_DIR / f"working_params_{args.model}.json")
    if Path(params_path).exists():
        with open(params_path) as f:
            params = json.load(f)
        print(f"作業用HP読み込み: {params_path}")
    else:
        # フォールバック: 軽量デフォルトパラメータ
        params = {"n_estimators": 500, "learning_rate": 0.05, "random_state": RANDOM_STATE,
                  "verbose": -1} if args.model == "lgb" else {}
        print("⚠ 作業用HPファイルが見つかりません。デフォルトパラメータを使用します。")
        print("  Stage 3 で scripts/optimize_hp.py を実行して作業用HPを確定してください。")

    # ベーススコアの計算
    print(f"\n【ベーススコア計算中】 特徴量数: {len(BASE_FEATURES)}")
    assert args.new_feature not in BASE_FEATURES, f"{args.new_feature} は既にベース特徴量に含まれています"
    X_base = train[BASE_FEATURES]
    base_oof, base_std = cv_score(X_base, y, params, args.model)
    print(f"  Base OOF: {base_oof:.5f} ± {base_std:.5f}")

    # 追加スコアの計算
    assert args.new_feature in train.columns, \
        f"{args.new_feature} が学習データに存在しません。特徴量生成スクリプトを先に実行してください。"
    print(f"\n【+{args.new_feature} スコア計算中】 特徴量数: {len(BASE_FEATURES) + 1}")
    X_new = train[BASE_FEATURES + [args.new_feature]]
    new_oof, new_std = cv_score(X_new, y, params, args.model)
    delta = new_oof - base_oof
    print(f"  New  OOF: {new_oof:.5f} ± {new_std:.5f}")

    # 判定
    if delta > 0.0003:
        verdict = "✅ 採用推奨"
    elif delta > 0.0001:
        verdict = "🔶 採用検討（他モデルでも確認を推奨）"
    elif delta > -0.0001:
        verdict = "⬜ ノイズ範囲（採用不要）"
    else:
        verdict = "❌ 棄却"

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 特徴量追加効果の計測結果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 特徴量    : {args.new_feature}
 モデル    : {args.model}
 Base OOF  : {base_oof:.5f} ± {base_std:.5f}
 New  OOF  : {new_oof:.5f} ± {new_std:.5f}
 ΔOOF      : {delta:+.5f}
 判定      : {verdict}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
次のステップ: /fe-hypothesis update H-NNN で結果を記録し、
             FEATURE_REPORT.md を更新してください。
""")


if __name__ == "__main__":
    main()
