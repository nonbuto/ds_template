"""
AV診断（Adversarial Validation）

train/test を結合し is_test を予測することで、分布シフトの有無を診断する。
PLAYBOOK.md#av-診断adversarial-validation 参照。Stage 4でFEが一段落した時点、
Stage 6移行前に必ず実施する。

使い方:
    uv run python -m scripts.av_check
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from src.config import PROCESSED_DATA_DIR, PLOTS_DIR, RANDOM_STATE, N_SPLITS
from scripts.train import FEATURES


def main():
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")

    X_train = train[FEATURES].copy()
    X_test = test[FEATURES].copy()

    X_av = pd.concat([X_train, X_test], ignore_index=True)
    y_av = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(X_av))
    importances = []

    params = dict(
        objective="binary",
        metric="auc",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_av, y_av)):
        X_tr, X_val = X_av.iloc[tr_idx], X_av.iloc[val_idx]
        y_tr, y_val = y_av[tr_idx], y_av[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
        importances.append(model.feature_importances_)

    av_auc = roc_auc_score(y_av, oof_pred)

    imp_df = pd.DataFrame(
        {"feature": FEATURES, "importance": np.mean(importances, axis=0)}
    ).sort_values("importance", ascending=False)
    imp_df.to_csv(PLOTS_DIR / "av_importance.csv", index=False)

    if av_auc < 0.55:
        verdict = "✅ シフトなし"
    elif av_auc < 0.65:
        verdict = "🔶 軽度シフト（importance weight 試行価値あり）"
    elif av_auc < 0.80:
        verdict = "⚠️ 中度シフト（上位重要度特徴量を drop 検討）"
    else:
        verdict = "❌ 強いシフト（drop 必須 or データ前処理の見直し）"

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AV診断結果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 特徴量数: {len(FEATURES)}
 AV-AUC  : {av_auc:.5f}
 判定    : {verdict}

 importance top5 (is_test予測への寄与度):
{imp_df.head(5).to_string(index=False)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
