---
description: 実験記録の規約（scripts/**とexperiments/runs/**に適用）
paths: ["scripts/**/*.py", "experiments/runs/**/*.py"]
---

# 実験トラッキング規約

## scripts/train.py での必須記録

すべての学習実行は `src.experiment` モジュールを通じて `experiments/log.csv` に記録する。

```python
from src.experiment import ExperimentTracker
from src.config import EXPERIMENT_NAME

tracker = ExperimentTracker(experiment_name=EXPERIMENT_NAME)
run_id = tracker.start_run(
    run_name="fold_training",
    description="実験の概略をここに記述",
    model="lgb",
    features="baseline_features",
)

# 学習ループ内で記録
tracker.log_fold_scores(fold, trn_score, val_score)

# 学習完了後
tracker.end_run(
    train_scores=train_scores,
    val_scores=val_scores,
    oof_score=oof_score,
    n_features=len(features),
)
```

## 実験名の命名規則

`{competition}_{model}_{feature_set}_{variant}`

例:
- `playground-series-s6e6_lgb_baseline`
- `playground-series-s6e6_lgb_fe_v2`
- `playground-series-s6e6_cb_optuna`
- `playground-series-s6e6_ensemble_lgb_cb`

## log.csvへの記録

`tracker.end_run()` が自動で `experiments/log.csv` に追記する。
`submit_score` と `lb_rank` は `/kaggle-submit` スキル実行後に自動追記される。

## experiments/runs/ スクリプトでの記録

`experiments/runs/exp{NNN}_s{stage}_{内容}.py` では、
`scripts/train.py` などを呼び出すか、同様のトラッキングコードを含める。

```python
# experiments/runs/exp042_s4_fe_col_a_interaction.py の例
from src.experiment import ExperimentTracker

tracker = ExperimentTracker(experiment_name="lgb_fe_col_a_interaction")
# ... 学習・記録 ...
tracker.end_run(...)
# end_run() がコミットメッセージの雛形を提案する:
# feat(exp042): col_A×col_Bの交互作用特徴量を追加
# OOF=0.91688  model=lgb  features=fe_v7_interaction
```
