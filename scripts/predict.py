"""
保存済み test 予測 → 提出ファイル生成スクリプト

保存済みの test 予測（`.npy`）から提出 CSV を生成する。
`submission_path()` の命名規約に従いファイル名を自動生成する。

**提出形式は `src/config.py` の `EVAL_METRIC` / `PROBLEM_TYPE` から決まる**
（`src.utils.finalize` が唯一の定義元）。以前はこのスクリプトだけが指標を見ず、
**常に `argmax` でハードラベルを書いていた** —— AUC コンペでは実測 **−0.074**。
しかも `feature_study.py` と `blend.py` が「次のステップ」としてこのコマンドを案内するので、
テンプレートの導線に沿って進むと必ずそこへ落ちる構造だった。

使い方:
    uv run python -m scripts.predict --test-npy data/output/oof/test_001_lgb.npy \\
        --model lgb --oof-score 0.44485 --exp-id 001
"""

import argparse

import numpy as np

from src.metrics import describe as describe_setup
from src.utils.finalize import save_run_outputs


def main():
    parser = argparse.ArgumentParser(description="保存済み test 予測から提出ファイルを生成する")
    parser.add_argument("--test-npy", type=str, required=True,
                        help="test 予測の .npy ファイルパス（分類は確率配列、回帰は 1 次元）")
    parser.add_argument("--model", type=str, required=True,
                        help="モデル識別子（例: lgb, cb, lgb_cb_blend）")
    parser.add_argument("--oof-score", type=float, required=True,
                        help="OOF スコア（ファイル名に埋め込まれる。指標は EVAL_METRIC）")
    parser.add_argument("--exp-id", type=str, default="",
                        help="experiment_id（log.csv と紐付け）")
    parser.add_argument("--oof-npy", type=str, default="",
                        help="OOF 予測の .npy（省略時は test 予測から再保存しない）")
    parser.add_argument("--is-ensemble", action="store_true",
                        help="派生アンサンブルならファイル名に _ens_ を入れる（プールの自己参照防止）")
    args = parser.parse_args()

    print(f"評価設定: {describe_setup()}")
    test = np.load(args.test_npy)
    oof = np.load(args.oof_npy) if args.oof_npy else np.empty(0)

    # 提出形式の判定・ID 整合の確認・命名規約は finalize が持つ（定義元を 1 つにする）
    save_run_outputs(
        exp_id=args.exp_id or "000",
        model=args.model,
        oof=oof,
        test=test,
        oof_score=args.oof_score,
        is_ensemble=args.is_ensemble,
        save_npy=False,          # npy は既にあるので上書きしない
    )
    print("次のステップ: /ds-kaggle-submit で提出してください")


if __name__ == "__main__":
    main()
