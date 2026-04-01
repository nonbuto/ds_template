"""
OOF予測 → 提出ファイル生成スクリプト

保存済みの test 予測（.npy）から提出CSVを生成する。
submission_path() の命名規約に従いファイル名を自動生成する。

使い方:
    # 単一モデルの提出ファイル
    uv run python scripts/predict.py --test-npy data/output/oof/test_042_lgb.npy --model lgb --oof-score 0.91688

    # 複数モデルのブレンド済み予測から提出ファイル
    uv run python scripts/predict.py --test-npy data/output/oof/blend_171.npy --model lgb_cb_blend --oof-score 0.91777 --exp-id 171
"""

import argparse

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, submission_path

# ──────────────────────────────────────────────
# TODO: コンペごとにここを変更する
# ──────────────────────────────────────────────
ID_COL = "id"         # TODO: 提出ファイルのID列名
TARGET_COL_OUT = "target"  # TODO: 提出ファイルのターゲット列名（コンペの要求に合わせる）


def main():
    parser = argparse.ArgumentParser(description="提出ファイルを生成する")
    parser.add_argument("--test-npy", type=str, required=True,
                        help="test予測の .npy ファイルパス")
    parser.add_argument("--model", type=str, required=True,
                        help="モデル識別子（例: lgb, cb, lgb_cb_blend）")
    parser.add_argument("--oof-score", type=float, required=True,
                        help="OOF AUCスコア（ファイル名に埋め込まれる）")
    parser.add_argument("--exp-id", type=str, default="",
                        help="experiment_id（log.csv と紐付け）")
    parser.add_argument("--clip", type=float, nargs=2, default=[0.01, 0.99],
                        help="予測値のクリップ範囲（デフォルト: 0.01 0.99）")
    args = parser.parse_args()

    # 予測値読み込み
    test_preds = np.load(args.test_npy)
    test_preds = np.clip(test_preds, args.clip[0], args.clip[1])

    # サンプル提出ファイルからID列を取得
    sample_sub_path = RAW_DATA_DIR / "sample_submission.csv"
    if sample_sub_path.exists():
        sample = pd.read_csv(sample_sub_path)
        sub = pd.DataFrame({
            ID_COL: sample[ID_COL],
            TARGET_COL_OUT: test_preds,
        })
    else:
        # sample_submission.csv がない場合は連番IDで生成
        print("⚠ sample_submission.csv が見つかりません。連番IDで生成します。")
        sub = pd.DataFrame({
            ID_COL: range(len(test_preds)),
            TARGET_COL_OUT: test_preds,
        })

    # 提出ファイルの保存
    out_path = submission_path(
        model=args.model,
        oof_score=args.oof_score,
        exp_id=args.exp_id,
    )
    sub.to_csv(out_path, index=False)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 提出ファイル生成完了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ファイル  : {out_path.name}
 件数      : {len(sub):,}
 予測値    : {test_preds.min():.4f} 〜 {test_preds.max():.4f}
 平均      : {test_preds.mean():.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
次のステップ: /kaggle-submit で提出してください
""")


if __name__ == "__main__":
    main()
