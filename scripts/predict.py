"""
OOF予測 → 提出ファイル生成スクリプト

保存済みの test 予測（.npy, shape=(N, N_CLASSES) の確率配列）から提出CSVを生成する。
submission_path() の命名規約に従いファイル名を自動生成する。

multiclass の場合は argmax でクラスを決定し、ラベル文字列に戻して出力する。

使い方:
    uv run python -m scripts.predict --test-npy data/output/oof/test_001_lgb.npy --model lgb --oof-score 0.44485 --exp-id 001
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import ID_COL, RAW_DATA_DIR, TARGET_COL, submission_path

# ──────────────────────────────────────────────
# TODO: コンペごとにここを変更する
# ──────────────────────────────────────────────



def main():
    parser = argparse.ArgumentParser(description="提出ファイルを生成する")
    parser.add_argument("--test-npy", type=str, required=True,
                        help="test予測の .npy ファイルパス（shape=(N, N_CLASSES) の確率配列）")
    parser.add_argument("--model", type=str, required=True,
                        help="モデル識別子（例: lgb, cb, lgb_cb_blend）")
    parser.add_argument("--oof-score", type=float, required=True,
                        help="OOF balanced_accuracy スコア（ファイル名に埋め込まれる）")
    parser.add_argument("--exp-id", type=str, default="",
                        help="experiment_id（log.csv と紐付け）")
    args = parser.parse_args()

    # 予測値読み込み（確率配列 → argmaxでクラスindex決定）
    test_probs = np.load(args.test_npy)
    pred_idx = np.argmax(test_probs, axis=1)

    # ラベルエンコードの復元（train.csv のクラス順で fit し直す）
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    le = LabelEncoder()
    le.fit(train[TARGET_COL])
    pred_labels = le.inverse_transform(pred_idx)

    # サンプル提出ファイルからID列を取得
    sample_sub_path = RAW_DATA_DIR / "sample_submission.csv"
    sample = pd.read_csv(sample_sub_path)
    sub = pd.DataFrame({
        ID_COL: sample[ID_COL],
        TARGET_COL: pred_labels,
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
 クラス分布: {sub[TARGET_COL].value_counts().to_dict()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
次のステップ: /ds-kaggle-submit で提出してください
""")


if __name__ == "__main__":
    main()
