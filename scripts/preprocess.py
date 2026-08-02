"""
前処理スクリプト

train/test の全カラムを保持したまま、以下を行い
train_features.pkl / test_features.pkl として保存する:
  - 数値カラムの欠損値を中央値で埋める
  - カテゴリカラムは欠損値を "missing" カテゴリとして明示し、pandas category 型に変換
    （LightGBM が categorical_feature としてネイティブに扱えるようにする）

全カラムを保持し削除しない方針（CLAUDE.md: 特徴量の全保持・学習時選択アーキテクチャ）。
学習時に使う列は scripts/train.py の FEATURES で選択する。

使い方:
    uv run python -m scripts.preprocess
"""

import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# TODO: コンペごとに埋める
NUMERIC_COLS: list[str] = []

CATEGORICAL_COLS: list[str] = []


def main():
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "test.csv")

    medians = train[NUMERIC_COLS].median()
    train[NUMERIC_COLS] = train[NUMERIC_COLS].fillna(medians)
    test[NUMERIC_COLS] = test[NUMERIC_COLS].fillna(medians)

    for col in CATEGORICAL_COLS:
        train[col] = train[col].fillna("missing").astype("category")
        # train/testでカテゴリの水準を揃える（testにしか無い値・train基準の欠損対応も含む）
        categories = train[col].cat.categories
        test[col] = test[col].fillna("missing").astype(
            pd.CategoricalDtype(categories=categories)
        )

    train.to_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test.to_pickle(PROCESSED_DATA_DIR / "test_features.pkl")

    print(f"train: {train.shape}, test: {test.shape}")
    print(f"saved to {PROCESSED_DATA_DIR}")
    for col in CATEGORICAL_COLS:
        print(f"  {col}: {list(train[col].cat.categories)}")


if __name__ == "__main__":
    main()
