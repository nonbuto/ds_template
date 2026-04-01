"""
EDA可視化スクリプト

/eda-visual スキルから呼び出す。画像を data/output/plots/ に保存し、
Claude が Read ツールで読み込んで対話に使う。

使い方:
    uv run python scripts/visualize.py --var tenure --theme target_dist
    uv run python scripts/visualize.py --theme overview
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非インタラクティブ（ファイル保存のみ）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PROCESSED_DATA_DIR, PLOTS_DIR, TARGET_COL

sns.set_theme(style="whitegrid", palette="Set2")


# ──────────────────────────────────────────────
# 個別テーマの可視化関数
# ──────────────────────────────────────────────

def plot_target_distribution(train: pd.DataFrame, out: Path) -> Path:
    """ターゲット分布（クラスバランス）"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = train[TARGET_COL].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values)
    axes[0].set_title("ターゲット件数")
    axes[0].set_xlabel(TARGET_COL)

    rates = train[TARGET_COL].value_counts(normalize=True).sort_index()
    axes[1].pie(rates.values, labels=[f"{k} ({v:.1%})" for k, v in zip(rates.index, rates.values)],
                autopct="%1.1f%%")
    axes[1].set_title("ターゲット比率")

    fig.suptitle("ターゲット変数の分布", fontsize=13)
    fig.tight_layout()
    path = out / "eda_target_distribution.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


def plot_missing_values(train: pd.DataFrame, out: Path) -> Path:
    """欠損値ヒートマップ"""
    missing = train.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("欠損値なし — スキップ")
        return out / "eda_missing_values.png"

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    missing.plot.barh(ax=ax)
    ax.set_xlabel("欠損率")
    ax.set_title("列ごとの欠損率")
    ax.axvline(0.1, color="red", linestyle="--", alpha=0.5, label="10%")
    ax.axvline(0.5, color="orange", linestyle="--", alpha=0.5, label="50%")
    ax.legend()
    fig.tight_layout()
    path = out / "eda_missing_values.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


def plot_target_dist_by_var(train: pd.DataFrame, var: str, out: Path) -> Path:
    """数値変数 × ターゲット: KDE/箱ひげ図"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, grp in train.groupby(TARGET_COL):
        sns.kdeplot(grp[var], ax=axes[0], label=f"{TARGET_COL}={label}", fill=True, alpha=0.4)
    axes[0].set_title(f"{var} の分布（ターゲット別）")
    axes[0].legend()

    sns.boxplot(data=train, x=TARGET_COL, y=var, ax=axes[1])
    axes[1].set_title(f"{var} の箱ひげ図（ターゲット別）")

    fig.suptitle(f"{var} × {TARGET_COL}", fontsize=13)
    fig.tight_layout()
    path = out / f"eda_{var}_target_dist.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


def plot_target_rate_by_cat(train: pd.DataFrame, var: str, out: Path) -> Path:
    """カテゴリ変数 × ターゲット率: 棒グラフ（件数併記）"""
    grp = train.groupby(var)[TARGET_COL].agg(["mean", "count"]).reset_index()
    grp.columns = [var, "target_rate", "count"]
    grp = grp.sort_values("target_rate", ascending=False)

    fig, ax1 = plt.subplots(figsize=(max(8, len(grp) * 1.2), 4))
    ax2 = ax1.twinx()

    ax1.bar(grp[var].astype(str), grp["target_rate"], alpha=0.7, color="steelblue", label="ターゲット率")
    ax2.plot(grp[var].astype(str), grp["count"], "o-", color="tomato", label="件数")
    ax1.set_ylabel("ターゲット率")
    ax2.set_ylabel("件数")
    ax1.set_title(f"{var} × {TARGET_COL}率（棒: 率, 折れ線: 件数）")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    path = out / f"eda_{var}_target_rate.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


def plot_train_test_compare(train: pd.DataFrame, test: pd.DataFrame, var: str, out: Path) -> Path:
    """Train/Test 分布比較（KDE重ね合わせ or 比率棒グラフ）"""
    if pd.api.types.is_numeric_dtype(train[var]):
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.kdeplot(train[var], ax=ax, label="Train", fill=True, alpha=0.4)
        sns.kdeplot(test[var], ax=ax, label="Test", fill=True, alpha=0.4)
        ax.set_title(f"{var}: Train vs Test 分布比較")
        ax.legend()
    else:
        tr_r = train[var].value_counts(normalize=True).rename("Train")
        te_r = test[var].value_counts(normalize=True).rename("Test")
        df = pd.concat([tr_r, te_r], axis=1).fillna(0)
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 4))
        df.plot.bar(ax=ax)
        ax.set_title(f"{var}: Train vs Test 比率比較")
        ax.set_ylabel("比率")
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    path = out / f"eda_{var}_train_test.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


# ──────────────────────────────────────────────
# メインエントリ
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA可視化スクリプト")
    parser.add_argument("--var", type=str, default="", help="対象変数名")
    parser.add_argument("--theme", type=str, default="target_dist",
                        choices=["target_dist", "target_rate", "missing", "train_test", "overview"],
                        help="可視化テーマ")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    test_path = PROCESSED_DATA_DIR / "test_features.pkl"
    test = pd.read_pickle(test_path) if test_path.exists() else None

    if args.theme == "overview":
        plot_target_distribution(train, PLOTS_DIR)
        plot_missing_values(train, PLOTS_DIR)
    elif args.theme == "missing":
        plot_missing_values(train, PLOTS_DIR)
    elif args.theme == "target_dist":
        assert args.var, "--var が必要です"
        plot_target_dist_by_var(train, args.var, PLOTS_DIR)
    elif args.theme == "target_rate":
        assert args.var, "--var が必要です"
        plot_target_rate_by_cat(train, args.var, PLOTS_DIR)
    elif args.theme == "train_test":
        assert args.var and test is not None, "--var と test データが必要です"
        plot_train_test_compare(train, test, args.var, PLOTS_DIR)


if __name__ == "__main__":
    main()
