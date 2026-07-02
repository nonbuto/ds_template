"""
EDA可視化スクリプト

/ds-eda-visual スキルから呼び出す。画像を data/output/plots/ に保存し、
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
# モデル/Blend 評価系プロット (Stage 4-6 向け)
# 「早期却下の禁止」原則のために必須
# ──────────────────────────────────────────────

def plot_feature_importance(fi_series: pd.Series, out: Path, title: str = "Feature importance (gain)", top_n: int = 30) -> Path:
    """LGB/XGB feature_importances_ を棒グラフで可視化

    使い方:
        fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        plot_feature_importance(fi, PLOTS_DIR, title="exp042 importance")
    """
    top = fi_series.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.25)))
    top[::-1].plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance (gain)")
    fig.tight_layout()
    path = out / f"feature_importance_{title.replace(' ', '_')}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_oof_distribution(y_true, oof_old, oof_new, out: Path, label_old: str = "old", label_new: str = "new") -> Path:
    """新旧 OOF 予測の分布比較 (前後評価)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # クラス別に分けてヒストグラム
    for ax, oof, lbl in zip(axes, [oof_old, oof_new], [label_old, label_new]):
        ax.hist(oof[y_true == 0], bins=50, alpha=0.5, label="class=0", density=True)
        ax.hist(oof[y_true == 1], bins=50, alpha=0.5, label="class=1", density=True)
        ax.set_title(f"{lbl}")
        ax.set_xlabel("Predicted probability")
        ax.legend()
    fig.tight_layout()
    path = out / f"oof_distribution_{label_old}_vs_{label_new}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_matrix(oof_dict: dict, out: Path, title: str = "OOF correlation") -> Path:
    """複数モデルの OOF 相関マトリクス (blend overfit / 冗長性チェック)

    使い方:
        oofs = {"lgb": oof_lgb, "xgb": oof_xgb, "cb": oof_cb}
        plot_correlation_matrix(oofs, PLOTS_DIR)
    """
    names = list(oof_dict.keys())
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = np.corrcoef(oof_dict[a], oof_dict[b])[0, 1]
    df_mat = pd.DataFrame(mat, index=names, columns=names)
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), max(5, len(names) * 0.5)))
    sns.heatmap(df_mat, annot=True, fmt=".3f", cmap="RdYlBu_r", vmin=0.95, vmax=1.0, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title(title)
    fig.tight_layout()
    path = out / f"correlation_matrix_{title.replace(' ', '_')}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lb_history(log_csv_path: Path, out: Path, title: str = "LB history") -> Path:
    """experiments/log.csv の submit_score を実験順にプロット (LB プラトー検出用)"""
    df = pd.read_csv(log_csv_path)
    df = df.dropna(subset=["submit_score"])
    df = df[df["submit_score"] > 0].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(df)), df["submit_score"], marker="o", linestyle="-")
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.annotate(row["experiment_id"], (i, row["submit_score"]), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Submission order")
    ax.set_ylabel("Public LB")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "lb_history.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# メインエントリ
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA可視化スクリプト")
    parser.add_argument("--var", type=str, default="", help="対象変数名")
    parser.add_argument("--theme", type=str, default="target_dist",
                        choices=["target_dist", "target_rate", "missing", "train_test", "overview", "lb_history"],
                        help="可視化テーマ")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.theme == "lb_history":
        from src.config import EXPERIMENTS_DIR
        plot_lb_history(EXPERIMENTS_DIR / "log.csv", PLOTS_DIR)
        return

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
