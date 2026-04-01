"""
特徴量レポート生成スクリプト

FEATURE_REPORT.md の可視化セクションに対応する画像を生成する。
Claude が Read ツールで読んで対話に使う。

使い方:
    uv run python scripts/feature_report.py
    uv run python scripts/feature_report.py --theme importance  # 重要度のみ
    uv run python scripts/feature_report.py --theme delta       # ΔOOF棒グラフのみ
    uv run python scripts/feature_report.py --theme corr        # 相関ヒートマップのみ
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR, OOF_DIR, PARAMS_DIR, PROCESSED_DATA_DIR

sns.set_theme(style="whitegrid", palette="Set2")


def plot_feature_importance(importance_path: Path, out: Path) -> Path:
    """LGB/CB特徴量重要度（gainベース）の棒グラフ。

    importance_path: feature_importance_{exp_id}.csv
        期待カラム: feature, importance
    """
    if not importance_path.exists():
        print(f"⚠ 重要度ファイルが見つかりません: {importance_path}")
        return out / "feature_importance_current.png"

    df = pd.read_csv(importance_path).sort_values("importance", ascending=True).tail(30)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    ax.barh(df["feature"], df["importance"])
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"特徴量重要度 Top{len(df)}")
    fig.tight_layout()
    path = out / "feature_importance_current.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


def plot_delta_oof(feature_report_md: Path, out: Path) -> Path:
    """FEATURE_REPORT.md のエンジニアリング済み特徴量テーブルからΔOOF棒グラフを生成。"""
    # Markdown テーブルをパースする（簡易実装）
    lines = feature_report_md.read_text().splitlines()
    rows = []
    in_table = False
    for line in lines:
        if "エンジニアリング済み特徴量" in line:
            in_table = True
            continue
        if in_table and line.startswith("|") and "---" not in line and "特徴量名" not in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 6 and not parts[0].startswith("*(例)"):
                try:
                    delta = float(parts[3])
                    status = "採用" if "✅" in parts[4] else "棄却"
                    rows.append({"feature": parts[0], "delta": delta, "status": status})
                except ValueError:
                    pass
        if in_table and line.startswith("##") and "エンジニアリング" not in line:
            in_table = False

    if not rows:
        print("⚠ FEATURE_REPORT.md にエンジニアリング済み特徴量の実データがありません")
        return out / "fe_delta_oof.png"

    df = pd.DataFrame(rows).sort_values("delta", ascending=True)
    colors = df["status"].map({"採用": "steelblue", "棄却": "tomato"})

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.5)))
    bars = ax.barh(df["feature"], df["delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔOOF (AUC)")
    ax.set_title("FE特徴量のΔOOF（青: 採用 / 赤: 棄却）")
    ax.set_xlim(df["delta"].min() * 1.3, df["delta"].max() * 1.3)
    for bar, val in zip(bars, df["delta"]):
        ax.text(val + (0.00002 if val >= 0 else -0.00002), bar.get_y() + bar.get_height() / 2,
                f"{val:+.5f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    fig.tight_layout()
    path = out / "fe_delta_oof.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


def plot_feature_correlation(train_path: Path, feature_cols: list[str], out: Path) -> Path:
    """採用済み特徴量間の相関ヒートマップ。"""
    if not train_path.exists():
        print(f"⚠ 学習データが見つかりません: {train_path}")
        return out / "feature_correlation.png"

    train = pd.read_pickle(train_path)
    cols = [c for c in feature_cols if c in train.columns]
    if not cols:
        print("⚠ 指定された特徴量が学習データに存在しません")
        return out / "feature_correlation.png"

    corr = train[cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.7), max(7, len(cols) * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0,
                cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
    ax.set_title("採用特徴量間の相関ヒートマップ")
    fig.tight_layout()
    path = out / "feature_correlation.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="特徴量レポート画像を生成する")
    parser.add_argument("--theme", type=str, default="all",
                        choices=["all", "importance", "delta", "corr"])
    parser.add_argument("--importance-csv", type=str, default="",
                        help="特徴量重要度CSV（省略時は最新ファイルを自動検索）")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_report_md = Path("FEATURE_REPORT.md")
    train_path = PROCESSED_DATA_DIR / "train_features.pkl"

    # 重要度ファイルの自動検索
    importance_path = Path(args.importance_csv) if args.importance_csv else None
    if importance_path is None:
        candidates = sorted(PLOTS_DIR.parent.glob("**/feature_importance_*.csv"))
        if candidates:
            importance_path = candidates[-1]
            print(f"重要度ファイル: {importance_path}")

    if args.theme in ("all", "importance") and importance_path:
        plot_feature_importance(importance_path, PLOTS_DIR)

    if args.theme in ("all", "delta"):
        plot_delta_oof(feature_report_md, PLOTS_DIR)

    if args.theme in ("all", "corr"):
        # 採用済み特徴量のリストを FEATURE_REPORT.md から抽出
        adopted = []
        if feature_report_md.exists():
            for line in feature_report_md.read_text().splitlines():
                if "✅ 採用" in line and line.startswith("|") and "*(例)" not in line:
                    cols = [p.strip() for p in line.split("|")[1:-1]]
                    if cols:
                        adopted.append(cols[0])
        plot_feature_correlation(train_path, adopted, PLOTS_DIR)

    print("\n✅ 特徴量レポート画像の生成が完了しました")
    print(f"   保存先: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
