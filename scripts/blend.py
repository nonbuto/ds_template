"""
アンサンブル・ブレンドスクリプト（Stage 6）

保存済みの OOF/test 予測（.npy）を読み込み、最適重みブレンド・Greedy HC を実行する。
CLAUDE.md の「アンサンブル探索手順（STEP 1〜4）」に対応している。

使い方:
    # STEP 1: 相関確認（必ず最初に実施）
    uv run python scripts/blend.py --mode corr --oofs lgb=path1.npy cb=path2.npy

    # STEP 2: 最適重みブレンド
    uv run python scripts/blend.py --mode optimize --oofs lgb=path1.npy cb=path2.npy \
        --tests lgb=test1.npy cb=test2.npy

    # STEP 3: Greedy Hill Climbing
    uv run python scripts/blend.py --mode greedy --oofs lgb=p1.npy cb=p2.npy xgb=p3.npy \
        --tests lgb=t1.npy cb=t2.npy xgb=t3.npy
"""

import argparse

import numpy as np
import pandas as pd
from src.metrics import get_metric, greater_is_better

from src.config import OOF_DIR, PROCESSED_DATA_DIR, TARGET_COL
from src.utils.ensemble import correlation_check, optimize_weights, greedy_ensemble


def _ascending_metric():
    """`optimize_weights` / `greedy_ensemble` は「大きいほど良い」前提なので向きを揃える。

    RMSE のように小さいほど良い指標は符号を反転して渡す。
    指標そのものは `src.metrics` が唯一の定義元（学習・HP と必ず同じものを使う）。
    """
    metric = get_metric()
    if greater_is_better():
        return metric
    return lambda y, p: -metric(y, p)


def parse_npy_args(args_list: list[str]) -> dict[str, np.ndarray]:
    """'name=path.npy' 形式の引数をパースして {name: array} に変換する。"""
    result = {}
    for item in args_list:
        name, path = item.split("=", 1)
        result[name] = np.load(path)
        print(f"  読み込み: {name} ({path})")
    return result


def main():
    parser = argparse.ArgumentParser(description="アンサンブル・ブレンドスクリプト")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["corr", "optimize", "greedy"],
                        help="実行モード: corr（相関確認）/ optimize（重み最適化）/ greedy（Greedy HC）")
    parser.add_argument("--oofs", nargs="+", required=True,
                        help="OOF予測ファイル（形式: モデル名=ファイルパス）")
    parser.add_argument("--tests", nargs="+", default=[],
                        help="Test予測ファイル（形式: モデル名=ファイルパス）")
    parser.add_argument("--corr-threshold", type=float, default=0.998,
                        help="相関確認のスキップ閾値（デフォルト: 0.998）")
    parser.add_argument("--out-prefix", type=str, default="blend",
                        help="出力ファイルのプレフィックス")
    args = parser.parse_args()

    # 正解ラベルの読み込み
    train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    # ラベルは train.py と同じく LabelEncoder を通す。生のまま渡すと文字列ラベルや
    # 多クラスで指標が落ちる（`multi_class must be in ('ovo','ovr')`）。
    # e2e が blend を通していなかったため、この不整合が長く残っていた。
    from sklearn.preprocessing import LabelEncoder
    y_raw = train[TARGET_COL]
    y = LabelEncoder().fit_transform(y_raw) if y_raw.dtype == object else y_raw.values

    print("\nOOF ファイル読み込み:")
    oofs = parse_npy_args(args.oofs)

    # blend は 1 次元の OOF（二値の陽性確率 / 回帰の予測値）を前提にした実装。
    # train.py は多クラスで (n, クラス数) の OOF を作るため、そのまま渡すと
    # sklearn の分かりにくいエラーになる。**ここで明示的に止める。**
    n_classes = len(np.unique(y))
    bad = [n for n, a in oofs.items() if np.asarray(a).ndim != 1]
    if bad or n_classes > 2:
        raise SystemExit(
            "\n❌ blend は二値分類・回帰の 1 次元 OOF のみ対応しています。\n"
            f"   クラス数={n_classes} / 1 次元でない OOF={bad or 'なし'}\n"
            "   多クラスをブレンドする場合は、クラスごとに 1 次元へ分けて実行するか、\n"
            "   `src/utils/ensemble.py` を直接使って確率行列を重み付けしてください\n"
            "   （多クラス対応は docs/TODO_TEMPLATE.md に課題として記録済み）。"
        )

    # ──────────────────────────────────────────────
    # STEP 1: 相関確認
    # ──────────────────────────────────────────────
    if args.mode == "corr":
        names = list(oofs.keys())
        print(f"\n【STEP 1: 相関確認】 モデル数: {len(names)}")
        print(f"{'モデルA':20s} {'モデルB':20s} {'相関':>8s} {'判定':>15s}")
        print("─" * 65)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                corr, skip = correlation_check(oofs[a], oofs[b], threshold=args.corr_threshold)
                verdict = "⚠ スキップ推奨" if skip else "✅ アンサンブル候補"
                print(f"{a:20s} {b:20s} {corr:8.4f} {verdict:>15s}")

        # 単体スコアも表示
        print("\n単体OOFスコア:")
        for name, oof in oofs.items():
            score = get_metric()(y, oof)
            print(f"  {name:20s}: {score:.5f}")
        return

    tests = parse_npy_args(args.tests) if args.tests else {}

    # ──────────────────────────────────────────────
    # STEP 2: 最適重みブレンド
    # ──────────────────────────────────────────────
    if args.mode == "optimize":
        names = list(oofs.keys())
        oofs_matrix = np.column_stack([oofs[n] for n in names])
        tests_matrix = np.column_stack([tests[n] for n in names]) if tests else None

        print("\n【STEP 2: 最適重みブレンド】")
        w_opt, best_score = optimize_weights(oofs_matrix, y, _ascending_metric())

        print("\nブレンド結果:")
        for name, w in zip(names, w_opt):
            single = get_metric()(y, oofs[name])
            print(f"  {name:20s}: weight={w:.3f}  (単体OOF={single:.5f})")
        print(f"\nブレンドOOF: {best_score:.5f}")

        if tests_matrix is not None:
            blend_test = tests_matrix @ w_opt
            out_oof = OOF_DIR / f"oof_{args.out_prefix}.npy"
            out_test = OOF_DIR / f"test_{args.out_prefix}.npy"
            np.save(out_oof, oofs_matrix @ w_opt)
            np.save(out_test, blend_test)
            print(f"\n保存: {out_oof.name}, {out_test.name}")
        return

    # ──────────────────────────────────────────────
    # STEP 3: Greedy Hill Climbing
    # ──────────────────────────────────────────────
    if args.mode == "greedy":
        print("\n【STEP 3: Greedy Hill Climbing】")
        selected, ens_oof, ens_test, final_score = greedy_ensemble(
            oofs=oofs, tests=tests, y=y, metric_fn=_ascending_metric(),
        )
        out_oof = OOF_DIR / f"oof_{args.out_prefix}_greedy.npy"
        out_test = OOF_DIR / f"test_{args.out_prefix}_greedy.npy"
        np.save(out_oof, ens_oof)
        if ens_test is not None:
            np.save(out_test, ens_test)

        print(f"\n保存: {out_oof.name}")
        print(f"次: scripts/predict.py --test-npy {out_test} --model greedy_ens --oof-score {final_score:.5f}")


if __name__ == "__main__":
    main()
