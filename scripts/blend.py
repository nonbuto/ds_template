"""
アンサンブル・ブレンドスクリプト（Stage 6）

保存済みの OOF/test 予測（.npy）を読み込み、最適重みブレンド・Greedy HC を実行する。
手順と棄却分析は `PLAYBOOK.md#アンサンブル探索の手順stage-6` に対応している。

使い方（**この順に進む**。詳細は `PLAYBOOK.md#アンサンブル探索の手順stage-6`）:

    # ① 相関確認 —— 高すぎる候補は追加しない
    uv run python -m scripts.blend --mode corr --oofs lgb=p1.npy cb=p2.npy

    # ② 構成を決める（Caruana: 復元あり + サブセット bagging）
    uv run python -m scripts.blend --mode hillclimb --oofs lgb=p1.npy cb=p2.npy \
        --tests lgb=t1.npy cb=t2.npy

    # ③ 結合方式を上げる（符号制約なし線形。弱い候補を引き算に使える）
    uv run python -m scripts.blend --mode stack --oofs lgb=p1.npy cb=p2.npy \
        --tests lgb=t1.npy cb=t2.npy

補助・互換:
    --mode optimize --n-seeds 8   # simplex の重み探索。天井帯の重み bagging（G-CEILING）
    --mode greedy                 # 非復元・等重みの旧実装。過去実験の再現用
"""

import argparse

import numpy as np
import pandas as pd
from src.metrics import get_metric, greater_is_better, is_regression

from src.config import OOF_DIR, PROCESSED_DATA_DIR, TARGET_COL
from src.utils.ensemble import (correlation_check, greedy_ensemble, hillclimb,
                                optimize_weights, signed_stack)


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
                        choices=["corr", "optimize", "greedy", "hillclimb", "stack"],
                        help="実行モード: corr（相関確認）/ optimize（simplex 重み最適化）/ "
                             "greedy（非復元の貪欲選択）/ hillclimb（Caruana: 復元あり + bagging）/ "
                             "stack（符号制約なし線形スタッキング）")
    parser.add_argument("--oofs", nargs="+", required=True,
                        help="OOF予測ファイル（形式: モデル名=ファイルパス）")
    parser.add_argument("--tests", nargs="+", default=[],
                        help="Test予測ファイル（形式: モデル名=ファイルパス）")
    parser.add_argument("--corr-threshold", type=float, default=0.998,
                        help="相関確認のスキップ閾値（デフォルト: 0.998）")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="hillclimb の 1 bag あたり選択回数")
    parser.add_argument("--n-bags", type=int, default=20,
                        help="hillclimb の bag 数（選択の過学習を抑える）")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="重み探索を独立に何本まわして平均するか（G-CEILING の重み bagging）。"
                             "天井帯では 8〜12 本を推奨")
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
    if is_regression():
        y = y_raw.to_numpy(dtype=float)
    else:
        y = LabelEncoder().fit_transform(y_raw) if y_raw.dtype == object else y_raw.values

    print("\nOOF ファイル読み込み:")
    oofs = parse_npy_args(args.oofs)

    # blend は 1 次元の予測（二値の陽性確率 / 回帰の予測値）を重み付けする実装。
    # **train.py は二値でも (n, 2) の確率行列を保存する**ので、そのまま渡すと
    # 「1 次元でない」と撥ねられていた —— テンプレートが出力したファイルを
    # テンプレートのブレンドが受け取れない状態だった。二値は陽性列に落として受け入れる。
    # **回帰では「ユニーク値の数」をクラス数と読んではいけない。** 連続値なら行数に近い値になり、
    # 以前は `n_classes > 2` が必ず真になって、Stage 6 が回帰コンペで入口ごと落ちていた
    # （しかもエラー文は「回帰は対応している」と言っていた）。
    n_classes = 1 if is_regression() else len(np.unique(y))

    def _as_1d(name: str, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 2 and n_classes == 2:
            return arr[:, 1]
        return arr

    oofs = {n: _as_1d(n, a) for n, a in oofs.items()}

    # ── 未予測行（NaN）を全モデル共通で落とす ──
    # `train.py` は TimeSeriesSplit の未予測行を **NaN** で保存する（`covered`）。
    # そのまま重み最適化に渡すと sklearn が `Input contains NaN` で落ちるか、
    # 相関チェックが NaN を返して**「追加を検討可」と答える**（実測）。
    # 学習側と同じ扱い ——「全モデルで予測されている行だけで評価する」に揃える。
    covered = np.ones(len(y), dtype=bool)
    for arr in oofs.values():
        covered &= np.isfinite(np.asarray(arr, dtype=float))
    if not covered.all():
        print(f"\n  ℹ️ 未予測行 {int((~covered).sum()):,} / {len(covered):,} 行を除外します"
              f"（全モデルで予測された行だけで評価）")
        if covered.sum() < 10:
            raise SystemExit("❌ 全モデルで予測されている行が 10 行未満です。OOF を確認してください")
        oofs = {n: np.asarray(a, dtype=float)[covered] for n, a in oofs.items()}
        y = y[covered]
    bad = [n for n, a in oofs.items() if a.ndim != 1]
    if bad or n_classes > 2:
        raise SystemExit(
            "\n❌ blend は二値分類・回帰の 1 次元 OOF のみ対応しています。\n"
            f"   クラス数={n_classes} / 1 次元でない OOF={bad or 'なし'}\n"
            "   多クラスをブレンドする場合は、クラスごとに 1 次元へ分けて実行するか、\n"
            "   `src/utils/ensemble.py` を直接使って確率行列を重み付けしてください\n"
            "   （多クラス対応は docs/TODO_TEMPLATE.md に課題として記録済み）。"
        )

    # ──────────────────────────────────────────────
    # ① 相関確認
    # ──────────────────────────────────────────────
    if args.mode == "corr":
        names = list(oofs.keys())
        print(f"\n【① 相関確認】 モデル数: {len(names)}")
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

    tests = {n: _as_1d(n, a) for n, a in parse_npy_args(args.tests).items()} if args.tests else {}

    # ──────────────────────────────────────────────
    # 補助: simplex の重み探索（天井帯の重み bagging 用）
    # ──────────────────────────────────────────────
    if args.mode == "optimize":
        names = list(oofs.keys())
        oofs_matrix = np.column_stack([oofs[n] for n in names])
        tests_matrix = np.column_stack([tests[n] for n in names]) if tests else None

        print("\n【補助: simplex 重み探索】※ 順路は ① corr → ② hillclimb → ③ stack")
        w_opt, best_score = optimize_weights(oofs_matrix, y, _ascending_metric(),
                                             n_seeds=args.n_seeds)

        print("\nブレンド結果:")
        for name, w in zip(names, w_opt):
            single = get_metric()(y, oofs[name])
            print(f"  {name:20s}: weight={w:.3f}  (単体OOF={single:.5f})")
        # `_ascending_metric()` は探索用に符号を反転している。**表示と保存には素の値を使う**
        # —— 反転値をそのまま出すと「単体 0.94 → ブレンド −0.73」のように、
        # 同じ画面に向きの違う 2 つの数字が並ぶ（RMSE コンペで実際にそうなっていた）。
        blend_oof_score = get_metric()(y, oofs_matrix @ w_opt)
        print(f"\nブレンドOOF: {blend_oof_score:.5f}")

        if tests_matrix is not None:
            blend_test = tests_matrix @ w_opt
            out_oof = OOF_DIR / f"oof_{args.out_prefix}.npy"
            out_test = OOF_DIR / f"test_{args.out_prefix}.npy"
            np.save(out_oof, oofs_matrix @ w_opt)
            np.save(out_test, blend_test)
            print(f"\n保存: {out_oof.name}, {out_test.name}")
        return

    # ──────────────────────────────────────────────
    # Caruana 型 ensemble selection（復元あり + サブセット bagging）
    # ──────────────────────────────────────────────
    if args.mode == "hillclimb":
        print("\n【② Caruana ensemble selection】")
        weights, ens_oof, _ = hillclimb(oofs, y, _ascending_metric(),
                                        n_iter=args.n_iter, n_bags=args.n_bags)
        final_score = get_metric()(y, ens_oof)          # 表示は素の指標値
        for name, w in sorted(weights.items(), key=lambda t: -t[1]):
            if w > 0:
                print(f"  {name:20s}: weight={w:.4f}  (単体OOF={get_metric()(y, oofs[name]):.5f})")
        print(f"\nアンサンブル OOF: {final_score:.5f}")
        out_oof = OOF_DIR / f"oof_{args.out_prefix}_hc.npy"
        np.save(out_oof, ens_oof)
        if tests and all(n in tests for n in weights):
            ens_test = np.column_stack([tests[n] for n in weights]) @ np.array(
                [weights[n] for n in weights])
            np.save(OOF_DIR / f"test_{args.out_prefix}_hc.npy", ens_test)
            print(f"保存: {out_oof.name}, test_{args.out_prefix}_hc.npy")
        else:
            print(f"保存: {out_oof.name}（--tests が無いため test 予測は作りません）")
        return

    # ──────────────────────────────────────────────
    # 符号制約なしスタッキング（simplex では表現できない結合）
    # ──────────────────────────────────────────────
    if args.mode == "stack":
        print("\n【③ 符号制約なしスタッキング】")
        coefs, ens_oof, ens_test, final_score = signed_stack(oofs, tests, y)
        for name, c in sorted(coefs.items(), key=lambda t: -abs(t[1])):
            print(f"  {name:20s}: coef={c:+.4f}")
        print(f"\nスタッキング OOF: {final_score:.5f}（fold 外で算出）")
        np.save(OOF_DIR / f"oof_{args.out_prefix}_stack.npy", ens_oof)
        if ens_test is not None:
            np.save(OOF_DIR / f"test_{args.out_prefix}_stack.npy", ens_test)
        print(f"保存: oof_{args.out_prefix}_stack.npy"
              + ("" if ens_test is not None else "（--tests が無いため test 予測は作りません）"))
        return

    # ──────────────────────────────────────────────
    # 互換: 非復元・等重みの旧実装（過去実験の再現用）（非復元・等重み。互換のため残す）
    # ──────────────────────────────────────────────
    if args.mode == "greedy":
        print("\n【互換: 非復元 Greedy】※ 新規は --mode hillclimb を使う")
        selected, ens_oof, ens_test, _ = greedy_ensemble(
            oofs=oofs, tests=tests, y=y, metric_fn=_ascending_metric(),
        )
        final_score = get_metric()(y, ens_oof)      # 表示・ファイル名は素の指標値
        out_oof = OOF_DIR / f"oof_{args.out_prefix}_greedy.npy"
        out_test = OOF_DIR / f"test_{args.out_prefix}_greedy.npy"
        np.save(out_oof, ens_oof)
        if ens_test is not None:
            np.save(out_test, ens_test)

        print(f"\n保存: {out_oof.name}")
        if ens_test is not None:
            print(f"次: uv run python -m scripts.predict --test-npy {out_test} "
                  f"--model greedy_ens --oof-score {final_score:.5f}")
        else:
            print("  ℹ️ --tests を渡していないため test 予測は作っていません"
                  "（提出するには --tests を付けて再実行してください）")


if __name__ == "__main__":
    main()
