"""
1列ΔOOF計測スクリプト（Stage 4: 段階的FE用）

ベースとなる特徴量セットに1列を追加し、OOFスコア（balanced_accuracy）の変化を計測する。
scripts/train.py の run_cv() を再利用し、multiclass/class_weight/評価指標を二重管理しない。

/ds-fe-hypothesis で仮説を立案した後、このスクリプトで効果を測定する。

このスクリプトの実行自体が1つの実験として experiments/log.csv に自動記録される
（採用・棄却を問わない）。"New"（base+新特徴量）のOOF/test予測は .npy 保存されるため、
仮説採用時は再学習せず、保存済みファイルをそのまま scripts/predict.py で提出ファイル化できる。

使い方:
    uv run python -m scripts.feature_study --new-feature gender --hypothesis-id H-001
    uv run python -m scripts.feature_study --new-feature gender --model lgb_balanced

結果の読み方（Balanced Accuracyは0〜1で変動幅が大きいため、AUC等より緩めの閾値）:
    +0.001 以上: 採用を強く推奨
    +0.0003〜+0.001: 採用を検討（他のモデルでも確認）
    ±0.0003 以内: ノイズ範囲（採用不要）
    マイナス: 棄却

【重要】ΔOOFの閾値判断だけで棄却を確定しない（Private LBでのshakedownリスク）:
    ΔOOFがノイズ範囲・棄却域でも、必ずtrain/val/gapのCV内部診断とimportanceの3点を併せて確認する。
    実証1（過去コンペ）: ΔOOF=-0.00009 だが gap 変化ほぼ0（純粋なノイズ）だった特徴量は LB でむしろ改善し、
    ΔOOF=-0.00020 で gap が拡大（軽度の過学習兆候）だった特徴量は LB でも一貫して悪化した。
    実証2（過去コンペ）: importance が 14 特徴量中 8 位（中位の既存採用特徴量を上回る）にもかかわらず
    ΔOOF はほぼゼロ〜マイナスという乖離が複数の特徴量で観測された。
    ΔOOF・gap・importanceのいずれか1つだけで判断すると見落としが生じるため、
    このスクリプトは3点セットを自動表示する。最終判断は必ずこの3点を総合して行う。
"""

import argparse
import json

import numpy as np

from scripts.train import run_cv, DEFAULT_PARAMS, FEATURES as BASE_FEATURES
from src.metrics import greater_is_better
from src.noise import (empirical_lb_floor, fold_paired_se,
                       min_detectable_difference, multiple_testing_note, paired_se)
from src.config import OOF_DIR, PLOTS_DIR, EXPERIMENT_NAME, RANDOM_STATE
from src.experiment import ExperimentTracker

# ──────────────────────────────────────────────
# TODO: ベース特徴量セットは scripts/train.py の FEATURES を使う（二重管理しない）
# ──────────────────────────────────────────────


GAP_NOTABLE = 0.0005      # 過学習の兆候（gap の拡大）。効果量とは別軸で見る


def _welch_df(se_boot: float, se_splits: float, n_splits: int) -> float | None:
    """合成した SE の**有効自由度**（Welch–Satterthwaite）。

    ブートストラップ由来の成分は `DEFAULT_N_BOOT-1` の自由度を持ち、
    分割由来は `n_splits-1` しか持たない。両者を合成した SE に
    「小さい方の自由度」を当てると過補正になる（検出力が 5.7% まで落ちた）。
    """
    from src.noise import DEFAULT_N_BOOT

    a, b = float(se_boot), float(se_splits)
    if not (np.isfinite(a) and np.isfinite(b)) or (a == 0 and b == 0):
        return None
    num = (a**2 + b**2) ** 2
    den = a**4 / max(DEFAULT_N_BOOT - 1, 1) + b**4 / max(n_splits - 1, 1)
    return float(num / den) if den > 0 else None


def build_verdict(delta: float, floor: float, z: float, gap_delta: float,
                  is_screening: bool) -> str:
    """ΔOOF・床・gap から判定文を作る。

    **関数に切り出しているのはテストのため。** main() の中に埋めると、
    判定の正しさを「ソースの字面」でしか検査できなくなる（L-30 で繰り返した失敗）。

    `is_screening`（分割 1 回）のときは**床が下限でしかない**ので、
    「採用推奨」とは言い切らず、次に何をするかを指定する。
    """
    if not np.isfinite(floor) or floor <= 0:
        # **床が出せないときに判定を出さない。** `src/noise.py` の `verdict()` は
        # これを明示的に潰しているのに、実際に判定を出すこの関数には無かった。
        # `se_rows` と `se_folds` が両方 NaN だと `nanmax` が NaN を返し、
        # `abs(delta) < nan` が False になるので**必ず「候補」側に落ちていた**。
        return "床を推定できませんでした（fold 数・指標・記録の精度を確認してください）"

    if abs(delta) < floor:
        # **「測れていない」と「効果がない」を混同しない。**
        # 前者は床を下げれば測れる可能性があり、後者は測った上で差が無い。
        base = "⬜ 測れていない（床未満）"
        if gap_delta > GAP_NOTABLE:
            return f"{base} — ただし gap が拡大しており過学習の兆候あり"
        return f"{base} — seed / fold を増やすか、集約へ（G-CEILING）"

    if delta > 0:
        if is_screening:
            if z >= 3:
                return f"🔶 候補（z={z:+.2f}）— **`--n-repeats 3` で測り直してから採否を決める**"
            return f"🔶 弱い候補（z={z:+.2f}）— 他の候補を先に測り、余裕があれば再計測"
        if z >= 3:
            return "✅ 採用推奨"
        return "🔶 採用検討（他モデル・他 seed でも確認を推奨）"

    if gap_delta > GAP_NOTABLE:
        return "❌ 棄却（過学習傾向を伴う明確な悪化）"
    return "❌ 棄却"


def _count_prior_feature_tests() -> int:
    """log.csv に残っている FE 計測の件数（`feature_study` が書いた行）を数える。

    **床は 1 回の判定を守るが、判定の繰り返しは守らない。** 何件試したかを知らないと、
    「87 件中 2 件が採用推奨」が本物なのか偶然なのか区別できない。
    """
    import csv as _csv

    from src.config import EXPERIMENTS_DIR

    path = EXPERIMENTS_DIR / "log.csv"
    if not path.exists():
        return 0
    try:
        with open(path, newline="") as f:
            return sum(1 for r in _csv.DictReader(f) if "ΔOOF=" in (r.get("notes") or ""))
    except OSError:
        return 0


def _cv_stats(result: dict) -> dict:
    """train/valのfold平均・std・gapをまとめる（過学習・CV安定性を毎回確認するため）。

    `gap` は**「train が val よりどれだけ良いか」**に揃える（`G-DIAG` の第1診断軸）。
    素の差 `train - val` だと、RMSE や logloss のように**小さいほど良い**指標では
    過学習しているほど gap が負に大きくなり、判定の向きが逆になる。
    """
    tr = np.array(result["train_scores"])
    va = np.array(result["val_scores"])
    sign = 1.0 if greater_is_better() else -1.0
    return {
        "train_mean": float(tr.mean()), "train_std": float(tr.std()),
        "val_mean": float(va.mean()), "val_std": float(va.std()),
        "gap": float(sign * (tr.mean() - va.mean())),
    }


def _print_cv_stats(label: str, stats: dict) -> None:
    print(f"  {label} train: mean={stats['train_mean']:.5f} std={stats['train_std']:.5f}")
    print(f"  {label} val  : mean={stats['val_mean']:.5f} std={stats['val_std']:.5f}")
    print(f"  {label} gap(train が val より良い分): {stats['gap']:.5f}")


def main():
    parser = argparse.ArgumentParser(description="1列追加のΔOOF計測")
    parser.add_argument("--new-feature", type=str, required=True,
                        help="追加する特徴量の列名。カンマ区切りで複数指定する場合は --allow-batch が必要")
    parser.add_argument("--model", type=str, default="lgb_balanced",
                        choices=["lgb", "lgb_balanced", "cb", "xgb"])
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="モデルの seed（初期化・サンプリング）")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="分割の seed（省略時は RANDOM_STATE）")
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="**分割を引き直す回数**。2 以上にすると split_seed を振って繰り返し、"
                             "分割由来の分散を床に含める（`G-FULLCV`）。既定 1 では"
                             "「この 1 つの分割の上での差」しか測れない")
    parser.add_argument("--n-splits", type=int, default=None,
                        help="fold 数（省略時は config の N_SPLITS）")
    parser.add_argument("--hypothesis-id", type=str, default="",
                        help="対応するFE_HYPOTHESES.mdの仮説ID（例: H-007）。log.csvに記録される")
    parser.add_argument("--params", type=str, default="",
                        help="作業用HP等のJSONファイルパス（省略時はデフォルトパラメータ）")
    parser.add_argument("--allow-batch", action="store_true",
                        help="複数列の同時投入を許可する（CLAUDE.md Stage4「1列ずつ」原則の明示的な例外）")
    parser.add_argument("--batch-reason", type=str, default="",
                        help="--allow-batch 使用時に必須。なぜ一括投入で良いのかの理由（log.csvに記録される）")
    args = parser.parse_args()

    # ── 一括投入ガード（CLAUDE.md 絶対規約「特徴量は 1 列ずつ計測する」/ `G-STEPWISE`）──
    # 「どの列が効き、どの列が相殺しているか」が分からなくなる事故を防ぐ。
    # 一括はスクリーニング用途に限り、採用・棄却の判断は必ず LOO 分解を経てから行う。
    new_cols = [c.strip() for c in args.new_feature.split(",") if c.strip()]
    is_batch = len(new_cols) > 1
    if is_batch and not args.allow_batch:
        parser.error(
            f"\n複数列（{len(new_cols)}列: {new_cols}）の同時投入が指定されました。\n"
            "CLAUDE.md Stage4 は『特徴量は必ず1列ずつ投入する』と定めています。\n"
            "一括投入は ΔOOF の帰属が不明になり、後で LOO 分解をやり直す手戻りを生みます。\n\n"
            "スクリーニング目的で意図的に一括投入する場合は、理由を添えて明示してください:\n"
            "  --allow-batch --batch-reason '<なぜ一括で良いのか>'\n"
            "※ その場合も採用・棄却の判断は LOO 分解の後に行ってください。"
        )
    if is_batch and not args.batch_reason:
        parser.error("--allow-batch 使用時は --batch-reason で理由を明示してください（log.csvに記録されます）")

    for c in new_cols:
        assert c not in BASE_FEATURES, f"{c} は既にベース特徴量に含まれています"

    params = DEFAULT_PARAMS[args.model].copy()
    if args.params:
        with open(args.params) as f:
            params.update(json.load(f))

    new_features = BASE_FEATURES + new_cols
    if is_batch:
        print(f"\n  ⚠️ 一括投入モード（{len(new_cols)}列）: 理由={args.batch_reason}")
        print("     → これはスクリーニングです。採用・棄却の判断は LOO 分解の後に行ってください")

    # ── 分割を引き直して繰り返す ──
    # **モデル seed だけを振っても分割は毎回同じ**なので、行のブートストラップでは
    # 分割由来の分散が床に入らない。`--n-repeats` を 2 以上にすると分割 seed を振り、
    # 「別の切り方でも同じ向きに動くか」を確かめられる（`G-FULLCV`）。
    split_seeds = [args.split_seed if args.split_seed is not None else RANDOM_STATE]
    if args.n_repeats > 1:
        split_seeds = [(args.split_seed or RANDOM_STATE) + i for i in range(args.n_repeats)]

    base_runs, new_runs = [], []
    for i, ss in enumerate(split_seeds, 1):
        tag = f"（分割 {i}/{len(split_seeds)}, split_seed={ss}）" if len(split_seeds) > 1 else ""
        print(f"\n【ベーススコア計算中】 特徴量数: {len(BASE_FEATURES)} {tag}")
        base_runs.append(run_cv(args.model, params, seed=args.seed, features=BASE_FEATURES,
                                split_seed=ss, n_splits=args.n_splits))
        print(f"  Base OOF: {base_runs[-1]['oof_score']:.5f}")
        print(f"【+{args.new_feature} スコア計算中】 特徴量数: {len(new_features)} {tag}")
        new_runs.append(run_cv(args.model, params, seed=args.seed, features=new_features,
                               split_seed=ss, n_splits=args.n_splits))
        print(f"  New  OOF: {new_runs[-1]['oof_score']:.5f}")

    base_result, new_result = base_runs[0], new_runs[0]
    base_oof = float(np.mean([r["oof_score"] for r in base_runs]))
    new_oof = float(np.mean([r["oof_score"] for r in new_runs]))
    base_stats = _cv_stats(base_result)
    _print_cv_stats("Base", base_stats)
    new_stats = _cv_stats(new_result)
    _print_cv_stats("New", new_stats)
    metric_dir = "大きいほど良い" if greater_is_better() else "小さいほど良い"

    # **改善方向に揃えた Δ**を判定に使う。素の `new - base` だと、RMSE・logloss・MAE の
    # ように小さいほど良い指標では符号が逆になり、**良い特徴量を棄却し悪い特徴量を採用する**。
    # feature_study は FE 判断の中核ツールなので、ここが逆だと全 FE の採否が反転する。
    raw_delta = new_oof - base_oof
    delta = raw_delta if greater_is_better() else -raw_delta
    gap_delta = new_stats["gap"] - base_stats["gap"]

    # ── 床は固定値ではなく**この 2 本から実測する** ──
    # 以前は ±0.0003 / +0.001 の絶対値だった。しかし ΔOOF 自身のばらつきは実測で
    # **SD 0.0011**（seed だけ振った 8 回）あり、**完全に無関係な列が seed 次第で
    # 「🔶 採用検討」「⬜ ノイズ範囲」「❌ 棄却」の 3 判定すべてを出す**。
    # しかも閾値は指標非依存の絶対値で、RMSE（目標σ=1000）でも同じ 0.0003 を使っていた。
    #
    # base と new は**同じ行を同じ分割で**予測しているので、対応のある比較ができる。
    # 行の当たり外れと fold の難易度は差を取った時点で相殺する（`src/noise.py`）。
    # 両方で予測された行だけを突き合わせる（TimeSeriesSplit の未予測行を混ぜない）
    both = base_result["covered"] & new_result["covered"]
    se_rows = paired_se(base_result["y_true"][both],
                        new_result["oof_preds"][both], base_result["oof_preds"][both])
    se_folds = fold_paired_se(new_result["val_scores"], base_result["val_scores"])
    # **分割を引き直したときのばらつき**（最も見落とされやすい成分）。
    # 行のブートストラップも fold 対応差も、分割が固定されている限りこれを再現しない。
    per_split = [(n["oof_score"] - b["oof_score"]) * (1 if greater_is_better() else -1)
                 for b, n in zip(base_runs, new_runs)]
    se_splits = (float(np.std(per_split, ddof=1) / np.sqrt(len(per_split)))
                 if len(per_split) > 1 else float("nan"))

    # ── 行由来と分割由来は**独立な成分**なので二乗和で合成する ──
    # 前ラウンドで「分割は 1 段上の不確実性だから下位を含む」と判断したが、**これは誤り**だった。
    # **全分割が同じ行集合を使う**ので、行由来の誤差は全分割に共通に乗る＝分割間分散から
    # 相殺されて消える。したがって分割を増やしても行由来の不確実性は減らない。
    #
    # モンテカルロ実測（真の効果ゼロ、σ_row=0.0024 / σ_split=0.0034、2σ 判定の偽陽性率）:
    #   分割数        3      5     10     20
    #   分割のみ    33.2%  34.5%  43.9%  55.0%   ← 分割を増やすほど**悪化**する
    #   二乗和        6.4%   4.9%   4.8%   4.8%   ← 設計値 5% に一致
    #
    # 分割が 1 回しかないときは分割成分を推定できないので、行・fold の大きい方を
    # 採り「下限」と明示する（`is_screening`）。
    # 合成した SE の自由度は成分ごとに違う。**支配的な `se_rows` は 400 回の
    # ブートストラップから推定されており自由度は実質無限大**なのに、
    # `df = m-1` を全体に当てると精度の高い成分まで不確かと見なす二重の罰になる。
    # 実測（σ_row=0.0024 / σ_split=0.0034、真の効果 +0.008、m=3）:
    #     t(m−1) を全体に当てる : 偽陽性 0.0% / **検出力 5.7%**  ← 直前の実装
    #     Welch の有効自由度     : 偽陽性 5.5% / 検出力 62.7%
    #     正規 2σ               : 偽陽性 6.6% / 検出力 72.8%
    # 「偽陽性 33%」を「検出力 5.7%」で置き換えては意味がない（`G-PERSIST`）。
    _hi = float(np.nanmax([se_rows, se_folds])) if np.isfinite([se_rows, se_folds]).any() else np.nan
    if len(per_split) >= 3:
        # 片方だけ欠けても分割の床は活かす（両方欠けたときだけ床なしにする）
        hi_safe = 0.0 if not np.isfinite(_hi) else _hi
        se = float(np.hypot(hi_safe, se_splits))
        df_eff = _welch_df(hi_safe, se_splits, len(per_split))
    else:
        se, df_eff = _hi, None
    split_deltas = (", ".join(f"{d:+.5f}" for d in per_split) if len(per_split) > 1
                    else "（--n-repeats 2 以上で分割を引き直せます）")
    # 分割から推定した SE は自由度 m−1。少ない繰り返しに正規の 2σ を当てると
    # 名目 5% のつもりで実際は 18%（m=3）になる（`min_detectable_difference` の docstring）。
    floor = min_detectable_difference(se, df=df_eff)
    z = delta / se if se > 0 else float("nan")

    # ── スクリーニングと採用判定を分ける ──
    # **分割を 1 回しか引かないと、床から最大成分が抜ける。** 実測（無関係な列）:
    #     1 分割: 行 0.00243 / fold 0.00126 / 分割 —    → 採用 0.00243
    #     4 分割: 行 0.00243 / fold 0.00126 / 分割 0.00341 → 採用 0.00341（**40% 大きい**）
    # かといって常に 3〜5 回引くと FE 1 列あたりの計測時間が 3〜5 倍になり、
    # 何十件も試す運用と両立しない。**用途で分ける**のが正解:
    #   スクリーニング（既定 1 回）= 候補を絞る。床は**下限**でしかないと明示する
    #   採用判定（3 回以上）      = 特徴量セットに入れる決定。分割由来の分散を床に含める
    is_screening = len(per_split) < 3
    mode_note = (
        "🔍 スクリーニング判定（--n-repeats=1）"
        "\n              この床は**下限**です（分割由来の分散を含まない。実測で最大成分になりうる）。"
        "\n              **採用を決めるときは `--n-repeats 3` 以上で測り直すこと。**"
        if is_screening else
        f"✅ 採用判定（{len(per_split)} 分割で計測。分割由来の分散を床に含む）"
    )

    # **提出実績から測った床**も併記する。ブートストラップの床は「この CV の上で測れるか」、
    # 実績床は「**LB に現れるか**」を答える。後者は分割の引き直し・分布差・Public の
    # 標本ゆらぎを含むので、提出する価値があるかの判断ではこちらが現実の壁になる。
    # これまでに何件 FE を計測したかを log.csv から数える（notes に "ΔOOF=" が入る）
    n_prior_tests = _count_prior_feature_tests()
    mt_note = multiple_testing_note(n_prior_tests + 1)

    lb_floor = empirical_lb_floor()
    lb_note = (f"{lb_floor}\n 今回の ΔOOF はその {lb_floor.ratio(delta):.1f} 倍"
               + ("  ← 床未満。LB には出ない公算が大きい" if lb_floor.ratio(delta) < 1 else "")
               if lb_floor else
               "LB 反映の床: まだ測れません（OOF と LB が揃った提出が 8 件未満）")

    verdict = build_verdict(delta, floor, z, gap_delta, is_screening)

    gap_warning = ""
    if gap_delta > 0.005:
        gap_warning = "\n ⚠️  train/val gapが拡大しています（過学習の兆候の可能性）"

    # importance確認（ΔOOF・gapだけでなくimportanceも併せて見る。中位以上でもΔOOFに寄与しない
    # ケース（H-006, sleep_deficit_amount/log等）があるため、この3点セットを常に表示する）
    importance_note = "importance計算不可（このモデルは feature_importances_ 非対応）"
    if new_result["importance_df"] is not None:
        imp_df = new_result["importance_df"].reset_index(drop=True)
        n_feat = len(imp_df)
        parts = []
        for c in new_cols:
            match = imp_df[imp_df["feature"] == c]
            if not match.empty:
                rank = match.index[0] + 1
                imp_val = match.iloc[0]["importance"]
                parts.append(f"{c}: importance={imp_val:.1f}（{n_feat}特徴量中{rank}位）")
        importance_note = " / ".join(parts)

    # ── 実験記録: "New"（base+新特徴量）をexpNNNとしてlog.csvに記録する ──
    # 採用・棄却を問わず記録することで、全FE仮説検証を一元的なexp番号で追跡できる。
    # OOF/test予測を保存するため、採用時は再学習せずscripts/predict.pyで提出ファイル化できる。
    hyp_note = f"{args.hypothesis_id}: " if args.hypothesis_id else ""
    batch_note = (
        f"  ⚠️一括投入{len(new_cols)}列(理由: {args.batch_reason})——採否判断はLOO分解後に行うこと"
        if is_batch else ""
    )
    tracker = ExperimentTracker(
        experiment_name=EXPERIMENT_NAME,
        model=args.model,
        features=f"{len(new_features)}features(+{args.new_feature})",
        notes=f"{hyp_note}ΔOOF={delta:+.5f} vs base({len(BASE_FEATURES)}features)  {verdict}{batch_note}",
    )
    tracker.start_run(
        description=f"FE仮説検証: {hyp_note}{args.new_feature}追加"
    )
    exp_id = tracker._experiment_id

    np.save(OOF_DIR / f"oof_{exp_id}_{args.model}.npy", new_result["oof_preds"])
    np.save(OOF_DIR / f"test_{exp_id}_{args.model}.npy", new_result["test_preds"])
    if new_result["importance_df"] is not None:
        new_result["importance_df"].to_csv(
            PLOTS_DIR / f"feature_importance_{exp_id}.csv", index=False
        )

    tracker.end_run(
        train_scores=new_result["train_scores"],
        val_scores=new_result["val_scores"],
        oof_score=new_oof,
        n_features=len(new_features),
    )

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 特徴量追加効果の計測結果（exp{exp_id}としてlog.csvに記録済み）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 特徴量    : {args.new_feature}
 モデル    : {args.model}

 [OOF]
 Base OOF  : {base_oof:.5f}
 New  OOF  : {new_oof:.5f}
 ΔOOF      : {delta:+.5f}   ← 改善方向に揃えた値（正なら改善）
 素の差     : {raw_delta:+.5f}（{metric_dir}）

 [この 2 本で実測した床]（固定閾値ではない → src/noise.py）
 対応差の床 : 行 1σ={se_rows:.5f} / fold 1σ={se_folds:.5f} / 分割 1σ={se_splits:.5f} → 採用 1σ={se:.5f}
 分割ごとのΔ: {split_deltas}
 計測の位置づけ: {mode_note}
 {mt_note}
 判定の境界 : 2σ={floor:.5f}   z={z:+.2f}
 {lb_note}

 [CV内部診断: train/val 平均・ばらつき・gap]
 Base: train={base_stats['train_mean']:.5f}±{base_stats['train_std']:.5f}  val={base_stats['val_mean']:.5f}±{base_stats['val_std']:.5f}  gap={base_stats['gap']:.5f}
 New : train={new_stats['train_mean']:.5f}±{new_stats['train_std']:.5f}  val={new_stats['val_mean']:.5f}±{new_stats['val_std']:.5f}  gap={new_stats['gap']:.5f}
 Δgap: {gap_delta:+.5f}{gap_warning}

 [Importance]
 {importance_note}
 ※ importanceが中位以上でもΔOOFに寄与しないケースがある（H-006等）。ΔOOF・gap・importanceの3点を総合して判断すること

 判定      : {verdict}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
次のステップ:
  1. /ds-fe-hypothesis update {args.hypothesis_id or 'H-NNN'} で結果を記録
  2. 採用の場合: scripts/train.py の FEATURES に '{args.new_feature}' を追加してから
     uv run python -m scripts.predict --test-npy {OOF_DIR}/test_{exp_id}_{args.model}.npy \\
       --model {args.model} --oof-score {new_oof:.5f} --exp-id {exp_id}
     で再学習せずそのまま提出ファイルを生成できます
  3. 棄却の場合: exp{exp_id}の記録はlog.csvに残るのみで、追加対応は不要です
""")


if __name__ == "__main__":
    main()
