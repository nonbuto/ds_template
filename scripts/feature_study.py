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
from src.config import OOF_DIR, PLOTS_DIR, EXPERIMENT_NAME
from src.experiment import ExperimentTracker

# ──────────────────────────────────────────────
# TODO: ベース特徴量セットは scripts/train.py の FEATURES を使う（二重管理しない）
# ──────────────────────────────────────────────


def _cv_stats(result: dict) -> dict:
    """train/valのfold平均・std・gapをまとめる（過学習・CV安定性を毎回確認するため）。"""
    tr = np.array(result["train_scores"])
    va = np.array(result["val_scores"])
    return {
        "train_mean": float(tr.mean()), "train_std": float(tr.std()),
        "val_mean": float(va.mean()), "val_std": float(va.std()),
        "gap": float(tr.mean() - va.mean()),
    }


def _print_cv_stats(label: str, stats: dict) -> None:
    print(f"  {label} train: mean={stats['train_mean']:.5f} std={stats['train_std']:.5f}")
    print(f"  {label} val  : mean={stats['val_mean']:.5f} std={stats['val_std']:.5f}")
    print(f"  {label} gap(train-val): {stats['gap']:.5f}")


def main():
    parser = argparse.ArgumentParser(description="1列追加のΔOOF計測")
    parser.add_argument("--new-feature", type=str, required=True,
                        help="追加する特徴量の列名。カンマ区切りで複数指定する場合は --allow-batch が必要")
    parser.add_argument("--model", type=str, default="lgb_balanced",
                        choices=["lgb", "lgb_balanced", "cb", "xgb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hypothesis-id", type=str, default="",
                        help="対応するFE_HYPOTHESES.mdの仮説ID（例: H-007）。log.csvに記録される")
    parser.add_argument("--params", type=str, default="",
                        help="作業用HP等のJSONファイルパス（省略時はデフォルトパラメータ）")
    parser.add_argument("--allow-batch", action="store_true",
                        help="複数列の同時投入を許可する（CLAUDE.md Stage4「1列ずつ」原則の明示的な例外）")
    parser.add_argument("--batch-reason", type=str, default="",
                        help="--allow-batch 使用時に必須。なぜ一括投入で良いのかの理由（log.csvに記録される）")
    args = parser.parse_args()

    # ── 一括投入ガード（CLAUDE.md Stage4 / 指針#13）─────────────────────────
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

    print(f"\n【ベーススコア計算中】 特徴量数: {len(BASE_FEATURES)}")
    base_result = run_cv(args.model, params, seed=args.seed, features=BASE_FEATURES)
    base_oof = base_result["oof_score"]
    base_stats = _cv_stats(base_result)
    print(f"  Base OOF: {base_oof:.5f}")
    _print_cv_stats("Base", base_stats)

    new_features = BASE_FEATURES + new_cols
    print(f"\n【+{args.new_feature} スコア計算中】 特徴量数: {len(new_features)}")
    if is_batch:
        print(f"  ⚠️ 一括投入モード（{len(new_cols)}列）: 理由={args.batch_reason}")
        print("     → これはスクリーニングです。採用・棄却の判断は LOO 分解の後に行ってください")
    new_result = run_cv(args.model, params, seed=args.seed, features=new_features)
    new_oof = new_result["oof_score"]
    new_stats = _cv_stats(new_result)
    print(f"  New  OOF: {new_oof:.5f}")
    _print_cv_stats("New", new_stats)

    delta = new_oof - base_oof
    gap_delta = new_stats["gap"] - base_stats["gap"]

    # ΔOOFがノイズ範囲・棄却域にある場合、gapの変化で「純粋なノイズ」か
    # 「軽度の過学習兆候」かを判別する（CV内部診断）。
    # 実績（過去コンペ）: gap_delta≈+0.00004（ノイズ）はLBで改善、gap_delta≈+0.0002前後（軽度過学習兆候）はLBでも悪化、
    # gap_delta>+0.01（明確な過学習）はLBで明確に悪化、という一貫した対応関係を確認済み。
    GAP_NOTABLE = 0.0005

    if delta > 0.001:
        verdict = "✅ 採用推奨"
    elif delta > 0.0003:
        verdict = "🔶 採用検討（他モデルでも確認を推奨）"
    elif delta > -0.0003:
        if gap_delta > GAP_NOTABLE:
            verdict = "⬜ ノイズ範囲だが軽度の過学習兆候あり（gap拡大、要注意）"
        else:
            verdict = "⬜ ノイズ範囲（過学習兆候なし、採用不要）"
    else:
        if gap_delta > GAP_NOTABLE:
            verdict = "❌ 棄却（過学習傾向を伴う明確な悪化）"
        else:
            verdict = "❌ 棄却"

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
 ΔOOF      : {delta:+.5f}

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
