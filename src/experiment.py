"""
実験トラッキングモジュール

experiments/log.csv への人間可読なサマリーの追記（Excel/Numbers等で開ける）と、
MLflow（任意インストール）によるアーティファクト管理を行う。

MLflowは必須ではありません。`uv add mlflow` で追加すると利用できます。
（※ pandas>=3 との互換性のある mlflow バージョンを確認してください）
"""

import csv
import json
import os
import re
import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:      # 型注釈のためだけの import（実行時には読み込まない）
    import pandas as pd

import numpy as np

from src.config import EXPERIMENTS_DIR, OOF_DIR, PLOTS_DIR, RANDOM_STATE

# MLflowはオプション依存
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

LOG_CSV_PATH = EXPERIMENTS_DIR / "log.csv"

LOG_CSV_COLUMNS = [
    "timestamp",
    "experiment_id",
    "experiment_name",
    "description",
    "model",
    "features",
    "cv_train_mean",
    "cv_train_std",
    "cv_val_mean",
    "cv_val_std",
    "fold_val_scores",   # fold ごとの val スコア（";" 区切り）。**対応差の床を出すのに要る**
    "oof_score",
    "submit_score",
    "lb_rank",
    "n_folds",
    "n_features",
    "duration_sec",       # start_run から end_run までの実測秒数（ETA 較正の材料）
    "git_hash",
    "git_branch",
    "notes",
    # 実験サイクル列（/ds-new-experiment と /ds-kaggle-submit スキルが記録）
    "experiment_question",  # この実験で何を明らかにしたいか（/ds-new-experiment が記録）
    "success_criteria",     # どんな結果なら成功か（/ds-new-experiment が記録）
    "abort_criteria",       # どんな結果なら中止するか（/ds-new-experiment が記録）
    "learning",             # 実験から何を学んだか（/ds-kaggle-submit が記録）
    "oof_lb_gap",           # oof_score − submit_score（/ds-kaggle-submit が記録）
]
# 注: ベスト実験の管理は SESSION.md のスコアテーブルで一元化する（is_best 列は持たない）


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


VIZ_GUARD_WINDOW = 5   # 直近N実験のあいだに1枚も可視化が無ければ警告する


def _check_visualization_guard(window: int = VIZ_GUARD_WINDOW) -> Optional[str]:
    """直近 `window` 件の実験期間中に可視化が生成されたかを機械的に判定する。

    CLAUDE.md 指針#9 の「必須発動条件③（直近5実験で可視化ゼロ）」を、AI の自己申告ではなく
    タイムスタンプ比較で判定する。log.csv の window 件前の実験時刻より新しい .png が
    PLOTS_DIR に1枚も無ければ警告文字列を返す（無ければ None）。

    背景: 「努力目標」→「発動条件の明示」の2世代とも実効性が無く、2コンペ連続で
    「可視化が最初の数日に集中し以降ほぼゼロ」という同一の形で形骸化した。
    AI の自己監査に依存しない機械的な検知が第3世代の対策（TODO_TEMPLATE 2026-08-01 CRITICAL）。
    """
    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if len(rows) < window:
        return None

    # window 件前の実験のタイムスタンプ = 判定の基準時刻
    try:
        since = datetime.strptime(rows[-window]["timestamp"][:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, KeyError):
        return None

    if not PLOTS_DIR.exists():
        recent_plots = []
    else:
        recent_plots = [
            p for p in PLOTS_DIR.glob("*.png")
            if datetime.fromtimestamp(p.stat().st_mtime) >= since
        ]
    if recent_plots:
        return None

    return (
        f"\n⚠️  可視化ガード発動: 直近{window}実験（{since:%Y-%m-%d %H:%M} 以降）で "
        f"{PLOTS_DIR.name}/ に新規の可視化が1枚もありません。\n"
        f"   CLAUDE.md 指針#9 の必須発動条件③に該当します。次の実験に進む前に実行してください:\n"
        f"     uv run python scripts/feature_report.py     # importance / ΔOOF\n"
        f"     uv run python scripts/visualize.py          # 分布・誤差分析\n"
        f"   （この警告は AI の自己申告ではなくタイムスタンプ比較による機械判定です）"
    )


DIAG_GUARD_WINDOW = 10   # 直近N実験の診断列の記録率を見る
DIAG_GUARD_MIN_RATE = 0.7


def _check_diagnostic_recording_guard(window: int = DIAG_GUARD_WINDOW) -> Optional[str]:
    """CV 内部診断（cv_train_mean / cv_val_std）が実際に記録されているかを判定する。

    CLAUDE.md `G-DIAG` は「OOF と必ず併記する」と定めるが、log.csv に列があっても
    使い捨てスクリプトが ExperimentTracker を経由しないと空欄のまま積み上がる。
    過去コンペでは記入率が cv_train_mean 28% / cv_val_std 21% まで落ち、
    「記録されない診断は存在しないのと同じ」状態になった。

    「規約を読んだか」は観測できないが「診断が記録されたか」は観測できる。
    導線（CONVENTIONS.md への参照）が機能しているかを、結果側から測るガード。
    """
    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    # スコアが入っている＝完了した実験だけを母数にする（予約行は除外）
    done = [r for r in rows if (r.get("oof_score") or "").strip()][-window:]
    if len(done) < window:
        return None

    filled = sum(
        1 for r in done
        if (r.get("cv_train_mean") or "").strip() and (r.get("cv_val_std") or "").strip()
    )
    rate = filled / len(done)
    if rate >= DIAG_GUARD_MIN_RATE:
        return None

    return (
        f"\n⚠️  診断記録ガード発動: 直近{len(done)}実験のうち CV 内部診断"
        f"（cv_train_mean / cv_val_std）が揃っているのは {filled} 件（{rate:.0%}）です。\n"
        f"   ExperimentTracker を経由しない使い捨てスクリプトが原因である可能性が高いです。\n"
        f"   `G-DIAG` の 3 診断軸が機能せず、ΔOOF が fold 間 std より小さいかを判定できません。\n"
        f"   → CONVENTIONS.md#experimenttracker-の使い方 を確認し、\n"
        f"     学習ループ内で tracker.log_fold_scores(fold, tr, val) を呼んでください。\n"
        f"   （この警告は AI の自己申告ではなく log.csv の記入率による機械判定です）"
    )


def _check_inference_artifact_guard(exp_id: str) -> Optional[str]:
    """OOF を保存したのに test 予測（＝提出ファイルの材料）が無い実験を検知する。

    CLAUDE.md `G-STEPWISE` は「学習 → OOF + test 予測 → 提出ファイル」を 1 つの流れとして
    完結させることを求める。学習だけして推論を省くと、後で「やっぱり提出したい」となった
    ときに**同じ学習をもう一度回すことになる**（過去コンペで多発し、数時間規模を空費した）。
    ここは自己申告では守られないので、成果物の有無から機械的に判定する（`G-MECH`）。
    """
    if not OOF_DIR.exists():
        return None
    has_oof = any(OOF_DIR.glob(f"oof_{exp_id}*.npy"))
    has_test = any(OOF_DIR.glob(f"test_{exp_id}*.npy"))
    if not has_oof or has_test:
        return None
    return (
        f"\n⚠️ 推論成果物ガード: exp{exp_id} は OOF のみで test 予測が保存されていません。\n"
        f"   → `test_{exp_id}_*.npy` を同じ実行内で保存してください"
        f"（学習 → OOF + test → 提出ファイルは 1 つの流れ・`G-STEPWISE`）。\n"
        f"   提出候補になりうる実験なら `submission_path()` で CSV まで作り切ること。\n"
        f"   ΔOOF スクリーニング目的（feature_study 等）で意図的に省く場合はこの警告を無視してよい。"
    )


INFER_GUARD_WINDOW = 5   # 直近N実験の推論成果物を見る


def _check_inference_artifacts_window(window: int = INFER_GUARD_WINDOW) -> Optional[str]:
    """直近 `window` 件のうち「OOF はあるのに test 予測が無い」実験を列挙する。

    `_check_inference_artifact_guard()` は tracker 経由（`end_run()`）でしか呼べない。
    log.csv へ直接追記する使い捨てスクリプトもカバーするため、hook 経路から呼ぶ窓版を用意する。
    """
    if not LOG_CSV_PATH.exists() or not OOF_DIR.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    missing = []
    for row in rows[-window:]:
        exp_id = (row.get("experiment_id") or "").strip()
        if not exp_id or not (row.get("oof_score") or "").strip():
            continue
        if any(OOF_DIR.glob(f"oof_{exp_id}*.npy")) and not any(OOF_DIR.glob(f"test_{exp_id}*.npy")):
            missing.append(exp_id)
    if not missing:
        return None

    ids = ", ".join(f"exp{i}" for i in missing)
    return (
        f"\n⚠️  推論成果物ガード: 直近{window}実験のうち {ids} は OOF のみで test 予測がありません。\n"
        f"   学習 → OOF + test → 提出ファイルは 1 つの流れです（CLAUDE.md `G-STEPWISE` / PLAYBOOK L-24）。\n"
        f"   → 実験スクリプトの最後で src.utils.finalize.save_run_outputs() を呼んでください。\n"
        f"   ΔOOF スクリーニング専用（feature_study 等）なら無視して構いません。"
    )


def _previous_fold_scores(exclude_id: Optional[str] = None,
                          model: Optional[str] = None,
                          features: Optional[str] = None) -> Optional[list[float]]:
    """比較可能な直近の実験の fold ごとの val スコアを返す。無ければ None。

    **同じ fold で比べた差**の標準誤差を出すために要る（`src/noise.py` の `fold_paired_se`）。
    fold 平均と `cv_val_std` だけでは、fold の難易度差が相殺されないので床が 10 倍高く出る。
    """
    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    for row in reversed(rows):
        if exclude_id and (row.get("experiment_id") or "").strip() == exclude_id:
            continue
        if model is not None and (row.get("model") or "").strip() != model:
            continue
        if features is not None and (row.get("features") or "").strip() != features:
            continue
        raw = (row.get("fold_val_scores") or "").strip()
        if not raw:
            continue
        try:
            return [float(v) for v in raw.split(";") if v.strip()]
        except ValueError:
            continue
    return None


def _previous_experiment_scores(exclude_id: Optional[str] = None,
                                model: Optional[str] = None,
                                features: Optional[str] = None) -> Optional[float]:
    """比較可能な直近の実験の oof_score を返す。無ければ None。

    `G-DIAG`「ΔOOF が fold 間 std より小さいなら、その差は測れていない」の自動判定に使う。

    `exclude_id` は**今まさに書き込んだ自分の行**を除くためのもの。以前はこの関数を
    行の書き込み**後**に呼んでいたため、直前の実験ではなく自分自身を拾い、
    **ΔOOF が常に ±0.00000 と表示されていた**（診断が常に「判別不能」を出す）。

    `model` / `features` を渡すと**条件が揃う実験だけ**を比較相手にする。
    以前は「最後に oof_score が入っている行」を無条件に取っていたため、
    CatBoost の直後に LightGBM を回すと**異種モデル間の差が「ΔOOF」として表示された**。
    診断機構そのものが `G-FAIR` 違反（条件の揃わない比較）を作っていた。
    条件が揃う行が無ければ None を返す —— 比較しないほうが、誤った比較より良い。
    """
    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    for row in reversed(rows):
        if exclude_id and (row.get("experiment_id") or "").strip() == exclude_id:
            continue
        if model is not None and (row.get("model") or "").strip() != model:
            continue
        if features is not None and (row.get("features") or "").strip() != features:
            continue
        raw = (row.get("oof_score") or "").strip()
        if not raw:
            continue
        # 以前は `[01]\.\d+` で、**0/1 で始まる値しか拾えず符号も落としていた**。
        # RMSE の 12.34 は "2.34" と読まれ、R² の -0.12 は +0.12 になる。
        m = re.search(r"-?\d+(?:\.\d+)?", raw)   # "0.95092(anchor)" のような表記にも対応
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                continue
    return None


RUNNING_DIR = EXPERIMENTS_DIR / ".running"


def _heartbeat_path(exp_id: str) -> Path:
    return RUNNING_DIR / f"{exp_id}.json"


def _heartbeat_write(exp_id: str, **fields) -> None:
    """実行中ジョブの状態を書く（存在＝実行中、更新時刻＝生存確認）。

    「まだ動いていますか」「また止まってませんか」を人が尋ねずに済むようにする。
    ハートビートが古ければハングまたはクラッシュと判定できる（`scripts/harness/job_status.py`）。
    """
    try:
        RUNNING_DIR.mkdir(parents=True, exist_ok=True)
        path = _heartbeat_path(exp_id)
        state = {}
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        state.update(fields)
        state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass   # ハートビートの失敗で学習を止めない


def _heartbeat_clear(exp_id: str) -> None:
    try:
        _heartbeat_path(exp_id).unlink(missing_ok=True)
    except Exception:
        pass


def _save_feature_snapshot(exp_id: str, feature_names: list[str],
                           model: str, oof_score: Optional[float]) -> None:
    """この実験の特徴量セットを JSON で残す（「今のベース」を機械可読にする）。"""
    from src.config import PARAMS_DIR

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = PARAMS_DIR / f"features_{exp_id}.json"
    path.write_text(json.dumps({
        "experiment_id": exp_id,
        "model": model,
        "oof_score": oof_score,
        "n_features": len(feature_names),
        "features": list(feature_names),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


PUB_OOF_GAP_WINDOW = 5        # 直近N実験を見る
PUB_OOF_GAP_THRESHOLD = 0.0005   # 基準線（中央値）からの許容幅


def _check_pub_oof_gap_guard(window: int = PUB_OOF_GAP_WINDOW) -> Optional[str]:
    """Public が OOF より過剰に浮いた実験を検知する（`G-TWOAXIS` の監視ルール）。

    CLAUDE.md は「全実験の pub_oof_gap 中央値を基準線として記録し、基準線 +0.0005 を
    超えたら Public 過剰浮上警告を記録する」と数値つきで定めているが、機構が無かった。
    数値つきの監視義務ほど、締切間際に自己申告では守られない（`G-MECH`）。

    符号の定義に注意: log.csv の `oof_lb_gap` = OOF − LB。
    `pub_oof_gap` = LB − OOF なので符号を反転して評価する。
    正に大きいほど Public が浮いており、シェイクダウンのリスクが高い。
    """
    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    def gap(row) -> Optional[float]:
        try:
            oof, lb = float(row["oof_score"]), float(row["submit_score"])
        except (KeyError, TypeError, ValueError):
            return None
        return lb - oof

    scored = [(r.get("experiment_id", "?"), gap(r)) for r in rows]
    values = [g for _, g in scored if g is not None]
    if len(values) < window:
        return None       # 基準線を引くだけの提出数がまだ無い

    # ── 基準線は「初期の安定期」で固定する ──
    # 以前は**全提出の中央値**を基準線にしていた。これには 2 つの欠陥があった。
    #
    # 1. 検知したい系統差そのものが基準線に吸収される。モンテカルロ（20,000 回）で
    #    帰無条件（真の gap=0）と真に危険な条件（全実験で gap=+0.004）の発火率が
    #    **93.1% vs 92.9% でほぼ同じ** —— この警告は情報を持っていなかった。
    # 2. 閾値 0.0005 が LB のノイズ床（実測 0.002 前後）より小さく、純粋なノイズで
    #    ほぼ常に発火していた（帰無条件で 84〜97%）。アラーム疲れを作るだけ。
    #
    # 直したいのは「**gap が後から広がっていないか**」なので、基準線は前半で固定し、
    # 閾値は LB のノイズ床から決める（`src/noise.py`）。
    # 5-fold OOF と全学習相当の test 予測を比べる以上、gap には正当な系統オフセットが
    # 常に乗る。それは初期の基準線に含まれるので、差分だけを見れば消える。
    n_base = max(window, len(values) // 2)
    baseline = float(np.median(values[:n_base]))
    limit = baseline + _pub_gap_threshold(rows)
    # **直近窓の中央値**で判定する。1 点でも超えたら鳴らす形だと、窓 5 件のうち
    # どれか 1 つが 2σ を超える確率が積み上がって偽陽性が増える（実測 20.7%）。
    # 見たいのは「後から系統的に浮いたか」なので、点ではなく水準で見る。
    recent_vals = [g for _, g in scored[-window:] if g is not None]
    if len(recent_vals) < window or float(np.median(recent_vals)) <= limit:
        return None
    recent = [(i, g) for i, g in scored[-window:] if g is not None and g > limit]

    lines = ", ".join(f"exp{i}({g:+.5f})" for i, g in recent)
    return (
        f"\n⚠️  Public 過剰浮上警告（`G-TWOAXIS`）: pub_oof_gap が初期の水準から離れました。\n"
        f"   基準線（最初の {n_base} 提出の中央値）= {baseline:+.5f} / 閾値 = {limit:+.5f}\n"
        f"   超過した実験: {lines}\n"
        f"   → Public が OOF より**後から**浮いている＝シェイクダウンのリスク。SESSION.md に記録し、\n"
        f"     Final 2 でこの系統に偏らせないこと（OOF を犠牲にして gap を操作しない）"
    )


def _pub_gap_threshold(rows: list) -> float:
    """Public 過剰浮上の閾値。**LB のノイズ床から決める**（固定値ではない）。

    `PUBLIC_TEST_ROWS` が設定されていれば解析式で床を出す。無ければ、
    観測された gap の**前半のばらつき**を床の代理として使う
    （後半のばらつきを混ぜると、検知したい変化そのものが閾値を押し上げる）。
    """
    from src.config import PUBLIC_POS_RATE, PUBLIC_TEST_ROWS
    from src.noise import min_detectable_difference, single_score_se

    gaps, scores = [], []
    for r in rows:
        try:
            oof, lb = float(r["oof_score"]), float(r["submit_score"])
        except (KeyError, TypeError, ValueError):
            continue
        gaps.append(lb - oof)
        scores.append(lb)

    if PUBLIC_TEST_ROWS and scores:
        try:
            # **陽性率を渡す。** 半々と決め打つと不均衡データで床が最大 4.6 倍過小になり、
            # このガードは「鳴りにくい方向」へずれる（実測: 陽性率 2% で 0.00463 vs 0.00101）。
            se = single_score_se(metric_name="auc", n=PUBLIC_TEST_ROWS,
                                 score=float(np.median(scores)),
                                 pos_rate=PUBLIC_POS_RATE or 0.5)
            return min_detectable_difference(se)
        except ValueError:
            pass          # auc 以外は解析式が無いので観測ベースへ落ちる

    half = gaps[: max(len(gaps) // 2, 3)]
    observed = float(np.std(half)) if len(half) >= 3 else PUB_OOF_GAP_THRESHOLD
    return max(min_detectable_difference(observed), PUB_OOF_GAP_THRESHOLD)


BELOW_FLOOR_WINDOW = 8       # 「床の下での探索」を判定する直近の実験数


def _check_below_floor_guard(window: int = BELOW_FLOOR_WINDOW) -> Optional[str]:
    """直近の実験がすべて「LB に現れる床」の下に収まっていないか。

    **これは飽和の宣言ではなく、測定の限界の通知。** 床の下で回し続けても、
    返ってくるのは「変わらなかった」という情報にならない結果だけになる。
    `G-PERSIST` は探索の縮小を禁じているが、**同じ土俵での微調整を続けること**は
    探索ではない。向きを変える判断材料としてこの警告を出す。

    前回コンペの実測（最終盤 47 提出）:
        LB = OOF + 0.00112 ± 0.00007  →  床 = 0.00013
        隣接実験の ΔOOF 中央値 = 0.000010、床を超えた隣接ペア = **0%**
        その間 8 日・32 提出で LB 更新はゼロだった。
    """
    from src.metrics import greater_is_better
    from src.noise import empirical_lb_floor

    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    floor = empirical_lb_floor(rows)
    if floor is None:
        return None

    scores = []
    for row in rows:
        raw = (row.get("oof_score") or "").strip()
        if raw:
            try:
                scores.append((row.get("experiment_id", "?"), float(raw)))
            except ValueError:
                continue
    if len(scores) < window + 1:
        return None

    best_before = max(v for _, v in scores[:-window])
    recent = scores[-window:]
    gains = [(eid, (v - best_before) * (1 if greater_is_better() else -1)) for eid, v in recent]
    # **「床を超えた」と数えてよいのは改善方向だけ。** 悪化の大きさで判定すると、
    # 大きく外した実験が 1 つあるだけで「まだ測れる領域にいる」と誤認する。
    if any(g > 0 and floor.ratio(g) >= 1 for _, g in gains):
        return None

    best_recent = max(gains, key=lambda t: t[1])
    return (
        f"\n⚠️  床下探索の通知（`G-CALIB-SUB`）: 直近 {window} 実験がすべて"
        f"「LB に現れる床」の下にあります。\n"
        f"   {floor}\n"
        f"   直近のベスト更新幅: exp{best_recent[0]} の {best_recent[1]:+.5f}"
        f"（床の {floor.ratio(best_recent[1]):.1f} 倍）\n"
        f"   → **飽和ではなく測定の限界**。同じ土俵の微調整を続けても結果は情報にならない。\n"
        f"     ①seed / fold を増やして床を下げる ②情報源を変える（`G-SOURCE`）\n"
        f"     ③集約に切り替える（`G-CEILING`）のいずれかを選ぶこと"
    )


LONG_RUN_THRESHOLD_SEC = 30 * 60   # CLAUDE.md「30分ルール」の閾値


def _format_duration(seconds: Optional[float]) -> str:
    """実行時間を表示用に整える。30 分ルールを超えたら実行環境の再検討を促す。"""
    if seconds is None:
        return "計測なし（start_run を経由していない）"
    text = f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    if seconds >= LONG_RUN_THRESHOLD_SEC:
        text += "  ⚠️ 30分超 — 次回は Kaggle Notebook GPU も選択肢に入れること"
    return text


def _ensure_log_csv() -> None:
    """log.csv を作成し、列が増えていれば既存ファイルを移行する。

    列を追加したあと既存ファイルへそのまま追記すると、ヘッダ（旧列数）と行（新列数）が
    食い違って**過去の実験記録が丸ごとずれる**。列の増減はテンプレート更新のたびに
    起こりうるので、追記の前に必ずここで整合させる。

    移行は**ロック下の原子的書き戻し**で行う。全行を読んで書き直す区間なので、
    その最中に別プロセスが追記すると、その行は書き戻しで消える（`src/utils/csvlock`）。
    """
    from src.utils.csvlock import locked_csv

    if not LOG_CSV_PATH.exists():
        with locked_csv(LOG_CSV_PATH, LOG_CSV_COLUMNS):
            pass                       # 区間を抜けるときにヘッダだけのファイルが書かれる
        return

    try:
        with open(LOG_CSV_PATH, newline="") as f:
            existing = list(csv.DictReader(f).fieldnames or [])
    except Exception:
        return

    missing = [c for c in LOG_CSV_COLUMNS if c not in existing]
    if not missing:
        return

    with locked_csv(LOG_CSV_PATH, LOG_CSV_COLUMNS) as rows:
        n = len(rows)                  # locked_csv が未知の列を末尾に残して書き戻す
    print(f"🔧 log.csv に列を追加しました（既存 {n} 行は保持）: {', '.join(missing)}")


def _fmt(value: float, digits: int = 5) -> str:
    """診断列の書式。測れなかった値（NaN）は "nan" ではなく**空欄**にする。

    "nan" を書くと診断記録ガードは「記入済み」と数えてしまい、記入率が実態より高く出る
    （ガードが空洞化する典型）。空欄なら未記入として正しく数えられる。
    """
    return "" if value != value else f"{value:.{digits}f}"


RUNNING_MARK_PREFIX = "（実行中"


def _running_mark() -> str:
    """行を「実行中」と印す。pid を含めるのは、落ちた実行の予約行を再利用できるようにするため。"""
    return f"{RUNNING_MARK_PREFIX} pid={os.getpid()} — end_run で結果を書き込む）"


def _is_claimed_and_alive(notes: str) -> bool:
    """その行を掴んでいるプロセスがまだ生きているか。

    死んでいれば（クラッシュ・kill）予約行を再利用できるようにする。
    生存確認をせずに「印があれば飛ばす」にすると、一度落ちた実験の予約行が
    永久に使えなくなり、`/ds-new-experiment` の予約が機能しなくなる。
    """
    if not notes.startswith(RUNNING_MARK_PREFIX):
        return False
    m = re.search(r"pid=(\d+)", notes)
    if not m:
        return True
    try:
        os.kill(int(m.group(1)), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # 別ユーザーのプロセス = 生きている


def _claim_experiment_id(experiment_name: str, model: str, description: str) -> str:
    """次の experiment_id を**ロック下で採番し、その場で行を確保する**。

    以前は採番（読むだけ）と記録（end_run での追記）が離れていたため、
    並行実行すると全員が同じ番号を名乗った。実測（8 プロセス同時）:

        ID: ['000', '000', '000', '000', '000', '000', '000', '000']

    `CLAUDE.md` は「バックグラウンド並行実行時も例外なし」として同時実行を前提にしているので、
    **番号を取ると同時に行を書き込み**、次のプロセスにその番号を見せる。
    `/ds-new-experiment` の予約行があるときは、新しい行を作らずその行を引き継ぐ。
    """
    from src.utils.csvlock import locked_csv

    with locked_csv(LOG_CSV_PATH, LOG_CSV_COLUMNS) as rows:
        # 予約行（目的は記入済み・スコアは空）があればその ID を再利用する
        for idx in range(len(rows) - 1, -1, -1):
            r = rows[idx]
            has_purpose = bool((r.get("experiment_question") or "").strip())
            no_score = (not (r.get("oof_score") or "").strip()
                        and not (r.get("cv_val_mean") or "").strip())
            if not (has_purpose and no_score):
                continue
            # **既に別プロセスが走り出している予約行は取らない。**
            # 以前はここで印を付けずに return していたため、ロックを抜けた瞬間に
            # 次のプロセスが同じ予約行を見つけ、8 プロセス同時で全員が同じ ID を名乗った
            # （実測: ['042'] × 8）。end_run は同じ ID の行を上書きするので、
            # **8 実験のうち 7 件分の記録が消える**。log.csv は唯一の台帳で git にも残らない。
            if _is_claimed_and_alive(r.get("notes") or ""):
                continue
            rid = (r.get("experiment_id") or "").strip()
            if rid:
                r["notes"] = _running_mark()          # ← ロック区間の中で印を付ける
                print(f"ℹ️  予約行を検出しました（experiment_id={rid}）。同じ行に結果を書き込みます",
                      file=sys.stderr)
                return rid

        ids = [int(r["experiment_id"]) for r in rows if (r.get("experiment_id") or "").isdigit()]
        exp_id = str((max(ids) + 1) if ids else 1).zfill(3)
        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": exp_id,
            "experiment_name": experiment_name,
            "description": description,
            "model": model,
            "notes": _running_mark(),
        })
    return exp_id


@dataclass
class ExperimentTracker:
    """実験トラッキングクラス。

    使い方:
        tracker = ExperimentTracker(experiment_name="playground-series-s6e3_lgb_baseline")
        tracker.start_run(description="ベースライン実験", model="lgb", features="raw_features")

        # 学習ループ内で
        tracker.log_fold_scores(fold=0, train_score=0.85, val_score=0.82)

        # 学習完了後
        tracker.end_run(train_scores=[...], val_scores=[...], oof_score=0.83, n_features=30)
    """

    experiment_name: str
    description: str = ""
    model: str = "lgb"
    features: str = ""
    notes: str = ""

    _experiment_id: Optional[str] = field(default=None, repr=False)
    _started_at: Optional[datetime] = field(default=None, repr=False)
    _fold_train_scores: list[float] = field(default_factory=list, repr=False)
    _fold_val_scores: list[float] = field(default_factory=list, repr=False)

    def start_run(
        self,
        description: str = "",
        model: str = "",
        features: str = "",
        notes: str = "",
        tags: Optional[dict] = None,
        skip_viz_check: bool = False,
    ) -> str:
        """実験を開始する。実験IDを返す。

        Args:
            skip_viz_check: 可視化ガードのブロックを明示的に無効化する（省略の意思表示）。
                環境変数 `DS_SKIP_VIZ_CHECK=1` でも同じ効果。

        Raises:
            RuntimeError: 可視化ガードが発動中（直近N実験で可視化ゼロ）のまま
                次の実験を開始しようとした場合。第4世代の対策として、警告の出力ではなく
                **実行を止める**（過去3世代とも「警告は出ていたが対応されない」形で形骸化した。
                特に締切直前ほど無視されやすい）。
        """
        if not (skip_viz_check or os.environ.get("DS_SKIP_VIZ_CHECK")):
            blocking = _check_visualization_guard()
            if blocking:
                raise RuntimeError(
                    f"{blocking}\n\n"
                    f"   ⛔ 可視化を実施するまで次の実験を開始できません（第4世代の機械的ゲート）。\n"
                    f"   意図的に省略する場合のみ、start_run(skip_viz_check=True) または\n"
                    f"   環境変数 DS_SKIP_VIZ_CHECK=1 を明示してください。"
                )

        if description:
            self.description = description
        if model:
            self.model = model
        if features:
            self.features = features
        if notes:
            self.notes = notes

        self._started_at = datetime.now()
        self._experiment_id = _claim_experiment_id(
            self.experiment_name, self.model, self.description)
        _heartbeat_write(self._experiment_id, experiment_name=self.experiment_name,
                         model=self.model, description=self.description,
                         started_at=self._started_at.strftime('%Y-%m-%d %H:%M:%S'),
                         folds_done=0, pid=os.getpid())

        if _MLFLOW_AVAILABLE:
            from src.config import MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.experiment_name)
            _tags = {"model": self.model, "features": self.features}
            if tags:
                _tags.update(tags)
            mlflow.start_run(tags=_tags)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("model", self.model)
            mlflow.log_param("features", self.features)

        print(f"🔬 実験開始: {self.experiment_name} (ID: {self._experiment_id})")
        return self._experiment_id

    def log_fold_scores(self, fold: int, train_score: float, val_score: float) -> None:
        """各フォールドのスコアを記録する。"""
        self._fold_train_scores.append(train_score)
        self._fold_val_scores.append(val_score)
        if self._experiment_id:
            _heartbeat_write(self._experiment_id, folds_done=len(self._fold_val_scores),
                             last_val_score=val_score)
        if _MLFLOW_AVAILABLE:
            mlflow.log_metric(f"fold_{fold}_train_score", train_score)
            mlflow.log_metric(f"fold_{fold}_val_score", val_score)

    def log_params(self, params: dict) -> None:
        """モデルパラメータを記録する。"""
        if _MLFLOW_AVAILABLE:
            mlflow.log_params(params)

    def save_oof_analysis(
        self,
        oof_preds: np.ndarray,
        labels: np.ndarray,
        feature_df: Optional["pd.DataFrame"] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """OOF 予測の誤差分析を出力する（問題種別に応じて内容を変える）。

        `G-MECH` は「③提出後（OOF-LB 乖離が大きいなら**誤差分析**）」を必須の可視化局面に
        挙げているが、`feature_report`（importance / ΔOOF）も `visualize`（分布）も
        誤差分析は担わない。ここが唯一の実装。

        以前は `roc_auc_score` と閾値 0.5 を直書きしており **二値分類でしか動かず、
        しかもどこからも呼ばれていなかった**（死蔵メソッド）。指標は `src.metrics` から取り、
        `PROBLEM_TYPE` で分岐するように直したうえで `scripts/train.py` から呼ぶようにした。
        """
        try:
            from src.config import PROBLEM_TYPE
            from src.metrics import get_metric, shape_for_metric
        except ImportError:
            return

        oof_preds, labels = np.asarray(oof_preds), np.asarray(labels)
        lines = ["\n📋 OOF 誤差分析"]

        try:
            # 整形は `shape_for_metric` に任せる（写経を増やさない。L-29 #2）
            lines.append(f"  OOF スコア: {get_metric()(labels, shape_for_metric(oof_preds)):.5f}")
        except Exception as e:
            lines.append(f"  OOF スコア: 計算できず（{type(e).__name__}）")

        if PROBLEM_TYPE == "regression":
            resid = labels - (oof_preds.ravel() if oof_preds.ndim > 1 else oof_preds)
            q = np.quantile(np.abs(resid), [0.5, 0.9, 0.99])
            lines += [f"  平均絶対誤差: {np.abs(resid).mean():.4f}",
                      f"  |残差| の分位点 50%/90%/99%: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}",
                      f"  最大の外れ {min(5, len(resid))} 件: "
                      f"{np.sort(np.abs(resid))[-5:][::-1].round(4).tolist()}"]
        elif oof_preds.ndim == 2 and oof_preds.shape[1] > 2:
            pred_cls = np.argmax(oof_preds, axis=1)
            lines.append(f"  正解率: {(pred_cls == labels).mean():.4f}")
            for c in np.unique(labels):
                mask = labels == c
                lines.append(f"    class {c}: n={mask.sum():,}  誤り {(pred_cls[mask] != c).sum():,} 件"
                             f"（{(pred_cls[mask] != c).mean():.1%}）")
        else:
            prob = oof_preds[:, 1] if oof_preds.ndim == 2 else oof_preds
            fp_high = int(((prob > 0.8) & (labels == 0)).sum())
            fn_high = int(((prob < 0.2) & (labels == 1)).sum())
            lines += [f"  平均絶対誤差: {float(np.abs(prob - labels).mean()):.4f}",
                      f"  高信頼度 FP (prob>0.8, label=0): {fp_high:,} 件",
                      f"  高信頼度 FN (prob<0.2, label=1): {fn_high:,} 件"]

        lines.append("  ※ 乖離が大きいときは /ds-eda-visual でセグメント別に可視化する")
        print("\n".join(lines))

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / f"oof_{self._experiment_id or '000'}.npy", oof_preds)

    def end_run(
        self,
        train_scores: Optional[list[float]] = None,
        val_scores: Optional[list[float]] = None,
        oof_score: Optional[float] = None,
        n_features: int = 0,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """実験を終了し、experiments/log.csv に追記する。

        Args:
            feature_names: この実験で実際に使った特徴量リスト。渡すと
                `params/features_{exp_id}.json` に保存され、「今どの特徴量がベースか」を
                機械可読な形で追える（`scripts/feature_report.py --sync` が参照する）。
                手書きの FEATURE_REPORT.md だけでは、試行が増えるとベースを見失う。
        """
        if feature_names is not None:
            n_features = n_features or len(feature_names)
        if train_scores is not None:
            self._fold_train_scores = train_scores
        if val_scores is not None:
            self._fold_val_scores = val_scores

        # `--resume` でキャッシュから復元した fold は train を再計算できないので NaN が入る。
        # **NaN を平均に混ぜると全体が NaN になり、`G-DIAG` の診断列が丸ごと空になる**ので、
        # 測れた fold だけで平均を取る（測れなかったことは fold 数の差として残る）。
        def _mean(vals) -> float:
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            return float(arr.mean()) if arr.size else float("nan")

        def _std(vals) -> float:
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            return float(arr.std()) if arr.size else float("nan")

        # **fold スコアを 1 度も記録しなかった実験は「0.00000」ではなく NaN。**
        # 以前は空リストのとき 0.0 を渡していたため、`_fmt()` の NaN → 空欄という対策に
        # 到達する前に無効化され、`log_fold_scores` を呼ばない実験が
        # 診断記録ガードに「記入済み」と数えられていた（実測で 100% すり抜け）。
        # しかも画面には**測っていない 0.00000 が診断値として表示される**。
        train_mean = _mean(self._fold_train_scores)
        train_std = _std(self._fold_train_scores)
        val_mean = _mean(self._fold_val_scores)
        val_std = _std(self._fold_val_scores)

        if _MLFLOW_AVAILABLE:
            mlflow.log_metric("cv_train_mean", train_mean)
            mlflow.log_metric("cv_val_mean", val_mean)
            if oof_score is not None:
                mlflow.log_metric("oof_score", oof_score)
            mlflow.end_run()

        _heartbeat_clear(self._experiment_id or "000")

        if feature_names is not None:
            _save_feature_snapshot(self._experiment_id or "000", feature_names,
                                   self.model, oof_score)

        _ensure_log_csv()
        duration = ((datetime.now() - self._started_at).total_seconds()
                    if self._started_at else None)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": self._experiment_id or "000",
            "experiment_name": self.experiment_name,
            "description": self.description,
            "model": self.model,
            "features": self.features,
            "cv_train_mean": _fmt(train_mean),
            "cv_train_std": _fmt(train_std),
            "cv_val_mean": _fmt(val_mean),
            "cv_val_std": _fmt(val_std),
            # fold 平均だけでは「同じ fold で比べた差のばらつき」が出せない。
            # 生の fold スコアを残すことで、次の実験が対応差の SE を計算できる（`src/noise.py`）
            # **表示用の 5 桁で保存しない。** 対応差は元の差が小さいので、丸めると
            # 差が全部 0 になり SE が 0 に潰れ、「z=+68 で改善」のような無意味な断定が出る。
            "fold_val_scores": ";".join(_fmt(v, digits=8) for v in self._fold_val_scores),
            "oof_score": f"{oof_score:.5f}" if oof_score is not None else "",
            "submit_score": "",          # /ds-kaggle-submit スキルが追記
            "lb_rank": "",               # /ds-kaggle-submit スキルが追記
            "n_folds": len(self._fold_val_scores),
            "n_features": n_features,
            "duration_sec": f"{duration:.0f}" if duration is not None else "",
            "git_hash": _get_git_hash(),
            "git_branch": _get_git_branch(),
            "notes": self.notes,
            "experiment_question": "",   # /ds-new-experiment スキルが記録
            "success_criteria": "",      # /ds-new-experiment スキルが記録
            "abort_criteria": "",        # /ds-new-experiment スキルが記録
            "learning": "",              # /ds-kaggle-submit スキルが記録
        }
        # 直前の実験のスコアは**自分の行を書く前に**読む（書いた後だと自分自身を拾う）
        prev = _previous_experiment_scores(exclude_id=row["experiment_id"],
                                           model=self.model, features=self.features)

        # 同じ experiment_id の行（start_run が確保した行、または /ds-new-experiment の
        # 予約行）があれば、追記ではなく**その行を上書き**する。目的・成功基準・撤退基準は
        # 予約時の記入を残す。読みから書き戻しまでを 1 つのロック区間に収めることが要点 ——
        # 別々にロックしても、その隙に他プロセスが追記すれば書き戻しで消える。
        from src.utils.csvlock import locked_csv

        merged_into_reserved = False
        with locked_csv(LOG_CSV_PATH, LOG_CSV_COLUMNS) as rows:
            for idx in range(len(rows) - 1, -1, -1):
                if (rows[idx].get("experiment_id") or "").strip() != row["experiment_id"]:
                    continue
                if (rows[idx].get("oof_score") or "").strip():
                    break                      # 結果まで入った行は上書きしない
                for key in ("experiment_question", "success_criteria", "abort_criteria"):
                    row[key] = rows[idx].get(key, "")
                for key in ("description", "notes"):
                    if not row[key] and rows[idx].get(key):
                        row[key] = rows[idx][key]
                if row["notes"].startswith(RUNNING_MARK_PREFIX):
                    row["notes"] = self.notes   # start_run が置いたプレースホルダは残さない
                rows[idx] = row
                merged_into_reserved = True
                break
            if not merged_into_reserved:
                rows.append(row)
        if merged_into_reserved:
            print(f"✅ 予約行（experiment_id={row['experiment_id']}）に結果をマージしました（行の重複なし）")

        oof_str = f"{oof_score:.5f}" if oof_score is not None else "N/A"
        exp_id = self._experiment_id or "000"
        branch = _get_git_branch()
        # gap は「train が val よりどれだけ**良い**か」。小さいほど良い指標（RMSE 等）では
        # 素の差の符号が逆になるため、指標の向きに合わせる（feature_study と同じ扱い）。
        from src.metrics import greater_is_better
        train_val_gap = (train_mean - val_mean) * (1 if greater_is_better() else -1)
        gap_note = "  ⚠️ gapが大きい可能性（過学習/校正不足を確認）" if train_val_gap > 0.01 else ""
        print(
            f"\n📊 実験記録完了 (ID: {exp_id})\n"
            f"  CV Train: {train_mean:.5f} ± {train_std:.5f}\n"
            f"  CV Val  : {val_mean:.5f} ± {val_std:.5f}\n"
            f"  Gap(train-val): {train_val_gap:.5f}{gap_note}\n"
            f"  OOF     : {oof_str}\n"
            f"  実行時間: {_format_duration(duration)}\n"
            f"  Branch  : {branch}\n"
            f"  log.csv : {LOG_CSV_PATH}"
        )

        # CV 内部診断（`G-DIAG`）。OOF/LB だけで判断させないための常設表示。
        if val_std > 0:
            print(f"\n🔍 CV内部診断（`G-DIAG`）\n"
                  f"  fold間 val std = {val_std:.5f}（CV 設計の安定性。**床ではない**）")
            if prev is None:
                print("     （同条件〔同じモデル・特徴量セット〕の直近実験が無いため ΔOOF は出しません）")
            elif oof_score is not None:
                # **床は fold 対応差の標準誤差で出す。** `cv_val_std` を床にしてはいけない ——
                # それは「fold ごとの難易度の差」を主成分に含み、同じ fold で 2 つを比べれば
                # 相殺する成分。実測では val std 0.01251 に対し正しい床は 0.00124（**10 倍**）で、
                # 実在する改善を体系的に「判別不能」と切り捨てていた。
                # これが L-19（個別 Δ≈0 が 13 系統累積すると確定的な正の差になった）の説明。
                from src.noise import fold_paired_se, verdict as noise_verdict

                d = (oof_score - prev) * (1 if greater_is_better() else -1)
                prev_folds = _previous_fold_scores(exclude_id=self._experiment_id or "",
                                                   model=self.model, features=self.features)
                se = (fold_paired_se(self._fold_val_scores, prev_folds)
                      if prev_folds and len(prev_folds) == len(self._fold_val_scores)
                      else float("nan"))
                print(f"     前実験 OOF={prev:.5f} → 今回 {oof_score:.5f}  ΔOOF={d:+.5f}")
                if se == se:      # NaN でない
                    # fold 差から推定した SE の自由度は fold 数 − 1。
                    # 正規の 2σ を当てると少数標本で甘くなる（`min_detectable_difference`）
                    df = max(len(self._fold_val_scores) - 1, 1)
                    print(f"     fold対応差の床: 1σ={se:.5f}  → {noise_verdict(d, se, df=df)}")
                else:
                    print("     fold対応差の床: 算出不可（前実験の fold スコアが無い/fold 数が違う）"
                          "\n     → 行単位で測るなら src.noise.paired_se(y, oof_new, oof_prev)")

                # **提出実績から測った「LB に現れるための床」**も出す。
                # CV 上で測れることと、LB に出ることは別問題（`G-CALIB-SUB`）。
                from src.noise import empirical_lb_floor

                lb_floor = empirical_lb_floor()
                if lb_floor is not None:
                    ratio = lb_floor.ratio(d)
                    tail = "  ← 床未満。LB には出ない公算が大きい" if ratio < 1 else ""
                    print(f"     {lb_floor}\n     今回の ΔOOF はその {ratio:.1f} 倍{tail}")
        if train_val_gap > 0.01:
            print(
                "  ⚠️ train−val 乖離が大きい。正則化に飛びつく前に、"
                "多クラス/不均衡タスクなら **校正不足**（class_weight・β・閾値）を先に疑うこと"
            )

        # コミットメッセージの提案（OOFスコア入り）
        commit_title = f"feat(exp{exp_id}): {self.description}"
        commit_body = f"OOF={oof_str}  model={self.model}  features={self.features}"
        print(
            f"\n💡 コミットメッセージ案:\n"
            f"  {commit_title}\n"
            f"  {commit_body}\n"
            f"  ↑ git add -p してから git commit -m '<上記>' で記録してください"
        )

        # 可視化ガード（機械判定。CLAUDE.md `G-MECH`）
        viz_warning = _check_visualization_guard()
        if viz_warning:
            print(viz_warning)

        # 診断記録ガード（機械判定。CLAUDE.md `G-DIAG` が実際に機能しているか）
        diag_warning = _check_diagnostic_recording_guard()
        if diag_warning:
            print(diag_warning)

        # 推論成果物ガード（機械判定。学習だけして提出材料が無い実験を検知する）
        infer_warning = _check_inference_artifact_guard(exp_id)
        if infer_warning:
            print(infer_warning)
