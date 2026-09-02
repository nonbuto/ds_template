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
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def _previous_experiment_scores() -> Optional[float]:
    """log.csv の最新行（＝直前の実験）の oof_score を返す。無ければ None。

    指針#31「ΔOOF が fold 間 std より小さいなら、その差は測れていない」の自動判定に使う。
    """
    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    for row in reversed(rows):
        raw = (row.get("oof_score") or "").strip()
        if not raw:
            continue
        m = re.search(r"[01]\.\d+", raw)     # "0.95092(anchor)" のような表記にも対応
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                continue
    return None


def _find_reserved_row() -> Optional[int]:
    """`/ds-new-experiment` が予約した行（目的は記入済み・スコアは空欄）の行番号を返す。

    予約行の条件: `experiment_question` が埋まっており、かつ `oof_score` と `cv_val_mean` が空。
    見つからなければ None。末尾から探索するため、最新の予約行が優先される。

    背景: 予約行を作る設計と `_get_next_experiment_id()`（最大ID+1 を採番）が噛み合わず、
    目的・成功基準だけの行とスコアだけの行に情報が分裂する事故が起きていた。
    """
    if not LOG_CSV_PATH.exists():
        return None
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    for idx in range(len(rows) - 1, -1, -1):
        r = rows[idx]
        has_purpose = bool((r.get("experiment_question") or "").strip())
        no_score = not (r.get("oof_score") or "").strip() and not (r.get("cv_val_mean") or "").strip()
        if has_purpose and no_score:
            return idx
    return None


def _get_next_experiment_id() -> str:
    """次の experiment_id を採番する。**予約行があればその ID を再利用する**。"""
    if not LOG_CSV_PATH.exists():
        return "001"
    with open(LOG_CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "001"

    reserved = _find_reserved_row()
    if reserved is not None:
        rid = (rows[reserved].get("experiment_id") or "").strip()
        if rid:
            print(f"ℹ️  予約行を検出しました（experiment_id={rid}）。同じ行に結果を書き込みます")
            return rid

    ids = [int(r["experiment_id"]) for r in rows if r.get("experiment_id", "").isdigit()]
    return str((max(ids) + 1) if ids else 1).zfill(3)


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

    baseline = float(np.median(values))
    limit = baseline + PUB_OOF_GAP_THRESHOLD
    recent = [(i, g) for i, g in scored[-window:] if g is not None and g > limit]
    if not recent:
        return None

    lines = ", ".join(f"exp{i}({g:+.5f})" for i, g in recent)
    return (
        f"\n⚠️  Public 過剰浮上警告（`G-TWOAXIS`）: pub_oof_gap が基準線 +{PUB_OOF_GAP_THRESHOLD} を超えました。\n"
        f"   基準線（全 {len(values)} 提出の中央値）= {baseline:+.5f} / 閾値 = {limit:+.5f}\n"
        f"   超過した実験: {lines}\n"
        f"   → Public が OOF より浮いている＝シェイクダウンのリスク。SESSION.md に記録し、\n"
        f"     Final 2 でこの系統に偏らせないこと（OOF を犠牲にして gap を操作しない）"
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
    """
    if not LOG_CSV_PATH.exists():
        with open(LOG_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
            writer.writeheader()
        return

    try:
        with open(LOG_CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception:
        return

    missing = [c for c in LOG_CSV_COLUMNS if c not in existing]
    if not missing:
        return

    # 既知の列は順序どおりに、未知の列（手で足したもの）は末尾に残す
    fieldnames = LOG_CSV_COLUMNS + [c for c in existing if c not in LOG_CSV_COLUMNS]
    with open(LOG_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"🔧 log.csv に列を追加しました（既存 {len(rows)} 行は保持）: {', '.join(missing)}")


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
        self._experiment_id = _get_next_experiment_id()
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
        feature_df: Optional["pd.DataFrame"] = None,  # type: ignore[name-defined]
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
            from src.metrics import get_metric, needs_proba
        except ImportError:
            return

        oof_preds, labels = np.asarray(oof_preds), np.asarray(labels)
        lines = ["\n📋 OOF 誤差分析"]

        try:
            pred_for_metric = oof_preds
            if not needs_proba() and oof_preds.ndim == 2:
                pred_for_metric = np.argmax(oof_preds, axis=1)
            elif needs_proba() and oof_preds.ndim == 2 and oof_preds.shape[1] == 2:
                pred_for_metric = oof_preds[:, 1]
            lines.append(f"  OOF スコア: {get_metric()(labels, pred_for_metric):.5f}")
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

        train_mean = float(np.mean(self._fold_train_scores)) if self._fold_train_scores else 0.0
        train_std = float(np.std(self._fold_train_scores)) if self._fold_train_scores else 0.0
        val_mean = float(np.mean(self._fold_val_scores)) if self._fold_val_scores else 0.0
        val_std = float(np.std(self._fold_val_scores)) if self._fold_val_scores else 0.0

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
            "cv_train_mean": f"{train_mean:.5f}",
            "cv_train_std": f"{train_std:.5f}",
            "cv_val_mean": f"{val_mean:.5f}",
            "cv_val_std": f"{val_std:.5f}",
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
        # 予約行（/ds-new-experiment が作った目的だけの行）があれば、追記ではなく**上書き**する。
        # 予約行の experiment_question / success_criteria / abort_criteria は保持する。
        reserved = _find_reserved_row()
        merged_into_reserved = False
        if reserved is not None:
            with open(LOG_CSV_PATH, newline="") as f:
                rows = list(csv.DictReader(f))
            if (rows[reserved].get("experiment_id") or "").strip() == row["experiment_id"]:
                for key in ("experiment_question", "success_criteria", "abort_criteria"):
                    row[key] = rows[reserved].get(key, "")
                # 予約時に記入済みの description / notes は、空でなければ活かす
                for key in ("description", "notes"):
                    if not row[key] and rows[reserved].get(key):
                        row[key] = rows[reserved][key]
                rows[reserved] = row
                with open(LOG_CSV_PATH, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                merged_into_reserved = True
                print(f"✅ 予約行（experiment_id={row['experiment_id']}）に結果をマージしました（行の重複なし）")

        if not merged_into_reserved:
            with open(LOG_CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
                writer.writerow(row)

        oof_str = f"{oof_score:.5f}" if oof_score is not None else "N/A"
        exp_id = self._experiment_id or "000"
        branch = _get_git_branch()
        train_val_gap = train_mean - val_mean
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

        # CV 内部診断（CLAUDE.md 指針#31）。OOF/LB だけで判断させないための常設表示。
        prev = _previous_experiment_scores()
        if val_std > 0:
            print(
                f"\n🔍 CV内部診断（指針#31）\n"
                f"  fold間 val std = {val_std:.5f}\n"
                f"  → **前実験との OOF 差がこの std を下回るなら、その差は「測れていない」**"
            )
            if prev is not None and oof_score is not None:
                d = oof_score - prev
                verdict = ("判別不能（std 未満）" if abs(d) < val_std
                           else "std を超える差")
                print(f"     前実験 OOF={prev:.5f} → 今回 {oof_score:.5f}  ΔOOF={d:+.5f}  … {verdict}")
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
