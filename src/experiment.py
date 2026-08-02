"""
実験トラッキングモジュール

experiments/log.csv への人間可読なサマリーの追記（Excel/Numbers等で開ける）と、
MLflow（任意インストール）によるアーティファクト管理を行う。

MLflowは必須ではありません。`uv add mlflow` で追加すると利用できます。
（※ pandas>=3 との互換性のある mlflow バージョンを確認してください）
"""

import csv
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import EXPERIMENTS_DIR, PLOTS_DIR, RANDOM_STATE

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


def _ensure_log_csv() -> None:
    if not LOG_CSV_PATH.exists():
        with open(LOG_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
            writer.writeheader()


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
    _fold_train_scores: list[float] = field(default_factory=list, repr=False)
    _fold_val_scores: list[float] = field(default_factory=list, repr=False)

    def start_run(
        self,
        description: str = "",
        model: str = "",
        features: str = "",
        notes: str = "",
        tags: Optional[dict] = None,
    ) -> str:
        """実験を開始する。実験IDを返す。"""
        if description:
            self.description = description
        if model:
            self.model = model
        if features:
            self.features = features
        if notes:
            self.notes = notes

        self._experiment_id = _get_next_experiment_id()

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
        """OOF予測の誤差分析レポートを出力する。

        - 高信頼度FP/FNの件数と主要セグメントを表示
        - feature_df を渡した場合はセグメント別残差を表示
        - output_dir を渡した場合は .npy ファイルとして保存
        """
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return

        auc = roc_auc_score(labels, oof_preds)
        global_mean = labels.mean()
        threshold = 0.5

        fp_high = ((oof_preds > 0.8) & (labels == 0)).sum()
        fn_high = ((oof_preds < 0.2) & (labels == 1)).sum()
        abs_err = float(np.abs(oof_preds - labels).mean())

        print(
            f"\n📋 OOF誤差分析\n"
            f"  OOF AUC: {auc:.5f}\n"
            f"  平均絶対誤差: {abs_err:.4f}\n"
            f"  高信頼度FP (prob>0.8, label=0): {fp_high:,}件\n"
            f"  高信頼度FN (prob<0.2, label=1): {fn_high:,}件\n"
            f"  ※ /ds-eda-visual でセグメント別残差を可視化できます"
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            exp_id = self._experiment_id or "000"
            np.save(output_dir / f"oof_{exp_id}.npy", oof_preds)

    def end_run(
        self,
        train_scores: Optional[list[float]] = None,
        val_scores: Optional[list[float]] = None,
        oof_score: Optional[float] = None,
        n_features: int = 0,
    ) -> None:
        """実験を終了し、experiments/log.csv に追記する。"""
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

        _ensure_log_csv()
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

        # 可視化ガード（機械判定。CLAUDE.md 指針#9 / TODO_TEMPLATE 2026-08-01 CRITICAL）
        viz_warning = _check_visualization_guard()
        if viz_warning:
            print(viz_warning)
