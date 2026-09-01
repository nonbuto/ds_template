"""ターン終了時の規律監査（`.claude/settings.json` の Stop hook から呼ばれる）。

`G-STEPWISE` は「1 実験 = 1 コミット」「OOF 判明後 5 分以内に commit」を定めるが、
これは s6e8 で最も頻繁に破られた規律であり、機構はゼロだった。守らせたい規律は
AI への指示ではなく**観測可能な結果の側から**測る（`G-MECH`）。

判定するもの（すべてファイル・git の実態から）:
  1. 未コミットの実験スクリプト（`experiments/runs/exp*.py`）
  2. log.csv に OOF が記録済みなのに、その実験 ID を含むコミットが存在しない
  3. 3 つの規律ガード（可視化・診断記録・推論成果物）

**ブロックはしない。** Stop hook でブロックすると停止と再開のループを招くため、
`systemMessage` でユーザーに提示するだけにする。ブロックしてよいのは実績のある
可視化ガード（`start_run()`）と、不可逆な提出ゲートだけ。

使い方（hook 経由。手動でも実行できる）:
    uv run python -m scripts.session_audit
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment import (  # noqa: E402
    LOG_CSV_PATH,
    _check_diagnostic_recording_guard,
    _check_inference_artifacts_window,
    _check_visualization_guard,
)

COMMIT_SCAN_LIMIT = 200   # git log を遡る件数
RECENT_ROWS = 5           # log.csv の直近何件をコミット照合するか


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=15).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def uncommitted_experiment_scripts() -> list[str]:
    """未コミット（未追跡 or 変更済み）の実験スクリプトを列挙する。"""
    # -uall: 未追跡ディレクトリを `?? dir/` に畳まず、ファイル単位で列挙させる
    out = _run(["git", "status", "--short", "-uall", "--", "experiments/runs/"])
    names = []
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        if path.endswith(".py") and Path(path).name.startswith("exp"):
            names.append(Path(path).name)
    return names


def experiments_without_commit() -> list[str]:
    """OOF が記録済みなのに、対応するコミットが見つからない実験 ID を列挙する。"""
    if not LOG_CSV_PATH.exists():
        return []
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []

    log_text = _run(["git", "log", f"-{COMMIT_SCAN_LIMIT}", "--pretty=%s%n%b"])
    missing = []
    for row in rows[-RECENT_ROWS:]:
        exp_id = (row.get("experiment_id") or "").strip()
        if not exp_id or not (row.get("oof_score") or "").strip():
            continue
        if f"exp{exp_id}" not in log_text:
            missing.append(exp_id)
    return missing


def build_report() -> str | None:
    """指摘があれば報告文を返す。何も無ければ None。"""
    sections: list[str] = []

    scripts = uncommitted_experiment_scripts()
    if scripts:
        shown = ", ".join(scripts[:5]) + ("…" if len(scripts) > 5 else "")
        sections.append(
            f"未コミットの実験スクリプト {len(scripts)} 件: {shown}\n"
            f"  → 1 実験 = 1 コミット（`G-STEPWISE`）。log.csv の記録と併せて消化してください"
        )

    no_commit = experiments_without_commit()
    if no_commit:
        ids = ", ".join(f"exp{i}" for i in no_commit)
        sections.append(
            f"OOF 記録済みだがコミットが見つからない実験: {ids}\n"
            f"  → OOF 判明後 5 分以内に commit する規約です（`G-STEPWISE`）"
        )

    for warning in (_check_visualization_guard(),
                    _check_diagnostic_recording_guard(),
                    _check_inference_artifacts_window()):
        if warning:
            sections.append(warning.strip())

    if not sections:
        return None
    body = "\n\n".join(f"• {s}" for s in sections)
    return f"📋 セッション監査（機械判定・作業はブロックしません）\n\n{body}"


def main() -> int:
    report = build_report()
    if report:
        print(json.dumps({"systemMessage": report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
