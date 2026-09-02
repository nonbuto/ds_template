"""状態ファイルの鮮度を機械判定する。

テンプレートは 7 つの状態ファイルを持つが、更新はすべて手動（スキルが促すだけ）で、
「更新されたか」は誰も観測していなかった。s6e8 では `FEATURE_REPORT.md` が
3 週間以上停滞し、「今どの特徴量がベースなのか」を追えなくなった。

判定は **log.csv の実験タイムスタンプと各ファイルの mtime の比較のみ**で行う。
内容の意味は見ない — 「規約を読んだか」は観測できないが「結果が記録されたか」は
観測できる、という `G-MECH` の設計原則どおり、観測可能な側だけで測る。

使い方:
    uv run python -m scripts.harness.state_audit
    uv run python -m scripts.harness.state_audit --window 10   # 何実験ぶん遡って見るか
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
sys.path.insert(0, str(ROOT))

from src.experiment import LOG_CSV_PATH  # noqa: E402

DEFAULT_WINDOW = 10   # 直近 N 実験のあいだに更新されたかを見る

# (ファイル名, 何を記録する場所か, 更新するスキル)
WATCHED: list[tuple[str, str, str]] = [
    ("state/SESSION.md", "現在地・次のアクション", "/ds-new-experiment · /ds-kaggle-submit"),
    ("state/FEATURE_REPORT.md", "各変数の特性・ΔOOF・採否", "/ds-eda-visual · /ds-fe-hypothesis"),
    ("state/FE_HYPOTHESES.md", "仮説の因果・棄却理由", "/ds-fe-hypothesis"),
    ("state/EDA_SUMMARY.md", "EDA の問いと発見", "/ds-eda-visual"),
    ("state/KAGGLE_RESEARCH.md", "外部調査の知見", "/ds-kaggle-research"),
]


def _experiments_since(window: int) -> tuple[datetime | None, int]:
    """判定の基準時刻（window 件前の実験時刻）と、全実験数を返す。"""
    if not LOG_CSV_PATH.exists():
        return None, 0
    try:
        with open(LOG_CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None, 0
    if len(rows) < window:
        return None, len(rows)
    try:
        since = datetime.strptime(rows[-window]["timestamp"][:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, KeyError, TypeError):
        return None, len(rows)
    return since, len(rows)


def stale_files(window: int = DEFAULT_WINDOW) -> list[tuple[str, str, str, int]]:
    """基準時刻以降に更新されていない状態ファイルを返す。

    Returns:
        (ファイル名, 役割, 更新するスキル, 最終更新からの日数) のリスト。
    """
    since, _ = _experiments_since(window)
    if since is None:
        return []

    stale = []
    now = datetime.now()
    for name, role, skill in WATCHED:
        path = ROOT / name
        if not path.exists():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if mtime < since:
            stale.append((name, role, skill, (now - mtime).days))
    return stale


def build_report(window: int = DEFAULT_WINDOW) -> str | None:
    stale = stale_files(window)
    if not stale:
        return None
    since, total = _experiments_since(window)
    lines = [
        f"状態ファイルの停滞: 直近 {window} 実験（{since:%Y-%m-%d %H:%M} 以降・全 {total} 件）"
        f"のあいだ更新されていないファイルがあります"
    ]
    for name, role, skill, days in stale:
        lines.append(f"  - {name}（{role}） 最終更新 {days} 日前 → {skill}")
    lines.append("  「考えたことは記録しなければ存在しなかったのと同じ」（CLAUDE.md 思考の外部化）")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="直近N実験を判定窓とする（default: %(default)s）")
    args = parser.parse_args()

    report = build_report(args.window)
    print(report if report else "✅ 状態ファイルの停滞はありません")
    return 0


if __name__ == "__main__":
    sys.exit(main())
