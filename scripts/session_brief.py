"""セッション開始時の現在地ブリーフ（`.claude/settings.json` の SessionStart hook から呼ばれる）。

CLAUDE.md は「新しいセッション開始時は必ず `/ds-resume` を実行する」と定めるが、
実行を保証する機構は無かった（AI が忘れれば文脈ゼロのまま作業が始まる）。
このスクリプトは `/ds-resume` の**機械的な部分**（現在地の復元）を自動化する。

`/ds-resume` スキルは不要にならない。スキルの本体は「ユーザーと次の一手を合意する対話」であり、
ここが担うのは対話の前提となる事実の提示だけ。

**30 行以内に収める。** SessionStart の出力は毎セッションのコンテキストを消費するため、
CLAUDE.md（650 行）と併せた予算を守る。

使い方（hook 経由。手動でも実行できる）:
    uv run python -m scripts.session_brief
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment import LOG_CSV_PATH  # noqa: E402

MAX_LINES = 30
RECENT_EXPERIMENTS = 3


def _section(text: str, heading: str) -> list[str]:
    """SESSION.md から `## <heading>` 直下の箇条書き・テーブル行を取り出す。"""
    lines, capture = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            capture = heading in line
            continue
        if capture:
            s = line.strip()
            if s and not s.startswith("<!--") and not s.startswith("---"):
                lines.append(s)
    return lines


def _is_placeholder(line: str) -> bool:
    """テンプレートの例示行（「（例: …）」を含む行・記号だけの行）を除外する。"""
    stripped = line.lstrip("-*0123456789. |").strip()
    return (not stripped) or ("（例:" in stripped) or set(stripped) <= set("—-| ")


def build_brief() -> str:
    out: list[str] = ["━━━ セッション現在地（SessionStart hook · 機械生成）━━━"]

    session_md = ROOT / "SESSION.md"
    if session_md.exists():
        text = session_md.read_text(encoding="utf-8")
        stage = [l for l in _section(text, "現在のステージ") if not _is_placeholder(l)]
        nxt = [l for l in _section(text, "次にやること") if not _is_placeholder(l)]
        blockers = [l for l in _section(text, "未解決の問い") if not _is_placeholder(l)]
        if stage:
            out += ["【ステージ】"] + [f"  {l}" for l in stage[:2]]
        if nxt:
            out += ["【次にやること】"] + [f"  {l}" for l in nxt[:3]]
        if blockers:
            out += ["【未解決の問い】"] + [f"  {l}" for l in blockers[:2]]
    else:
        out.append("SESSION.md がありません → /ds-kickoff から始めてください")

    # ── 直近の実験 ──
    if LOG_CSV_PATH.exists():
        try:
            with open(LOG_CSV_PATH, newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []
        if rows:
            out.append(f"【直近の実験】（全 {len(rows)} 件）")
            for row in rows[-RECENT_EXPERIMENTS:]:
                oof = (row.get("oof_score") or "—").strip() or "—"
                lb = (row.get("submit_score") or "—").strip() or "—"
                dur = (row.get("duration_sec") or "").strip()
                dur_s = f" {int(float(dur)) // 60}分" if dur else ""
                out.append(f"  exp{row.get('experiment_id', '?')} {row.get('model', '')} "
                           f"OOF={oof} LB={lb}{dur_s} — {(row.get('description') or '')[:40]}")
        else:
            out.append("【直近の実験】まだ記録がありません（Stage 1 の最小ベースラインから）")

    # ── 規律監査（Stop hook と同じ判定を再利用）──
    try:
        from scripts.session_audit import build_report
        report = build_report()
    except Exception:
        report = None
    if report:
        out.append("【要対応】")
        out += [f"  {l}" for l in report.splitlines()[2:] if l.strip()][:6]

    out.append("→ 対話の再開は /ds-resume（このブリーフは事実の提示のみ）")

    if len(out) > MAX_LINES:
        out = out[:MAX_LINES - 1] + ["  …（省略。詳細は /ds-resume）"]
    return "\n".join(out)


def main() -> int:
    brief = build_brief()
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": brief,
        }
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
