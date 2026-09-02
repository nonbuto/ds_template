"""現在地ブリーフ。`SessionStart` と `PostCompact` の 2 つの hook から呼ばれる。

**これは `/ds-resume` を強制するものではない。** `/ds-resume` はユーザーの儀式であり、
その価値は「現在地を合意して次の一手を決める対話」にある。ここが担うのは**下限の保証**——
儀式が挟まらない開始（本題から直接入る・`--continue`・**コンテキスト圧縮の直後**）でも、
AI がゼロ文脈で走り出さないようにする。

とくに `PostCompact` が効く運用がある。夜間の長時間学習などでセッションを起動したまま
使い続けると、新セッションは滅多に始まらない代わりに**圧縮が繰り返し起きる**。
圧縮は数万トークンを要約へ潰すので、そこへ 15 行のブリーフを戻す費用は失うものの 1% 未満。
`PreCompact`（`session_snapshot.py`）が SESSION.md へ書き、ここが圧縮後に読み直す。

**`MAX_LINES` 行以内に収める。** 出力はそのままコンテキストを消費する。

使い方（hook 経由。手動でも実行できる）:
    uv run python -m scripts.harness.session_brief
    uv run python -m scripts.harness.session_brief --event PostCompact
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
sys.path.insert(0, str(ROOT))

from src.config import SESSION_MD  # noqa: E402
from src.experiment import LOG_CSV_PATH  # noqa: E402

MAX_LINES = 30
MAX_LINE_CHARS = 100   # 1 行あたりの上限。**費用は行数ではなく文字数（トークン）で決まる**
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


def _clip(line: str) -> str:
    """1 行を `MAX_LINE_CHARS` に丸める。

    行数だけを見ていると、SESSION.md に長い 1 行（コンペ終了報告など 150 字超）が
    あったときに予算を素通りする。コンテキストの費用は行数ではなく文字数で決まる。
    """
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS - 1] + "…"


def _is_placeholder(line: str) -> bool:
    """テンプレートの例示行（「（例: …）」を含む行・記号だけの行）を除外する。"""
    stripped = line.lstrip("-*0123456789. |").strip()
    return (not stripped) or ("（例:" in stripped) or set(stripped) <= set("—-| ")


def build_brief(event: str = "SessionStart") -> str:
    out: list[str] = [f"━━━ セッション現在地（{event} hook · 機械生成）━━━"]

    session_md = SESSION_MD
    if session_md.exists():
        text = session_md.read_text(encoding="utf-8")
        stage = [l for l in _section(text, "現在のステージ") if not _is_placeholder(l)]
        nxt = [l for l in _section(text, "次にやること") if not _is_placeholder(l)]
        blockers = [l for l in _section(text, "未解決の問い") if not _is_placeholder(l)]
        if stage:
            out += ["【ステージ】"] + [f"  {_clip(l)}" for l in stage[:2]]
        if nxt:
            out += ["【次にやること】"] + [f"  {_clip(l)}" for l in nxt[:3]]
        if blockers:
            out += ["【未解決の問い】"] + [f"  {_clip(l)}" for l in blockers[:2]]
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
                out.append(_clip(f"  exp{row.get('experiment_id', '?')} {row.get('model', '')} "
                                 f"OOF={oof} LB={lb}{dur_s} — {(row.get('description') or '')[:40]}"))
        else:
            out.append("【直近の実験】まだ記録がありません（Stage 1 の最小ベースラインから）")

    # ── 実行中のジョブ（夜間の長時間学習から戻ったとき最初に知りたいこと）──
    try:
        from src.experiment import RUNNING_DIR
        running = sorted(RUNNING_DIR.glob("*.json")) if RUNNING_DIR.exists() else []
    except Exception:
        running = []
    if running:
        ids = ", ".join(f"exp{p.stem}" for p in running)
        out.append(f"【実行中のジョブ】{ids}"
                   f" → `uv run python -m scripts.harness.job_status`（生存・進捗・ETA）")

    # ── 規律監査（Stop hook と同じ判定を再利用）──
    try:
        from scripts.harness.session_audit import build_report
        report = build_report()
    except Exception:
        report = None
    if report:
        out.append("【要対応】")
        out += [f"  {_clip(l)}" for l in report.splitlines()[2:] if l.strip()][:6]

    out.append("→ 対話の再開は /ds-resume（このブリーフは事実の提示のみ）")

    if len(out) > MAX_LINES:
        out = out[:MAX_LINES - 1] + ["  …（省略。詳細は /ds-resume）"]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="SessionStart",
                        choices=["SessionStart", "PostCompact"],
                        help="呼び出し元の hook イベント名")
    args = parser.parse_args()

    brief = build_brief(args.event)
    payload: dict = {
        "hookSpecificOutput": {
            "hookEventName": args.event,
            "additionalContext": brief,
        }
    }
    if args.event == "PostCompact":
        brief = ("（コンテキスト圧縮が起きました。以下はファイルから読み直した現在地です）\n"
                 + brief)
        payload["hookSpecificOutput"]["additionalContext"] = brief
        # additionalContext の注入が効かなかった場合に備え、ユーザーには必ず見える形で
        # 退避先を知らせる（SESSION.md の恒久スナップショットは PreCompact が書いている）。
        payload["systemMessage"] = (
            "コンテキスト圧縮後に現在地を再注入しました"
            "（復元できない場合は SESSION.md の自動スナップショットを参照）"
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
