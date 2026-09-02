"""コンテキスト圧縮の直前に、機械が知っている事実を SESSION.md へ退避する。

`.claude/settings.json` の PreCompact hook から呼ばれる。

長いセッションではコンテキストが自動圧縮され、そのたびに「今どこにいるか」の復元に
コストがかかる（s6e8 のセッションは実際に複数回圧縮された）。圧縮で失われるのは
会話の細部だが、**直近の実験・実行中のジョブ・コミット状況は圧縮とは無関係に
ファイルから読める事実**なので、消える前にディスクへ書いておけば復元は確実になる。

書き込むのは SESSION.md の**マーカーで囲まれた 1 ブロックだけ**。
人・AI が書いた各セクション（ステージ・次にやること・方針）には触れない。

使い方（hook 経由。手動でも実行できる）:
    uv run python -m scripts.harness.session_snapshot
"""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
sys.path.insert(0, str(ROOT))

from src.config import SESSION_MD  # noqa: E402
from src.experiment import LOG_CSV_PATH, RUNNING_DIR  # noqa: E402

BEGIN = "<!-- BEGIN:auto-snapshot (scripts/harness/session_snapshot.py が生成・手で編集しない) -->"
END = "<!-- END:auto-snapshot -->"
RECENT = 3


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def build_block() -> str:
    lines = [BEGIN, "## 自動スナップショット（圧縮直前に機械記録・上限行数の対象外）",
             f"- **記録時刻**: {datetime.now():%Y-%m-%d %H:%M:%S}"]

    if LOG_CSV_PATH.exists():
        try:
            with open(LOG_CSV_PATH, newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []
        if rows:
            recent = " / ".join(
                f"exp{r.get('experiment_id', '?')} {r.get('model', '')}"
                f" OOF={(r.get('oof_score') or '—').strip() or '—'}"
                f" LB={(r.get('submit_score') or '—').strip() or '—'}"
                for r in rows[-RECENT:]
            )
            lines.append(f"- **直近の実験**（全 {len(rows)} 件）: {recent}")

    running = sorted(RUNNING_DIR.glob("*.json")) if RUNNING_DIR.exists() else []
    if running:
        ids = ", ".join(f"exp{p.stem}" for p in running)
        lines.append(f"- **実行中のジョブ**: {ids} → `uv run python -m scripts.harness.job_status` で確認")

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "?"
    dirty = len([l for l in _run(["git", "status", "--short"]).splitlines() if l])
    head = _run(["git", "log", "-1", "--pretty=%h %s"])
    lines.append(f"- **git**: `{branch}` / 未コミット {dirty} 件 / HEAD: {head}")
    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    if not SESSION_MD.exists():
        return 0
    text = SESSION_MD.read_text(encoding="utf-8")
    block = build_block()

    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"

    SESSION_MD.write_text(text, encoding="utf-8")
    print(f"📌 SESSION.md に自動スナップショットを記録しました（{SESSION_MD}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
