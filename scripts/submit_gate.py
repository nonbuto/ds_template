"""Kaggle 提出の PreToolUse ゲート（`.claude/settings.json` の hook から呼ばれる）。

CLAUDE.md「Kaggle提出ルール」は **提出前に「対象ファイル名・本日の提出回数・残り回数」を
表示してユーザーの確認を取る**ことを AI に義務づけ、省略を「テンプレート違反」と定めている。
しかし従来これは AI の自己申告に完全に依存しており、機構はゼロだった（`G-MECH` 違反）。

このゲートは 2 つのことをする:

1. **事実を実測する** — 本日の提出数・締切までの残り時間は Kaggle API と時計から取得する。
   過去コンペの事故（古い時刻確認に基づく残り時間の誤り・提出枠の自己申告ズレ）は、
   すべて「提示された数字が記憶であって実測でなかった」ことが原因だった。
2. **人間に判断を渡す** — `permissionDecision: "ask"` を返すと Claude Code は
   **ユーザー本人に**承認プロンプトを出す。AI が確認を省略して実行することはできない
   （Bash が allowlist に入っていても ask が優先される）。

使い方（hook 経由。手動確認もできる）:
    echo '{"tool_name":"Bash","tool_input":{"command":"kaggle competitions submit -c x -f y.csv -m z"}}' \
      | uv run python -m scripts.submit_gate
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.deadline_status import (  # noqa: E402
    DAILY_LIMIT,
    count_todays_submissions,
    parse_deadline,
    read_deadline_from_competition_md,
)

SEPARATORS = {";", "&&", "||", "|", "&", "(", "{"}
ENV_ASSIGN_RE = re.compile(r"^\w+=")
SUBCOMMANDS = {"competitions", "c"}


def is_submit_command(command: str) -> bool:
    """実際に Kaggle 提出を実行するコマンドか判定する。

    提出コマンドを単に**文字列として含むだけ**のコマンド（ドキュメントの編集・grep・
    heredoc など）を誤検知しないよう、**コマンド位置**（先頭、または `;` `&&` `|` などの
    直後）に現れる場合だけ true にする。クォートで囲まれた部分は shlex が 1 トークンに
    まとめるため、文字列としての言及は自然に除外される。
    （この判定を入れる前は、本ファイルを編集する Bash 自身がブロックされた）
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        # クォートが閉じていない等でパースできないときは、安全側（確認を求める）に倒す
        return "submit" in command and "kaggle" in command

    at_command_position = True
    for i, tok in enumerate(tokens):
        if tok in SEPARATORS:
            at_command_position = True
            continue
        if at_command_position and ENV_ASSIGN_RE.match(tok):
            continue          # `VAR=value cmd` の環境変数はコマンド位置のまま
        if (at_command_position and tok == "kaggle"
                and tokens[i + 1:i + 2] and tokens[i + 1] in SUBCOMMANDS
                and tokens[i + 2:i + 3] == ["submit"]):
            return True
        at_command_position = False
    return False


def _extract_file(command: str) -> str | None:
    """`-f` / `--file` で指定された提出ファイルパスを取り出す。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for i, tok in enumerate(tokens):
        if tok in ("-f", "--file") and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith("--file="):
            return tok.split("=", 1)[1]
    return None


def _git_status() -> str:
    try:
        out = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "確認不能"
    if not out:
        return "clean ✅"
    return f"⚠️ 未コミット {len(out.splitlines())} 件（提出前は clean が規約）"


def build_brief(command: str) -> tuple[str, bool]:
    """提出内容の実測ブリーフを組み立てる。戻り値は (本文, 致命的エラーか)。"""
    lines: list[str] = ["Kaggle 提出の確認（数値はすべて実測値です）", ""]
    fatal = False

    # ── 対象ファイル ──
    path_str = _extract_file(command)
    if path_str is None:
        lines.append("対象ファイル : ⚠️ -f が見つかりません")
        fatal = True
    else:
        path = Path(path_str)
        abs_path = path if path.is_absolute() else ROOT / path
        if not abs_path.exists():
            lines.append(f"対象ファイル : ❌ 存在しません — {path_str}")
            fatal = True
        else:
            size_kb = abs_path.stat().st_size / 1024
            n_rows = sum(1 for _ in abs_path.open()) - 1
            conv = "✅ 命名規約に適合" if abs_path.name.startswith("sub_") else \
                   "⚠️ submission_path() 由来ではない可能性（sub_ で始まっていない）"
            lines.append(f"対象ファイル : {abs_path.name}")
            lines.append(f"               {n_rows:,} 行 / {size_kb:,.0f} KB / {conv}")

    # ── 本日の提出枠（UTC）──
    competition = None
    m = re.search(r"(?:-c|--competition)[= ]+(\S+)", command)
    if m:
        competition = m.group(1).strip("\"'")
    if competition is None:
        try:
            from src.config import COMPETITION
            competition = COMPETITION
        except Exception:
            competition = None

    now = datetime.now(timezone.utc)
    lines.append("")
    lines.append(f"現在時刻(UTC): {now:%Y-%m-%d %H:%M}")
    if competition:
        used = count_todays_submissions(competition)
        if used is None:
            lines.append(f"本日の提出   : ⚠️ 取得失敗（kaggle CLI を確認）  コンペ: {competition}")
        else:
            left = DAILY_LIMIT - used
            mark = "🔴" if left <= 0 else ("🟡" if left <= 2 else "🟢")
            lines.append(f"{mark} 本日の提出 : {used}/{DAILY_LIMIT} 使用済み → "
                         f"**これが {used + 1} 回目・提出後の残り {left - 1} 枠**")
            if left <= 0:
                lines.append("               ❌ 本日の上限に達しています（提出は失敗します）")
                fatal = True
    else:
        lines.append("本日の提出   : ⚠️ コンペ slug 不明")

    # ── 締切 ──
    dl_text = read_deadline_from_competition_md()
    deadline = parse_deadline(dl_text) if dl_text else None
    if deadline:
        hours = (deadline - now).total_seconds() / 3600
        mark = "🔴" if hours < 6 else ("🟡" if hours < 24 else "🟢")
        lines.append(f"{mark} 締切まで   : {int(hours)}時間{int((deadline - now).total_seconds() % 3600 // 60)}分"
                     f"（{deadline:%Y-%m-%d %H:%M} UTC）")
    else:
        lines.append("締切まで     : 不明（COMPETITION.md に締切行が無い）")

    lines.append(f"git status   : {_git_status()}")
    lines.append("")
    lines.append("提出後は log.csv の submit_score / oof_lb_gap と SESSION.md を更新すること。")
    return "\n".join(lines), fatal


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0   # hook の入力が読めないときは素通しする（作業を止めない）

    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    if not is_submit_command(command):
        return 0

    brief, fatal = build_brief(command)
    decision = "deny" if fatal else "ask"
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": brief,
        }
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
