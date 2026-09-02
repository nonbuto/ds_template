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
      | uv run python -m scripts.harness.submit_gate
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
sys.path.insert(0, str(ROOT))

from scripts.harness.deadline_status import (  # noqa: E402
    DAILY_LIMIT,
    count_todays_submissions,
    parse_deadline,
    read_deadline_from_competition_md,
)

ENV_ASSIGN_RE = re.compile(r"^\w+=")
SUBCOMMANDS = {"competitions", "c"}
# 実行を包むだけで本体が次のトークンになるもの。透過して次を見る。
WRAPPERS = {"uv", "run", "nohup", "time", "env", "sudo", "xargs", "command", "exec",
            "poetry", "pipenv", "do", "then", "else", "elif"}
# shlex が演算子として返すトークン。ここでコマンド位置がリセットされる。
OPERATORS = {";", "&", "&&", "|", "||", "(", ")", "{", "}", ";;", "\n"}


def is_submit_command(command: str) -> bool:
    """実際に Kaggle 提出を実行するコマンドか判定する。

    **見逃しは提出の無確認実行に直結する**（この gate は不可逆な操作を止める唯一の関門）。
    以前は `shlex.split()` で空白分割してからコマンド位置を探していたが、
    **`shlex.split` は `;` や改行を演算子として扱わない**（`"echo a; kaggle …"` は
    `'a;'` に融合する）。そのため複数行コマンド・`;` 区切り・`uv run` 前置・絶対パスが
    すべて素通りしていた（8 パターン中 6 件）。

    いまは `punctuation_chars=True` の `shlex` で**クォートを尊重しつつ演算子も分離**する。

    **誤検知と見逃しは非対称**: 誤検知は確認を 1 回求めるだけだが、見逃しは
    無確認の提出になる。判断に迷う入力は**検知側（確認を求める）に倒す**。
    """
    for line in command.splitlines():          # 改行もコマンド区切り
        if not line.strip():
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            # クォートが閉じていない等。安全側（確認を求める）に倒す
            if "kaggle" in line and "submit" in line:
                return True
            continue

        k = 0
        while k < len(tokens):
            tok = tokens[k]
            if tok in OPERATORS or ENV_ASSIGN_RE.match(tok) or Path(tok).name in WRAPPERS:
                k += 1
                continue
            if (Path(tok).name == "kaggle"
                    and tokens[k + 1:k + 2] and tokens[k + 1] in SUBCOMMANDS
                    and tokens[k + 2:k + 3] == ["submit"]):
                return True
            # コマンド本体が来たので、次の演算子までは引数。読み飛ばす
            while k < len(tokens) and tokens[k] not in OPERATORS:
                k += 1
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
    # 端末から起動された（＝hook 入力が来ない）場合は待たずに終わる。
    # `json.load(sys.stdin)` は stdin が閉じられないとブロックし続けるため、
    # PreToolUse hook が毎回の Bash を最大 timeout 秒ハングさせる事故になる。
    if sys.stdin.isatty():
        print("hook 入力（JSON）が stdin に無いため何もしません。"
              "手動確認は次のように渡してください:\n"
              '  echo \'{"tool_name":"Bash","tool_input":{"command":"..."}}\''
              " | uv run python -m scripts.harness.submit_gate", file=sys.stderr)
        return 0
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0   # hook の入力が読めないときは素通しする（作業を止めない）

    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    if not is_submit_command(command):
        return 0

    # **内部エラーで素通しさせない。** 例外がそのまま出ると hook は非ゼロ終了し、
    # 提出コマンドは確認を経ずに実行されうる。ゲートが守るのは
    # 「取り消せない・回数制限つき・外部に見える」唯一の操作なので、
    # 壊れたときは通すのではなく**必ず確認を求める**側に倒す。
    try:
        brief, fatal = build_brief(command)
    except Exception as exc:                       # noqa: BLE001 — 全例外を確認に倒す
        brief = (f"⚠️ 提出前チェックが内部エラーで完了しませんでした（{type(exc).__name__}: {exc}）。\n"
                 "   ファイル名・本日の提出回数・残り枠を手で確認してから実行してください。")
        fatal = False
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
