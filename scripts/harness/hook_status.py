"""どの hook が実際に発火したかを実測で表示する。

**なぜ必要か**: hook を登録しても「本当に発火しているか」は誰も観測していなかった。
とくに次の 2 点は設定ファイルを眺めても分からない:

  1. **セッション中に追加した hook が、その走行中のセッションに反映されるか**
     （反映されないなら、追加した hook は次のセッションまで一度も動かない）
  2. **自動圧縮（コンテキスト上限による発火）でも `PreCompact` / `PostCompact` が呼ばれるか**
     （手動 `/compact` では呼ばれても、自動では呼ばれない可能性を排除できない）

推測で埋めず、`.claude/settings.json` の各 hook 先頭で発火時刻を追記させ、
ここで集計する。テンプレートの原則どおり、自己申告ではなく**結果の側から**測る。

使い方:
    uv run python -m scripts.harness.hook_status
    uv run python -m scripts.harness.hook_status --tail 20   # 直近の発火を時系列で見る
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
LOG_PATH = ROOT / "experiments" / ".hook_log"

EXPECTED = ["SessionStart", "PreToolUse", "PostToolUse", "Stop", "PreCompact", "PostCompact"]

NOTES = {
    "SessionStart": "セッション開始時のみ。長時間セッション運用ではほとんど発火しない",
    "PreToolUse": "Bash 実行のたび。提出コマンドの検知に使う",
    "PostToolUse": "Bash 実行のたび。log.csv 更新時だけガードを走らせる",
    "Stop": "ターン終了のたび",
    "PreCompact": "圧縮の直前。**自動圧縮でも発火するかがここで分かる**",
    "PostCompact": "圧縮の直後。**現在地の再注入が働いたかがここで分かる**",
}


def read_events() -> list[tuple[str, str]]:
    if not LOG_PATH.exists():
        return []
    events = []
    for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            events.append((parts[0], parts[1]))
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=int, default=0, help="直近N件を時系列で表示する")
    args = parser.parse_args()

    events = read_events()
    if not events:
        print(f"発火ログがありません（{LOG_PATH}）")
        print("hook を登録した直後は空です。しばらく作業してから再実行してください。")
        return 0

    counts = Counter(ev for _, ev in events)
    last = {}
    for ts, ev in events:
        last[ev] = ts

    print(f"hook 発火の実測（{LOG_PATH.name} / 全 {len(events)} 件）")
    print(f"  記録開始: {events[0][0]}\n")
    for ev in EXPECTED:
        n = counts.get(ev, 0)
        mark = "✅" if n else "❌"
        seen = f"{n:>5} 回  最終 {last.get(ev, '—')}" if n else "    未発火"
        print(f"  {mark} {ev:<14}{seen}")
        print(f"       {NOTES[ev]}")

    unknown = set(counts) - set(EXPECTED)
    if unknown:
        print(f"\n  想定外のイベント: {sorted(unknown)}")

    if not counts.get("PreCompact") and not counts.get("PostCompact"):
        print("\n⚠️ 圧縮系の hook がまだ一度も発火していません。判断がつくのは次のいずれか:")
        print("   (a) この期間に圧縮が起きていない（＝まだ何も言えない）")
        print("   (b) 自動圧縮では発火しない、または設定変更が走行中セッションに未反映")
        print("   → 圧縮が起きたと分かっているのに未発火なら (b) が濃厚。"
              "SESSION.md の自動スナップショット有無も併せて確認する")

    if args.tail:
        print(f"\n直近 {args.tail} 件:")
        for ts, ev in events[-args.tail:]:
            print(f"  {ts}  {ev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
