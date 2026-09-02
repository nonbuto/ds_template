"""締切までの残り時間と本日（UTC）の提出使用数を 1 コマンドで表示する。

長時間ジョブ（30 分超）を開始する前と、前回の時刻確認から 30 分以上経過している場合は
必ずこれを実行してから着手すること（→ CLAUDE.md「提出枠の管理方針」）。

過去コンペでは、古い時刻確認に基づく「残り約 9 時間」という見積もり（実際は約 4.5 時間）で
5 時間規模の学習ジョブを開始しかけ、締切直前まで気づかない一歩手前だった。

使い方:
    uv run python -m scripts.harness.deadline_status
    uv run python -m scripts.harness.deadline_status --deadline "2026-08-31 23:59"  # 手動指定
"""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# 提出上限の定義元は `src/config.py`。ここに直書きすると、コンペごとに変わる値が
# ハーネスの中に隠れて、submit_gate の表示と食い違う（実際に 2 箇所に散っていた）。
from src.config import DAILY_SUBMISSION_LIMIT as DAILY_LIMIT


def read_deadline_from_competition_md() -> str | None:
    """COMPETITION.md の基本情報テーブルから締切を読む（`| 締め切り | ... |` 行）。"""
    from src.config import COMPETITION_MD
    path = COMPETITION_MD
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "締め切り" in line and "|" in line:
            cells = [c.strip() for c in line.split("|")]
            for c in cells:
                if re.search(r"\d{4}-\d{2}-\d{2}", c):
                    return c
    return None


def parse_deadline(text: str) -> datetime | None:
    """`2026-08-31 23:59 UTC` のような文字列を UTC の datetime にする。"""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}))?", text)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = int(m.group(4)) if m.group(4) else 23
    mm = int(m.group(5)) if m.group(5) else 59
    return datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)


def count_todays_submissions(competition: str) -> int | None:
    """`kaggle competitions submissions` から本日（UTC）の提出数を数える。"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        out = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", competition, "--csv", "--page-size", "200"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    # kaggle CLI は警告行を混ぜることがあるため、日付フィールドの一致だけを数える
    return sum(1 for line in out.splitlines() if today in line)


SUBMISSION_CACHE = Path(__file__).resolve().parents[2] / "experiments" / ".statusline_cache.json"
CACHE_TTL_SEC = 600   # statusLine が読む側の許容鮮度


def write_submission_cache(competition: str, used: int, limit: int) -> None:
    """Kaggle API を叩いたついでに提出枠をキャッシュする。

    `statusline.py` は毎ターン呼ばれるため**同期ネットワーク呼び出しができない**。
    API を叩くのはこちら（deadline_status / submit_gate）だけにし、
    statusLine はこのキャッシュを読むだけにする。
    """
    import json
    try:
        SUBMISSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SUBMISSION_CACHE.write_text(json.dumps({
            "competition": competition, "used": used, "limit": limit,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }), encoding="utf-8")
    except Exception:
        pass          # キャッシュの失敗で本処理を止めない


def read_submission_cache(ttl_sec: int = CACHE_TTL_SEC) -> dict | None:
    """鮮度内のキャッシュを返す。無い・古い・壊れている場合は None。"""
    import json
    try:
        d = json.loads(SUBMISSION_CACHE.read_text(encoding="utf-8"))
        at = datetime.strptime(d["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if (datetime.now(timezone.utc) - at).total_seconds() > ttl_sec:
            return None
        return d
    except Exception:
        return None


def runtime_stats() -> list[tuple[str, int, float]]:
    """log.csv の `duration_sec` からモデル別の実測中央値を返す。

    30 分ルールは「推定実行時間」を求めるが、推定を較正する**実測**がどこにも残らなかった。
    過去コンペでは推定 1 時間の実験が実測 4 時間 17 分（しかも 4/25 fold）で、
    締切直前の貴重な時間を溶かした。推定の根拠はここから取る。
    """
    import csv
    from statistics import median

    path = Path(__file__).resolve().parents[2] / "experiments" / "log.csv"
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []

    by_model: dict[str, list[float]] = {}
    for row in rows:
        raw = (row.get("duration_sec") or "").strip()
        model = (row.get("model") or "?").strip() or "?"
        if raw:
            try:
                by_model.setdefault(model, []).append(float(raw))
            except ValueError:
                continue
    return sorted(((m, len(v), median(v)) for m, v in by_model.items()),
                  key=lambda t: -t[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=str, default=None, help="例: '2026-08-31 23:59'（UTC）")
    parser.add_argument("--limit", type=int, default=DAILY_LIMIT, help="1 日あたりの提出上限")
    parser.add_argument("--competition", type=str, default=None, help="コンペ slug（省略時は src.config から）")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    print(f"現在時刻 (UTC): {now:%Y-%m-%d %H:%M:%S}")

    deadline_text = args.deadline or read_deadline_from_competition_md()
    deadline = parse_deadline(deadline_text) if deadline_text else None
    if deadline:
        remaining = deadline - now
        hours = remaining.total_seconds() / 3600
        mark = "🔴" if hours < 6 else ("🟡" if hours < 24 else "🟢")
        print(f"締切     (UTC): {deadline:%Y-%m-%d %H:%M}")
        print(f"{mark} 残り時間     : {int(hours)}時間{int(remaining.total_seconds() % 3600 // 60)}分")
        if hours < 0:
            print("   ⚠️ 締切を過ぎています")
    else:
        print("締切     (UTC): 不明（COMPETITION.md の「締め切り」行に日付が無い。--deadline で指定可）")

    competition = args.competition
    if competition is None:
        try:
            from src.config import COMPETITION

            competition = COMPETITION
        except Exception:
            competition = None

    if competition:
        used = count_todays_submissions(competition)
        if used is None:
            print(f"本日の提出   : 取得失敗（kaggle CLI を確認）  コンペ: {competition}")
        else:
            write_submission_cache(competition, used, args.limit)
            left = args.limit - used
            mark = "🔴" if left == 0 else ("🟡" if left <= 2 else "🟢")
            print(f"{mark} 本日の提出   : {used}/{args.limit} 使用済み（残り {left} 枠）  コンペ: {competition}")
    else:
        print("本日の提出   : コンペ slug 不明（--competition で指定可）")

    stats = runtime_stats()
    if stats:
        print("\n実測の実行時間（log.csv の duration_sec・中央値）— 推定はここから取る:")
        for model, n, med in stats[:5]:
            mark = "⚠️" if med >= 30 * 60 else "  "
            print(f"  {mark} {model:<20} {med / 60:6.1f} 分  (n={n})")
    else:
        print("\n実測の実行時間: まだ記録がありません（duration_sec は start_run 経由で自動記録）")

    if deadline and (deadline - now).total_seconds() / 3600 < 12:
        print("\n⚠️ 締切まで 12 時間を切っています。"
              "30 分を超える長時間ジョブを始める前に、実測ペースと残り時間を必ず突き合わせること。")


if __name__ == "__main__":
    main()
