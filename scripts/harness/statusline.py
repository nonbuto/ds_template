"""ターミナルのステータスバーに「今どうなっているか」を常時表示する。

`.claude/settings.json` の `statusLine` から呼ばれる。Claude Code は stdin に
セッション情報の JSON を渡し、標準出力の 1 行をステータスバーに表示する。

**なぜこれが要るか**: 実験の現在地・実行中ジョブ・締切と提出枠は、これまで
「AI に尋ねる」か「コマンドを叩く」でしか分からなかった。過去コンペでは
「動いていますか？」「また止まってませんか？」の確認が繰り返され、
古い締切見積もりのまま長時間ジョブを始めかけた事故もあった。
ステータスバーなら**コンテキストを 1 文字も消費せず**に常時見える。

**性能上の絶対条件**: 毎ターン（`refreshInterval` を設定すれば数秒ごとにも）呼ばれる。
- **同期ネットワーク呼び出しをしない。** 提出枠は `deadline_status` が API を叩いた
  ついでに書いたキャッシュを読むだけ。キャッシュが無い/古ければ「—」と出す
- ファイル読み取りだけで完結させ、**失敗しても例外を投げず短い文字列を返す**
  （ここで落ちるとステータスバーが壊れるだけでなく、毎ターンエラーが出る）

使い方（手動確認）:
    echo '{}' | uv run python -m scripts.harness.statusline
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
sys.path.insert(0, str(ROOT))

SEP = "  │  "
STALE_MIN = 15   # ハートビートがこれ以上古ければ「?」を付ける


def _experiment() -> str:
    """直近の実験の現在地（ID・モデル・OOF・前実験比の Δ）。"""
    path = ROOT / "experiments" / "log.csv"
    if not path.exists():
        return ""
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return ""
    scored = [r for r in rows if (r.get("oof_score") or "").strip()]
    if not scored:
        return f"exp— ({len(rows)}件)" if rows else ""

    last = scored[-1]
    out = f"exp{(last.get('experiment_id') or '?').strip()}"
    model = (last.get("model") or "").strip()
    if model:
        out += f" {model[:12]}"
    try:
        oof = float(last["oof_score"])
        out += f" {oof:.5f}"
        if len(scored) >= 2:
            d = oof - float(scored[-2]["oof_score"])
            out += f" ({d:+.5f})"
    except (ValueError, KeyError):
        pass
    return out


def _jobs() -> str:
    """実行中ジョブ（無ければ空文字＝表示しない）。"""
    d = ROOT / "experiments" / ".running"
    if not d.exists():
        return ""
    files = sorted(d.glob("*.json"))
    if not files:
        return ""

    now, parts, stale = datetime.now(), [], False
    for p in files[:2]:                       # 表示は 2 件まで
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        seg = f"exp{st.get('experiment_id', p.stem)}"
        folds = st.get("folds_done")
        if folds is not None:
            seg += f" f{folds}"
        try:
            started = datetime.strptime(st["started_at"], "%Y-%m-%d %H:%M:%S")
            seg += f" {int((now - started).total_seconds() // 60)}分"
        except Exception:
            pass
        try:
            updated = datetime.strptime(st["updated_at"], "%Y-%m-%d %H:%M:%S")
            if (now - updated).total_seconds() / 60 >= STALE_MIN:
                stale = True                  # 無応答＝ハングの疑い
        except Exception:
            pass
        parts.append(seg)
    if not parts:
        return ""
    more = f"+{len(files) - len(parts)}" if len(files) > len(parts) else ""
    return f"▶{'⚠' if stale else ''} {' '.join(parts)}{more}"


def _deadline_and_slots() -> str:
    """締切までの残り時間と本日の提出枠（枠はキャッシュのみ・API は叩かない）。"""
    out = []
    try:
        from scripts.harness.deadline_status import (
            parse_deadline, read_deadline_from_competition_md, read_submission_cache)
    except Exception:
        return ""

    try:
        text = read_deadline_from_competition_md()
        dl = parse_deadline(text) if text else None
        if dl:
            rest = (dl - datetime.now(timezone.utc)).total_seconds()
            if rest < 0:
                out.append("締切超過")
            else:
                h = int(rest // 3600)
                mark = "🔴" if h < 6 else ("🟡" if h < 24 else "")
                out.append(f"{mark}残{h}h{int(rest % 3600 // 60):02d}m")
    except Exception:
        pass

    try:
        c = read_submission_cache()
        out.append(f"提出{c['used']}/{c['limit']}" if c else "提出—")
    except Exception:
        pass
    return " ".join(out)


def main() -> int:
    try:
        sys.stdin.read()          # hook 入力は使わないが、読み捨てて詰まりを防ぐ
    except Exception:
        pass
    try:
        parts = [p for p in (_experiment(), _jobs(), _deadline_and_slots()) if p]
        print(SEP.join(parts) if parts else "ds-template")
    except Exception:
        print("ds-template")      # ここで落とさない（毎ターン出るエラーになる）
    return 0


if __name__ == "__main__":
    sys.exit(main())
