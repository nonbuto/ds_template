"""実行中の学習ジョブの状態を表示する（生きているか・どこまで進んだか・あと何分か）。

過去コンペでは長時間ジョブのあいだ「動いていますか？」「また止まってませんか？」を
人が繰り返し尋ねる必要があった。プロセスの生死・fold の進捗・残り時間はすべて
機械が答えられる情報なので、ここで一括表示する。

判定材料:
  - `experiments/.running/{exp_id}.json` の存在（= 実行中）と `updated_at`（= 生存確認）
  - 記録された PID が実在するか（プロセスが消えていれば異常終了）
  - log.csv の `duration_sec` 実測から、同じモデルの所要時間を引いて ETA を出す

使い方:
    uv run python -m scripts.harness.job_status
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
sys.path.insert(0, str(ROOT))

from scripts.harness.deadline_status import runtime_stats  # noqa: E402
from src.experiment import RUNNING_DIR  # noqa: E402

STALE_MINUTES = 15   # これ以上ハートビートが更新されないならハングを疑う


def _pid_alive(pid: int | None) -> bool | None:
    """PID が生きているか。判定できない場合は None。"""
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True     # 存在するが権限が無い
    except Exception:
        return None


def _eta(model: str, elapsed_min: float, folds_done: int) -> str:
    """同じモデルの実測中央値から残り時間を見積もる。"""
    median_by_model = {m: med for m, _n, med in runtime_stats()}
    med = median_by_model.get(model)
    if med:
        remaining = med / 60 - elapsed_min
        return (f"残り約 {remaining:.0f} 分（{model} の実測中央値 {med / 60:.0f} 分から）"
                if remaining > 0 else
                f"実測中央値（{med / 60:.0f} 分）を超過中")
    if folds_done >= 1:
        per_fold = elapsed_min / folds_done
        return f"1 fold あたり {per_fold:.1f} 分（実測中央値なし）"
    return "見積もり不可（実測データなし）"


def main() -> int:
    if not RUNNING_DIR.exists() or not any(RUNNING_DIR.glob("*.json")):
        print("実行中のジョブはありません")
        print("（`start_run()` を経由した実験だけがここに表示されます）")
        return 0

    now = datetime.now()
    print(f"実行中のジョブ（{now:%Y-%m-%d %H:%M:%S}）")
    for path in sorted(RUNNING_DIR.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(f"  {path.stem}: 状態ファイルを読めません")
            continue

        exp_id = state.get("experiment_id", path.stem)
        model = state.get("model", "?")
        folds = state.get("folds_done", 0)
        try:
            started = datetime.strptime(state["started_at"], "%Y-%m-%d %H:%M:%S")
            elapsed_min = (now - started).total_seconds() / 60
        except Exception:
            elapsed_min = 0.0
        try:
            updated = datetime.strptime(state["updated_at"], "%Y-%m-%d %H:%M:%S")
            silent_min = (now - updated).total_seconds() / 60
        except Exception:
            silent_min = 0.0

        alive = _pid_alive(state.get("pid"))
        if alive is False:
            status = "❌ プロセスが存在しない（異常終了。状態ファイルが残っています）"
        elif silent_min >= STALE_MINUTES:
            status = f"⚠️ {silent_min:.0f} 分間ハートビート更新なし（ハングを疑う）"
        else:
            status = "🟢 稼働中"

        print(f"\n  exp{exp_id}  {model}  {status}")
        print(f"    経過      : {elapsed_min:.0f} 分 / fold 完了 {folds}")
        print(f"    最終更新  : {silent_min:.0f} 分前")
        print(f"    ETA       : {_eta(model, elapsed_min, folds)}")
        if state.get("description"):
            print(f"    内容      : {state['description']}")

    print("\n異常終了で残った状態ファイルは削除して構いません: experiments/.running/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
