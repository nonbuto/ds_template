"""実験の「目的・成功基準・撤退基準」を log.csv に予約行として記録する。

**なぜこれがあるか**: `/ds-new-experiment` は「log.csv に予約追記する」と指示しながら、
**その手段を用意していなかった**。結果、機械化されている列だけが埋まり、
手作業の列は落ちた（前コンペ 271 実験の実測）:

    experiment_question  35%
    success_criteria     35%
    abort_criteria       35%
    learning             88%   ← end_run / submit の経路が支えている

`G-MECH` の教科書どおりの形 —— **手で書けと言うだけの規律は守られない**。

予約行を作ると `ExperimentTracker.start_run()` がその ID を引き継ぎ、
学習結果が同じ行にマージされる（目的とスコアが 1 行に揃う）。

使い方:
    uv run python -m scripts.harness.reserve_experiment \\
        --name "lgb_h012_age_ratio" \\
        --question "age_ratio を足すと OOF が改善するか" \\
        --success "ΔOOF が採用判定の床を超える（--n-repeats 3）" \\
        --abort   "ΔOOF が負、または gap が +0.0005 以上拡大"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="実験名（log.csv の experiment_name）")
    parser.add_argument("--question", required=True,
                        help="この実験で**何を明らかにするか**（手段ではなく目的）")
    parser.add_argument("--success", required=True, help="どうなれば「この方向で正しい」か")
    parser.add_argument("--abort", required=True, help="どうなれば「中止・別の方向へ」か")
    parser.add_argument("--model", default="", help="使うモデル（分かっていれば）")
    parser.add_argument("--description", default="", help="1 行の説明")
    args = parser.parse_args()

    from src.experiment import LOG_CSV_COLUMNS, LOG_CSV_PATH, _ensure_log_csv
    from src.utils.csvlock import locked_csv

    _ensure_log_csv()
    with locked_csv(LOG_CSV_PATH, LOG_CSV_COLUMNS) as rows:
        ids = [int(r["experiment_id"]) for r in rows if (r.get("experiment_id") or "").isdigit()]
        exp_id = str((max(ids) + 1) if ids else 1).zfill(3)
        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": exp_id,
            "experiment_name": args.name,
            "description": args.description,
            "model": args.model,
            "experiment_question": args.question,
            "success_criteria": args.success,
            "abort_criteria": args.abort,
        })

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 実験を予約しました（experiment_id={exp_id}）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 目的     : {args.question}
 成功基準 : {args.success}
 撤退基準 : {args.abort}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
次に学習を回すと、`ExperimentTracker` がこの行に結果をマージします
（目的とスコアが 1 行に揃う）。**結果が出たら 5 分以内に commit してください。**
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
