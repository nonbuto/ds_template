"""
規律ガード（機械的強制）— CLAUDE.md `G-MECH` / `G-DIAG`

2 つの規律を、AI の自己申告ではなくファイルの実態から判定する:
  - **可視化ガード**: 直近 N 実験のあいだに `data/output/plots/` へ新規 .png があるか
    （log.csv のタイムスタンプと .png の更新時刻の比較）
  - **診断記録ガード**: 直近 N 実験で `cv_train_mean` / `cv_val_std` が記録されているか
    （tracker を経由しない使い捨てスクリプトが `G-DIAG` を空洞化させていないか）

AI の自己申告・記憶に依存しないことが唯一の設計目的。

2 つの経路から呼ばれる:
  1. `ExperimentTracker.end_run()` の末尾（`_check_visualization_guard()` を直接呼ぶ）
     → `scripts/train.py` 等 tracker を使う実験を確実にカバーする
  2. `.claude/settings.json` の PostToolUse hook（本スクリプト）
     → tracker を経由せず log.csv へ直接追記する使い捨て実験スクリプトもカバーする

使い方（`src` を import するため -m 形式で起動する）:
    uv run python -m scripts.viz_guard            # 2つのガードを判定（常に exit 0）
    uv run python -m scripts.viz_guard --window 5 # 判定窓（実験数）を変える
"""

import argparse
import sys

from src.experiment import (VIZ_GUARD_WINDOW, _check_diagnostic_recording_guard,
                            _check_visualization_guard)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=VIZ_GUARD_WINDOW,
                        help="直近N実験を判定窓とする（default: %(default)s）")
    args = parser.parse_args()

    for warning in (_check_visualization_guard(window=args.window),
                    _check_diagnostic_recording_guard()):
        if warning:
            print(warning)
    # 警告は「気づかせる」ためのものでありワークフローを止めない。常に正常終了する。
    return 0


if __name__ == "__main__":
    sys.exit(main())
