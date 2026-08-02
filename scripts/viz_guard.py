"""
可視化ガード（機械的強制）— CLAUDE.md 指針#9

直近 N 実験のあいだに `data/output/plots/` へ新規の可視化が生成されているかを、
log.csv のタイムスタンプと .png の更新時刻の比較だけで判定する。
AI の自己申告・記憶に依存しないことが唯一の設計目的。

2 つの経路から呼ばれる:
  1. `ExperimentTracker.end_run()` の末尾（`_check_visualization_guard()` を直接呼ぶ）
     → `scripts/train.py` 等 tracker を使う実験を確実にカバーする
  2. `.claude/settings.json` の PostToolUse hook（本スクリプト）
     → tracker を経由せず log.csv へ直接追記する使い捨て実験スクリプトもカバーする

使い方（`src` を import するため -m 形式で起動する）:
    uv run python -m scripts.viz_guard            # 判定して警告を出す（常に exit 0）
    uv run python -m scripts.viz_guard --window 5 # 判定窓（実験数）を変える
"""

import argparse
import sys

from src.experiment import VIZ_GUARD_WINDOW, _check_visualization_guard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=VIZ_GUARD_WINDOW,
                        help="直近N実験を判定窓とする（default: %(default)s）")
    args = parser.parse_args()

    warning = _check_visualization_guard(window=args.window)
    if warning:
        print(warning)
    # 警告は「気づかせる」ためのものでありワークフローを止めない。常に正常終了する。
    return 0


if __name__ == "__main__":
    sys.exit(main())
