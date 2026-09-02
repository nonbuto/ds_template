"""
規律ガード（機械的強制）— CLAUDE.md `G-MECH` / `G-DIAG`

2 つの規律を、AI の自己申告ではなくファイルの実態から判定する:
  - **可視化ガード**: 直近 N 実験のあいだに `data/output/plots/` へ新規 .png があるか
    （log.csv のタイムスタンプと .png の更新時刻の比較）
  - **診断記録ガード**: 直近 N 実験で `cv_train_mean` / `cv_val_std` が記録されているか
    （tracker を経由しない使い捨てスクリプトが `G-DIAG` を空洞化させていないか）
  - **推論成果物ガード**: 直近 N 実験に「OOF はあるのに test 予測が無い」ものがないか
    （学習だけして推論を省くと、提出時に同じ学習をやり直すことになる → `G-STEPWISE`）
  - **Public 過剰浮上ガード**: pub_oof_gap が基準線 +0.0005 を超えていないか（→ `G-TWOAXIS`）

AI の自己申告・記憶に依存しないことが唯一の設計目的。

2 つの経路から呼ばれる:
  1. `ExperimentTracker.end_run()` の末尾（`_check_visualization_guard()` を直接呼ぶ）
     → `scripts/train.py` 等 tracker を使う実験を確実にカバーする
  2. `.claude/settings.json` の PostToolUse hook（本スクリプト）
     → tracker を経由せず log.csv へ直接追記する使い捨て実験スクリプトもカバーする

使い方（`src` を import するため -m 形式で起動する）:
    uv run python -m scripts.harness.viz_guard            # 2つのガードを判定（常に exit 0）
    uv run python -m scripts.harness.viz_guard --window 5 # 判定窓（実験数）を変える
"""

import argparse
import json
import sys

from src.experiment import (VIZ_GUARD_WINDOW, _check_diagnostic_recording_guard,
                            _check_inference_artifacts_window,
                            _check_pub_oof_gap_guard,
                            _check_visualization_guard)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=VIZ_GUARD_WINDOW,
                        help="直近N実験を判定窓とする（default: %(default)s）")
    args = parser.parse_args()

    warnings = [w for w in (_check_visualization_guard(window=args.window),
                            _check_diagnostic_recording_guard(),
                            _check_inference_artifacts_window(),
                            _check_pub_oof_gap_guard()) if w]
    if not warnings:
        return 0

    text = "\n".join(warnings)
    if sys.stdin.isatty():
        # 手動起動（ターミナル）—— そのまま読ませる
        print(text)
        return 0

    # hook として起動された場合。**PostToolUse の素の stdout は AI にもユーザーにも
    # 届かない**（トランスクリプト表示にしか出ない）ため、警告を出したつもりで
    # 誰も読んでいなかった —— ガードの空洞化そのもの。JSON で明示的に両方へ渡す。
    print(json.dumps({
        "systemMessage": text,                       # ユーザーの画面に出す
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": text,               # AI の文脈に入れる
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
