"""並行 start_run の採番を別プロセスで走らせる補助（テストから subprocess 起動）。

`multiprocessing` の spawn では stdin 起動のスクリプトを子が読めないため、実ファイルにする。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    log = Path(sys.argv[1])
    os.environ["DS_SKIP_VIZ_CHECK"] = "1"
    from src import experiment as ex
    ex.LOG_CSV_PATH = log
    print(ex._claim_experiment_id(f"pid{os.getpid()}", "lgb", "並行採番テスト"))
