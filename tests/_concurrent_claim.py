"""並行 start_run の採番を別プロセスで走らせる補助（テストから subprocess 起動）。

`multiprocessing` の spawn では stdin 起動のスクリプトを子が読めないため、実ファイルにする。

第 2 引数に秒数を渡すと、採番後その秒数だけ生き続ける。**実際の学習は数時間走り続ける**ので、
予約行の「掴んでいるプロセスが生きているか」を試すには、この生存が要る
（採番直後に終了すると、次のプロセスが正当に予約行を再利用してしまい、
競合の有無を判定できない）。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    log = Path(sys.argv[1])
    hold_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    os.environ["DS_SKIP_VIZ_CHECK"] = "1"
    from src import experiment as ex
    ex.LOG_CSV_PATH = log
    exp_id = ex._claim_experiment_id(f"pid{os.getpid()}", "lgb", "並行採番テスト")
    print(exp_id, flush=True)
    if hold_sec:
        time.sleep(hold_sec)
