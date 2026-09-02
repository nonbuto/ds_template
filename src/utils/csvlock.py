"""log.csv の read-modify-write を、並行実行しても壊れない形に揃える。

**なぜこのモジュールがあるか**: `experiment.py` は log.csv を「全部読む → 加工する →
書き戻す」形で更新していたが、ロックが無かった。`CLAUDE.md` は
**「バックグラウンド並行実行時も例外なし」**として複数実験の同時実行を前提にしているのに、
その前提で壊れる作りだった。

実測（8 プロセス同時、`multiprocessing`）:

    ID: ['000', '000', '000', '000', '000', '000', '000', '000']
    重複 ID: 7 件

全員が「まだ 0 行」を読んでから追記するため、**8 実験が同じ experiment_id を名乗る**。
さらに全書き換え（列の移行・予約行のマージ）中に別プロセスが追記すると、
その追記は書き戻しで**丸ごと消える**。log.csv は実験の唯一の台帳なので、
消えた行は git 履歴にも残らない。

使い方:

    with locked_csv(LOG_CSV_PATH) as rows:      # 読み → 排他 → 書き戻しまで一括
        rows.append(new_row)

    with log_lock(LOG_CSV_PATH):                # 読みだけを守りたいとき
        ...
"""

from __future__ import annotations

import csv
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

LOCK_TIMEOUT_SEC = 30.0


@contextmanager
def log_lock(path: Path, timeout: float = LOCK_TIMEOUT_SEC):
    """`path` への排他ロックを取る。

    ロックはサイドカー（`<path>.lock`）に取る —— log.csv 自体を `os.replace` で
    差し替えるため、本体に取った fd は差し替えの瞬間に別の inode を指してしまう。

    **タイムアウトしても例外にせず、ロック無しで続行する。**
    学習が終わった直後の記録でプロセスを落とすと、数時間の計算結果が log に残らない。
    競合の可能性を警告して進める方が、実害が小さい。
    """
    import fcntl
    import time

    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    print(f"⚠️ {path.name} のロック取得が {timeout:.0f} 秒で失敗しました。"
                          "ロック無しで続行します（他プロセスと競合する可能性）")
                    break
                time.sleep(0.05)
        yield acquired
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def write_rows_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """同じディレクトリの一時ファイルに書いてから `os.replace` で差し替える。

    直接 `open(path, "w")` すると、書いている途中にプロセスが死んだ場合に
    **台帳が途中まで書かれた状態で残る**（ヘッダだけ、など）。`os.replace` は
    同一ファイルシステム上で原子的なので、読み手は必ず旧版か新版のどちらかを見る。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def locked_csv(path: Path, fieldnames: list[str], timeout: float = LOCK_TIMEOUT_SEC):
    """読み込み → 編集 → 原子的書き戻しを、ロックを保持したまま 1 区間で行う。

    yield されたリストをその場で編集する（append / 要素の差し替え）。
    区間を抜けると、そのリストがファイルの新しい内容になる。

        with locked_csv(LOG_CSV_PATH, LOG_CSV_COLUMNS) as rows:
            rows.append(row)                    # ← 他プロセスは待たされる

    **読んだ内容が書き戻しまで有効であること**がこの型の要点。
    「読む」と「書く」を別々にロックしても、その間に他プロセスが割り込めば同じ事故になる。
    """
    with log_lock(path, timeout=timeout):
        rows: list[dict] = []
        existing: list[str] = []
        if path.exists():
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                existing = list(reader.fieldnames or [])
                rows = list(reader)
        # 手で足された未知の列は末尾に残す（勝手に消さない）
        out_fields = list(fieldnames) + [c for c in existing if c not in fieldnames]
        yield rows
        write_rows_atomic(path, out_fields, rows)
