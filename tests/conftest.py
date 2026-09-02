"""テストの最終実行時刻を記録する。

`session_audit`（Stop hook）が「ハーネスを変更したのに pytest を走らせていない」を
検知するために使う。`.pytest_cache` の mtime は**内容が変わったときしか更新されない**ため、
実行のたびに確実に更新されるマーカーをこちらで書く。
"""

from pathlib import Path

import pytest

MARKER = Path(__file__).resolve().parents[1] / "experiments" / ".pytest_last_run"


@pytest.fixture(scope="session", autouse=True)
def _record_test_run():
    yield
    try:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.touch()
    except Exception:
        pass          # 記録の失敗でテストを落とさない
