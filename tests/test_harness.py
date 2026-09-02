"""ハーネス（規律を守らせる仕組み）が壊れていないかを検証する。

**なぜテストが要るか**: ガード 6 種・hook 6 種・エージェント 4 種の動作確認を、
これまで変更のたびに手作業で回していた（ある 1 日だけで 7 コミット分）。
手作業は締切前に最初に省略される。**ハーネスを守るのはハーネス自身のテスト**。

実行:
    uv run pytest tests/ -v
    uv run pytest tests/ -m "not slow"   # 重いものを除く
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HARNESS_SCRIPTS = [
    "doc_audit", "viz_guard", "state_audit", "session_brief", "session_audit",
    "session_snapshot", "job_status", "hook_status", "deadline_status", "statusline",
]


# ──────────────────────────────────────────────────────────
# 1. 提出ゲート —— 提出コマンドの検知が誤検知も見逃しもしないこと
# ──────────────────────────────────────────────────────────

SUB = "kaggle" + " competitions submit"      # 文字列として書くと自分がゲートに引っかかる


@pytest.mark.parametrize("command,expected,label", [
    (f"{SUB} -c x -f y.csv -m z", True, "素の提出コマンド"),
    (f"cd /tmp && {SUB} -c x -f y.csv", True, "&& の後ろ"),
    ("kaggle c submit -c x -f y.csv", True, "短縮形"),
    (f"KAGGLE_CONFIG_DIR=/tmp {SUB} -c x -f y.csv", True, "環境変数つき"),
    (f"grep -rn '{SUB}' CONVENTIONS.md", False, "grep での言及"),
    (f"echo '| hook | {SUB} を検知 |' >> doc.md", False, "ドキュメント編集"),
    ("kaggle competitions submissions -c x", False, "submissions は提出ではない"),
    ("ls -la", False, "無関係"),
])
def test_submit_gate_detection(command, expected, label):
    """提出コマンドの検知はコマンド位置で判定する（文字列としての言及を誤検知しない）。

    導入直後、この判定が無かったために**本ファイルを編集する Bash 自身がブロックされた**。
    """
    from scripts.harness.submit_gate import is_submit_command
    assert is_submit_command(command) is expected, label


def test_submit_gate_does_not_hang_without_stdin():
    """stdin が閉じられなくてもハングしない。

    PreToolUse hook は毎回の Bash の前に走るため、ここでブロックすると
    すべての Bash が timeout 秒止まる。
    """
    r = subprocess.run(["uv", "run", "python", "-m", "scripts.harness.submit_gate"],
                       cwd=ROOT, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0


# ──────────────────────────────────────────────────────────
# 2. log.csv の列追加マイグレーション —— 過去の記録を壊さないこと
# ──────────────────────────────────────────────────────────

def test_log_csv_migration_preserves_rows(tmp_path, monkeypatch):
    """列を足したとき、既存行の値が 1 つもずれないこと。

    ヘッダと行の列数が食い違うと**過去の実験記録が丸ごとずれる**ため、
    ここが壊れると失うものが大きい。
    """
    from src import experiment as ex

    log = tmp_path / "log.csv"
    old_cols = [c for c in ex.LOG_CSV_COLUMNS if c != "duration_sec"]
    sample = [
        {"experiment_id": "042", "model": "lgb", "oof_score": "0.91688",
         "submit_score": "0.91393", "notes": "カンマ, を含む値", "learning": "OOF↑LB↑"},
        {"experiment_id": "043", "model": "cb", "oof_score": "0.91701",
         "submit_score": "", "notes": "", "learning": ""},
    ]
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_cols)
        w.writeheader()
        for row in sample:
            w.writerow(row)

    monkeypatch.setattr(ex, "LOG_CSV_PATH", log)
    ex._ensure_log_csv()

    with open(log, newline="") as f:
        reader = csv.DictReader(f)
        header, rows = list(reader.fieldnames), list(reader)

    assert "duration_sec" in header
    assert len(rows) == len(sample)
    for got, want in zip(rows, sample):
        for key, value in want.items():
            assert got[key] == value, f"{key} がずれた"
    assert all(r["duration_sec"] == "" for r in rows)

    ex._ensure_log_csv()          # 冪等
    with open(log, newline="") as f:
        assert len(list(csv.DictReader(f))) == len(sample)


# ──────────────────────────────────────────────────────────
# 3. fold キャッシュ —— 中断した学習を再開できること
# ──────────────────────────────────────────────────────────

def test_foldcache_resumes_only_missing_folds(tmp_path):
    """中断後の再実行で、終わった fold は再学習しない。

    4 時間超まわした学習を fold 4/25 で打ち切り、その分が全損した事故への対策。
    """
    from src.utils.foldcache import FoldCache

    n_splits = 5
    rng = np.random.default_rng(0)
    val = [rng.random((10, 2)) for _ in range(n_splits)]
    test = [rng.random((7, 2)) for _ in range(n_splits)]
    cache = FoldCache(tag="unittest", seed=42, n_splits=n_splits, cache_dir=tmp_path)

    def run(stop_after=None):
        trained = []
        for fold in range(n_splits):
            if stop_after is not None and fold > stop_after:
                break
            if cache.load(fold) is not None:
                continue
            trained.append(fold)
            cache.save(fold, val[fold], test[fold])
        return trained

    assert run(stop_after=2) == [0, 1, 2]      # 中断
    assert run() == [3, 4]                     # 再開時は残りだけ
    assert cache.completed_folds() == list(range(n_splits))
    assert all(np.array_equal(cache.load(f)[0], val[f]) for f in range(n_splits))

    # tag が違えば再利用しない（条件が変わったら混ぜない = G-FAIR）
    other = FoldCache(tag="unittest_v2", seed=42, n_splits=n_splits, cache_dir=tmp_path)
    assert other.completed_folds() == []


# ──────────────────────────────────────────────────────────
# 4. サブエージェント定義 —— tools による強制が壊れていないこと
# ──────────────────────────────────────────────────────────

READONLY_AGENTS = {"fe-ideator", "experiment-reviewer"}
ALLOWED_TOOLS = {"Read", "Grep", "Glob", "Bash", "WebFetch", "WebSearch"}


def _agent_frontmatter(path: Path) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), re.S)
    assert m, f"{path.name}: frontmatter が無い"
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm


@pytest.mark.parametrize("path", sorted((ROOT / ".claude" / "agents").glob("*.md")),
                         ids=lambda p: p.stem)
def test_agent_definition(path):
    """エージェントは `tools` で「学習実行・commit をさせない」ことを保証している。

    指示ではなく道具で縛る設計なので、ここが緩むと保証そのものが消える。
    """
    fm = _agent_frontmatter(path)
    assert fm.get("name") == path.stem, "name がファイル名と一致しない"
    assert fm.get("description"), "description が無い（常時ロードされる要素）"
    tools = {t.strip() for t in fm.get("tools", "").split(",") if t.strip()}
    assert tools, "tools が無い（無制限になる）"
    assert not (tools - ALLOWED_TOOLS), f"想定外の tools: {sorted(tools - ALLOWED_TOOLS)}"
    if path.stem in READONLY_AGENTS:
        assert "Bash" not in tools, "読み取り専用のはずが Bash を持っている"


# ──────────────────────────────────────────────────────────
# 5. ハーネスのスクリプトが起動すること
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", HARNESS_SCRIPTS)
def test_harness_script_runs(name):
    """ハーネスの各スクリプトが例外を出さずに終了する。

    hook から呼ばれるものが落ちると、毎ターンエラーが出続ける。
    """
    r = subprocess.run(["uv", "run", "python", "-m", f"scripts.harness.{name}"],
                       cwd=ROOT, input="{}", capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-400:]


def test_statusline_is_fast_and_offline():
    """statusLine は毎ターン呼ばれるので、ネットワークを待たず即座に返ること。"""
    import time
    t0 = time.time()
    r = subprocess.run(["uv", "run", "python", "-m", "scripts.harness.statusline"],
                       cwd=ROOT, input="{}", capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert len(r.stdout.strip().splitlines()) == 1, "出力は 1 行であること"
    assert time.time() - t0 < 15, "遅すぎる（同期ネットワーク呼び出しを疑う）"


def test_hooks_pass_through_when_scripts_absent(tmp_path):
    """スクリプトが無いブランチでは、全 hook が黙って素通しすること。

    ハーネスは git 管理下なのでブランチを切り替えると消える。
    素通ししないと、コンペブランチで毎回の Bash がエラーになる。
    """
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    (tmp_path / "experiments").mkdir()
    for event, entries in settings["hooks"].items():
        for entry in entries:
            for hook in entry["hooks"]:
                r = subprocess.run(["bash", "-c", hook["command"]], cwd=tmp_path,
                                   capture_output=True, text=True, timeout=60)
                assert r.returncode == 0, f"{event} が非ゼロで終了した"


# ──────────────────────────────────────────────────────────
# 6. ドキュメント階層
# ──────────────────────────────────────────────────────────

def test_doc_audit_has_no_errors():
    """ドキュメント階層の検査が ERROR 0 であること。"""
    r = subprocess.run(["uv", "run", "python", "-m", "scripts.harness.doc_audit"],
                       cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert "ERROR 0" in r.stdout, r.stdout[-1200:]


def test_all_modules_import():
    """src/ と scripts/ の全モジュールが import できること。"""
    failed = []
    for path in sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))):
        module = str(path.relative_to(ROOT))[:-3].replace("/", ".")
        r = subprocess.run(["uv", "run", "python", "-c", f"import {module}"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            failed.append(module)
    assert not failed, f"import できないモジュール: {failed}"
