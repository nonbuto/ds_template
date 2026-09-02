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
    # 検知すべき —— 見逃しは**無確認の提出**に直結する。
    # 以前は `shlex.split` で空白分割していたため、`;`・改行・`uv run` 前置・絶対パスが
    # すべて素通りしていた（8 パターン中 6 件）。テストが `&&` の形しか見ておらず
    # 誤った安心を与えていたので、ここを厚くする。
    (f"{SUB} -c x -f y.csv -m z", True, "素の提出コマンド"),
    (f"cd /tmp && {SUB} -c x -f y.csv", True, "&& の後ろ"),
    (f"uv run {SUB} -c x -f y.csv", True, "uv run 前置"),
    (f"echo start; {SUB} -c x -f y.csv", True, "; 区切り（空白なし）"),
    (f"echo start\n{SUB} -c x -f y.csv", True, "改行区切り（複数行コマンド）"),
    (f"nohup {SUB} -c x -f y.csv", True, "nohup 前置"),
    (f"time {SUB} -c x -f y.csv", True, "time 前置"),
    (f"/opt/homebrew/bin/{SUB} -c x -f y.csv", True, "絶対パス"),
    (f"KAGGLE_CONFIG_DIR=/tmp {SUB} -c x -f y.csv", True, "環境変数つき"),
    ("kaggle c submit -c x -f y.csv", True, "短縮形 c submit"),
    (f"for i in 1; do {SUB} -c x -f y.csv; done", True, "for ループ内"),
    (f"ls | head && {SUB} -c x -f y.csv", True, "パイプの後"),
    # 無視すべき —— 誤検知はドキュメント編集を妨げる
    (f"grep -rn '{SUB}' CONVENTIONS.md", False, "grep での言及"),
    (f"echo '| hook | {SUB} を検知 |' >> doc.md", False, "ドキュメント編集"),
    ("kaggle competitions submissions -c x", False, "submissions は提出ではない"),
    ('python3 -c "print(\'kaggle competitions submit\')"', False, "python 内の文字列"),
    ("ls -la", False, "無関係"),
])
def test_submit_gate_detection(command, expected, label):
    """提出コマンドの検知。**見逃しと誤検知は非対称**（見逃し＝無確認の提出）。

    シェルの区切り（`;` `&&` `|` 改行 `(` `)`）をクォートを尊重しつつ分離し、
    ラッパー（`uv run` `nohup` `time`）と環境変数代入を透過して先頭を見る。
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


# ──────────────────────────────────────────────────────────
# 7. 評価指標と CV —— 設定から 1 箇所で決まること
# ──────────────────────────────────────────────────────────

def test_metric_and_cv_come_from_config():
    """指標と分割器が `src/config.py` の設定から決まること。

    以前は train.py と optimize_hp.py が独立に指標を呼んでおり、
    片方だけ変えると **HP 最適化が別の指標を最適化する**（静かに壊れる）状態だった。
    """
    from src import metrics as M
    assert callable(M.get_metric())
    assert M.get_cv() is not None
    assert isinstance(M.describe(), str)


@pytest.mark.parametrize("name,y,pred", [
    ("auc", np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])),
    ("auc", np.array([0, 1, 2, 0]), np.eye(3)[[0, 1, 2, 0]] * 0.7 + 0.1),   # 多クラス
    ("balanced_accuracy", np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1])),
    ("f1", np.array([0, 1, 2, 0]), np.array([0, 1, 2, 1])),                  # 多クラス
    ("rmse", np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 2.8])),
])
def test_metric_handles_shapes(name, y, pred):
    """二値・多クラス・回帰のいずれでも指標が計算できること。

    多クラス × AUC は `multi_class` 引数が要る。呼び出し側に任せると
    train と optimize_hp で扱いがずれるので、指標モジュール側で吸収する。
    """
    from src.metrics import get_metric
    assert isinstance(get_metric(name)(y, pred), float)


def test_unknown_metric_fails_loudly():
    """未対応の指標は黙って動かず、選択肢を示して失敗すること。"""
    from src.metrics import get_metric
    with pytest.raises(ValueError, match="未対応"):
        get_metric("nonexistent_metric")


def test_metric_is_defined_in_one_place():
    """train.py と optimize_hp.py が指標を直接呼んでいないこと（定義元は src/metrics.py だけ）。"""
    for name in ["train.py", "optimize_hp.py", "blend.py"]:
        code = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        body = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
        for bad in ["balanced_accuracy_score(", "roc_auc_score(", "log_loss(",
                    "mean_squared_error(", "StratifiedKFold("]:
            assert bad not in body, f"{name} が {bad} を直接呼んでいる（定義元は src/metrics.py）"


def test_oof_analysis_handles_all_problem_types(capsys):
    """誤差分析が二値・多クラス・回帰のいずれでも例外を出さないこと。

    以前は roc_auc と閾値 0.5 を直書きした**二値専用かつ呼び出し元ゼロ**の死蔵メソッドだった。
    """
    import importlib
    import src.config as cfg
    from src.experiment import ExperimentTracker

    rng = np.random.default_rng(0)
    cases = [
        ("binary_classification", "auc", rng.random(40), rng.integers(0, 2, 40)),
        ("multiclass", "balanced_accuracy", rng.random((40, 3)), rng.integers(0, 3, 40)),
        ("regression", "rmse", rng.random(40) * 10, rng.random(40) * 10),
    ]
    original = (cfg.PROBLEM_TYPE, cfg.EVAL_METRIC)
    try:
        for ptype, metric, oof, y in cases:
            cfg.PROBLEM_TYPE, cfg.EVAL_METRIC = ptype, metric
            import src.metrics as M
            importlib.reload(M)
            ExperimentTracker(experiment_name="t").save_oof_analysis(oof, y)
            assert "OOF 誤差分析" in capsys.readouterr().out, ptype
    finally:
        cfg.PROBLEM_TYPE, cfg.EVAL_METRIC = original
        import src.metrics as M
        importlib.reload(M)


def test_experiment_template_follows_conventions():
    """実験の雛形が作法（tracker / metrics / foldcache / finalize）を満たしていること。"""
    tpl = ROOT / "experiments" / "runs" / "_TEMPLATE_exp000_s0_example.py"
    assert tpl.exists(), "実験の雛形が無い"
    body = tpl.read_text(encoding="utf-8")
    for required in ["ExperimentTracker", "log_fold_scores", "end_run(",
                     "get_metric", "get_cv", "FoldCache", "save_run_outputs"]:
        assert required in body, f"雛形に {required} が無い"


def test_doc_audit_guards_are_not_hollow():
    """ガード自身が「0 件しか検査していない」状態になっていないこと（C15）。

    「問題 0 件」と「検査対象 0 件」は表示上どちらも ✅ になる。
    README の自己申告値・指針の ID 定義・文書中のコマンドで、実際に 3 度見逃した。
    """
    import importlib
    from scripts.harness import doc_audit as D
    importlib.reload(D)
    results: list = []
    D.check(results)
    for key in ["C2", "C3", "C6", "C11", "C13", "C14"]:
        assert D.CHECKED.get(key, 0) > 0, f"{key} の検査対象が 0 件（ガードが空洞）"
    assert len(results) == D.TOTAL_CHECKS, "TOTAL_CHECKS が実際の検査数と不一致"


def test_blend_rejects_multiclass_clearly():
    """blend は 1 次元 OOF 前提。多クラスでは**沈黙せず明確に**失敗すること。"""
    body = (ROOT / "scripts" / "blend.py").read_text(encoding="utf-8")
    assert "二値分類・回帰" in body, "多クラスを検知して止める処理が無い"
    assert "LabelEncoder" in body, "ラベルエンコードを通していない（文字列ラベルで落ちる）"


def test_config_values_are_valid_for_metrics():
    """`src/config.py` の設定値が `src/metrics.py` で実際に使えること。

    `/ds-kickoff` が Kaggle の表記（AreaUnderROCCurve 等）をそのまま書くと、
    学習の開始時に ValueError で止まる。設定と実装の乖離をここで捕まえる。
    """
    from src import config as cfg
    from src.metrics import _METRICS, get_cv, get_metric

    assert cfg.EVAL_METRIC.lower() in _METRICS, (
        f"EVAL_METRIC='{cfg.EVAL_METRIC}' は未対応。有効値: {sorted(_METRICS)}")
    assert callable(get_metric()), "指標関数を作れない"
    assert get_cv() is not None, f"CV_STRATEGY='{cfg.CV_STRATEGY}' から分割器を作れない"
    assert cfg.PROBLEM_TYPE in {"regression", "binary_classification", "multiclass"}, (
        f"PROBLEM_TYPE='{cfg.PROBLEM_TYPE}' が想定外")


def test_skills_reference_existing_files():
    """スキルが言及するファイルパスが実在すること。

    state/ や docs/ へ移設したときに、スキル側の記述が取り残されると
    「README を頼りに探した人が見つけられない」状態になる。
    """
    import re
    missing = []
    for skill in sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md")):
        text = skill.read_text(encoding="utf-8")
        for m in re.finditer(r"`((?:scripts|src|state|docs|tests)/[\w/.]+|[A-Z_]+\.md)`", text):
            if not (ROOT / m.group(1)).exists():
                missing.append(f"{skill.parent.name}: {m.group(1)}")
    assert not missing, f"スキルが存在しないファイルを参照している: {sorted(set(missing))}"


# ──────────────────────────────────────────────────────────
# 8. 実行時ガードが「実際に発火する」こと
# ──────────────────────────────────────────────────────────
# これまで**ガードを `return None` に潰しても全件合格**していた。
# 「壊れを検知するテスト」が無く、リファクタや締切前の緩和で規律機構が
# 静かに死んでも全部グリーンのままだった（L-06 の形骸化と同じ構造）。

def _fake_log(tmp_path, n_rows: int, **extra) -> "Path":
    """指定件数の実験行を持つ log.csv を作る。"""
    import csv as _csv
    from src import experiment as ex
    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
        w.writeheader()
        for i in range(n_rows):
            row = {"experiment_id": f"{i:03d}", "timestamp": "2020-01-01 00:00:00",
                   "model": "lgb", "oof_score": "0.5"}
            row.update(extra)
            w.writerow(row)
    return log


def test_visualization_guard_fires(tmp_path, monkeypatch):
    """直近 N 実験で可視化ゼロなら発火し、新しい .png があれば黙ること。"""
    from src import experiment as ex
    monkeypatch.setattr(ex, "LOG_CSV_PATH", _fake_log(tmp_path, ex.VIZ_GUARD_WINDOW))
    plots = tmp_path / "plots"
    plots.mkdir()
    monkeypatch.setattr(ex, "PLOTS_DIR", plots)
    assert ex._check_visualization_guard() is not None, "可視化ゼロなのに発火しない"

    import time
    png = plots / "x.png"
    png.write_bytes(b"x")
    import os
    os.utime(png, (time.time(), time.time()))
    assert ex._check_visualization_guard() is None, "新しい .png があるのに発火する"


def test_diagnostic_recording_guard_fires(tmp_path, monkeypatch):
    """診断列が空の実験が続けば発火し、埋まっていれば黙ること。"""
    from src import experiment as ex
    monkeypatch.setattr(ex, "LOG_CSV_PATH", _fake_log(tmp_path, ex.DIAG_GUARD_WINDOW))
    assert ex._check_diagnostic_recording_guard() is not None, "診断列が空なのに発火しない"

    monkeypatch.setattr(ex, "LOG_CSV_PATH",
                        _fake_log(tmp_path / "b", ex.DIAG_GUARD_WINDOW,
                                  cv_train_mean="0.9", cv_val_std="0.01")
                        if (tmp_path / "b").mkdir() is None else None)
    assert ex._check_diagnostic_recording_guard() is None, "診断列が埋まっているのに発火する"


def test_inference_artifact_guard_fires(tmp_path, monkeypatch):
    """OOF はあるのに test 予測が無い実験を検知すること。"""
    from src import experiment as ex
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    monkeypatch.setattr(ex, "OOF_DIR", oof_dir)
    monkeypatch.setattr(ex, "LOG_CSV_PATH", _fake_log(tmp_path, 1))

    (oof_dir / "oof_000_lgb.npy").write_bytes(b"x")
    assert ex._check_inference_artifact_guard("000") is not None, "test が無いのに発火しない"
    assert ex._check_inference_artifacts_window() is not None, "窓版が発火しない"

    (oof_dir / "test_000_lgb.npy").write_bytes(b"x")
    assert ex._check_inference_artifact_guard("000") is None, "test があるのに発火する"
    assert ex._check_inference_artifacts_window() is None, "窓版が発火し続ける"


def test_pub_oof_gap_guard_fires(tmp_path, monkeypatch):
    """Public が OOF より基準線 +0.0005 を超えて浮いたら発火すること。"""
    import csv as _csv
    from src import experiment as ex

    def write(gaps):
        log = tmp_path / f"log_{len(gaps)}_{gaps[-1]}.csv"
        with open(log, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
            w.writeheader()
            for i, g in enumerate(gaps):
                w.writerow({"experiment_id": f"{i:03d}", "oof_score": "0.90000",
                            "submit_score": f"{0.90000 + g:.5f}"})
        return log

    flat = [0.0] * 10
    monkeypatch.setattr(ex, "LOG_CSV_PATH", write(flat))
    assert ex._check_pub_oof_gap_guard() is None, "gap が一定なのに発火する"

    spiked = [0.0] * 10 + [0.01] * 3        # 直近が基準線から大きく浮く
    monkeypatch.setattr(ex, "LOG_CSV_PATH", write(spiked))
    assert ex._check_pub_oof_gap_guard() is not None, "Public が浮いているのに発火しない"


# ──────────────────────────────────────────────────────────
# 9. log.csv の並行更新（実験の台帳が壊れないこと）
# ──────────────────────────────────────────────────────────

def test_concurrent_experiment_ids_are_unique(tmp_path):
    """同時に走る実験が別々の experiment_id を取ること。

    修正前は採番（読むだけ）と記録（end_run の追記）が離れており、
    8 プロセス同時で**全員が `000` を名乗った**（重複 7 件）。
    `CLAUDE.md` は「バックグラウンド並行実行時も例外なし」として同時実行を前提にしている。
    """
    import subprocess
    import sys

    log = tmp_path / "log.csv"
    helper = Path(__file__).parent / "_concurrent_claim.py"
    procs = [subprocess.Popen([sys.executable, str(helper), str(log)],
                              stdout=subprocess.PIPE, text=True) for _ in range(8)]
    ids = [p.communicate()[0].strip().splitlines()[-1] for p in procs]

    assert len(set(ids)) == 8, f"experiment_id が重複した: {sorted(ids)}"
    import csv as _csv
    with open(log, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 8, f"確保した行が失われた（{len(rows)} 行）"


def test_atomic_write_survives_reader(tmp_path):
    """書き戻しの途中でも、読み手はヘッダだけの壊れたファイルを見ないこと。"""
    import csv as _csv
    from src.utils.csvlock import write_rows_atomic

    path = tmp_path / "log.csv"
    cols = ["experiment_id", "oof_score"]
    write_rows_atomic(path, cols, [{"experiment_id": "001", "oof_score": "0.9"}])
    write_rows_atomic(path, cols, [{"experiment_id": "001", "oof_score": "0.9"},
                                   {"experiment_id": "002", "oof_score": "0.8"}])
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert [r["experiment_id"] for r in rows] == ["001", "002"]
    assert not list(path.parent.glob(".*.tmp")), "一時ファイルが残っている"


# ──────────────────────────────────────────────────────────
# 10. 指標の向き・提出形式・クラス数（静かに間違う系）
# ──────────────────────────────────────────────────────────
# いずれも例外を出さず、それらしい数字を返しながら結論だけが逆になる型。
# テストが無いと「動いている」ことと「正しい」ことが区別できない。

def test_shape_for_metric_matches_metric_kind(monkeypatch):
    """整形の写経（6 箇所にあった三項演算子）が 1 箇所に集約され、指標種別に従うこと。"""
    import numpy as np
    from src import metrics as m

    proba2 = np.array([[0.3, 0.7], [0.6, 0.4]])
    proba3 = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])

    assert m.shape_for_metric(proba2, "auc").tolist() == [0.7, 0.4], "二値は陽性列を渡す"
    assert m.shape_for_metric(proba3, "auc").shape == (2, 3), "多クラス確率はそのまま"
    assert m.shape_for_metric(proba3, "accuracy").tolist() == [2, 0], "ラベル指標は argmax"
    assert m.shape_for_metric(np.array([1.5, 2.5]), "rmse").tolist() == [1.5, 2.5], "回帰は素通し"


@pytest.mark.parametrize("metric,expected", [
    ("auc", True), ("logloss", False), ("rmse", False), ("r2", True),
    ("accuracy", True), ("balanced_accuracy", True), ("f1", True), ("mae", False),
])
def test_metric_direction_is_declared(metric, expected):
    """全指標に改善の向きが定義されていること（feature_study の ΔOOF 判定の前提）。"""
    from src.metrics import greater_is_better
    assert greater_is_better(metric) is expected


def test_feature_study_delta_orients_to_improvement():
    """ΔOOF が**改善方向**に揃うこと。

    修正前は `new - base` をそのまま判定に使っていたため、RMSE・logloss・MAE の
    コンペでは**良い特徴量を棄却し、悪い特徴量を採用する**判定が出ていた。
    feature_study は FE 判断の中核ツールなので、ここが逆だと全 FE の採否が反転する。
    """
    src = Path("scripts/feature_study.py").read_text(encoding="utf-8")
    assert "greater_is_better()" in src, "指標の向きを見ていない"
    assert "delta = raw_delta if greater_is_better() else -raw_delta" in src


def test_submission_format_follows_metric(monkeypatch):
    """提出形式が `EVAL_METRIC` から決まること（AUC でハードラベルを出さない）。"""
    from src.utils import finalize

    monkeypatch.setattr(finalize, "is_regression", lambda: False)
    monkeypatch.setattr(finalize, "needs_proba", lambda: True)
    assert finalize._resolve_submit_mode("auto") == "proba", "AUC/logloss は確率で提出する"

    monkeypatch.setattr(finalize, "needs_proba", lambda: False)
    assert finalize._resolve_submit_mode("auto") == "label", "accuracy 系はラベルで提出する"

    monkeypatch.setattr(finalize, "is_regression", lambda: True)
    assert finalize._resolve_submit_mode("auto") == "value", "回帰は予測値で提出する"


def test_default_config_is_internally_consistent():
    """clone 直後の設定でクラス数と目的関数が矛盾しないこと。

    修正前は `train.py` が `N_CLASSES = 3` を直書きし、config の既定
    （binary_classification / auc）と食い違っていた。Stage 1 の最小ベースラインが
    **clone 直後には動かない**状態で、テンプレートの入口が壊れていた。
    """
    from scripts.train import DEFAULT_PARAMS, build_params
    from src.config import PROBLEM_TYPE
    from src.metrics import n_classes

    if PROBLEM_TYPE == "binary_classification":
        assert DEFAULT_PARAMS["lgb"]["objective"] == "binary"
        assert "num_class" not in DEFAULT_PARAMS["lgb"], "二値に num_class は渡さない"
        assert n_classes() == 2
    assert build_params("lgb", 3)["num_class"] == 3, "多クラスではクラス数が入る"
    assert build_params("xgb", 2)["objective"] == "binary:logistic"


def test_train_importance_is_gain_based():
    """importance が gain ベースで取り出されること（文書・軸ラベルは "gain" と書いている）。

    `feature_importances_` は LightGBM では既定が split（分岐回数）で、gain とは別物。
    `G-DIAG` の第3診断軸をこの値で判断するので、定義がずれていると解釈が狂う。
    """
    src = Path("scripts/train.py").read_text(encoding="utf-8")
    assert 'importance_type="gain"' in src
    assert "feature_importances_" not in src.split("def extract_importance")[0], \
        "extract_importance を経由せず split を拾っている箇所が残っている"


def test_resume_does_not_fake_train_scores():
    """キャッシュ再利用の fold が train スコアを詐称しないこと。

    修正前は `tr_score = val_score` を入れていた。これは `cv_train_mean` に嘘を書くのと同じで、
    `--resume` のたびに `G-DIAG` の train−val 乖離が「乖離ゼロ」に化けていた。
    """
    src = Path("scripts/train.py").read_text(encoding="utf-8")
    assert "tr_score = val_score" not in src
    assert 'tr_score = float("nan")' in src


def test_experiment_diag_columns_blank_when_unmeasured():
    """測れなかった診断値が "nan" ではなく空欄で記録されること。

    "nan" と書くと診断記録ガードが「記入済み」と数え、記入率が実態より高く出る
    （ガードの空洞化そのもの）。
    """
    from src.experiment import _fmt
    assert _fmt(float("nan")) == ""
    assert _fmt(0.5) == "0.50000"


def test_timeseries_split_leaves_rows_unpredicted():
    """TimeSeriesSplit が先頭行を一度も検証しないこと（未予測行が存在する前提の確認）。

    この性質があるため、OOF 配列をゼロ初期化したまま全行で評価すると
    「全部クラス0と予測した」ことになり、**スコアが実力と無関係に動く**。
    train.py / optimize_hp.py は `covered` マスクで評価対象を絞る。
    """
    import numpy as np
    from src.metrics import get_cv

    cv = get_cv(strategy="TimeSeriesSplit", n_splits=5)
    X = np.arange(100).reshape(-1, 1)
    covered = np.zeros(100, dtype=bool)
    for _, val_idx in cv.split(X):
        covered[val_idx] = True
    assert not covered.all(), "この前提が崩れたらマスク処理を見直す"
    assert covered.sum() < 100
    # 未予測行があるとき、ゼロのまま評価するとどれだけ結論が動くか
    assert (~covered).sum() >= 10, f"未予測は {(~covered).sum()} 行"


def test_unpredicted_rows_excluded_from_scoring():
    """学習スクリプトが未予測行を評価から外していること。"""
    for path in ("scripts/train.py", "scripts/optimize_hp.py"):
        src = Path(path).read_text(encoding="utf-8")
        assert "covered" in src, f"{path} に未予測行のマスクが無い"
        assert "covered[val_idx] = True" in src, f"{path} でマスクを立てていない"


def test_hp_search_does_not_inject_class_weight():
    """HP 探索が特定モデルにだけ class_weight を混ぜないこと。

    修正前は `MULTICLASS_OVERRIDES` が lgb にだけ `class_weight="balanced"` を付けており、
    lgb は重み付き・xgb/cb は重みなしという条件の揃わない比較になっていた
    ——**テンプレート自身が `G-FAIR` 違反を作っていた**。
    """
    # 説明のためにコメントで言及するのは可。**実行されるコード**に無いことを見る
    import ast
    tree = ast.parse(Path("scripts/optimize_hp.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "MULTICLASS_OVERRIDES" not in names, "多クラス前提の上書き定数が残っている"
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert "balanced" not in consts, "探索空間に class_weight='balanced' が混ざっている"


def test_beta_calibration_respects_metric_direction():
    """beta 較正が指標の向きに従い、確率指標では argmax を挟まないこと。"""
    import numpy as np
    import pandas as pd
    from scripts import optimize_hp as o

    # 確率指標（AUC）では較正を使わない = 確率のまま評価される
    assert o._use_beta_calibration(2) is False
    score, beta = o._score_with_beta(np.array([[0.2, 0.8], [0.7, 0.3]]),
                                     pd.Series([1, 0]), np.array([0.5, 0.5]), 2)
    assert beta == 1.0 and score == 1.0

    src = Path("scripts/optimize_hp.py").read_text(encoding="utf-8")
    assert "pick = max if greater_is_better() else min" in src, "最良 beta の選び方が向きに従っていない"
    assert 'trial.set_user_attr("beta", beta)' in src, "最良 beta を保存していない"


def test_previous_score_excludes_own_row(tmp_path, monkeypatch):
    """ΔOOF の比較相手が「直前の実験」であって自分自身でないこと。

    修正前はこの関数を自分の行を書いた**後**に呼んでいたため、
    ΔOOF が常に ±0.00000 と表示され、`G-DIAG` の「std 未満なら測れていない」判定が
    毎回「判別不能」に張り付いていた。
    """
    import csv as _csv
    from src import experiment as ex

    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
        w.writeheader()
        w.writerow({"experiment_id": "001", "oof_score": "0.90000"})
        w.writerow({"experiment_id": "002", "oof_score": "0.95000"})
    monkeypatch.setattr(ex, "LOG_CSV_PATH", log)

    assert ex._previous_experiment_scores() == 0.95
    assert ex._previous_experiment_scores(exclude_id="002") == 0.90, "自分の行を除けていない"


@pytest.mark.parametrize("raw,expected", [
    ("0.95092(anchor)", 0.95092),      # 注釈つきの表記
    ("12.3456", 12.3456),              # RMSE のスケール（旧実装は "2.34" と誤読）
    ("-0.1234", -0.1234),              # R² が負（旧実装は符号を落とした）
    ("0.5", 0.5),
])
def test_score_parsing_handles_all_metric_scales(raw, expected, tmp_path, monkeypatch):
    """スコアの読み取りが 0/1 始まりの値に限定されないこと。"""
    import csv as _csv
    from src import experiment as ex

    log = tmp_path / f"log_{raw}.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
        w.writeheader()
        w.writerow({"experiment_id": "001", "oof_score": raw})
    monkeypatch.setattr(ex, "LOG_CSV_PATH", log)
    assert ex._previous_experiment_scores() == expected


def test_group_cv_is_usable():
    """GroupKFold 系が実際に分割できること（`groups` が渡る経路があること）。

    修正前は `get_cv()` が返せても呼び出し側が `groups` を渡しておらず、
    設定として選べるのに**必ず ValueError で落ちる**状態だった。
    """
    import pandas as pd
    from src import metrics as m

    monkey = pytest.MonkeyPatch()
    monkey.setattr(m, "GROUP_COL", "g")
    try:
        df = pd.DataFrame({"a": range(20), "g": [i // 4 for i in range(20)]})
        cv = m.get_cv(strategy="GroupKFold", n_splits=5)
        groups = m.get_groups(df, "GroupKFold")
        folds = list(cv.split(df[["a"]], [0, 1] * 10, groups=groups))
        assert len(folds) == 5
        for _, va in folds:
            assert len(set(df["g"].iloc[va])) == 1, "同じグループが複数 fold に分かれている"
    finally:
        monkey.undo()

    for path in ("scripts/train.py", "scripts/optimize_hp.py"):
        src = Path(path).read_text(encoding="utf-8")
        assert "groups=groups" in src, f"{path} が groups を渡していない"


def test_blend_accepts_binary_probability_matrix():
    """blend が train.py の出力（二値 (n,2)）を受け取れること。

    修正前は「1 次元でない」と撥ねており、**テンプレートが出力したファイルを
    テンプレートのブレンドが受け取れない**状態だった。
    """
    src = Path("scripts/blend.py").read_text(encoding="utf-8")
    assert "_as_1d" in src
    assert "arr.shape[1] == 2 and n_classes == 2" in src


def test_greedy_ensemble_without_tests(capsys):
    """`--tests` を渡さなくても greedy が最後で落ちないこと。

    修正前は探索が全部終わった最後の 1 行で `tests[n]` が KeyError になり、
    **計算結果ごと失われていた**。
    """
    import numpy as np
    from src.utils.ensemble import greedy_ensemble

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    oofs = {f"m{i}": np.clip(y * 0.6 + rng.normal(0.2, 0.3, 200), 0, 1) for i in range(3)}
    selected, ens_oof, ens_test, score = greedy_ensemble(
        oofs=oofs, tests={}, y=y, metric_fn=lambda a, b: float(((b > 0.5) == a).mean()))
    assert selected and ens_test is None
    assert ens_oof.shape == (200,)


def test_agent_removal_is_detected():
    """エージェントを 1 つ消したら doc_audit が気づくこと。

    ファイル側だけを検査していると、消しても「残った分は全部正しい」で ✅ のまま通る。
    テストも glob の結果を回すだけなので、**入力が消えれば検査項目ごと消える**。
    参照する側（文書）から見て初めて「消えたこと」が検知できる。
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    # git clone ではなく**作業ツリーを複製する** —— clone は HEAD を取るので、
    # 未コミットの doc_audit を検査できず「直したのにテストが落ちる」ことになる。
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "repo"
        work.mkdir()
        for rel in ("scripts", "src", ".claude", "CLAUDE.md", "GUIDELINES.md",
                    "CONVENTIONS.md", "PLAYBOOK.md", "README.md"):
            srcp = ROOT / rel
            if srcp.is_dir():
                shutil.copytree(srcp, work / rel,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif srcp.exists():
                shutil.copy2(srcp, work / rel)

        (work / ".claude" / "agents" / "fe-ideator.md").unlink()
        r = subprocess.run([sys.executable, "-m", "scripts.harness.doc_audit"],
                           cwd=work, capture_output=True, text=True)
        assert "fe-ideator" in r.stdout, "エージェントを消しても doc_audit が黙っている"


def test_submit_gate_asks_when_internal_check_fails(monkeypatch):
    """提出前チェックが内部エラーでも、素通しせず確認を求めること。

    ゲートが守るのは「取り消せない・回数制限つき・外部に見える」唯一の操作なので、
    壊れたときは通す側ではなく**確認を求める側**に倒す。
    """
    import io
    import json as _json
    import sys as _sys
    from scripts.harness import submit_gate as g

    monkeypatch.setattr(g, "build_brief",
                        lambda cmd: (_ for _ in ()).throw(RuntimeError("kaggle CLI 異常")))
    payload = _json.dumps({"tool_name": "Bash",
                           "tool_input": {"command": f"{SUB} -c x -f y.csv -m t"}})
    stdin = io.StringIO(payload)
    stdin.isatty = lambda: False
    monkeypatch.setattr(_sys, "stdin", stdin)
    buf = io.StringIO()
    monkeypatch.setattr(_sys, "stdout", buf)
    g.main()
    out = _json.loads(buf.getvalue())["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask"
    assert "内部エラー" in out["permissionDecisionReason"]


def test_submission_limit_has_single_definition():
    """提出上限の定義元が 1 つであること（表示ごとに違う値になるのを防ぐ）。"""
    from src.config import DAILY_SUBMISSION_LIMIT
    from scripts.harness import deadline_status, submit_gate

    assert submit_gate.DAILY_LIMIT == deadline_status.DAILY_LIMIT == DAILY_SUBMISSION_LIMIT
    src = Path("scripts/harness/deadline_status.py").read_text(encoding="utf-8")
    assert "DAILY_LIMIT = 10" not in src, "ハーネス側に上限が直書きされている"


def test_viz_guard_reaches_the_conversation():
    """ガードの警告が hook 経由で AI とユーザーに届く形で出ること。

    PostToolUse の素の stdout はどちらにも届かない（トランスクリプトにしか出ない）。
    警告を出したつもりで**誰も読んでいなかった**状態がガードの空洞化そのもの。
    """
    src = Path("scripts/harness/viz_guard.py").read_text(encoding="utf-8")
    assert "systemMessage" in src, "ユーザーに届く経路が無い"
    assert "additionalContext" in src, "AI の文脈に入る経路が無い"


# ──────────────────────────────────────────────────────────
# 11. 学習・アンサンブルの設計上のリーク
# ──────────────────────────────────────────────────────────

def test_early_stopping_does_not_watch_the_oof_fold():
    """early stopping の監視先が既定で train fold の内側であること。

    検証 fold（= OOF になる行）で木の本数を決めると、OOF はその選択を経た予測になる。
    合成データでの実測差は **AUC +0.00467**（ノイズ床 ±0.0002 の 20 倍以上）で、
    `G-OOF`「OOF を足切りに使う」という前提が静かに崩れる大きさ。
    """
    from src.config import EARLY_STOPPING_ON

    assert EARLY_STOPPING_ON == "inner", "既定が検証 fold を見る設定になっている"
    src = Path("scripts/train.py").read_text(encoding="utf-8")
    for fn in ("train_fold_lgb", "train_fold_cb", "train_fold_xgb"):
        body = src.split(f"def {fn}")[1].split("\ndef ")[0]
        assert "_split_for_fit" in body, f"{fn} が early stopping 用の分割を経ていない"


def test_early_stopping_inner_split_is_stratified():
    """内側分割がクラス比を保つこと（少数クラスが消えると early stopping が壊れる）。"""
    import numpy as np
    import pandas as pd
    import scripts.train as t

    X = pd.DataFrame({"a": range(200)})
    y = pd.Series([0] * 180 + [1] * 20)
    X_fit, y_fit, X_es, y_es = t._split_for_fit(X, y, X, y)
    assert len(X_fit) + len(X_es) == 200
    assert set(np.unique(y_es)) == {0, 1}, "内側分割で少数クラスが消えた"


def test_stacking_meta_predictions_are_out_of_fold():
    """スタッキングの train 側予測が in-sample でないこと。

    全行で学習したメタモデルで同じ全行を予測すると、返り値をそのまま
    スタッキングの OOF として評価したとき**必ず楽観的に出る**。
    """
    src = Path("src/utils/ensemble.py").read_text(encoding="utf-8")
    body = src.split("def stacking_blend")[1].split("\ndef ")[0]
    assert "cross_val_predict" in body

    import numpy as np
    from src.utils.ensemble import stacking_blend
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    oof = np.column_stack([np.clip(y * 0.5 + rng.normal(0.25, 0.3, 400), 0, 1) for _ in range(3)])
    tr, te = stacking_blend(oof, np.column_stack([rng.random(50)] * 3), y)
    assert tr.shape == (400,) and te.shape == (50,)


def test_weight_bagging_is_available():
    """重み探索の bagging が使えること（`G-CEILING` の集約戦略 (a)）。

    過去コンペの Private 最高は 12 シードの重み bagging だった（L-03）が、
    以前の `optimize_weights` には seed も複数開始点も無く**その戦略が実行できなかった**。
    """
    import inspect
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.utils.ensemble import optimize_weights

    sig = inspect.signature(optimize_weights).parameters
    assert "n_seeds" in sig and "seed" in sig

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    oofs = np.column_stack([np.clip(y * 0.5 + rng.normal(0.25, s, 400), 0, 1)
                            for s in (0.30, 0.32, 0.35)])
    w, score = optimize_weights(oofs, y, roc_auc_score, n_seeds=5)
    assert abs(w.sum() - 1.0) < 1e-6 and 0 < score <= 1


def test_lightgbm_eval_api_is_current():
    """LightGBM の非推奨 API を使っていないこと（警告が出続けると本命の警告が埋もれる）。"""
    import warnings

    import pandas as pd
    from sklearn.datasets import make_classification
    import scripts.train as t

    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        t.train_fold_lgb(X[:150], y[:150], X[150:], y[150:],
                         {"n_estimators": 20, "verbose": -1})


def test_imputation_statistics_come_from_train_only():
    """欠損補完の統計量が test を含まないこと（train/test 混合は明確なリーク）。"""
    src = Path("scripts/preprocess.py").read_text(encoding="utf-8")
    assert "medians = train[NUMERIC_COLS].median()" in src
    assert "concat" not in src, "train と test を連結してから統計量を取っている"
