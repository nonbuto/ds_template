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
    # 各プロセスは採番後 3 秒生き続ける（学習中に相当）。採番直後に終了させると、
    # 次のプロセスが「掴んでいたプロセスが死んだ」と見て正当に再利用してしまい、
    # 競合の有無を判定できない。
    procs = [subprocess.Popen([sys.executable, str(helper), str(log), "3"],
                              stdout=subprocess.PIPE, text=True) for _ in range(8)]
    ids = [p.stdout.readline().strip() for p in procs]
    for p in procs:
        p.wait()

    assert len(set(ids)) == 8, f"experiment_id が重複した: {sorted(ids)}"
    import csv as _csv
    with open(log, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 8, f"確保した行が失われた（{len(rows)} 行）"


def test_concurrent_claim_with_reserved_row(tmp_path):
    """**予約行があるとき**も並行実験が別々の ID を取ること。

    `/ds-new-experiment` が置いた予約行を引き継ぐ経路だけ、ID を返すだけで
    行に印を付けていなかった。ロックを抜けた瞬間に次のプロセスが同じ予約行を見つけ、
    8 プロセス同時で**全員が `042` を名乗った**（重複 7 件）。
    `end_run` は同じ ID の行を上書きするので、**8 実験のうち 7 件分の記録が消える**。
    log.csv は唯一の台帳で、git 履歴にも残らない。
    """
    import csv as _csv
    import subprocess
    import sys as _sys
    from src.experiment import LOG_CSV_COLUMNS

    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
        w.writeheader()
        w.writerow({"experiment_id": "042", "experiment_question": "特徴量Xは効くか"})

    helper = Path(__file__).parent / "_concurrent_claim.py"
    procs = [subprocess.Popen([_sys.executable, str(helper), str(log), "3"],
                              stdout=subprocess.PIPE, text=True) for _ in range(8)]
    ids = [p.stdout.readline().strip() for p in procs]
    for p in procs:
        p.wait()

    assert len(set(ids)) == 8, f"予約行を取り合った: {sorted(ids)}"
    assert "042" in ids, "予約行が誰にも引き継がれていない"
    with open(log, newline="") as f:
        assert len(list(_csv.DictReader(f))) == 8


def test_dead_process_releases_its_reservation(tmp_path, monkeypatch):
    """掴んでいたプロセスが落ちたら、その予約行を再利用できること。

    生存確認をせず「印があれば飛ばす」にすると、一度クラッシュした実験の予約行が
    永久に使えなくなり、`/ds-new-experiment` の予約が機能しなくなる。
    """
    import csv as _csv
    from src import experiment as ex

    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
        w.writeheader()
        # 存在しない pid が掴んでいる状態（= クラッシュ後）
        w.writerow({"experiment_id": "042", "experiment_question": "Q",
                    "notes": f"{ex.RUNNING_MARK_PREFIX} pid=999999 — end_run で結果を書き込む）"})
    monkeypatch.setattr(ex, "LOG_CSV_PATH", log)
    assert ex._claim_experiment_id("x", "lgb", "d") == "042", "落ちた実行の予約行が再利用されない"


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


def test_foldcache_signature_separates_conditions(tmp_path):
    """特徴量や HP が変わったら別のキャッシュになること。

    tag を `f"{model}_{len(FEATURES)}f"` で作ると、モデル名と**特徴量の本数**しか
    区別しない。列を入れ替えても HP を変えても本数が同じなら前回の予測が再利用され、
    `--resume` を付けた瞬間に別条件の結果が混ざる（しかも表示上は普通に完走する）。
    """
    import numpy as np
    from src.utils.foldcache import FoldCache

    def cache(features, lr):
        return FoldCache(tag="lgb_2f", seed=42, n_splits=3, cache_dir=tmp_path,
                         signature={"features": features, "params": {"lr": lr}})

    base = cache(["a", "b"], 0.05)
    base.save(0, np.ones((4, 2)), np.ones((2, 2)))

    assert base.load(0) is not None
    assert cache(["a", "c"], 0.05).load(0) is None, "列を変えたのに再利用された"
    assert cache(["a", "b"], 0.10).load(0) is None, "HP を変えたのに再利用された"
    assert cache(["a", "b"], 0.05).load(0) is not None, "同条件なのに再利用されない"


def test_train_passes_cache_signature():
    """学習スクリプトがキャッシュに条件を渡していること。"""
    for path in ("scripts/train.py", "experiments/runs/_TEMPLATE_exp000_s0_example.py"):
        src = Path(path).read_text(encoding="utf-8")
        assert "signature={" in src, f"{path} が FoldCache に条件を渡していない"


# ──────────────────────────────────────────────────────────
# 12. end-to-end —— clone 直後の状態でパイプラインが通ること
# ──────────────────────────────────────────────────────────

@pytest.mark.slow
def test_end_to_end_pipeline():
    """合成データで preprocess → train → blend を通す。

    個々の関数が正しくても**繋がっていない**ことがある。実際、
    「AUC 設定なのにハードラベルを提出」「clone 直後は N_CLASSES=3 で動かない」
    「blend が train.py の出力を撥ねる」は、どれも単体テストでは見えなかった。
    """
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "_e2e_pipeline.py"), str(ROOT)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"e2e 失敗:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"
    assert "e2e 通過" in r.stdout


@pytest.mark.parametrize("command,expected,label", [
    (f'bash -c "{SUB} -c x -f y.csv"', True, "bash -c で包む"),
    (f'sh -c "{SUB} -c x -f y.csv"', True, "sh -c で包む"),
    (f'zsh -c "cd /tmp && {SUB} -c x -f y.csv"', True, "zsh -c の中の && の後"),
    (f"bash -c 'bash -c \"{SUB} -c x\"'", True, "入れ子の -c"),
    (f"grep -rn '{SUB}' CONVENTIONS.md", False, "grep（クォート内は見ない）"),
    (f'python3 -c "print(1)"  # {SUB}', False, "コメント中は実行されない"),
    (f'{SUB} -m "閉じ忘れ', True, "解釈不能な行は安全側（確認）に倒す"),
])
def test_submit_gate_sees_through_shell_wrappers(command, expected, label):
    """`bash -c "…"` の引数は文字列ではなく**実行されるコマンド**なので中身を見ること。

    クォート内を見ないのは意図的（ドキュメント編集を誤検知しないため）だが、
    `sh|bash|zsh -c` だけは例外 —— そこは実行される。17 パターンのテストに
    `-c` ラッパーが 1 件も無く、実測で 5 パターンが素通りしていた。
    """
    from scripts.harness.submit_gate import is_submit_command
    assert is_submit_command(command) is expected, label


def test_submission_count_reports_failure_as_none():
    """提出回数が取得できなかったとき、0 ではなく None を返すこと。

    以前は `returncode` を見ずに stdout だけを数えていたため、403・トークン失効・
    オフライン・コンペ未参加のいずれでも **「本日 0/10 使用済み」という捏造値**が
    提出ゲートに表示されていた。ゲートの存在理由は「提示された数字が記憶であって
    実測でなかった事故を潰す」ことなので、唯一の実測数字が嘘になるのは無いより悪い。
    """
    from scripts.harness.deadline_status import count_todays_submissions
    assert count_todays_submissions("this-competition-does-not-exist-xyz-000") is None


# ──────────────────────────────────────────────────────────
# 13. HP 探索が「本番と同じもの」を最適化していること
# ──────────────────────────────────────────────────────────

def _search_params(model: str, n_cls: int = 2) -> dict:
    import optuna
    from scripts import optimize_hp as o

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return o.build_search_params(optuna.create_study().ask(), model, n_cls)


@pytest.mark.parametrize("model", ["lgb", "cb", "xgb"])
def test_hp_search_uses_production_tree_count(model):
    """探索が本番と同じ木の本数で行われること。

    以前は写しキーに `n_estimators` / `iterations` が無く、LGB / XGB は sklearn 既定の
    **100 本**で探索していた（本番は 1000 本）。100 本で選んだ learning_rate を
    1000 本で使う構図で、低 lr が構造的に選ばれない。合成データでの実測では
    探索が lr=0.05 を選ぶ一方、1000 本での最適は lr=0.01（**−0.00097**）。
    さらに CatBoost だけ既定 1000 本で、モデル間の比較も不公正だった（`G-FAIR`）。
    """
    from scripts.train import build_params

    search = _search_params(model)
    prod = build_params(model, 2)
    key = "iterations" if model == "cb" else "n_estimators"
    assert search[key] == prod[key], f"{model}: 探索 {search.get(key)} vs 本番 {prod[key]}"


@pytest.mark.parametrize("model", ["lgb", "cb", "xgb"])
def test_early_stopping_metric_follows_eval_metric(model):
    """early stopping の監視指標が `EVAL_METRIC` に従うこと。

    とくに CatBoost は探索空間側の `eval_metric="AUC"` が上書きされずに残り、
    **RMSE コンペでも AUC で best iteration を選んでいた**（例外は出ない）。
    """
    from src.config import EVAL_METRIC
    from src.metrics import native_eval_metric

    search = _search_params(model)
    got = search.get("metric") or search.get("eval_metric")
    expected = native_eval_metric(model, 2)
    if expected is not None:
        assert got == expected, f"{model}: {got!r} は EVAL_METRIC={EVAL_METRIC!r} に対応しない"


def test_hp_spaces_have_no_task_dependent_keys():
    """探索空間にタスク依存キー（目的関数・評価指標）が直書きされていないこと。

    直書きすると、多クラス・回帰コンペでも二値前提のキーが探索空間から入り込み、
    `build_params()` が上書きしないキー（CatBoost の `eval_metric`）はそのまま生き残る。
    """
    import optuna
    from src import hp_spaces

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial = optuna.create_study().ask()
    for name, fn in (("lgb", hp_spaces.lgb_space), ("xgb", hp_spaces.xgb_space),
                     ("cb", hp_spaces.cb_space)):
        space = fn(trial)
        for key in ("objective", "metric", "eval_metric", "loss_function", "num_class"):
            assert key not in space, f"{name}_space にタスク依存キー {key} が直書きされている"


def test_optimize_hp_shares_the_training_code(monkeypatch):
    """HP 探索が**本番と同じ学習関数**を通ること（別実装を持たないこと）。

    探索と本番で別の学習コードを持つと、early stopping のプロトコルや目的関数が
    静かにずれる（実際にずれていた: 探索だけ検証 fold を監視し、
    LGB/XGB は 100 本・CB は 1000 本で比較していた）。
    **字面ではなく、実際に TRAIN_FN が呼ばれることを見る。**
    """
    import numpy as np
    import optuna
    import pandas as pd
    from sklearn.datasets import make_classification

    from scripts import optimize_hp as o
    from scripts import train as t

    calls = []

    def spy(X_tr, y_tr, X_val, y_val, params):
        calls.append(len(X_tr))
        return t.train_fold_lgb(X_tr, y_tr, X_val, y_val, params)

    monkeypatch.setitem(o.TRAIN_FN, "lgb", spy)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y)
    trial = optuna.create_study().ask()
    o.objective(trial, X, y, "lgb", np.array([0.5, 0.5]), 2)

    assert calls, "探索が TRAIN_FN を経由していない（別の学習コードを持っている）"
    # early stopping 用の内側分割を通っているので、学習行数は fold の train より少ない
    assert all(n < 300 for n in calls)


def test_optuna_study_name_includes_condition_hash(tmp_path):
    """study 名に条件のハッシュが入り、条件が変われば別 study になること。

    以前は `{model}_{tag}` だけで `load_if_exists=True` だったため、FEATURES や指標を
    変えて再実行すると**旧条件の trial と混ざり、best が旧セットから返り得た**。
    FoldCache に `signature` を入れて塞いだのと同型の問題。
    """
    from src.utils.foldcache import _signature_hash

    a = _signature_hash({"features": ["a", "b"], "metric": "auc"})
    b = _signature_hash({"features": ["a", "c"], "metric": "auc"})
    c = _signature_hash({"features": ["a", "b"], "metric": "logloss"})
    assert len({a, b, c}) == 3, "条件が変わってもハッシュが同じ"

    src = Path("scripts/optimize_hp.py").read_text(encoding="utf-8")
    assert "_signature_hash(" in src and "study_name = f\"{args.model}_{args.tag}_{sig}\"" in src


# ──────────────────────────────────────────────────────────
# 14. 自分が守るはずの事故で自分が壊れないこと
# ──────────────────────────────────────────────────────────

def test_foldcache_survives_corrupted_file(tmp_path, capsys):
    """保存中に落ちて壊れた .npy を「無い」として扱うこと。

    このモジュールの存在理由は「kill・クラッシュで fold の計算が失われるのを防ぐ」ことなのに、
    保存の最中に落ちると中途半端な .npy が残り、次の `--resume` が
    `ValueError: EOF: reading array header` で落ちていた（復旧手段は手動削除だけ）。
    """
    import numpy as np
    from src.utils.foldcache import FoldCache

    c = FoldCache(tag="t", seed=42, n_splits=3, cache_dir=tmp_path)
    c.save(0, np.ones((4, 2)), np.ones((2, 2)))
    assert c.load(0) is not None

    p = c._path(0, "test")
    p.write_bytes(p.read_bytes()[:20])          # 保存中クラッシュを再現
    assert c.load(0) is None, "壊れたキャッシュで例外を投げている"
    assert c.completed_folds() == [], "report() が落ちる"


def test_foldcache_save_is_atomic(tmp_path):
    """保存が一時ファイル + os.replace であること（中途半端な .npy を残さない）。"""
    import numpy as np
    from src.utils.foldcache import FoldCache

    c = FoldCache(tag="t", seed=1, n_splits=2, cache_dir=tmp_path)
    c.save(0, np.arange(6).reshape(3, 2), np.arange(4).reshape(2, 2))
    leftovers = [f.name for f in tmp_path.iterdir() if "tmp" in f.name]
    assert not leftovers, f"一時ファイルが残っている: {leftovers}"
    src = Path("src/utils/foldcache.py").read_text(encoding="utf-8")
    assert "os.replace(tmp, path)" in src


def test_unmeasured_diagnostics_are_blank_not_zero(tmp_path, monkeypatch):
    """fold スコアを 1 度も記録しない実験が、診断列に 0.00000 を書かないこと。

    以前は空リストのとき 0.0 を渡していたため、`_fmt()` の NaN → 空欄という対策に
    到達する前に無効化され、`log_fold_scores` を呼ばない実験が診断記録ガードに
    **「記入済み」と数えられていた**（実測 100% すり抜け）。
    しかも画面には測っていない 0.00000 が診断値として表示されていた。
    """
    import numpy as np
    from src import experiment as ex

    assert ex._fmt(float("nan")) == ""
    arr = np.asarray([], dtype=float)
    assert np.isnan(float(arr.mean()) if arr.size else float("nan"))

    src = Path("src/experiment.py").read_text(encoding="utf-8")
    body = src.split("def end_run")[1].split("\n    def ")[0]
    assert "if self._fold_train_scores else 0.0" not in body, \
        "測っていない診断値に 0.0 を入れている"


def test_delta_oof_compares_same_conditions(tmp_path, monkeypatch):
    """ΔOOF の比較相手が同条件（同じモデル・特徴量セット）に限られること。

    以前は「最後に oof_score が入っている行」を無条件に取っていたため、
    CatBoost の直後に LightGBM を回すと**異種モデル間の差が「ΔOOF」として表示された**。
    診断機構そのものが `G-FAIR` 違反を作っていた。
    """
    import csv as _csv
    from src import experiment as ex

    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
        w.writeheader()
        w.writerow({"experiment_id": "001", "model": "lgb", "features": "7features",
                    "oof_score": "0.900"})
        w.writerow({"experiment_id": "002", "model": "cb", "features": "7features",
                    "oof_score": "0.950"})
    monkeypatch.setattr(ex, "LOG_CSV_PATH", log)

    assert ex._previous_experiment_scores(model="lgb", features="7features") == 0.900
    assert ex._previous_experiment_scores(model="xgb", features="7features") is None, \
        "条件が揃う実験が無いのに比較相手を返した"


def test_av_check_shares_training_protocol():
    """AV 診断が学習側と同じ early stopping / importance の定義を使うこと。

    train と test を完全に同じ分布から作った帰無条件（真の AV-AUC = 0.500）の実測:
        検証 fold で ES : 0.5166 / 0.5039 / 0.5176
        内側分割で ES   : 0.4951 / 0.4949 / 0.4957
    判定帯を +0.005〜+0.018 押し上げるので、境界付近では判定が逆になる。
    importance も split ベースだと、高カーディナリティ列が上位に来やすい指標で
    **「上位重要度特徴量を drop 検討」という破壊的な判断**を決めることになる。
    """
    src = Path("scripts/av_check.py").read_text(encoding="utf-8")
    assert "_split_for_fit" in src, "検証 fold で early stopping している"
    assert "extract_importance" in src, "importance が split ベースのまま"
    assert "model.feature_importances_" not in src


def test_leakage_check_detects_duplicate_rows():
    """train と test に同一行があることを検知すること。

    `preprocess.py` のコメントは「train/test に同一行が混入する事故を保存前に止める」と
    主張していたのに実装が無く、完全に同じ 3 行を test に入れても `✅ PASS` を返していた。
    """
    import pandas as pd
    from src.validation import validate_no_leakage

    tr = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "target": [0, 1, 0, 1]})
    dup = pd.DataFrame({"a": [1, 2, 9], "b": [5, 6, 10]})       # 先頭 2 行が train と同一
    clean = pd.DataFrame({"a": [9, 10], "b": [11, 12]})

    assert validate_no_leakage(tr, dup, "target").passed is False
    assert validate_no_leakage(tr, clean, "target").passed is True
    assert validate_no_leakage(tr, tr.drop(columns=[]), "target").passed is False, \
        "test に target 列がある場合も落ちること"


def test_latest_submission_is_chosen_by_mtime(tmp_path):
    """「最新の提出ファイル」を辞書順ではなく更新時刻で選ぶこと。

    命名は `sub_{exp_id}_{model}_{score}_{ts}.csv` なので、辞書順では先頭の exp_id が効き
    時刻は効かない（exp010 を後から作っても exp100 が選ばれる）。
    Notebook コンペではこれが**間違った CSV の提出**になる。
    """
    import os
    import time

    (tmp_path / "sub_100_lgb_0.9_20260101_0000.csv").write_text("x")
    time.sleep(0.01)
    newest = tmp_path / "sub_010_lgb_0.9_20260601_0000.csv"
    newest.write_text("x")

    by_name = sorted(tmp_path.glob("sub_*.csv"))[-1]
    by_mtime = sorted(tmp_path.glob("sub_*.csv"), key=os.path.getmtime)[-1]
    assert by_name != newest and by_mtime == newest, "この前提が崩れたらテストを見直す"

    src = Path("scripts/to_kaggle_nb.py").read_text(encoding="utf-8")
    assert "key=os.path.getmtime" in src, "辞書順で最新を選んでいる"


def test_shape_logic_is_not_copied_around():
    """指標に合わせた整形が `shape_for_metric` の外に写経されていないこと。

    同じ三項演算子が 6 箇所に写経され、片方だけ直る事故が起きた（L-29 #2）。
    その後も `save_oof_analysis` と実験雛形に残っていた。
    """
    import re

    pattern = re.compile(r"needs_proba\(\).*shape\[1\] == 2")
    for path in Path("src").rglob("*.py"):
        if path.name == "metrics.py":
            continue
        assert not pattern.search(path.read_text(encoding="utf-8")), f"{path} に写経が残っている"
    for path in list(Path("scripts").rglob("*.py")) + list(Path("experiments").rglob("*.py")):
        assert not pattern.search(path.read_text(encoding="utf-8")), f"{path} に写経が残っている"


# ──────────────────────────────────────────────────────────
# 15. 振る舞いで検証する（ソースの字面ではなく）
# ──────────────────────────────────────────────────────────
# L-28 で「ガードが発火するテストを書く」と結論したが、実際に書いたのは
# **過去のパッチの字面のテスト**だった。字面テストは「同じ diff を revert する」ことしか
# 検知せず、意味的に別の書き方で同じ欠陥を入れると必ず通る。
# 実測（変異注入 25 件）で 13 件がすり抜け、欠陥 3 件が同時に存在した状態で全件緑だった。
# ここでは**実装を呼んで出力を確かめる**。


def _synth_classification(n=600, n_features=6, seed=0):
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=4,
                               random_state=seed)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), pd.Series(y)


def test_importance_is_gain_not_split_by_value():
    """`extract_importance()` の戻り値が gain と一致し、split とは一致しないこと。

    字面テスト（`'importance_type="gain"' in src`）は XGBoost 分岐の同じ文字列で
    満たされてしまい、LightGBM だけ split に戻しても通っていた（実測）。
    **値で確かめれば、どう書き換えても検知できる。**
    """
    import lightgbm as lgb
    import numpy as np
    from scripts.train import extract_importance

    X, y = _synth_classification()
    model = lgb.LGBMClassifier(n_estimators=30, verbose=-1).fit(X, y)
    got = extract_importance(model, list(X.columns))

    gain = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(model.booster_.feature_importance(importance_type="split"), dtype=float)
    assert np.allclose(got, gain), "gain と一致しない"
    assert not np.allclose(got, split), "split を返している（gain とは別物）"


def test_stacking_train_predictions_are_not_in_sample():
    """スタッキングの train 側予測が、全行学習の in-sample 予測と**数値として異なる**こと。

    字面テスト（`"cross_val_predict" in body`）は関数内の import 行で満たされ、
    実装を in-sample に戻しても通っていた（実測）。
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.utils.ensemble import stacking_blend

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    oof = np.column_stack([np.clip(y * 0.5 + rng.normal(0.25, 0.3, 400), 0, 1)
                           for _ in range(3)])
    test = np.column_stack([rng.random(60) for _ in range(3)])

    train_preds, _ = stacking_blend(oof, test, y)
    in_sample = (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
                                                                   random_state=42))
                 .fit(oof, y).predict_proba(oof)[:, 1])
    assert not np.allclose(train_preds, in_sample), "in-sample 予測を返している"


def test_test_imputation_uses_train_statistics(tmp_path):
    """`preprocess` が test の欠損を **train の中央値**で埋めること。

    字面テスト（該当行の存在 + `concat` を含まない）は、test 側を test 自身の中央値で
    埋めるよう書き換えても通っていた（実測）。**実際に走らせて値を見る。**
    """
    import subprocess
    import sys
    import shutil

    import pandas as pd

    work = tmp_path / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".venv", "kaggle_nb", "data"))
    raw = work / "data" / "raw"
    raw.mkdir(parents=True)
    (work / "data" / "processed").mkdir(parents=True)

    # train と test で中央値が明確に違うデータ（train=10, test=100）
    train = pd.DataFrame({"num0": [10.0] * 9 + [None], "target": [0, 1] * 5})
    test = pd.DataFrame({"num0": [100.0] * 9 + [None]})
    train.to_csv(raw / "train.csv", index=False)
    test.to_csv(raw / "test.csv", index=False)

    src = work / "scripts" / "preprocess.py"
    src.write_text(src.read_text().replace("NUMERIC_COLS: list[str] = []",
                                           'NUMERIC_COLS: list[str] = ["num0"]'))
    r = subprocess.run([sys.executable, "-m", "scripts.preprocess"], cwd=work,
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(work)})
    assert r.returncode == 0, r.stderr[-1500:]

    filled = pd.read_pickle(work / "data" / "processed" / "test_features.pkl")["num0"].iloc[-1]
    assert filled == 10.0, f"test を train の中央値(10)で埋めていない（実際: {filled}）"


def test_weight_bagging_actually_changes_weights():
    """`n_seeds` を増やすと**重みが実際に変わる**こと。

    字面テスト（シグネチャに `n_seeds` があること・重みの和が 1）は、
    `n_seeds` を完全に無視する実装でも通っていた（実測）。
    この関数は「bagging を実行できるようにする」ために作られたので、
    実行できない状態に戻ったら落ちなければ意味がない。
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.utils.ensemble import optimize_weights

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    oofs = np.column_stack([np.clip(y * 0.5 + rng.normal(0.25, s, 600), 0, 1)
                            for s in (0.30, 0.34, 0.40, 0.5)])

    w1, _ = optimize_weights(oofs, y, roc_auc_score, n_seeds=1)
    w8, _ = optimize_weights(oofs, y, roc_auc_score, n_seeds=8)
    assert not np.allclose(w1, w8), "n_seeds を変えても重みが同じ（bagging が効いていない）"


def test_unpredicted_rows_do_not_change_the_score():
    """未予測行がスコアに混ざらないこと（`covered` マスクが実際に効いていること）。

    字面テスト（`"covered" in src`）は、マスクを作るだけで評価に使わない実装でも
    通っていた（実測）。ここでは**同じ値になるべき 2 つのスコア**を比べる。
    """
    import numpy as np
    from src.metrics import get_metric, shape_for_metric

    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    proba = np.clip(np.column_stack([1 - (y * 0.6 + rng.normal(0.2, 0.2, n)),
                                     y * 0.6 + rng.normal(0.2, 0.2, n)]), 0, 1)
    covered = np.ones(n, dtype=bool)
    covered[:100] = False              # TimeSeriesSplit の先頭を模す
    oof = proba.copy()
    oof[~covered] = 0.0                # 未予測はゼロのまま

    metric = get_metric()
    masked = metric(y[covered], shape_for_metric(oof[covered]))
    naive = metric(y, shape_for_metric(oof))
    truth = metric(y[covered], shape_for_metric(proba[covered]))

    assert abs(masked - truth) < 1e-12, "マスクした評価が実力と一致しない"
    assert abs(naive - truth) > 1e-4, \
        f"未予測行を混ぜてもスコアが動かない（masked={masked:.5f} naive={naive:.5f}）"


def test_start_run_blocks_when_visualization_is_missing(tmp_path, monkeypatch):
    """可視化ガードが**実行を止める**こと（警告ではなくブロック）。

    `start_run` の docstring は「第4世代の対策として、警告の出力ではなく実行を止める
    （過去3世代とも『警告は出ていたが対応されない』形で形骸化した）」と書いているのに、
    **テストファイル全体に `start_run` という文字列が 1 度も出てこなかった**。
    `raise` を `print` に格下げしても誰も気づかない状態だった。
    """
    import csv as _csv
    from src import experiment as ex

    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
        w.writeheader()
        for i in range(ex.VIZ_GUARD_WINDOW):
            w.writerow({"experiment_id": f"{i:03d}", "timestamp": "2020-01-01 00:00:00",
                        "model": "lgb", "oof_score": "0.5"})
    plots = tmp_path / "plots"
    plots.mkdir()
    monkeypatch.setattr(ex, "LOG_CSV_PATH", log)
    monkeypatch.setattr(ex, "PLOTS_DIR", plots)
    monkeypatch.delenv("DS_SKIP_VIZ_CHECK", raising=False)

    tracker = ex.ExperimentTracker(experiment_name="t", model="lgb", features="1f")
    with pytest.raises(RuntimeError, match="可視化"):
        tracker.start_run(description="ブロックされるべき")

    # 明示的に省略の意思表示をしたときは通ること（逃げ道が塞がっていないこと）
    assert tracker.start_run(description="省略を明示", skip_viz_check=True)


def test_atomic_write_is_never_observed_half_written(tmp_path):
    """書き戻しの最中に読んでも、壊れた状態を観測しないこと。

    以前のテストは名前に反して**並行読み手を一切作らず**、逐次に 2 回書いて内容を
    見るだけだった。非原子的な `open(w)` に戻しても通っていた（実測）。
    """
    import csv as _csv
    import subprocess
    import sys
    import textwrap

    path = tmp_path / "log.csv"
    cols = ["experiment_id", "oof_score"]
    from src.utils.csvlock import write_rows_atomic

    rows = [{"experiment_id": f"{i:03d}", "oof_score": "0.9"} for i in range(300)]
    write_rows_atomic(path, cols, rows)

    reader_src = tmp_path / "reader.py"
    reader_src.write_text(textwrap.dedent(f"""
        import csv, time
        bad = 0
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            try:
                with open({str(path)!r}, newline="") as f:
                    n = len(list(csv.DictReader(f)))
                if n not in (300, 301):
                    bad += 1
            except (FileNotFoundError, PermissionError):
                bad += 1
        print(bad)
    """))
    reader = subprocess.Popen([sys.executable, str(reader_src)],
                              stdout=subprocess.PIPE, text=True)
    for _ in range(60):
        write_rows_atomic(path, cols, rows + [{"experiment_id": "999", "oof_score": "0.1"}])
        write_rows_atomic(path, cols, rows)
    bad = int(reader.communicate()[0].strip())
    assert bad == 0, f"書き換え中に壊れた状態を {bad} 回観測した"

    with open(path, newline="") as f:
        assert len(list(_csv.DictReader(f))) == 300


def test_run_cv_excludes_unpredicted_rows_from_its_score(tmp_path, monkeypatch):
    """`run_cv` が報告する OOF スコアが、実際に予測された行だけで計算されていること。

    前のテスト（`test_unpredicted_rows_do_not_change_the_score`）は指標側のロジックしか
    見ておらず、`run_cv` が `covered` を作るだけで**評価に使わない**実装に戻しても通った
    （変異注入で実測）。ここでは実際に `run_cv` を呼び、報告値を突き合わせる。

    TimeSeriesSplit は先頭の行をどの fold の検証にも入れないので、ゼロのまま採点すると
    「全部クラス0と予測した」ことになり、**スコアが実力と無関係に動く**。
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    from scripts import train as t
    from src import metrics as m

    X, y = make_classification(n_samples=600, n_features=5, n_informative=3, random_state=0)
    cols = [f"f{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    df.to_pickle(tmp_path / "train_features.pkl")
    df[cols].to_pickle(tmp_path / "test_features.pkl")

    monkeypatch.setattr(t, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(m, "CV_STRATEGY", "TimeSeriesSplit")

    result = t.run_cv("lgb", t.build_params("lgb", 2) | {"n_estimators": 20}, seed=0,
                      features=cols)

    oof = np.asarray(result["oof_preds"], dtype=float)
    covered = ~np.isnan(oof).any(axis=1)
    assert not covered.all(), "TimeSeriesSplit なのに全行が予測されている（前提の確認）"

    metric = m.get_metric()
    on_covered = metric(y[covered], m.shape_for_metric(oof[covered]))
    filled = np.nan_to_num(oof, nan=0.0)
    naive = metric(y, m.shape_for_metric(filled))

    assert abs(result["oof_score"] - on_covered) < 1e-12, \
        f"報告値 {result['oof_score']:.6f} が予測済み行のスコア {on_covered:.6f} と一致しない"
    assert abs(naive - on_covered) > 1e-4, \
        f"未予測行を混ぜてもスコアが動かない（この差が検知の根拠。naive={naive:.6f}）"


@pytest.mark.slow
def test_mutations_are_detected():
    """実装に欠陥を注入したら、テストが必ず落ちること。

    **「テストが通ること」と「テストが守っていること」は別物**で、後者は欠陥を入れて
    初めて測れる。L-28 で「ガードが発火するテストを書く」と結論した直後に書いたテストは
    ソースの字面の grep で、変異注入 25 件のうち **13 件がすり抜け**、
    欠陥 3 件が同時に存在した状態で 98 件全件が緑だった（L-30）。

    この検査自体が「置換対象が実装に無い」場合も落ちる —— 検査対象が消えたことを
    合格にすると、L-28 #2（入力が消えれば検査項目ごと消える）と同じ穴になる。
    """
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "_mutation_check.py"), str(ROOT)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"検知できない変異がある:\n{r.stdout}\n{r.stderr[-1000:]}"
    assert "❌" not in r.stdout, r.stdout


# ──────────────────────────────────────────────────────────
# 16. 判断の床が実測であること（固定値の表ではない）
# ──────────────────────────────────────────────────────────

def test_auc_noise_floor_matches_hanley_mcneil():
    """AUC の床が Hanley-McNeil 式と一致し、旧テーブルの ±0.0001 とは桁で違うこと。

    `G-NOISE` の表は「Hanley-McNeil 由来」と明記しながら ±0.0001 を掲げていたが、
    式に n_pos=n_neg=5,000 / AUC=0.9 を入れると **0.0032**（32 倍）。
    ±0.0001 は実際には「相関 0.999 のペア差」の値で、それを単体スコアの床として掲げ、
    さらに「paired は 5-10x 小さい」と書いていたため実効閾値が 10〜20 倍甘くなっていた。
    **これが L-21（OOF 有意な 6 件が全部 LB に再現しなかった）を説明する。**
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.noise import auc_se, single_score_se

    assert auc_se(0.90, 5_000, 5_000) > 0.002, "旧テーブルの値では説明できない大きさ"

    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.integers(0, 2, n)
    pred = y * 1.9 + rng.normal(0, 1, n)
    got = single_score_se(y, pred, n_boot=200)
    analytic = auc_se(roc_auc_score(y, pred), n // 2, n // 2)
    assert 0.5 < got / analytic < 2.0, f"実測 {got:.5f} と解析式 {analytic:.5f} が桁で違う"


def test_paired_floor_is_much_smaller_than_single_floor():
    """対応差の床が単体スコアの床より 1〜2 桁小さいこと。

    **用途が違う 2 つの床を取り違えると、判断が桁で狂う。**
    単体の床で FE を判定すると実在する改善を切り捨て、
    対応差の床で LB を判定すると偶然を「突破」と呼ぶ。
    """
    import numpy as np
    from src.noise import paired_se, single_score_se

    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.integers(0, 2, n)
    a = y * 1.9 + rng.normal(0, 1, n)
    b = a + rng.normal(0, 0.05, n)          # ほぼ同じ 2 本

    single = single_score_se(y, a, n_boot=150)
    paired = paired_se(y, a, b, n_boot=150)
    assert paired < single / 10, f"対応差 {paired:.5f} が単体 {single:.5f} と近すぎる"


def test_fold_paired_se_is_smaller_than_val_std():
    """fold 対応差の SE が、fold 間 val std より小さいこと。

    `G-DIAG` は「ΔOOF が fold 間 std より小さいなら測れていない」としていたが、
    val std は「fold ごとの難易度の差」を主成分に含み、同じ fold で 2 つを比べれば
    相殺する成分。実測では val std 0.01251 に対し正しい床は 0.00124（**10 倍**）で、
    実在する改善を体系的に「判別不能」と切り捨てていた
    （L-19: 個別 Δ≈0 が 13 系統累積すると確定的な正の差になった）。
    """
    import numpy as np
    from src.noise import fold_paired_se

    # fold の難易度が大きく違い、2 本の差は小さい状況
    difficulty = np.array([0.80, 0.86, 0.90, 0.94, 0.99])
    a = difficulty + np.array([0.0010, 0.0012, 0.0009, 0.0011, 0.0008])
    b = difficulty

    val_std = float(np.std(a))
    se = fold_paired_se(a, b)
    assert se < val_std / 10, f"fold 対応差の SE {se:.5f} が val std {val_std:.5f} と近い"
    assert abs((a - b).mean()) > 2 * se, "この設定では差が検出できるはず（前提の確認）"


def test_feature_study_uses_measured_floor():
    """`feature_study` が固定閾値ではなく実測した床で判定すること。

    旧実装の ±0.0003 / +0.001 は指標非依存の絶対値で、ΔOOF 自身の seed 間ばらつき
    （実測 SD 0.0011）の 1/4 しかなかった。**完全に無関係な列が seed 次第で
    「🔶 採用検討」「⬜ ノイズ範囲」「❌ 棄却」の 3 判定すべてを出す。**
    実際、合成データで無関係な列の ΔOOF は +0.00081 になり、
    旧閾値なら「採用検討」と判定されていた（新しい床では z=+0.33 で「測れていない」）。
    """
    src = Path("scripts/feature_study.py").read_text(encoding="utf-8")
    assert "paired_se" in src and "fold_paired_se" in src
    assert "min_detectable_difference" in src
    for old in ("delta > 0.001", "delta > 0.0003", "delta > -0.0003"):
        assert old not in src, f"固定閾値 {old} が残っている"


def test_pub_oof_gap_guard_carries_information(tmp_path, monkeypatch):
    """Public 過剰浮上ガードが、帰無条件と真の危険を区別できること。

    修正前は基準線が**全提出の中央値**だったため、検知したい系統差そのものが
    基準線に吸収されていた。モンテカルロ（20,000 回）で帰無条件 93.1% /
    真に危険な条件 92.9% と**発火率がほぼ同じ** —— この警告は情報を持っていなかった。
    さらに閾値 0.0005 が LB のノイズ床（実測 0.002 前後）より小さく、
    純粋なノイズで 84〜97% 発火していた。
    """
    import csv as _csv
    import numpy as np
    from src import experiment as ex

    def build(gap_fn, lb_se=0.002, n=25, seed=0):
        rng = np.random.default_rng(seed)
        log = tmp_path / f"log_{gap_fn.__name__}_{seed}.csv"
        with open(log, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
            w.writeheader()
            for i in range(n):
                lb = 0.97 + gap_fn(i) + rng.normal(0, lb_se)
                w.writerow({"experiment_id": f"{i:03d}", "oof_score": "0.97000",
                            "submit_score": f"{lb:.5f}"})
        return log

    def flat(i):
        return 0.0

    def constant_offset(i):
        return 0.004        # 5-fold OOF vs full-train test の**正当な**オフセット

    def drift(i):
        return 0.0 if i < 15 else 0.006      # 後から浮く = 検知したい形

    fired = {}
    for fn in (flat, constant_offset, drift):
        hits = 0
        for seed in range(12):
            monkeypatch.setattr(ex, "LOG_CSV_PATH", build(fn, seed=seed))
            hits += ex._check_pub_oof_gap_guard() is not None
        fired[fn.__name__] = hits / 12

    assert fired["flat"] <= 0.25, f"帰無条件で鳴りすぎ（{fired['flat']:.0%}）"
    assert fired["constant_offset"] <= 0.25, \
        f"正当な一定オフセットで鳴っている（{fired['constant_offset']:.0%}）"
    assert fired["drift"] >= 0.6, f"後から浮いたのに鳴らない（{fired['drift']:.0%}）"


def test_fold_scores_are_recorded_for_paired_comparison():
    """fold ごとの val スコアが log.csv に残ること（対応差の床を出すのに要る）。"""
    from src.experiment import LOG_CSV_COLUMNS

    assert "fold_val_scores" in LOG_CSV_COLUMNS
    src = Path("src/experiment.py").read_text(encoding="utf-8")
    assert "fold_paired_se" in src, "診断が fold 対応差の床を使っていない"
    assert "abs(d) < val_std" not in src, "val std を床に使う旧判定が残っている"

    # **実際に記録して読み戻せること**を見る（字面ではなく振る舞い）
    import csv as _csv
    import tempfile
    from pathlib import Path as _Path
    from src import experiment as ex

    tmp = _Path(tempfile.mkdtemp()) / "log.csv"
    with open(tmp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
        w.writeheader()
        w.writerow({"experiment_id": "001", "model": "lgb", "features": "7f",
                    "oof_score": "0.90000",
                    "fold_val_scores": ";".join(ex._fmt(v, digits=8)
                                                for v in (0.901234567, 0.902345678))})
    old = ex.LOG_CSV_PATH
    try:
        ex.LOG_CSV_PATH = tmp
        folds = ex._previous_fold_scores(model="lgb", features="7f")
    finally:
        ex.LOG_CSV_PATH = old
    assert folds is not None and len(folds) == 2
    # 丸めで差が消えていないこと（5 桁だと対応差の SE が 0 に潰れる）
    assert abs((folds[1] - folds[0]) - 0.001111111) < 1e-7, f"精度が足りない: {folds}"


# ──────────────────────────────────────────────────────────
# 17. 実績から測る床（LB に現れるか）
# ──────────────────────────────────────────────────────────

def _floor_log(tmp_path, gaps, base_oof=0.9690):
    """指定した gap 列を持つ log.csv を作る。"""
    import csv as _csv
    from src.experiment import LOG_CSV_COLUMNS

    log = tmp_path / f"log_{len(gaps)}_{gaps[0]:.5f}.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
        w.writeheader()
        for i, g in enumerate(gaps):
            oof = base_oof + i * 1e-5
            w.writerow({"experiment_id": f"{i:03d}", "model": "lgb", "features": "7f",
                        "oof_score": f"{oof:.5f}", "submit_score": f"{oof + g:.5f}"})
    return log


def test_empirical_floor_measures_gap_dispersion(tmp_path):
    """床が gap の**散らばり**から出ること（平均オフセットは床に影響しないこと）。

    `gap = LB − OOF` の平均は 5-fold OOF と全学習相当の test 予測の差による
    正当なオフセットで、毎回同じだけ乗るので差を取れば消える。
    **閾値の情報を持つのは SD の方。**
    """
    import csv as _csv
    from src.noise import empirical_lb_floor

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.0001, 20)

    def read(log):
        with open(log, newline="") as f:
            return empirical_lb_floor(list(_csv.DictReader(f)))

    tight = read(_floor_log(tmp_path, 0.001 + noise))
    shifted = read(_floor_log(tmp_path, 0.005 + noise))       # オフセットだけ違う
    loose = read(_floor_log(tmp_path, 0.001 + noise * 10))    # 散らばりが 10 倍

    assert abs(tight.floor - shifted.floor) < 1e-6, "平均オフセットが床を動かしている"
    assert loose.floor > tight.floor * 5, "散らばりが床に反映されていない"
    assert shifted.offset > tight.offset, "オフセット自体は報告されるべき"


def test_empirical_floor_needs_enough_history(tmp_path):
    """実績が足りないときは床を出さない（推測で数字を作らない）。"""
    import csv as _csv
    from src.noise import EMPIRICAL_MIN_N, empirical_lb_floor

    few = _floor_log(tmp_path, [0.001] * (EMPIRICAL_MIN_N - 1))
    with open(few, newline="") as f:
        assert empirical_lb_floor(list(_csv.DictReader(f))) is None


def test_below_floor_guard_fires_only_when_nothing_clears_the_floor(tmp_path, monkeypatch):
    """直近の実験がすべて床の下なら通知し、1 件でも床を超える改善があれば黙ること。

    s6e8 の最終盤では**隣接実験の ΔOOF が 1 件残らず床の下**にあり、
    8 日・32 提出のあいだ LB は 1 度も更新されなかった。
    「飽和した」のではなく**検出可能な大きさの差を作れていなかった**（`G-CALIB-SUB`）。
    """
    import csv as _csv
    from src import experiment as ex

    rng = np.random.default_rng(0)

    def build(recent_gains):
        log = tmp_path / f"g_{recent_gains[0]:.5f}_{len(recent_gains)}.csv"
        with open(log, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=ex.LOG_CSV_COLUMNS)
            w.writeheader()
            for i in range(20):                       # 床を測るための実績
                oof = 0.9690 + i * 1e-5
                w.writerow({"experiment_id": f"{i:03d}", "oof_score": f"{oof:.5f}",
                            "submit_score": f"{oof + 0.0011 + rng.normal(0, 0.00007):.5f}"})
            best = 0.9690 + 19 * 1e-5
            for j, gain in enumerate(recent_gains):    # 直近の実験（未提出でよい）
                w.writerow({"experiment_id": f"{100 + j}", "oof_score": f"{best + gain:.5f}"})
        return log

    below = [1e-5, -2e-5, 5e-6, -1e-5, 0.0, 3e-6, -4e-5, 1e-5]     # 全部床の下
    monkeypatch.setattr(ex, "LOG_CSV_PATH", build(below))
    assert ex._check_below_floor_guard() is not None, "床の下ばかりなのに通知しない"

    cleared = below[:4] + [0.0020] + below[5:]                      # 1 件だけ床超えの改善
    monkeypatch.setattr(ex, "LOG_CSV_PATH", build(cleared))
    assert ex._check_below_floor_guard() is None, "床を超える改善があるのに通知している"

    # **悪化の大きさを「床超え」と数えない**（符号の取り違え）
    worse = below[:4] + [-0.0050] + below[5:]
    monkeypatch.setattr(ex, "LOG_CSV_PATH", build(worse))
    assert ex._check_below_floor_guard() is not None, \
        "大きく悪化した実験を「まだ測れる領域」と誤認している"


def test_submit_gate_reports_floor_ratio(tmp_path, monkeypatch):
    """提出ゲートが「今回の ΔOOF は床の何倍か」を出すこと。

    提出枠は限られた資源で、LB は分散の大きい観測器。床未満の差を提出しても、
    返るのは情報にならない結果だけ。
    """
    import csv as _csv
    from src import config
    from scripts.harness import submit_gate

    log = tmp_path / "log.csv"
    from src.experiment import LOG_CSV_COLUMNS
    rng = np.random.default_rng(0)
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
        w.writeheader()
        for i in range(20):
            oof = 0.9690 + i * 1e-5
            w.writerow({"experiment_id": f"{i:03d}", "oof_score": f"{oof:.5f}",
                        "submit_score": f"{oof + 0.0011 + rng.normal(0, 0.00007):.5f}"})
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)

    tiny = submit_gate._floor_lines("sub_900_blend_0.96921_20260902_1200.csv")   # +0.00002
    worse = submit_gate._floor_lines("sub_902_blend_0.96900_20260902_1200.csv")  # ベスト以下
    high = submit_gate._floor_lines("sub_901_blend_0.97500_20260902_1200.csv")   # 大幅改善

    assert any("床" in line for line in tiny)
    assert any("🟡" in line for line in tiny), f"床未満なのに警告が出ない: {tiny}"
    assert any("上回っていません" in line for line in worse), \
        f"ベストより悪い提出が警告されない: {worse}"
    assert any("🟢" in line for line in high) and not any("🟡" in line for line in high), \
        f"床を超える改善なのに警告が出ている: {high}"


# ──────────────────────────────────────────────────────────
# 18. 測定装置（分割の引き直し）とアンサンブル器
# ──────────────────────────────────────────────────────────

def test_split_seed_is_wired_through(tmp_path, monkeypatch):
    """`split_seed` が実際に分割を変えること（引数はあるが効かない、を防ぐ）。

    `get_cv(seed=)` は前ラウンドで追加したのに、**呼び出し 3 箇所すべてが引数なし**で、
    271 実験すべてが単一の分割（seed=42）に条件付いていた。
    行のブートストラップは分割由来の分散を再現しないので、
    「OOF 有意なのに LB に再現しない」の構造的な原因になる。
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    from scripts import train as t
    from src import metrics as m

    X, y = make_classification(n_samples=400, n_features=5, n_informative=3, random_state=0)
    cols = [f"f{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    df.to_pickle(tmp_path / "train_features.pkl")
    df[cols].to_pickle(tmp_path / "test_features.pkl")
    monkeypatch.setattr(t, "PROCESSED_DATA_DIR", tmp_path)

    params = t.build_params("lgb", 2) | {"n_estimators": 15}
    a = t.run_cv("lgb", params, seed=0, features=cols, split_seed=1)
    b = t.run_cv("lgb", params, seed=0, features=cols, split_seed=2)
    assert not np.allclose(a["oof_preds"], b["oof_preds"]), "split_seed を変えても分割が同じ"

    # fold 数も効くこと
    c = t.run_cv("lgb", params, seed=0, features=cols, n_splits=3)
    assert len(c["val_scores"]) == 3
    assert len(a["val_scores"]) == m.N_SPLITS

    for path in ("scripts/train.py", "scripts/feature_study.py"):
        src = Path(path).read_text(encoding="utf-8")
        assert "split_seed" in src, f"{path} が split_seed を扱っていない"


def test_feature_study_includes_split_variance():
    """`feature_study` が分割由来の分散を床に含めること。

    実測: **完全に無関係な列**の ΔOOF が分割次第で −0.00008〜+0.01481 まで動き、
    分割 1σ=0.00341 が行 1σ=0.00243・fold 1σ=0.00126 を上回った。
    **最も見落とされやすい成分が、実は最大だった。**
    """
    src = Path("scripts/feature_study.py").read_text(encoding="utf-8")
    assert "--n-repeats" in src and "se_splits" in src
    assert "np.nanmax([se_rows, se_folds, se_splits])" in src, "分割の床が採用されていない"


def test_zero_floor_is_not_reported_as_a_result():
    """床がゼロ同然に潰れたとき「測れた」と言わないこと。

    記録の丸めや fold 数の不足で対応差がすべて同一になると SE が 0 近くに潰れ、
    z が発散して「z=+68 で改善」のような無意味な断定が出る（実際に出した）。
    """
    from src.noise import verdict

    assert "床が算出できません" in verdict(0.00009, 1.3e-9)
    assert "測れていない" in verdict(0.00009, 0.00035)


def test_fold_scores_keep_enough_precision():
    """fold スコアが対応差を計算できる精度で保存されること。

    表示用の 5 桁で保存すると、差が小さいときに全部 0 になり SE が潰れる。
    """
    src = Path("src/experiment.py").read_text(encoding="utf-8")
    assert "_fmt(v, digits=8)" in src, "fold スコアを丸めて保存している"
    from src.experiment import _fmt
    assert _fmt(0.123456789, digits=8) == "0.12345679"


def test_hillclimb_uses_replacement_and_bagging():
    """Caruana 型の選択が、非復元・等重みの `greedy_ensemble` より表現力を持つこと。

    `greedy_ensemble` は非復元 + 等重み平均しか作れない。
    Playground の定石は復元あり hillclimb + サブセット bagging で、
    **選ばれた回数がそのまま重みになる**。
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.utils.ensemble import hillclimb

    rng = np.random.default_rng(0)
    n = 1500
    y = rng.integers(0, 2, n)
    oofs = {f"m{i}": np.clip(y * 0.5 + rng.normal(0.25, s, n), 0, 1)
            for i, s in enumerate((0.30, 0.33, 0.36, 0.50, 0.70))}

    weights, ens, score = hillclimb(oofs, y, roc_auc_score, n_iter=30, n_bags=6, verbose=False)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert score >= max(roc_auc_score(y, v) for v in oofs.values()), "単体ベストを下回った"
    # 等重みでない = 復元選択が効いている
    positive = [w for w in weights.values() if w > 0]
    assert len(set(round(w, 6) for w in positive)) > 1, "等重みしか作れていない"


def test_signed_stack_can_use_negative_coefficients():
    """符号制約なしスタッキングが、simplex では表現できない結合を作れること。

    `optimize_weights` は非負かつ合計 1 なので、**弱いメンバーを引き算に使う経路が
    構造的にふさがれている**。前コンペで終盤に唯一 LB で確認できた改善
    （累計 +0.00035、2σ 超え）は、まさにこの結合方式の変更によるものだった。
    """
    import numpy as np
    from src.utils.ensemble import signed_stack

    rng = np.random.default_rng(0)
    n = 1200
    y = rng.integers(0, 2, n)
    signal = y * 0.5 + rng.normal(0.25, 0.30, n)
    bias = rng.normal(0, 1, n)
    oofs = {"good": np.clip(signal, 0, 1),
            "biased": np.clip(signal + 0.8 * bias, 0, 1),   # 誤差を引き算で消せる
            "noise": np.clip(bias, 0, 1)}

    coefs, oof, test, score = signed_stack(oofs, None, y, verbose=False)
    assert any(c < 0 for c in coefs.values()), "負の係数を一度も使っていない"
    assert oof.shape == (n,) and test is None


def test_blend_exposes_new_modes():
    """`blend.py` から新しいアンサンブル器を呼べること（実装しても導線が無いと使われない）。"""
    src = Path("scripts/blend.py").read_text(encoding="utf-8")
    assert '"hillclimb"' in src and '"stack"' in src
    assert "signed_stack(" in src and "hillclimb(" in src


# ──────────────────────────────────────────────────────────
# 19. 定石の実装（TE / pseudo / 後処理）—— リークしないこと
# ──────────────────────────────────────────────────────────

def test_target_encoding_does_not_leak():
    """**行ごとに一意な列**を target encoding しても情報が漏れないこと。

    素朴な TE（全 train で集計）だとその列は target そのものになる。実測:
        素朴な TE  : AUC = 1.00000  ← 完全なリーク
        fold 外 TE : AUC = 0.50000
    TE のリークは**エラーを出さない**。学習時だけスコアが跳ね、LB で落ちる形で現れる。
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from src.utils.encoders import add_target_encoding

    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({"row_id": [f"id{i}" for i in range(n)],
                       "city": rng.choice([f"c{i}" for i in range(8)], n)})
    effect = {f"c{i}": i / 7 for i in range(8)}
    y = (rng.random(n) < np.array([0.2 + 0.6 * effect[c] for c in df.city])).astype(int)

    te, _ = add_target_encoding(df, None, ["row_id", "city"], y)

    leaked = roc_auc_score(y, te["row_id_te"])
    assert abs(leaked - 0.5) < 0.05, f"一意な列から情報が漏れている（AUC={leaked:.5f}）"

    signal = roc_auc_score(y, te["city_te"])
    assert signal > 0.65, f"本物の信号まで消えている（AUC={signal:.5f}）"


def test_target_encoding_test_side_uses_full_train():
    """test 側は train 全体で計算され、未知カテゴリは事前平均になること。"""
    import numpy as np
    import pandas as pd

    from src.utils.encoders import add_target_encoding

    train = pd.DataFrame({"k": ["a"] * 20 + ["b"] * 20})
    test = pd.DataFrame({"k": ["a", "b", "zzz"]})       # zzz は train に無い
    y = np.array([1] * 20 + [0] * 20)

    _, te_test = add_target_encoding(train, test, ["k"], y, smoothing=0.0)
    vals = te_test["k_te"].to_numpy()
    assert vals[0] > vals[1], "a(全部1) が b(全部0) より高くない"
    assert abs(vals[2] - y.mean()) < 1e-9, "未知カテゴリが事前平均になっていない"


def test_count_encoding_needs_no_target():
    """count encoding が target を使わないこと（リークしようがない）。"""
    import inspect

    from src.utils import encoders

    sig = inspect.signature(encoders.add_count_encoding).parameters
    assert "y" not in sig, "count encoding が target を受け取っている"

    import pandas as pd
    tr = pd.DataFrame({"k": ["a", "a", "b"]})
    te = pd.DataFrame({"k": ["a", "c"]})
    out_tr, out_te = encoders.add_count_encoding(tr, te, ["k"], normalize=False)
    assert out_tr["k_count"].tolist() == [3, 3, 1]      # train+test 合算で a は 3 回
    assert out_te["k_count"].tolist() == [3, 1]


def test_pseudo_labeling_stays_inside_the_fold():
    """擬似ラベルが**この fold の学習部分だけ**から作られること。

    「全 train で学習したモデルで pseudo を作る」実装は、その予測が検証 fold の
    情報を含むため OOF を必ず楽観側へ寄せる（そして LB では再現しない）。
    """
    import numpy as np
    import pandas as pd

    from src.utils.pseudo import make_fold_pseudo

    seen = {}

    def spy_train(X_tr, y_tr, X_val, y_val, params):
        seen["n_rows"] = len(X_tr)

        class M:
            def predict_proba(self, X):
                p = np.linspace(0.01, 0.99, len(X))
                return np.column_stack([1 - p, p])

        return M(), None

    X_tr = pd.DataFrame({"f": range(100)})
    X_test = pd.DataFrame({"f": range(200, 260)})
    y_tr = pd.Series(np.repeat([0, 1], 50))

    X_aug, y_aug, w_aug = make_fold_pseudo(X_tr, y_tr, X_test, spy_train, threshold=0.9)

    assert seen["n_rows"] == 100, "擬似ラベル生成に fold の学習部分以外を渡している"
    assert len(X_aug) == len(y_aug) == len(w_aug)
    assert len(X_aug) > 100, "擬似ラベルが 1 件も足されていない"
    assert (w_aug[:100] == 1.0).all() and (w_aug[100:] < 1.0).all(), \
        "擬似ラベル行が本物と同じ重みになっている"


def test_pseudo_selection_respects_threshold_and_cap():
    """確信度の閾値と上限件数が効くこと。"""
    import numpy as np

    from src.utils.pseudo import describe_pseudo, select_confident

    proba = np.column_stack([1 - np.linspace(0.01, 0.99, 100),
                             np.linspace(0.01, 0.99, 100)])
    mask_hi, _ = select_confident(proba, threshold=0.99)
    mask_lo, _ = select_confident(proba, threshold=0.6)
    assert mask_hi.sum() < mask_lo.sum(), "閾値が効いていない"

    capped, labels = select_confident(proba, threshold=0.6, max_n=5)
    assert capped.sum() == 5, "上限件数が効いていない"
    assert len(labels) == 5

    empty = np.zeros(10, dtype=bool)
    assert "採用 0 件" in describe_pseudo(empty, np.array([]), 100), \
        "0 件のときに警告が出ない（「効かなかった」の前に「実行されたか」を見るための表示）"


def test_unify_duplicates_reduces_variance():
    """重複行の予測統一が、実際にスコアを動かすこと。"""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from src.utils.postprocess import unify_duplicates

    rng = np.random.default_rng(0)
    n = 2000
    base = rng.integers(0, 12, (n, 2))
    feat = pd.DataFrame(base, columns=list("ab"))
    p_true = 1 / (1 + np.exp(-(base[:, 0] - 6) / 2))
    y = (rng.random(n) < p_true).astype(int)
    pred = np.clip(p_true + rng.normal(0, 0.12, n), 0.001, 0.999)

    after, n_dup = unify_duplicates(pred, feat)
    assert n_dup > 0, "この設定では重複があるはず（前提の確認）"
    assert roc_auc_score(y, after) > roc_auc_score(y, pred), "統一しても改善しない"


def test_rank_transform_preserves_auc_exactly():
    """rank 変換が AUC を変えないこと（順序を保つ変換なので保証される）。"""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from src.utils.postprocess import rank_transform

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    pred = np.clip(y * 0.5 + rng.normal(0.25, 0.3, 800), 0, 1)
    assert abs(roc_auc_score(y, rank_transform(pred)) - roc_auc_score(y, pred)) < 1e-12


def test_postprocess_skips_rank_unless_auc(monkeypatch):
    """AUC 以外では rank 変換を実行しないこと（値そのものを見る指標で予測を別物にしない）。"""
    import numpy as np
    import pandas as pd

    from src.utils import postprocess as pp

    pred = np.linspace(0.01, 0.99, 50)
    feat = pd.DataFrame({"a": range(50)})

    out, note = pp.apply_postprocess(pred, feat, rank=True)
    assert "rank 変換（" in note, f"AUC 設定なのに rank が効いていない: {note}"

    import src.config as cfg
    monkeypatch.setattr(cfg, "EVAL_METRIC", "logloss")
    out2, note2 = pp.apply_postprocess(pred, feat, rank=True)
    assert "スキップ" in note2, f"logloss で rank を実行している: {note2}"
    assert np.allclose(out2, pred), "予測が変わってしまっている"


def test_clip_uses_training_range():
    """clip が学習データの範囲を使い、変更行数を報告すること。"""
    import numpy as np

    from src.utils.postprocess import clip_predictions

    y = np.array([10.0, 20.0, 30.0])
    pred = np.array([5.0, 20.0, 99.0])
    out, n = clip_predictions(pred, y_train=y)
    assert out.tolist() == [10.0, 20.0, 30.0] and n == 2
