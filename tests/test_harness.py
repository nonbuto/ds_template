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
