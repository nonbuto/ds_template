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
    monkeypatch.setattr(g, "_read_stdin_with_deadline", lambda timeout=1.0: payload)
    buf = io.StringIO()
    monkeypatch.setattr(_sys, "stdout", buf)
    g.main()
    out = _json.loads(buf.getvalue())["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask"
    assert "内部エラー" in out["permissionDecisionReason"]


def test_submission_limit_has_single_definition():
    """提出上限の定義元が 1 つであること（表示ごとに違う値になるのを防ぐ）。"""
    from src.config import DAILY_SUBMISSION_LIMIT
    from scripts.harness import deadline_status

    assert deadline_status.DAILY_LIMIT == DAILY_SUBMISSION_LIMIT
    src = Path("scripts/harness/deadline_status.py").read_text(encoding="utf-8")
    assert "DAILY_LIMIT = 10" not in src, "ハーネス側に上限が直書きされている"
    # **submit_gate はトップレベルで import しない**（config 破損でゲートごと落ちるため）
    gate_src = Path("scripts/harness/submit_gate.py").read_text(encoding="utf-8")
    head = gate_src.split("def ")[0]
    assert "from scripts.harness.deadline_status import" not in head, \
        "トップレベル import に戻っている（config 破損で fail-open する）"


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

    # 既定の**値**ではなく「検証 fold を覗かないこと」を検証する
    # （`inner` / `inner_refit` はどちらも覗かない。前者は速く、後者は学習量を取り戻す）
    assert EARLY_STOPPING_ON in ("inner", "inner_refit"), \
        f"既定が検証 fold を見る設定になっている（{EARLY_STOPPING_ON}）"
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
    """スタッキングの train 側予測が in-sample でないこと（`signed_stack`）。

    全行で学習したメタモデルで同じ全行を予測すると、返り値をそのまま
    スタッキングの OOF として評価したとき**必ず楽観的に出る**。
    """
    src = Path("src/utils/ensemble.py").read_text(encoding="utf-8")
    body = src.split("def signed_stack")[1].split("\ndef ")[0]
    assert "cross_val_predict" in body

    import numpy as np
    from src.utils.ensemble import signed_stack
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    oofs = {f"m{i}": np.clip(y * 0.5 + rng.normal(0.25, 0.3, 400), 0, 1) for i in range(3)}
    tests = {f"m{i}": rng.random(50) for i in range(3)}
    _, tr, te, _ = signed_stack(oofs, tests, y, verbose=False)
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

    from src.utils.ensemble import signed_stack

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    oofs = {f"m{i}": np.clip(y * 0.5 + rng.normal(0.25, 0.3, 400), 0, 1) for i in range(3)}
    tests = {f"m{i}": rng.random(60) for i in range(3)}
    X = np.column_stack([oofs[f"m{i}"] for i in range(3)])

    _, train_preds, _, _ = signed_stack(oofs, tests, y, verbose=False)
    in_sample = (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                                   random_state=42))
                 .fit(X, y).predict_proba(X)[:, 1])
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
    # **実リポジトリの `experiments/.running/` を汚染しない。**
    # ここを差し替え忘れると、テストが書いたハートビートが残り、
    # ユーザーの statusline に「9 時間実行中の実験」として出続ける（実際に出ていた）。
    monkeypatch.setattr(ex, "RUNNING_DIR", tmp_path / "running")
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
    import inspect

    from scripts import feature_study as fs

    src = inspect.getsource(fs.main)
    assert "--n-repeats" not in src or True      # 引数の存在は下の CLI 検査で見る
    # **行由来と分割由来は独立成分なので二乗和で合成する。**
    # 分割だけを採ると、分割を増やすほど床が 0 に近づき偽陽性が増える（実測 33%→55%）。
    assert "np.hypot(" in src, "床を二乗和で合成していない"
    assert "np.nanmax([se_rows, se_folds])" in src, "分割が無いときの下限を採っていない"

    help_text = subprocess.run(
        [sys.executable, "-m", "scripts.feature_study", "--help"],
        cwd=ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}).stdout
    assert "--n-repeats" in help_text, "分割を引き直す手段が CLI に無い"


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

    te, _ = add_target_encoding(df, None, ["row_id", "city"], y, is_fold_subset=True)

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

    _, te_test = add_target_encoding(train, test, ["k"], y, smoothing=0.0,
                                     is_fold_subset=True)
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


# ──────────────────────────────────────────────────────────
# 20. 常時走る仕組みが、詰まらない・嘘をつかない
# ──────────────────────────────────────────────────────────

def test_statusline_does_not_block_on_stdin(monkeypatch, capsys):
    """端末から起動しても stdin で固まらないこと。

    `sys.stdin.read()` はパイプが閉じられるまで戻らない。statusLine は **30 秒ごとに走る**
    ので、ここで詰まると影響が大きい（submit_gate で塞いだのと同じ事故）。
    """
    import io
    import sys as _sys

    from scripts.harness import statusline

    class NeverEnding(io.StringIO):
        def isatty(self):
            return True

        def read(self, *a):                      # 呼ばれたら失敗させる
            raise AssertionError("端末から起動されたのに stdin を読んでいる")

    monkeypatch.setattr(_sys, "stdin", NeverEnding())
    assert statusline.main() == 0
    assert capsys.readouterr().out.strip(), "何も表示していない"


def test_statusline_does_not_show_dead_jobs(tmp_path, monkeypatch):
    """死んだプロセスの状態ファイルを「実行中」と表示しないこと。

    異常終了すると状態ファイルが残り続け、statusline だけが何時間も「実行中」と言い続ける。
    実際に**テストが書いたハートビートが 9 時間「実行中」と表示されていた**。
    `job_status` は同じファイルを見て「プロセスが存在しない」と正しく報告しており、
    **2 つのツールが同じ状態を見て食い違っていた**。
    """
    import json as _json
    import os

    from scripts.harness import statusline

    running = tmp_path / "running"
    running.mkdir()
    monkeypatch.setattr(statusline, "RUNNING_DIR", running)

    dead = {"experiment_id": "005", "started_at": "2020-01-01 00:00:00",
            "updated_at": "2020-01-01 00:00:00", "folds_done": 0, "pid": 999999}
    (running / "005.json").write_text(_json.dumps(dead))
    out = statusline._jobs()
    assert "exp005" not in out, f"死んだジョブを実行中として表示している: {out!r}"
    assert "残骸" in out, f"残骸の存在を伝えていない: {out!r}"

    alive = dict(dead, experiment_id="006", pid=os.getpid())
    (running / "006.json").write_text(_json.dumps(alive))
    out2 = statusline._jobs()
    assert "exp006" in out2, f"生きているジョブが表示されない: {out2!r}"


def test_tests_do_not_pollute_the_real_running_dir():
    """テストが実リポジトリの `experiments/.running/` にハートビートを残さないこと。

    `start_run` を呼ぶテストで `RUNNING_DIR` を差し替え忘れると、
    ユーザーの statusline に「何時間も実行中の実験」として出続ける（実際に出ていた）。
    """
    src = Path("tests/test_harness.py").read_text(encoding="utf-8")
    body = src.split("def test_start_run_blocks_when_visualization_is_missing")[1]
    body = body.split("\ndef ")[0]
    assert 'monkeypatch.setattr(ex, "RUNNING_DIR"' in body, \
        "start_run を呼ぶテストが実 .running を汚染する"


def test_mutation_check_does_not_touch_the_working_tree():
    """変異注入が作業ツリーを書き換えないこと。

    最初の実装は `ROOT / rel` を直接書き換えていた。`try/finally` で戻してはいたが、
    **中断されると変異が残り**（実際に 1 回発生）、実行中の数分間は `src/*.py` が
    壊れた状態になる。その間に長時間の学習や並行して読むエージェントが同じファイルを読む。
    """
    src = Path("tests/_mutation_check.py").read_text(encoding="utf-8")
    assert "shutil.copytree(ROOT, work" in src, "複製せずに変異させている"
    assert "p = work / rel" in src, "作業ツリーのファイルを直接指している"
    assert "ROOT / rel" not in src


# ──────────────────────────────────────────────────────────
# 21. 床が「1 回の判定」しか守らないことへの対処
# ──────────────────────────────────────────────────────────

def test_screening_and_adoption_are_distinguished():
    """分割 1 回の計測を「採用推奨」と言い切らないこと。

    分割を 1 回しか引かないと床から**最大成分が抜ける**。実測（無関係な列）:
        1 分割: 行 0.00243 / fold 0.00126 / 分割 —      → 採用 0.00243
        4 分割: 行 0.00243 / fold 0.00126 / 分割 0.00341 → 採用 0.00341（**40% 大きい**）
    かといって常に 3〜5 回引くと FE 1 列の計測時間が 3〜5 倍になり、
    何十件も試す運用と両立しない。**用途で分ける**のが正解。
    """
    from scripts.feature_study import build_verdict

    # 同じ入力でも、スクリーニングか採用判定かで結論が変わる
    strong = dict(delta=0.005, floor=0.001, z=5.0, gap_delta=0.0)
    screening = build_verdict(**strong, is_screening=True)
    adoption = build_verdict(**strong, is_screening=False)

    assert "採用推奨" not in screening, f"床が下限なのに言い切っている: {screening}"
    assert "--n-repeats 3" in screening, f"次に何をするかを指定していない: {screening}"
    assert "採用推奨" in adoption, f"分割を引き直したのに採用と言わない: {adoption}"

    # 床未満なら、どちらのモードでも「測れていない」
    for mode in (True, False):
        v = build_verdict(delta=0.0001, floor=0.001, z=0.2, gap_delta=0.0, is_screening=mode)
        assert "測れていない" in v and "効果がない" not in v

    # 悪化は gap の拡大を併記して棄却
    v = build_verdict(delta=-0.005, floor=0.001, z=-5.0, gap_delta=0.002, is_screening=False)
    assert "棄却" in v and "過学習" in v


@pytest.mark.parametrize("n_tests,sigma,expected_range", [
    (87, 2.0, (1.5, 2.5)),     # 前コンペの FE 仮説数はちょうど 87 件だった
    (87, 3.0, (0.05, 0.3)),
    (10, 2.0, (0.1, 0.4)),
])
def test_expected_false_positives(n_tests, sigma, expected_range):
    """多重比較の期待偽陽性数が正しく計算されること。

    **床は 1 回の判定を守るもので、判定の繰り返しは守らない。**
    2σ で 87 件試せば効果ゼロでも期待 2.0 件が「採用推奨」に見える。
    """
    from src.noise import expected_false_positives

    got = expected_false_positives(n_tests, sigma)
    lo, hi = expected_range
    assert lo <= got <= hi, f"n={n_tests}, {sigma}σ → {got:.2f}（期待 {lo}〜{hi}）"


def test_multiple_testing_note_appears_only_when_relevant():
    """多重比較の注意が、件数が積み上がってから出ること。"""
    from src.noise import multiple_testing_note

    assert multiple_testing_note(3) == "", "件数が少ないうちから出している"
    note = multiple_testing_note(87)
    assert "87 件" in note and "2.0 件" in note


def test_feature_study_counts_prior_tests(tmp_path, monkeypatch):
    """これまでの FE 計測件数を log.csv から数えられること。"""
    import csv as _csv
    import importlib

    import src.config as cfg
    from src.experiment import LOG_CSV_COLUMNS

    with open(tmp_path / "log.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
        w.writeheader()
        for i in range(7):
            w.writerow({"experiment_id": f"{i:03d}", "notes": f"H-{i} ΔOOF=+0.00001 vs base"})
        w.writerow({"experiment_id": "900", "notes": "普通の学習実験（FE 計測ではない）"})

    monkeypatch.setattr(cfg, "EXPERIMENTS_DIR", tmp_path)
    fs = importlib.reload(importlib.import_module("scripts.feature_study"))
    try:
        assert fs._count_prior_feature_tests() == 7
    finally:
        monkeypatch.undo()
        importlib.reload(fs)


def test_screening_floor_is_a_lower_bound_of_the_adoption_floor():
    """スクリーニングの床が、採用判定の床より小さくなりうること（＝下限であること）。

    実測: 1 分割 0.00243 に対し 4 分割 0.00341（**40% 大きい**）。
    この差があるからこそ、スクリーニングで「採用推奨」と言ってはいけない。
    """
    import numpy as np

    # 分割ごとに Δ が動く状況（効果ゼロの特徴量で実際に起きる）
    per_split = np.array([0.00081, 0.00491, 0.01481, -0.00008])
    se_splits = float(np.std(per_split, ddof=1) / np.sqrt(len(per_split)))
    se_rows, se_folds = 0.00243, 0.00126

    screening = float(np.nanmax([se_rows, se_folds]))
    adoption = se_splits
    assert adoption > screening, \
        f"この設定では分割の床が大きいはず（採用 {adoption:.5f} / スクリーニング {screening:.5f}）"

    from src.noise import min_detectable_difference
    delta = float(per_split.mean())
    assert abs(delta) < min_detectable_difference(adoption), \
        "効果ゼロの列が採用判定の床を超えている（前提の確認）"


# ──────────────────────────────────────────────────────────
# 22. DS 視点の指摘（入れ子リーク・床の合成・NN の取り違え）
# ──────────────────────────────────────────────────────────

def test_nn_kind_is_not_consumed_from_the_shared_params():
    """`--model tabm` が全 fold で TabM を使うこと。

    `run_cv` は fold ループの**外**で params を 1 個作り、全 fold に同じ dict を渡す。
    `train_fold_nn` が `params.pop("_nn_kind")` すると、
    **fold0 だけ TabM・残り 4 fold は RealMLP** になる（実測）。
    例外も警告も出ず、log.csv には `tabm` と記録される。
    """
    from scripts.train import build_params

    params = build_params("tabm", 2)
    picked = []
    for _ in range(5):                       # fold ループを模す
        picked.append(params.get("_nn_kind", "realmlp"))
    assert set(picked) == {"tabm"}, f"fold ごとに実装が変わる: {picked}"

    src = Path("scripts/train.py").read_text(encoding="utf-8")
    assert 'params.pop("_nn_kind"' not in src, "共有 dict を破壊している"
    # モデルへは内部キーを渡さない
    assert 'if not k.startswith("_")' in src


def test_hp_search_keeps_the_nn_kind():
    """`--model tabm` の HP 探索が TabM を最適化すること。

    写しキーに `_nn_kind` が無いと、探索は既定の RealMLP を学習し、
    `best_params_tabm_*.json` として**別モデルの HP** が保存される（実測）。
    """
    import optuna

    from scripts.optimize_hp import build_search_params

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for model in ("realmlp", "tabm"):
        params = build_search_params(optuna.create_study().ask(), model, 2)
        assert params.get("_nn_kind") == model, f"{model} の探索が別モデルになる: {params.get('_nn_kind')}"


def test_split_and_row_uncertainty_are_combined_not_replaced():
    """分割を増やしても行由来の不確実性が消えないこと。

    前ラウンドで「分割は 1 段上だから下位を含む」と判断したが**誤り**だった。
    **全分割が同じ行集合を使う**ので、行由来の誤差は全分割に共通に乗り、
    分割間分散から相殺されて消える。モンテカルロ実測（真の効果ゼロ・2σ 判定）:

        分割数        3      5     10     20
        分割のみ    33.2%  34.5%  43.9%  55.0%   ← 増やすほど**悪化**
        二乗和        6.4%   4.9%   4.8%   4.8%   ← 設計値 5% に一致
    """
    import numpy as np

    rng = np.random.default_rng(0)
    s_row, s_split = 0.0024, 0.0034

    def fpr(m, combine):
        hits = 0
        for _ in range(1500):
            e_row = rng.normal(0, s_row)          # 分割によらず共通
            d = e_row + rng.normal(0, s_split, m)
            hits += abs(d.mean()) >= 2 * combine(s_row, d.std(ddof=1) / np.sqrt(m))
        return hits / 1500

    only_split = fpr(20, lambda r, s: s)
    combined = fpr(20, lambda r, s: np.hypot(r, s))
    assert only_split > 0.3, f"この前提が崩れたら再検討（{only_split:.1%}）"
    assert combined < 0.10, f"合成しても偽陽性が高い（{combined:.1%}）"

    src = Path("scripts/feature_study.py").read_text(encoding="utf-8")
    assert "np.hypot(" in src, "床を二乗和で合成していない"


def test_target_encoding_must_be_built_inside_the_fold():
    """fold ループの内側で TE を作る API が用意され、その旨が明示されていること。

    前処理で 1 本作って使い回すと**入れ子のリーク**が残る ——
    学習 fold A の行の TE は「A 以外の全 fold」で集計されており、
    そこにモデルの**検証 fold B の target が入っている**。
    実測（効果ゼロのカテゴリ列・400 カテゴリ・8 seed）:
        前処理で 1 本 : OOF AUC = 0.5194（z = +3.28）
        fold 内で作る : OOF AUC = 0.5080（z = +1.77）
    """
    import numpy as np
    import pandas as pd

    from src.utils.encoders import add_target_encoding_in_fold

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"c": [f"v{v}" for v in rng.integers(0, 20, n)]})
    y = rng.integers(0, 2, n)
    tr, va = np.arange(0, 300), np.arange(300, n)
    X_test = pd.DataFrame({"c": [f"v{v}" for v in rng.integers(0, 20, 50)]})

    tr_out, va_out, te_out = add_target_encoding_in_fold(
        X.iloc[tr], y[tr], X.iloc[va], X_test, ["c"])
    assert "c_te" in tr_out and "c_te" in va_out and "c_te" in te_out
    assert len(tr_out) == 300 and len(va_out) == 100 and len(te_out) == 50
    assert tr_out["c_te"].notna().all() and va_out["c_te"].notna().all()

    doc = Path("src/utils/encoders.py").read_text(encoding="utf-8")
    assert "入れ子のリーク" in doc, "危険が文書化されていない"


def test_clip_does_not_propagate_nan(capsys):
    """`y_train` に欠損があっても予測を全滅させないこと。

    `min`/`max` は NaN を伝播するので、目的変数に欠損が 1 個あるだけで範囲が
    (nan, nan) になり、`np.clip` が**全予測を NaN にする**。
    提出直前に呼ばれる関数なので、中身が全部 NaN の CSV ができうる。
    """
    import numpy as np

    from src.utils.postprocess import clip_predictions

    out, n = clip_predictions(np.array([1.0, 2.0, 3.0]),
                              y_train=np.array([np.nan, 1.0, 2.0]))
    assert np.isfinite(out).all(), f"NaN が伝播している: {out}"
    assert out.tolist() == [1.0, 2.0, 2.0]

    all_nan, n2 = clip_predictions(np.array([1.0, 2.0]), y_train=np.array([np.nan, np.nan]))
    assert np.isfinite(all_nan).all() and n2 == 0, "全欠損でも予測を壊さない"

    with pytest.raises(ValueError, match="範囲が逆"):
        clip_predictions(np.array([1.0]), lo=5.0, hi=1.0)


def test_postprocess_refuses_to_return_broken_predictions(monkeypatch):
    """後処理の結果に有限でない値があれば、黙って返さないこと（提出直前の最後の砦）。"""
    import numpy as np

    from src.utils import postprocess as pp

    monkeypatch.setattr(pp, "unify_duplicates",
                        lambda preds, feats, how="mean": (np.array([np.nan] * len(preds)), 0))
    import pandas as pd
    with pytest.raises(ValueError, match="有限でない"):
        pp.apply_postprocess(np.array([0.1, 0.2]), pd.DataFrame({"a": [1, 2]}))


@pytest.mark.parametrize("pos_rate,min_ratio", [(0.1, 1.8), (0.02, 4.0)])
def test_auc_floor_accounts_for_class_imbalance(pos_rate, min_ratio):
    """AUC の床が陽性率を反映すること。

    AUC の SE は**少数クラスの件数に支配される**。半々と決め打つと不均衡データで
    最も楽観的な床を返す。実測（n=100,000・AUC=0.9）:
        陽性率 10% → 真 0.00209 / 半々仮定 0.00101（**2.1 倍過小**）
        陽性率  2% → 真 0.00463 / 半々仮定 0.00101（**4.6 倍過小**）
    """
    from src.noise import single_score_se

    balanced = single_score_se(metric_name="auc", n=100_000, score=0.9)
    actual = single_score_se(metric_name="auc", n=100_000, score=0.9, pos_rate=pos_rate)
    assert actual / balanced >= min_ratio, f"陽性率が床に反映されていない（{actual/balanced:.2f} 倍）"


def test_empirical_floor_uses_sqrt2_for_comparing_two_submissions():
    """2 提出の LB 差の床が、1 本の gap の √2 倍であること。

    LB₁ − LB₂ = ΔOOF + (gap₁ − gap₂) なので、差のばらつきは gap の SD の √2 倍。
    1 本の 2σ を床にすると **1.41 倍甘く**なり、
    「LB に出るはず」と判断して出した提出が出ない、が体系的に起こる。
    """
    import numpy as np

    from src.noise import EmpiricalFloor

    f = EmpiricalFloor(sd=0.001, n=20, oof_lo=0.9, oof_hi=0.97, offset=0.001)
    assert abs(f.floor - f.single_floor * np.sqrt(2)) < 1e-12
    assert f.floor > f.single_floor


def test_min_detectable_difference_uses_t_for_small_df():
    """繰り返しが少ないとき、正規の 2σ ではなく t 臨界値を使うこと。

    推奨手順の `--n-repeats 3` は自由度 2。正規の 2σ を当てると
    **名目 5% のつもりで実際は 18.3%** の偽陽性率になる（正しい t 臨界値は 4.30）。
    """
    from src.noise import min_detectable_difference as mdd

    assert mdd(1.0) == 2.0
    assert mdd(1.0, df=2) > 4.0, "自由度 2 で正規の臨界値を使っている"
    assert mdd(1.0, df=7) > 2.3
    # 自由度が増えれば正規に近づく
    assert 2.0 < mdd(1.0, df=100) < 2.05


def test_fold_paired_se_applies_nadeau_bengio():
    """fold 対応差の SE が、学習集合の重なりを補正していること。

    fold ごとの差は独立ではない（学習集合が重なる）。素の SE/√k は
    5-fold で 1.50 倍・10-fold で 1.45 倍過小になる。
    """
    import numpy as np

    from src.noise import fold_paired_se

    # 差が完全に一定だと SD=0 になるので、ばらつきのある例で比べる
    a2 = np.array([0.90, 0.915, 0.92, 0.945, 0.94])
    b2 = np.array([0.899, 0.916, 0.918, 0.947, 0.938])
    naive2 = float(np.std(a2 - b2, ddof=1) / np.sqrt(len(a2)))
    got2 = fold_paired_se(a2, b2)
    assert got2 > naive2 * 1.3, f"補正が効いていない（{got2/naive2:.2f} 倍）"
    # 差が一定なら SE は 0（補正しても 0 のまま。浮動小数の丸めは許容する）
    assert fold_paired_se(np.arange(5) * 0.01, np.arange(5) * 0.01 - 0.001) < 1e-12


def test_build_verdict_refuses_when_floor_is_unavailable():
    """床が出せないときに判定を出さないこと。

    `se_rows` と `se_folds` が両方 NaN だと `nanmax` が NaN を返し、
    `abs(delta) < nan` が False になるので**必ず「候補」側に落ちていた**。
    `src/noise.py` の `verdict()` はこれを潰しているのに、
    実際に判定を出す `build_verdict` には同じガードが無かった。
    """
    from scripts.feature_study import build_verdict

    for bad in (float("nan"), 0.0, -1.0):
        v = build_verdict(delta=0.0001, floor=bad, z=float("nan"),
                          gap_delta=0.0, is_screening=False)
        assert "床を推定できません" in v, f"床 {bad} で判定を出している: {v}"


def test_pseudo_labeling_rejects_regression(monkeypatch):
    """回帰で pseudo-labeling を黙って実行しないこと。

    回帰では確信度が定義できない。予測値をそのまま確信度として扱うと、
    値が大きい行ほど確信度が高いという無意味な基準で選び、
    しかも 0/1 の擬似ラベルを連続値ターゲットに混ぜる（例外は出なかった）。
    """
    import pandas as pd

    from src.utils import pseudo

    import src.metrics as m
    monkeypatch.setattr(m, "PROBLEM_TYPE", "regression")

    with pytest.raises(ValueError, match="分類専用"):
        pseudo.make_fold_pseudo(pd.DataFrame({"f": [1, 2]}), [0.5, 1.5],
                                pd.DataFrame({"f": [3]}), lambda *a, **k: (None, None))


def test_nested_te_leak_is_measurably_removed():
    """入れ子リークが実際に減ることを、**リポジトリの API で**測る。

    「前処理で 1 本作って使い回す」と「fold ループの内側で作る」を同じ合成データで比べる。
    効果ゼロのカテゴリ列なので、正しければどちらも OOF AUC ≈ 0.5 になるはず。
    実測では前者が 0.5194（z vs 0.5 = +3.28）、後者が 0.5080（z = +1.77）。
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier

    from src.utils.encoders import add_target_encoding, add_target_encoding_in_fold

    rng = np.random.default_rng(0)
    n, k = 3000, 300
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"c": [f"v{v}" for v in rng.integers(0, k, n)]})   # target と無関係
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def oof_preprocessed():
        # わざと誤用する（前処理で 1 本作って使い回す）。ガードは明示で外す
        te, _ = add_target_encoding(df, None, ["c"], y, cv=cv, smoothing=10.0,
                                    is_fold_subset=True)
        oof = np.zeros(n)
        for tr, va in cv.split(df, y):
            m = DecisionTreeClassifier(max_depth=6, random_state=0)
            m.fit(te[["c_te"]].iloc[tr], y[tr])
            oof[va] = m.predict_proba(te[["c_te"]].iloc[va])[:, 1]
        return roc_auc_score(y, oof)

    def oof_in_fold():
        oof = np.zeros(n)
        inner = StratifiedKFold(5, shuffle=True, random_state=1)
        for tr, va in cv.split(df, y):
            tr_te, va_te, _ = add_target_encoding_in_fold(
                df.iloc[tr], y[tr], df.iloc[va], None, ["c"], inner_cv=inner, smoothing=10.0)
            m = DecisionTreeClassifier(max_depth=6, random_state=0)
            m.fit(tr_te[["c_te"]], y[tr])
            oof[va] = m.predict_proba(va_te[["c_te"]])[:, 1]
        return roc_auc_score(y, oof)

    leaky, safe = oof_preprocessed(), oof_in_fold()
    assert leaky > safe, f"入れ子リークが再現しない（前処理 {leaky:.4f} / fold 内 {safe:.4f}）"
    assert abs(safe - 0.5) < abs(leaky - 0.5), "fold 内で作った方が 0.5 に近くない"


def test_low_level_te_api_warns_about_its_scope():
    """低レベル API の docstring が「渡すのは fold の学習部分」だと明示していること。

    この関数の「fold 外」は**渡された train の中での fold 外**でしかない。
    train 全体を渡して 1 本作れば、モデルの CV から見れば漏れている。
    """
    from src.utils.encoders import add_target_encoding

    doc = add_target_encoding.__doc__ or ""
    assert "その fold の学習部分" in doc
    assert "入れ子" in doc
    assert "add_target_encoding_in_fold" in doc, "正しい API へ誘導していない"

    # **文書だけでなく実行時に止める**（`G-MECH`: 呼び出し側の記憶に任せない）
    import pandas as pd
    with pytest.raises(ValueError, match="fold の学習部分"):
        add_target_encoding(pd.DataFrame({"k": ["a", "b"]}), None, ["k"], [0, 1])


# ──────────────────────────────────────────────────────────
# 23. 過補正の是正（偽陽性を潰して検出力を失わない）
# ──────────────────────────────────────────────────────────

def test_welch_df_restores_detection_power():
    """合成した SE に「小さい方の自由度」を当てないこと。

    支配的な `se_rows` は 400 回のブートストラップから推定され自由度は実質無限大なのに、
    `df = m−1` を全体に当てると精度の高い成分まで不確かと見なす二重の罰になる。
    実測（σ_row=0.0024 / σ_split=0.0034、真の効果 +0.008、m=3）:

        t(m−1) を全体に当てる : 偽陽性 0.0% / **検出力 5.7%**
        Welch の有効自由度     : 偽陽性 5.5% / 検出力 62.7%

    **「偽陽性 33%」を「検出力 5.7%」で置き換えては意味がない**（`G-PERSIST`）。
    """
    import numpy as np

    from scripts.feature_study import _welch_df
    from src.noise import min_detectable_difference as mdd

    for m in (3, 5, 10):
        se_sp = 0.0034 / np.sqrt(m)
        df_eff = _welch_df(0.0024, se_sp, m)
        assert df_eff > m - 1, f"m={m}: 有効自由度 {df_eff:.1f} が m-1 以下"
        se = float(np.hypot(0.0024, se_sp))
        assert mdd(se, df=df_eff) < mdd(se, df=m - 1), f"m={m}: 床が縮んでいない"

    # 片方の成分が 0 でも壊れない
    assert _welch_df(0.0, 0.001, 5) is not None
    assert _welch_df(np.nan, 0.001, 5) is None


def test_floor_survives_when_one_component_is_missing():
    """行・fold の一方が NaN でも、分割の床は活かすこと。

    `nanmax([nan, nan])` は NaN を返し `hypot(nan, x)` も NaN になるので、
    有効な分割の床があるのに「床を推定できません」になっていた。
    """
    import numpy as np

    src = Path("scripts/feature_study.py").read_text(encoding="utf-8")
    assert "hi_safe = 0.0 if not np.isfinite(_hi) else _hi" in src
    assert np.isfinite(np.hypot(0.0, 0.002)), "前提の確認"


def test_verdict_accepts_degrees_of_freedom():
    """`verdict()` が自由度を受け取り、少数標本で床を広げること。

    `end_run` の診断は fold 差 5 個から SE を出すので、正規の 2σ では甘くなる。
    `feature_study` で潰した問題が、**毎回必ず表示される診断**では残っていた。
    """
    from src.noise import verdict

    loose = verdict(0.001, 0.0004)                 # 正規 2σ
    strict = verdict(0.001, 0.0004, df=4)          # fold 5 個から推定
    assert "改善" in loose and "測れていない" in strict

    src = Path("src/experiment.py").read_text(encoding="utf-8")
    assert "noise_verdict(d, se, df=df)" in src, "end_run が自由度を渡していない"


def test_public_pos_rate_reaches_the_guard():
    """陽性率が Public 過剰浮上ガードの閾値まで届いていること。

    `pos_rate` を API に足しても**呼び出し元が渡さなければ**床は半々仮定のまま
    （不均衡で最大 4.6 倍過小 ＝ 警告が鳴りにくい）。
    """
    from src import config

    assert hasattr(config, "PUBLIC_POS_RATE")
    src = Path("src/experiment.py").read_text(encoding="utf-8")
    assert "pos_rate=PUBLIC_POS_RATE or 0.5" in src, "ガードが陽性率を使っていない"


def test_optuna_categorical_choices_are_scalars():
    """Optuna の選択肢がスカラーであること（コンテナだと警告が出続ける）。

    `list` → `tuple` にしても**要素がコンテナである限り警告は消えない**。
    前回「直した」と報告したが実際には出続けていた。
    """
    import warnings

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from src.hp_spaces import nn_space

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        params = nn_space(optuna.create_study().ask())     # 警告が出れば例外になる
    assert isinstance(params["hidden_sizes"], list)
    assert all(isinstance(v, int) for v in params["hidden_sizes"])


def test_low_level_te_requires_explicit_acknowledgement():
    """旧 TE API が**実行時に**誤用を止めること（文書の警告だけにしない）。

    `G-MECH`: 「規約を読んだか」は観測できないが「明示的に申告したか」は観測できる。
    前処理で 1 行書けば実測 +0.019 AUC の楽観が静かに戻る種類の誤用。
    """
    import pandas as pd

    from src.utils.encoders import add_target_encoding

    df = pd.DataFrame({"k": ["a", "b"] * 10})
    y = [0, 1] * 10
    with pytest.raises(ValueError, match="fold の学習部分"):
        add_target_encoding(df, None, ["k"], y)

    out, _ = add_target_encoding(df, None, ["k"], y, is_fold_subset=True)
    assert "k_te" in out


def test_in_fold_te_does_not_run_inner_cv_twice():
    """test を渡しても内側 CV を 2 回まわさないこと（実測 1.88 倍の無駄だった）。"""
    import numpy as np
    import pandas as pd

    from src.utils import encoders

    calls = {"n": 0}
    original = encoders.add_target_encoding

    def counting(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    encoders.add_target_encoding = counting
    try:
        X = pd.DataFrame({"c": [f"v{i % 5}" for i in range(60)]})
        y = np.arange(60) % 2
        encoders.add_target_encoding_in_fold(X.iloc[:40], y[:40], X.iloc[40:50],
                                             X.iloc[50:], ["c"])
    finally:
        encoders.add_target_encoding = original
    assert calls["n"] == 1, f"内側 CV を {calls['n']} 回まわしている"


# ──────────────────────────────────────────────────────────
# 24. early stopping 後の再学習（学習データを取り戻す）
# ──────────────────────────────────────────────────────────

def _synth_frames(tmp_path, n=1500, n_features=10, seed=0):
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=5,
                               flip_y=0.2, random_state=seed)
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    df.to_pickle(tmp_path / "train_features.pkl")
    df[cols].to_pickle(tmp_path / "test_features.pkl")
    return cols


def test_refit_uses_the_whole_fold_train(tmp_path, monkeypatch):
    """`inner_refit` が **学習 fold 100%** で本数固定の学習をし直すこと。

    `inner` は検証 fold を覗かない代わりに学習 fold からさらに 15% を抜くので、
    **最終モデルは全データの 0.8 × 0.85 = 68%** でしか学習していない。
    実測（合成データ・8 seed の対応比較・LightGBM）:
        Δ(refit − inner) = **+0.00122 ± 0.00049（z=+2.48）**
        内側の取り分 10% → +0.00128 / 15% → +0.00202 / 25% → +0.00261
    """
    import numpy as np
    import pandas as pd

    from scripts import train as t

    n_rows = {"fit": []}
    original = t._refit_on_full_fold

    def spy(Est, params, X_tr, y_tr, best_iter, n_key, fit_kwargs=None):
        n_rows["fit"].append((len(X_tr), best_iter))
        return original(Est, params, X_tr, y_tr, best_iter, n_key, fit_kwargs)

    monkeypatch.setattr(t, "_refit_on_full_fold", spy)
    monkeypatch.setattr(t, "EARLY_STOPPING_ON", "inner_refit")

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(600, 6)),
                     columns=[f"f{i}" for i in range(6)])
    y = pd.Series((np.arange(600) % 2))
    params = t.build_params("lgb", 2) | {"n_estimators": 120}
    model, _ = t.train_fold_lgb(X, y, X.iloc[:100], y.iloc[:100], params)

    assert n_rows["fit"], "再学習が呼ばれていない"
    fitted_rows, best = n_rows["fit"][0]
    assert fitted_rows == 600, f"学習 fold 全体を使っていない（{fitted_rows} 行）"
    assert model.n_estimators == best, "early stopping が選んだ本数に固定されていない"
    assert model.n_estimators < 120, "本数が上限のまま（early stopping が効いていない）"


def test_refit_is_off_for_other_modes(tmp_path, monkeypatch):
    """`inner` / `val` では再学習しないこと（速さのための選択肢が機能すること）。"""
    import numpy as np
    import pandas as pd

    from scripts import train as t

    called = {"n": 0}
    monkeypatch.setattr(t, "_refit_on_full_fold",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or None)

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(400, 5)),
                     columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.arange(400) % 2)
    params = t.build_params("lgb", 2) | {"n_estimators": 60}

    for mode in ("inner", "val"):
        monkeypatch.setattr(t, "EARLY_STOPPING_ON", mode)
        t.train_fold_lgb(X, y, X.iloc[:80], y.iloc[:80], params)
    assert called["n"] == 0, f"{mode} で再学習している"


def test_inner_split_follows_the_model_seed():
    """内側分割の seed がモデル seed に追従すること。

    固定にすると multi-seed avg で seed を振っても **ES 用の 15% が毎回同じ行**になり、
    seed 由来の多様性がその分だけ出ない。
    """
    import numpy as np
    import pandas as pd

    from scripts import train as t

    X = pd.DataFrame({"a": range(200)})
    y = pd.Series(np.arange(200) % 2)

    a = t._split_for_fit(X, y, X, y, {"random_state": 0})[0]
    b = t._split_for_fit(X, y, X, y, {"random_state": 7})[0]
    assert not a.index.equals(b.index), "seed を変えても内側分割が同じ"

    # CatBoost 系の名前でも拾えること
    c = t._split_for_fit(X, y, X, y, {"random_seed": 7})[0]
    assert c.index.equals(b.index), "random_seed を見ていない"


def test_best_iteration_is_read_across_libraries():
    """各ライブラリの best iteration をライブラリ差を吸収して取り出せること。"""
    from scripts.train import _best_iteration

    class LGBLike:
        best_iteration_ = 42

    class XGBLike:
        best_iteration = 17

    class CBLike:
        def get_best_iteration(self):
            return 99

    class NoES:
        pass

    assert _best_iteration(LGBLike()) == 42
    assert _best_iteration(XGBLike()) == 17
    assert _best_iteration(CBLike()) == 99
    assert _best_iteration(NoES()) is None


def test_early_stopping_mode_is_selectable_from_cli():
    """`--early-stopping` で 1 回だけ方式を変えられること（設定を書き換えずに済む）。"""
    help_text = subprocess.run(
        [sys.executable, "-m", "scripts.train", "--help"],
        cwd=ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}).stdout
    assert "--early-stopping" in help_text
    for mode in ("inner_refit", "inner", "val"):
        assert mode in help_text, f"{mode} が選べない"


# ──────────────────────────────────────────────────────────
# 25. プロセス制御 —— どう終わり、何が残るか
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("command,expected,label", [
    (f"uv run --no-project {SUB} -c x -f a.csv", True, "uv のオプション付き"),
    (f"timeout 60 {SUB} -c x -f a.csv", True, "timeout でラップ"),
    (f"nice {SUB} -c x -f a.csv", True, "nice でラップ"),
    (f"sudo -u me {SUB} -c x -f a.csv", True, "sudo のオプション付き"),
    (f"env -i {SUB} -c x -f a.csv", True, "env -i"),
    (f"xargs -I{{}} {SUB} -c x -f {{}}", True, "xargs のオプション付き"),
    (f"uv --directory . run {SUB} -c x", True, "uv のサブコマンド前オプション"),
    (f"grep -rn '{SUB}' CONVENTIONS.md", False, "grep（誤検知してはいけない）"),
    ("kaggle competitions submissions -c x", False, "submissions は提出ではない"),
])
def test_submit_gate_ignores_wrapper_flags(command, expected, label):
    """ラッパーの**オプション**をコマンド本体と誤認しないこと。

    以前はラッパーを透過して「コマンド本体」を追っていたため、そこにオプションが来ると
    本体と誤認して**次の演算子まで全部読み飛ばして**いた。実測で 6 パターンが素通り
    （`uv run --no-project` が最も現実的）。
    コマンド位置を追うのをやめ、全トークン位置で三つ組を探せば、
    **ラッパーの一覧を網羅する必要そのものが消える**。
    """
    from scripts.harness.submit_gate import is_submit_command
    assert is_submit_command(command) is expected, label


def test_submit_gate_does_not_import_config_at_module_level():
    """提出ゲートが config 破損で **fail-open** しないこと。

    `deadline_status` をトップレベルで import すると `src/config.py` を読むので、
    config を編集中・venv が壊れている・`uv sync` 途中のいずれでも
    **main() に到達する前**に落ちる。PreToolUse がツールを止めるのは exit 2 だけなので、
    exit 1 で落ちた提出コマンドは**確認なしで実行される**。
    **ゲートが壊れたときだけゲートが消える**という最悪の形だった。
    """
    src = Path("scripts/harness/submit_gate.py").read_text(encoding="utf-8")
    head = src.split("\ndef ")[0]
    assert "from scripts.harness.deadline_status import" not in head
    assert "from src.config import" not in head


@pytest.mark.slow
def test_submit_gate_asks_even_when_config_is_broken(tmp_path):
    """config を壊した複製で、提出コマンドが確認を求められること（実際に走らせる）。"""
    import json as _json
    import shutil
    import subprocess
    import sys as _sys

    work = tmp_path / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".venv*", "kaggle_nb", "data", "*.db"))
    (work / "src" / "config.py").write_text(
        (work / "src" / "config.py").read_text() + "\nthis is not python(((\n")

    r = subprocess.run([_sys.executable, "-m", "scripts.harness.submit_gate"], cwd=work,
                       input=_json.dumps({"tool_name": "Bash",
                                          "tool_input": {"command": f"{SUB} -c x -f y.csv"}}),
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(work)})
    assert r.returncode == 0, f"exit {r.returncode}（hook が落ちると素通りする）"
    decision = _json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"]
    assert decision == "ask", f"config 破損で {decision} になっている"


def test_submit_gate_rejects_truncated_submission(tmp_path, monkeypatch):
    """切り詰められた提出ファイルを通さないこと。

    `to_csv` は非原子的なので、書き込み中に落ちると**途中まで書かれた CSV** が残る。
    ゲートは行数を**表示するだけ**で `sample_submission.csv` と突き合わせていなかった
    （実測: 301 行のはずが 38 行でも `decision=ask` で提出できた）。
    """
    from scripts.harness import submit_gate

    monkeypatch.setattr(submit_gate, "_expected_rows", lambda: 300)
    src = Path("scripts/harness/submit_gate.py").read_text(encoding="utf-8")
    assert "_expected_rows()" in src and "fatal = True" in src
    assert "行数が sample_submission と違います" in src


def test_finalize_writes_atomically():
    """提出 CSV / npy が一時ファイル経由で書かれること。

    途中で落ちると「壊れたファイルが存在する」状態になり、
    推論成果物ガードの glob は「ある」と判定して通す（`FoldCache` で塞いだのと同じ穴）。
    """
    src = Path("src/utils/finalize.py").read_text(encoding="utf-8")
    assert "_save_atomic(" in src and "os.replace(tmp, path)" in src
    assert "sub.to_csv(sub_path, index=False)" not in src, "非原子書き込みが残っている"


def test_heartbeat_is_written_atomically():
    """ハートビートが原子的に書かれること。

    truncate 書きだと読み手（statusline / job_status）が書き換えの途中を読む。
    実測（書き手 1・読み手 1・3 秒）: **77,054 回中 1,807 回（2.3%）がパース失敗**。
    """
    import json as _json
    import tempfile
    import threading
    import time

    from src.experiment import _write_json_atomic

    path = Path(tempfile.mkdtemp()) / "hb.json"
    _write_json_atomic(path, {"folds_done": 0})
    state = {"stop": False, "bad": 0, "reads": 0}

    def writer():
        i = 0
        while not state["stop"]:
            _write_json_atomic(path, {"folds_done": i, "pad": "x" * 200})
            i += 1

    def reader():
        while not state["stop"]:
            state["reads"] += 1
            try:
                _json.loads(path.read_text())
            except Exception:
                state["bad"] += 1

    threads = [threading.Thread(target=f) for f in (writer, reader)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    state["stop"] = True
    for t in threads:
        t.join()

    assert state["reads"] > 100, "読み取り回数が少なすぎて判定できない"
    assert state["bad"] == 0, f"{state['reads']:,} 回中 {state['bad']} 回が壊れた状態を読んだ"


def test_foldcache_signature_includes_early_stopping():
    """`--early-stopping` を変えたら別のキャッシュになること。

    実測: `--early-stopping val` で作った fold を `inner_refit` の実験として
    `--resume` が再利用し、**表示上は普通に完走した**。
    「val を覗いた条件」の予測を「覗かない条件」の OOF として log.csv に記録することになる。
    """
    src = Path("scripts/train.py").read_text(encoding="utf-8")
    assert '"early_stopping": EARLY_STOPPING_ON' in src, "signature に方式が入っていない"
    # 事後に区別できるよう log にも残す
    assert '"_early_stopping": EARLY_STOPPING_ON' in src, "log に学習プロトコルが残らない"

    from src.utils.foldcache import _signature_hash
    base = {"features": ["a"], "params": {}, "n_splits": 5, "split_seed": None}
    refit = _signature_hash({**base, "early_stopping": "inner_refit"})
    leaky = _signature_hash({**base, "early_stopping": "val"})
    assert refit != leaky, "方式が違ってもハッシュが同じ（キャッシュを取り違える）"


def test_temp_copies_exclude_all_virtualenvs():
    """複製が `.venv-autogluon` のような別 venv も除外すること。

    `.venv` の完全一致だと 580 MB が毎回コピーされる。
    実測: 504 MB / 1.8 秒 → 1.8 MB / 0.0 秒。
    """
    for path in ("tests/_e2e_pipeline.py", "tests/_mutation_check.py"):
        src = Path(path).read_text(encoding="utf-8")
        assert '".venv*"' in src, f"{path} が別 venv を除外していない"

    # e2e は失敗時も複製を消す
    e2e = Path("tests/_e2e_pipeline.py").read_text(encoding="utf-8")
    assert "finally:" in e2e and "shutil.rmtree(work.parent" in e2e


def test_stdin_readers_have_a_deadline():
    """常時走るスクリプトが stdin で無限に待たないこと。

    `isatty()` だけでは足りない —— 端末でなくても、パイプが開いたまま EOF が
    来なければ `read()` は戻らない。statusLine は 30 秒ごと、提出ゲートは
    **毎回の Bash の前**に走る。
    """
    import subprocess
    import sys as _sys

    for mod in ("scripts.harness.statusline", "scripts.harness.submit_gate"):
        src = Path(mod.replace(".", "/") + ".py").read_text(encoding="utf-8")
        assert "_read_stdin_with_deadline" in src, f"{mod} が期限つき読み取りでない"

        p = subprocess.Popen([_sys.executable, "-m", mod], cwd=ROOT,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                             env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)})
        try:
            p.wait(timeout=15)          # stdin を閉じずに待つ
        except subprocess.TimeoutExpired:
            p.kill()
            pytest.fail(f"{mod} が stdin でブロックしている")
        finally:
            if p.poll() is None:
                p.kill()


def test_unreachable_public_helper_is_detected(tmp_path):
    """公開ヘルパーに到達経路が無ければ C17 が鳴ること。

    **テンプレートでは「未使用」より「見つけられない」が問題。**
    利用者のコンペコードから呼ばれる道具なので `src/` 内に呼び出しが無いのは正常だが、
    scripts からも文書からも辿れないなら**作ったのに知られないまま埋もれる**
    （導入時の実測で 20 件あった）。
    """
    import shutil
    import subprocess
    import sys as _sys

    work = tmp_path / "repo"
    for rel in ("scripts", "src", ".claude", "CLAUDE.md", "GUIDELINES.md",
                "CONVENTIONS.md", "PLAYBOOK.md", "README.md"):
        srcp = ROOT / rel
        if srcp.is_dir():
            shutil.copytree(srcp, work / rel,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif srcp.exists():
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcp, work / rel)

    target = work / "src" / "utils" / "postprocess.py"
    target.write_text(target.read_text(encoding="utf-8")
                      + "\n\ndef orphan_helper_nobody_can_find(x):\n"
                        '    """誰からも辿れない公開関数。"""\n    return x\n',
                      encoding="utf-8")
    r = subprocess.run([_sys.executable, "-m", "scripts.harness.doc_audit"],
                       cwd=work, capture_output=True, text=True)
    assert "orphan_helper_nobody_can_find" in r.stdout, \
        "到達経路の無い公開ヘルパーを検知していない"


# ──────────────────────────────────────────────────────────
# 26. ガードの費用対効果（安いのは、何もしていないからではないか）
# ──────────────────────────────────────────────────────────

def _post_tool_use_command() -> str:
    import json as _json

    hooks = _json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    for h in hooks["hooks"]["PostToolUse"]:
        for c in h["hooks"]:
            if "viz_guard" in c["command"]:
                return c["command"]
    raise AssertionError("PostToolUse に viz_guard の hook が無い")


def test_post_tool_use_hook_does_not_use_a_time_window():
    """可視化ガードの起動判定が「時間の窓」でないこと。

    以前は「log.csv が 20 秒以内に更新されていたら走る」だった。すると:
      A) 前景で学習 → Bash 終了直後に更新 → 窓の内（動く）
      B) **背景で学習** → 完了時に Bash 呼び出しが無い → 窓の外（**動かない**）
      C) 学習後に別作業を挟む → 次の Bash が 20 秒超 → 窓の外（**動かない**）
    実測: このセッションの 1,319 回の発火機会に対し**一度も実際には走らなかった**。
    **コストが安いのは、ほとんど何もしていないから**だった。
    """
    cmd = _post_tool_use_command()
    assert "-lt 20" not in cmd, "20 秒の時間窓が残っている"
    assert "-nt" in cmd and "viz_guard_seen" in cmd, "更新検知（マーカー比較）になっていない"


@pytest.mark.slow
def test_post_tool_use_hook_runs_once_per_log_change(tmp_path):
    """log.csv が変わったときだけ 1 回走り、平時は走らないこと。

    実測: マーカーなし 676 ms（走る）→ 変化なし 4 ms（走らない）→
    更新後 697 ms（再び走る）→ 4 ms。
    背景実行で時間が空いても、次の Bash で必ず 1 回走る（旧実装は永久に走らなかった）。
    """
    import os
    import shutil
    import subprocess
    import time

    work = tmp_path / "repo"
    for rel in ("scripts", "src", ".claude"):
        shutil.copytree(ROOT / rel, work / rel,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (work / "experiments").mkdir(parents=True, exist_ok=True)
    log = work / "experiments" / "log.csv"
    log.write_text("experiment_id,oof_score\n")

    cmd = _post_tool_use_command()

    def elapsed() -> float:
        t0 = time.time()
        subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True)
        return time.time() - t0

    first = elapsed()          # マーカーが無い → 走る
    second = elapsed()         # 変化なし → 走らない
    time.sleep(1.1)
    os.utime(log, None)        # 背景の学習が log を更新したと想定
    time.sleep(1.0)            # その後しばらくしてから次の Bash（旧実装なら窓の外）
    third = elapsed()
    fourth = elapsed()

    assert first > 0.2, f"初回にガードが走っていない（{first*1000:.0f} ms）"
    assert second < 0.2, f"変化が無いのに走っている（{second*1000:.0f} ms）"
    assert third > 0.2, f"log.csv 更新後に走っていない（{third*1000:.0f} ms）"
    assert fourth < 0.2, f"2 回目も走っている（{fourth*1000:.0f} ms）"


def test_guard_costs_are_bounded():
    """毎ツール前に走るゲートが軽いこと（重い import を遅延していること）。

    実測: PreToolUse 32 ms（提出ゲート）。`uv run` の起動だけで 23 ms なので、
    ゲート自身の処理は 10 ms 程度。numpy/pandas を読むと 650 ms 級になる
    （viz_guard がそれ。だから毎回は走らせない）。
    """
    src = Path("scripts/harness/submit_gate.py").read_text(encoding="utf-8")
    head = src.split("\ndef ")[0]
    for heavy in ("import numpy", "import pandas", "from src.experiment"):
        assert heavy not in head, f"提出ゲートがトップレベルで {heavy} している"


# ──────────────────────────────────────────────────────────
# 27. 未予測行と回帰 —— 黙って間違った答えを返さない
# ──────────────────────────────────────────────────────────

def test_correlation_check_refuses_to_answer_with_nan():
    """未予測行（NaN）があるとき、黙って「追加を検討可」と答えないこと。

    `np.corrcoef` は NaN を伝播し、**`nan < threshold` は False** なので、
    スキップ判定が `False`（＝追加してよい）になる。実測でそう答えていた。
    TimeSeriesSplit の OOF には未予測行が NaN で入る（`train.py` の `covered`）。
    """
    import numpy as np

    from src.utils.ensemble import correlation_check

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    a = np.clip(y * 0.5 + rng.normal(0.25, 0.3, n), 0, 1)
    b = a + rng.normal(0, 0.1, n)

    clean_corr, _ = correlation_check(a, b)
    with_nan = a.copy()
    with_nan[:50] = np.nan
    nan_corr, _ = correlation_check(with_nan, b)

    assert np.isfinite(nan_corr), "NaN を返している（判定が意味を成さない）"
    assert abs(nan_corr - clean_corr) < 0.05, "未予測行を除いた相関が大きくずれている"

    # 相関が定義できない入力は**答えない**（定数列など）
    with pytest.raises(ValueError, match="定数"):
        correlation_check(np.full(n, 0.5), b)


def test_blend_drops_rows_not_predicted_by_all_models():
    """`blend` が未予測行を全モデル共通で落とすこと。

    そのまま重み最適化に渡すと sklearn が `Input contains NaN` で落ちる。
    学習側と同じ扱い ——「全モデルで予測されている行だけで評価する」に揃える。
    """
    src = Path("scripts/blend.py").read_text(encoding="utf-8")
    assert "covered &= np.isfinite" in src, "未予測行のマスクを作っていない"
    assert "y = y[covered]" in src, "正解側にマスクを適用していない"

    # 実際に NaN 入りで重み最適化が落ちることの確認（前提）
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from src.utils.ensemble import optimize_weights

    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    a = np.clip(y * 0.5 + rng.normal(0.25, 0.3, n), 0, 1)
    bad = np.column_stack([a, a + rng.normal(0, 0.1, n)])
    bad[:30, 0] = np.nan
    with pytest.raises(ValueError):
        optimize_weights(bad, y, roc_auc_score)

    good = bad[30:]
    w, _ = optimize_weights(good, y[30:], roc_auc_score)
    assert np.isfinite(w).all(), "マスク後も NaN が残っている"


def test_select_confident_refuses_regression(monkeypatch):
    """下位関数を直接呼んでも回帰で素通りしないこと。

    `make_fold_pseudo` だけ塞いでも、**公開されている以上は直接呼ばれうる**。
    実測: 予測 `[-3.2, 0.4, 120.0, 0.51]` → 確信度 `[4.2, 0.6, 120.0, 0.51]` /
    ラベル `[0,0,1,1]` を例外なく返していた（値が大きい行ほど確信度が高い、という無意味な基準）。
    """
    import numpy as np

    import src.metrics as m
    from src.utils.pseudo import select_confident

    monkeypatch.setattr(m, "PROBLEM_TYPE", "regression")
    with pytest.raises(ValueError, match="分類専用"):
        select_confident(np.array([-3.2, 0.4, 120.0, 0.51]))

    monkeypatch.setattr(m, "PROBLEM_TYPE", "binary_classification")
    mask, labels = select_confident(np.array([0.99, 0.5, 0.01]), threshold=0.9)
    assert mask.tolist() == [True, False, True]
    assert labels.tolist() == [1, 0]


# ──────────────────────────────────────────────────────────
# 28. 検査が「.md にしか届いていない」問題
# ──────────────────────────────────────────────────────────

def _audit_in_copy(tmp_path, mutate) -> str:
    """作業ツリーの複製に手を入れて doc_audit を走らせ、出力を返す。"""
    import shutil
    import subprocess
    import sys as _sys

    work = tmp_path / f"repo_{abs(hash(mutate)) % 10000}"
    for rel in ("scripts", "src", ".claude", "experiments", "CLAUDE.md", "GUIDELINES.md",
                "CONVENTIONS.md", "PLAYBOOK.md", "README.md"):
        srcp = ROOT / rel
        if srcp.is_dir():
            shutil.copytree(srcp, work / rel,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.db"))
        elif srcp.exists():
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcp, work / rel)
    mutate(work)
    return subprocess.run([_sys.executable, "-m", "scripts.harness.doc_audit"],
                          cwd=work, capture_output=True, text=True).stdout


def test_broken_command_in_python_is_detected(tmp_path):
    """`.py` の中の `uv run python scripts/x.py` も検知すること。

    C14 はこの誤り（`src` を import できず `ModuleNotFoundError`）を潰すために作られたのに、
    **走査対象が `.md` だけ**だった。その結果、**可視化ガードがブロック時に印字する
    唯一の解除手順**が動かないコマンドのまま残っていた（実測で 12 箇所）。
    """
    def mutate(work):
        p = work / "src" / "utils" / "postprocess.py"
        p.write_text(p.read_text(encoding="utf-8")
                     + '\n\n# 実行例: uv run python scripts/postprocess.py\n', encoding="utf-8")

    out = _audit_in_copy(tmp_path, mutate)
    assert "postprocess.py" in out and "-m 形式" in out, f"検知していない:\n{out[-800:]}"


def test_dangling_doc_reference_in_python_is_detected(tmp_path):
    """`.py` から存在しない CLAUDE.md の節を参照していたら検知すること。

    v6.6 で CLAUDE.md を憲法に絞った際、中身は GUIDELINES / CONVENTIONS へ移したのに、
    `.py` の参照は旧節名を指したままだった（`_iter_docs` が `.md` しか見ないため）。
    """
    def mutate(work):
        p = work / "src" / "utils" / "postprocess.py"
        p.write_text(p.read_text(encoding="utf-8")
                     + '\n\n# 詳細は CLAUDE.md の「存在しない架空の節」を参照\n', encoding="utf-8")

    out = _audit_in_copy(tmp_path, mutate)
    assert "存在しない架空の節" in out, f"検知していない:\n{out[-800:]}"


def test_missing_config_key_is_detected(tmp_path):
    """config から設定が消えたら検知すること（`ImportError` になる前に）。

    スキルが「ブロックごと置き換える」と指示していたため、**指示どおりに適用すると
    6 キーが消えて `ImportError: cannot import name 'GROUP_COL'`** になった。
    C16 は「config → 文書」の片方向だけで、**消えたことを見ていなかった**。
    """
    def mutate(work):
        p = work / "src" / "config.py"
        p.write_text("\n".join(ln for ln in p.read_text(encoding="utf-8").splitlines()
                               if not ln.startswith("GROUP_COL")), encoding="utf-8")

    out = _audit_in_copy(tmp_path, mutate)
    assert "GROUP_COL" in out and "ImportError" in out, f"検知していない:\n{out[-800:]}"


def test_setup_and_kickoff_do_not_replace_the_config_block():
    """初期化スキルが config を「ブロックごと置換」と指示していないこと。

    現在の config は 12 設定あり、6 キーのスニペットで上書きすると学習が始まらない。
    **次のコンペの Step 2 で確実に踏む**種類の事故。
    """
    kickoff = Path(".claude/skills/ds-kickoff/SKILL.md").read_text(encoding="utf-8")
    assert "ブロックごと置き換えない" in kickoff, "破壊的な指示のまま"
    assert "該当行だけ" in kickoff
    # config が名指ししている設定を kickoff が扱っていること
    for key in ("DAILY_SUBMISSION_LIMIT", "PUBLIC_TEST_ROWS"):
        assert key in kickoff, f"config が /ds-kickoff に委ねている {key} が手順に無い"


def test_visualization_guard_prints_runnable_commands():
    """ブロック時に印字される解除手順が、実際に起動できる形式であること。

    ガードが解けない手順を印字している状態は、**ガードへの信頼を最初に壊す**。
    """
    import re

    src = Path("src/experiment.py").read_text(encoding="utf-8")
    body = src.split("def _check_visualization_guard")[1].split("\ndef ")[0]
    cmds = re.findall(r"uv run python (\S+)", body)
    assert cmds, "解除手順が印字されていない"
    for c in cmds:
        assert c == "-m", f"`{c}` 形式は src を import できない（-m 形式にする）"


def test_experiment_reservation_has_a_command(tmp_path, monkeypatch):
    """実験の目的・成功基準・撤退基準を**コマンドで**予約できること。

    以前は `/ds-new-experiment` が「log.csv に予約追記する」と指示しながら
    **手段を用意していなかった**。前コンペ 271 実験の実測:

        experiment_question / success_criteria / abort_criteria : **35%**
        learning（end_run / submit が支える）                    : **88%**

    **手で書けと言うだけの規律は守られない**（`G-MECH`）。
    """
    import csv as _csv
    import subprocess
    import sys as _sys

    from src.experiment import LOG_CSV_COLUMNS

    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        _csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS).writeheader()

    # 予約コマンドは既定パスに書くので、config を差し替えた環境で走らせる
    runner = tmp_path / "run.py"
    runner.write_text(
        "import sys; sys.path.insert(0, %r)\n"
        "import src.experiment as ex; from pathlib import Path\n"
        "ex.LOG_CSV_PATH = Path(%r)\n"
        "sys.argv = ['x', '--name', 't', '--question', 'Qです',\n"
        "            '--success', 'Sです', '--abort', 'Aです']\n"
        "from scripts.harness import reserve_experiment as r\n"
        "import src.experiment\n"
        "r.main()\n" % (str(ROOT), str(log)), encoding="utf-8")
    r = subprocess.run([_sys.executable, str(runner)], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr[-800:]

    with open(log, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["experiment_question"] == "Qです"
    assert rows[0]["success_criteria"] == "Sです"
    assert rows[0]["abort_criteria"] == "Aです"
    assert not rows[0]["oof_score"], "スコア列は空でなければならない"


def test_reserved_row_is_inherited_by_start_run(tmp_path, monkeypatch):
    """予約行の ID を `start_run` が引き継ぐこと（目的とスコアが 1 行に揃う）。"""
    import csv as _csv
    from datetime import datetime

    from src import experiment as ex
    from src.utils.csvlock import locked_csv

    monkeypatch.setattr(ex, "LOG_CSV_PATH", tmp_path / "log.csv")
    monkeypatch.setattr(ex, "RUNNING_DIR", tmp_path / "running")
    ex._ensure_log_csv()
    with locked_csv(ex.LOG_CSV_PATH, ex.LOG_CSV_COLUMNS) as rows:
        rows.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "experiment_id": "001", "experiment_name": "t",
                     "experiment_question": "Qです", "success_criteria": "Sです",
                     "abort_criteria": "Aです"})

    assert ex._claim_experiment_id("t", "lgb", "説明") == "001", "予約行を引き継いでいない"
    with open(ex.LOG_CSV_PATH, newline="") as f:
        row = list(_csv.DictReader(f))[0]
    assert row["experiment_question"] == "Qです", "目的が失われた"


def test_new_experiment_skill_points_to_the_command():
    """`/ds-new-experiment` が予約コマンドを案内していること（手段の無い指示にしない）。"""
    skill = Path(".claude/skills/ds-new-experiment/SKILL.md").read_text(encoding="utf-8")
    assert "scripts.harness.reserve_experiment" in skill
    assert "手で書かない" in skill
