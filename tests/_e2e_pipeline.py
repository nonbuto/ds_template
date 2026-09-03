"""合成データで preprocess → train → predict → blend を通し、成果物が揃うか確かめる。

`tests/test_harness.py::test_end_to_end_pipeline` から subprocess で起動される。
**作業ツリーを複製して別プロセスで走らせる**のは、`src/config.py` のパス定数が
モジュールロード時に決まるため（同一プロセスで差し替えると他のテストに漏れる）。

**二値（auc）と回帰（rmse）の 2 タスクを通す。** 以前は二値しか通しておらず、
回帰では `blend.py` が入口で必ず落ちる・実験雛形が `AxisError` になる・
CatBoost の探索 HP に `eval_metric="AUC"` が残る、といった欠陥が全部素通りしていた。
**タスクを 1 つ増やすだけで、単体テストでは見えない欠陥がまとめて落ちる。**

ここが通ることは、次を同時に保証する:
  - clone 直後の設定で最小ベースラインが動く
  - 学習 → OOF + test 予測 → 提出 CSV が 1 回の実行で出る（`G-STEPWISE`）
  - 指標に合った提出形式になる（AUC は確率、回帰は予測値）
  - `--resume` がキャッシュ経由で完走する
  - `predict.py` が保存済み npy から同じ形式の提出を作る
  - `blend.py` が train.py の出力を受け取り、表示スコアの向きが揃っている
  - 実験雛形（`_TEMPLATE_exp000_s0_example.py`）が実際に完走する
  - NN 系（pytabkit の RealMLP）が tree 系と同じ入口で完走する
  - `--split-seed` / `--n-splits` が実際に配線されている

単独実行:
    uv run python tests/_e2e_pipeline.py .
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

ROOT = Path(sys.argv[1]).resolve()
COLS = [f"num{i}" for i in range(6)]


def build_workdir(task: str) -> Path:
    """タスク別に、合成データと config を仕込んだ作業ディレクトリを作る。"""
    work = Path(tempfile.mkdtemp()) / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
        # `.venv` の完全一致だと **`.venv-autogluon`（580 MB）が毎回コピーされる**。
        # 実測: 504 MB / 1.8 秒 → 1.8 MB / 0.0 秒
        "__pycache__", "*.pyc", ".git", ".venv*", "kaggle_nb", "data"))
    raw = work / "data" / "raw"
    raw.mkdir(parents=True)
    for d in ("processed", "output/submissions", "output/oof", "output/models",
              "output/params", "output/plots"):
        (work / "data" / d).mkdir(parents=True, exist_ok=True)

    if task == "binary":
        X, y = make_classification(n_samples=1200, n_features=6, n_informative=4,
                                   random_state=0)
    else:
        X, y = make_regression(n_samples=1200, n_features=6, n_informative=4,
                               noise=10.0, random_state=0)

    tr = pd.DataFrame(X[:900], columns=COLS)
    tr["id"] = range(900)
    tr["target"] = y[:900]
    te = pd.DataFrame(X[900:], columns=COLS)
    te["id"] = range(900, 1200)
    tr.loc[tr.index[:20], "num0"] = np.nan            # 欠損補完の経路を通す
    tr.to_csv(raw / "train.csv", index=False)
    te.to_csv(raw / "test.csv", index=False)
    pd.DataFrame({"id": te["id"], "target": 0}).to_csv(raw / "sample_submission.csv",
                                                       index=False)

    def patch(path, subs):
        p = work / path
        s = p.read_text()
        for a, b in subs:
            assert a in s, f"{path}: 置換対象が見つからない: {a[:50]}"
            s = s.replace(a, b, 1)
        p.write_text(s)

    patch("scripts/preprocess.py", [("NUMERIC_COLS: list[str] = []",
                                     f"NUMERIC_COLS: list[str] = {COLS}")])
    patch("scripts/train.py", [("FEATURES: list[str] = []", f"FEATURES: list[str] = {COLS}")])
    patch("experiments/runs/_TEMPLATE_exp000_s0_example.py",
          [("FEATURES: list[str] = []", f"FEATURES: list[str] = {COLS}")])
    if task == "regression":
        patch("src/config.py", [
            ('PROBLEM_TYPE = "binary_classification"', 'PROBLEM_TYPE = "regression"'),
            ('EVAL_METRIC = "auc"', 'EVAL_METRIC = "rmse"'),
            ('CV_STRATEGY = "StratifiedKFold"', 'CV_STRATEGY = "KFold"'),
        ])
    return work


def run(work: Path, *args: str) -> str:
    r = subprocess.run([sys.executable, "-m", *args], cwd=work, capture_output=True,
                       text=True, env={"PATH": "/usr/bin:/bin", "DS_SKIP_VIZ_CHECK": "1",
                                       "PYTHONPATH": str(work)})
    print(f"  $ {' '.join(args)}  → exit {r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-2500:])
        sys.exit(1)
    return r.stdout


def check_submission(work: Path, task: str) -> None:
    subs = sorted((work / "data/output/submissions").glob("*.csv"))
    assert subs, "提出 CSV が作られていない（学習→提出の一連が途切れている）"
    sub = pd.read_csv(subs[-1])
    assert len(sub) == 300 and list(sub.columns) == ["id", "target"], f"提出の形が不正: {sub.shape}"
    vals = sub["target"]
    if task == "binary":
        assert vals.between(0, 1).all() and vals.nunique() > 2, \
            f"AUC 設定なのに確率でない（ユニーク {vals.nunique()} 種）"
        print(f"  提出値: 確率 {vals.min():.4f}〜{vals.max():.4f}（{vals.nunique()} 種）")
    else:
        assert vals.nunique() > 2, "回帰なのに離散値を提出している"
        print(f"  提出値: 連続値 {vals.min():.2f}〜{vals.max():.2f}（{vals.nunique()} 種）")


def run_task(task: str) -> None:
    """1 タスク分を通す。**失敗しても複製を残さない**（`try/finally`）。

    以前は成功時にしか `rmtree` しておらず、`run()` が `sys.exit(1)` する経路や
    assert で落ちる経路で複製が残り続けた（実測で 46 件・20.8 GB が残っていた）。
    """
    print(f"\n──── {task} ────")
    work = build_workdir(task)
    try:
        _run_task_body(work, task)
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)


def _run_task_body(work: Path, task: str) -> None:
    run(work, "scripts.preprocess")
    run(work, "scripts.train", "--model", "lgb")
    check_submission(work, task)

    oof = sorted((work / "data/output/oof").glob("oof_*.npy"))
    test = sorted((work / "data/output/oof").glob("test_*.npy"))
    assert oof and test, "OOF / test 予測が保存されていない"

    run(work, "scripts.train", "--model", "lgb", "--resume")
    run(work, "experiments.runs._TEMPLATE_exp000_s0_example", "--model", "lgb")

    # **NN 系も同じ入口で通ること。** ここが通らないと FE の ΔOOF 計測が
    # tree 系だけに対して行われ、特徴量セットが tree に偏って最適化される。
    run(work, "scripts.train", "--model", "realmlp", "--n-splits", "3")
    check_submission(work, task)

    # 分割の引き直しが配線されていること（単一分割では「たまたま良い」を選び続ける）
    run(work, "scripts.train", "--model", "lgb", "--split-seed", "7", "--n-splits", "3")

    # 保存済み npy からの提出も、同じ形式で作られること
    n_before = len(list((work / "data/output/submissions").glob("*.csv")))
    run(work, "scripts.predict", "--test-npy", str(test[0]), "--model", "replay",
        "--oof-score", "0.5", "--exp-id", "900")
    assert len(list((work / "data/output/submissions").glob("*.csv"))) == n_before + 1
    check_submission(work, task)

    out = run(work, "scripts.blend", "--mode", "corr", "--oofs",
              f"a={oof[0]}", f"b={oof[-1]}")
    assert "単体OOFスコア" in out


if __name__ == "__main__":
    for t in ("binary", "regression"):
        run_task(t)
    print("\n✅ e2e 通過（binary / regression）")
