"""合成データで preprocess → train → blend を通し、成果物が揃うか確かめる。

`tests/test_harness.py::test_end_to_end_pipeline` から subprocess で起動される。
**作業ツリーを複製して別プロセスで走らせる**のは、`src/config.py` のパス定数が
モジュールロード時に決まるため（同一プロセスで差し替えると他のテストに漏れる）。

ここが通ることは、次を同時に保証する:
  - clone 直後の設定（binary_classification / auc）で最小ベースラインが動く
  - 学習 → OOF + test 予測 → 提出 CSV が 1 回の実行で出る（`G-STEPWISE`）
  - AUC 設定で**確率**が提出される（ハードラベルでない）
  - `--resume` がキャッシュ経由で完走する
  - blend が train.py の出力（二値 (n,2)）を受け取れる

単独実行:
    uv run python tests/_e2e_pipeline.py .
"""
import shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(sys.argv[1])
work = Path(tempfile.mkdtemp()) / "repo"
shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", ".venv", "kaggle_nb", "data"))

raw = work / "data" / "raw"; raw.mkdir(parents=True)
for d in ("processed", "output/submissions", "output/oof", "output/models",
          "output/params", "output/plots"):
    (work / "data" / d).mkdir(parents=True, exist_ok=True)

X, y = make_classification(n_samples=1200, n_features=6, n_informative=4, random_state=0)
cols = [f"num{i}" for i in range(6)]
tr = pd.DataFrame(X[:900], columns=cols); tr["id"] = range(900); tr["target"] = y[:900]
te = pd.DataFrame(X[900:], columns=cols); te["id"] = range(900, 1200)
tr.loc[tr.index[:20], "num0"] = np.nan            # 欠損補完の経路を通す
tr.to_csv(raw / "train.csv", index=False); te.to_csv(raw / "test.csv", index=False)
pd.DataFrame({"id": te["id"], "target": 0}).to_csv(raw / "sample_submission.csv", index=False)

def patch(path, subs):
    p = work / path; s = p.read_text()
    for a, b in subs: 
        assert a in s, f"{path}: 置換対象が見つからない: {a[:40]}"
        s = s.replace(a, b)
    p.write_text(s)

patch("scripts/preprocess.py", [("NUMERIC_COLS: list[str] = []", f"NUMERIC_COLS: list[str] = {cols}")])
patch("scripts/train.py", [("FEATURES: list[str] = []", f"FEATURES: list[str] = {cols}")])

def run(*args, **kw):
    r = subprocess.run([sys.executable, "-m", *args], cwd=work, capture_output=True,
                       text=True, env={**kw.get("env", {}), "PATH": "/usr/bin:/bin",
                                       "DS_SKIP_VIZ_CHECK": "1",
                                       "PYTHONPATH": str(work)})
    print(f"  $ {' '.join(args)}  → exit {r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-2500:]); sys.exit(1)
    return r.stdout

run("scripts.preprocess")
out = run("scripts.train", "--model", "lgb")
print("   ", [l for l in out.splitlines() if "OOF" in l or "提出" in l][:4])

oof = sorted((work / "data/output/oof").glob("oof_*.npy"))
test = sorted((work / "data/output/oof").glob("test_*.npy"))
subs = sorted((work / "data/output/submissions").glob("*.csv"))
print(f"  成果物: OOF={len(oof)} / test={len(test)} / 提出CSV={len(subs)}")
assert oof and test and subs, "学習→提出の一連が途切れている"

sub = pd.read_csv(subs[0])
assert len(sub) == 300 and list(sub.columns) == ["id", "target"]
vals = sub["target"]
assert vals.between(0, 1).all() and vals.nunique() > 2, \
    f"AUC 設定なのに確率でない（ユニーク {vals.nunique()} 種）"
print(f"  提出値: 確率 {vals.min():.4f}〜{vals.max():.4f}（{vals.nunique()} 種）← ハードラベルでない")

# --resume が同じ結果で通ること
run("scripts.train", "--model", "lgb", "--resume")
run("scripts.train", "--model", "lgb", "--resume")
print("  --resume 2 回目もキャッシュ経由で完走")

# blend が train.py の出力（二値 (n,2)）を受け取れること
run("scripts.blend", "--mode", "corr", "--oofs", f"a={oof[0]}", f"b={oof[0]}")
print("\n✅ e2e 通過")
