"""実装に欠陥を注入して、テストが落ちるかを測る（必ず元に戻す）。

**「テストが通ること」と「テストが守っていること」は別物**で、後者は欠陥を入れて初めて測れる。
L-28 で「ガードが発火するテストを書く」と結論した直後に書いたテストは**ソースの字面の grep**
で、変異注入 25 件のうち 13 件がすり抜け、欠陥 3 件が同時に存在した状態で全件緑だった（L-30）。

`tests/test_harness.py::test_mutations_are_detected` から起動される（`slow` マーカー）。
手で回す場合:

    uv run python tests/_mutation_check.py .

**変異は必ず `try/finally` で元に戻す。** 途中で落ちたときのために、
実行後は `git status` がクリーンであることを確認すること。
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
MUTANTS = [
    ("scripts/train.py",
     'return np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)',
     'return np.asarray(model.feature_importances_, dtype=float)',
     "importance を split に戻す"),
    ("src/utils/ensemble.py",
     'train_preds = cross_val_predict(_pipe(), X_meta_train, y_train,\n                                    cv=get_cv(), method="predict_proba")[:, 1]',
     'train_preds = _pipe().fit(X_meta_train, y_train).predict_proba(X_meta_train)[:, 1]',
     "stacking を in-sample に戻す"),
    ("scripts/preprocess.py",
     "test[NUMERIC_COLS] = test[NUMERIC_COLS].fillna(medians)",
     "test[NUMERIC_COLS] = test[NUMERIC_COLS].fillna(test[NUMERIC_COLS].median())",
     "test を test 自身の中央値で補完"),
    ("src/experiment.py",
     "                raise RuntimeError(",
     "                print(",
     "可視化ガードを警告に格下げ"),
    ("src/utils/csvlock.py",
     "        os.replace(tmp, path)",
     "        import shutil as _sh; _sh.copyfile(tmp, path); Path(tmp).unlink()",
     "原子的書き戻しをやめる"),
    ("src/utils/ensemble.py",
     "    weights = [_one(x0, seed + i) for i, x0 in enumerate(starts)]",
     "    weights = [_one(starts[0], seed)]",
     "n_seeds を無視する"),
    ("scripts/train.py",
     "    oof_score = metric(y[covered], shape_for_metric(oof_preds[covered]))",
     "    oof_score = metric(y, shape_for_metric(oof_preds))",
     "covered を評価に使わない（run_cv）"),
]

def run_tests() -> bool:
    r = subprocess.run(["uv", "run", "pytest", "tests/", "-q", "-x", "-m", "not slow", "-p", "no:randomly"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0

print(f"  {'検知':<6} {'変異':<40}")
print("  " + "-" * 50)
all_caught = True
for rel, old, new, label in MUTANTS:
    p = ROOT / rel
    orig = p.read_text()
    if old not in orig:
        # 置換対象が消えた = 実装が変わった。**変異が「効かない」ことを合格にしない** ——
        # それを許すと、検査対象が消えたことに気づけない（L-28 #2 と同じ構造）
        print(f"  {'❌ 対象なし':<6} {label:<40} ← 置換文字列が実装に無い")
        all_caught = False
        continue
    try:
        p.write_text(orig.replace(old, new, 1))
        caught = not run_tests()
    finally:
        p.write_text(orig)
    all_caught = all_caught and caught
    print(f"  {'✅' if caught else '❌ 素通り':<6} {label:<40}")

# 呼び出し側（テスト）が結果を判定できるよう、1 件でもすり抜けたら異常終了する
sys.exit(0 if all_caught else 1)
