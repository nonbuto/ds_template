"""実装に欠陥を注入して、テストが落ちるかを測る。**作業ツリーには触らない。**

**「テストが通ること」と「テストが守っていること」は別物**で、後者は欠陥を入れて初めて測れる。
L-28 で「ガードが発火するテストを書く」と結論した直後に書いたテストは**ソースの字面の grep**
で、変異注入 25 件のうち 13 件がすり抜け、欠陥 3 件が同時に存在した状態で全件緑だった（L-30）。

**なぜ複製の上で変異させるか**: 最初の実装は作業ツリーのファイルを直接書き換えていた。
`try/finally` で戻してはいたが、それでは足りない:

- **中断されると変異が残る**（このセッションで実際に 1 回発生し、`git checkout` で復旧した）
- **実行中の数分間、`src/*.py` が壊れた状態になる**。その間に別の作業
  （長時間の学習・並行して読むエージェント）が同じファイルを読むと壊れた実装を読む
- `uv run pytest` を回すたびに作業ツリーが揺れるので、**安心して並行実行できない**

`e2e` と同じく複製して実行すれば、これらは構造的に起こらない。

`tests/test_harness.py::test_mutations_are_detected` から起動される（`slow` マーカー）。
手で回す場合:

    uv run python tests/_mutation_check.py .
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()

MUTANTS = [
    ("scripts/train.py",
     'return np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)',
     'return np.asarray(model.feature_importances_, dtype=float)',
     "importance を split に戻す"),
    ("src/utils/ensemble.py",
     '            pred = cross_val_predict(pipe, X, y, cv=get_cv(), method="predict_proba")[:, 1]',
     '            pred = pipe.fit(X, y).predict_proba(X)[:, 1]',
     "signed_stack を in-sample に戻す"),
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
    # 今回追加した定石モジュール（L-35）。新しいコードほどテストの実効性が未検証
    ("src/utils/encoders.py",
     "                mapping = _smoothed_map(keys_tr.iloc[tr_idx], tgt[tr_idx], prior, smoothing)",
     "                mapping = _smoothed_map(keys_tr, tgt, prior, smoothing)",
     "target encoding を全 train で集計（リーク）"),
    ("src/utils/pseudo.py",
     "    model, _ = train_fn(X_tr, y_tr, X_tr, y_tr, params)",
     "    model, _ = train_fn(X_tr, y_tr, X_test.iloc[:1], None, params)",
     "pseudo の生成元を変える"),
    ("src/noise.py",
     "    if se < ZERO_FLOOR_EPS:",
     "    if False:",
     "床がゼロでも判定を出す"),
    ("src/utils/postprocess.py",
     '        if EVAL_METRIC.lower() == "auc":',
     "        if True:",
     "AUC 以外でも rank 変換する"),
]


def run_tests(work: Path) -> bool:
    """複製の中でテストを回す（作業ツリーの venv をそのまま使う）。"""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-x", "-m", "not slow",
         "-p", "no:randomly"],
        cwd=work, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(work),
             "DS_SKIP_VIZ_CHECK": "1"},
    )
    return r.returncode == 0


def main() -> int:
    work = Path(tempfile.mkdtemp()) / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
        # `.venv` の完全一致だと **`.venv-autogluon`（580 MB）が毎回コピーされる**。
        # 実測: 504 MB / 1.8 秒 → 1.8 MB / 0.0 秒
        "__pycache__", "*.pyc", ".git", ".venv*", "kaggle_nb", "data", "*.db"))

    print(f"  {'検知':<6} {'変異':<40}")
    print("  " + "-" * 50)
    all_caught = True
    try:
        for rel, old, new, label in MUTANTS:
            p = work / rel
            orig = p.read_text(encoding="utf-8")
            if old not in orig:
                # 置換対象が消えた = 実装が変わった。**変異が「効かない」ことを合格にしない**
                # —— それを許すと、検査対象が消えたことに気づけない（L-28 #2 と同じ構造）
                print(f"  {'❌ 対象なし':<6} {label:<40} ← 置換文字列が実装に無い")
                all_caught = False
                continue
            try:
                p.write_text(orig.replace(old, new, 1), encoding="utf-8")
                caught = not run_tests(work)
            finally:
                p.write_text(orig, encoding="utf-8")
            all_caught = all_caught and caught
            print(f"  {'✅' if caught else '❌ 素通り':<6} {label:<40}")
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)

    # 呼び出し側（テスト）が結果を判定できるよう、1 件でもすり抜けたら異常終了する
    return 0 if all_caught else 1


if __name__ == "__main__":
    sys.exit(main())
