"""学習の成果物（OOF / test 予測 / 提出ファイル）を 1 回の呼び出しで出し切るヘルパー。

**なぜ必要か**: FE 実験などで「学習して OOF だけ見て終わり」にすると、後からその構成を
提出したくなったときに**同じ学習をもう一度回す**ことになる（過去コンペで多発）。
CLAUDE.md `G-STEPWISE` の「学習 → OOF + test 予測 → 提出ファイル」を 1 つの流れにするため、
実験スクリプトの最後は原則この関数 1 本で締める。

使い方:
    from src.utils.finalize import save_run_outputs

    paths = save_run_outputs(
        exp_id="123", model="lgb_h012",
        oof=oof_preds, test=test_preds, oof_score=oof_score,
    )
    # paths.oof / paths.test / paths.submission
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import ID_COL, OOF_DIR, RAW_DATA_DIR, TARGET_COL, submission_path
from src.metrics import is_regression, needs_proba


@dataclass
class RunOutputs:
    oof: Path
    test: Path
    submission: Optional[Path]


def _assert_row_alignment(n_sample: int, test: np.ndarray) -> None:
    """test 予測の行数が `sample_submission.csv` と一致することを確かめる。

    提出 CSV は `sample[ID_COL]` と test 予測を**位置で**貼り合わせている。
    前処理に `sort_values` / `merge` / 重複除去が入ると行の対応が崩れ、
    **エラーを出さないまま全行ずれた提出ファイル**ができる。スコアは出るので気づけない
    （`PLAYBOOK.md` の L-29 が並べた「静かに間違う」型）。
    行数の一致は最低限の防波堤で、これだけでも並べ替え以外の事故は止まる。
    """
    n_test = len(np.asarray(test))
    if n_test != n_sample:
        raise ValueError(
            f"test 予測の行数（{n_test:,}）が sample_submission.csv（{n_sample:,}）と一致しません。\n"
            "   前処理で行を落とした・並べ替えた可能性があります。"
            "提出ファイルは ID と予測を位置で貼り合わせるため、このまま出すと全行ずれます。"
        )


def _resolve_submit_mode(submit: str) -> str:
    """`"auto"` を、評価指標とタスク種別から具体的な提出形式に落とす。

    **なぜ auto が既定か**: 以前の既定は `"label"` で、しかも呼び出し側は誰も指定していなかった。
    つまり AUC コンペでも**ハードラベル（0/1）を提出していた**。AUC は順序の指標なので、
    確率を 2 値に潰すと情報が消える —— 過去コンペのデータで実測すると **AUC −0.074**。
    しかもスコアは出るのでエラーにならず、**気づかないまま提出枠を消費する**。
    指標が確率を要るかは `src/metrics.py` が既に知っているので、そこから決める。
    """
    if submit != "auto":
        return submit
    if is_regression():
        return "value"
    return "proba" if needs_proba() else "label"


def _to_submission_values(test: np.ndarray, submit: str) -> np.ndarray:
    """test 予測配列を提出列の値に変換する。"""
    if submit == "value":
        # 回帰: 予測値をそのまま提出する
        return np.asarray(test).ravel()
    if submit == "proba":
        # 二値分類（AUC 等）: 陽性クラスの確率をそのまま提出する
        return test[:, 1] if test.ndim == 2 and test.shape[1] == 2 else test.ravel()
    # ラベル提出: argmax → 元のクラスラベルへ復元
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(pd.read_csv(RAW_DATA_DIR / "train.csv")[TARGET_COL])
    return le.inverse_transform(np.argmax(test, axis=1))


def save_run_outputs(
    exp_id: str,
    model: str,
    oof: np.ndarray,
    test: np.ndarray,
    oof_score: float,
    submit: Literal["auto", "label", "proba", "value"] = "auto",
    make_submission: bool = True,
    is_ensemble: bool = False,
    save_npy: bool = True,
) -> RunOutputs:
    """OOF・test 予測・提出 CSV をまとめて保存する。

    Args:
        exp_id: 実験 ID（`ExperimentTracker` が採番したもの）。
        model: モデル識別子（例 `lgb_h012`）。ファイル名に入る。
        oof / test: 予測配列。**test を省いた実験は再学習を招くので必ず渡す。**
        oof_score: OOF スコア（提出ファイル名に埋め込む）。
        submit: `"auto"`（既定）= `EVAL_METRIC` / `PROBLEM_TYPE` から決める。
            `"label"` = argmax してクラスラベルを提出（accuracy・f1 等）。
            `"proba"` = 陽性確率をそのまま提出（AUC・logloss）。`"value"` = 回帰の予測値。
        make_submission: False にすると CSV を作らない。**ΔOOF スクリーニング専用の
            実験でのみ使う**（提出候補になりうる実験では必ず True のままにする）。
        is_ensemble: 派生アンサンブルなら True。ファイル名に `_ens_` を入れて
            プールの自己参照混入を防ぐ（→ `CONVENTIONS.md` の OOF 命名規約）。
        save_npy: False にすると npy を書かない（既存の npy から提出だけ作る場合）。

    Returns:
        RunOutputs: 保存した 3 つのパス（submission を作らない場合は None）。
    """
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{'ens_' if is_ensemble else ''}{model}"
    oof_path = OOF_DIR / f"oof_{exp_id}_{tag}.npy"
    test_path = OOF_DIR / f"test_{exp_id}_{tag}.npy"
    if save_npy:
        np.save(oof_path, oof)
        np.save(test_path, test)

    sub_path: Optional[Path] = None
    sample_path = RAW_DATA_DIR / "sample_submission.csv"
    if make_submission and not sample_path.exists():
        # ここで例外を投げると、学習が終わっているのに成果物を失う。
        # npy は保存済みなので、提出 CSV だけ諦めて続行する。
        print(f"⚠️ {sample_path} が無いため提出 CSV は作りません（OOF / test は保存済み）")
        make_submission = False
    mode = _resolve_submit_mode(submit)
    if make_submission:
        sample = pd.read_csv(sample_path)
        _assert_row_alignment(len(sample), test)
        sub = pd.DataFrame({
            ID_COL: sample[ID_COL],
            TARGET_COL: _to_submission_values(test, mode),
        })
        sub_path = submission_path(model=tag, oof_score=oof_score, exp_id=exp_id)
        sub.to_csv(sub_path, index=False)

    print(
        f"\n💾 成果物を保存しました（学習→提出まで 1 フロー）\n"
        f"  OOF : {oof_path.name}\n"
        f"  test: {test_path.name}\n"
        f"  提出: {sub_path.name if sub_path else '（未作成: make_submission=False）'}"
        f"{f'  ← 形式={mode}' if sub_path else ''}"
    )
    return RunOutputs(oof=oof_path, test=test_path, submission=sub_path)
