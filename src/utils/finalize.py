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

from src.config import OOF_DIR, RAW_DATA_DIR, TARGET_COL, submission_path

ID_COL = "id"


@dataclass
class RunOutputs:
    oof: Path
    test: Path
    submission: Optional[Path]


def _to_submission_values(test: np.ndarray, submit: str) -> np.ndarray:
    """test 予測配列を提出列の値に変換する。"""
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
    submit: Literal["label", "proba"] = "label",
    make_submission: bool = True,
    is_ensemble: bool = False,
) -> RunOutputs:
    """OOF・test 予測・提出 CSV をまとめて保存する。

    Args:
        exp_id: 実験 ID（`ExperimentTracker` が採番したもの）。
        model: モデル識別子（例 `lgb_h012`）。ファイル名に入る。
        oof / test: 予測配列。**test を省いた実験は再学習を招くので必ず渡す。**
        oof_score: OOF スコア（提出ファイル名に埋め込む）。
        submit: `"label"` = argmax してクラスラベルを提出（多クラス精度系）。
            `"proba"` = 陽性確率をそのまま提出（AUC 系）。
        make_submission: False にすると CSV を作らない。**ΔOOF スクリーニング専用の
            実験でのみ使う**（提出候補になりうる実験では必ず True のままにする）。
        is_ensemble: 派生アンサンブルなら True。ファイル名に `_ens_` を入れて
            プールの自己参照混入を防ぐ（→ `CONVENTIONS.md` の OOF 命名規約）。

    Returns:
        RunOutputs: 保存した 3 つのパス（submission を作らない場合は None）。
    """
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{'ens_' if is_ensemble else ''}{model}"
    oof_path = OOF_DIR / f"oof_{exp_id}_{tag}.npy"
    test_path = OOF_DIR / f"test_{exp_id}_{tag}.npy"
    np.save(oof_path, oof)
    np.save(test_path, test)

    sub_path: Optional[Path] = None
    sample_path = RAW_DATA_DIR / "sample_submission.csv"
    if make_submission and not sample_path.exists():
        # ここで例外を投げると、学習が終わっているのに成果物を失う。
        # npy は保存済みなので、提出 CSV だけ諦めて続行する。
        print(f"⚠️ {sample_path} が無いため提出 CSV は作りません（OOF / test は保存済み）")
        make_submission = False
    if make_submission:
        sample = pd.read_csv(sample_path)
        sub = pd.DataFrame({
            ID_COL: sample[ID_COL],
            TARGET_COL: _to_submission_values(test, submit),
        })
        sub_path = submission_path(model=tag, oof_score=oof_score, exp_id=exp_id)
        sub.to_csv(sub_path, index=False)

    print(
        f"\n💾 成果物を保存しました（学習→提出まで 1 フロー）\n"
        f"  OOF : {oof_path.name}\n"
        f"  test: {test_path.name}\n"
        f"  提出: {sub_path.name if sub_path else '（未作成: make_submission=False）'}"
    )
    return RunOutputs(oof=oof_path, test=test_path, submission=sub_path)
