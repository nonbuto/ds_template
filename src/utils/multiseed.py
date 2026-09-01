"""multi-seed avg アンサンブルの実行ヘルパー（既存 seed 結果の再利用つき）。

**なぜ再利用が要るか**: multi-seed avg5 は「基本 seed（`RANDOM_STATE`）を含む 5 seed」で
構成されることが多いが、基本 seed の学習は単体モデルの実験で**既に実行済み**であることが
ほとんど。それを毎回再学習すると、1 回の avg5 につき 1/5（NN 系なら十数分〜数十分）を無駄に使う。
既存の `oof_{tag}_s{seed}.npy` / `test_{tag}_s{seed}.npy` があれば読み込み、残り 4 seed だけを回す。

使い方:
    from src.utils.multiseed import run_multiseed

    result = run_multiseed(
        train_fn=lambda seed: my_train(X, y, X_test, seed=seed),   # (oof, test) を返す
        tag="lgb_h012",
        reuse_base_seed=True,      # oof_{tag}_s42.npy があれば再利用
    )
    result.oof, result.test        # avg5 の平均済み予測
    result.seed_scores             # seed ごとの OOF スコア（渡した場合）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from src.config import OOF_DIR, RANDOM_STATE

# 基本 seed を先頭に置く（再利用の対象は先頭 = RANDOM_STATE）
DEFAULT_SEEDS: tuple[int, ...] = (RANDOM_STATE, 0, 1, 7, 2026)


@dataclass
class MultiSeedResult:
    oof: np.ndarray
    test: np.ndarray
    seeds: list[int]
    reused_seeds: list[int] = field(default_factory=list)
    seed_scores: list[float] = field(default_factory=list)


def _seed_paths(tag: str, seed: int) -> tuple:
    return OOF_DIR / f"oof_{tag}_s{seed}.npy", OOF_DIR / f"test_{tag}_s{seed}.npy"


def run_multiseed(
    train_fn: Callable[[int], tuple[np.ndarray, np.ndarray]],
    tag: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    reuse_base_seed: bool = True,
    score_fn: Optional[Callable[[np.ndarray], float]] = None,
    save_per_seed: bool = True,
    verbose: bool = True,
) -> MultiSeedResult:
    """seed を変えて学習し、OOF / test 予測を平均する。既存 seed 結果は再利用する。

    Args:
        train_fn: `seed` を受け取り `(oof, test)` を返す関数。
        tag: 保存・再利用に使う識別子（`oof_{tag}_s{seed}.npy` の形で保存される）。
        seeds: 使用する seed 列。既定は `(RANDOM_STATE, 0, 1, 7, 2026)`。
        reuse_base_seed: True なら**既に保存済みの seed 結果を読み込んで再学習を省く**。
            部分的に保存済みでも、存在するものだけ再利用して残りを学習する。
        score_fn: OOF 配列を受け取りスコアを返す関数（渡すと seed ごとのスコアを記録）。
        save_per_seed: seed ごとの予測を保存するか（次回の再利用のため既定 True）。

    Returns:
        MultiSeedResult: 平均済み `oof` / `test`、再利用した seed の一覧など。

    Note:
        再利用は「**同じ特徴量セット・同じ HP** で作られた結果である」ことが前提。
        特徴量や HP を変えたら `tag` も変えること（古い結果の混入は `G-FAIR` 違反になる）。
    """
    oofs: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    reused: list[int] = []
    scores: list[float] = []

    for seed in seeds:
        oof_path, test_path = _seed_paths(tag, seed)
        if reuse_base_seed and oof_path.exists() and test_path.exists():
            oof, test = np.load(oof_path), np.load(test_path)
            reused.append(seed)
            if verbose:
                print(f"  seed={seed:>5}  既存結果を再利用（学習スキップ）")
        else:
            if verbose:
                print(f"  seed={seed:>5}  学習中...", flush=True)
            oof, test = train_fn(seed)
            if save_per_seed:
                np.save(oof_path, oof)
                np.save(test_path, test)

        oofs.append(oof)
        tests.append(test)
        if score_fn is not None:
            s = score_fn(oof)
            scores.append(s)
            if verbose:
                print(f"  seed={seed:>5}  OOF={s:.5f}")

    avg_oof = np.mean(oofs, axis=0)
    avg_test = np.mean(tests, axis=0)

    if verbose:
        saved = len(reused)
        print(f"\n  avg{len(seeds)} 完了（{saved} seed を再利用し {len(seeds) - saved} seed を学習）")
        if score_fn is not None:
            print(f"  単一seed平均: {np.mean(scores):.5f}   avg{len(seeds)}: {score_fn(avg_oof):.5f}")

    return MultiSeedResult(
        oof=avg_oof, test=avg_test, seeds=list(seeds),
        reused_seeds=reused, seed_scores=scores,
    )
