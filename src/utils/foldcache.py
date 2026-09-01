"""fold 単位のチェックポイント（中断した学習を途中から再開する）。

`src/utils/multiseed.py` の「既存 seed 結果を再利用する」を **1 段下げた**もの。
seed 単位では救えない中断（kill・クラッシュ・見積もり外れによる打ち切り・
締切に追われての中断）で、それまでに終わった fold の計算がすべて失われるのを防ぐ。

過去コンペでは 4 時間超まわした学習を fold 4/25 の時点で打ち切ることになり、
その 4 fold 分の計算が丸ごと消えた。fold ごとの保存はコストがほぼゼロなのに、
やり直しは学習コスト全額を払い直す — `finalize.py` と同じ非対称性がここにもある。

使い方:
    from src.utils.foldcache import FoldCache

    cache = FoldCache(tag="lgb_h012", seed=42, n_splits=5)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        cached = cache.load(fold)
        if cached is not None:
            oof[va_idx], test_parts = cached           # 学習をスキップ
            continue
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        va_pred = model.predict_proba(X.iloc[va_idx])
        te_pred = model.predict_proba(X_test)
        cache.save(fold, va_pred, te_pred)             # ここで中断しても次回は再開できる
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.config import OOF_DIR


class FoldCache:
    """fold ごとの val 予測・test 予測をディスクに保存し、再実行時に再利用する。

    Attributes:
        tag: 特徴量セットと HP を一意に表す識別子。
            **条件が変われば必ず変える**（古い fold が混ざると不公正比較になる → `G-FAIR`）。
        seed: 乱数シード。`multiseed.run_multiseed()` と組み合わせて使える。
    """

    def __init__(self, tag: str, seed: int, n_splits: int,
                 cache_dir: Optional[Path] = None, enabled: bool = True) -> None:
        self.tag = tag
        self.seed = seed
        self.n_splits = n_splits
        self.enabled = enabled
        self.dir = Path(cache_dir) if cache_dir else OOF_DIR / "_foldcache"
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, fold: int, kind: str) -> Path:
        return self.dir / f"{kind}_{self.tag}_s{self.seed}_f{fold}.npy"

    def load(self, fold: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """保存済みなら `(val 予測, test 予測)` を返す。無ければ None。"""
        if not self.enabled:
            return None
        va, te = self._path(fold, "val"), self._path(fold, "test")
        if va.exists() and te.exists():
            return np.load(va), np.load(te)
        return None

    def save(self, fold: int, val_pred: np.ndarray, test_pred: np.ndarray) -> None:
        """fold の予測を保存する。**fold ループの中で毎回呼ぶこと。**"""
        if not self.enabled:
            return
        np.save(self._path(fold, "val"), val_pred)
        np.save(self._path(fold, "test"), test_pred)

    def completed_folds(self) -> list[int]:
        """既に計算済みの fold 番号を返す。"""
        if not self.enabled:
            return []
        return sorted(f for f in range(self.n_splits) if self.load(f) is not None)

    def report(self) -> str:
        done = self.completed_folds()
        if not done:
            return f"  foldcache[{self.tag} s{self.seed}]: 再利用可能な fold なし（最初から学習）"
        return (f"  foldcache[{self.tag} s{self.seed}]: "
                f"fold {done} を再利用（{len(done)}/{self.n_splits} スキップ）")

    def clear(self) -> int:
        """この tag / seed のキャッシュを削除する。戻り値は削除したファイル数。"""
        paths = list(self.dir.glob(f"*_{self.tag}_s{self.seed}_f*.npy"))
        for p in paths:
            p.unlink()
        return len(paths)
