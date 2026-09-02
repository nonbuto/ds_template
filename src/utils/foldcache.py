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

import os

import numpy as np

from src.config import OOF_DIR


def _signature_hash(signature: object) -> str:
    """条件（特徴量リスト・HP 等）を 8 桁の短いハッシュにする。

    JSON にできないオブジェクトは `repr` で落とす（型が変われば文字列も変わるので、
    「条件が変わったことを検知する」目的には足りる）。
    """
    import hashlib
    import json

    try:
        text = json.dumps(signature, sort_keys=True, default=repr, ensure_ascii=False)
    except TypeError:
        text = repr(signature)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


class FoldCache:
    """fold ごとの val 予測・test 予測をディスクに保存し、再実行時に再利用する。

    Attributes:
        tag: 特徴量セットと HP を一意に表す識別子。
            **条件が変われば必ず変える**（古い fold が混ざると不公正比較になる → `G-FAIR`）。
        seed: 乱数シード。`multiseed.run_multiseed()` と組み合わせて使える。
        signature: 特徴量リストや HP など、**条件が変わったことを機械が判定できる材料**。
            渡すとその内容のハッシュが tag に足される。

    Note:
        呼び出し側は tag を `f"{model}_{len(FEATURES)}f"` の形で作りがちだが、
        それはモデル名と**特徴量の本数**しか区別しない。列を入れ替えても、HP を変えても、
        本数が同じなら**前回の予測がそのまま再利用される** —— `--resume` を付けた瞬間に
        別条件の結果が混ざり、しかも表示上は普通に完走する。
        `signature` を渡せば条件が変わった時点で別のキャッシュになる。
    """

    def __init__(self, tag: str, seed: int, n_splits: int,
                 cache_dir: Optional[Path] = None, enabled: bool = True,
                 signature: Optional[object] = None) -> None:
        if signature is not None:
            tag = f"{tag}_{_signature_hash(signature)}"
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
        """保存済みなら `(val 予測, test 予測)` を返す。無ければ None。

        **壊れたファイルは「無い」として扱う。** このモジュールの存在理由は
        「kill・クラッシュ・打ち切りで fold の計算が失われるのを防ぐ」ことなのに、
        保存の最中に落ちると中途半端な .npy が残り、次の `--resume` が
        `ValueError: EOF: reading array header` で落ちていた（復旧手段は手動削除だけ）。
        **自分が守るはずの事故で自分が壊れる**のは避ける。
        """
        if not self.enabled:
            return None
        va, te = self._path(fold, "val"), self._path(fold, "test")
        if not (va.exists() and te.exists()):
            return None
        try:
            return np.load(va), np.load(te)
        except (ValueError, OSError, EOFError) as exc:
            print(f"  ⚠️ foldcache[{self.tag}] fold {fold} が壊れています（{type(exc).__name__}）。"
                  "この fold は学習し直します")
            for path in (va, te):
                path.unlink(missing_ok=True)
            return None

    def save(self, fold: int, val_pred: np.ndarray, test_pred: np.ndarray) -> None:
        """fold の予測を保存する。**fold ループの中で毎回呼ぶこと。**

        一時ファイルに書いてから `os.replace` で差し替える（`csvlock` と同じ理由）。
        `np.save` は非原子的なので、途中で落ちると壊れたファイルが「存在する」状態で残る。
        """
        if not self.enabled:
            return
        for path, arr in ((self._path(fold, "val"), val_pred),
                          (self._path(fold, "test"), test_pred)):
            # `np.save` は拡張子が .npy でないと勝手に付け足すので、一時名も .npy で終える
            tmp = path.with_name(f".{path.name}.tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, path)

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
