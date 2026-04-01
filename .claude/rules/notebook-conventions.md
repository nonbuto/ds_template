---
description: スクリプト作成・編集規約（marimo廃止・通常.pyのみ使用）
---

# スクリプト規約

## スクリプトの種類と置き場所

| 種類 | 場所 | 命名 | 説明 |
|---|---|---|---|
| **汎用骨格スクリプト** | `scripts/` | `動詞.py`（例: `train.py`） | テンプレート本体。コンペ開始時に TODO を埋めて使う |
| **実験スクリプト** | `experiments/runs/` | `exp{NNN}_s{stage}_{内容}.py` | コンペ固有の1回限りスクリプト |
| **可視化スクリプト** | `scripts/visualize.py` | （固定） | 画像を `data/output/plots/` に保存。Claude が Read で読む |

## スクリプトの標準構成

```python
"""
スクリプトの説明

使い方:
    uv run python scripts/xxx.py --option value
"""

# 1. 標準ライブラリ
# 2. サードパーティ
# 3. ローカル（src.*）— パスは必ず src.config からインポート

from src.config import PROCESSED_DATA_DIR, TARGET_COL, RANDOM_STATE

# ──────────────────────────────────────────────
# TODO: コンペごとにここを変更する
# ──────────────────────────────────────────────
FEATURES: list[str] = []  # コンペ固有の特徴量リスト

# メイン処理
def main():
    ...

if __name__ == "__main__":
    main()
```

## データ読み込み・保存

```python
import pandas as pd
from src.config import PROCESSED_DATA_DIR, OOF_DIR

# 読み込み
train = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")

# 保存（役割別サブディレクトリを使う）
import numpy as np
np.save(OOF_DIR / "oof_042_lgb.npy", oof_preds)
```

## 可視化の規約

```python
import matplotlib
matplotlib.use("Agg")  # 非インタラクティブ（ファイル保存のみ）
import matplotlib.pyplot as plt
from src.config import PLOTS_DIR

fig, ax = plt.subplots(figsize=(10, 4))
# ... 描画 ...
fig.savefig(PLOTS_DIR / "eda_tenure_target_dist.png", dpi=120, bbox_inches="tight")
plt.close(fig)
```

- 画像は `data/output/plots/` に保存する（直接表示しない）
- Claude が `Read` ツールで読んで対話に使う
- ファイル名: `{prefix}_{変数名}_{テーマ}.png`

## スクリプトの実行

```bash
# 通常実行
uv run python scripts/train.py

# 引数付き
uv run python scripts/train.py --model lgb --params data/output/params/best_params_lgb_working.json

# 実験スクリプト（experiments/runs/ 内）
uv run python experiments/runs/exp042_s4_fe_tenure.py
```
