---
name: kaggle-research
description: FE棄却が3連続したとき・Stage 6 で外部公開予測を row-level voting に活用したいとき・「自分が思いつかない角度を探したい」ときに呼ぶ。Kaggle Discussion・上位カーネル・公開データセットを CLI で系統的に調査して競合アプローチを自前実験の仮説に接続する。「飽和した」と言う前に必ずこのスキルを実行すること。
disable-model-invocation: true
---

# /kaggle-research スキル

## このスキルの役割

LB 上位者のコードや公開予測ファイルを系統的に収集・分析する。
「なぜあのスコアが出るのか」を逆算して、自前の実験仮説に接続することが目的。

**いつ使うか:**
- FE 棄却が3連続した後（飽和宣言する前に他の参加者の知見を確認する）
- Stage 6 で外部公開予測を row-level voting に活用したいとき
- 「自分が思いつかない角度」を探したいとき

---

## 実行手順

### フェーズ1: カーネル・ノートブックの調査

```bash
# 1. コンペ名でカーネルを投票数順に一覧取得
kaggle kernels list --search "<コンペ名>" --sort-by voteCount 2>&1 | head -30

# 2. 特定ユーザーのカーネル一覧
kaggle kernels list --user <username> 2>&1

# 3. カーネルをローカルに取得（.ipynb / .py をダウンロード）
kaggle kernels pull <username>/<kernel-slug> -p /tmp/kaggle_<slug>

# 4. 複数カーネルをまとめて取得したい場合
for slug in user1/kernel1 user2/kernel2; do
  kaggle kernels pull $slug -p /tmp/kaggle_$(echo $slug | tr '/' '_')
done
```

**カーネル内容の解析ポイント（ノートブック）:**
```python
import json
with open('/tmp/kaggle_<slug>/<slug>.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells']:
    src = ''.join(cell['source'])
    if src.strip():
        print(f'=== {cell["cell_type"]} ===')
        print(src[:800])
```

### フェーズ2: 公開データセットの調査・取得

```bash
# 1. データセット一覧（キーワード検索）
kaggle datasets list --search "<コンペ略称>" --sort-by voteCount 2>&1 | head -20

# 2. 特定ユーザーのデータセット一覧
kaggle datasets list --user <username> 2>&1

# 3. データセットのメタ情報確認（ファイル名・サイズ）
kaggle datasets files <username>/<dataset-slug> 2>&1

# 4. データセットをダウンロード（zip で落ちる → Python で unzip）
kaggle datasets download <username>/<dataset-slug> -p /tmp/<dirname>

# unzip（Python 経由が確実）
python3 -c "
import zipfile, os
z = zipfile.ZipFile('/tmp/<dirname>/<slug>.zip')
print(z.namelist())   # まずファイル一覧確認
z.extractall('/tmp/<dirname>')
"

# 5. CSV の先頭を確認
head -3 /tmp/<dirname>/*.csv
```

### フェーズ3: ディスカッションの検索

```bash
# ディスカッション一覧（コンペ slug 確認は kaggle competitions list で）
# ※ Kaggle CLI にはディスカッション取得コマンドがないため WebSearch を使う
# → WebSearch: site:kaggle.com/competitions/<slug>/discussion <キーワード>
```

**ディスカッション調査の代替手段（CLI 不可の場合）:**
- `WebSearch: kaggle <competition-slug> discussion "<スコア値>"` で Google 経由検索
- カーネルのマークダウンセルに Discussion URL が記載されていることが多い → そこから辿る

### フェーズ4: 公開予測ファイルを活用する場合

公開されている予測 CSV（submission形式）をブレンドする場合の手順:

```python
import pandas as pd
import numpy as np

# 1. 公開予測の読み込み
sub_external = pd.read_csv('/tmp/<dirname>/<filename>.csv')

# 2. 自前予測との一致率確認
sub_own = pd.read_csv('data/output/submissions/<own_sub>.csv')
merged = pd.merge(sub_external, sub_own, on='id', suffixes=('_ext', '_own'))
agree_rate = (merged['target_ext'] == merged['target_own']).mean()
print(f"Agreement rate: {agree_rate:.4f}")

# 3. 不一致行を確認
disagree = merged[merged['target_ext'] != merged['target_own']]
print(f"Disagreement rows: {len(disagree)}")
print(disagree['target_ext'].value_counts())
print(disagree['target_own'].value_counts())

# 4. Row-level voting（2モデル合意 → そちら採用、不一致 → 第3モデルで決着）
def vote3(row):
    if row['pred_a'] == row['pred_b']:
        return row['pred_a']
    elif row['pred_a'] == row['pred_c']:
        return row['pred_a']
    elif row['pred_b'] == row['pred_c']:
        return row['pred_b']
    else:
        return row['pred_c']  # 全不一致時のフォールバック
```

---

## 調査フロー（標準手順）

```
1. LB スコア上位者を特定
   └── kaggle kernels list --search <comp> --sort-by voteCount

2. 注目カーネルを pull
   └── kaggle kernels pull <user>/<slug> -p /tmp/

3. カーネルが参照しているデータセットを確認
   └── カーネル内の path 変数や import を見る（例: /kaggle/input/datasets/<user>/<slug>）

4. 関連データセットを調査・DL
   └── kaggle datasets download <user>/<slug> -p /tmp/

5. ファイル内容を分析
   └── 特徴量エンジニアリングの違い・公開予測 CSV の利用可否を判断

6. 新発見を SESSION.md の「未解決の問い」または FE_HYPOTHESES.md に記録
   └── 「なぜそのスコアが出るか」のメカニズムを1〜3行で言語化
```

---

## 注意事項

- **公開予測 CSV の利用はKaggleルール上 OK**（Public Dataset として共有されたもの）
- ローカルの `/tmp/` キャッシュは再起動で消える → 再利用時は再 DL する
- カーネルが参照している Dataset slug は `path` 変数や `pd.read_csv(...)` パスから確認できる
  - 例: `/kaggle/input/datasets/<user>/<slug>/` → `<user>/<slug>`
- `kaggle datasets download` は zip で落ちる。`--unzip` フラグは古いバージョンでは使えないため Python の `zipfile` で解凍する
- 公開予測を pseudo source に使うと自前モデルが蒸留になりうる（CLAUDE.md の「Pseudo 源泉の品質とリーク診断」を参照）
