---
name: ds-kickoff
description: コンペ参加直後に1回だけ呼ぶ。データを見る前に「そのデータが何者か」を理解するためのスキル。評価指標・データ種別（合成/実データ）・外部データ有無・CV設計の初期判断を COMPETITION.md と src/config.py に記録する。EDA や最初の実験を始める前に必ず実施すること。
argument-hint: "<コンペ名 or URL>"
---

# /ds-kickoff スキル

## このスキルの役割

EDAの前に「そのデータが何者か」を理解する。

EDAは手元のデータを見る作業だが、kickoff はデータが生まれた文脈を理解する作業。
この順序を守ることで、EDA・FEの方針が「思いつきの探索」ではなく「文脈に基づく問い」になる。

**このスキルはコンペ参加直後に一度だけ実行する。**

---

## 実行手順

### フェーズ0: 環境確認とデータの存在確認（セーフティネット）

フェーズ1の前に、実行環境と `data/raw/` を確認する。

**環境確認（ローカル vs Kaggle Notebook）:**

```python
from src.config import IS_KAGGLE, RAW_DATA_DIR
print(f"IS_KAGGLE={IS_KAGGLE}, RAW_DATA_DIR={RAW_DATA_DIR}")
```

- **IS_KAGGLE=True（Kaggle Notebook）**: `/kaggle/input/` にコンペデータが自動マウントされる。ダウンロード不要。フェーズ1へ進む。
- **IS_KAGGLE=False（ローカル）**: `data/raw/` を確認してダウンロードが必要か判断する。

**ローカル環境でのデータ確認:**

```bash
ls data/raw/ 2>/dev/null
```

- **train/test 等が存在する** → そのままフェーズ1へ
- **空 or 存在しない** → `src/config.py` の `COMPETITION` を読み、ダウンロードを提案する:

  ```
  data/raw/ が空です。COMPETITION = "<slug>" のデータをダウンロードしますか？
    A) 容量確認してからダウンロード（kaggle competitions files -c <slug>）
    B) そのままダウンロード
    C) スキップ（後で手動）
  ```

  実行コマンド:
  ```bash
  uv run kaggle competitions download -c <slug> -p data/raw/
  cd data/raw/ && unzip -q "*.zip" 2>/dev/null; cd -
  ```

> `COMPETITION` がまだプレースホルダー（`your-competition-name`）の場合は、
> 先にユーザーへコンペスラッグを確認してから `src/config.py` を更新する。

**ハードウェア確認（長時間実験の実行環境計画に使う）:**

```bash
uv run python -c "
import platform
print(platform.machine(), platform.system())
try:
    import torch
    print('cuda:', torch.cuda.is_available(),
          '/ mps:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
except ImportError:
    print('torch 未導入（NN系を使う段階で確認）')
"
```

→ 結果（CUDA/MPS/CPU・GPU有無）を `COMPETITION.md` の「実行環境」に記録する
→ ローカルGPUが無い・弱い場合、NN 系や multi-fold × multi-seed の重い学習は
  **Kaggle Notebook GPU**（`PLAYBOOK.md#kaggle-gpu-ワークフローcsv提出コンペ`）を主実行環境として計画する
→ 以降、**推定30分超の実験は毎回「ローカル vs Kaggle GPU」の選択肢を提示する**（CLAUDE.md 環境・ツール参照）

---

### フェーズ1: コンペ概要の精読（思考層）

ユーザーにコンペ概要ページまたはデータ説明を共有・確認してもらいながら、以下を順番に問いかける。

---

**Q1: データの種類は何ですか？**

以下の3分類で答えてもらう:

| 種別 | 説明 | FE方針への影響 |
|---|---|---|
| **実データ** | 実際の観測・計測値 | ドメイン知識・欠損パターンが重要 |
| **合成データ** | 元データから統計的に生成 | 元データの外部シグナルが有効になる可能性が高い |
| **半合成データ** | 実データに合成で拡張 | 実データ部分と合成部分の分離が有効な場合がある |

→ **合成データと判明した場合**: Q2へ進む前に「生成元データは公開されていますか？」を確認する。
　 公開されている場合、外部シグナル特徴量（カテゴリ別ターゲット率・数値分布距離）が
　 内部FEより先に試す価値があることをこの時点で記録する。

**Q2: 評価指標の特性を確認しましょう**

| 指標 | 特性 | 影響 |
|---|---|---|
| AUC | 閾値不要・順位のみ | Calibrationより予測の順序が重要 |
| RMSE/MAE | 外れ値に敏感/鈍感 | 外れ値処理の優先度が変わる |
| Logloss | 確率のCalibrationが重要 | 予測値のクリッピングが必要 |
| F1/Precision/Recall | 閾値依存 | 閾値最適化が必要 |

→ 「この評価指標で特に注意すべき点は何だと思いますか？」と問いかける。

**Q3: 提供データの構造を確認しましょう**

- `data/raw/` のファイル一覧を表示して確認する
- train/test 以外に追加ファイル（external data, supplemental など）が含まれているか
- カラム数・行数の規模感（学習データが小さい場合は CV 設計が重要になる）

**ターゲット列の自動検出（フェーズ3 の config 自動補完に使う）:**

`sample_submission.csv` のヘッダーから `TARGET_COL` を推定する:
```bash
head -1 data/raw/sample_submission.csv
```
- 先頭は通常 ID 列、2 列目以降がターゲット列
- 2 列構成（id, target）→ 2 列目が `TARGET_COL`
- 3 列以上（multiclass の確率列等）→ multiclass の可能性。ユーザーに確認する

検出結果をユーザーに提示して確認する:
```
sample_submission.csv のヘッダー: id,<col>
→ TARGET_COL = "<col>" と推定しました（フェーズ3 で config に反映）。正しいですか？
```

**Q4: Discussion / Notebooks の初期調査**

「コンペの Discussion や公開 Notebook を確認しましたか？」と問いかける。

確認すべき点:
- 外部データを活用している先行 Notebook はあるか
- データ品質の問題（リーク・ラベルノイズ）が報告されているか
- ベースラインの LB スコア帯はどのくらいか（自分のスコアの位置づけの基準）
- CV 設計の議論（通常の StratifiedKFold で良いか）

**Q5: ドメイン知識先行ヒアリング（EDA より前に必須）**

「このターゲット変数に **当然影響するであろう変数** をドメイン直観で 5-10 個列挙してください」と問う。

例（ドメインごとに想定影響変数を列挙）:
- スポーツ戦略予測 (例: 競技中の意思決定): 「装備寿命、消耗状態、天候、競技中断要因、パフォーマンス劣化、戦略窓、競技固有指標」
- 顧客 churn: 「契約年数、月額料金、サービス利用率、サポート問い合わせ数、競合プラン」
- 価格予測 (不動産・商品等): 「立地、面積、築年数/経年、需給ピーク、競合価格、季節要因」

**重要な使い方:**
- 列挙された変数が **現データに無い** 場合 → 外部データ取得の検討シグナル
- 列挙された変数が **粗い形 (binary 等) でしかない** 場合 → 詳細データ拡張の検討
- ドメイン専門家がいる場合は **ML パイプラインを動かす前** にヒアリングする

→ 列挙された変数を `COMPETITION.md` の「ドメイン知識先行リスト」セクションに記録する

**Q6: 外部データインベントリ（強制）**

`ls data/external/` を実行し、存在する全ファイルを列挙する（無い場合はスキップ）。

各ファイルについてユーザーと以下を判定する:

| 判定 | 意味 | 次のアクション |
|---|---|---|
| **使う** | Stage 1 または Stage 4 で投入する | 投入時期を明示 |
| **保留** | 価値不明だが捨てない | **Stage 4 終了までに必ず再評価** |
| **skip** | 不要と判断 | 理由を記録（重複・低品質等） |

> **「後で見る」は許可しない。** すべて 3 択のいずれかに分類する。
> 教訓 (過去事例): ダウンロード済みの外部時系列データがコンペ最終日まで未使用で放置され、+1σ LB 改善を取り逃しかけた

→ 判定結果を `COMPETITION.md` の「外部データインベントリ」セクションに表形式で記録する

**Q7: このコンペで何を重視しますか？（コンペ戦略軸）**

順位・スコアの最大化だけがコンペの目的ではない。ユーザーの戦略軸を最初に言語化しておくことで、
終盤（特に Final 2 選定）の議論が「スコア期待値のみ」に偏ることを防ぐ。

例（複数選択・自由記述可）:
- **スコア最大化**: 使える手段は全て使う（外部公開予測・公開カーネルの知見も積極活用）
- **自前モデルの限界追求**: 外部知見のブレンドより自前パイプラインの完成度を優先する
- **新手法の習得**: 特定アーキテクチャ・手法の実戦練習を優先する
- **プロセス検証**: テンプレート・ワークフローの改善検証を優先する

→ 回答を `COMPETITION.md` の「コンペ戦略軸」セクションに記録する
→ **AI はこの戦略軸を終盤まで保持する義務がある**: Final 2 選定時に必ず再掲し、
  スコア期待値と戦略軸が対立する場合は「スコア軸では A、戦略軸では B」と両論併記して
  ユーザーの判断を仰ぐ（AI が一方の軸だけで推奨を組み立てない）

---

### フェーズ2: キックオフサマリーの記録（実行層）

フェーズ1の回答を `COMPETITION.md` の先頭に記録する:

```markdown
## Kickoff サマリー

- **データ種別**: <実データ / 合成データ（元データ: XXX, 公開: Yes/No）/ 半合成>
- **評価指標**: <指標名> — <特性と注意点>
- **データ規模**: train=X行 × Y列 / test=Z行
- **実行環境**: <ローカル: CUDA/MPS/CPU、GPU名> — 30分超の実験は Kaggle GPU を都度検討
- **Discussion 初期調査**:
  - <先行例や注意点を箇条書き>
- **FE方針の初期判断**:
  - <外部シグナルを先に試す / 内部FEから始める、理由>
- **CV設計の初期判断**:
  - <StratifiedKFold / GroupKFold / TimeSeriesSplit、理由>

### ドメイン知識先行リスト (Q5 回答)
ターゲットに **当然影響するであろう変数** をドメイン直観で列挙:
- <変数 1>: 影響メカニズム / 現データでの有無
- <変数 2>: ...
- <変数 N>: ...

→ **現データに無い変数**は外部データ取得の検討対象

### 外部データインベントリ (Q6 回答)
`data/external/` 内の全ファイル判定:

| ファイル | 判定 | 投入時期 / 理由 |
|---|---|---|
| <file_1> | 使う / 保留 / skip | <Stage 1 / Stage 4 / Stage 4 終了までに再評価 / 重複> |
| ... | ... | ... |

> **「後で見る」は許可されない。** 全ファイルを 3 択分類。
> 保留ファイルは Stage 4 終了までに必ず再評価する。

### コンペ戦略軸 (Q7 回答)
- **重視する軸**: <スコア最大化 / 自前モデルの限界追求 / 新手法の習得 / その他>
- **具体的な意味**: <例: 外部公開予測のブレンドは Final 2 に含めない、等>

> Final 2 選定時に必ずこのセクションを再掲する（`/ds-kaggle-submit` フェーズ5 Step 0）。
> コンペ途中で軸が変わったら更新してよい（変更履歴を1行残す）。
```

---

### フェーズ3: src/config.py のコンペ設定を自動補完（実行層）

**ユーザーが手で入力するのは `COMPETITION` だけ**（`/ds-kaggle-setup` が設定済み）。
残りの項目は以下の情報源から **自動で決定** し、`src/config.py` の TODO セクションを上書きする:

```python
# ===== コンペティション設定（/ds-kickoff スキルが更新する） =====
COMPETITION = "<slug>"             # /ds-kaggle-setup が設定済み（変更不要）
TARGET_COL  = "<col>"              # ← sample_submission.csv の 2 列目（Q3 で検出）
PROBLEM_TYPE = "<type>"            # ← 評価指標 + ターゲット列の値域から推定
EVAL_METRIC  = "<metric>"          # ← Kaggle メタデータから取得
CV_STRATEGY  = "<strategy>"        # ← CV 設計の初期判断（Q4）から
N_SPLITS     = 5
```

**各項目の自動決定ロジック:**

| 項目 | 情報源 | 決定方法 |
|---|---|---|
| `TARGET_COL` | `sample_submission.csv` のヘッダー | 2 列目（ID 以外）。Q3 で検出・確認済み |
| `EVAL_METRIC` | Kaggle メタデータ | `kaggle competitions view -c <slug>` の評価指標欄を参照（取得不可なら Q2 の回答から） |
| `PROBLEM_TYPE` | `EVAL_METRIC` + ターゲット値域 | auc/logloss→分類、rmse/mae→回帰、確率列が 3+→multiclass |
| `CV_STRATEGY` | Q4 の CV 設計判断 | StratifiedKFold が既定。時系列→TimeSeriesSplit、グループ構造→GroupKFold |

```bash
# 評価指標の取得（取得できれば EVAL_METRIC / PROBLEM_TYPE の根拠にする）
kaggle competitions view -c <slug> 2>/dev/null
```

**推定結果は config に書き込む前に必ずユーザーへ提示して確認する:**
```
config を以下で自動補完します（COMPETITION は設定済み）:
  TARGET_COL   = "<col>"      （sample_submission 2 列目）
  EVAL_METRIC  = "<metric>"   （Kaggle メタデータ）
  PROBLEM_TYPE = "<type>"     （指標+値域から推定）
  CV_STRATEGY  = "<strategy>" （Q4 の判断）
この内容でよいですか？
```

---

### フェーズ4: SESSION.md の初期化と次ステップへの接続（実行層）

`src/config.py` 更新後、`SESSION.md` を初期化する:

- `現在のステージ`: Stage 0 — Kickoff 完了
- `次にやること`:
  1. `/ds-new-experiment` で最小ベースライン実験を開始（Stage 1）
  2. ベースラインをLBに提出してCV/LB相関を確立
  3. CV/LB相関が確認できたら `/ds-eda-visual` へ（Stage 2）
- `今サイクルで決めた重要な方針`: FE方針・CV設計の初期判断を転記

記録後、以下を提示して完了:

```
Kickoff 完了。確認した内容:
  データ種別: <種別>
  外部データ: <有/無>
  評価指標:   <指標>
  CV設計:     <StratifiedKFold など>

src/config.py を更新しました:
  COMPETITION  = "<コンペ名>"
  TARGET_COL   = "<ターゲット列名>"
  PROBLEM_TYPE = "<タスク種別>"
  EVAL_METRIC  = "<評価指標>"
  CV_STRATEGY  = "<CV戦略>"

SESSION.md を初期化しました。

次のステップ:
  Stage 1: /ds-new-experiment で最小ベースライン実験を開始する
    → 前処理不要な数値カラムのみ・デフォルトHPで学習
    → scripts/train.py の FEATURES に数値カラムのみを設定
    → LBに提出してCV/LB相関を確立する

※ 合成データかつ元データが入手可能な場合は、
  ベースライン確立後の /ds-eda-visual で Q4（元データ vs コンペデータ比較）を必ず実施。
```

---

## 設計の意図

- **EDAの前に文脈を持つ**: データを見る前に「そのデータが何者か」を知ることで、EDAの問いの質が上がる
- **外部データの早期発見**: 合成データコンペでは元データの存在がコンペ概要に明記されていることが多い。見落とさないための構造化された確認ステップ
- **Discussion の活用**: 先行 Notebook のベースライン LB を把握しておくことで、自分のスコアの位置づけが即座に判断できる
- **CV設計の初期判断**: データの性質（時系列/グループ/クラス不均衡）によって CV 戦略が変わる。この判断を後回しにすると CV/LB 乖離の原因になる
- **コンペ戦略軸の事前言語化 (Q7)**: 「何を重視するか」はユーザーの領分。序盤に明文化しておかないと、終盤の Final 2 議論が AI のスコア期待値軸に偏り、ユーザーの意図（例: 自前モデル重視）との擦り合わせに提出枠を浪費する
