# DS Template v4.1 — Kaggle Competition Workspace

Claude Code と連携して動く Kaggle コンペ用データサイエンステンプレートです。
「実験の目的を先に言語化する」「1列ずつΔAUCを計測する」「学びをサイクルとして蓄積する」という
3つの原則を、スキルとスクリプトで仕組みとして強制します。

**ローカル環境と Kaggle Notebook 環境の両方に対応しています。**

**実践コンペ**:
- v1: s6e3（Churn Prediction）
- v2: s6e4（Irrigation Need Prediction, rank 293/4316）
- v3: s6e5 (Predicting F1 Pit Stops)
- v4 / **v4.1: s6e6 (SDSS Star Classification)** で開発・改良

---

## v4.1 で追加された主な改善

s6e6 の全実験（104本）を振り返り、「スコア上昇」と「user との共創」の観点でテンプレートを精査。
原則を1つも削らずに、探索効率と認知負荷を改善した。

### ① 外部知見調査をワークフロー前段に配置

`/kaggle-research` を「行き詰まりの応急処置」から「主軸アーキテクチャ決定の前提入力」へ格上げ。

- **フェーズ0（序盤アーキテクチャ調査）** を新設。Stage 1.5 の直前に上位カーネルのアーキテクチャ分布を調べ、サーベイ候補に反映する
- Playground Series は上位解法のアーキテクチャがスコアを最も動かす。それを終盤ではなく序盤の方向づけに使う

### ② CLAUDE.md の2層分離（Core + PLAYBOOK）

CLAUDE.md 1246行 → 734行（-41%）。実行レシピを `PLAYBOOK.md`（650行）へ分離。

- **CLAUDE.md** = 毎ターン守る「原則・判断基準（精神）」。AI指針 #1-22 は全文保持
- **PLAYBOOK.md** = その局面で読む「実行レシピ（手順・コード・コマンド）」
- 判断に必要な情報だけで意思決定でき、実行時に該当セクションを参照する。認知負荷を下げてルール形骸化を防止

### ③ FE 仮説にドメイン実体を引き出す Q0 を追加

`/fe-hypothesis` の因果メカニズム掘り下げに **Q0（ドメイン実体の確認）** を新設。

- AI が因果を推測する前に、user のドメイン知識から変数の実体（物理的/業務的な意味）を引き出す
- 「相関がありそう」から「この物理量がこう振る舞うからターゲットが決まる」という鋭い仮説へ
- 「共創の原則」を Kickoff だけでなく FE サイクル全体へ展開

### ④ 提出のたびに「伸びしろの所在」を可視化

`/kaggle-submit` に LBトップとの差分＋その差を埋める最有力候補（アーキテクチャ/情報次元/多様性）を毎回提示。

- 局所改善（+ノイズ床の FE）に没入して最大の伸びしろ（別アーキテクチャ等）を見落とすことを防ぐ
- 埋めるべきは「構造的な伸びしろ」であって Public LB スコアそのものではない（AI指針 #17/#18/#21 と整合）

---

## v4 で追加された主な改善

s6e6 実践（3クラス Balanced Accuracy）を経て、アーキテクチャ探索戦略と OOF/LB 評価軸を大幅強化。

### AI 行動規範 #21-22（CLAUDE.md）

- **#21**: OOF最大化 + pub_oof_gap最小化の二軸評価（gap最大化戦略を廃止）
  - 実証: OOF→Private r=+0.998、pub_oof_gap→シェイクダウン量 r=+0.853
  - 第一目標: OOF最大化。第二目標: 同OOF水準なら pub_oof_gap 最小化
  - モデルファミリー別 OOF 信頼性テーブル（NN系/Tree系/Blend の傾向）
- **#22**: アーキテクチャ乗り換え時の公正比較義務
  - 同一特徴量セット × 作業用HP確定済み × 同一CV の3条件を揃えてから比較
  - 「最適化済み LGB vs デフォルト HP の RealMLP」は不公正比較として禁止

### Stage 1.5（早期アーキテクチャサーベイ）新設

FE探索を始める前に「主軸アーキテクチャ」を決定するステージを追加。

```
Stage 1（最小ベースライン）完了後 → Stage 1.5 → Stage 2（EDA）開始
```

- 候補アーキテクチャ（LGB / CatBoost / RealMLP 等）を最小特徴量 × 作業用HP × 同一CVで評価
- 各アーキテクチャの OOF と pub_oof_gap を記録し、主軸を1つ決定
- 副軸（主軸と OOF 差 10% 以内）は Stage 6 アンサンブル候補として記録

> **教訓**: 過去コンペで LGB 主軸のまま 40+ 実験を費やした後、RealMLP に移行したら提出効率が 50x 改善（+0.000007 LB/提出 → +0.000343 LB/提出）。Stage 1.5 で早期に特定すべきだった。

### FE 有効性のアーキテクチャ依存性明記

- LGB で棄却した FE が RealMLP では有効なケースに対応
- FE 棄却記録に「棄却したアーキテクチャ名」の明記を義務化
- Stage 4 → Stage 6 移行時に全候補アーキテクチャへ同一 FE を移植して再評価

### Kaggle Notebook 環境サポート追加

`src/config.py` が `/kaggle/input` の存在を自動検出し、パスを切り替える。

```python
from src.config import IS_KAGGLE, RAW_DATA_DIR, OOF_DIR
# ローカル: IS_KAGGLE=False、RAW_DATA_DIR=data/raw/
# Kaggle:  IS_KAGGLE=True、RAW_DATA_DIR=/kaggle/input/
```

---

## 必須要件

- [uv](https://github.com/astral-sh/uv) — パッケージ・Python バージョン管理
- [Claude Code](https://claude.ai/claude-code) — AI ペアプログラミング（スキルの実行環境）
- Kaggle API（`~/.kaggle/kaggle.json` 設定済み）

---

## 新しいコンペを始める手順（ローカル環境）

ユーザーが手で入力するのは **コンペスラッグ 1 つだけ**。残りはすべて 2 つのスキルが自動化する。

### Step 1: `/kaggle-setup <コンペ名>` でセットアップ

**親となる作業フォルダの中で**、コンペスラッグを渡して実行するだけ:

```
/kaggle-setup playground-series-s6e6
```

この 1 コマンドで、**コンペを始められる状態までのすべての準備が自動で整う**:

1. **GitHub（ds_template）から clone** され、コンペ名から導出したフォルダ名に配置される
2. `comp/<slug>` ブランチを作成
3. **Kaggle からデータを取得**（`data/raw/` へ）
4. **Python 基本環境を構築**（`uv sync`。**Python 3.12** を `.python-version` で固定）
5. `src/config.py` の `COMPETITION` を自動設定
6. `SESSION.md` / `experiments/log.csv` を初期化

### Step 2: `/kickoff` でコンペ文脈を記録 & config を自動補完

```
/kickoff
```

- データ種別・評価指標・外部データ・CV 設計を対話で記録 → `COMPETITION.md`
- `src/config.py` の残り項目を自動補完（手作業不要）
- データ未取得なら自動ダウンロード（セーフティネット）

→ 完了後は学習サイクルへ。`/new-experiment` で最小ベースライン実験を開始する。

### （参考）スキルを使わない手動フロー

```bash
git clone https://github.com/nonbuto/ds_template.git my-competition
cd my-competition
git checkout -b comp/<slug>
uv sync                                                  # .python-version=3.12 で構築される
# src/config.py の COMPETITION だけ設定（残りは /kickoff が埋める）
uv run kaggle competitions download -c <slug> -p data/raw/
# Claude Code を起動して /kickoff
```

---

## Kaggle Notebook 環境での使い方

### セットアップ

1. このリポジトリを Kaggle Dataset として登録する
2. Notebook に Dataset を追加（`/kaggle/input/<dataset-name>/` にマウント）
3. Notebook の最初のセルで実行:

```python
import sys
sys.path.insert(0, "/kaggle/input/<dataset-name>")

from src.config import IS_KAGGLE, RAW_DATA_DIR, COMPETITION
print(f"IS_KAGGLE={IS_KAGGLE}")        # → True
print(f"RAW_DATA_DIR={RAW_DATA_DIR}")  # → /kaggle/input/
```

### 実験スクリプト実行

```python
import subprocess
result = subprocess.run(
    ["python", "/kaggle/input/<dataset-name>/experiments/runs/exp001_s1_lgb_baseline.py"],
    capture_output=True, text=True
)
print(result.stdout)
```

### 提出

```python
from src.config import submission_path, COMPETITION
import subprocess

# 提出ファイル生成（/kaggle/working/data/output/submissions/ に保存）
sub_path = submission_path(model="lgb", oof_score=0.91234, exp_id="001")
sub_df.to_csv(sub_path, index=False)

# Kaggle CLI で直接提出（Internet access を有効にすること）
subprocess.run([
    "kaggle", "competitions", "submit",
    "-c", COMPETITION, "-f", str(sub_path), "-m", "exp001 lgb baseline"
])
```

> **注意**: Kaggle Notebook は `/kaggle/working/` のみ書き込み可能。セッションをまたぐ場合は成果物を Dataset に保存して持ち出す。

---

## 学習サイクル（スキルとスクリプトの使い方）

```
/ds-resume ──── 毎セッション開始時に必ず呼ぶ
    ↓
/kickoff ─────── コンペ参加直後に1回だけ
    ↓ データ種別・合成データ判定・外部データ確認
/new-experiment ─ 最小ベースライン（数値列のみ・デフォルトHP）
    ↓
/kaggle-submit ── CV/LB相関を確立する（以降の改善判断の基準点）
    ↓
/kaggle-research ─ 上位解法のアーキテクチャ分布を調べる（フェーズ0・序盤調査）
    ↓ 「上位が何のアーキテクチャで勝っているか」を主軸決定の前提入力にする
Stage 1.5 ──────── 早期アーキテクチャサーベイ（LGB/RealMLP等を公正比較 → 主軸を決定）
    ↓ OOFとpub_oof_gapを記録。主軸1つ・副軸候補を保持
/eda-visual ───── 「何を知りたいか」を先に言語化してから可視化
    ↓ FE仮説の種を /fe-hypothesis に登録しながら進む
Optuna 軽量 ──── 作業用HP（20〜30試行）。ΔAUCのノイズを低減する目的
    ↓
/fe-hypothesis ── 「なぜ効くか」の因果を言語化 → 実装 → 可視化確認 → ΔAUCを計測
    ↓ 必ず1列ずつ feature_study.py で投入。複数列の一括追加は禁止
    ↓ FE確定後、全候補アーキテクチャに同一FEを移植して再評価
Optuna フル ───── 確定した特徴量セットで100試行以上
    ↓
/kaggle-submit ── OOF/LBギャップを解釈して学びを言語化
    ↓
/new-experiment ─ 次のサイクルへ（アンサンブルへ移行 or FEに戻る）
```

> FE棄却が3連続したら → `/kaggle-research` で上位者の知見を確認してから次の仮説を立てる

---

## スクリプトの実行

```bash
# Stage 2: EDA可視化（画像を data/output/plots/ に保存 → Claude が Read で読む）
uv run python scripts/visualize.py

# Stage 1・4: CV学習
uv run python scripts/train.py --model lgb

# Stage 3: 作業用HP（FE中のΔAUCノイズ低減）
uv run python scripts/optimize_hp.py --model lgb --n-trials 25 --tag working

# Stage 4: 1列ΔCV計測（FE仮説の効果測定）
uv run python scripts/feature_study.py --new-feature <feature_name>

# Stage 5: 本格HP最適化
uv run python scripts/optimize_hp.py --model lgb --n-trials 150 --tag full

# 提出ファイル生成
uv run python scripts/predict.py --exp-id 042 --model lgb --oof-score 0.91688

# Stage 6: アンサンブル（相関確認 → Simple Blend → Greedy HC）
uv run python scripts/blend.py --mode corr   --oofs lgb=oof_042.npy cb=oof_070.npy
uv run python scripts/blend.py --mode blend  --oofs lgb=oof_042.npy cb=oof_070.npy
uv run python scripts/blend.py --mode greedy --oofs lgb=oof_042.npy cb=oof_070.npy \
    --tests lgb=test_042.npy cb=test_070.npy

# 特徴量レポート（重要度・ΔOOF棒グラフ → Claude が Read で読む）
uv run python scripts/feature_report.py
```

---

## Claude Code スキル一覧

| スキル | タイミング | 役割 |
|---|---|---|
| `/ds-resume` | **毎セッション開始時（必須）** | SESSION.md + log.csv + FE_HYPOTHESES.md を読み「今どこにいるか」を1画面で復元 |
| `/kickoff` | コンペ参加直後（1回のみ） | データ種別・外部データ有無・CV設計の初期判断を COMPETITION.md に記録 |
| `/new-experiment` | 実験開始前 | 目的・成功基準・撤退基準を言語化してからブランチとインフラを整備 |
| `/kaggle-submit` | 提出前後 | 提出前確認 → LBスコア取得 → OOF/LB乖離分析 → 学びを log.csv に記録 |
| `/eda-visual` | Stage 2 | 「問い→可視化→発見→FE仮説の種」の対話型EDA |
| `/fe-hypothesis` | Stage 4 | FE仮説の立案・実装後可視化確認・検証・棄却理由の構造化 |
| `/kaggle-research` | **Stage 1.5 の前（序盤調査）**・FE棄却3連続後・Stage 6 外部予測活用時 | 上位解法のアーキテクチャ分布調査（フェーズ0）／ Kaggle Discussion / Dataset / Kernel を CLI で系統的に調査 |
| `/template-update` | 随時 | テンプレート改善アイデアを TODO_TEMPLATE.md に記録 |

---

## ディレクトリ構成

```
├── CLAUDE.md              # Claude Code プロジェクト設定（原則・AI行動ルール・ステージ定義）※毎ターン参照
├── PLAYBOOK.md            # 実行レシピ集（アンサンブル手順・GPUワークフロー・Final 2 等）※局面参照
├── COMPETITION.md         # コンペ固有メモ（/kickoff が生成・更新）
├── FE_HYPOTHESES.md       # FE仮説の立案・検証・棄却記録（/fe-hypothesis が管理）
├── FEATURE_REPORT.md      # 特徴量の生きたレポート（EDA・FE段階を通じて記入）
├── EDA_SUMMARY.md         # EDA対話の発見まとめ（/eda-visual が生成）
├── SESSION.md             # セッション現在地・次のアクション（/ds-resume で参照）
├── TODO_TEMPLATE.md       # テンプレート改善タスク（/template-update が追記）
│
├── scripts/               # 汎用骨格スクリプト（コンペ開始時に TODO を埋めて使う）
│   ├── train.py           # CV学習（LGB / CB / XGB 切り替え）
│   ├── feature_study.py   # 1列ΔOOF計測（Stage 4 FE仮説の効果測定）
│   ├── optimize_hp.py     # Optuna HP最適化（Stage 3: 軽量 / Stage 5: フル）
│   ├── predict.py         # OOF・test 予測 → 提出ファイル生成
│   ├── blend.py           # アンサンブル（相関確認 / 重み最適化 / Greedy HC）
│   ├── visualize.py       # EDA可視化 → data/output/plots/ に画像保存
│   └── feature_report.py  # 特徴量重要度・ΔOOF棒グラフを画像生成
│
├── experiments/
│   ├── log.csv            # 全実験サマリー（OOF・LB・oof_lb_gap・学びを記録）
│   └── runs/              # コンペ固有の1回限りスクリプト
│       └── exp{NNN}_s{stage}_{内容}.py
│
├── src/                   # 共通ライブラリ
│   ├── config.py          # パス・コンペ設定・命名規約（IS_KAGGLE 自動検出）
│   ├── experiment.py      # 実験トラッキング（log.csv 書き込み）
│   ├── validation.py      # データバリデーション
│   ├── hp_spaces.py       # Optuna サーチスペース定義
│   ├── feature_registry.py # 特徴量レジストリ
│   └── utils/
│       ├── ensemble.py    # correlation_check / optimize_weights / greedy_ensemble
│       └── logger.py      # ロガー
│
├── data/                  # ← Git 管理外（.gitignore で除外）
│   ├── raw/               # 生データ（読み取り専用）
│   ├── processed/         # 前処理済みデータ（.pkl）
│   └── output/
│       ├── submissions/   # 提出CSV（submission_path() で命名）
│       ├── oof/           # OOF・test予測（.npy）
│       ├── models/        # 学習済みモデル
│       ├── params/        # best_params JSON
│       └── plots/         # 可視化画像（Claude が Read で読んで対話に使う）
│
└── .claude/
    ├── skills/            # Claude Code スキル（上記スキル一覧）
    └── rules/             # コーディング規約
```

---

## 設計上の主要な判断

| 判断 | 理由 |
|---|---|
| **可視化は画像ファイルで保存** | Claude Code は marimo のレンダリングを認識できない。`data/output/plots/` に `.png` を保存し、Read ツールで読んで対話する |
| **FEは1列ずつ計測** | 複数列を一度に追加すると「どれが効いたか」が分からなくなる。`feature_study.py` で1列ずつΔAUCを計測する |
| **実験の目的を先に記録** | 結果が出てから目的を決めると合理化が起きる。`/new-experiment` で「何を明らかにするか」を先に log.csv に記録する |
| **SESSION.md は上書き原則** | 履歴を追記すると80行を超えて読めなくなる。各セクションは常に最新1件だけ上書きし、詳細は git log で追跡する |
| **OOF最大化を第一目標・pub_oof_gap最小化を第二目標** | OOF→Private 相関が極めて高い（r≈0.998）。gap最大化は Private で逆効果（r≈−0.51）と実証済み |
| **Stage 1.5 でアーキテクチャを早期決定** | FE探索後のアーキテクチャ乗り換えは探索効率が大幅に落ちる。最小特徴量の段階で公正比較して主軸を固める |
| **IS_KAGGLE 自動検出** | ローカルと Kaggle Notebook でパスが異なる。`/kaggle/input` の存在確認で自動切り替えし、コードの分岐を最小化する |

---

## 技術スタック

| ツール | 用途 |
|---|---|
| `uv` | パッケージ管理（pip/conda 不使用） |
| `LightGBM` | デフォルトモデル |
| `XGBoost` / `CatBoost` | アンサンブル用追加モデル |
| `RealMLP` | NN系主軸候補（pub_oof_gap 小・OOF信頼性高） |
| `Optuna` | ハイパーパラメータ最適化 |
| `matplotlib` / `seaborn` | 可視化（非インタラクティブ・画像保存） |
| `SHAP` | 特徴量重要度の説明 |
| `scikit-learn` | CV / 前処理ユーティリティ |
