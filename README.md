# DS Template v6.6 — Kaggle Competition Workspace

Claude Code と連携して動く Kaggle コンペ用データサイエンステンプレートです。
「実験の目的を先に言語化する」「1列ずつΔOOFを計測する」「学びをサイクルとして蓄積する」という
3つの原則を、スキルとスクリプトで仕組みとして強制します。

**ローカル環境と Kaggle Notebook 環境の両方に対応しています。**

**実践コンペ**:
- v1: s6e3（Churn Prediction）
- v2: s6e4（Irrigation Need Prediction, rank 293/4316）
- v3: s6e5 (Predicting F1 Pit Stops)
- v4 / v4.1: s6e6 (SDSS Star Classification) で開発・改良
- v4.2: s6e6 振り返りの完全反映 + 汎用DSテンプレートへの純化
- v5: s6e7 (Predicting Student Health Risk, **rank 93/3356 = 上位2.8%**) の総括を反映
- **v6 系**: ドキュメント階層の再設計（4層モデル + SSoT と規律の機械検証）
  - v6: 4層化・恒久ID・`doc_audit` 新設 ／ v6.1: 導線の実装漏れ修正 ／ v6.2: 自己診断の追加
  - v6.3: 本番投入前の最終点検 ／ v6.4: README ドリフトの機械検知（C11）
  - v6.5: 規律の機械化（hook 6 種 × ガード 5 種）・学習成果を失わない仕組み・死蔵の解消
  - **v6.6: CLAUDE.md を憲法へ（3,342 字）・判断指針を GUIDELINES.md へ分離・構成の是正**

---

> 版ごとの改善履歴は [CHANGELOG.md](CHANGELOG.md) を参照。

---

## 現在の構成（実測値・`doc_audit` が検証する）

| 項目 | 現在値 |
|---|---|
| 毎セッション自動ロード（CLAUDE.md） | **3,510 字（-94%）**（59 行。上限は 3,000〜5,000 字かつ 60 行 / v5 は約 56,000 字） |
| ルート直下 | テンプレート文書 6 件のみ（CLAUDE / GUIDELINES / CONVENTIONS / PLAYBOOK / README / CHANGELOG）。コンペごとに育つ記録は `state/`、改善の記録は `docs/` |
| ドキュメント階層 | L0 CLAUDE.md（憲法）/ GUIDELINES.md（判断指針 `G-*`）/ CONVENTIONS.md（辞書）/ PLAYBOOK.md（手順・教訓）/ `.claude/skills/`（対話） |
| `doc_audit` のチェック | **C1-C13**（C4 は実測値、C12 は指針の索引と本文、C13 はエージェント定義を検査） |
| 規律の機械化 | hook 6 種 + statusLine + ガード 6 種 + サブエージェント 4 種（調査・提案・審査のみ／`tools` で学習実行を封じる） |

> この表の数値は `uv run python -m scripts.harness.doc_audit` の C11 が実態と突き合わせる。
> **ズレたら WARNING が出る**ので、README が現実から乖離したまま放置されない。

---

## 必須要件

- [uv](https://github.com/astral-sh/uv) — パッケージ・Python バージョン管理
- [Claude Code](https://claude.ai/claude-code) — AI ペアプログラミング（スキルの実行環境）
- Kaggle API（`~/.kaggle/kaggle.json` 設定済み）

---

## 新しいコンペを始める手順（ローカル環境）

ユーザーが手で入力するのは **コンペスラッグ 1 つだけ**。残りはすべて 2 つのスキルが自動化する。

### Step 1: `/ds-kaggle-setup <コンペ名>` でセットアップ

**親となる作業フォルダの中で**、コンペスラッグを渡して実行するだけ:

```
/ds-kaggle-setup playground-series-s6e6
```

この 1 コマンドで、**コンペを始められる状態までのすべての準備が自動で整う**:

1. **GitHub（ds_template）から clone** され、コンペ名から導出したフォルダ名に配置される
2. `comp/<slug>` ブランチを作成
3. **Kaggle からデータを取得**（`data/raw/` へ）
4. **Python 基本環境を構築**（`uv sync`。**Python 3.12** を `.python-version` で固定）
5. `src/config.py` の `COMPETITION` を自動設定
6. `state/SESSION.md` / `experiments/log.csv` を初期化

### Step 2: `/ds-kickoff` でコンペ文脈を記録 & config を自動補完

```
/ds-kickoff
```

- データ種別・評価指標・外部データ・CV 設計を対話で記録 → `state/COMPETITION.md`
- `src/config.py` の残り項目を自動補完（手作業不要）
- データ未取得なら自動ダウンロード（セーフティネット）

→ 完了後は学習サイクルへ。`/ds-new-experiment` で最小ベースライン実験を開始する。

### （参考）スキルを使わない手動フロー

```bash
git clone https://github.com/nonbuto/ds_template.git my-competition
cd my-competition
git checkout -b comp/<slug>
uv sync                                                  # .python-version=3.12 で構築される
# src/config.py の COMPETITION だけ設定（残りは /ds-kickoff が埋める）
uv run kaggle competitions download -c <slug> -p data/raw/
# Claude Code を起動して /ds-kickoff
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
/ds-kickoff ─────── コンペ参加直後に1回だけ
    ↓ データ種別・合成データ判定・外部データ確認
/ds-new-experiment ─ 最小ベースライン（数値列のみ・デフォルトHP）
    ↓
/ds-kaggle-submit ── CV/LB相関を確立する（以降の改善判断の基準点）
    ↓
/ds-kaggle-research ─ 上位解法のアーキテクチャ分布を調べる（フェーズ0・序盤調査）
    ↓ 「上位が何のアーキテクチャで勝っているか」を主軸決定の前提入力にする
Stage 1.5 ──────── 早期アーキテクチャサーベイ（LGB/RealMLP等を公正比較 → 主軸を決定）
    ↓ OOFとpub_oof_gapを記録。主軸1つ・副軸候補を保持
/ds-eda-visual ───── 「何を知りたいか」を先に言語化してから可視化
    ↓ FE仮説の種を /ds-fe-hypothesis に登録しながら進む
Optuna 軽量 ──── 作業用HP（20〜30試行）。ΔOOFのノイズを低減する目的
    ↓
/ds-fe-hypothesis ── 「なぜ効くか」の因果を言語化 → 実装 → 可視化確認 → ΔOOFを計測
    ↓ 必ず1列ずつ feature_study.py で投入。複数列の一括追加は禁止
    ↓ FE確定後、全候補アーキテクチャに同一FEを移植して再評価
Optuna フル ───── 確定した特徴量セットで100試行以上
    ↓
/ds-kaggle-submit ── OOF/LBギャップを解釈して学びを言語化
    ↓
/ds-new-experiment ─ 次のサイクルへ（アンサンブルへ移行 or FEに戻る）
```

> FE棄却が3連続したら → `/ds-kaggle-research` で上位者の知見を確認してから次の仮説を立てる

---

## スクリプトの実行

```bash
# Stage 2: EDA可視化（画像を data/output/plots/ に保存 → Claude が Read で読む）
uv run python scripts/visualize.py

# Stage 1・4: CV学習
uv run python scripts/train.py --model lgb

# Stage 3: 作業用HP（FE中のΔOOFノイズ低減）
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
| `/ds-resume` | **毎セッション開始時（必須）** | state/SESSION.md + log.csv + state/FE_HYPOTHESES.md を読み「今どこにいるか」を1画面で復元 |
| `/ds-kickoff` | コンペ参加直後（1回のみ） | データ種別・外部データ有無・CV設計の初期判断を state/COMPETITION.md に記録 |
| `/ds-new-experiment` | 実験開始前 | 目的・成功基準・撤退基準を言語化してからブランチとインフラを整備 |
| `/ds-kaggle-submit` | 提出前後 | 提出前確認 → LBスコア取得 → OOF/LB乖離分析 → 学びを log.csv に記録 |
| `/ds-eda-visual` | Stage 2 | 「問い→可視化→発見→FE仮説の種」の対話型EDA |
| `/ds-fe-hypothesis` | Stage 4 | FE仮説の立案・実装後可視化確認・検証・棄却理由の構造化 |
| `/ds-kaggle-research` | **Stage 1.5 の前（序盤調査）**・FE棄却3連続後・Stage 6 外部予測活用時 | 上位解法のアーキテクチャ分布調査（フェーズ0）／ Kaggle Discussion / Dataset / Kernel を CLI で系統的に調査 |
| `/ds-template-update` | 随時 | テンプレート改善アイデアを docs/TODO_TEMPLATE.md に記録 |

> `/ds-kaggle-setup` は上表に含まれない。**repo を clone する側**なので repo 内には置けず、
> `~/.claude/skills/` の個人スキルとして持つ（新コンペの Step 1 で使う）。
>
> `/ds-eda-report` は v6 で削除済み。機能は `/ds-eda-visual` と `state/FEATURE_REPORT.md` に統合済み。

---

## ディレクトリ構成

```
├── CLAUDE.md              # L0 原則: 判断の憲法（恒久ID G-XXX・ステージ定義）※毎セッション自動ロード
├── CONVENTIONS.md         # L1 規約: パス・命名・log.csv列・コミット形式 ※引くときだけ読む
├── PLAYBOOK.md            # L2 手順+史料: 実行レシピ + 教訓アーカイブ(L-NN) ※局面参照
├── state/COMPETITION.md         # コンペ固有メモ（/ds-kickoff が生成・更新）
├── state/FE_HYPOTHESES.md       # FE仮説の立案・検証・棄却記録（/ds-fe-hypothesis が管理）
├── state/FEATURE_REPORT.md      # 特徴量の生きたレポート（EDA・FE段階を通じて記入）
├── state/EDA_SUMMARY.md         # EDA対話の発見まとめ（/ds-eda-visual が生成）
├── state/SESSION.md             # セッション現在地・次のアクション（/ds-resume で参照）
├── docs/TODO_TEMPLATE.md       # テンプレート改善タスク（/ds-template-update が追記）
│
├── scripts/               # 汎用骨格スクリプト（コンペ開始時に TODO を埋めて使う）
│   ├── doc_audit.py       # ドキュメント階層の検査（SSoT・行数予算・実測値の保存）
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
    ├── skills/            # L3 進行台本: 対話フェーズと質問文面
    └── settings.json      # PostToolUse hook（可視化ガードの自動起動）
```

---

## 設計上の主要な判断

| 判断 | 理由 |
|---|---|
| **可視化は画像ファイルで保存** | Claude Code は marimo のレンダリングを認識できない。`data/output/plots/` に `.png` を保存し、Read ツールで読んで対話する |
| **FEは1列ずつ計測** | 複数列を一度に追加すると「どれが効いたか」が分からなくなる。`feature_study.py` で1列ずつΔOOFを計測する |
| **実験の目的を先に記録** | 結果が出てから目的を決めると合理化が起きる。`/ds-new-experiment` で「何を明らかにするか」を先に log.csv に記録する |
| **state/SESSION.md は上書き原則** | 履歴を追記すると80行を超えて読めなくなる。各セクションは常に最新1件だけ上書きし、詳細は git log で追跡する |
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
