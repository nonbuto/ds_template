# CONVENTIONS — 規約リファレンス

> **この層（L1）の役割**: 迷ったら引く「辞書」。**決まりだけを書き、判断はしない。**
> 規範文（〜せよ／禁止／必須）は `CLAUDE.md`（L0）が SSoT。ここには置かない。
>
> オンデマンドで読む。毎ターン読む必要はないが、**下記のいずれかを実装/記録するときは必ず参照する**。

---

## ディレクトリ規約

| ディレクトリ | 用途 |
|---|---|
| `data/raw/` | 生データ（読み取り専用） |
| `data/processed/` | 加工済みデータ（pickle形式） |
| `data/output/submissions/` | 提出CSVのみ（`submission_path()` で生成） |
| `data/output/oof/` | OOF・test予測の `.npy` ファイル |
| `data/output/models/` | 学習済みモデルファイル |
| `data/output/params/` | best_params JSON |
| `data/output/plots/` | 可視化画像（`.png`）。Claudeが読んで対話に使う |
| `experiments/` | log.csv + MLflowアーティファクト |
| `scripts/` | 再利用可能なスクリプト（後述） |
| `experiments/runs/` | 実験ごとの1回限りスクリプト |
| `state/` | **コンペごとに育つ作業記録**（SESSION / COMPETITION / FE_HYPOTHESES / FEATURE_REPORT / EDA_SUMMARY / KAGGLE_RESEARCH）。パスは `src.config` の `STATE_DIR` から取る |
| `docs/` | テンプレート改善の記録（TODO_TEMPLATE / TODO_ARCHIVE） |

> `data/output/` 直下にファイルを置かない。役割別サブディレクトリを必ず使う。

---

## スクリプト構成

**スクリプトは 2 系統に分ける**（判定基準: 「実験・学習・提出物を作る」= コンペ用 ／
「規律を守らせる・現在地を知らせる」= ハーネス用）:

| 置き場所 | 起動 | 役割 |
|---|---|---|
| `scripts/` | `uv run python -m scripts.<名前>` | コンペ用。前処理・学習・FE計測・HP探索・提出物生成 |
| `scripts/harness/` | `uv run python -m scripts.harness.<名前>` | ハーネス用。hook から呼ばれるものと、規律・現在地の診断ツール |

`__init__.py` は置かない（名前空間パッケージ）。ハーネス側の `ROOT` は
`Path(__file__).resolve().parents[2]` でリポジトリルートを指す（1 階層深いため）。

**`scripts/`（コンペ用）**

| ファイル | Stage | 役割 |
|---|---|---|
| `scripts/preprocess.py` | 0 | 生データ → `train_features.pkl` / `test_features.pkl`（**パイプラインの最初の工程**。保存前に `src/validation.py` のスキーマ・リーク・欠損検証を通す）|
| `scripts/train.py` | 1・4 | CV学習の汎用骨格（モデル・特徴量をconfigで切り替え） |
| `scripts/feature_study.py` | 4 | 1列ΔCV計測（FE仮説の効果測定） |
| `scripts/optimize_hp.py` | 3・5 | Optuna HP探索 |
| `scripts/predict.py` | 全般 | OOF予測→提出ファイル生成 |
| `scripts/blend.py` | 6 | アンサンブル・ブレンド |
| `scripts/visualize.py` | 2 | EDA可視化→`data/output/plots/`に画像保存 |
| `scripts/feature_report.py` | 随時 | 特徴量重要度・ΔOOF棒グラフを画像生成 |
| `scripts/harness/deadline_status.py` | 随時 | 現在UTC・締切・残り時間・本日の提出使用枠を一括表示 |
| `scripts/av_check.py` | 4 | AV診断（train/test 分布シフトの検出） |
| `scripts/harness/doc_audit.py` | 随時 | ドキュメント階層の検査（重複・SSoT違反・参照切れ） |
| `scripts/harness/viz_guard.py` | 随時 | 可視化・診断記録・推論成果物の機械チェック |
| `scripts/harness/state_audit.py` | 随時 | 状態ファイルの停滞を検知（log.csv の実験時刻 vs mtime） |
| `scripts/harness/session_brief.py` | hook | SessionStart / PostCompact で現在地を提示 |
| `scripts/harness/session_audit.py` | hook | Stop でコミット規律・状態鮮度・3ガードを監査 |
| `scripts/harness/submit_gate.py` | hook | PreToolUse で Kaggle 提出にユーザー承認を要求 |
| `scripts/harness/session_snapshot.py` | hook | PreCompact で state/SESSION.md へ状態を退避 |
| `scripts/harness/job_status.py` | 随時 | 実行中ジョブの生存・fold 進捗・ETA を表示 |
| `scripts/harness/hook_status.py` | 随時 | **どの hook が実際に発火したか**を実測ログから集計 |

**`src/`（コア）**: `config.py`（パス・コンペ設定）、`experiment.py`（`ExperimentTracker` と各ガード）、
`hp_spaces.py`（Optuna 探索空間）、`validation.py`（スキーマ・リーク・欠損の検証。`preprocess.py` が呼ぶ）

**`src/utils/`（共通ヘルパー）**: `ensemble.py`（重み最適化・相関チェック）、
`logger.py`、`plot_style.py`（日本語フォント設定・可視化の命名規則ヘルパー）、
`finalize.py`（OOF + test 予測 + 提出 CSV を 1 回で保存）、
`multiseed.py`（multi-seed avg の実行・既存 seed 結果の再利用）、
`foldcache.py`（fold 単位チェックポイント・中断した学習の再開）

**`experiments/runs/`（コンペ固有・使い捨て）**

命名規約: `exp{NNN}_s{stage}_{内容}.py`

```
experiments/runs/
  exp001_s1_lgb_baseline.py       ← Stage 1: 最小ベースライン
  exp003_s3_hp_lgb_optuna.py      ← Stage 3: 作業用HP調整
  exp042_s4_fe_age_sq.py          ← Stage 4: 特徴量追加
  exp099_s5_hp_lgb_full.py        ← Stage 5: 本格HP最適化
  exp171_s6_lgb_cb_blend.py       ← Stage 6: アンサンブル
```

- `exp{NNN}`: `experiments/log.csv` の `experiment_id` と一致させる
- `s{stage}`: どのステージの実験かが一目で分かる
- `scripts/` のスクリプトを呼び出すラッパーとして書くことを推奨

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

**学習スクリプトの終わり方（必須）**: 学習した実験は、同じ実行の中で
**OOF・test 予測・提出 CSV の 3 点を出し切る**。`src/utils/finalize.py` の
`save_run_outputs()` を最後に 1 回呼べば揃う。

```python
from src.utils.finalize import save_run_outputs
save_run_outputs(exp_id=exp_id, model="lgb_h012", oof=oof, test=test, oof_score=oof_score)
```

- **例外は ΔOOF スクリーニング専用の実験だけ**（`scripts/feature_study.py` 等）。
  提出候補になりうる実験で test 予測を省くと、提出時に同じ学習をやり直すことになる（CLAUDE.md `G-STEPWISE`）
- `ExperimentTracker.end_run()` が「OOF はあるのに test が無い」実験を機械検知して警告する

### multi-seed avg（avg5 等）の実行規約

`src/utils/multiseed.py` の `run_multiseed()` を使う。**基本 seed（`RANDOM_STATE`）の結果が
既にディスクにあれば再学習せず再利用し、残りの seed だけを回す**（avg5 なら学習時間が 1/5 削減される）。

- seed ごとの予測は `oof_{tag}_s{seed}.npy` / `test_{tag}_s{seed}.npy` で保存する（次回の再利用のため）
- **再利用の前提は「同じ特徴量セット・同じ HP」**。FE や HP を変えたら `tag` も変える
  （古い seed 結果が混ざると不公正比較になる → CLAUDE.md `G-FAIR`）
- 既定 seed 列は `(RANDOM_STATE, 0, 1, 7, 2026)`。先頭が再利用対象になる

### fold 単位チェックポイント（長時間の学習）

`src/utils/foldcache.py` の `FoldCache` を fold ループに挟む。seed 単位では救えない
**中断（kill・クラッシュ・見積もり外れによる打ち切り）から再開できる**ようにする。

- 保存名は `val_{tag}_s{seed}_f{fold}.npy` / `test_{tag}_s{seed}_f{fold}.npy`
- 保存先は `data/output/oof/_foldcache/`（`.gitignore` 済み。再開用の一時成果物）
- **再利用の前提は multi-seed avg と同じ**（同じ特徴量セット・同じ HP。変えたら `tag` も変える）
- `cache.report()` を学習開始時に print すると、何 fold をスキップしたかが記録に残る

### 実行中ジョブの確認

`start_run()` が `experiments/.running/{exp_id}.json` にハートビートを書き、
`log_fold_scores()` が fold 進捗で更新、`end_run()` が削除する。

```bash
uv run python -m scripts.harness.job_status   # 生存・fold 進捗・ETA・ハング検知
```

「まだ動いていますか」「また止まってませんか」を人が尋ねずに済ませるための機構。
PID が消えていれば異常終了、ハートビートが 15 分以上古ければハングを疑う。

---

## marimo の用途

- **可視化 EDA では使わない** — Claude は marimo のレンダリング結果を認識できない。可視化はスクリプトから `data/output/plots/` に画像保存し、Claude が Read で読んで対話する
- **`.ipynb` 変換に使う** — `scripts/to_kaggle_nb.py` が marimo 形式スクリプトを `marimo export ipynb` で変換する。Kaggle Notebook 実行用の `.ipynb` 生成に使う

---

## コーディング規約

- パスは必ず `src.config` からインポート（ハードコード禁止）
- 乱数シードは `RANDOM_STATE`（`src.config`から）
- 特徴量名: snake_case・スペースなし
- `src/` 配下に型ヒントを付ける

---

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
- ファイル名: **`{実験ID3桁}_{通し番号2桁}_{内容}.png`**（例: `042_01_importance_top30.png`）
  - EDA 段階など実験 ID が確定していない場合は `eda_{変数名}_{テーマ}.png` を使う
  - **数字プレフィックスを必ず先頭に置く**理由: `ls` のソート順が作成順と一致し、
    「どの実験に紐づく、いつ作られた図か」が後から追える。過去コンペでは
    `oof_dist_079_vs_080_xgb.png` / `loo_mlp_delta_oof_loss.png` のように
    命名がその場限りになり、時系列も紐付けも追えなくなった

**日本語フォント（必須の初期設定）**: uv 管理の venv には CJK フォントが入っていないため、
matplotlib で日本語ラベルを描くと文字化け（tofu 表示）する。可視化スクリプトの冒頭で
`src/utils/plot_style.py` の `setup_japanese_font()` を呼ぶ（`scripts/visualize.py` 等では実施済み）。

> この節が可視化規約の **SSoT**。CLAUDE.md・各スキルはここを参照するだけにする
> （v5 では 4 箇所に重複していた）。

---

## Optuna study の永続化と命名

`scripts/optimize_hp.py` は study を `data/output/params/optuna_studies/{model}_{tag}.db`
（SQLite）に永続化し、同名 study があれば `load_if_exists=True` で**再開**する。

- `study_name` は `{model}_{tag}` （例: `lgb_working`, `xgb_full`）。**同じ tag を使えば試行が積み上がる**
- 探索をやり直したい場合は tag を変える（例: `--tag full_v2`）か、該当 `.db` を削除する
- `.db` は数十 KB〜数 MB。`best_params_*.json` は「最良の 1 点」しか残さないが、
  study を残せば **追加試行・fANOVA による寄与分析・探索履歴の監査**ができる

---

## 提出の手動チェックリスト

**前提**（スキル経由でもこの 3 つは変わらない）: ①提出は `/ds-kaggle-submit` 経由（直接 CLI は禁止）
②提出前に git working tree が clean ③提出後に `submit_score` / `lb_rank` / `learning` を log.csv に記録。

`/ds-kaggle-submit` が使えないとき、AI がスキルのフローを代替するための手順。
**チェックリストの省略は禁止**（規範は CLAUDE.md「Kaggle提出ルール」）。

スキル経由が不可でも、以下のチェックリストを手動で実施してから CLI 提出する:

1. `git status` が clean か確認する
2. 提出ファイルは `submission_path()` 生成のものか確認する（→ `CONVENTIONS.md#提出ファイルの命名規約`）
3. 提出後 `kaggle competitions submissions | head -3` で LB スコアを確認する
4. log.csv の `submit_score` と `oof_lb_gap`（= `oof_score` − `submit_score`）を更新する
5. state/SESSION.md のスコアテーブルを更新する（**OOF-LB 乖離列を必ず記入**）+ 本日の提出数を記録する
6. `git commit` で LB 結果を記録する

スキルが提供するフローをAIが代替する。チェックリストの省略は禁止。

---

## 提出ファイルの命名規約

提出CSVは必ず `submission_path()` ヘルパーで生成する:

```python
from src.config import submission_path
sub_path = submission_path(model="lgb_cb_blend", oof_score=0.91777, exp_id="171")
# → data/output/submissions/sub_171_lgb_cb_blend_0.91777_20260331_2347.csv
sub.to_csv(sub_path, index=False)
```

命名規約: `sub_{exp_id}_{model}_{oof_score:.5f}_{yyyymmdd_HHMM}.csv`

- `exp_id`: `experiments/log.csv` の `experiment_id` と紐付ける（省略可）
- `model`: ブレンド内容が分かる短い識別子（例: `lgb`, `lgb_cb_blend`, `greedy_ens`）
- `oof_score`: ファイル名だけで品質が分かるようにする
- タイムスタンプ: 同名ファイルの上書き防止と生成順の追跡

---

## OOF / test 予測ファイルの命名規約（派生アンサンブルの識別）

`data/output/oof/` に保存する `oof_{tag}.npy` / `test_{tag}.npy` は、
**「独立した基底モデルの予測」か「既存メンバーを結合した派生アンサンブルの出力」か**を
ファイル名だけで判別できる状態に保つ。

- **派生アンサンブル（ブレンド・スタッキング・rank average 等の出力）には必ず `_ens_` を含める**
  例: `oof_216_ens_stack_logit_pall.npy`, `oof_263_ens_p5_signed.npy`
- 独立した基底モデル（単一アーキテクチャの学習結果）には `_ens_` を含めない
  例: `oof_063_lgb_h012.npy`, `oof_144_realmlp_tuned_multiseed5.npy`

> **なぜ必須か**: プール自動探索（`discover_pool()` 等）は「派生アンサンブルを除外語の
> ブラックリストで弾く」実装になりがちだが、**新しい命名パターンを登録し忘れると
> 自分自身の出力を「独立メンバー」として再投入してしまう**（自己参照リーク）。
> 過去コンペでは同一コンペ内で 2 回発生し、2 回目は bagged OOF が異常値（seed 間 std が
> 通常の 17 倍）を示したことで偶然発覚した。異常値がレンジ内に収まっていれば
> 気づかず採用していた（→ `PLAYBOOK.md#教訓アーカイブ実測値つき` L-23）。
> `_ens_` マーカーを規約にすれば、除外側は 1 パターンの検査で済み、列挙漏れが構造的に消える。

**新しい派生アンサンブルのスクリプトを書いたら、プールへ投入する前に必ず確認する**:

```bash
python -c "import re; print(bool(re.search(r'_ens_', '<新しいファイル名>')))"  # True であること
```

---

## サブエージェントの運用規約

`.claude/agents/` に 4 つ定義している。いずれも**過去に実際に起きた失敗**（`PLAYBOOK.md` の `L-NN`）
に紐づく。

| エージェント | 解く失敗 | 起動する局面 |
|---|---|---|
| `fe-ideator` | L-15: brainstorm が既存変数の組み合わせに偏った | `/ds-fe-hypothesis` モード3。**視点ごとに並列** |
| `experiment-reviewer` | L-02 / L-16 / L-20: 不公正比較による誤棄却が 3 回 | `/ds-new-experiment` フェーズ3 の後、**学習を回す前** |
| `blocker-investigator` | L-12: ブロッカー 4 件を原因未特定のまま回避 | `G-BLOCKER` の局面。**回避策と並行して** |
| `kaggle-researcher` | L-07: 外部データを締切前日まで放置 | `/ds-kaggle-research` フェーズ0-2 |

**並列化してよい境界**（`G-STEPWISE` との整合）:

| 並列化してよい | 並列化しない |
|---|---|
| 調査・FE 仮説の生成・スクリプトの審査・ブロッカーの原因調査（成果物がテキスト） | 学習の実行・log.csv への記録・commit・提出 |

`G-STEPWISE` が縛るのは**実験の実行**。**仮説を 3 つ並列で「出す」ことは侵さないが、
3 つ並列で「回す」ことは侵す。**

**起動ポリシー**: **AI の裁量では起動しない。** 上表の局面で、ユーザーの承認を得てから起動する
（テンプレートの「1 実験ずつ確認を挟む」設計と整合させるため）。

**`tools` を絞る理由**: 「学習を実行しない・commit しない」を**指示ではなく道具で縛る**（`G-MECH`）。
`fe-ideator` と `experiment-reviewer` は読み取り専用（Bash を渡さない）。
この保証が壊れていないかは `doc_audit` の **C13** が検査する。

**成果物の統合**: 親が受け取り、**ユーザーに提示して選ばせる**。
`state/FE_HYPOTHESES.md` への登録・`state/KAGGLE_RESEARCH.md` への追記は**親が 1 件ずつ**行う。
エージェントは提案を返すだけで、状態ファイルを書き換えない。

---

## hook とガードの一覧

`.claude/settings.json` に **`statusLine`（ステータスバー）と 6 つの hook** を登録している。

**`statusLine`** は毎ターン（`refreshInterval` 秒ごとにも）呼ばれるため、
**同期ネットワーク呼び出しを一切しない**。提出枠は `deadline_status` が Kaggle API を
叩いたついでに書いたキャッシュ（`experiments/.statusline_cache.json`・TTL 10 分）を読むだけで、
無い/古い場合は「提出—」と表示する。失敗しても例外を投げず短い文字列を返す
（ここで落ちると毎ターンエラーが出る）。

規律は AI への指示ではなく
**観測可能な結果の側から**測る（CLAUDE.md `G-MECH`）。

| hook | 実行するもの | 何をするか |
|---|---|---|
| `SessionStart` | `scripts/harness/session_brief.py` | 現在地（ステージ・次アクション・直近の実験・要対応）を提示。`/ds-resume` の機械部分。行数上限は同ファイルの `MAX_LINES`（毎セッションのコンテキストを消費するため） |
| `PreToolUse` (Bash) | `scripts/harness/submit_gate.py` | Kaggle 提出コマンドを検知し、**実測した**提出枠・締切・git 状態を添えて**ユーザー承認を要求**（`permissionDecision: "ask"`） |
| `PostToolUse` (Bash) | `scripts/harness/viz_guard.py` | log.csv が 20 秒以内に更新されていたら 3 ガードを判定 |
| `Stop` | `scripts/harness/session_audit.py` | 未コミットの実験スクリプト・OOF 記録済みで未コミットの実験・状態ファイルの停滞・3 ガードを監査（**ブロックしない**） |
| `PreCompact` | `scripts/harness/session_snapshot.py` | コンテキスト圧縮の**直前**に、直近の実験・実行中ジョブ・git 状態を state/SESSION.md へ退避 |
| `PostCompact` | `scripts/harness/session_brief.py --event PostCompact` | 圧縮の**直後**に現在地を再注入する |

**ガードの手動実行**:

```bash
uv run python -m scripts.harness.viz_guard        # 可視化・診断記録・推論成果物・Public過剰浮上
uv run python -m scripts.harness.state_audit      # 状態ファイルの停滞（log.csv の実験時刻 vs mtime）
uv run python -m scripts.harness.session_audit    # 上記すべて + コミット規律
uv run python -m scripts.harness.session_brief    # 現在地ブリーフ
uv run python -m scripts.harness.doc_audit        # ドキュメント階層（11 チェック）
```

**設計上の約束**:

- **ブロックしてよいのは 2 つだけ** — 可視化ガード（`start_run()` が `RuntimeError`）と
  提出ゲート（不可逆な外部作用）。それ以外は警告に留める
  （Stop hook でブロックすると停止と再開のループを招く）
- 提出ゲートが検証するのは「人間が OK と言ったか」ではなく**「提示された数字が実測か」**。
  過去コンペの事故はすべて「提示された数字が記憶であって実測でなかった」ことが原因だった
- 提出コマンドの検知は **shlex でコマンド位置を判定**する。文字列として含むだけの
  Bash（ドキュメント編集・grep）を誤検知しない（導入時に実際に自分をブロックした）
- **`settings.json` は共有（hook 定義・git 管理下）、`settings.local.json` は個人の許可設定で
  git 管理外**（`.gitignore` 済み）。ローカル設定を追跡すると、個人の許可リストや
  そのセッション固有のパスが配布物に混入する
- hook が落ちても作業は止めない（入力が読めない・API 取得失敗時は素通しする）
- **ハーネスはブランチに閉じている。** `.claude/settings.json` も `scripts/*.py` も git 管理下なので、
  **別ブランチに切り替えると hook 構成ごと入れ替わる**（コンペブランチには存在しないことがある）。
  そのため各 hook は先頭で `[ -f scripts/<script>.py ] || exit 0` を通し、
  **スクリプトが無いブランチでは黙って素通しする**（無いと PreToolUse が毎回の Bash でエラーを出す）。
  スクリプトが在るのに落ちた場合はエラーを出す——存在しないことと壊れていることは区別する
- **stdin を読む hook は TTY を検知して即終了する。** `json.load(sys.stdin)` は stdin が
  閉じられないとブロックし続け、PreToolUse なら毎回の Bash を timeout 秒ハングさせる
- **各 hook は先頭で発火時刻を `experiments/.hook_log` へ追記する。**
  「登録した hook が本当に発火しているか」は設定ファイルを見ても分からない——とくに
  ①走行中セッションに設定変更が反映されるか ②**自動圧縮**でも `PreCompact` / `PostCompact` が
  呼ばれるか、の 2 点。推測で埋めず `uv run python -m scripts.harness.hook_status` で実測を見る
- **`PreCompact` と `PostCompact` は対で使う**。長時間セッション（夜間の学習を回し続ける等）では
  新セッションが滅多に始まらない代わりに圧縮が繰り返し起きる。圧縮は数万トークンを要約へ潰すので、
  直後にブリーフ十数行を戻す費用は失うものの 1% 未満——**「コンテキスト節約 vs 文脈欠如」は
  釣り合っておらず、再注入する側が明確に得**。`PreCompact` が state/SESSION.md へ恒久記録し、
  `PostCompact` が読み直す（注入が効かなかった場合の保険として state/SESSION.md 側が残る）

---

## スキル呼び出しが無くても守るプロトコル

CLAUDE.md「セッションの作法」が求める対応表。**スキル呼び出しは「このプロトコルで進めて」という
ユーザーの意思表示であって、規律の起動条件ではない。**

**CLAUDE.md の原則は常に active。**「skill 経由ではないから略式で OK」は**禁止**。
スキル呼び出しはユーザーが「このプロトコルで進めて」と意思表示する儀式であって、規律の起動条件ではない。

| 場面 | スキル呼び出しが無くても従うプロトコル |
|---|---|
| 新規実験開始 | `ds-new-experiment` フェーズ1-2: 目的・成功基準・撤退基準を**実験開始前に**明文化 |
| FE 仮説立案 | `ds-fe-hypothesis` の因果連鎖言語化: 「なぜ効くか」を仮説段階で記録 |
| Kaggle 提出前後 | `ds-kaggle-submit` フェーズ1-4: 提出前確認 → Plateau 検出 → OOF-LB gap の解釈 → learning 必須記入 |
| EDA | `ds-eda-visual` の「問い → 発見 → FE 仮説の種」3 段階 |
| 最終日 | `ds-kaggle-submit` フェーズ5: Final 2 Persona 投票 |
| 学習完了・OOF 判明直後 | 「OOF=X、前ベスト比 ΔOOF=±X。commit しますか？」を**能動提示**（→ `G-STEPWISE`） |
| 実験 3 回完了ごと | state/SESSION.md の「直近の実験」を更新し、次の方向を問いかける |
| FE 棄却 3 連続 | Discussion 調査を提案し「未試行の情報次元」を自発的に列挙（→ `G-PERSIST`） |

**スキルを呼ぶべき節目**（AI は能動的に提案する）: セッション開始 `/ds-resume`（**必須**・文脈リセット防止）／ `/ds-kickoff`（参加直後 1 回）／ **主軸アーキテクチャ決定の前 `/ds-kaggle-research`**（Stage 1.5 の前提入力。序盤の外部調査を省くと伸びしろの所在に気づくのが遅れる）／ FE 棄却 3 連続・LB プラトー時 `/ds-kaggle-research`／ 振り返り `/ds-template-update`。**AI から起動できないのは `/ds-kaggle-submit` だけ**（不可逆な外部作用のため `disable-model-invocation` を維持。他は読み取り・記録のみなので AI 起動可）。submit がヒットしたら黙って止まらず理由を明示し、代替（下記の手動チェックリスト）を提示する。

---

## 学習サイクルの 11 ステップ

**すべての実験は独立したイベントではなく、学習サイクルの一部として扱う。**
以下は順序と「次に進んでよい」判断の一覧（規範は CLAUDE.md「絶対規約」）。
**行き詰まったらステージを戻ることを厭わない**（サイクルは一方向ではない）。

| # | ステップ | 移行の条件（次に進んでよい判断） |
|---|---|---|
| 1 | `/ds-resume` | **新セッション開始時は必ず呼ぶ。** state/SESSION.md + log.csv + state/FE_HYPOTHESES.md 索引で現在地を復元 |
| 2 | `/ds-kickoff` | 「そのデータが何者か」を文脈から理解（参加直後に一度だけ）→ state/COMPETITION.md に記録 |
| 3 | `/ds-new-experiment` | 最小ベースライン（前処理不要な数値カラムのみ・デフォルト HP） |
| 4 | `/ds-kaggle-submit` | LB 提出で **CV/LB 相関を確立**。以降すべての改善はこの基準点からの Δ で判断する |
| 5 | `/ds-kaggle-research` | 上位解法のアーキテクチャ分布を調べる（フェーズ0）。**主軸決定の前提入力にする** |
| 6 | Stage 1.5 | 早期アーキテクチャサーベイ → OOF と pub_oof_gap を記録し**主軸を 1 つ決定** |
| 7 | `/ds-eda-visual` | 「何を知りたいか」を先に言語化する（Kickoff の記録と基準点を持ち込む）|
| 8 | Optuna 軽量 | 作業用 HP（20〜30 試行）。目的は最適化ではなく **ΔOOF 計測のノイズ低減** |
| 9 | `/ds-fe-hypothesis` | 因果連鎖を言語化 → **必ず 1 列ずつ** `feature_study.py` で ΔOOF 計測（複数列の一括追加は禁止）。ΔOOF が閾値以下でも importance (gain) を確認してから棄却する |
| 10 | Optuna フル | FE 収束後（追加 FE の ΔOOF が `G-NOISE` の閾値未満 かつ importance が BASE 最下位未満 が続いたら）100 試行以上 |
| 11 | `/ds-kaggle-submit` | OOF/LB のギャップを解釈し「学び」を言語化 → 次サイクルの仮説を更新 |

---

## 作業ステージのゲート条件

CLAUDE.md「作業ステージとゲート」の完了条件（規範は L0、ここは条件の一覧）。
**次のステージへ進む前に、その行の完了条件を満たしているか確認する。**

| Stage | 目的 | 完了条件 | スキル・ツール |
|---|---|---|---|
| **0. Kickoff** | データの文脈理解 | `state/COMPETITION.md` にデータ種別・外部データ有無・評価指標特性・CV設計の初期判断を記録済み | `/ds-kickoff` |
| **1. 最小ベースライン** | CV/LB相関の確立 | 前処理不要な数値カラムのみ・デフォルトHPでモデルを学習し、LBに提出してCV/LB相関を確認済み。以降すべての改善はこの基準点からのΔで判断する | `/ds-new-experiment` + `/ds-kaggle-submit` |
| **1.5. 早期アーキテクチャサーベイ** | 主軸アーキテクチャの決定 | 候補アーキテクチャ（Tree/NN/Linear等）を最小特徴量セット + 作業用HPで評価し、OOFとpub_oof_gapを記録。「主軸アーキテクチャ」を1つ決定済み。手順は `PLAYBOOK.md#早期アーキテクチャサーベイの手順stage-15` に従い実施 | `/ds-new-experiment` |
| **2. EDA** | 問いとFE仮説の種を獲得 | `/ds-eda-visual` で「問い→発見→FE仮説の種」の対話完了。合成データの場合は元データとの分布比較も含む | `/ds-eda-visual` |
| **3. 作業用HP調整** | FE計測の安定化 | Optuna 20〜30試行でFE実験中に使う「作業用HP」を確定済み。目的は完全最適化ではなくΔOOF計測のノイズ低減。**不安定な大型アーキでは単一 fold・サブサンプルでの HP 選定を禁止**（→ `G-FULLCV`）。study は SQLite に永続化され同じ tag で追加試行できる（→ `CONVENTIONS.md#optuna-study-の永続化と命名`） | Optuna（軽量） |
| **4. 段階的FE** | 有効な特徴量の特定 | `state/FE_HYPOTHESES.md` に採用・棄却含む仮説5件以上、棄却理由が分類記録済み。**特徴量は必ず1列ずつ** `scripts/feature_study.py` で投入し ΔOOF と importance を計測済み。AV 診断で分布シフト確認済み。FE 確定後、全候補アーキテクチャへ移植して再評価済み（詳細 → `PLAYBOOK.md#stage-4-stage-5-のゲート詳細`） | `/ds-fe-hypothesis` + `scripts/feature_study.py` + AV診断 |
| **5. 本格HP最適化** | 確定特徴量での性能最大化 | 特徴量セット確定後に Optuna 100 試行以上を実施し、ΔOOF が指標別閾値以内（`G-NOISE`）で収束済み。FE が ±20% 以上変動したら HP retune を再実行する。**単体ベストを state/SESSION.md に記録する**（詳細 → `PLAYBOOK.md#stage-4-stage-5-のゲート詳細`） | Optuna（フルサーチ） |
| **6. アンサンブル** | モデル多様性の活用 | 特徴量・HP飽和を確認済み。手順は `PLAYBOOK.md#アンサンブル探索の手順stage-6` に従い実施済み | `src/utils/ensemble.py` |

**Stage 6 の判断原則**（実行手順は L2、ここは着手可否の判断材料）:

- **着手前に問う**: 「単体モデルで要件を満たすか。アンサンブルの複雑性コストに見合う伸びがあるか」
  （実務デプロイ観点。天井帯では単体強化に戻る判断も有効）
- **相関確認を最初に**: 追加候補と既存モデルの OOF 相関 **≥ 0.998 なら実装前にスキップ**（コスト浪費防止）
- **多様性 > 単体 OOF**: blend 改善には「誤差の方向が違う」ことが必要。高 OOF でも誤差が相関すれば無意味
- **Multi-seed avg5 をデフォルト化**: tree モデルは **+0.00010〜0.00020 OOF** の安定改善（CB は特に効果大）
- **Pseudo-labeling のリーク診断必須**: OOF↑ かつ LB↓ なら leakage 確定。源泉を train fold 内に変更する

**ステージを飛ばすと何が起きるか**（CLAUDE.md「ステージを飛ばさない」の根拠）:

- Stage 1 を省くと **CV/LB 乖離に気づくのが遅れる**
- Stage 3 を省くと Stage 4 の **ΔOOF 計測がノイズに埋もれる**
- Stage 4 で `feature_study.py` を使わず複数列を一度に追加すると、**どの特徴量が効いたか分からなくなる**
- Stage 5 は Stage 4 完了後でないと **最適 HP が変わるため意味が薄い**
- Stage 6 の相関確認を省くと、実装・学習コストをかけてから「重みゼロ」と判明する

---

## 実験管理（log.csv）

`experiments/log.csv` の主要カラム:

| カラム | 記録タイミング | 説明 |
|---|---|---|
| `experiment_question` | `/ds-new-experiment` | この実験で何を明らかにしたいか |
| `success_criteria` | `/ds-new-experiment` | どんな結果なら成功か |
| `abort_criteria` | `/ds-new-experiment` | どんな結果なら中止するか |
| `cv_val_mean` / `oof_score` | 学習完了時 | OOFスコア |
| `duration_sec` | 学習完了時（自動） | `start_run` → `end_run` の実測秒数。**30分ルールの推定はこの実測から取る**（`uv run python -m scripts.harness.deadline_status` がモデル別中央値を表示） |
| `submit_score` | `/ds-kaggle-submit` | LBスコア |
| `oof_lb_gap` | `/ds-kaggle-submit` | OOF tuned − LB（正=OOF過大評価、負=OOF過小評価）。乖離が大きい実験は汎化リスクあり |
| `learning` | `/ds-kaggle-submit` | この実験から何を学んだか |

> **ベスト実験の管理は state/SESSION.md のスコアテーブルで一元化する。** log.csv にベストフラグ列（`is_best` 等）を持たない
> — フラグ方式は過去コンペで 100 実験超のうちほぼ全行が未記入となり形骸化した。二重管理をやめ、state/SESSION.md の上書き更新に集約する。

**列を追加するとき**: `LOG_CSV_COLUMNS`（`src/experiment.py`）に足すだけでよい。
`_ensure_log_csv()` が既存ファイルのヘッダを見て不足列を空値で補い、行を保ったまま移行する。
**この移行を省くとヘッダと行の列数が食い違い、過去の実験記録が丸ごとずれる。**

---

## ExperimentTracker の使い方

すべての学習実行は `src.experiment` モジュールを通じて `experiments/log.csv` に記録する。

```python
from src.experiment import ExperimentTracker
from src.config import EXPERIMENT_NAME

tracker = ExperimentTracker(experiment_name=EXPERIMENT_NAME)
run_id = tracker.start_run(
    run_name="fold_training",
    description="実験の概略をここに記述",
    model="lgb",
    features="baseline_features",
)

# 学習ループ内で記録
tracker.log_fold_scores(fold, trn_score, val_score)

# 学習完了後
tracker.end_run(
    train_scores=train_scores,
    val_scores=val_scores,
    oof_score=oof_score,
    n_features=len(features),
)
```

## 実験名の命名規則

`{competition}_{model}_{feature_set}_{variant}`

例:
- `<competition-slug>_lgb_baseline`
- `<competition-slug>_lgb_fe_v2`
- `<competition-slug>_cb_optuna`
- `<competition-slug>_ensemble_lgb_cb`

---

## 記録先の対応表

CLAUDE.md「思考の外部化の原則」が求める記録を、どのファイルへ・どのスキルで残すか。

| 記録すべきもの | どこに | スキル |
|---|---|---|
| 「何を知りたいか」「ドメイン知識」 | state/EDA_SUMMARY.md | `/ds-eda-visual` |
| 各変数の特性・ΔOOF・採否 | state/FEATURE_REPORT.md | `/ds-eda-visual` · `/ds-fe-hypothesis` が記入を促す。**「現在の特徴量セット」節は機械生成**（`end_run(feature_names=...)` → `scripts.feature_report --sync`）。採否の表は自由形式の検証でも確定時に手動更新する — 過去コンペでは 3 週間超の FE 成果が未反映のまま「今どれがベースか」を追えなくなった |
| 特徴量の仮説・因果・棄却理由 | state/FE_HYPOTHESES.md | `/ds-fe-hypothesis` |
| 実験の目的・成功基準・撤退基準 | experiments/log.csv | `/ds-new-experiment` |
| 実験から何を学んだか | experiments/log.csv | `/ds-kaggle-submit` |
| テンプレートへの汎用的な気づき | docs/TODO_TEMPLATE.md | `/ds-template-update` |
| **現在地・次のアクション・未解決の問い** | **state/SESSION.md** | **`/ds-new-experiment` · `/ds-kaggle-submit` が自動更新** |

---

## SESSION.md の構成と上限

**state/SESSION.md の更新タイミング（自動）:**
- `/ds-kickoff` 実行時 → Stage 0 完了・次のアクション（最小ベースライン）を記録
- `/ds-eda-visual` 実行時 → Stage 2 完了・次のFE仮説リストを記録
- `/ds-fe-hypothesis` 実行時（新規） → 仮説登録・次のアクション（実装→計測）を記録
- `/ds-new-experiment` 実行時 → 実験開始・次のアクションを記録
- `/ds-kaggle-submit` 実行時 → LBスコア・OOF-LB乖離・**本日の提出数（例: 3/10）**・学び・次の方向性を記録

state/SESSION.md は「今どこにいるか」を1画面で示すライブダッシュボード。
**アペンドではなく各セクションを上書き更新する**（蓄積禁止）。

**固定構成**（この順序を守り、追加セクションを作らない）:

| # | セクション | 上限・注意 |
|---|---|---|
| 1 | ファイルヘッダー | 最終更新日時 |
| 2 | 現在のステージ | 1〜2行。「次にやること」を1行目に |
| 3 | スコア状況 | **ベストのみ1テーブル**（新テーブル追加禁止） |
| 4 | 直近の実験 | **最大10件**。11件目以降は最古から削除 |
| 5 | 次にやること | 箇条書き最大5件 |
| 6 | 未解決の問い | ブロッカーのみ。解決済みは削除 |
| 7 | 重要な方針 | 実験を通じて確定した原則のみ |

**スコアテーブルの形式**（乖離列で「OOFは高いがLBで悪化」を早期検知する）:

| 指標 | OOF tuned | LB | OOF-LB乖離 | 実験ID |
|---|---|---|---|---|
| ベスト | 0.XXXXX | 0.XXXXX | ±0.XXXXX | expNNN |
| **単体ベスト** | 0.XXXXX | 0.XXXXX | ±0.XXXXX | expNNN |

**「単体ベスト」行は常設**（アンサンブルを組んでいる期間も維持する）。理由は 2 つ:
①実務デプロイでは 400 メンバーのスタックより単体モデルの方が扱いやすく、単体精度そのものに価値がある
②アンサンブルの伸びが止まった局面で「単体を鍛え直す」という選択肢を視野から外さないため（→ CLAUDE.md Stage 5）

**上限値の定義**: **ファイル全体は 80 行**、そのうち **「直近の実験」は最大 10 件**。
この 2 つは別物であり両立する（v5 では別々のファイルに書かれ非同期だった）。
`/ds-resume` が 80 行超過を検知したら、完了済みエントリを削除して収める。

**例外**: `<!-- BEGIN:auto-snapshot -->` 〜 `<!-- END:auto-snapshot -->` のブロックは
上限の対象外。PreCompact hook（`scripts/harness/session_snapshot.py`）が**追記ではなく置換**するため
蓄積せず、内容も直近の実験・実行中ジョブ・git 状態に限定され行数が構造的に有界だから。
手で編集しない（次回の圧縮で上書きされる）。

**禁止パターン**: 「最後に完了したこと」を複数回追記する / 複数のスコアテーブルを並存させる /
過去セッションの履歴を蓄積する（git history に残るため state/SESSION.md には不要）。

---

## 改善を 4 層のどこに入れるか

CLAUDE.md「テンプレート改善プロトコル」の振り分け表。**判断基準は L0、ここは対応の一覧。**

| 内容の性質 | 行き先 | 例 |
|---|---|---|
| 規範・判断基準・閾値 | **L0 `CLAUDE.md`** | 「〜せよ」「〜は禁止」「相関 0.998 以上ならスキップ」 |
| パス・命名・列定義・コマンド規約 | **L1 `CONVENTIONS.md`** | ディレクトリ、log.csv の列、コミット形式 |
| 実行手順・コード・**教訓の本文** | **L2 `PLAYBOOK.md`** | STEP 1-8、ノイズ床の計算、`L-NN` の実測記録 |
| 対話の進行・質問文面 | **L3 `.claude/skills/*`** | フェーズ構成、ユーザーへの問いかけ |

**振り分けの判断基準**: 「**毎ターン参照すべきか**（= L0）」「**引くだけか**（= L1）」
「**その局面で読めばよいか**（= L2 / L3）」。

**SSoT 4 原則**（同じ内容が複数箇所に散ると、更新時に必ず食い違う）:

1. **1 情報 = 1 ファイル** — 重複したら下の層を削り、上の層への参照に置換する
2. 参照の向きは常に「下 → 上」
3. **教訓の本文は `PLAYBOOK.md#教訓アーカイブ実測値つき` のみ。** 他所からは
   「結論 1 文 + 鍵の数値 + リンク」で参照する。**数値は残す** ——
   規範を「守るべきもの」に変える payload だから
4. **L0 にコードフェンスを書かない**（手順を書きたくなったら L2 に置くべきサイン。`doc_audit` C9）

**L0 の退避ポリシー — 入れた分の「量」を出す。** CLAUDE.md は上限 5,000 字（`doc_audit` C1 が ERROR）。
**行数で測ってはいけない** —— 2 つの箇条書きを 1 行に結合すれば行数は減るが中身は減らない
（実際それで上限を素通りした: 行数 −2 に対し文字数 +1,898）。出す候補は ①実行手順・コード例
②詳細な列挙（閾値の根拠・例示）③1 コンペ限定の記述。**削るのが惜しいものほど L2 へ移す**
（消すのではなく参照に置換する）。L2 に手順を足したら、L1/L3 の対応する箇所からリンクを張る。

---

## main マージ前チェックリスト

コンペ用の改良をテンプレート本体（`main`）へ戻すときは、コンペ固有の痕跡を落とす。

- [ ] コンペ名・ターゲット列のハードコードを `src/config.py` の変数に置換
- [ ] 回帰・分類の両方に対応（またはどちらか明記）
- [ ] 新依存関係を `pyproject.toml` に追加済み
- [ ] カスタマイズ箇所を `# TODO:` コメントで明示
- [ ] `uv run python -m scripts.harness.doc_audit` が ERROR 0（C7 がコンペ識別子の混入を検出する）

---

## ブランチ管理

```
main              ← テンプレート本体（コンペ固有コード禁止）
comp/<competition> ← コンペ適用ブランチ（日々の実験コミットの置き場）
exp/<実験名>      ← 大きな方向転換のみ（下記基準参照）
template/fix-XXX  ← テンプレ改善ブランチ
```

**`exp/` ブランチを作る基準（すべての実験には不要）:**

| 作る | 作らない |
|---|---|
| 新しいアルゴリズムの追加（XGB, NN, RF など） | FEの1列追加 |
| 特徴量セットの大幅再設計（列数 ±20% 以上） | HPチューニング（Optuna） |
| CV戦略の変更（StratifiedKFold → GroupKFold など） | ブレンド重みの調整 |
| アーキテクチャ変更（Stacking の試験的導入） | 既存スクリプトのバグ修正 |

→ 上記に当てはまらない実験は `comp/<competition>` ブランチ上でコミットしてよい。

---

## コミット規約

**コミットのタイミング（3つのルール + 並行実行ルール）:**

1. **学習完了直後にコミットする** — OOFスコアが判明した直後 **5 分以内**。時間を置かない
2. **1実験 = 1コミット** — 複数の変更を一度のコミットにまとめない。何が効いたか追跡できなくなる
3. **`/ds-kaggle-submit` の前にコミット済みであること** — `git status` がcleanでなければ提出しない

**並行実行時の特例ルール（バックグラウンド実行時も厳守）:**

複数の実験をバックグラウンドで並行実行している場合でも、**各実験の OOF 判明ごとに個別 commit する**:

```
❌ NG パターン:
  exp_A 完了 → 待機 → exp_B 完了 → 待機 → exp_C 完了 → まとめて 1 commit

✅ OK パターン:
  exp_A 完了 → commit_A → exp_B 完了 → commit_B → exp_C 完了 → commit_C
```

待ち時間の活用:
- バックグラウンド実行中の「次の実験設計」は OK
- しかし完了した実験の commit は **絶対に後回しにしない**
- log.csv の更新も同じタイミングで（バッチ更新は禁止）

> **教訓 (過去事例)**: 7 実験を 1 コミットにまとめ、log.csv 更新を最終日に一括実施した結果、後追いで「どの変更が効いたか」が追跡困難になった

**実験番号の衝突防止:**

新しい実験番号を決める前に、必ず以下で既存ファイルを確認する:
```bash
ls experiments/runs/ | grep "^exp" | sort | tail -5
# log.csv の最大 experiment_id も確認
tail -3 experiments/log.csv | cut -d',' -f2
```
未コミットの実験スクリプトが `experiments/runs/` に存在する場合（`git status` で `??` 表示）、
それらの番号は使用済みとして扱い、それより大きい番号を使う。

**コミットメッセージの形式:**

```
feat(expNNN): <実験の目的を1文で>

OOF=<score>  model=<model>  features=<feature_set>
```

例:
```
feat(exp042): col_A×col_B の交互作用特徴量を追加

OOF=0.91688  model=lgb  features=fe_v7_interaction
```

- `expNNN` は `experiments/log.csv` の `experiment_id` と一致させる
- 本文行（2行目）は `tracker.end_run()` が自動提案する
- `feat` / `fix` / `refactor` を使い分ける（FE追加=feat, バグ修正=fix, リファクタ=refactor）

---

## 付録A. 旧番号と恒久 ID の対応表

v5 までの `指針#N` は v6 で恒久 ID に置き換えた。**この表は v6 限りの移行用**（v7 で削除する）。
過去の state/SESSION.md・log.csv・state/FE_HYPOTHESES.md に残る旧番号を読むときに使う。

| 恒久 ID | 旧番号 | 主題 |
|---|---|---|
| `G-PURPOSE` | #1, #3 | 目的を先に言語化し、直前の学びに接続する |
| `G-CAUSAL` | #2 | FE は「なぜ効くか」を説明できるものだけ提案する |
| `G-STEPWISE` | #6, #15 | 1 実験 = 1 つの問い = 1 コミット |
| `G-PERSIST` | #4, #7, #8, #12, #14 | 「飽和」を宣言しない。問いを立て直す |
| `G-SOURCE` | #5, #10, #11 | モデルの多様性より情報源の多様性 |
| `G-FAIR` | #13, #22 | 棄却の前に比較条件を揃える |
| `G-DIAG` | #31 | CV 内部診断を常設の判断軸にする |
| `G-NOISE` | #17, #23 | その差は測れるのか（量子とノイズ床） |
| `G-OOF` | #18 | OOF は足切りに使い、1 位の選定には使わない |
| `G-TWOAXIS` | #21 | OOF 最大化と pub_oof_gap 最小化の二軸評価 |
| `G-OVERFIT` | #20, #24, #25 | 「OOF↑なのに LB↓」の 3 変奏 |
| `G-FULLCV` | #26 | 単一分割の兆候はフル CV で再確認する |
| `G-INFOCEIL` | #28 | 情報天井の判定と情報源への切り替え |
| `G-CEILING` | #19, #27 | 天井帯では「当てにいく」のをやめ「集約する」 |
| `G-CALIB-SUB` | #29 | 提出枠を判断基準の検証にも使う |
| `G-MECH` | #9, #16 | 規律は機械に守らせる |
| `G-BLOCKER` | #30 | 技術的ブロッカーを「回避＝解決」としない |

**なぜ番号をやめたか**: 指針を統合・並べ替えるたびに `#N` の参照が全ファイルで壊れ、v5 では
「ヘッダーが `#1-22` のまま」「`#20` と `#17` の取り違え」等のバグが 8 件蓄積していた。
恒久 ID なら統合しても参照が壊れず、未定義 ID は `scripts/harness/doc_audit.py` の C3 が機械的に検出する。
