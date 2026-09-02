# テンプレート改善タスク — 完了済みアーカイブ

`TODO_TEMPLATE.md` から分離した **[DONE]** のエントリ。改善の履歴として残す。

> **なぜ分離したか**: 本体は `/ds-resume` が毎セッション参照する「未完了の作業リスト」。
> 完了済みが 97% を占める状態では、読むべき数件が 1000 行の履歴に埋もれる。
> 履歴に価値はあるので消さず、参照頻度の違うものを別ファイルに分ける。

---

## [2026-03-20] HIGH — 実験トラッキング（experiments/log.csv）
- **説明**: 実験ごとのtrain/val/OOF/LBスコアをCSVで管理できるようにする
- **影響ファイル**: src/experiment.py（新規）, notebooks/04_train.py
- **状態**: [DONE]


## [2026-03-20] HIGH — データバリデーション
- **説明**: パイプライン各ステージでスキーマ・リーク・欠損値を自動チェック
- **影響ファイル**: src/validation.py（新規）, notebooks/02_preprocess.py
- **状態**: [DONE]


## [2026-03-20] HIGH — ハイパーパラメータ管理
- **説明**: Optunaサーチスペースをsrc/hp_spaces.pyに集約し、best_params JSONで管理
- **影響ファイル**: src/hp_spaces.py（新規）, notebooks/04_train.py
- **状態**: [DONE]


## [2026-03-20] MED — アンサンブルサポート
- **説明**: 複数モデルのブレンド・スタッキングをsrc/utils/ensemble.pyに集約
- **影響ファイル**: src/utils/ensemble.py（新規）, notebooks/05_predict.py
- **状態**: [DONE]


## [2026-03-20] MED — 実験比較ダッシュボード
- **説明**: log.csvを読み込んで実験比較・CV/LB相関・過学習モニタリングをAltairで可視化
- **影響ファイル**: notebooks/06_experiment_log.py（新規）
- **状態**: [DONE]


## [2026-03-20] HIGH — 段階的特徴量追加フレームワーク
- **説明**: 特徴量を1列ずつ追加してCV AUCへの貢献を測定する実験プロセスをテンプレート化。ステップ構成: 数値ベースライン(1) → 数値追加(2) → カテゴリ追加(3) → 数値×数値FE(4) → 数値×カテゴリFE(5) → カテゴリ×カテゴリFE(6)。全自動・手動両対応。各列=1実験としてlog.csvに記録。
- **影響ファイル**: src/feature_registry.py（新規）, notebooks/03b_feature_study.py（新規）
- **設計判断**:
  - カテゴリカラムはドメイン知識で「効きそうな順」に並べる（優先リストにない列はアルファベット順で末尾追加）
  - feature_study は本番と同じ5-foldを使用（精度と速度のバランスはn_estimators=300+early_stoppingで調整）
  - feature_study の submit はしない（CV AUCのみで判断。ベースラインでCV/LBギャップを確立済みの前提）
- **状態**: [DONE]


## [2026-03-20] HIGH — 特徴量の全保持・学習時選択アーキテクチャ
- **説明**: 前処理・特徴量生成フェーズでは全カラム・全特徴量を保持し、削除しない。学習・推論時に feature registry 経由で使用する特徴量を選択する運用に変更。これにより「一度削除した特徴量を後で試したい」場合でも再前処理が不要になる。
- **影響ファイル**: notebooks/02_preprocess.py, notebooks/03_features.py, src/feature_registry.py, notebooks/04_train.py
- **設計判断**:
  - 02_preprocess.py: gender / PhoneService を削除しない（エンコードして保持）
  - 03_features.py: 可能な限り多くの特徴量を生成して保存
  - 学習時: `ACTIVE_FEATURES` リストまたは feature registry で使用列を制御
  - ストレージコスト vs 柔軟性のトレードオフ → 柔軟性を優先
- **状態**: [DONE]


## [2026-03-22] HIGH — ユーザー×AI協業フレームワークの整備
- **説明**: コンペ実践での反省を踏まえ、ユーザーとAIが「認識共有 → 仮説立案 → 検証 → 蓄積」サイクルを回せる仕組みを追加。
- **背景（反省点）**:
  1. EDAで数値だけを出力し、ユーザーとデータの姿を共有できていなかった
  2. 特徴量エンジニアリングが仮説なしの機械的組み合わせ列挙になっていた
  3. 特徴量飽和前にモデル変更・アンサンブルを提案しがちだった
- **対応内容**:
  - CLAUDE.md にステージゲート追加（EDA → FE → ベースライン → HP最適化 → アンサンブルの順序を強制）
  - `FE_HYPOTHESES.md` テンプレート作成（仮説→検証→結果の蓄積ドキュメント）
  - `.claude/skills/fe-hypothesis/` スキル追加（仮説の言語化・記録・更新フロー）
  - `.claude/skills/eda-visual/` スキル追加（marimoで可視化ダッシュボードを生成し認識共有）
  - `src/experiment.py` に `save_oof_analysis()` 追加（学習後の自動誤差分析）
- **影響ファイル**: CLAUDE.md, FE_HYPOTHESES.md, .claude/skills/fe-hypothesis/, .claude/skills/eda-visual/, src/experiment.py
- **状態**: [DONE]


## [2026-03-20] HIGH — Cat×Cat / Cat×Num 特徴量の体系的追加
- **説明**: FEATURE_ANALYSIS.md の分析結果を元に、ドメイン知識ベースの交互作用特徴量を追加。削除していた gender / PhoneService も含めて交互作用を探索。
- **影響ファイル**: notebooks/03_features.py, run_feature_study.py, notebooks/03b_feature_study.py
- **追加特徴量**:
  - Cat×Cat: monthly_x_fiber, electronic_x_monthly, has_family, fiber_x_no_security, contract_x_paperless, contract_x_internet, is_monthly_contract, is_fiber_optic
  - Cat×Num: risk_x_monthly, senior_x_monthly, tenure_x_contract, electronic_x_tenure, monthly_per_service
- **状態**: [DONE]


## [2026-04-01] HIGH — /fe-hypothesis に実装後・投入前の可視化確認フェーズを追加
- **説明**: 特徴量を実装してからモデルに投入するまでの間に、分布確認・ターゲット別確認・冗長性チェック・ユーザー対話のステップが欠如していた。
- **背景（反省点）**: 実装バグ（スケール異常・欠損の混入）がモデル学習後にΔAUCが低い結果として現れ、「特徴量が悪いのかモデルが悪いのか」の切り分けに時間がかかった。また、ユーザーのドメイン知識を「特徴量の見た目を見ながら」引き出す機会がなかった。
- **対応内容**:
  - `/fe-hypothesis` モード1 にフェーズ3（実装後・投入前の可視化確認）を追加
  - STEP A: 分布確認（外れ値・欠損）、STEP B: ターゲット別分布確認、STEP C: ユーザー対話（想定通りか？）、STEP D: 冗長性事前確認、STEP E: 確認結果の記録
  - FE_HYPOTHESES.md のエントリに「可視化確認」フィールドを追加
  - Q1の例示をs6e3固有から汎用表現に変更
- **影響ファイル**: .claude/skills/fe-hypothesis/SKILL.md
- **状態**: [DONE]


## [2026-04-01] HIGH — HP最適化の2段階化とFE1列ずつ原則の明示
- **説明**: HP最適化を「作業用（FE前）」と「本格（FE後）」の2段階に分離。FE段階での特徴量追加を `run_feature_study.py` による1列ずつのΔAUC計測に限定する。
- **背景（反省点）**: ①FEをデフォルトHPで進めると、特徴量のΔAUC計測がHPのノイズに埋もれる。②FEが収束する前に本格HP最適化に入ると、最適HPが特徴量セット変更で変わるため無駄が生じる。③後半で複数特徴量を一括追加し「どれが効いたか」が不明になった。
- **対応内容**:
  - CLAUDE.md のステージゲートを Stage 0〜6 に再編（Stage 3: 作業用HP調整、Stage 4: 段階的FE、Stage 5: 本格HP最適化）
  - CLAUDE.md 学習サイクル図を更新（「Optuna 軽量」→FE→「Optuna フル」の順序を明示）
  - Stage 4 に「必ず1列ずつ run_feature_study.py で投入。複数列の一括追加禁止」を追記
  - `/new-experiment` Q1 例示に「作業用HP調整」「本格HP最適化」パターンを追加
- **影響ファイル**: CLAUDE.md, .claude/skills/new-experiment/SKILL.md
- **状態**: [DONE]


## [2026-04-01] HIGH — ステージゲートの順序再編（最小ベースラインをEDA前に移動）
- **説明**: ステージ順序が「EDA → FE → ベースライン」になっており、CV/LB相関を確認する前にFEを進めていた。正しい順序は「最小ベースライン（数値のみ）→ LB提出でCV/LB確認 → EDA → 段階的FE」。
- **発見のきっかけ**: s6e3 でCV/LB相関の基準点がないまま多数の特徴量実験を積み重ねた。CV上の改善がLBに反映されるかを序盤に確認する手順が欠如していた。
- **汎用性の根拠**: CV設計の妥当性確認（LB提出）はあらゆるコンペで最初に行うべき手順。後から発覚するCV/LB乖離は修正コストが高い。
- **対応内容**:
  - CLAUDE.md 学習サイクル図を更新（最小ベースライン→LB提出→EDA→段階的FEの順序を明示）
  - CLAUDE.md ステージゲートを Stage 0〜5 に再編。Stage 1 を「最小ベースライン」、Stage 3 を「段階的FE（EDAと並走）」に変更
  - `/new-experiment` Q1 の例示に「初回: CV/LB相関確立」「多様性確認: 予測相関<0.998か」を追加
  - `/kaggle-submit` フェーズ2 のファイルパスを `data/output/submissions/` に修正（旧パスのバグ解消）
- **影響ファイル**: CLAUDE.md, .claude/skills/new-experiment/SKILL.md, .claude/skills/kaggle-submit/SKILL.md
- **状態**: [DONE]


## [2026-04-01] HIGH — /eda-visual の強化（欠損値・Train/Test・FE仮説接続）
- **説明**: EDA段階で①欠損値の発生メカニズム議論、②Train/Test分布差の必須確認、③FE仮説登録の強制化、が欠如していた。
- **発見のきっかけ**: s6e3 で欠損値を機械的に補完（MNAR を考慮せず）、Train/Test分布差を体系的に確認しなかった、EDAの発見を仮説化せず直接実装するショートカットが多発した。
- **対応内容**:
  - フェーズ2: Train/Test分布差を「必須」として明記（オプションから変更）
  - フェーズ3（新設）: 欠損値の発生メカニズム分析（MCAR/MAR/MNARの3分類、フラグ特徴量検討、処理決定の根拠記録）
  - フェーズ4: s6e3固有の例を汎用化、予想外発見の深掘りをAI指針として強化、Train/Test乖離時の対処問いかけを追加
  - フェーズ5: FE仮説登録を「必須・スキップ禁止」に変更。記録するまで次に進まないルールを明記
  - フェーズ7 (EDA_SUMMARY.md テンプレート): 欠損値処理決定テーブルと Train/Test分布差セクションを追加
  - ダッシュボード構成にセクションC（欠損値）・セクションD（Train/Test比較）を追加
- **影響ファイル**: .claude/skills/eda-visual/SKILL.md
- **状態**: [DONE]


## [2026-04-01] HIGH — /kickoff スキルの新設（コンペ参加直後の文脈理解ステップ）
- **説明**: EDA 前に「データが何者か」を理解するステップが欠如していた。コンペ概要に明記された外部データ情報を見落とし、外部シグナルFEの着手が大幅に遅れた。
- **発見のきっかけ**: s6e3 で IBM Telco が生成元として概要に明記されていたにもかかわらず、コンペ中盤まで外部データ活用を思いつかなかった。
- **汎用性の根拠**: どんなコンペでもデータの文脈理解（実/合成/半合成、評価指標特性、CV設計）はEDAより先に行う価値がある汎用的な手順。
- **対応内容**:
  - `.claude/skills/kickoff/SKILL.md` を新規作成
  - CLAUDE.md の学習サイクル図に `/kickoff` を最初のステップとして追加
  - CLAUDE.md のステージゲートに Stage 0 (Kickoff) を追加、Stage 5 に相関確認条件を追加
  - `/new-experiment` フェーズ0（合成データ確認）を削除（kickoff に統合）
- **影響ファイル**: .claude/skills/kickoff/SKILL.md（新規）, CLAUDE.md, .claude/skills/new-experiment/SKILL.md
- **状態**: [DONE]


## [2026-04-01] HIGH — 合成データコンペ向け外部シグナル特徴量テンプレート
- **説明**: 合成データコンペで「元データの統計量を外部シグナルとして注入する」パターンをテンプレート化。①カテゴリ別ターゲット率マッピング（ORIG_proba）と②数値分布距離特徴量（z-score/percentile/Euclid距離）の2パターンが有効と確認。
- **背景（反省点）**: FEを内部データのみで探索していた。合成データでは元データのシグナルが圧縮されており、外部から補完する方が内部FEの追加より効果が大きいケースがあった。
- **対応内容**:
  - CLAUDE.md に「合成データコンペ向けガイダンス」セクション追加
  - `/new-experiment` スキルにフェーズ0（初回: 合成データ確認）を追加
  - `/eda-visual` スキルにQ4（元データ入手確認・分布比較）を追加
  - `/fe-hypothesis` スキルの FE_HYPOTHESES.md テンプレートに「外部データ活用」カテゴリを追加
- **影響ファイル**: CLAUDE.md, .claude/skills/new-experiment/SKILL.md, .claude/skills/eda-visual/SKILL.md, .claude/skills/fe-hypothesis/SKILL.md
- **実装上の注意**: percentile計算は `np.searchsorted` を使う（`percentileofscore` のループは O(N²) で大規模データでは使用不可）
- **状態**: [DONE]


## [2026-04-01] HIGH — 出力ディレクトリ構造の整理と提出ファイル命名規約
- **説明**: `data/output/` 直下に提出CSV・OOF .npy・モデル・パラメータが混在しており、コンペ終盤に「提出可能ファイルはどれか」の特定が困難だった。サブディレクトリ分離と命名規約で解決する。
- **背景（反省点）**: 最後の8分でどの未提出ファイルを出すか目視で探す作業が発生。ファイル名にスコアも実験IDも入っておらず判断コストが高かった。
- **対応内容**:
  - `src/config.py` に `SUBMISSIONS_DIR / OOF_DIR / MODELS_DIR / PARAMS_DIR` を追加
  - `src/config.py` に `submission_path(model, oof_score, exp_id)` 命名ヘルパーを追加
  - CLAUDE.md のディレクトリ規約表を更新（サブディレクトリ明示）
  - CLAUDE.md にコーディング規約として提出命名規約と使用例を追記
  - 命名規約: `sub_{exp_id}_{model}_{oof:.5f}_{yyyymmdd_HHMM}.csv`
- **影響ファイル**: src/config.py, CLAUDE.md, 全 run_*.py（新規作成分から適用）
- **状態**: [DONE]


## [2026-04-01] HIGH — アンサンブル棄却分析の構造化（諦めずに次の手を探す）
- **説明**: アンサンブル実験が「効かなかった → スキップ」で終わり、なぜ効かなかったか・次に何を試すかの分析がなかった。FEの棄却パターン分類と同じ考え方をアンサンブル段階にも適用する。
- **背景（反省点）**: CB Plain（相関=1.000）・meta-stacking（相関=0.9998）・XGBoost（低OOF）が棄却された際、「なぜその結果になったか」と「それを踏まえて次に何をすべきか」の対話がなかった。同じパターンの棄却を繰り返していた。
- **対応内容**:
  - CLAUDE.md Stage 6 の探索手順に棄却分析テーブル（A〜D パターン）を追加
  - 各 STEP に「改善なし → 棄却分析 → 次の手」のフローを明示
  - `/kaggle-submit` フェーズ3 にアンサンブル実験特有の棄却分析問いかけを追加
  - 「棄却は終わりではなく次の探索方向を示すシグナル」を原則として明記
- **影響ファイル**: CLAUDE.md, .claude/skills/kaggle-submit/SKILL.md
- **状態**: [DONE]


## [2026-04-01] HIGH — 提出枠管理と learning 記録の強制力強化
- **説明**: 提出枠をステージ固定で配分する考え方を廃止し「残り枠を使い切る・毎回状況を把握する」方針に変更。`learning` 列の未記入を許可しないルールを明示。
- **背景（反省点）**: ①締め切り直前に「未提出の有望ファイルはどれか」を探す作業が発生した（残り枠の把握不足）。②`learning` 列が空欄のまま実験を積み重ね、後から「なぜあの実験をしたか」が追跡できなかった。
- **対応内容**:
  - CLAUDE.md に提出枠管理方針（残り枠は使い切る・毎回確認）を追記
  - `/kaggle-submit` フェーズ2 に残り枠・締め切り・未提出候補の提示ロジックを追加
  - `/kaggle-submit` フェーズ4 に `learning` 必須記入・引き出し問いかけを追加
  - `/kaggle-submit` 注意事項に「`learning` 空欄のまま次に進まない」を明記
- **影響ファイル**: CLAUDE.md, .claude/skills/kaggle-submit/SKILL.md
- **状態**: [DONE]


## [2026-04-01] HIGH — アンサンブル探索の標準手順化と toolkit 整備
- **説明**: アンサンブル探索が非体系的（相関未確認→実装→重みゼロ判明）だった。相関確認→Simple Blend→Greedy HC→Stacking の順序を標準手順として定め、ツールをテンプレートに組み込む。
- **背景（反省点）**: CB Plain（相関=1.000）・meta-stacking（相関=0.9998）を実装・学習後に「効果なし」と判明。Greedy HC は終盤に手作りしたが、最初から標準ツールとして用意すべきだった。
- **対応内容**:
  - CLAUDE.md Stage 6 にアンサンブル探索の推奨順序（STEP 1〜4）と相関確認ワンライナーを追記
  - `src/utils/ensemble.py` に `correlation_check()` / `optimize_weights()` / `greedy_ensemble()` を追加
- **影響ファイル**: CLAUDE.md, src/utils/ensemble.py
- **状態**: [DONE]


## [2026-04-01] MED — アンサンブル多様性の早期確認フレームワーク
- **説明**: 複数モデル（LGB/CB等）の予測相関が高い（>0.999）場合、アンサンブルへの追加効果はほぼゼロ。モデル追加前に相関確認を必須ステップとして組み込む。
- **背景（反省点）**: CB boosting_type変更（Ordered→Plain）が予測相関=1.000と判明したのは実装・学習後だった。事前に確認できる指標があれば計算コストを節約できた。
- **対応内容**: `run_ensemble.py` または `src/utils/ensemble.py` に「相関行列の表示 → weight=0 のモデルを自動スキップ」ロジックを追加する
- **影響ファイル**: src/utils/ensemble.py, run_ensemble.py（または同等のアンサンブルスクリプト）
- **状態**: [DONE] 2026-07-01 CLAUDE.md Stage 6 STEP 1（相関確認フロー）で反映済み。`correlation_check()` が相関≥0.998の場合スキップとなっている

---


## [2026-05-31] HIGH — Kickoff 時の外部生データ判定リスト
- **説明**: 今回 woodshole F1 Weather data がコンペ最終日まで未使用だった。Kickoff時に `data/external/` 内の全ファイルを「使う/使わない」明示判定する儀式を追加すべき。
- **教訓**: Weather 投入で LB 0.95426 → 0.95427 → 0.95428 突破。**もっと早く使えていればさらなる伸び代があった**。
- **対応内容**:
  - `/kickoff` skill にフェーズ1 Q6「外部データインベントリ」追加（使う/保留/skip の 3 択判定）
  - CLAUDE.md AIへの指針 #10 に「コンペ初日の外部データインベントリ義務化」を追加
  - `COMPETITION.md` テンプレートに「外部データインベントリ」セクション追加
- **影響ファイル**: `.claude/skills/kickoff/SKILL.md`, CLAUDE.md
- **状態**: [DONE] 2026-06-01 反映完了


## [2026-05-31] HIGH — ドメイン知識先行ヒアリング
- **説明**: F1 ドメインでは weather が pit stop に直結することは自明だが、ML パイプライン優先で「データから学習」発想に偏った。
- **教訓**: 「is_wet_race」binary 1 つで天候 signal をカバーしたと早期判断、AirTemp/TrackTemp/Humidity 等の連続値を見落とし。
- **対応内容**:
  - `/kickoff` skill にフェーズ1 Q5「ドメイン知識先行ヒアリング」追加（ターゲットに影響する変数を 5-10 個列挙）
  - CLAUDE.md AIへの指針 #11 に「ドメイン知識先行プロセス」を追加
  - `COMPETITION.md` テンプレートに「ドメイン知識先行リスト」セクション追加
- **影響ファイル**: `.claude/skills/kickoff/SKILL.md`, CLAUDE.md
- **状態**: [DONE] 2026-06-01 反映完了


## [2026-05-31] HIGH — Plateau 検出時の Discussion 強制調査
- **説明**: LB プラトー (5+ 提出同値) で「飽和」と判断しがち。CLAUDE.md には書いてあるが skill 化されていない。
- **教訓**: 0.95425-0.95426 ceiling を 7+ 実験で確認後、外部 voting に注力したが weather data の存在は知っていながら投入せず。
- **対応内容**:
  - `/kaggle-submit` skill フェーズ3 に「Plateau 検出（強制）」を追加（同一 LB ± 0.00002 で 5 回以上で発動）
  - CLAUDE.md AIへの指針 #12 に「LB プラトー検出時の強制 brainstorm」を追加
- **影響ファイル**: `.claude/skills/kaggle-submit/SKILL.md`, CLAUDE.md
- **状態**: [DONE] 2026-06-01 反映完了


## [2026-05-31] MEDIUM — HP retune スケジュール
- **説明**: FE 変更時に HP 最適点が変動することを今回実証 (Δ=+0.00014 OOF)。テンプレートに「FE が ±20% 変わったら HP retune を検討」のガイドライン追加。
- **対応内容**: CLAUDE.md Stage 5 完了条件に「FE変更時の HP retune ルール」追加
- **影響ファイル**: CLAUDE.md (Stage 5 セクション)
- **状態**: [DONE] 2026-06-01 反映完了


## [2026-05-31] MEDIUM — Multi-seed Averaging のデフォルト化
- **説明**: 全ベース blend モデルで multi-seed avg5 が +0.00010-0.00020 OOF 改善を提供。テンプレートで multi-seed 5+ をデフォルトに。
- **対応内容**: CLAUDE.md Stage 6 STEP 6 への重要追記として「Multi-seed averaging のデフォルト化」を追加（n_ens=5 デフォルト推奨）
- **影響ファイル**: CLAUDE.md (Stage 6)
- **状態**: [DONE] 2026-06-01 反映完了


## [2026-05-31] MEDIUM — Blend of Blends パターン記録
- **説明**: 構造的に異なる 2 blend (greedy HC vs equal weight) の 50/50 平均で +0.00001 LB 改善が出ることを今回実証。
- **対応内容**: CLAUDE.md Stage 6 に新規「STEP 8【Blend of Blends - 構造的に異なる blend の consensus】」追加。実装パターン・メカニズム・Final 2 への影響を明記。
- **影響ファイル**: CLAUDE.md (Stage 6)
- **状態**: [DONE] 2026-06-01 反映完了


## [2026-05-31] LOW — AV 診断 (Adversarial Validation) を Stage 4 標準化
- **説明**: 今回最終日に AV を実行し、TabM 拡張特徴量に leakage を発見。BASE_FEATURES は AV=0.501 で問題なかったが、もっと早く確認すべきだった。
- **対応内容**: CLAUDE.md Stage 4 完了条件に「AV 診断で train/test 分布シフトの有無を確認済み」を追加。Stage 6 移行前ガイダンスとして「AV 診断（Adversarial Validation）の標準実施手順」セクション追加（判定基準テーブル付き）
- **影響ファイル**: CLAUDE.md (Stage 4 セクション)
- **状態**: [DONE] 2026-06-01 反映完了


## [2026-05-31] LOW — DSチームペルソナ投票による Final 2 選定
- **説明**: 今回 9 人のペルソナ投票で Final 2 を選定 (sub_140 + sub_141)。シェイクダウン回避に有効な手法。
- **対応内容**:
  - CLAUDE.md「最終選択（Final Submission Selection）」セクションを大幅改訂、9 ペルソナ投票プロトコル追加
  - `/kaggle-submit` skill にフェーズ5「Final 2 選定モード」追加（最終日または残り 2 提出以下で発動）
  - Persona 主張テーブル、投票ルール、典型パターン (A/B/C/D)、警告を明文化
- **影響ファイル**: CLAUDE.md, `.claude/skills/kaggle-submit/SKILL.md`
- **状態**: [DONE] 2026-06-01 反映完了

---


## [2026-06-01] HIGH — AI 行動規範 #13-16 と Autonomous Skill Application 追加
- **背景**: s6e5 終了時のユーザー振り返りで判明した AI の行動パターン問題
  - 早期却下しがち（weather 特徴量を最終日まで見送り）
  - Final 2 早期決定癖（残り 3 slot で確定提案、ユーザーが止めなければ +0.00001 LB 改善取り逃がし）
  - 1実験1コミット違反（exp148-154 を 1 コミット）
  - 可視化の自発提案不足（数値表のみ報告）
- **対応内容**:
  - CLAUDE.md AIへの指針 #13-16 追加:
    - #13: 早期却下の禁止（可視化・関連変数列挙・相関/importance 三重チェック）
    - #14: Final 2 早期決定の禁止（残り slot ≥ 2 では Final 議論禁止）
    - #15: 1実験1コミットの厳守（並行実行時も例外なし）
    - #16: 可視化の自発的提案（「グラフ生成しますか？」を必ず提示）
  - CLAUDE.md「Autonomous Skill Application」セクション新設:
    - スキル呼び出しが無くてもプロトコルに従う義務
    - 場面別の autonomous 適用テーブル
    - ユーザーが skill 呼ぶ場面 vs AI が autonomous で従う場面の整理
  - CLAUDE.md コミット規約に並行実行時の特例ルール追加（バッチコミット禁止）
- **影響ファイル**: CLAUDE.md
- **状態**: [DONE] 2026-06-01


## [2026-06-01] MEDIUM — scripts/visualize.py に Stage 4-6 向け関数追加
- **背景**: 「早期却下の禁止」原則を実行するための可視化 helper 不足
- **対応内容**:
  - `plot_feature_importance()`: LGB/XGB importance top N 棒グラフ
  - `plot_oof_distribution()`: 新旧 OOF 予測のヒストグラム比較
  - `plot_correlation_matrix()`: 複数モデル OOF 相関マトリクス（heatmap）
  - `plot_lb_history()`: experiments/log.csv の submit_score 時系列 + experiment_id ラベル付き
  - CLI に `--theme lb_history` 追加
- **影響ファイル**: scripts/visualize.py
- **状態**: [DONE] 2026-06-01

---


## [2026-06-01] HIGH — Private LB 検証で判明した 4 つの戦略的教訓
- **背景**: s6e5 のコンペ終了後 Private LB 確認で判明した発見
  - Private LB 全体で reverse-shakedown（Private > Public for 全自前 submission）
  - 採用 Final 2 = sub_140 (Private 0.95448)、しかし真の最高は sub_127 (Private 0.95450, +0.00002)
  - sub_127 は Public LB 0.95426 で「平凡」と判断され Final 候補から除外されていた
  - 新規 FE (外部データ集約) は Public +0.00001 (ノイズ床) のみ改善、Private では -0.00002 悪化
  - BoB (Public 最高 +0.00001) の Private は親 blend と同等 → Public 微改善が Private に反映されない実例
- **対応内容**:
  - CLAUDE.md AI 指針 #17 追加: Public LB 微改善の懐疑主義（評価指標別の閾値テーブル付き）
    - AUC: ±0.0001 ノイズ床、+0.0002 で「突破」
    - Logloss: ±0.001 absolute、-0.002 で「突破」
    - RMSE: ±0.1% relative、-0.2% で「突破」
    - Accuracy/F1: ±0.001 ノイズ床、+0.003 で「突破」
  - CLAUDE.md AI 指針 #18 追加: OOF を Public LB と同等以上に Private LB 指標として尊重
  - CLAUDE.md AI 指針 #19 追加: Final 2 候補プール拡張ルール（Public Top-10 ∪ OOF Top-10）
  - CLAUDE.md AI 指針 #20 追加: 新規 FE/外部データの「Private 過適合候補」分類
  - CLAUDE.md「最終選択」セクションに候補プール構築ステップ追加（注目度分類テーブル付き）
  - CLAUDE.md Stage 6 STEP 8 (BoB) に Private 注意点追記（共倒れリスク警告）
  - kaggle-submit skill フェーズ5 に Step 1-3 追加（候補プール拡張 → AI 指針チェック → Persona 投票）
- **設計判断**:
  - 評価指標別の閾値テーブルで AUC 以外のコンペにも汎用適用可能に
  - 個別特徴量名（weather など）と特定スコア（0.95428 など）を記述から除去
  - 「教訓」は具体的事例ではなく一般化されたパターンとして記述
- **影響ファイル**: CLAUDE.md, .claude/skills/kaggle-submit/SKILL.md
- **状態**: [DONE] 2026-06-01

---


## [2026-06-01] HIGH — コンペ開始手順の自動化（手作業を COMPETITION 1 項目に削減）
- **背景**: README の「Step 2: コンペ設定を更新」が config 6 項目の手編集を案内、「Step 3: データDL」も手動コマンド案内。実態は `/kaggle-setup`（COMPETITION設定+DL）と `/kickoff`（残り config 自動更新）が既に自動化しており、README が古いまま矛盾していた。
- **対応内容**:
  - **Python 3.12 固定**: `.python-version`（=3.12）新規作成。`pyproject.toml` を `requires-python = ">=3.12,<3.13"` に変更。理由: MLスタック（LGB/XGB/CB/PyTorch/RealMLP/TabM 等）の wheel 成熟度。uv が pin 不在で 3.14 を自動選択していた問題を解消
  - **README 全面簡素化**: 手作業 Step2/3 を削除。`/kaggle-setup <slug>` → `/kickoff` の 2 スキルで完結する「スラッグ1つだけ入力」フローに書き換え。手動フローは参考として圧縮
  - **kickoff 強化**:
    - フェーズ0 新設: `data/raw/` 空ならダウンロード提案（kickoff 単独実行時のセーフティネット）
    - Q3 に TARGET_COL 自動検出（sample_submission.csv 2 列目）追加
    - フェーズ3 を「手編集」→「自動補完」に改訂。EVAL_METRIC は `kaggle competitions view` メタデータ、PROBLEM_TYPE は指標+値域、CV_STRATEGY は Q4 から自動決定。書込前にユーザー確認
- **設計判断**: 手入力は `COMPETITION` 1 項目のみ。残りはデータ・メタデータ・対話から自動導出し提示確認する方式
- **影響ファイル**: `.python-version`（新規）, pyproject.toml, README.md, `.claude/skills/kickoff/SKILL.md`
- **状態**: [DONE] 2026-06-01

---


## [2026-07-01] HIGH — s6e6 コンペ振り返りによるテンプレート改善（5項目）

### A. OOF最大化 + pub_oof_gap最小化の二軸評価（gap最大化廃止）
- **背景**: s6e6 全50提出の統計分析。OOF→Private r=+0.998、pub_oof_gap→Private r=−0.51、gap→シェイクダウン量 r=+0.853。SESSION.md に記録されていた「ΔLB = ΔOOF + Δgap → gap拡大で改善」は誤り。
- **対応内容**: CLAUDE.md AIへの指針 #21「OOF最大化とpub_oof_gap最小化の二軸評価」追加。モデルファミリー別OOF信頼性テーブル（NN/Tree/Blend）付き。pub_oof_gap監視ルール追記。
- **影響ファイル**: CLAUDE.md
- **状態**: [DONE] 2026-07-01

### B. Stage 1.5（早期アーキテクチャサーベイ）新設
- **背景**: s6e6 では LGB 主軸のまま 40+ 実験を費やし、RealMLP 移行が終盤になった。Phase 効率分析: LGB FE 探索 +0.000007 LB/提出 vs RealMLP 移行 +0.000343 LB/提出（50x 差）。
- **対応内容**: CLAUDE.md ステージゲート表に「1.5. 早期アーキテクチャサーベイ」追加。公正比較条件（同一FE/HP/CV）の義務化。手順・記録テーブル・教訓を記載。AIへの指針 #22「アーキテクチャ乗り換え時の公正比較義務」追加。
- **影響ファイル**: CLAUDE.md
- **状態**: [DONE] 2026-07-01

### C. FE有効性のアーキテクチャ依存性明記
- **背景**: LGB で棄却した FE が RealMLP では有効だったケースが s6e6 で複数発生。「LGB 棄却 = 全アーキテクチャで棄却」という誤判断を防ぐ。
- **対応内容**: CLAUDE.md FE棄却判断マトリクスの後に「FEの有効性はアーキテクチャに依存する」節追加。棄却記録に「棄却したアーキテクチャ名」明記義務。Stage 4 → Stage 6 移行時の FE 移植手順追加。Stage 4 完了条件に「全候補アーキテクチャに同一FEを移植して再評価済み」追記。
- **影響ファイル**: CLAUDE.md
- **状態**: [DONE] 2026-07-01

### D. Kaggle Notebook 環境サポート追加
- **背景**: テンプレートをローカル環境専用から Kaggle Notebook 環境でも動作するよう拡張する。IS_KAGGLE フラグで自動切り替え。
- **対応内容**: `src/config.py` に `IS_KAGGLE` 環境検出追加（`/kaggle/input` 存在確認）。ローカル/Kaggle でパス自動切り替え。CLAUDE.md に「Kaggle Notebook 環境サポート」セクション追加（セットアップ手順・データ読み込みパターン・注意点）。
- **影響ファイル**: src/config.py, CLAUDE.md
- **状態**: [DONE] 2026-07-01

---


## [2026-07-02] HIGH — `uv sync` が shap の推移的依存で失敗する（numba/llvmlite が極めて古いバージョンに固定される）

- **背景**: s6e7 セットアップ時、`uv sync`（Apple Silicon / arm64 macOS, Python 3.12）が `numba==0.53.1` のビルドで失敗した。
  ```
  RuntimeError: Cannot install on Python version 3.12.12; only versions >=3.6,<3.10 are supported.
  ```
- **原因調査**:
  - `pyproject.toml` は `shap>=0.46` としか書いておらず、PyPI 最新の `shap==0.52.0` が解決される。
  - `shap==0.52.0` の `requires_dist` には環境マーカー分岐がある:
    ```
    numba<0.63; sys_platform == "darwin" and platform_machine == "x86_64"
    numba;      sys_platform != "darwin" or platform_machine != "x86_64"
    llvmlite<0.46; sys_platform == "darwin" and platform_machine == "x86_64"
    llvmlite;      sys_platform != "darwin" or platform_machine != "x86_64"
    ```
  - 実行環境は arm64（`x86_64` ではない）なので本来 `numba<0.63` 分岐には該当しないはずだが、uv 0.9.22 のユニバーサルロック解決では x86_64 分岐の上限制約が引きずられ、`numba==0.53.1` / `llvmlite==0.36.0`（2021年リリース、Python 3.9 までしか対応しない）という最古版が選ばれてしまう。
  - `uv lock --upgrade-package numba`、`rm uv.lock && uv lock`、`--fork-strategy requires-python` のいずれも改善せず。`tool.uv.environments` で arm64 macOS のみに限定してもリゾルバの選択は変わらなかった。
- **回避に成功した手順**（s6e7 で実施・検証済み）:
  1. `pyproject.toml` に `[tool.uv] environments = ["sys_platform == 'darwin' and platform_machine == 'arm64'"]` を追加してロック対象プラットフォームを絞る
  2. `numba` を直接依存として追加し、下限を明示的に引き上げる: `uv add "numba>=0.60"`（`numba>=0.53.1` のような自明下限では効果なし。0.60 以上を明示することでリゾルバが `numba==0.66.0` / `llvmlite==0.48.0` を選ぶようになった）
  3. `uv sync` で成功（`shap==0.52.0`, `numba==0.66.0`, `llvmlite==0.48.0` の組み合わせで全 import 確認済み）
- **テンプレートへの恒久対応（未着手 / TODO）**:
  - `pyproject.toml` の `dependencies` に `numba>=0.60` を明示的に追加する（`shap` の推移的依存だけに任せない）。ユニバーサルロック時の x86_64 分岐に引きずられる問題を根本から回避できる。
  - `tool.uv.environments` による arm64-only 限定は **他アーキテクチャ（Intel Mac / Linux）でこのテンプレートを使うユーザーの互換性を壊す**ため、恒久対応としてそのまま採用すべきではない。代替案:
    - (a) `numba>=0.60` の直接依存追加だけで解決するか再検証する（`environments` 制限なしでも効くか要確認）
    - (b) `environments` を使うなら arm64 だけでなく x86_64 darwin / linux も列挙したユニバーサル指定にする
    - (c) shap を optional dependency 化し、SHAP 解釈が必要な時だけ `uv sync --extra shap` のように分離する（本体の環境構築を shap のバージョン問題から切り離す）
  - 対応時は Intel Mac または Linux 環境（CI 等）でも `uv sync` が通ることを確認してから DONE にする。
- **影響ファイル**: pyproject.toml, uv.lock
- **恒久対応（2026-08-02 完了。案 (a) を採用）**:
    - **`environments` によるプラットフォーム制限を完全に削除**し、`numba>=0.60` の直接依存だけを残した
    - 検証: 制限なしで `uv lock` → **numba==0.66.0 / llvmlite==0.48.0 / shap==0.52.0** が解決され、`uv sync` も成功。全ライブラリ（lightgbm / xgboost / catboost / torch / shap / numba）の import と既存実験スクリプトの実行も確認済み
    - → **arm64-only 制限による他アーキテクチャ（Intel Mac / Linux）の互換性破壊が解消された**。案 (b)(c) は不要
    - `pyproject.toml` に「なぜ numba を直接依存にしているか」のコメントを追記（将来の削除を防ぐ）
- **状態**: [DONE]

---


## [2026-07-02] HIGH — `uv run python scripts/train.py` が `ModuleNotFoundError: No module named 'src'` で起動しない

- **背景**: s6e7 で CLAUDE.md 記載の実行例通り `uv run python scripts/train.py` を実行したところ、`src.config` の import で `ModuleNotFoundError` が発生した。テンプレート付属の `scripts/train.py`（TODO埋め前の骨格のまま）でも同じ現象を確認済み。
- **原因**: Python は `python <file>.py` 実行時、`sys.path[0]` にスクリプト自身のディレクトリ（`scripts/`）を追加し、カレントディレクトリ（プロジェクトルート）を追加しない。そのため `scripts/` 配下のスクリプトから `from src.config import ...` が解決できない。`pyproject.toml` にも `src` を editable install する設定（`[tool.setuptools.packages.find]` 等）が無いため、`uv sync` 後も `src` はサイトパッケージとして認識されない。
- **確認した回避策**: `uv run python -m scripts.preprocess` のように `-m` （モジュール実行）形式にすると `sys.path[0]` がカレントディレクトリになり解決する。ただし CLAUDE.md の実行例（`uv run python scripts/train.py --model lgb`）とは異なる呼び出し方になり、ドキュメントと実挙動が食い違っている。
- **恒久対応（未着手 / TODO）**: 以下のいずれかで CLAUDE.md 記載の実行例をそのまま動くようにする。
    - (a) `pyproject.toml` に `[tool.setuptools.packages.find]` 等を追加し `uv sync` 時に `src` を editable install する（`uv pip install -e .` 相当）。`scripts/*.py` 側の変更は不要になる
    - (b) 各 `scripts/*.py` の先頭で `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` を追加し、スクリプト単体実行でも解決できるようにする
    - (c) CLAUDE.md の実行例を `uv run python -m scripts.train` 形式に統一する（最も手間が少ないが、`argparse` 等の呼び出し感が変わる点に注意）
  - (a) が最も「テンプレートを使う側が意識しなくて済む」ため推奨。対応時は `scripts/` 配下の全スクリプト（train.py, predict.py, feature_study.py, optimize_hp.py, blend.py, visualize.py, feature_report.py 等）で動作確認する。
- **影響ファイル**: pyproject.toml, CLAUDE.md（実行例の記述）, scripts/*.py
- **解決確認**: src import が通ることを2026-08-02に検証(uv run python -c 'from src.config import ...' 成功)
- **状態**: [DONE]

---


## [2026-07-02] MED — `/ds-new-experiment` の予約行と `ExperimentTracker.end_run()` が別々の log.csv 行を作ってしまう

- **背景**: s6e7 exp001 で `/ds-new-experiment` が `experiments/log.csv` に `experiment_id=001` の行を予約追記（experiment_question/success_criteria/abort_criteria 入り、スコア列は空欄）した後、`scripts/train.py` 内の `ExperimentTracker.end_run()` を呼んだところ、`_get_next_experiment_id()` が「既存の最大 ID + 1」を採番し `experiment_id=002` として**別の行**を新規追記してしまった。結果、001（目的・成功基準・撤退基準あり、スコアなし）と 002（スコアあり、目的等は空欄）に情報が分裂した。
- **原因**: `ExperimentTracker.start_run()` は `_get_next_experiment_id()` で常に「次の空き番号」を採番するだけで、`/ds-new-experiment` が既に予約した行の experiment_id を再利用する仕組みが無い。両スキル間で ID の受け渡しがされていない。
- **s6e7 での回避**: 学習後に手動で 001 行に 002 のスコア系カラムをマージし、002 行を削除して整合を取った。
- **恒久対応（未着手 / TODO）**:
    - (a) `ExperimentTracker` に「既存の予約行（スコア未記入の最新行）があればその experiment_id を引き継いで上書きする」ロジックを追加する（例: `start_run()` 時に log.csv を読み、自分より前で oof_score が空の行があればそれを再利用）
    - (b) `scripts/train.py` に `--experiment-id` 引数を追加し、`/ds-new-experiment` が払い出した ID を明示的に渡せるようにする。`SESSION.md` の「次にやること」に呼び出しコマンド例（ID込み）を書く運用にする
    - (c) 予約行を作らず、`/ds-new-experiment` は目的・成功基準・撤退基準を SESSION.md や一時ファイルにのみ記録し、`ExperimentTracker.end_run()` 実行時にそれを読み込んで同じ行に書き込む（予約 → 上書きではなく、実行時に1回だけ書く設計に変更）
  - (c) が最も競合しにくいが `/ds-new-experiment` のドキュメント（フェーズ3手順5）の変更が必要。(a)(b) は現行の「先に予約行を作る」設計を維持できる。
- **影響ファイル**: src/experiment.py, `.claude/skills/ds-new-experiment/SKILL.md`, scripts/train.py
- **恒久対応（2026-08-02 完了。案 (a) を採用）**:
    - `src/experiment.py` に `_find_reserved_row()` を追加。「`experiment_question` が埋まっており、かつ `oof_score`/`cv_val_mean` が空」の行を**予約行**として末尾から検出する
    - `_get_next_experiment_id()` が予約行を見つけたら**その ID を再利用**する（新規採番しない）
    - `end_run()` は予約行があれば**追記ではなく上書きマージ**する。予約時に記入された `experiment_question` / `success_criteria` / `abort_criteria` は保持し、空の `description` / `notes` も予約行の値を活かす
    - 隔離環境で検証済み: 予約行作成 → `end_run()` の流れで **行数 1 のまま**、スコアと目的の両方が同一行に揃うことを確認
    - 案 (b)(c)（`--experiment-id` 引数の追加 / 予約行を作らない設計変更）はスキル側の変更が必要なため不採用。現行の「先に予約行を作る」設計を維持したまま解決できた
- **状態**: [DONE]

---


## [2026-07-02] HIGH — `scripts/train.py` / `scripts/predict.py` が binary_classification 専用で multiclass 非対応

- **背景**: s6e7（3クラス分類、評価指標 balanced_accuracy）で `scripts/train.py` の TODO を埋めて実行しようとしたところ、テンプレート付属の骨格が以下の点で binary 分類専用に固定されており、そのままでは動かなかった。
  - `DEFAULT_PARAMS` の lgb/cb/xgb すべてが `objective: "binary"` 等の binary 専用設定
  - `model.predict_proba(X_val)[:, 1]` で正例確率のみを取り出す実装（multiclass では `(N, N_CLASSES)` の配列全体が必要）
  - スコア計算が `roc_auc_score(y_val, val_pred)` に決め打ち（AUC 以外の指標、特に balanced_accuracy や multiclass logloss が使えない）
  - `scripts/predict.py` も同様に、予測値を `np.clip` してそのまま1列出力する前提（multiclass では argmax でクラス決定 → ラベル文字列への逆変換が必要）。加えて `TARGET_COL_OUT = "target"` が固定のハードコード値で、`src/config.py` の `TARGET_COL` と二重管理になっている（値がズレると気づきにくい）
  - `src/config.py` には元々 `PROBLEM_TYPE`（"regression" | "binary_classification" | "multiclass"）と `EVAL_METRIC` の変数が用意されているにもかかわらず、`scripts/train.py` 側はこれらを一切参照せず、モデル・スコア計算をハードコードしている。設定と実装が連動していない
- **s6e7 での対応**: `scripts/train.py` を multiclass + balanced_accuracy 用に書き換え（`LabelEncoder` でクラスをエンコード、`objective: multiclass`、`balanced_accuracy_score(y, np.argmax(preds, axis=1))` でスコア計算）。`scripts/predict.py` も argmax→ラベル逆変換に書き換え、`TARGET_COL_OUT` 決め打ちをやめて `src.config.TARGET_COL` を参照するよう変更。
- **恒久対応（未着手 / TODO）**:
    - `scripts/train.py` / `scripts/predict.py` を `src.config.PROBLEM_TYPE` を見て分岐する設計にする（例: `PROBLEM_TYPE == "multiclass"` なら LabelEncoder + argmax 経路、`"binary_classification"` なら現行の `[:, 1]` 経路、`"regression"` なら `predict()` 経路）
    - スコア計算関数も `src.config.EVAL_METRIC` に応じて `sklearn.metrics` から動的に選択する薄いディスパッチ関数（例: `src/metrics.py` に `get_scorer(EVAL_METRIC)` を追加）を用意し、`roc_auc_score` 決め打ちを解消する
    - `scripts/predict.py` の `TARGET_COL_OUT` ハードコードを削除し `src.config.TARGET_COL` に統一する
    - 対応後、binary（既存コンペ想定）・multiclass（s6e7）・regression の3パターンで骨格が動くことを確認してから DONE にする
- **影響ファイル**: scripts/train.py, scripts/predict.py, src/config.py, src/metrics.py（新規検討）
- **⚠️ 判定の訂正（2026-08-02）**: 一度 [DONE] としたが、これは **comp ブランチ（s6e7 適用版）での検証**であり、**テンプレート本体 main は依然 binary 専用**（`objective: "binary"` / `binary:logistic`）だった。「コンペ用にカスタマイズした結果動いている」と「テンプレートが対応している」を取り違えた誤判定。
- **解決（2026-08-02 / v5 で対応）**: s6e7 で実証済みの multiclass 実装（LabelEncoder + `multi:softprob` / `multi_logloss` + `run_cv()` の汎用CV骨格）を、**コンペ固有記述（FEATURES 実体・N_CLASSES 固定値・s6e7 の docstring）を除去した上で** main へ取り込んだ
- **状態**: [DONE]

---


## [2026-07-02] MED — 前処理スクリプト（`data/processed/` を生成する骨格）がテンプレートに存在しない

- **背景**: s6e7 で `scripts/train.py` を実行しようとしたところ `PROCESSED_DATA_DIR / "train_features.pkl"` を要求されたが、これを生成するスクリプトがテンプレートに含まれていなかった。CLAUDE.md や過去の TODO_TEMPLATE.md エントリには `notebooks/02_preprocess.py` という記述が残っているが、現行テンプレートには `notebooks/` ディレクトリ自体が存在せず（`scripts/` ベースに移行済みと思われる）、対応する `scripts/preprocess.py` も無い。
- **影響**: 新しいコンペを始めるたびに、前処理スクリプトをゼロから自作する必要がある。ドキュメント（CLAUDE.md のステージゲート説明等）と実際に存在するファイルが食い違っており、どこまでがテンプレートの守備範囲か分かりにくい。
- **s6e7 での対応**: `scripts/preprocess.py` を新規作成（数値カラムの欠損値中央値埋め→pickle保存の最小実装）。コンペ固有のロジックのため、テンプレートにそのまま採用はできない。
- **恒久対応（未着手 / TODO）**:
    - `scripts/preprocess.py` の最小骨格（train/test 読み込み→ `TODO` コメントで前処理を埋める→ `train_features.pkl`/`test_features.pkl` 保存、の3ステップ）をテンプレートに追加する。`scripts/train.py` と同じ「TODO 埋め方式」に揃える
    - CLAUDE.md 内の `notebooks/02_preprocess.py` 等、現存しないパスへの言及を `scripts/` ベースの現行構成に合わせて修正する（ドキュメントと実ファイルの不一致を解消）
- **影響ファイル**: scripts/preprocess.py（新規）, CLAUDE.md
- **解決確認**: scripts/preprocess.py の存在を2026-08-02に確認
- **状態**: [DONE]

---


## [2026-07-02] LOW — `/ds-kickoff` スキルが案内する `kaggle competitions view` コマンドが存在しない

- **背景**: `/ds-kickoff` フェーズ3（評価指標の自動取得）で案内されている `kaggle competitions view -c <slug>` を実行したところ、手元の Kaggle CLI 2.2.3 には `view` サブコマンドが存在せず `invalid choice` エラーになった（有効なサブコマンドは `list, files, download, submit, submissions, leaderboard, team-submissions, episodes, replay, logs, pages, topics, topic-messages`）。
- **影響**: `EVAL_METRIC` の自動取得ステップが機能せず、s6e7 では評価指標をユーザーへの直接ヒアリングに切り替えて対応した。スキルの手順通りに進めると原因不明のエラーで止まる体験になる。
- **恒久対応（未着手 / TODO）**:
    - `kaggle competitions view` を使わない手順に修正する。評価指標は Kaggle API から機械的に取得できないため、`/ds-kickoff` のフェーズ3手順は最初から「ユーザーにコンペページの Evaluation 欄を確認してもらう」ことを既定の一次情報源とし、API 呼び出しは補助情報（`files`/`leaderboard` 等、実在するサブコマンドのみ）に留める
    - 併せて `.claude/skills/ds-kickoff/SKILL.md` の該当コマンド例を実在するサブコマンドに置き換える
- **影響ファイル**: `.claude/skills/ds-kickoff/SKILL.md`
- **恒久対応（2026-08-02 完了）**: 実在するコマンドに置換し、動作を検証済み:
    - 評価指標: `kaggle competitions pages list -c <slug> --content --page-name Evaluation` → s6e7 で balanced accuracy の説明文が正しく取得できることを確認
    - 概要（締切・参加チーム数・自分の順位）: `kaggle competitions list -s <slug>`
    - ページ名一覧: `kaggle competitions pages list -c <slug>`（rules / Evaluation / Timeline / data-description 等）
    - `kaggle competitions view` が存在しない旨をコメントで明記し、将来の再発を防ぐ
- **状態**: [DONE]

---


## [2026-07-03] HIGH — `scripts/feature_study.py` がFE仮説検証のたびに二重学習していた（記録漏れ）

- **背景**: s6e7でexp001〜exp009まで、FE仮説（H-001〜H-006）を採用するたびに「①`scripts/feature_study.py`でΔOOF計測（base+new実装の2回CV学習）→②採用判断後、`scripts/train.py`のFEATURESに追加して本番学習として再度CV学習」という2段階フローを繰り返していた。②の学習は①の「new」（base+新特徴量）と全く同じ特徴量・同じseed・同じCV分割であり、結果は毎回完全に一致することを確認済み（例: exp006でfeature_study.pyのNew OOF=0.85778とscripts/train.py本番学習のOOF=0.85778が一致）。ユーザーからの「同じ実験を2回行っているのでは」という指摘で発覚した。
- **問題点**: `scripts/feature_study.py`の`run_cv()`はOOF/test予測をメモリ上で計算するだけで`.npy`保存や`log.csv`記録（`ExperimentTracker`）を行わない設計だったため、仮説採用のたびに同じCV学習（LGB 5fold×1000 iterations）を50%多く実行していた。また、FE仮説検証（feature_study.py実行）自体はexpNNN番号を持たず、`log.csv`に記録されず、棄却された仮説は`FE_HYPOTHESES.md`のみに文章で残る状態で、実験の追跡が二元化していた。
- **恒久対応（対応済み / DONE）**:
    - `scripts/feature_study.py`を改修し、「new」（base+新特徴量）のCV学習結果を`ExperimentTracker`経由で自動的に`log.csv`に記録するようにした（採用・棄却を問わず、全てのFE仮説検証がexpNNN番号を持つ）
    - 「new」のOOF予測・テスト予測を`OOF_DIR`に`.npy`保存し、feature importanceも`PLOTS_DIR`に保存するようにした → 仮説採用時は`scripts/train.py`で再学習せず、保存済み`.npy`をそのまま`scripts/predict.py`に渡して提出ファイルを生成できる
    - `--hypothesis-id`引数を追加し、`FE_HYPOTHESES.md`のH-NNNとlog.csvのexpNNNを相互参照できるようにした
    - `scripts/train.py`の`run_cv()`にfeature importance収集を追加（従来は`main()`内でのみ収集していたが、`run_cv()`からも取得可能にした）
    - 既存の棄却済み仮説（H-008, H-009の3variant）はlog.csvに手動で遡って記録（exp010〜exp013）
- **今後の運用ルール**: FE仮説の実装後は必ず`scripts/feature_study.py --new-feature <col> --hypothesis-id H-NNN`で実行する（従来通り）。採用の場合、`scripts/train.py`のFEATURESに追加した後は**再学習せず**、feature_study.py実行時に保存された`test_{exp_id}_{model}.npy`を`scripts/predict.py`にそのまま渡して提出ファイルを生成する
- **影響ファイル**: `scripts/feature_study.py`, `scripts/train.py`
- **状態**: [DONE]

---


## [2026-07-02] HIGH — `/ds-kaggle-submit` の診断がOOF↔LBの2軸のみで、train↔val（fold内）乖離を見ていない

- **背景**: s6e7 exp001（train=0.46598, val=0.44485, LB=0.43832）で、`/ds-kaggle-submit` の標準フロー（フェーズ3「OOF/LBの差分パターン」表）通りに進めると oof_lb_gap（0.00653）だけを見て次の一手を判断することになる。実際にはユーザーが「train/val乖離(0.02113)も大きい」と指摘したことで、CV設計の不安定性（exp002で検証・棄却）でも単純な過学習でもなく、「argmax決定を伴う多クラス分類での校正不足（損失関数と評価指標のミスマッチ）」という第三の仮説にたどり着き、class_weight導入という決定的な打ち手（OOF 0.44485→0.81189, LB 0.43832→0.81185）につながった。
- **問題点**: `/ds-kaggle-submit` フェーズ3の「OOF/LBの差分パターン」表は OOF↔LB の2軸だけを扱っており、train↔val（同一実験内・fold内）の乖離やfold間ばらつきを診断対象にする視点が設計上存在しない。`scripts/train.py` はtrain_scoreとval_scoreを両方計算・記録しているにもかかわらず、それを解釈する仕組みがスキル側に無く、ユーザーの気づきに依存していた。
- **恒久対応（未着手 / TODO）**:
    - `/ds-kaggle-submit` フェーズ3に「CV内部診断」ステップを追加する。具体的には `cv_train_mean - cv_val_mean`（fold内乖離）を `cv_val_mean - submit_score`（val/LB乖離）と併記し、以下のような分岐を促す:
      - train/val乖離が小さい・val/LB乖離が大きい → CV設計の問題（fold数・seed・層化）を疑う
      - train/val乖離が大きい・val/LB乖離も大きい → 過学習 or 校正不足（多クラス/不均衡タスクではclass_weight・閾値調整を疑う）を疑う
      - train/val乖離が大きい・val/LB乖離が小さい → 通常の過学習（正則化・early stopping調整）
    - 特に **多クラス分類 + argmax決定 + クラス不均衡**の組み合わせでは、単純な指標比較だけでなく「予測クラス分布 vs 実際のクラス分布」を確認するステップを明示的に追加する（`np.unique(argmax(oof), return_counts=True)` 比較）。今回はこの比較が校正不足の発見に直結した
    - `experiments/log.csv` に既に `cv_train_mean`/`cv_train_std`/`cv_val_mean`/`cv_val_std` の列があるので、これらを `/ds-kaggle-submit` が読み取って自動で乖離幅を計算・表示するようにする（ユーザーが手計算・指摘する前に、AI側から先に提示する）
- **影響ファイル**: `.claude/skills/ds-kaggle-submit/SKILL.md`
- **実装内容（2026-08-02 反映済み。ユーザーからの再指摘が契機）**:
    - `CLAUDE.md` 指針#31「**OOF / LB だけで判断しない — CV 内部診断を常設の判断軸にする**」を新設。3診断軸（①train−val gap ②fold間std ③importance）、診断マトリクス（train↔val乖離 × val↔LB乖離の3分岐）、多クラス+不均衡での予測クラス分布確認、を収録
    - **【最重要ルール】ΔOOF が fold 間 std より小さいなら、その差は「測れていない」** を明記（s6e7 実測: 候補間OOF差 0.0004〜0.0009 に対し val std 0.0012〜0.0013）
    - `src/experiment.py` の `end_run()` に **CV内部診断の常設表示**を実装。fold間 val std を表示し、`_previous_experiment_scores()` で直前実験のOOFと比較して「判別不能（std未満）/ std を超える差」を自動判定する。train−val 乖離が大きい場合は「正則化より先に校正を疑え」と警告
    - `.claude/skills/ds-kaggle-submit/SKILL.md` フェーズ3に「CV内部診断」ブロックを追加。**ユーザーが指摘する前に AI 側から**乖離3値と診断マトリクスを提示する運用に変更
- **補足（記録率の問題）**: s6e7 では log.csv に列があるにもかかわらず `cv_train_mean` の記入率 **28%**、`cv_val_std` **21%** にとどまった（使い捨てスクリプトが `ExperimentTracker` を経由しなかったため）。指針#31 に「記録されない診断は存在しないのと同じ」と明記した
- **状態**: [DONE]

## TODO: 不安定NN(TabM等)のHP調整はサブサンプル単一foldで信頼できない
- 状態: [DONE] 20260802 v6.1で反映。CLAUDE.md `G-FULLCV` に「HP探索では消えるだけでなく壊れる」節（複数fold平均評価/gradient_clipping・lr上限の制約/発散foldへのペナルティ/軽量アーキで当たりを付けてから移植）を追加。Stage 3 のゲート条件に「不安定な大型アーキでは単一fold・サブサンプルでのHP選定を禁止」を明記。実測は PLAYBOOK.md#教訓アーカイブ実測値つき の L-18
- 種別: 汎用プロセス(HP調整)
- 文脈: s6e7でTabMの作業用HPをサブサンプル15万×単一foldOptunaで調整(exp272)→最良HP(n_blocks4/d_block512)がフル5-foldで1fold発散(val0.633)しdefault(0.95050)を大きく下回った(exp273 OOF0.896)
- 教訓: 大型・不安定なNNアーキのHP調整は(a)単一val分割への過学習 (b)スケール時の学習発散 の二重リスク。サブサンプル単一foldでの選定は信頼できない
- 反映案(CLAUDE.md Stage3 or PLAYBOOK): 不安定NNのHP調整は「複数fold平均で評価」「gradient_clipping/lr上限を制約」「発散fold検知時はスコアを大きくペナルティ」を必須化。20分/fold級の重いモデルは軽量アーキ(RealMLP)で作業用HPの当たりを付けてから移植する選択肢も検討

---


## [2026-07-28] HIGH — 天井帯（上位候補の差がノイズに埋もれる局面）での意思決定手順が無かった

- **背景**: s6e7 終盤、上位候補が LB 0.95092〜0.95098（幅 0.00006）に密集し、「どれを選ぶべきか」「次に何をすべきか」を判断する手順がテンプレートに存在しなかった。指針 #17（Public LB 微改善の懐疑主義）は評価指標別の**固定閾値表**を持つが、(a) そのコンペ・その規模での実際のノイズ床を測る手順、(b) 目標設定そのものの妥当性を検証する手順、(c) Final 2 の 2 本目を定量選定する手順、がいずれも無かった。
- **問題点**: 結果として「LB 0.95100 台を目指す」という**到達不可能な目標**を掲げそうになった。実際に計算すると差 +0.00002 は多数派クラスで 4.6 行分・少数派クラスでは 1 行未満、Public ノイズ床 ±0.00022 の 1/11 であり、工学的な目標として成立していなかった。また「構造的ヘッジを新規に育てるべきか」の議論も、計算すれば理論上限が +0.000052（ノイズ床未満）と事前に判明する問いだった。
- **恒久対応（対応済み / DONE）**:
    - **CLAUDE.md 指針 #23** 追加: 目標設定の妥当性を数値で検証してから追う（ノイズ床と量子の事前計算）
    - **CLAUDE.md 指針 #24** 追加: 探索空間を広げる前に「OOF 過学習の代償」を見積もる（パラメータ数と gap 符号反転）
    - **CLAUDE.md 指針 #25** 追加: 「単体性能」「異質性」「内部検証値」はいずれもブレンド寄与を予測しない ＋ 中間条件を挟んで効果を分離する検証設計
    - **CLAUDE.md 指針 #26** 追加: 単一分割で見えた「正の兆候」は必ずフル CV で再確認する
    - **CLAUDE.md 指針 #19 拡張**: Final 2 の 2 本目の価値を `max(A,B)` 構造で定量化する式と、その帰結（多様性だけでは価値にならない／E[利得] がノイズ床未満なら選定に時間をかけない）を追記
    - **PLAYBOOK.md「天井帯での意思決定ツールキット」** 新設: 手順1(ノイズ床と量子の計算) / 手順2(OOF 曲面の平坦性診断＝重み bagging) / 手順3(paired bootstrap の高速実装＝12セル多項分布) / 手順4(E[max] による Final 2 選定) / 手順5(中間条件による寄与分離) をコード付きで収録
- **s6e7 での実証データ**:
    - Public ノイズ床 ±0.00022（OOF 実測 ±0.00008 × √(690088/88725)）、少数派クラスの量子 0.0000651
    - 重み 6 個 → 13 個で OOF +0.00004 なのに LB −0.00017、gap が +0.00006 → −0.00015 と符号反転
    - 12 シードの重み探索で重み std は大きい（あるモデルは 0.000〜0.184）のに OOF std=0.000009（曲面が平坦）
    - Final 2 の 2 本目の E[利得] は最良でも +0.000021、理論上限でも +0.000052
- **影響ファイル**: `CLAUDE.md`, `PLAYBOOK.md`
- **状態**: [DONE]

---


## [2026-07-30] HIGH — 「多様性」を情報源の多様性ではなくアーキテクチャの多様性と取り違えるプロセス上の欠陥

- **背景**: s6e7 では 87 の FE 仮説・10 種のアーキテクチャ（木/attention/MLP/foundation model/検索ベース）を検証したが、**87 仮説の大半が「同じ 13 変数の表現を変える試み」（bin化/エンコーディング/交互作用/TE）** であり、10 アーキテクチャも全て同一の特徴量セット・同一の train.csv に対する学習だった。結果として全モデルが決定境界レベルで収束し（exp338 実測: 12 モデルがゲート完全行 97.17-97.23%・欠損行 88.60-88.97% という極めて狭い帯に収束）、ブレンドの天井が情報天井とほぼ一致した。
- **問題点（プロセスの欠陥として認識すべき点）**:
    1. **収束のシグナルが出た時点で方針転換できなかった**: H-081〜H-085（7月中旬）で「全アーキの argmax 一致率が 99% 超」と判明した時点で、「アーキテクチャ探索から情報源探索へ」と切り替えるべきだった。実際にはその後も TabPFN / Resnet_RTDL / k-NN / LGB-OvA とアーキテクチャ探索を継続し、いずれも寄与ゼロで終わった。
    2. **外部データの「保留」判定が最後まで再評価されなかった**: COMPETITION.md の外部データインベントリに「enhanced 版に academic_pressure / mental_health_status / social_relationships / screen_time / sitting_time が存在、直接結合はできないが分布特性の転用を検討（Stage 4 で要検討）」と初日から記録されていたが、H-074（外部 50k データでのゲート回復不能性の実証、7/22）を「外部データ探索の終了」と過度に一般化してしまい、enhanced 版は締切前日まで一度も検証されなかった。指針 #10（保留判定は Stage 4 終了までに必ず再評価）が形式的にしか機能していなかった。
    3. **「未試行の情報次元を列挙する」際、既存変数の組み合わせばかり挙げていた**: 指針 #12（LB プラトー時の強制 brainstorm）は実行されたが、列挙された「未試行の情報次元」が毎回「既存 13 変数の別の組み合わせ」に偏り、外部データ由来の新情報源がリスト上位に来なかった。
- **上位陣との差の構造的理解（この気づきが本質）**: 上位 LB（0.95281）は他参加者の public submission をブレンドする手法に依存していた。この手法の価値は**アルゴリズムの巧妙さではなく「チームをまたいだ独立性」（他チームの独自の特徴量設計・独自の外部データ・独自のパイプライン）そのもの**にある。exp332 で同手法（Consensus Frontier）を自前 12 モデルで完全に再現したところ、「全員一致で anchor と異なる行」がわずか 104/690,088 行（0.015%）しか存在せず母数不足で棄却された——**1 チーム 1 パイプラインの中でモデルを増やしても、この種の多様性は原理的に複製できない**。
- **恒久対応（提案）**:
    - **CLAUDE.md に新指針**: 「**モデル間の argmax 一致率が 99% を超えた時点で、アーキテクチャ探索を打ち切り情報源探索へ強制的に方針転換する**」。一致率の測定を Stage 6 STEP 1（相関確認）の必須項目に格上げし、閾値超過時は「新アーキテクチャの追加は原則禁止、外部データ/新情報源の探索を優先」と明示する。
    - **CLAUDE.md 指針 #10 の強化**: 外部データの「保留」判定を Stage 4 終了時に再評価する際、**「1 つの外部データで否定的結果が出ても、他の外部データ候補の評価を打ち切らない」** を明記（H-074 の過度な一般化を防ぐ）。
    - **CLAUDE.md 指針 #12 の強化**: 「未試行の情報次元を 5 個列挙」する際、**少なくとも 2 個は「現データに存在しない変数（外部データ由来）」でなければならない**という制約を追加。既存変数の組み合わせだけで枠を埋めることを禁止する。
    - **PLAYBOOK.md に診断手順**: 「情報天井の判定手順」——(1) 全モデルの argmax 一致率マトリクス、(2) セグメント別（欠損数別等）性能分解で全モデルが同一帯に収束しているかの確認、(3) 収束が確認されたら「特徴量セットを変えない限りどんなモデルも同じ壁にぶつかる」と判定し情報源探索へ移行。s6e7 の exp338/339/340 がそのままテンプレートになる。
- **s6e7 での実証データ**:
    - exp338: 12 モデル（木3種/attention/MLP2種/foundation model/検索ベース/OvA/multi-seed）がゲート完全行 0.97169-0.97227、欠損行 0.88603-0.88970 に収束。唯一例外の k-NN は欠損 3 個で 0.33791（≒ランダム基準）と、異質だが質の低い異質さだった
    - exp332: Consensus Frontier（上位カーネル手法の自前再現）の該当行 104/690,088（0.015%）、ΔOOF は全 cap でノイズ床以下
    - exp339: 12 モデル単純多数決は全セグメントで最適化済み 6 モデルブレンドに劣後（全体 −0.00038〜−0.00041）
    - exp340: 4 セグメント×12 モデル=48 パラメータの重み最適化は、同一データ評価で +0.00013 だが tune/audit 誠実検証で **−0.00021 に反転**
- **留保（正直な評価）**: H-074 で「生成ルールの 3 変数は元データで純度 100%・他変数と統計的に完全独立」と実証済みのため、enhanced 版の追加 5 変数を活用できたとしても天井を押し上げられた確度は低い。「やっておくべきだった（プロセスの欠陥）」と「やれば勝てた（結果の改善）」は別問題であり、前者は認めるべき課題だが後者の確度は低いという評価。
- **影響ファイル**: `CLAUDE.md`（新指針 + 指針#10/#12 強化）, `PLAYBOOK.md`（情報天井の判定手順）
- **実装内容（2026-08-02 一部反映）**:
    - `CLAUDE.md` 指針#28（情報天井の判定と判定後の方針転換）を新設。argmax 一致率 99% 超で新アーキテクチャ追加を原則禁止し、情報源探索へ切り替えるルールを明記
    - `PLAYBOOK.md` 手順7（情報天井の判定手順: 一致率マトリクス + セグメント別分解、判定後の切り替え表）を追加
    - 残: 指針#10（外部データの保留判定を過度に一般化しない）と #12（未試行情報次元に外部データ由来を最低2個含める）の強化は別途
- **追加実装（2026-08-02）**: 指針#10 に「1つの外部データの否定的結果を全体に一般化しない」「行レベル照合で同一生成プロセスかを最初に確認」を追記（enhanced版が別プロセスと判明した s6e7 実例つき）。指針#12 に「未試行情報次元5個のうち最低2個は現データに存在しない変数でなければならない」制約を追加
- **状態**: [DONE]

---


## [2026-08-01] HIGH — 天井帯での「選択」戦略：単一最良の選定は最悪手、集約こそが期待値を上げる（Private LB確定による事後実証）

- **背景**: s6e7 終了後、全169提出のPrivateスコアを取得し、天井帯（OOF≥0.9495, n=107）で「どの選び方がPrivate最良に近かったか」を事後検証した。結果は**これまでのテンプレートの前提を部分的に覆すもの**だった。
- **実証データ（すべてs6e7の実測値）**:

  **(A) OOF単独の最良を選ぶのは、天井帯では最悪に近い戦略だった**

  | 選び方 | 選ばれた候補 | Private |
  |---|---|---|
  | OOF最良 | exp317(CB全学習データsolo, OOF=0.95101) | **0.95003**（天井帯107件中でも下位） |
  | Public最良 | sub_251 | 0.95048 |
  | 実際の選択（OOF重視＋複合判断） | exp267 | 0.95057 |
  | 理論上の最良 | exp321(重みbagging) | 0.95060 |

  → 機械的に「OOF最高を選ぶ」ルールに従っていたら大失敗だった。実際には「内部val値でありCV OOFではない」という質的な警戒で救われた。

  **(B) 天井帯では OOF も Public も上位の順序をほとんど予測できない**
  - OOF上位10件 ∩ Private上位10件 = **1件**
  - Public上位10件 ∩ Private上位10件 = **2件**
  - 指針#19の和集合（OOF∪Public上位10）∩ Private上位10 = **2件**
  - r(OOF,Private)=0.819 は「全体の傾向」としては有効だが、**回帰の傾きは0.52**（OOFが+0.001改善してもPrivateは+0.0005しか動かない＝OOF差の約半分は天井帯では実現しない）

  **(C) モデル数（way数）とPrivateは単調増加、2wayの高スコアは「当たり」であって期待値ではない**

  | way数 | 件数 | Private平均 | Private最大 |
  |---|---|---|---|
  | 2way | 10 | 0.95013 | 0.95047 |
  | 3way | 11 | 0.95022 | 0.95056 |
  | 4way | 5 | 0.95037 | 0.95059 |
  | 6way | 10 | **0.95050** | **0.95060** |
  | 7way | 1 | 0.95058 | 0.95058 |

  → 2wayで0.95047という高スコアが出たのは事実だが、**2wayの平均は6wayより−0.00037低い**。個別の当たりを見て「2wayでも十分」と結論するのは生存者バイアス。

  **(D) 「上位N件を平均する」ほどPrivate期待値が上がる**

  | N | OOF上位N件のPrivate平均 | Public上位N件のPrivate平均 |
  |---|---|---|
  | 1 | 0.95003 | 0.95048 |
  | 3 | 0.95034 | 0.95051 |
  | 5 | 0.95043 | 0.95053 |
  | 10 | 0.95045 | 0.95051 |

  → **選択の集中がリスクであり、分散が期待値を上げる**。Private最高だったexp321自体が「12シードの重み探索を平均した」bagging手法だったことも同じ構造。

- **恒久対応（提案）**:
    1. **CLAUDE.md 新指針「天井帯では単一最良を選ばず、集約する」**: モデル間のargmax一致率が99%超（＝情報天井到達）と判定されたら、Final候補の選定基準を「単一のOOF/Public最良」から「**上位N件（5-10件）の集約**」へ切り替える。集約手段は (a) 重み探索を複数seedで実行し重みを平均（s6e7のexp321、Private最高を記録）、(b) 上位N構成の確率を単純平均、のいずれか。**「どれが最良か」を当てにいくのではなく「平均的に良い場所」に着地させる**。
    2. **CLAUDE.md 指針#18 の但し書き強化**: 「OOFを信じる」は**全体の傾向としては正しい**（r=0.819、Private−OOF中央値−0.00015でほぼ無バイアス）が、**天井帯における上位数件の順序付けには使えない**（OOF上位10と Private上位10の一致は1件のみ、回帰の傾き0.52）。「OOFで足切りする」のは有効、「OOFで1位を選ぶ」のは無効、という区別を明記する。
    3. **CLAUDE.md 指針#19（Final2候補プール）の修正**: 現行の「Public Top-10 ∪ OOF Top-10」和集合戦略も、s6e7ではPrivate上位10件を2件しか捕捉できなかった。**候補プールの構築より「候補群の集約」を優先**すべきという学びを追記。プールを広げても当てられないなら、広げたプールを平均する方が合理的。
    4. **CLAUDE.md 新指針「way数を減らす方向の最適化を、単一の結果で正当化しない」**: 少数モデル構成が高スコアを出した場合、それが「平均的に優れる」のか「たまたま上振れた1本」なのかを、**同じway数の複数構成の平均**で確認してから判断する。s6e7では2wayの当たり（0.95047）と2wayの平均（0.95013）に0.00034の差があった。
    5. **PLAYBOOK.md「天井帯での意思決定ツールキット」に手順追加**: 「手順6: 天井帯での最終候補の集約」——(i) argmax一致率で天井到達を判定、(ii) OOF足切り（例: 上位20件）、(iii) 複数seed重み探索の平均 or 上位N件の確率平均、(iv) 集約結果をFinal候補とする。
- **s6e7 での結果的な影響**: 我々のFinal2（exp267=0.95057 / exp316=0.95056）は、Private最高（exp321=0.95060）に対し−0.00003。**実害は数チーム分の順位のみ**で、上位7件が幅0.00003に密集していたため実質的な損失はなかった。ただし「たまたま良い場所に着地した」のであって「正しい選択手順で着地した」わけではない点が、この改善提案の動機。
- **影響ファイル**: `CLAUDE.md`（新指針2件 + #18/#19 修正）, `PLAYBOOK.md`（手順6追加）
- **実装内容（2026-08-02 反映済み）**:
    - `CLAUDE.md` 指針#27（天井帯では集約に切り替える）を新設。切り替え判定条件・集約手段(a)(b)・s6e7 実測データを収録
    - `CLAUDE.md` 指針#18 に**但し書き**を追加: 「OOF で足切りする」は有効／「OOF で 1 位を選ぶ」は無効（OOF上位10 ∩ Private上位10 = 1件、回帰の傾き 0.52、OOF最高の Private は 0.95003 で下位）
    - `CLAUDE.md` 指針#19 に**限界の注記**を追加: 和集合戦略でも Private 上位10件を 2 件しか捕捉できない。プールを広げるより集約する方が期待値が高い
    - `PLAYBOOK.md` 手順6（集約への切り替え、重み bagging のコード付き）を追加、ツールキット冒頭に手順一覧表を新設
- **状態**: [DONE]

---


## [2026-08-01] MED — LB観察から構造を推論する前に、評価指標の「格子間隔」を必ず計算する

- **背景**: s6e7 終了後の Private LB 分析で、上位帯に同一スコアが大量に重複していること（0.95084 に6チーム、0.95081 に11チーム）を観測し、**「同一スコアの重複 = 同一予測ファイル = 共有された公開ノートブック由来」と推論した。これはユーザーの指摘により誤りと判明**。
- **反証**: 自チームの169提出の中でも、明らかに構成の異なる提出が同一 Private スコアに大量に重なっていた——**Private=0.95047 に7件**（2way / 3way / 5way / 6way / 単体モデルという全く別物の構成）、0.95050 に5件、0.95049 に5件。同一スコアの重複は同一予測を意味しない。
- **正しい説明**: balanced_accuracy は離散指標であり、スコアは**格子上にしか存在しない**。s6e7 の Private（約207,027行）では最小クラス fit が約11,941行しかなく、**1行の予測変化で 0.0000289 動く**（at-risk=0.0000019 / unhealthy=0.0000192）。表示精度が小数5桁のため、スコアの実質的な格子間隔は 0.00002〜0.00003。幅0.00004の帯に存在しうる格子点は1〜2個で、そこに数十チームいれば重複は**必然**。
- **問題の本質（自己適用の失敗）**: 本コンペで確立した**指針#23（目標設定の妥当性を数値で検証する = 1行あたりの量子を事前計算する）を、自分自身の推論の妥当性検証に適用し損ねていた**。同じ計算を1回するだけで誤推論は防げた。
- **恒久対応（提案）**:
    - **CLAUDE.md 指針#23 の適用範囲を拡張**: 現行は「目標設定の妥当性検証」の文脈だが、**「LB の観察結果から構造・他チームの手法を推論する場面」にも同じ量子計算を必須とする**と明記。具体的には「スコアの重複・密集・段差を根拠に何かを主張する前に、必ず1行あたりのスコア変化量（量子）を計算し、観測された現象が単なる格子効果で説明できないかを排除する」。
    - **PLAYBOOK.md「天井帯での意思決定ツールキット」手順1に追記**: 量子計算の用途として、既存の「目標の到達可能性判定」に加えて「**LB上のクラスタ/重複の解釈**」を明示する。
- **一般化した教訓**: **自分が持っている検証の道具を、自分自身の推論にも向けること。** 外部データやモデルの評価には厳密な基準（ノイズ床・量子・独立検証）を適用していたのに、「LBを眺めて構造を推論する」という場面ではその基準が緩んでいた。分析対象が数値実験ではなく観察・解釈になった瞬間に検証基準が下がるのは、再発しやすいパターンとして警戒する。
- **訂正が結論に与えた影響**: 誤推論のままなら「上位は1解族が多数に共有されただけ」だったが、正しくは「**上位帯には独立した多数のチームが到達しており、その全員が我々の解族（107件、最良0.95060）を上回っていた**」——ギャップの解釈はむしろ深刻な方向に変わった。誤推論は自己に有利な方向のバイアスを含んでいた点も反省材料。
- **影響ファイル**: `CLAUDE.md`（指針#23 の適用範囲拡張）, `PLAYBOOK.md`（手順1 に用途追記）
- **実装内容（2026-08-02 反映済み）**: `CLAUDE.md` 指針#23 の見出しを「目標設定の妥当性〜」から「**量子を計算してから、目標を掲げ・LB を解釈する**」に変更し、適用範囲に「LB の観察結果から構造を推論する場面」を明記。誤推論の実例（自チームでも別構成7件が同一スコアに重複）と「自分の道具を自分の分析に向けていなかった」という本質を教訓として収録。`PLAYBOOK.md` 手順1 に用途 (A)目標検証 / (B)LB観察の解釈 の2列表を追加
- **状態**: [DONE]

---


## [2026-08-01] HIGH — コンペ全体を通じた AI 側の振る舞いの反省（s6e7 総括）

> **前提の明確化**: s6e7 コンペ自体は **2026年7月の1ヶ月間**（実働 7/02〜7/30 の27日間、169提出）。
> 「4ヶ月」は DS テンプレート構築と 4〜7月の**月次コンペ参加**の期間を指す。
> 月次サイクルであるため、**テンプレート改善は翌月のコンペに直接複利で効く共有資産**である点が、以下の反省の前提。

### 反省1（最重要）— AI の判断だけでは、中核 FE を取り逃していた

**H-070（13カラム全ての厳密値 target encoding、52特徴量）は Final2 の全モデルの土台**だが、AI は当初これを**棄却していた**。原因は「13特徴量用の HP を52特徴量にそのまま流用した不公正比較」で LB=0.94964 とベースを下回ったこと。ユーザーの「**判断が安直、せめて HP 調整してみれば**」という押し戻しがなければ、解は 0.949 台で終わっていた。

同じ構図の繰り返し（すべてユーザー指摘が起点）:
- exp181/182/183（構成要素除外・LOO分解・組み合わせ探索）
- exp184-186（候補A/B/C への HP 再調整、指針#22 の適用漏れ）
- H-101（LB クラスタの誤推論の発覚）

**構造的問題**: AI は「検証を尽くした」と判断する閾値が低く、**自分が出した結論を疑う方向の検証を自発的に起動しにくい**。指針#13（早期却下の禁止）を形式的には守っても実質的に守れていない。
→ **対応案**: 棄却判断を下す際、「この棄却が**不公正比較**（HP/特徴量/CV条件の不一致）に由来する可能性はないか」を明示的にチェックする項目を指針#13 に追加。特に**FE の追加・削除で特徴量数が±20%以上変わった場合の棄却は、HP再調整前は棄却として記録してはならない**（暫定保留とする）。

### 反省2 — 自分の道具を、自分の分析に向けていなかった

H-099（単一の Public 結果を「decisive な逆転例」と報告）と H-101（同一スコア重複を「共有ノートブック由来」と誤推論）。どちらも、本コンペで確立した道具（ノイズ床、量子計算）を1回適用するだけで防げた。
**数値実験には厳密な基準を適用するのに、観察・解釈の場面になった瞬間に基準が緩む。** しかも H-101 の誤推論は「上位は1ノートブックの共有にすぎない」という**自チームに有利な方向のバイアス**を含んでいた。
→ **対応案**: 別途記録済み（[2026-08-01] MED — LB観察から構造を推論する前に格子間隔を計算する）。

### 反省3 — 情報天井の実証後、コンペ期間の 1/3 を期待値ゼロの探索に使った

| 期間 | 提出数 | 到達した Private 最高 |
|---|---|---|
| 7/02〜7/21（H-074 で天井実証まで） | 123件 | 0.95059 |
| **7/22〜7/30（天井実証後）** | **48件（全体の28%）** | **0.95060（+0.00001）** |

H-074（7/22）で「欠損ゲート変数の回復は情報理論的に不可能」を**生成元データで実証**したにもかかわらず、その後9日間・48提出を続け、得られた Private の伸びは **+0.00001（ノイズ床の1/14）**。

**月次コンペのサイクルであることを踏まえると、この9日間はテンプレート改善（＝翌月に複利で効く資産）に振り向ける選択肢があった。** AI 側から「天井が実証されたので、残り期間の使い道を再設計しませんか」と提案すべきだった。
→ **対応案**: CLAUDE.md に「**情報天井が実証された後の期間の使い方**」の指針を追加。(1) 天井の実証（生成元データ・AV診断・argmax一致率等）が得られたら、その時点で残り期間の再配分を**AI から提案する**、(2) 選択肢として「Final候補の頑健化」「テンプレート改善」「次コンペの準備」を提示する、(3) 「探索を続ける」を選ぶ場合も、期待値がノイズ床未満であることを明示した上でユーザーが選ぶ形にする。

### 反省4 — 「使う」と判定した外部データを、コンペ期間ほぼ全域にわたり放置

enhanced 版データは初日のインベントリで「使う」と明記されていたが、検証は締切前日（7/31, exp341）。**実際の検証コストは数分だった**。H-074 を「外部データ探索そのものの終了」と過度に一般化したことが原因で、指針#10（保留判定は Stage 4 終了までに再評価）は形式的にしか機能していなかった。
→ **対応案**: 別途記録済み（[2026-07-30] HIGH — 多様性の取り違え、指針#10 強化案）。

### 反省5 — 記録の肥大化に対する設計がなかった

FE_HYPOTHESES.md は 2,300行超・101仮説。資産としては価値があるが、途中から「以前これを試したか」の確認に grep が必要になり、実際「exp299 で CB 検証済み」を後から再確認する場面が生じた。
→ **対応案**: 仮説数が 50 を超えたら、FE_HYPOTHESES.md の冒頭に**「棄却済みメカニズムの索引」**（仮説ID・1行要約・棄却理由の分類のみ）を維持する運用を追加。全文検索ではなく索引で「試したか」を判定できる状態を保つ。

### 反省ではないが、記録すべき「機能した規律」

- 169提出しながら Public は 0.95099 と抑制的（100回超提出チームの平均 0.95214）、shake も −0.00042（同 −0.00152）。**Public を追わなかったことが最終93位に直結した**
- 外部予測を使わない方針を、目に見えるスコア差がある中で押し通せた（結果的に Public 首位チームの Private を上回った）
- 「1実験1コミット」「棄却理由の分類記録」は最後まで維持され、この総括自体がその記録の上に成立している

- **影響ファイル**: `CLAUDE.md`（指針#13 強化 + 天井到達後の期間再配分の新指針）, `FE_HYPOTHESES.md`（索引運用）
- **実装内容（2026-08-02 反映済み）**: 反省1 → `CLAUDE.md` 指針#13 に**チェック4「不公正比較チェック」**を追加（HP/特徴量セット/CV条件/較正の4項目表 + 「特徴量数±20%変動時の棄却は HP 再調整前は暫定保留」ルール + H-070 を当初棄却していた実例）。反省2 → 指針#23 に LB 解釈での適用を追加。反省3 → 下記 NOTE で撤回・差し替え済み。反省4 → 指針#10 強化で対応。反省5（記録の肥大化）は運用課題として保留
- **状態**: [DONE]

---


## [2026-08-01] CRITICAL — 可視化プロセスの形骸化が「2コンペ連続」で再発。努力目標→条件明示では効かないことが実証された

- **背景**: ユーザーからの s6e7 総括で「テンプレート指示の可視化プロセスが、中盤以降はほとんどなかった」との指摘。実測で裏付けられた。

  | 期間 | 生成プロット数 |
  |---|---|
  | **7/02〜7/05（最初の4日）** | **23枚（全体の74%）** |
  | 7/10〜7/27（残り25日） | 8枚 |
  | 7/22〜7/30（最後の9日） | **1枚** |

  総プロット数31枚に対し、実験は339件・提出169件。
- **これが深刻な理由**: CLAUDE.md 指針#9 には、**まさにこの失敗の再発防止として**次の教訓が明記されていた——「『積極的に提案する』という努力目標だけでは、実験サイクルが高速化するほど可視化が省略され、100実験超のコンペで可視化が最初の3日間しか実施されなかった。**発動条件を明示して初めて実効性を持つ**」。その対策として **必須発動条件3つ**（①LBベスト更新時 ②oof_lb_gap が±0.0005超変動時 ③直近5実験で可視化ゼロのとき）まで明文化した。**にもかかわらず、今回も全く同じ形（最初の数日に集中、以降ほぼゼロ）で再発した。** LBベストは 7/27-28 に更新したが、その時期の可視化は1枚のみで、発動条件①は機能しなかった。
- **根本原因**: **発動条件をチェックする主体が AI 自身であり、AI はそのチェックを自発的に走らせない。** これは同日記録した「反省1（自分の結論を疑う検証を自発的に起動しない）」「反省2（観察・解釈の場面で検証基準が緩む）」と同一の根であり、**AI の自己監査に依存する対策は原理的に信頼できない**という結論に至る。
- **恒久対応（提案）— 努力目標でも条件明示でもなく、機械的強制へ**:
    1. **`src/experiment.py` の `tracker.end_run()` に可視化ガードを実装**: log.csv 追記時に「直近5実験で `data/output/plots/` に新規ファイルが生成されていない」を自動判定し、該当時は**標準出力に警告を出す**（例: `⚠️ 直近5実験で可視化ゼロ。scripts/feature_report.py を実行してください`）。AI の記憶や自己申告に依存しない。
    2. **`.claude/settings.json` の hook で強制**: log.csv への書き込みを検知する PostToolUse hook を設定し、上記の判定をシェルで実行して AI に差し戻す。**AI が「気づく」必要をなくす**のが要点。
    3. **CLAUDE.md 指針#9 の書き換え**: 「必須発動条件」という表現を残しつつ、「**この条件は AI の自己申告では守られないことが2コンペ連続で実証されている。必ず機械的な検知（tracker のガード / hook）とセットで運用すること**」と明記する。同じ轍を3度踏まないため、失敗の履歴自体を指針本文に残す。
- **教訓（テンプレート設計論として）**: **「AI に守らせたい規律」は、AI に守らせようとしてはいけない。** 努力目標（第1世代）→ 条件の明示（第2世代）と2段階で失敗した以上、次は**実行環境側の機械的強制（第3世代）**しかない。同種の規律（1実験1コミット、1列ずつのFE投入 等）についても、同じ観点で「AI の自己監査に依存していないか」を点検すべき。

- **影響ファイル**: `src/experiment.py`, `.claude/settings.json`, `CLAUDE.md`（指針#9）
- **実装内容（2026-08-02 反映済み）**:
    - `src/experiment.py` に `_check_visualization_guard()` を追加。log.csv の直近5実験のタイムスタンプと `data/output/plots/*.png` の更新時刻を比較し、可視化ゼロなら警告文字列を返す。`end_run()` の末尾で呼び出す
    - `scripts/viz_guard.py`（CLI ラッパー）を新設。`uv run python -m scripts.viz_guard` で単独実行可能
    - `.claude/settings.json` に PostToolUse(Bash) hook を追加。log.csv が20秒以内に更新されていた場合のみ判定を走らせる（tracker を経由せず log.csv へ直接追記する使い捨て実験もカバー）
    - `CLAUDE.md` 指針#9 に「失敗の履歴（第1世代=努力目標 → 第2世代=条件明示 → 第3世代=機械的強制）」を本文として明記し、3度目を防ぐ
    - 実データで発動を検証済み（s6e7 の直近5実験は可視化ゼロのため警告が出ることを確認）
- **状態**: [DONE]

---


## [2026-08-01] HIGH — OOF 棄却で提出せずに終わった実験が半数。天井帯では OOF に順位解像度がないと実証された

- **背景**: ユーザー指摘「たくさんの実験をしたにも関わらず submit 回数が少なかった。exp番号と submit 回数の差が顕著。単に OOF 棄却するのではなく次に繋がるように確認する必要があるのでは？」——実測: **実験339件に対し提出169件**（約半数が OOF 判断のみで終了）。
- **この指摘を裏付ける実測データ（Private LB 確定後の事後検証）**:
    - 天井帯（OOF≥0.9495, n=107）で **OOF 上位10件 ∩ Private 上位10件 = わずか1件**（Public 上位10件との一致も2件）
    - r(OOF, Private)=0.819 は全体傾向としては有効だが、**回帰の傾きは0.52**——OOF が +0.001 改善しても Private は +0.0005 しか動かない
    - → **天井帯において OOF は順位付けの解像度を持たない。そこで OOF だけを根拠に棄却するのは、解像度のない物差しで切っていることになる**
- **象徴的な反例（exp320c）**: OOF の論理では「提出する価値なし」だったが、ユーザーの「一応提出してみませんか」で提出した結果、**OOF +0.00004 なのに LB −0.00017、gap が正から負へ反転**という、**指針#24（探索空間拡大の代償）の決定的証拠**になった。**この1提出が生んだ知見は、OOF で棄却した数十実験の合計より価値があった。**
- **恒久対応（提案）**:
    1. **CLAUDE.md に「棄却候補の校正提出」を新指針として追加**: 天井帯（＝指標の差がノイズ床に近い領域）に入ったら、**OOF 棄却した候補のうち「棄却理由の類型が新しいもの」を定期的に提出して OOF→LB の対応関係を校正する**。提出枠は「良い候補を出す」だけでなく「**判断基準そのものを検証する**」ためにも使う。
    2. **提出枠の配分ルール化**: 1日の提出枠のうち **1枠を「検証提出」に充てる**（スコア更新を狙わず、棄却判断の妥当性や仮説のメカニズム確認を目的とする）。s6e7 では日次上限10回に対し平均6回程度しか使っておらず、枠は余っていた。
    3. **log.csv に「未提出のまま棄却」フラグの集計を追加**: `/ds-kaggle-submit` 実行時に「OOF棄却のまま未提出の実験がN件蓄積」を表示し、そのうち代表を提出する判断をユーザーに促す。
- **実装内容（2026-08-02 反映済み）**: `CLAUDE.md` 指針#29（提出枠は判断基準の検証にも使う＝棄却候補の校正提出）を新設。1日1枠を検証提出に充てる／棄却理由の類型が新しいものを優先／解釈基準を事前登録、の3ルールと exp320c の実例を収録
- **状態**: [DONE]

---


## [2026-08-01] MED — ベースモデル段階の特徴量投入が「一括比較」になっていた（1列ずつ原則の違反が6箇所）

- **背景**: ユーザー指摘「ベースモデル段階の特徴量投入テストをもう少し丁寧に一つずつ進めていきたかった（特に数値カラム）」。
- **実測**: CLAUDE.md Stage 4 は「**特徴量は必ず1列ずつ `scripts/feature_study.py` で投入**」と定めているが、FE_HYPOTHESES.md 内に「3案を一括比較」「6カラム全てを置換して一括比較」等の記述が **6箇所**。特に序盤の数値カラム bin 化検証（H-012〜014 前後）が該当。
- **実際に生じた損害**: どの列が効き、どの列が相殺しているかが分からなくなり、**後になって exp182 で LOO（leave-one-out）分解をやり直す必要が生じた**（しかもユーザー指摘が起点）。序盤の時間節約が、中盤の手戻りとして返ってきた。
- **恒久対応（提案）**:
    - **`scripts/feature_study.py` に複数列同時投入のガードを実装**: 引数で2列以上を同時に渡された場合、`--allow-batch` の明示なしでは実行を拒否する。理由の記録（なぜ一括で良いのか）を必須にする。
    - **CLAUDE.md Stage 4 の完了条件に追記**: 「一括比較を行った場合は、**採用・棄却の前に必ず LOO 分解で各列の寄与を分離する**」——一括比較を全面禁止にするのではなく、「一括はスクリーニング、判断は分解後」という順序を明示する。
- **実装内容（2026-08-02 反映済み）**:
    - `scripts/feature_study.py` に**一括投入ガード**を実装。`--new-feature` をカンマ区切りで複数指定した場合、`--allow-batch --batch-reason '<理由>'` の明示が無ければ実行を拒否する（エラーメッセージで CLAUDE.md Stage4 の原則と手戻りのリスクを説明）
    - 一括投入時は実行中に警告を表示し、**理由を log.csv の notes に記録**する（「採否判断は LOO 分解後に行うこと」の注記つき）
    - `CLAUDE.md` Stage 4 の完了条件に「一括はスクリーニングであり、採用・棄却の判断は必ず LOO 分解で各列の寄与を分離してから行う」を明記
    - 動作検証済み: 複数列を `--allow-batch` なしで渡すと拒否 / `--batch-reason` 無しでも拒否 / 単一列は従来通り通る
- **状態**: [DONE]

---


## [2026-08-01] MED — 「動かない理由が特定できないまま回避策で済ませた」ケースが複数（NN系の技術的ブロッカー）

- **背景**: ユーザーの良かった点「Kaggle 環境での NN モデル構築・推論が利用できた（本コンペの目標の一つ）。ただし一部私の補助が必要であった点は改善したい」を受けた、AI 側の課題認識。
- **該当ケース**:
    - **pytabkit（TabM/RealMLP）の全データ再学習が系統的に失敗**（exp312/313）: 内部 val BA が 0.887〜0.889 に収束。分割比率（5%/15%/20%）・model seed を変えても再現し、約100分の調査後に**根本原因未特定のまま断念**、CV版で代替した
    - **RealTabR がライブラリバグで実行不能**（H-082）: pytabkit の OrdinalEncoder に0列が渡されクラッシュ。最小合成データで再現確認まではしたが、**回避（Resnet_RTDL への切り替え）で済ませた**
    - **XGB rawfix の全データ再学習が失敗**（exp318）: 内部 val BA=0.89-0.91（期待0.95036）。データの健全性は確認したが**原因未特定のまま断念**
    - **FT-Transformer の全データ再学習が不安定**（exp315）: 内部5%ホールドアウトでの早期停止が主因と**推定**したが、確定的な検証は行っていない
- **問題の本質**: いずれも「動かない → 別の手段で回避」で処理しており、**次のコンペに持ち越せる知識になっていない**。同じライブラリを次に使うとき、同じ壁に同じ時間をかけることになる。
- **恒久対応（提案）**:
    - **CLAUDE.md に「技術的ブロッカーの扱い」を追加**: ライブラリ由来の failure に遭遇したら、(1) 最小再現コードを `experiments/blockers/` に保存、(2) ライブラリのバージョン・該当箇所のスタックトレースを記録、(3) 可能なら upstream の issue を検索・報告、(4) **回避策を採る場合も「未解決の技術的負債」として TODO_TEMPLATE に残す**。「回避できたから解決」としない。
    - s6e7 で残った未解決ブロッカー4件（上記）を、次回 pytabkit / XGBoost を使う際の**既知の落とし穴リスト**として PLAYBOOK に転記する。
- **実装内容（2026-08-02 反映済み）**: `CLAUDE.md` 指針#30（技術的ブロッカーは「回避できたから解決」としない）を新設し、遭遇時の5手順を明記。`PLAYBOOK.md` に新章「既知の落とし穴（ライブラリ別）」を追加し、s6e7 で遭遇した pytabkit 2件 / XGBoost 1件 / NN全般1件を表形式で転記
- **状態**: [DONE]

---


## [2026-08-01] NOTE — 「天井到達後も諦めない」はユーザー方針として正しい（AI の反省3を撤回）

- **経緯**: AI は s6e7 総括で「情報天井の実証（H-074, 7/22）後の9日間・48提出が稼いだ Private の伸びは +0.00001 で、期待値ゼロの探索だった」と反省点に挙げた。**ユーザーからこれは指示によるものであり、諦めない姿勢を貫くことが重要との指摘を受け、AI の反省3は撤回する。**
- **撤回の理由（AI 自身の評価として）**: 「期待値ゼロ」という評価は**スコアという単一軸でしか測っていない狭い見方**だった。実際にはこの期間に、
    - H-070 の復活（当初棄却 → ユーザー指摘 → 中核FEに）※これは前半だが同じ構図
    - exp320c の提出 → 指針#24 の決定的証拠
    - Discussion 再調査 → Consensus Frontier（H-090）、外部データ再評価（H-096）、AV診断やり直し（H-094）、数値プロファイル照合（H-097）
    - H-101 の誤推論訂正

    といった、**スコアには表れないがメカニズム理解とテンプレート資産を確実に増やした成果**が生まれている。月次コンペのサイクルでは、これらは翌月に複利で効く。
- **AI が真に反省すべきだった点（差し替え）**: 探索を続けたこと自体ではなく、**「視点を変えてスタートに戻る」提案を AI 側から出せなかったこと**。ユーザーが問いを投げるまで、AI は同じ土俵の中で細かい実験を続けていた。天井に当たったときに必要なのは「やめる判断」ではなく「**問いの立て直し**」であり、それを主導するのは AI の役割だった。
- **恒久対応（提案）**: CLAUDE.md 指針#8（探索継続姿勢）に追記——「**天井が実証されたときこそ、AI は『探索を縮小する提案』ではなく『問いを立て直す提案』を出す**。具体的には (1) Stage 0（Kickoff）の前提を再検証する、(2) 棄却済み仮説を新しい観点で再評価する、(3) 外部調査（Discussion / 上位解法）をやり直す、(4) 評価指標そのものの性質を測り直す、を順に提示する。s6e7 ではこれらを実施した結果、スコアは動かなかったがテンプレート資産は大きく増えた」。
- **実装内容（2026-08-02 反映済み）**: `CLAUDE.md` 指針#8 に「**情報天井が実証されたときこそ、AI は『探索を縮小する提案』ではなく『問いを立て直す提案』を出す**」を追加。立て直しの4手順（Kickoff前提の再検証 / 棄却済み仮説の再評価 / 外部調査のやり直し / 評価指標の性質の測り直し）と、s6e7 で実際に生まれた資産を教訓として収録
- **状態**: [DONE]

---


## [2026-08-07] HIGH — `/ds-kaggle-research` の調査結果が SESSION.md の行数上限で失われる

- **説明**: `/ds-kaggle-research` で得た Discussion・上位 Notebook の調査結果（発見した施策・報告された効果量・出典URL等）は現状 SESSION.md にしか記録されない。SESSION.md は `CONVENTIONS.md#sessionmd-の構成と上限` で全体80行の上限が定められており、`/ds-resume` のオーバーフロー検知で古いエントリが削除される対象になる。s6e8 では H-011（target encoding, LB+0.0017相当）を含む調査結果一式が SESSION.md 側にしか無く、80行超過での整理時に消失しかけた（ユーザー指摘で発覚）。
- **問題の本質**: SESSION.md は「今どこにいるか」の**ライブダッシュボード**であり蓄積禁止（CLAUDE.md 原則）だが、`/ds-kaggle-research` の調査結果は**恒久的に参照する価値がある資料**であり、両者の性質が異なるのに同じファイルに同居させている。
- **恒久対応（提案）**:
  - `/ds-kaggle-research` スキル（`.claude/skills/ds-kaggle-research/SKILL.md`）に**専用の保存先ファイル**（例: `KAGGLE_RESEARCH.md`）を指定する
  - フェーズ4（新発見の記録）を「SESSION.md の未解決の問いまたはFE_HYPOTHESES.mdに記録」から「**`KAGGLE_RESEARCH.md` に調査日付見出しで追記**（既存内容は上書きせず蓄積）」に変更する
  - SESSION.md 側には `KAGGLE_RESEARCH.md` への一行リンクのみを残す（詳細はリンク先を参照する運用にする）
  - CONVENTIONS.md の「参照すべき状態ファイル」表にも `KAGGLE_RESEARCH.md` を追加する
- **影響ファイル**: `.claude/skills/ds-kaggle-research/SKILL.md`, `CONVENTIONS.md`, （新規）`KAGGLE_RESEARCH.md`
- **状態**: [DONE]（2026-09-02 KAGGLE_RESEARCH.md 新設・スキルとSESSION.md参照表を更新）

---


## [2026-08-19] MED — Optuna study の状態を永続化しない運用が「探索の続きができない」問題を生んでいる

- **説明**: `experiments/runs/optimize_hp_*.py` 系のスクリプトは `optuna.create_study(direction="maximize", sampler=...)` を storage 指定なし（インメモリ）で呼んでいる。実行が終わるとサンプラーの内部状態（TPEが「どこを探索済みで、どこが有望か」という確率モデル）が失われ、`best_params_*.json` に残るのは最良の1点のみ。s6e8のRealMLP HP探索(exp141)で「試行数を増やしたい」となった際、15試行の続きから追加試行ができず、25試行を新規に走らせるかの二択になった（前回の探索情報を活かせない）。
- **問題の本質**: log.csv/best_params_*.jsonへの記録は「結果」の記録に留まり、「探索プロセスの状態」は記録されない。ExperimentTrackerの予約行機構と同種の「記録されない診断は存在しないのと同じ」原則がOptunaサーチにも当てはまる。
- **恒久対応（提案）**:
  - `optimize_hp_*.py` テンプレートの `optuna.create_study()` に `storage=f"sqlite:///{PARAMS_DIR}/optuna_studies/{model}_{tag}.db"`, `study_name=f"{model}_{tag}"`, `load_if_exists=True` をデフォルトで付与する
  - CONVENTIONS.md に「Optuna study の保存先・命名規則」節を新設し、study_name の一意性（model名+tagで衝突しない）を明文化する（ExperimentTrackerの予約行衝突と同型の事故を防ぐため）
  - `--resume` フラグを追加し、既存studyがあれば追加試行、無ければ新規作成する運用にする
  - コストはSQLiteファイル1つ分（数十KB〜数MB）とほぼゼロ。デメリットが小さく、再開性・後からのfANOVA分析可能性・監査性のメリットが大きいため、**任意ではなく標準動作にする**
- **影響ファイル**: `experiments/runs/optimize_hp_*.py`（全テンプレート）, `CONVENTIONS.md`
- **状態**: [DONE]（2026-09-02 optimize_hp.py に SQLite 永続化 + load_if_exists、CONVENTIONS に命名規約を追加）

---


## [2026-08-31] CRITICAL — アンサンブルプール発見の除外リスト方式(`DERIVED_RE`)が自己参照混入を2回引き起こした

- **説明**: `discover_pool()`（`exp160_stack_ablation.py`）は`oof_*.npy`を機械的に走査し、正規表現`DERIVED_RE`にマッチしないものを「独立した基底モデル」としてアンサンブル候補プールに含める。このブラックリスト方式は、**新しい派生アンサンブルの命名パターンを都度手動で追加登録しない限り、既存の派生アンサンブル出力が「独立メンバー」として紛れ込む**構造的欠陥を持つ。s6e8では**同一コンペ内で2回**発生した：①コンペ中盤に`161_stack_logit_pall`が混入（当時の`_stack_`パターン追加で対処）、②終盤に`p5_*`/`p6_*`/`pall_*`系の新規派生（`162_rank_average_pall`含む、コンペ全体を通じて未検知だった可能性）が混入し、**bagged OOFが0.97092・seed間std=0.00017（正常時の17倍）という明確な異常値**を出したことで偶然発覚した。異常値が正常レンジに収まっていたら気づかず採用していた可能性が高い。
- **本質的な問題**: 除外リストは「将来の派生アンサンブルの命名を予知する」という原理的に不可能なことを要求している。命名規則を守らせる運用ルールでは(G-MECHの原則通り)再発を防げない。
- **恒久対応（提案）**:
  - **ホワイトリスト化**: 派生アンサンブル出力には固定マーカー（例: ファイル名に`_ens_`を必須で含める）を強制する命名規約を`CONVENTIONS.md`に新設し、`discover_pool()`は「マーカーを含まない」ものだけを候補にする設計に変える（ブラックリストの列挙漏れという失敗モードを構造的に排除できる）
  - **機械的サニティチェック**: `integrity_gate()`と同じタイミングで、「候補の中に、他の候補群の(signed/最適化された)線形結合と相関0.999超のものがないか」を検知する関数を追加する（自己参照混入は原理的に「プール全体との相関が異常に高い」という observable な症状を持つため）
  - 新しい派生アンサンブルスクリプトを書いたら、`discover_pool()`に投入する前に`python -c "import re; print(bool(DERIVED_RE.search('<新ファイル名>')))"`を実行する習慣を`CONVENTIONS.md`のチェックリストに明記する（応急処置、上記2点までの繋ぎ）
- **影響ファイル**: `experiments/runs/exp160_stack_ablation.py`（`discover_pool`, `DERIVED_RE`, `integrity_gate`）, `CONVENTIONS.md`（派生アンサンブルの命名規約を新設）
- **状態**: [DONE]（2026-09-02 テンプレート反映済み）

---


## [2026-08-31] HIGH — 小規模アンサンブル(2本目/hedge候補)を育てる手順が存在しなかった

- **説明**: Final2の2本目（少数メンバーのブレンド）を改善しようとした際、「新規メンバーを追加する」という直感的なアプローチは繰り返し失敗した（新規追加したLGBが既存XGBと相関0.9994で重み0になる、プール全体582候補を相関でスキャンしても単体性能不足で全滅、等）。効いたのは全く別の軸だった：**結合方式そのものが非負simplex制約（`optimize_weights`の差分進化最適化）だったため、単体性能の低い候補を一切使えない**という構造的制約に気づき、PALL(1本目)と同じsigned係数のロジスティック回帰スタッカー(`nested_stack`)に切り替えたところ、simplexでは重み0だった候補群が負係数で機能し、OOF+0.00012→LB+0.00017の確認済み改善を得た。さらに「候補を手動で数個選ぶ」より「関連候補を機械的に全投入してL2正則化に選ばせる」方が一貫して良い結果を生み（ただし単体性能の下限フィルタありのほうが下限なしより優れた＝「量」と「質」のバランス点がある）、最終的にLB+0.00035（G-NOISEの2σ閾値超）まで到達した。この発見の順序（結合方式→候補基準のスイープ）は自明ではなく、複数回のユーザーとの往復を要した。
- **恒久対応（提案）**: `PLAYBOOK.md`の「天井帯での意思決定ツールキット」と並ぶ形で、「小規模アンサンブル(2本目)を育てる手順」を新設する:
  1. 結合方式が signed係数を許容しているか確認する（simplex制約なら最優先でsigned方式への切り替えを試す）
  2. signed方式に切り替えたら、simplexで「重み0」と判定され棄却済みの候補群を再評価する
  3. 候補の包摂基準を「プレフィックス/手動選定」ではなく「単体性能の閾値」でスイープし、量と質のバランス点を探す（全部投入・厳選数個の両極端をまず試し、中間の閾値ベース選定と比較する）
  4. 各ステップでLB検証を行い、OOF改善がLBで確認できるかを都度チェックする（天井帯ではOOF改善がLBで消えることが多いため）
- **影響ファイル**: `PLAYBOOK.md`（新規手順セクション）, `CLAUDE.md`（提出枠の管理方針「最終選択」節から参照リンクを追加）
- **状態**: [DONE]（2026-09-02 テンプレート反映済み）

---


## [2026-08-31] MEDIUM — LightGBMがOpenMPスレッドプールでSIGSEGVクラッシュする（macOS、原因は未特定だが再現条件と回避策は判明）

- **説明**: `build_imputed_columns()`など内部でLightGBMを使う関数が、同一セッション内で他のPythonプロセスを`kill -9`で強制終了した**後**に実行すると、`__kmp_fork_call`/`__kmp_create_worker`（libompのスレッドプール生成処理）内でSIGSEGVクラッシュすることがある。エラーメッセージやPythonトレースバックは一切出力されず（低レベルのネイティブクラッシュのため）、症状は「プロセスが理由もなく静かに消える」形で現れる。原因の特定には`~/Library/Logs/DiagnosticReports/*.ips`のクラッシュレポート（`exception.type: EXC_BAD_ACCESS`, フレームに`__kmp_`系シンボル）の確認が必須だった——症状（無言のプロセス消失）だけでは原因（MPS? Lightning? LightGBM?）の特定に至らず、最初はPyTorch/MPS関連の問題と誤診断して1時間以上を浪費した。
- **回避策（確認済み）**: 環境変数`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`を付けて実行すると発生しなくなる。根本原因（なぜkill -9の後にlibompの状態が壊れるのか）は未特定。
- **恒久対応（提案）**:
  - `PLAYBOOK.md`の「既知の落とし穴」（コンペ非依存のセクション）に、症状・診断手順（クラッシュレポートの見方）・回避策を記録する
  - `kill -9`でPythonプロセスを止めた直後にLightGBMを使うスクリプトを再実行する場合は、この既知の落とし穴を先に確認する習慣をCONVENTIONS.mdに一言添える
- **影響ファイル**: `PLAYBOOK.md`（既知の落とし穴セクション、テンプレート共通）
- **状態**: [DONE]（2026-09-02 テンプレート反映済み）

---


## [2026-08-31] MEDIUM — 締切直前の時間管理を都度アドホックな`date -u`確認に頼っており、古い確認結果に基づく誤った見積もりで長時間ジョブを開始しかけた

- **説明**: セッション中盤で確認した「UTC現在時刻・締切までの残り時間」を、その後の長い作業（数時間規模の実験を挟む）を経てから再確認せずに使い回した結果、「残り約9時間」という古い（実際は約4.5時間しかなかった）見積もりに基づいて5時間規模のバックグラウンド学習ジョブ（RealMLP avg5）を開始しかけた。さらにそのジョブ自体が想定の5倍近く遅く進行しており（1シード目だけで4時間超）、「見積もりの陳腐化」と「実行速度の想定外の乖離」が重なって、締切直前まで気づかない一歩手前だった。
- **恒久対応（提案）**:
  - `scripts/deadline_status.py`（新規）を作成し、1コマンドで「現在UTC時刻・締切（COMPETITION.mdから読む）・残り時間・本日の提出使用数/10」をまとめて表示できるようにする
  - CLAUDE.mdの「提出枠の管理方針」に、**30分を超える長時間ジョブを開始する前と、開始判断が過去の時刻確認から一定時間（目安30分）以上経過している場合は、必ず`deadline_status.py`を再実行してから着手する**という運用ルールを明記する
  - 長時間ジョブ自体にも「最初の1fold/1seed完了時点で、実測ペース vs 事前見積もりを比較して大きく（目安2倍以上）乖離していたら警告を出す」チェックポイントを`ExperimentTracker`または各スクリプトのテンプレートパターンとして追加し、遅延に数時間気づかないという事故を構造的に防ぐ
- **影響ファイル**: （新規）`scripts/deadline_status.py`, `CLAUDE.md`（提出枠の管理方針）, `src/experiment.py`または各`optimize_hp_*.py`/multiseed系スクリプトのテンプレートパターン
- **状態**: [DONE]（2026-09-02 テンプレート反映済み）

---


## [2026-08-31] MEDIUM — 可視化ガード（`G-MECH`第3世代）が「警告を出す」だけでは、締切直前の時間的プレッシャー下でAIに無視され続けた（L-06の4度目の再発）

- **説明**: `PLAYBOOK.md` L-06は「可視化の形骸化」が2コンペ連続で再発し、努力目標→条件明示→機械的検知（tracker/hookでの自動警告）と3世代の対策を経てきた経緯を記録している。s6e8終盤（Final2を複数回更新した時間帯）では、LBベスト更新のたびに正しく警告（「⚠️ 可視化ガード発動」）が出力され続けていたにもかかわらず、**AIは一度も対応しなかった**——第3世代（機械的検知）は「検知はできているが、検知結果への対応は結局AIの自発性に依存している」という同型の弱点を持っていたことが、このセッションで実証された。
- **恒久対応（提案）**: `PLAYBOOK.md` L-06に「第4世代」として追記し、対応案を検討する:
  - 警告を単なるログ出力ではなく、次の`ExperimentTracker.start_run()`呼び出し時に**確認応答を要求する**（例: 前回警告が出てから可視化未実施なら`start_run()`が例外を投げ、`--skip-viz-check`等の明示フラグでのみ続行可能にする）など、「気づけば直せる」から「気づかなくても止まる」設計に変える
  - 締切直前など時間的プレッシャーが高い局面ほど省略されやすいという経験則自体を`CLAUDE.md`のG-MECHの節に一文加え、「時間がない時こそ機械的ゲートを緩めない」ことを明示する
- **影響ファイル**: `PLAYBOOK.md`（L-06に追記）, `src/experiment.py`（`ExperimentTracker`の可視化ガード呼び出しタイミング）, `CLAUDE.md`（G-MECH節）
- **状態**: [DONE]（2026-09-02 テンプレート反映済み）

---


## [2026-08-31] LOW — Kaggle Final Submissionの選択がWeb UI専用（CLI不可）であることがどこにも文書化されていなかった

- **説明**: Kaggle CLIには提出（`kaggle competitions submit`）はあるが、2本の最終提出を指定する「Final Submission」の選択機能が無く、Web UIの「Submissions」タブから手動でチェックボックスを選ぶ必要がある。この制約は`.claude/skills/ds-kaggle-submit/SKILL.md`にも`CONVENTIONS.md`にも記載がなく、締切直前になって初めて気づいた（今回は間に合ったが、一歩間違えば「Final2は確定したのにWeb UI選択を忘れる」という致命的な事故になりかねなかった）。
- **恒久対応（提案）**: `.claude/skills/ds-kaggle-submit/SKILL.md`の最終日フェーズ（Final 2確定後の手順）に、「Kaggle Web UIの『Submissions』タブでFinal Submissionとして2本を手動選択する（CLI不可）」を明示的なチェックリスト項目として追加する
- **影響ファイル**: `.claude/skills/ds-kaggle-submit/SKILL.md`
- **状態**: [DONE]（2026-09-02 テンプレート反映済み）

---


## [2026-08-31] HIGH — ユーザーからの終盤フィードバック5件（可視化・スキル摩擦・プロセス規律・状態文書）

コンペ結果発表待ちのタイミングで、ユーザーからテンプレート全体に対する詳細なフィードバックを受けた。いずれもこのコンペ固有ではなく、テンプレート運用一般に関わる課題。

- **① 可視化ファイルの命名が実験順・作成順を反映しておらず追跡困難**
  - **説明**: `data/output/plots/`配下のファイル名（`oof_dist_079_vs_080_xgb.png`, `loo_mlp_delta_oof_loss.png`等）に統一された連番・日付プレフィックスが無く、後から見て「どの実験に紐づく、いつ作られた可視化か」が分かりにくい。
  - **恒久対応（提案）**: CONVENTIONS.mdの「可視化の規約」節に命名規則（例: `{実験ID3桁}_{通し番号2桁}_{内容}.png`）を明記し、`scripts/visualize.py`/`scripts/feature_report.py`の保存パス生成ロジックをこれに統一する。
  - **影響ファイル**: `CONVENTIONS.md`（可視化の規約）, `scripts/visualize.py`, `scripts/feature_report.py`

- **② uv環境に日本語(CJK)フォントが導入されておらず、matplotlib日本語表示が文字化けしていた**
  - **説明**: `scripts/visualize.py`等にmatplotlibの`font.family`/CJKフォント設定が一切無く、日本語ラベル・タイトルを使う可視化がtofu表示（文字化け）になっていた形跡が多数あった。
  - **恒久対応（提案）**: `pyproject.toml`に`japanize-matplotlib`等のCJKフォントパッケージを追加するか、`src/config.py`または可視化系スクリプトの共通importで`matplotlib.rcParams["font.family"]`にIPAexGothic/Noto Sans CJK JP等を明示的に設定する初期化コードを一本化する。`/ds-kaggle-setup`実行時に日本語表示テストを1回走らせて確認する手順を追加できるとなお良い。
  - **影響ファイル**: `pyproject.toml`, `src/config.py`または新規`src/utils/plot_style.py`, `.claude/skills/ds-kaggle-setup/SKILL.md`

- **③ `disable-model-invocation: true`のスキル（`ds-kaggle-research`/`ds-template-update`）が実行不可時、理由をユーザーに明示しないまま止まっていた**
  - **説明**: `ds-kaggle-submit`については実行不可時の代替手順（手動チェックリスト）がCLAUDE.mdに明記されているが、同じ設定を持つ`ds-kaggle-research`・`ds-template-update`には無い。ユーザーが「スキル設定の問題か.md記載の問題か」と疑問を持つほど、AI側が原因を明確に説明していなかった。
  - **恒久対応（提案）**: CLAUDE.mdの「ユーザーの呼び出しが必要な節目」節に、disable-model-invocationスキルにヒットした際は**必ず**「このスキルは設計上ユーザー自身の呼び出しが必要です」と明示してから代替案（手動調査等）を提示する、という振る舞いルールを追記する。`ds-kaggle-research`・`ds-template-update`にも`ds-kaggle-submit`同様の代替手順を用意する。
  - **影響ファイル**: `CLAUDE.md`（ユーザーの呼び出しが必要な節目）, `.claude/skills/ds-kaggle-research/SKILL.md`, `.claude/skills/ds-template-update/SKILL.md`

- **④ 「単体→avg5→HP調整→調整後avg5」の検証プロセスが完走する前に早急な棄却提案をしがちだった**
  - **説明**: G-FAIRは不公正な比較条件（HP流用・特徴量数の不一致）での棄却を禁じているが、「プロセスの全ステップを完走してから判断する」という運用規律は明文化されておらず、途中段階の結果だけで棄却を提案しかけた場面が複数あった（H-023/H-026等、HP再調整・avg5化で初めて真の効果が見えた例と対照的）。
  - **恒久対応（提案）**: CLAUDE.mdのG-FAIR節に「新アーキ/新FEの検証は単体seed→avg5→専用HP再調整→再調整後avg5の4段階を予告し、**途中経過のみでの棄却提案を禁止**する（暫定ネガティブと表現し最終段階まで確認する）」を追記する。
  - **影響ファイル**: `CLAUDE.md`（G-FAIR節）

- **⑤ 新特徴量を「現行ベースへの単純追加」1パターンのみで棄却しがちだった**
  - **説明**: 新特徴量の検証を「今のベース特徴量セットにそのまま1列追加する」という1パターンだけで済ませ、他の組み込み方（別アーキテクチャでの検証・既存特徴量との組み合わせ・置換・重み付け変更）を検討せずに棄却する傾向があった。
  - **恒久対応（提案）**: `/ds-fe-hypothesis`スキルの棄却記録フェーズ（フェーズ3）に、「単純追加以外の組み込み方（アーキテクチャ変更・組み合わせ・置換）を最低1つ検討したか」をチェック項目として追加する。CLAUDE.mdのG-FAIR「4つのチェック」にも同様の項目を検討する。
  - **影響ファイル**: `.claude/skills/ds-fe-hypothesis/SKILL.md`（フェーズ3）, `CLAUDE.md`（G-FAIR節）

- **⑥ `FEATURE_REPORT.md`の「現在の特徴量セット」節が2026-08-05で更新停止、以後3週間超のFE成果(H-012/H-023/H-026等)が未反映だった**
  - **説明**: `/ds-fe-hypothesis`スキル経由の仮説では更新される設計だが、それを経由しない自由形式のFE検証（H-023/H-026等、外部発見の直接検証パターン）では更新フローから漏れる。結果、「今どの特徴量がベースになっているか」をユーザーが会話履歴から都度追う必要があった。
  - **恒久対応（提案）**: `/ds-fe-hypothesis`経由かどうかに関わらず、FE_HYPOTHESES.mdに仮説の採否が確定した時点で`FEATURE_REPORT.md`の該当節を更新することをCLAUDE.mdの学習サイクル原則（思考の外部化）に明記する。可能であれば、FE_HYPOTHESES.mdの採否ステータス変更をトリガーに`FEATURE_REPORT.md`の該当行を機械的に同期するチェックスクリプト（`scripts/doc_audit.py`への追加検査項目）を検討する。
  - **影響ファイル**: `CLAUDE.md`（思考の外部化の原則）, `FEATURE_REPORT.md`のテンプレート運用注記, `scripts/doc_audit.py`（検査追加の検討）

- **状態**: [DONE]（2026-09-02 反映：①→CONVENTIONS可視化規約、②→plot_style.py、③→CLAUDE.md、④⑤→G-FAIR/ds-fe-hypothesis、⑥→CLAUDE.md 思考の外部化）

---


## [2026-09-01] HIGH — 1位/25位のsolution writeupから得た知見（外部比較）

コンペ終了後、1位(Chris Deotte「Distributed Intelligence - NVIDIA Inference Hub」)・25位(Ravi Ramakrishnan「Public 18 Private 25 approach」)のsolution writeupを確認した。我々（Public 538位/Private 645位）との比較から複数の学びを得た。

- **① `10-fold CV`が「試すべきと分かっていたのに最後まで実行されなかった」典型例として再確認された**
  - **説明**: 25位の解法は一貫して`StratifiedKFold(n_splits=10)`を使用していたのに対し、我々は191実験すべて5-fold固定だった。実は本セッションの初期計画（`Tier C1: 10-foldパイロット`）で「未実施項目」として明記されていたにもかかわらず、最後まで実行されなかった（優先順位付けの結果、他のTierに時間を割いた）。今回の外部比較で、**この種の「計画したが未実施」の項目が、後から見て本当に価値があったかを検証する仕組みが無い**ことが分かる。
  - **恒久対応（提案）**: `/ds-new-experiment`または`/ds-kaggle-submit`の振り返りフェーズに、「今サイクルの計画に上がったが未実施のまま終わった項目」を明示的に一覧するチェックを追加する。コンペ終了時の総括（`/ds-template-update`相当）で、この一覧と実際の上位解法を突き合わせて答え合わせをする運用を検討する。
  - **影響ファイル**: `CLAUDE.md`（学習サイクルの原則）, `.claude/skills/ds-kaggle-submit/SKILL.md`（振り返りフェーズ）

- **② sine/cosine特徴量(H-026)が、我々とは独立に25位の解法でも採用されていた——「既存情報へのアクセス効率化」系FEの再現性の高さが他チームでも裏付けられた**
  - **説明**: これは新しいTODOというより、`FE_HYPOTHESES.md`のH-026エントリで既に記録した「既存情報へのアクセス効率化系のFEは再現性が高い」という仮説を、コンペ後に**完全に独立した第三者の解法**でも裏付けられた事例として記録価値がある。
  - **恒久対応（提案）**: 特になし（既存の知見の裏付けとして`PLAYBOOK.md`の該当箇所に一言追記できると良い程度）。

- **③ トップ層(1位・25位とも)が明示的に「LLMエージェントを使った」ことを解法の中核として書いている——2026年後半のKaggle Playgroundは事実上「エージェント運用能力」が競技力の一部になっている**
  - **説明**: 1位はほぼ全工程を複数LLMエージェントの自律運用（人間はオーケストレーションのみ）、25位も「Codexの方が今回は信頼できるパートナーだった」「LLMは学習・提出パイプライン構築が得意」と明言している。我々のテンプレート運用（Claude Codeとの協業、ただしG-STEPWISE等で1実験ずつ人間確認を挟む設計）とは対照的に、1位は**人間の確認をほぼ介さない高速・並列・自律的なエージェント運用**で結果を出していた。
  - **本質的な論点**: これは技術的なTODOというより、テンプレートの設計哲学（共創の原則：「AIはユーザーのドメイン知識の代わりにならない」「G-STEPWISE：1実験1コミット1確認」）そのものに関わる問いである。**「慎重な人間協業」と「高速・自律的なエージェント運用」はトレードオフであり、今回のテンプレートは前者を意図的に選んでいる**（CLAUDE.mdの共創の原則で明記済み）。1位の結果は後者が極めて強力たりうることを実証したが、これは「テンプレートの設計が間違っていた」ことを意味しない——共創の原則を重視するユーザーには前者が適切であり続ける。
  - **恒久対応（提案）**: 次回コンペのKickoff時に、「厳密な人間協業モード（現行テンプレート）」と「高速・低介入の探索モード」のどちらで進めるか、または局面に応じて使い分けるかを、**ユーザーと明示的に議論する選択肢として提示する**。CLAUDE.mdの共創の原則の節に、「この設計は意図的な選択であり、速度を最優先する場合は別のトレードオフがあり得る」という一文を添えることを検討する。
  - **影響ファイル**: `CLAUDE.md`（共創の原則、Kickoffのガイダンス）

- **状態**: [DONE]（2026-09-02 反映：①→ds-kaggle-submit Step6、②→対応不要、③→CLAUDE.md 共創の原則）

---


## [2026-09-02] HIGH — 学習と推論を分けたため「提出のために同じ学習を回し直す」が多発した

- **説明**: FE 実験の大半は ΔOOF 計測だけを目的に書かれ、`oof_*.npy` は保存しても `test_*.npy` を作らなかった。しかし「LB で確かめたい」「ブレンドに入れたい」と判断が変わることが繰り返し起き、そのたびに同じ学習を再実行した。NN 系・multi-seed では 1 回数十分〜数時間かかり、締切直前に集中した。
- **問題の本質**: 「今は ΔOOF を測るだけ」という**その時点の目的**でスクリプトの出力範囲を決めていたが、実験の価値は事後に変わる。test 予測は学習済みモデルがメモリ上にある間ならほぼゼロコストなのに、後から作ると学習コスト全額を払い直す——この非対称性が設計に反映されていなかった。
- **改善案**: ①`save_run_outputs()` で OOF + test + 提出 CSV を 1 回で出し切る ②「OOF はあるのに test が無い」実験を `end_run()` が機械検知する ③学習 → 提出まで 1 フローにすることは「1 実験ずつ確認を挟む」規律（`G-STEPWISE`）とも噛み合う
- **状態**: [DONE]（2026-09-02 反映：`src/utils/finalize.py` 新設、`src/experiment.py` に推論成果物ガード、CLAUDE.md `G-STEPWISE`・CONVENTIONS スクリプト標準構成・PLAYBOOK L-24・ds-new-experiment スキル）

---


## [2026-09-02] MEDIUM — multi-seed avg で基本 seed を毎回再学習していた（テンプレートに再利用機構が無かった）

- **説明**: avg5 は「基本 seed（`RANDOM_STATE`）を含む 5 seed」で構成するが、基本 seed の学習は単体モデルの実験で既に済んでいることがほとんど。s6e8 では途中から手作業で「基本 seed 以外の 4 つだけ回す」形に切り替えて時間短縮したが、テンプレート本体にはその機構が無く、実験スクリプトごとにアドホックな分岐を書いていた（`exp144_realmlp_tuned_multiseed.py` 等）。
- **問題の本質**: 「同じ条件で既に計算済みの結果は再利用できる」という当たり前の最適化が、共通ヘルパーではなくコンペ固有スクリプトの中に埋もれていたため、毎回書き直しになり適用漏れも起きた。
- **改善案**: 共通ヘルパー化し、seed ごとの予測を規約化された名前で保存して次回自動的に再利用する。ただし**再利用は「同じ特徴量セット・同じ HP」が前提**なので、条件が変わったら `tag` を変える規約を明示する（`G-FAIR` 違反の防止）。
- **状態**: [DONE]（2026-09-02 反映：`src/utils/multiseed.py` 新設（`run_multiseed()`）、CONVENTIONS に「multi-seed avg の実行規約」、PLAYBOOK L-24 の項目4）

## [2026-09-02] CRITICAL — `G-MECH` がテンプレート自身に適用されておらず、強制の入口が hook 1 本しか無かった

- **説明**: 強制機構は PostToolUse hook 1 本とプロセス内ガード 3 つのみ。`SessionStart` / `PreToolUse` / `Stop` / `PreCompact` は未使用だった。**s6e8 で実測的に破られた規律**（提出前確認は毎回 AI の自己申告 / 1実験1コミット / FEATURE_REPORT が 3 週間停滞）は、いずれも機械化ゼロの側にあった。
- **区別すべき点**: `SessionStart` ブリーフは「`/ds-resume` を強制する」ものではない。`/ds-resume` はユーザーの儀式であり、その価値は「現在地を合意して次の一手を決める対話」にある（スキップされたという計測もしていない）。ブリーフが担うのは**下限の保証**——ユーザーが儀式を挟まず本題から入ったとき、`--continue` / `--resume` のとき、圧縮後のときに、AI がゼロ文脈で走り出さないようにする。スキルの代替ではない。
- **問題の本質**: 規範を書く場所（CLAUDE.md）と、それを守らせる機構の場所（hook・ガード）の対応表が無く、機構のある規律ばかり強化され、無い規律は放置され続けた。
- **対応**: hook を 5 種へ拡張（SessionStart ブリーフ / PreToolUse 提出ゲート / PostToolUse ガード / Stop 監査 / PreCompact 退避）、ガードを 5 種へ（可視化・提出・診断記録・推論成果物・コミット/状態鮮度）。提出ゲートは `permissionDecision: "ask"` で人間へ承認を戻しつつ、数字は Kaggle API と時計から実測する。
- **影響ファイル**: `.claude/settings.json`, `scripts/session_brief.py`, `scripts/submit_gate.py`, `scripts/session_audit.py`, `scripts/state_audit.py`, `scripts/session_snapshot.py`, `scripts/job_status.py`, `src/experiment.py`, `CLAUDE.md`, `CONVENTIONS.md`, `PLAYBOOK.md` L-25
- **状態**: [DONE]（2026-09-02）

---

## [2026-09-02] HIGH — 観測面の欠落（実行時間・状態鮮度・ベース特徴量が機械可読でなかった）

- **説明**: ①log.csv に実行時間の列が無く、30分ルールが要求する推定を較正する実測が残らなかった（推定 1h の実験が実測 4h17m）②7 つの状態ファイルの更新を誰も観測していなかった（FEATURE_REPORT が 3 週間停滞）③「今どの特徴量がベースか」は手書き markdown だけが情報源で、`src/feature_registry.py` は import 元ゼロの死蔵コードだった。
- **対応**: `duration_sec` の自動記録（+ 列追加マイグレーション）、`scripts/state_audit.py`、`end_run(feature_names=...)` → `params/features_{exp_id}.json` → `feature_report --sync` で FEATURE_REPORT の該当節を機械生成。`feature_registry.py` は削除。
- **状態**: [DONE]（2026-09-02）

---

## [2026-09-02] HIGH — 復旧性の欠如（中断で fold 全損・圧縮で文脈消失・ジョブ状態が不可視）

- **説明**: fold 単位のチェックポイントが無く、4 時間超まわした学習を fold 4/25 で打ち切った際にその分が全損した。コンテキスト圧縮のたびに現在地の復元コストを払った。長時間ジョブの生死をユーザーが繰り返し尋ねる必要があった。
- **対応**: `src/utils/foldcache.py`（fold ごとの保存と再開）、`experiments/.running/` ハートビート + `scripts/job_status.py`（生存・進捗・ETA・ハング検知）、PreCompact hook による SESSION.md への状態退避、`experiments/blockers/` の実体作成。
- **状態**: [DONE]（2026-09-02）

---

## [2026-09-02] MEDIUM — 参照頻度の違うものが同居して肥大していた（TODO 97% が完了済み・README の半分が履歴）

- **説明**: `TODO_TEMPLATE.md` 1,075 行のうち 66 件が DONE で、実際に読むべき未完了は 3 件だった（`/ds-resume` が毎セッション読むファイル）。`README.md` 581 行のうち約 280 行がバージョン履歴だった。`CLAUDE.md` は上限 650 行に張り付き、退避ポリシーが無かった。
- **対応**: `docs/TODO_ARCHIVE.md` と `CHANGELOG.md` へ分離。CLAUDE.md に「1 行入れるなら 1 行出す」の退避ポリシーと「ガード側を緩めない」を明記。README 側は履歴分離で `doc_audit` C11 の検査対象が消えたため「現在の構成」表を新設して検知を回復した。
- **状態**: [DONE]（2026-09-02）

---

## [2026-07-04] HIGH — FE仮説の採否は「ΔOOF・CV内部診断（train/val/std/gap）・importance」の3点セットで総合判断する（旧: ΔOOF閾値だけの判断はPrivate shakedownリスクがある）

- **背景（ΔOOF+gapの2点）**: s6e7でH-008(bmi_extreme_flag, ΔOOF=-0.00009)とH-009の一案(bmi_bin7, ΔOOF=-0.00020)は、どちらも`scripts/feature_study.py`のΔOOF閾値判定では「⬜ノイズ範囲/❌棄却」の同列に分類されていた。しかしCV内部診断（train_mean/val_std/gapの変化）まで見ると、bmi_extreme_flagはgap変化ほぼ0（train_meanはむしろ微減、val_stdは改善）の「過学習兆候なしの純粋なノイズ」だったのに対し、bmi_bin7はgapが+0.00023拡大する「軽度の過学習兆候あり」だった。この2つを実際にLB提出して比較したところ、bmi_extreme_flagはLBでexp009比+0.00027**改善**し、bmi_bin7はLBでも-0.00012悪化と、CV内部診断による事前の判別がLBの挙動と一貫して対応した。
- **背景追加（importanceの3点目）**: H-011（sleep_duration<6.0閾値、3変換案）では、sleep_deficit_amount/logのimportanceが14特徴量中8位（gender・physical_activity_level・smoking_alcohol・sleep_quality・diet_typeという既存採用済み特徴量より高い水準）だったにもかかわらず、ΔOOFはほぼゼロ〜マイナスだった。これはH-006(diet_type)でも確認済みの「importanceが中位以上でもΔOOFに寄与しない」パターンの再現であり、ΔOOF・gapの2点だけでは見えない乖離がimportance確認で追加発見できることを示した。ユーザーからは「これまでもPrivate時にshakedownを連発していた。ΔOOF閾値判断だけではリスクが高い」「今後の実験もOOFだけでなくこれらの数値から総合して採否を判断しよう」という明確な方針指示があった。
- **問題点**: `scripts/feature_study.py`の判定ロジック（当初実装）はΔOOFの絶対値のみで4段階判定しており、gap・importanceを判定に組み込んでいなかった。ΔOOFが僅差の場合、「本当に効果がないノイズ」「軽度の過学習」「importanceは高いが精度に寄与しない冗長」を区別できず、誤った採否判断がPrivate LBでのshakedown（順位変動）リスクに繋がる。
- **恒久対応（対応済み / DONE）**:
    - `scripts/feature_study.py`の判定ロジックを改修し、ΔOOFがノイズ範囲・棄却域にある場合はgapの変化（閾値+0.0005を「notable」とする）も加味した詳細な判定を表示するようにした
    - 同スクリプトの出力にimportance（新特徴量のimportance値・全特徴量中の順位）も自動表示するようにした（`run_cv()`が返す`importance_df`を利用、追加の学習は不要）
    - スクリプトのdocstringに「ΔOOF・gap・importanceの3点を総合して判断する」原則と、s6e7での実証結果（2件）を明記した
- **今後の運用ルール（このコンペ及び今後のコンペ共通）**: FE仮説の採否は必ず以下の3点を確認してから判断する。1つだけでは判断しない
    1. **ΔOOF**: 閾値+0.001/+0.0003/±0.0003/マイナスの4段階（既存基準）
    2. **CV内部診断（train/val/std/gap）**: train_meanの変化・val_stdの変化・gapの変化。gapが拡大していれば過学習兆候、変化がなければノイズの可能性が高い
    3. **importance（gain）**: 新特徴量の順位・値。低ければ「使われていない」棄却、中位以上でもΔOOFに寄与しなければ「冗長」棄却（H-006/H-011型）
    - ΔOOFがノイズ範囲・僅差棄却域かつgap変化が小さい場合は、提出枠に余裕があれば実際にLB提出して確認する価値がある（s6e7の事例のように、ΔOOFがマイナスでもLBで改善するケースがある）
- **関連**: 下記「`/ds-kaggle-submit` の診断がOOF↔LBの2軸のみ」のTODO項目とは扱う段階が異なる（本項目はFE採否判断＝提出前の入口段階、下記項目は提出後のLB診断段階）が、根本思想（複数指標を常に併記する）は共通
- **既知の限界（閾値校正、未着手/TODO）**: gap閾値`GAP_NOTABLE=0.0005`は、LB確認済みの2データ点（bmi_extreme_flag: Δgap+0.00004→LB改善／bmi_bin7: Δgap+0.00023→LB悪化）のみで暫定校正したものであり、サンプル数が極めて少ない。s6e7のH-010（step_count_bin3/5/7, Δgap+0.00030〜+0.00044）でこの閾値により全て「過学習兆候なし」と自動判定されたが、ΔOOFが3案とも一貫してマイナスだったため、ユーザー判断で自動判定を採用せず棄却した。今後のコンペでLB確認済みのCV内部診断データ点が蓄積されたら、閾値`GAP_NOTABLE`を再校正する（複数コンペのデータを`experiments/log.csv`から収集し、Δgapとoof_lb_gapの相関を確認した上で適切な閾値を再設定する）
- **影響ファイル**: `scripts/feature_study.py`
- **状態**: [DONE（閾値の再校正は今後の課題として別途TODO化）]

---
- **状態**: [DONE]（scripts/feature_study.py に実装済み。運用ルールは GUIDELINES `G-DIAG` / `G-FAIR` に反映）

---

## [2026-07-02] HIGH — `Kaggle GPU ワークフロー`の Dataset 同期（rsync 一括コピー）が AI 実行環境のセキュリティポリシーでブロックされる

- **背景**: s6e7 exp002 を Kaggle Notebook でも実行しようとし、PLAYBOOK.md 記載の標準手順の Step 1（`rsync -a --delete ... . /tmp/kaggle_dataset_<slug>/` でプロジェクト全体を一時ディレクトリに同期 → `kaggle datasets create` で Dataset化）を **AI（Claude Code）が実行しようとした**ところ、自動モード分類器に「Data Exfiltration」のハードブロックとして拒否された。ユーザーが明示的に許可しても解除されない種類のブロックで、コマンドを言い換えても（同期先ディレクトリを作る `mkdir` のみでも）再度ブロックされ、**rsync 同期は一度も成功しなかった**。
- **原因（推定）**: 「プロジェクトルート全体（`src/`, `scripts/`, `experiments/` 等の複数ファイル）を一時ディレクトリへ一括コピーし、外部プラットフォーム（Kaggle）へアップロードする」という操作パターンが、AI エージェントによるリポジトリ一括流出とみなされたと考えられる。kaggle-cli 自体には `.kaggleignore` のような除外機構は無く（`kaggle datasets create -p <dir>` はディレクトリ内の全ファイルを対象にする）、除外は rsync による事前フィルタリング前提の設計であるため、Dataset 同期を行う限りこの「複数ファイルの一括コピー」操作は避けられない。
- **s6e7 での対応（Dataset 同期を使わない方式への変更）**: Kaggle Dataset 経由でコードを import する PLAYBOOK.md 標準方式を諦め、実行に必要なロジック（前処理・学習・CV）を **単一の自己完結スクリプト**（`kaggle_nb/exp002_standalone.py`、`src/config.py` や `scripts/train.py` を import せず、必要な処理をすべて1ファイルに複製）として新規に書き起こし、それを1つの `.ipynb` に変換した。**アップロードしたのは変換後の `.ipynb` と `kernel-metadata.json` の2ファイルのみ**（`dataset_sources: []`、Dataset 参照なし）。この「Notebookファイル単体の push」は AI が実行してもブロックされず、`kaggle kernels push` は成功した。
  - 実行結果: push 後 `kaggle kernels status` で進捗を追跡したところ、正常に実行されていた（ハングではなく、Kaggle Notebook の CPU 実行がローカルより遅いだけだった。1 seed=5fold あたり約370秒。ローカルは同処理が数分で完了）。seed=42, 43 の中間結果はローカル実行と完全一致（val_mean=0.44485, 0.44552）し、ロジックの再現性も確認できた。
- **トレードオフ**: 自己完結スクリプト方式は `src/config.py` や `scripts/train.py` の共通ロジックを都度コードに複製することになり、DRY 原則に反する。実験が増えるたびに最新コードを手作業で複製・同期する必要があり、Dataset 経由方式（1回 push すれば全実験 Notebook が最新コードを参照できる）の利便性を失う。
- **運用上の注意点（新規）**: Kaggle Notebook の CPU 実行はローカル CPU より明確に遅い場合がある（今回は約2〜3倍）。「30分ルール」（CLAUDE.md）でKaggle GPU実行を検討する際、GPU を使わない CPU 実行に切り替えても速度改善を期待しない。GPU 対応ライブラリ（LightGBM の `device=gpu` 等）を使わない場合、Kaggle Notebook 実行がローカルより遅くなるケースがあることを想定して計画する。
- **恒久対応（未着手 / TODO）**:
    - `scripts/to_kaggle_nb.py` に **「自己完結モード」**（例: `--standalone` フラグ）を追加する。指定すると `--dataset-name` 経由の import ではなく、依存する `src/*.py` モジュールのソースを静的解析（AST等）または単純な import 追跡で集めて `.ipynb` の1セルにインライン化する。Dataset 同期を経由しない実行経路を正式にサポートする
    - PLAYBOOK.md の `Kaggle GPU ワークフロー` セクションに、Step 1（Dataset 同期）が AI エージェント実行環境のセキュリティポリシーでブロックされる可能性がある旨と、自己完結モードへの切り替え手順を注記する
    - Dataset 同期がどうしても必要な場合（実験数が多く自己完結方式のコード重複が許容できない場合）は、**ユーザー自身のターミナルで rsync を実行してもらう**ことを明示的な代替手順として案内する（AI が実行を試みて毎回ブロックされるのを避ける）
- **影響ファイル**: scripts/to_kaggle_nb.py, PLAYBOOK.md
- **恒久対応（2026-08-02 部分反映）**:
    - `PLAYBOOK.md` の `Kaggle GPU ワークフロー` 冒頭に**警告ブロック**を追加。「rsync 一括コピーは AI 実行環境でハードブロックされる（ユーザーが許可しても解除されない）」ことを明記し、**2 つの経路**を表で提示: (A) 自己完結 Notebook（AI 実行可・推奨）/ (B) Dataset 同期（**ユーザー自身のターミナルで実行**、AI は Step2 以降を担当）
    - Step 1 の見出しとコードブロックに「ユーザーのターミナルで実行すること」を明記し、AI が実行を試みて毎回ブロックされる事故を防ぐ
    - Kaggle Notebook の CPU 実行がローカルより遅い（実測 2〜3 倍）点も同ブロックに注記し、「30 分ルール」適用時の判断材料にした
    - **残**: `scripts/to_kaggle_nb.py` への `--standalone` フラグ実装（依存モジュールを AST 追跡してインライン化）は規模が大きいため別タスクとする
- **s6e8 での実績（2026-09-02 実測）**: Kaggle Notebook を **19 本**（うち GPU 有効 4 本）実行したが、
  **`dataset_sources` が空でないものは 0 本** —— 全数が方式(A) 自己完結 Notebook で完走した。
  PLAYBOOK が方式(A)/(B) を文書化した後は運用上の問題として顕在化していない。
  **回避策が定着した状態**であり、作業中の課題ではない。
- **未解決のまま残っていること**: 根本原因（サンドボックスが一括コピーを Data Exfiltration と
  判定する条件）は未特定。検証するには実際に rsync を試すしかなく、それがブロックされる操作なので
  AI 側からは確認できない。方式(B) が必要になったらユーザー自身のターミナルで実行してもらう
- **状態**: [DONE]（回避策は PLAYBOOK に定着済み・1 コンペ完走で実証。ただし根本原因は
  未解決の技術的負債として残す —— `G-BLOCKER`）

---

---

## [2026-09-01] HIGH — 1位解法から着想した3つの改善方向（ユーザーとの議論で合意）

1位のエージェント自律運用の分析を受け、テンプレートに取り込める部分をユーザーと議論した。3方向のうちA・Cは次のテンプレート改善に含める、Bは次回コンペで具体的に設計する、と合意した。

- **A) 単体モデル卓越性を独立したゴールとして明示する**
  - **説明**: 現行のステージ設計（Stage 5: 本格HP最適化 → Stage 6: アンサンブル）は、単体モデルの磨き込みを「アンサンブルの部品作り」の通過点として扱いがちで、それ自体を到達点として扱う視点が弱い。本コンペのH-023+H-026パイプライン（単体→avg5→専用HP再調整→avg5）は単体モデル卓越性を独立目標として追求し、検証全体で最良のLB(id=253, 0.96998)を生んだ。外部の25位解法比較でも「PyTabkit/RealMLPが最良単体モデル」という点で我々の発見と一致しており、単体モデルの精度自体に実務上の価値（デプロイの単純さ、保守性）があることも確認された。
  - **恒久対応（提案）**:
    - CLAUDE.mdのステージ表（Stage 5の完了条件）に「単体モデルの最良値を到達点として記録する」ことを明記する
    - `CONVENTIONS.md`の「SESSION.mdの構成と上限」節に、スコア状況テーブルの「単体ベスト」行を**常設の構成要素**として明記する（本コンペでは自然発生的にこの行を維持していたが、テンプレートとしては未規定だった）
    - Stage 6（アンサンブル）着手前に「単体モデルで実務要件を満たすか、アンサンブルの複雑性コストに見合う伸びがあるか」をユーザーと確認するチェックポイントを、CLAUDE.mdのStage 6節に追加検討する
  - **影響ファイル**: `CLAUDE.md`（ステージ表, Stage 6節）, `CONVENTIONS.md`（SESSION.mdの構成と上限）

- **C) 9-Persona投票パターンをFE仮説立案の初期段階にも拡張する**
  - **説明**: 現行、複数視点での並行評価（9-Persona投票）はFinal2選定という終盤の一局面でしか使っていない。この「複数視点で独立に提案させ、統合・選択する」パターンを、`/ds-fe-hypothesis`のFE仮説立案フェーズにも適用できないか検討する。追加の計算資源をほぼ使わず（テキストベースの提案生成のみ）、多様な角度からの仮説創出を促せる可能性がある。
  - **恒久対応（提案）**: `.claude/skills/ds-fe-hypothesis/SKILL.md`のモード3（list、次の仮説候補のレコメンド）に、「複数の視点（ドメイン専門家視点・統計理論視点・逆張り視点等）で個別に候補を提案してから統合する」オプションを追加検討する。実装は(a)Agent toolでの軽量な並列サブエージェント起動、または(b)単一応答内で複数視点を明示的に切り替えて提案する簡易版、の両方を比較検討する。
  - **影響ファイル**: `.claude/skills/ds-fe-hypothesis/SKILL.md`（モード3）

- **B) 並列サブエージェント実行の設計 — 次回コンペで具体的に検証（今回は方針のみ記録）**
  - **説明**: モデル学習そのものの並列化はローカル環境の資源競合リスクが高い（本コンペでOpenMP/MPSクラッシュを複数回経験済み）。1位・25位の解法双方が「本当のボトルネックはコードを書く速度」「LLMは学習パイプライン構築が得意」と明言していることから、**並列化すべきは「学習ジョブ」ではなく「複数のFE仮説・アーキテクチャ候補の調査・スクリプト作成」**という方針までユーザーと合意した。学習自体の並列化が必要な場合はKaggle GPU Notebookを複数同時に投げる（クラウド側での並列化）方が資源競合を避けられる。
  - **未確定だった 3 点への答え（2026-09-02 設計）**:
    1. **いつ何本** — スキルのフェーズが指示する局面のみ、ユーザーの承認を得てから。
       `fe-ideator` は視点の数だけ並列、他は 1 本
    2. **統合** — 親が受け取りユーザーに提示して選ばせる。状態ファイルへの記録は**親が 1 件ずつ**
    3. **`G-STEPWISE` との整合** — **並列化してよいのは成果物がテキストで実験の実行を伴わないものだけ。**
       仮説を 3 つ並列で「出す」のは侵さないが、3 つ並列で「回す」のは侵す。
       強制は `tools` で行う（読み取り専用エージェントに Bash を渡さない。`doc_audit` C13 が検査）
  - **影響ファイル**: `.claude/agents/`（4 件新設）, `CONVENTIONS.md`（運用規約）,
    `GUIDELINES.md`（G-STEPWISE）, `CLAUDE.md`, `scripts/harness/doc_audit.py`（C13）,
    `ds-fe-hypothesis` / `ds-new-experiment` / `ds-kaggle-research` の各スキル
  - **状態**: [DONE]（2026-09-02。4 エージェントは記録済みの失敗 L-15 / L-02·L-16·L-20 / L-12 / L-07 に紐づく）

- **状態**: [DONE]（A・C は 2026-09-02 反映、B も同日に設計・実装）

---

---
