# テンプレート改善タスク

コンペ作業中に発見したテンプレート改善項目を記録するファイルです。
`/template-update` スキルで追記されます。

## フォーマット

```markdown
## [YYYY-MM-DD] <優先度> — <タイトル>
- **説明**: ...
- **影響ファイル**: ...
- **状態**: [TODO / IN PROGRESS / DONE]
```

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
- **状態**: [TODO]

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
