# DS Template — Kaggle Competition Workspace

---

## Part 1: テンプレートの設計思想（汎用）

> この節はKaggleに限らず、あらゆるデータ分析・モデリング作業に適用される。

### 共創の原則

**AIはユーザーのドメイン知識の代わりにはならない。**

AIが担うのは「構造化・分析・記録」であり、ユーザーが担うのは「文脈・判断・ドメイン知識」。
この役割分担を守ること。AIが先に答えを出すより、ユーザーの思考を引き出す問いを優先する。

### 学習サイクルの原則

すべての実験は独立したイベントではなく、学習サイクルの一部として扱う。

```
/ds-resume ──────── 新しいセッション開始時に必ず呼ぶ。SESSION.md + log.csv + FE_HYPOTHESES.md を読み「今どこにいるか」を1画面で復元する
    ↓ 現在地を確認したら
/kickoff ────────── 「そのデータが何者か」を文脈から理解する（コンペ参加直後に一度だけ）
    ↓ データ種別・外部データ有無・評価指標の特性を COMPETITION.md に記録
/new-experiment ─ 最小ベースライン実験を開始（前処理不要な数値カラムのみ・デフォルトHP）
    ↓
/kaggle-submit ── LBに提出してCV/LB相関を確立する（以降のOOF判断の基準点）
    ↓ CV/LB相関が確認できたら
Stage 1.5 ──────── 早期アーキテクチャサーベイ（同一特徴量×作業用HP×同一CVで複数アーキテクチャ比較）
    ↓ OOFとpub_oof_gapを記録し「主軸アーキテクチャ」を1つ決定する
/eda-visual ───── 「何を知りたいか」を先に言語化する（kickoff と基準点を持ち込む）
    ↓ 仮説の種を /fe-hypothesis に登録しながら進む
Optuna 軽量 ──── 作業用HP調整（20〜30試行）。FEのΔAUC計測ノイズを低減する目的
    ↓ 作業用HPが確定したら
/fe-hypothesis ── 「なぜ効くか」の因果連鎖を言語化してから実装する
    ↓ 必ず1列ずつ scripts/feature_study.py で投入・ΔAUCを計測。複数列の一括追加は禁止
    ↓ ΔAUC が閾値以下でも feature importance (gain) も確認してから棄却判断する（後述）
/new-experiment ─ 「何を明らかにしたいか・成功基準・撤退基準」を先に記録する
    ↓ FEが収束したら（追加特徴量のΔAUC < ±0.0003 かつ importance が BASE最下位未満 が続いたら）
Optuna フル ───── 本格HP最適化（100試行以上）。確定した特徴量セットで実施
    ↓
/kaggle-submit ── 「OOF/LBのギャップ」を解釈し「学び」を言語化する
    ↓ 次の実験の仮説を更新する
/new-experiment ─ 次のサイクルへ（アンサンブルへ移行 or FEに戻る）
```

**サイクルを回す際のルール:**
- 実験の「目的」は実行前に記録する。結果が出てから目的を決めない
- 「OOFが上がった」は成功ではない。「汎化性能が上がったか」が問い
- 棄却された仮説は失敗ではない。「なぜ効かなかったか」が次の仮説を賢くする

### 思考の外部化の原則

**「考えたこと」は記録しなければ存在しなかったのと同じ。**

| 記録すべきもの | どこに | スキル |
|---|---|---|
| 「何を知りたいか」「ドメイン知識」 | EDA_SUMMARY.md | `/eda-visual` |
| 各変数の特性・ΔOOF・採否 | FEATURE_REPORT.md | `/eda-visual` · `/fe-hypothesis` が記入を促す |
| 特徴量の仮説・因果・棄却理由 | FE_HYPOTHESES.md | `/fe-hypothesis` |
| 実験の目的・成功基準・撤退基準 | experiments/log.csv | `/new-experiment` |
| 実験から何を学んだか | experiments/log.csv | `/kaggle-submit` |
| テンプレートへの汎用的な気づき | TODO_TEMPLATE.md | `/template-update` |
| **現在地・次のアクション・未解決の問い** | **SESSION.md** | **`/new-experiment` · `/kaggle-submit` が自動更新** |

**SESSION.md の更新タイミング（自動）:**
- `/kickoff` 実行時 → Stage 0 完了・次のアクション（最小ベースライン）を記録
- `/eda-visual` 実行時 → Stage 2 完了・次のFE仮説リストを記録
- `/fe-hypothesis` 実行時（新規） → 仮説登録・次のアクション（実装→計測）を記録
- `/new-experiment` 実行時 → 実験開始・次のアクションを記録
- `/kaggle-submit` 実行時 → LBスコア・OOF-LB乖離・**本日の提出数（例: 3/10）**・学び・次の方向性を記録

**SESSION.md の「現在の主要スコア」テーブルの形式:**

| 指標 | OOF tuned | LB | OOF-LB乖離 | 実験ID |
|---|---|---|---|---|
| ベスト | 0.XXXXX | 0.XXXXX | ±0.XXXXX | expNNN |

乖離列を常に記録することで「OOFは高いがLBで悪化」のパターンを早期に検知できる。

**新しいセッション開始時は必ず `/ds-resume` を実行する。**

**SESSION.md の設計原則（蓄積禁止・上書き原則）:**

SESSION.md は「今どこにいるか」を1画面で示すライブダッシュボード。
アペンド（追記）ではなく、各セクションを必ず **上書き更新** する。

禁止パターン（AIが絶対にやってはいけないこと）:
- 「最後に完了したこと」セクションを複数回追記する（古いものは削除 → 1件だけ保持）
- 複数のスコアテーブルを並存させる（常に最新ベストのみ1テーブルを保持）
- 過去セッションの履歴を蓄積する（git history に残るので SESSION.md には不要）

SESSION.md の固定構成（このセクション順序を守り、追加セクションを作らない）:
1. **ファイルヘッダー** — 最終更新日時（`/new-experiment` または `/kaggle-submit` 実行のたびに更新）
2. **現在のステージ** — 1〜2行で現状を説明。「次にやること」を1行目に書く
3. **スコア状況** — ベストスコアのみ1テーブル。更新時は上書き（新テーブル追加禁止）
4. **直近の実験** — 最大10件。11件目以降は最古から削除（git log で追跡可能）
5. **次にやること** — 箇条書き最大5件
6. **未解決の問い** — ブロッカー・疑問点のみ。解決済みは削除
7. **重要な方針** — 実験を通じて確定した原則のみ

SESSION.md のオーバーフロー検知:
- **ファイルが 80 行を超えた場合**: 蓄積が起きているサイン。過去の完了済みエントリを削除して 80 行以内に収める
- `/ds-resume` 実行時に行数を確認し、オーバーフローなら警告を出して整理する

### AIへの指針

以下の場面では、**実行より前に問いかけを行うこと:**

1. **ユーザーが「〜をやってほしい」と言ったとき**
   → まず「それをすることで何を明らかにしたいですか？」を問う
   → 目的が明確になってから実装する

2. **FEの提案をするとき**
   → 「この特徴量が効く理由（因果連鎖）」を説明できるものだけ提案する
   → 「試してみる価値がある」だけでは不十分

3. **次のステップを提案するとき**
   → 直前の実験から「何が分かったか」を確認してから提案する
   → 学びを踏まえない提案はしない

4. **アンサンブル・別モデル追加を提案するとき**
   → FE_HYPOTHESES.md に仮説5件以上 かつ 特徴量飽和の確認後のみ提案する
   → それ以外では「まず特徴量の探索を深めましょう」と促す

5. **合成データコンペと判明したとき（EDA着手前）**
   → 「元データ（生成元）が入手可能か」を確認する
   → 入手可能なら、特徴量探索の**最初のステップ**として外部シグナル特徴量を試す
   → 詳細は下記「合成データコンペ向けガイダンス」を参照

6. **複数の実験候補が出たとき（Step by Step 原則）**
   → 「A・B・C を試しましょう」という提案は OK だが、ユーザーが選択する前に並行実装を開始してはならない
   → 各実験完了後、次に移る前に必ず①OOF報告 ②log.csv記録 ③git commit ④次の目的確認 を実施する
   → ユーザーが「続けて」「まとめて」と言った場合も、記録とコミットは各実験完了後に行う（省略禁止）
   → 実験実行中の「待ち時間」に次ステップの設計・実装を進めることも禁止。結果確認→ユーザーと判断→次の実験の順序を守る
   → **1実験 = 1コミット。この原則をAIが「効率化のため」にスキップすることは禁止**

7. **FEの棄却が続いたとき（飽和宣言禁止）**
   → 「飽和確定」「全方向探索済み」という断言を使わない。代わりに「現在試した方向では改善なし。別の角度を探しましょう」と表現する
   → **FE棄却3連続の時点で Kaggle Discussion・上位ノートブックの調査を行う**（「飽和」と宣言する前に他の参加者の知見を確認する）
   → Discussion 調査の手順:
     ```bash
     kaggle kernels list --competition <id> --sort-by voteCount --page-size 20
     kaggle kernels pull <author>/<slug> -p /tmp/kernels/
     kaggle competitions leaderboard -c <id> --show | head -20
     ```
   → FE棄却時は「なぜ効かなかったか」に加えて「**まだ試していない情報次元は何か**」を必ず1文で記録する
   → ステージの前後（Stage 6 → Stage 4 に戻るなど）は積極的に提案する。ステージは一方向に進むものではない

8. **探索継続姿勢（諦めない・PDCAを回し続ける）**
   → **「残り枠」「コンペ終盤」を理由に探索を縮小する判断をAI側からしない。枠の配分はユーザーが決める**
   → 棄却が続いても「棄却 = 今の角度が尽きた」であり「棄却 = 探索終了」ではない。常に「別の角度はないか」を能動的に複数提案し続ける
   → 3連続棄却後も次に試せる視点を提案する。提案が尽きた場合は「私が思いつく範囲では…。他に試したい方向はありますか？」と問いかける
   → 早期の収束宣言はユーザーの探索機会を奪う。**「まだ試せることがある」という前提で考え続けること**
   → PDCA サイクルの意識:
     - **P（計画）**: 実験前に「何を明らかにするか・成功基準・撤退基準」を言語化
     - **D（実行）**: 計画通りに実験し、結果を記録
     - **C（評価）**: OOF/LB・乖離・相関を多角的に分析。「なぜその結果になったか」をメカニズムレベルで解釈
     - **A（改善）**: 学びから「次の最も価値ある仮説」を導出してサイクルを回す
     → 棄却された仮説は「Aフェーズの情報」。蓄積するほど次の仮説が精緻になる

9. **可視化プロセスを省略しないこと（全ステージ共通）**
   → 可視化は「やれたらやる」ではなく「やらなければ次に進めない」チェックポイントとして扱う
   → 省略しがちな3つの場面:
     - **FE実装後・モデル投入前**（`/fe-hypothesis` フェーズ3）: 実装バグと特徴量の非効果を混同しないために必須
     - **学習完了後**（`/new-experiment` 完了時）: 特徴量重要度・OOF分布を確認し「何がスコアを動かしたか」を把握する
     - **提出後**（`/kaggle-submit` 後）: OOF-LB乖離が大きい場合は誤差分析（どのサンプルで外れたか）を行う
   → AI側から「可視化しましょうか？」と積極的に提案する。ユーザーが「省略」と言った場合のみスキップ可

10. **コンペ初日の外部データインベントリ義務化**
   → Kickoff 時に `data/external/` 内の全ファイル（あれば）を列挙し、各ファイルについて「使う/skip/保留」を明示判定する
   → 判定結果は `COMPETITION.md` の「外部データインベントリ」セクションに記録する
   → 「後で見る」を許可しない。`保留` 判定したファイルは Stage 4 終了までに必ず再評価する
   → **教訓 (過去事例)**: 公開済みの外部時系列データがコンペ最終日まで未使用で放置され、終盤数日でようやく投入したが +1σ LB 改善のみで終わった

11. **ドメイン知識先行プロセス（EDA より前）**
   → Kickoff の Q5 として「このドメインで **ターゲット変数に当然影響するであろう変数** を 5-10 個列挙してください」と問う
   → 列挙された変数が現データに無い場合 → 外部データ取得を検討するシグナル
   → ドメイン専門家がいる場合は **ML パイプラインを動かす前** にヒアリング
   → **教訓 (過去事例)**: ドメイン的に自明な影響変数（天候/季節/地域特性 等）を簡易な binary 1 つで済ませ、連続値の詳細を見落とした

12. **LB プラトー検出時の強制 brainstorm**
   → 同一 LB ± 0.00002 で 5 回以上提出が続いたら、自動的に強制 brainstorm モードに入る
   → 強制ステップ:
     1. **「未試行情報次元」を 5 個以上列挙** (e.g., 天候、地域、時刻、外部統計、ドメイン特化指標)
     2. **`data/external/` の `保留` 判定ファイル**を再評価
     3. **Kaggle Discussion / 上位 Notebook** を `kaggle kernels list --sort-by voteCount` で調査
   → 「飽和」「全方向探索済み」という断言を使わない。**「現在の角度では伸びていない」** と表現
   → **教訓 (過去事例)**: 同じ LB 帯を 7 回以上確認しても保留中の外部データを投入しなかった。プラトー検出を強制化していれば 1 週間早く突破できた

13. **早期却下の禁止（FE/施策提案の判断プロセス）**
   → FE や施策を「却下」「スキップ」「不要」と判断する前に、**3 つのチェックを強制実施**:
     1. **可視化**: 当該特徴量の分布、ターゲットとの関係をプロット (`scripts/visualize.py`)
     2. **ドメイン関連変数の列挙**: 「この特徴量に関連するであろう変数を 5 個」列挙、各々が現データにあるか確認
     3. **相関とimportance分析**: 既存特徴量との相関 + LGB importance gain を計測
   → 上記をクリアして初めて却下可能。**「既存の Y と似ている」「ΔOOF が小さい」だけでは却下理由にならない**
   → **教訓 (過去事例)**: 既存の簡易特徴量 1 つで「カバー済み」と暗黙判断し、関連する連続値特徴量群を試さなかった

14. **Final 2 早期決定の禁止**
   → 残り提出 slot ≥ 2 の段階で「Final 2 を確定しましょう」は **禁句**
   → 残り 1 slot になる **最終提出後にのみ** Final 2 議論を解禁
   → 最終日まで「未試行の大型アイデア」を 1 つは温存する（諦めない姿勢）
   → 最終日に強制的に **「最後の足掻きで試すべき最大のアイデア」を 3 個自発的に提示する**
   → **教訓 (過去事例)**: 残り 3-4 slot の段階で複数回「Final 2 確定」を提案、ユーザーが止めなければ最終日の +1σ LB 改善は生まれなかった

15. **1実験1コミットの厳守（バックグラウンド実行時も例外なし）**
   → OOF 判明後 **5 分以内に commit** する
   → 複数実験のバックグラウンド並行実行時:
     - 各実験の OOF 判明ごとに個別 commit
     - 待ち時間に commit を消化する習慣化
   → 複数実験をまとめて 1 コミットにする「効率化」は **禁止**（追跡不能になる）
   → log.csv も同じタイミングで更新（バッチ更新禁止）
   → **教訓 (過去事例)**: 並行実行中の複数実験を 1 コミットにまとめた結果、各実験の効果が後追いで追跡困難になった

16. **可視化の自発的提案**
   → 数値変化を報告する時、必ず **「グラフを生成しますか？」を提示**
   → 特に必須の場面:
     - FE 追加後の importance gain (top 30 棒グラフ)
     - OOF 分布の比較（前後）
     - 相関マトリクス（新モデルが既存と冗長でないか）
     - LB plateau 時の提出履歴グラフ
   → ユーザーが「省略」と明示しない限り、可視化を実施

17. **Public LB 微改善の懐疑主義（評価指標別の閾値）**
   → Public LB は通常 test の 30% sample で計算され、評価指標に応じた sampling ノイズを持つ
   → 「真の改善」を主張できる最小閾値（経験的目安、絶対値）:

   | 指標 | Public ノイズ床 (典型) | 「突破」最小改善 (2σ) | 備考 |
   |---|---|---|---|
   | AUC | ±0.0001 (n_pos > 5K) | +0.0002 | Hanley-McNeil 由来 |
   | Logloss | ±0.001 absolute | -0.002 absolute | n_test ~ 10K-100K で |
   | RMSE | ±0.1% relative | -0.2% relative | target σ に比例 |
   | Accuracy / F1 | ±0.001 (n > 10K) | +0.003 | threshold 依存も考慮 |

   - **paired comparison**（同一 test set 上の 2 提出比較）は上記より 5-10x 小さい noise
   - 上記未満の改善は **「Public で改善 / Private で要確認」** と表現する
   - **「天井突破」「ブレイクスルー」と呼ぶには 2σ 以上の改善が必要**
   - **教訓**: 過去コンペで Public +1σ 改善を「天井突破」と呼んで新規 FE を採用、結果 Private で逆効果と判明した事例あり

18. **OOF と Private LB の関係性（OOF を信じる）**
   → 多くのコンペで **OOF-Private gap < OOF-Public gap**（Public は test の 30% で variance が大きいため）
   → 結論: **OOF は Public LB と同等以上に Private LB の指標として尊重する**
   → Public と OOF が乖離する候補は要注目:
     - **Public 平凡 + OOF Top** → Public sampling で過小評価された可能性（Private で勝つ）
     - **Public Top + OOF 平凡** → Public 過適合の可能性（Private で負ける）
   → 候補評価表は必ず Public LB と OOF を併記する
   → **教訓**: 過去コンペで OOF Top の候補が Public LB で平凡だったため Final 2 から除外、結果 Private LB で最高だったことが事後判明

21. **OOF最大化とpub_oof_gap最小化の二軸評価（gap最大化は禁止）**

   **データに基づく根拠（過去コンペ全提出の実証）:**
   - OOF → Private の相関: **r=+0.998**（OOF最大化がPrivate最大化に直結）
   - pub_oof_gap → Private の相関: **r=−0.51**（gap拡大はPrivateに有害）
   - pub_oof_gap → シェイクダウン量 の相関: **r=+0.853**（gapが大きいほど必ずシェイクダウン）

   **廃止する考え方:**
   - × 「ΔLB = ΔOOF + Δgap → gap拡大でLBが上がる」
   - × 「Pub-OOF gapを大きく保つほどPrivateに有利」
   - × 「OOF↓かつPublic↑ならgap効果で良いFE」

   **採用する考え方（優先順）:**
   1. **第一目標: OOF最大化** — r=0.998でPrivateに直結。これが唯一の主目標
   2. **第二目標: 同OOF水準ならpub_oof_gap最小化** — シェイクダウンリスクを下げる
   3. pub_oof_gap は「Privateの予測因子」ではなく **「シェイクダウンの警戒指標」** として記録する

   **pub_oof_gap 監視ルール:**
   - 全実験の pub_oof_gap 中央値を「基準線」として記録する
   - 新実験で基準線 + 0.0005 を超えた場合: SESSION.md に「Public過剰浮上警告」を記録
   - 「OOF↓かつPublic↑」は外部データ由来FEを除き原則棄却（OOFを犠牲にしてgapを操作しない）

   **モデルファミリー別OOF信頼性（コンペごとに早期確認）:**
   | ファミリー | 傾向（実証値） | 解釈 |
   |---|---|---|
   | NN系（RealMLP等） | pub_oof_gap小、OOF→Private r≈1.000 | OOFが信頼できる → 主軸候補 |
   | Tree系（LGB等） | pub_oof_gap大、Private≈OOF+0.0002 | 外部テストで浮上しやすい → 補完候補 |
   | Blend/EoS | pub_oof_gap最小でもPriv-OOF負 | OOFがCV誤差相殺で過大評価 → 要注意 |
   → Stage 1.5（後述）でファミリー別pub_oof_gapを記録し「信頼できるOOF」を持つモデルを早期特定する

   → **教訓**: 「gap拡大仮説（gap大=LB高）」はPublicに対しては局所的に観察されるが、Privateに対しては r=−0.51 の逆効果。OOF最大化こそが唯一信頼できる戦略。

22. **アーキテクチャ乗り換え時の公正比較義務**

   新しいアーキテクチャを既存と比較するとき、**以下の条件を揃えることは AI の義務**:

   1. **同一特徴量セット**: 主軸アーキテクチャで確定した FE セットを新アーキテクチャにも適用する
   2. **作業用 HP の確保**: デフォルト HP のまま比較しない。Optuna 20-30 試行以上で作業用 HP を確定してから比較する
   3. **同一 CV 戦略**: fold 数・シード・分割戦略を揃える

   比較前に必ず以下を宣言する:
   ```
   比較条件:
     特徴量: <特徴量セット名>（主軸と同一）
     HP: 作業用 HP 確定済み / デフォルト（要 Optuna）
     CV: <fold数>-fold × <seed数> seed avg
   ```

   **不公正比較のパターン（禁止）:**
   - × 最適化済み LGB（40 実験分の HP + FE）vs デフォルト HP の RealMLP（Stage 1 FE）
   - × 「既存モデルに特化した特徴量セット」で新モデルを評価
   - × 特徴量を絞って「軽量版」で比較（アーキテクチャ差と FE 差が混在）

   > **教訓 (s6e6 事例)**: RealMLP を「FE 削減版 + デフォルト HP」で評価し「LGB より劣る」と誤判断。
   > 後に公正条件（同一 FE + 作業用 HP）で評価したところ RealMLP が主軸として有効と判明した。

19. **Final 2 候補プールの拡張ルール（Public Top + OOF Top の和集合）**
   → Public LB Top-N だけのスクリーニングは Public 過適合候補を優先しがち
   → 候補プール構築: **Public LB Top-10 ∪ OOF Top-10**（重複除去で 10-15 個）
   → 各候補について以下を併記して評価:

   | 候補 | OOF rank | Public rank | OOF-Public gap | プロファイル分類 |
   |---|---|---|---|---|
   | sub_A | #1 | #1 | 標準 | Public+OOF Top |
   | sub_B | #2 | #25 | 大 | **OOF Top のみ (注目!)** |
   | sub_C | #25 | #2 | 大 | **Public Top のみ (Public 過適合の可能性)** |

   → 9 Persona 投票はこの拡張プールに対して実施する
   → **教訓**: Public LB ベースのみのスクリーニングで「OOF Top + Public 平凡」候補を除外、真の最高 Private LB を取り逃した

20. **新規 FE / 外部データの「Private 過適合候補」分類**
   → 新規 FE（特に外部データ、高次元集約特徴量、interaction）が以下を **全て満たす場合** 「Private 過適合候補」と明示マーク:
     1. Public LB 改善が #17 のノイズ床未満（評価指標別の閾値）
     2. OOF 改善が +0.0003 (AUC) / +0.1% (相対 Logloss/RMSE) / +0.001 (Acc/F1) 未満
     3. 既存 FE/モデルとの予測相関が >0.998
   → 「過適合候補」マーク済みの FE/blend は **Final 2 必須採用を避け、別 family の hedge を残す**
   → 採用する場合は SESSION.md に「Private 過適合の可能性」を明示記録
   → **教訓**: 外部生データから生成した集約特徴量が Public +ノイズ床のみ改善し、Private で逆効果になることがある（特に AUC コンペで顕著）

### Autonomous Skill Application（スキル呼び出しが無くてもプロセスを守る）

**AI は明示的なスキル呼び出しが無くても、対応する場面では skill のフェーズプロトコルに従う:**

| 場面 | Autonomous で従うべきプロトコル |
|---|---|
| 新規実験開始 | `/new-experiment` フェーズ1-2: 目的・成功基準・撤退基準を実験開始前に明文化 |
| FE 仮説立案 | `/fe-hypothesis` の因果連鎖言語化: 「なぜ効くか」「メカニズム」を仮説段階で記録 |
| Kaggle 提出前後 | `/kaggle-submit` フェーズ1-4: 提出前確認、Plateau 検出、learning 必須記入 |
| 提出後の OOF/LB 比較 | OOF-LB gap パターンに応じた問いかけ（kaggle-submit フェーズ3） |
| EDA | `/eda-visual` の「問い→発見→FE仮説の種」3 段階 |
| 最終日 | `/kaggle-submit` フェーズ5: Final 2 Persona 投票 |

**ユーザーの skill 呼び出しが必要な場面（friction が必要な節目）**:
- セッション開始: `/ds-resume` （文脈復元）
- Kickoff: `/kickoff` （1 回のみ、コンペ参加直後）
- 振り返り: `/template-update` （改善記録）

**判断基準:**
- スキル呼び出し = ユーザーが明示的に「このプロトコルで進めて」と意思表示する儀式
- 私が autonomous で従う = 規律を守るためのデフォルト動作

> **私自身の責任**: ユーザーがスキルを呼ばなくても、CLAUDE.md の原則は **常に active**。
> 「skill 経由ではないから略式で OK」は **禁止**。

**長時間セッション運用ルール（セッションを切らずに学習監視している場合）:**

学習実行中にセッションを継続している場合、`/ds-resume` を呼ぶ機会が無い。
この状況でも以下のマイルストーンで AI が自発的に「現在地サマリー」を出力する:

| マイルストーン | AI の自発アクション |
|---|---|
| **学習完了・OOF 確認直後** | 「OOF=X.XXXXX、前ベスト比 ΔOOF=±X。commit しますか？」を能動提示 |
| **実験 3 回完了ごと** | SESSION.md の「直近の実験」を更新し「次の方向確認」を問いかける |
| **LB 提出後** | OOF-LB 乖離・pub_oof_gap を表示し「学び」の言語化を促す |
| **FE 棄却 3 連続** | Discussion 調査を提案し「未試行の情報次元」を自発的に列挙 |

→ これらのタイミングで AI が能動的に動くことで、`/ds-resume` を呼ばなくてもサイクルが維持される。
→ ただし **新セッション開始時は必ず `/ds-resume` を呼ぶこと**（文脈リセットの防止）。

### 合成データコンペ向けガイダンス

**なぜ外部シグナルが効くか（メカニズム）:**

合成データは元データを基に統計的に生成されるが、**ターゲットとの細かい相関関係は圧縮・平滑化**される傾向がある。
元データの統計量を特徴量として注入することで、合成プロセスで失われたシグナルを補完できる。

**優先して試す3パターン（内部特徴量より先に試す価値がある）:**

| パターン | 内容 | 実装コスト |
|---|---|---|
| **カテゴリ別ターゲット率** | 元データの各カテゴリ列×ターゲット率をマッピング（外部ターゲットエンコーディング） | 低 |
| **数値分布特徴量** | 元データのターゲット群/非ターゲット群の分布との距離（z-score, percentile, Euclid距離） | 中 |
| **生成ルール逆算** | 元データで `DecisionTreeClassifier(max_depth=None)` を学習し、`tree.apply(X)`（葉ノードID）をfold内TargetEncoderでエンコード。元データでBA=1.0を達成できる場合は生成ルールを直接注入できる可能性がある | 中 |

**生成ルール逆算の注意点:**
- まず `DecisionTreeClassifier(max_depth=None).fit(orig_X, orig_y)` で元データに対して BA=1.0 を達成できるか確認する
- BA=1.0 を達成できない場合はこのパターンは無意味
- 達成できた場合でも、LGB が内部で同等の分割を既に学習している可能性がある（ΔOOF がゼロに近い）
- `tree.predict_proba()` の硬確率（0/1）を特徴量にするのは NG。必ず `tree.apply()`（葉ノードID）＋ fold内TargetEncoder を使うこと

**実装上の注意点:**

- カテゴリ別ターゲット率は1-wayで十分。2-way以上はtree splitsと冗長になりやすい
- percentile計算は `percentileofscore` をループすると O(N²) になる。大規模データでは `np.searchsorted` を使う:
  ```python
  # 高速版（O(N log N)）
  sorted_ref = np.sort(ref_vals)
  df["pct"] = np.searchsorted(sorted_ref, df[col].values, side="right") / len(sorted_ref)
  ```
- 元データが小規模（数千件）でも有効。合成データが数十万件あっても外部シグナルは機能する

---

## Part 2: このプロジェクトの設定

> この節はコンペ開始時に `src/config.py` と合わせて更新する。
> コンペ固有の詳細は `COMPETITION.md` に記録する（この節は環境・規約のみ）。

### コンペ概要

> `/kickoff` 実行後、`COMPETITION.md` を参照。

### 環境・ツール

- パッケージ管理: **uv のみ**（pip・conda 禁止）
- スクリプト実行: `uv run python scripts/<script>.py`
- スクリプト実行例: `uv run python scripts/train.py --model lgb`

**marimoは使用しない。** 可視化はスクリプトから画像ファイルとして出力し、Claudeが読んで対話する。
→ 理由: Claudeはmarimoのレンダリング結果を認識できず、可視化→対話のループが回らないため。

### ディレクトリ規約

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

> `data/output/` 直下にファイルを置かない。役割別サブディレクトリを必ず使う。

### スクリプト構成

**`scripts/`（テンプレート本体・再利用可能）**

| ファイル | Stage | 役割 |
|---|---|---|
| `scripts/train.py` | 1・4 | CV学習の汎用骨格（モデル・特徴量をconfigで切り替え） |
| `scripts/feature_study.py` | 4 | 1列ΔCV計測（FE仮説の効果測定） |
| `scripts/optimize_hp.py` | 3・5 | Optuna HP探索 |
| `scripts/predict.py` | 全般 | OOF予測→提出ファイル生成 |
| `scripts/blend.py` | 6 | アンサンブル・ブレンド |
| `scripts/visualize.py` | 2 | EDA可視化→`data/output/plots/`に画像保存 |
| `scripts/feature_report.py` | 随時 | 特徴量重要度・ΔOOF棒グラフを画像生成 |

**`experiments/runs/`（コンペ固有・使い捨て）**

命名規約: `exp{NNN}_s{stage}_{内容}.py`

```
experiments/runs/
  exp001_s1_lgb_baseline.py       ← Stage 1: 最小ベースライン
  exp003_s3_hp_lgb_optuna.py      ← Stage 3: 作業用HP調整
  exp042_s4_fe_tenure.py          ← Stage 4: 特徴量追加
  exp099_s5_hp_lgb_full.py        ← Stage 5: 本格HP最適化
  exp171_s6_lgb_cb_blend.py       ← Stage 6: アンサンブル
```

- `exp{NNN}`: `experiments/log.csv` の `experiment_id` と一致させる
- `s{stage}`: どのステージの実験かが一目で分かる
- `scripts/` のスクリプトを呼び出すラッパーとして書くことを推奨

### コーディング規約

- パスは必ず `src.config` からインポート（ハードコード禁止）
- 乱数シードは `RANDOM_STATE`（`src.config`から）
- 特徴量名: snake_case・スペースなし
- `src/` 配下に型ヒントを付ける

### Kaggle Notebook 環境サポート

このテンプレートはローカル環境と Kaggle Notebook 環境の両方で動作するよう設計されている。
`src/config.py` が自動的に環境を検出し、パスを切り替える。

**環境検出の仕組み:**

```python
from src.config import IS_KAGGLE, RAW_DATA_DIR, OOF_DIR

# ローカル環境: IS_KAGGLE = False
#   RAW_DATA_DIR = <project_root>/data/raw/
#   OOF_DIR      = <project_root>/data/output/oof/

# Kaggle Notebook 環境: IS_KAGGLE = True
#   RAW_DATA_DIR = /kaggle/input/   ← コンペデータが自動マウント
#   OOF_DIR      = /kaggle/working/data/output/oof/
```

**Kaggle Notebook でのセットアップ手順:**

1. このリポジトリを Kaggle Dataset として登録する（または `kaggle datasets create`）
2. Notebook に Dataset を追加すると `/kaggle/input/<dataset-name>/` にマウントされる
3. Notebook の最初のセルで以下を実行する:

```python
# セル1: リポジトリをパスに追加
import sys
sys.path.insert(0, "/kaggle/input/<dataset-name>")

# セル2: 設定確認
from src.config import IS_KAGGLE, RAW_DATA_DIR, OOF_DIR
print(f"IS_KAGGLE={IS_KAGGLE}")        # → True
print(f"RAW_DATA_DIR={RAW_DATA_DIR}")  # → /kaggle/input/
print(f"OOF_DIR={OOF_DIR}")            # → /kaggle/working/data/output/oof/
```

**Kaggle Notebook での実験スクリプト実行:**

```python
# Notebook セルから実験スクリプトを実行
import subprocess
result = subprocess.run(
    ["python", "/kaggle/input/<dataset-name>/experiments/runs/exp001_s1_lgb_baseline.py"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)  # エラーがある場合はここに出る
```

**Kaggle Notebook でのデータ読み込みパターン:**

```python
import pandas as pd
from src.config import RAW_DATA_DIR, IS_KAGGLE, COMPETITION

if IS_KAGGLE:
    # コンペデータは /kaggle/input/<competition-slug>/ 以下にある
    train = pd.read_csv(RAW_DATA_DIR / COMPETITION / "train.csv")
    test  = pd.read_csv(RAW_DATA_DIR / COMPETITION / "test.csv")
else:
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test  = pd.read_csv(RAW_DATA_DIR / "test.csv")
```

**Kaggle Notebook からの提出フロー:**

```python
# 1. 予測・提出ファイル生成（submission_path() で /kaggle/working/ 以下に保存）
from src.config import submission_path
sub_path = submission_path(model="lgb", oof_score=0.91777, exp_id="001")
sub_df.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}")

# 2. Kaggle CLI で直接提出（Notebook 内から）
import subprocess
result = subprocess.run(
    ["kaggle", "competitions", "submit",
     "-c", COMPETITION,
     "-f", str(sub_path),
     "-m", "exp001 lgb baseline"],
    capture_output=True, text=True
)
print(result.stdout)
```

**注意点:**

- Kaggle Notebook は `/kaggle/working/` のみ書き込み可能（`/kaggle/input/` は読み取り専用）
- GPU 利用時は `device = "cuda"` を config で設定する（LightGBM は `device = "gpu"`）
- セッションをまたぐ場合、`/kaggle/working/` 以下の成果物は消えることがある（Dataset に保存して持ち出す）
- Kaggle CLI で提出するには Notebook の「Internet access」を有効にする必要がある
- `submission_path()` が生成する CSV は `/kaggle/working/data/output/submissions/` に保存される

### 提出ファイルの命名規約

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

### 実験管理（log.csv）

`experiments/log.csv` の主要カラム:

| カラム | 記録タイミング | 説明 |
|---|---|---|
| `experiment_question` | `/new-experiment` | この実験で何を明らかにしたいか |
| `success_criteria` | `/new-experiment` | どんな結果なら成功か |
| `abort_criteria` | `/new-experiment` | どんな結果なら中止するか |
| `cv_val_mean` / `oof_score` | 学習完了時 | OOFスコア |
| `submit_score` | `/kaggle-submit` | LBスコア |
| `oof_lb_gap` | `/kaggle-submit` | OOF tuned − LB（正=OOF過大評価、負=OOF過小評価）。乖離が大きい実験は汎化リスクあり |
| `learning` | `/kaggle-submit` | この実験から何を学んだか |

### 作業ステージとゲート

| Stage | 目的 | 完了条件 | スキル・ツール |
|---|---|---|---|
| **0. Kickoff** | データの文脈理解 | `COMPETITION.md` にデータ種別・外部データ有無・評価指標特性・CV設計の初期判断を記録済み | `/kickoff` |
| **1. 最小ベースライン** | CV/LB相関の確立 | 前処理不要な数値カラムのみ・デフォルトHPでモデルを学習し、LBに提出してCV/LB相関を確認済み。以降すべての改善はこの基準点からのΔで判断する | `/new-experiment` + `/kaggle-submit` |
| **1.5. 早期アーキテクチャサーベイ** | 主軸アーキテクチャの決定 | 候補アーキテクチャ（Tree/NN/Linear等）を最小特徴量セット + 作業用HPで評価し、OOFとpub_oof_gapを記録。「主軸アーキテクチャ」を1つ決定済み。後述「早期アーキテクチャサーベイの手順」に従い実施 | `/new-experiment` |
| **2. EDA** | 問いとFE仮説の種を獲得 | `/eda-visual` で「問い→発見→FE仮説の種」の対話完了。合成データの場合は元データとの分布比較も含む | `/eda-visual` |
| **3. 作業用HP調整** | FE計測の安定化 | Optuna 20〜30試行でFE実験中に使う「作業用HP」を確定済み。目的は完全最適化ではなくΔAUC計測のノイズ低減 | Optuna（軽量） |
| **4. 段階的FE** | 有効な特徴量の特定 | `FE_HYPOTHESES.md` に採用・棄却含む仮説5件以上、棄却理由が分類記録済み。**特徴量は必ず1列ずつ `scripts/feature_study.py` で投入**してΔAUC と feature importance (gain) を計測済み。合成データの場合は外部シグナルFEを先に検証済み。**AV 診断**（adversarial validation）で train/test 分布シフトの有無を確認済み。**FE確定後、全候補アーキテクチャに同一FEを移植して再評価済み** | `/fe-hypothesis` + `scripts/feature_study.py` + AV診断 |
| **5. 本格HP最適化** | 確定特徴量での性能最大化 | Stage4の特徴量セットが確定した状態でOptuna 100試行以上を実施済み。ΔAUCの改善が±0.0002以内で収束していること。**FE変更時の HP retune ルール**: Stage 4 以降に FE が ±20% 以上変動した場合、または domain-specific 新特徴量を追加した場合、HP retune を再実行する（FE変更で HP 最適点は確実に変動。過去事例では HP retune で +1σ OOF 改善を実証） | Optuna（フルサーチ） |
| **6. アンサンブル** | モデル多様性の活用 | 特徴量・HP飽和を確認済み。下記「アンサンブル探索手順」に従い実施済み | `src/utils/ensemble.py` |

**早期アーキテクチャサーベイの手順（Stage 1.5）:**

Stage 1（最小ベースライン）完了直後に実施する。FE探索を始める前に「主軸アーキテクチャ」を決定する。

```
目的: 「このデータに最も合うアーキテクチャ」を最小コストで特定する
実施タイミング: Stage 1 完了後・Stage 2（EDA）開始前
```

**実施手順:**

1. **候補アーキテクチャの選定**: 最低3種を評価する（例: LightGBM / CatBoost / RealMLP / TabNet）
2. **共通評価条件（公正比較のための必須条件）**:
   - 同一の特徴量セット（Stage 1 と同じ最小特徴量）
   - 同一の CV 戦略（fold 数・シード）
   - 作業用 HP（Optuna 20-30 試行、または文献推奨デフォルト）
3. **記録項目**: 各アーキテクチャについて `OOF` と `pub_oof_gap` を記録する

   | アーキテクチャ | OOF | pub_oof_gap | 処理時間 | 採否 |
   |---|---|---|---|---|
   | LightGBM | 0.XXXX | -0.000XX | X min | 主軸候補 |
   | RealMLP | 0.XXXX | -0.000XX | X min | 副軸候補 |
   | … | … | … | … | … |

4. **主軸の決定**: OOF が最高 かつ pub_oof_gap が最小 のアーキテクチャを主軸とする。両者が競合する場合は **OOF を優先**（AI 指針 #21）
5. **副軸の保持**: 主軸と 10% 以内の OOF 差のアーキテクチャは「Stage 6 アンサンブル候補」として記録しておく

**公正比較の注意点（過去事例の教訓）:**

- ❌ 「最適化済み LGB vs デフォルト HP の RealMLP」は **不公正比較**。新アーキテクチャは必ず作業用 HP を揃えてから比較する
- ❌ 特徴量セットを変えての比較は NG（アーキテクチャ差と FE 差が混在する）
- ✅ 「Stage 1 特徴量 × 作業用 HP × 同一 CV」の条件を揃える
- ✅ FE が完成した後に **再評価**する（Stage 4 完了後に全候補アーキテクチャへ同一 FE を移植）

> **教訓 (s6e6 事例)**: LGB 主軸のまま 40+ 実験を費やし、RealMLP を試したのがコンペ終盤だった。
> 早期サーベイで RealMLP の優位性（提出効率 50x）を特定できていれば、探索効率が大幅に改善した。
> Phase 効率分析: LGB FE 探索（31 提出）= +0.000007 LB/提出、RealMLP 移行（8 提出）= +0.000343 LB/提出

**AV 診断（Adversarial Validation）の標準実施手順:**

Stage 4 で特徴量追加が一段落した時点、および Stage 6 移行前に必ず実施する。

```python
# 簡易版: train+test 結合データで is_test を予測
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

X_av = pd.concat([X_train, X_test], ignore_index=True)
y_av = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
# 3-fold CV で AV-AUC を測定
```

**判定基準:**

| AV-AUC | 判定 | 対応 |
|---|---|---|
| < 0.55 | ✅ シフトなし | そのまま継続 |
| 0.55-0.65 | 🔶 軽度シフト | importance weight 試行価値あり |
| 0.65-0.80 | ⚠️ 中度シフト | 上位重要度特徴量を drop 検討 |
| > 0.80 | ❌ 強いシフト | drop 必須 or データ前処理の見直し |

> **教訓 (過去事例)**: BASE_FEATURES では AV-AUC=0.5（無問題）だったが、後追いで拡張した特徴量で AV-AUC=1.0（count 系特徴量が train/test 構造差を leakage していた）。最終日まで気づかなかった

**アンサンブル探索の推奨順序（Stage 6 の標準手順）:**

```
STEP 1【相関確認】← 必ず最初に実施
  追加候補モデルのOOFと既存モデル群のOOFの相関を計算する:
  ```python
  from src.utils.ensemble import correlation_check
  corr, skip = correlation_check(oof_existing, oof_candidate)
  ```
  → 相関 < 0.998: STEP 2 へ
  → 相関 ≥ 0.998: スキップ。ただし【棄却分析A】を実施してから次へ

STEP 2【Simple Blend】
  既存モデル群との最適重みブレンドを試す（`optimize_weights()` で重み探索）
  → OOFが改善: STEP 3 へ
  → OOFが改善しない: 【棄却分析B】を実施してから次へ

STEP 3【Greedy Hill Climbing】
  保有する全OOFファイルを対象に系統的に探索する（`greedy_ensemble()`）
  → 改善するモデルの組み合わせを特定する
  → 全モデルで改善なし: 【棄却分析C】を実施

STEP 4【Stacking】
  LGB/CB以外に予測パターンが異なるモデルが2種以上ある場合のみ検討する
  → 相関が高いモデル同士のStackingは効果がない（前提の再確認）

STEP 5【Pseudo-labeling】
  アンサンブルの多様性が飽和した場合に有効な代替戦略。
  test の高信頼度サンプルに疑似ラベルを付与し、train に追加して再学習する:
  ```python
  test_proba = <最良モデルの test 予測>
  mask = test_proba.max(axis=1) >= threshold  # 閾値: 0.95 を最初に試す
  pseudo_df = test[mask].copy()               # 疑似ラベルサンプル
  # 各 fold の train に pseudo を追加して学習。OOF は元 train のみで評価
  ```
  探索順序:
    1. threshold=0.95 で OOF 改善を確認
    2. 改善あれば threshold=0.99/0.90 も比較（OOF-LB 乖離に注意）
    3. 改善あれば Iterative（前回の test 予測を次回の pseudo 源泉に）を2回まで試す
       → 3回目以降はラベルノイズ増幅で悪化するケースが多い
  注意: OOF-LB 乖離が拡大する閾値は LB 悪化リスクあり。乖離を記録すること

  **Pseudo 源泉の品質とリーク診断:**
  - pseudo 源泉の優先順位: train fold 内の高確信度サンプル > 自前 test 予測 > 外部公開予測
  - **外部予測を pseudo source に使う = モデル蒸留**（genuine pseudo-labeling ではない）
    → test 予測が外部予測と 99%+ 一致し、独立シグナルを失う。Highクラスのみなど部分的でも同様
  - **リーク診断テスト**（pseudo 採用前に必ず実施）:
    ```
    同一構成で比較:
      pseudo あり  → (OOF_A, LB_A)
      pseudo なし  → (OOF_B, LB_B)
    OOF_A > OOF_B かつ LB_A < LB_B → leakage 確定。pseudo 源泉を見直す
    ```
    → 外部予測由来 pseudo は高確率でこのパターンに該当する

STEP 6【自前マルチシード EoS voting】
  外部公開予測が存在しなくても使える row-level voting 手法。
  同一アーキテクチャを複数シードで学習し、シード間で予測が割れる「disagree 行」を
  別系統モデル（Aux）で解決する。

  **設計パターン:**
  1. ベースモデルを N シード × K fold で学習（推奨: 5シード × 5fold）
  2. シードごとの test 予測を保存: `test_{exp_id}_seed{s}_proba.npy`（後続分析で再学習不要にする）
  3. 全シードが一致する「agree 行」はそのまま採用
  4. シード間で割れる「disagree 行」を Aux モデルで上書き

  **disagree 行の分類（5シードの場合）:**
  - **4:1 スプリット**（N-1 シードが同方向）: 決定論的ルール特徴量などが N-1 シードを同方向に引く系統誤りケース。LB への貢献が大きい
  - **3:2 スプリット**（真に曖昧）: 補正効果は小さいが加法的に寄与する

  **Aux モデルの選定基準:**
  - **OOF 精度の最低閾値**: ベースモデルの OOF との差が **-0.002 以内**であること
    → それ以上低い Aux は disagree 行の大半を誤って変更し LB を悪化させる
  - **系統的独立性**: ベースの系統誤りを引き起こす特徴量を「含まない」Aux を選ぶ
  - **変化行数で品質を判断**: 変化行が少ないほど Aux の選択精度が高い（量より質）

  **Union Aux による漸進的改善（収穫逓減に注意）:**
  - Aux A（N 行変更）+ Aux B のユニーク行（M 行）→ LB は M 行分だけ追加改善
  - M が小さくなるにつれて改善はゼロに収束する。**ユニーク行が 15 行未満の Aux 追加に提出枠を使うのは非推奨**

  **保存規約（再学習を防ぐための必須設計）:**
  ```python
  # 学習時: シードごとに test 予測を保存
  np.save(OOF_DIR / f"test_{exp_id}_seed{seed}_proba.npy", test_preds)
  # 分析時: 保存済み予測を読み込んで EoS バリアント生成（再学習不要）
  seed_preds = [np.load(OOF_DIR / f"test_{exp_id}_seed{s}_proba.npy") for s in seeds]
  ```

STEP 7【外部公開予測の Row-level Voting】
  Kaggle Datasets や Discussion に他の参加者が公開した高LB予測ファイルが存在する場合、
  それらを「外部モデル」として row-level voting に活用できる。

  **確認手順:**
  ```bash
  # 公開データセット一覧（コンペ名で検索）
  kaggle datasets list --search "<competition-id>" --sort-by voteCount
  # ダウンロード
  kaggle datasets download <author>/<dataset-slug> -p /tmp/external/
  ```

  **Row-level voting の基本パターン:**
  1. 自前モデル群の bias 修正済み予測 vs 外部予測 を行単位で多数決
  2. 不一致行のみを別系統モデル（Aux）で解決（EoS voting パターン）
  3. 外部予測を pseudo source として自前モデルを再学習

  **重要な注意点:**
  - **精度↔独立性のジレンマ**: 外部予測を pseudo source にすると自前モデルが外部予測の蒸留になる
    → OOF は改善するが test 予測が外部予測と 99%+ 一致し、独立シグナルを失う
    → pseudo source に外部予測を使う場合は `agree_rate = (pred_test == ext_pred).mean()` を必ず計測する
  - **差分役割モデルの選び方**: voting で「差分を解決する役割（Aux）」には「最低限の精度」と「系統的独立性」の両方が必要
    → OOF が高い Aux モデル同士を比較した場合、より高精度な Aux は主モデルと同じ予測に収束しがちで差分を解決できない（Aux 精度↑→LB↓ の逆相関が生じる）

  **外部知見系の安定ピーク検知（Public LB 過適合を防ぐ 3 シグナル）:**

  外部 Row-level Voting は自前モデルの OOF 上限を超えられるが、Public LB を最大化し続けると
  Private LB で過適合が起きやすい。以下のシグナルで「安定ピーク」を見極める:

  1. **クラスター収束**: 補正セットが異なる実験 3 本以上が ±0.00002 の Public スコアに収束したら
     その帯域が安定ピーク。クラスターの最もシンプルな 1 本を最終選択候補として即座に確保する。
     → 1 本だけ跳ねた実験は Public テストサンプルへの偶然ヒットの可能性がある（確認前に手放さない）

  2. **限界改善の縮小**: 補正行数が増えているのに Public の増分が縮んでいるとき、
     1 行あたりの改善が Public テストの統計的雑音水準に達したサイン。
     目安: 多数決クラス補正の理論的最小単位を下回る改善は Private には出ない可能性が高い。

  3. **補正ルールの複雑化**: 単純な多数決閾値から複合条件フィルタへ移行するほど
     Public 過適合リスクが増加する。複雑さが増した実験が Public を超えても hedge は外さない。

  **自前系 vs 外部知見系の並行管理:**
  - 自前系（自前モデルの補正）と外部知見系（外部 voting）は Private ピークが一致しない
  - 外部知見系は Public を高く引き上げられる一方、安定ピークを過ぎると Private が落ちる
  - OOF 変化なしで Public だけ上昇が続く状況は「外部知見系の安定ピーク超過」の典型サイン
  - 最終 2 本の選択は「外部知見系 Public 最高 1 本 ＋ 外部知見系安定ピーク 1 本」を基本とする
    （自前系と外部知見系を並行している場合は「外部知見系安定ピーク ＋ 自前系ベスト」も有効）

STEP 8【Blend of Blends - 構造的に異なる blend の consensus】
  Stage 6 で複数の異なる philosophy の blend が同等 LB に到達した時、
  それらの平均がさらに僅かな改善を提供することがある。

  **適用条件:**
  - 2 つ以上の blend が同等 Public LB を持つ
  - 各 blend が構造的に異なる（例: greedy HC vs equal weight）
  - 各 blend が overfit リスクの異なる profile を持つ

  **実装パターン:**
  ```python
  # 構造的に異なる 2 つの blend を 50/50 で平均
  final = 0.5 * blend_greedy_hc.test_pred + 0.5 * blend_equal_weight.test_pred
  ```

  **メカニズム:**
  - Greedy HC は OOF ノイズに重みを最適化 → OOF overfit bias
  - Equal weight は variance reduction だが weighting suboptimal → variance bias
  - 2 つの異なる bias の consensus 効果で個別エラーが部分的に補完

  **期待改善:**
  - Public LB +0.00000〜+0.00002 (微小、AI 指針 #17 のノイズ床近辺)
  - 統計的にはノイズ範囲内のことが多い → 必ず submit して確認

  **Private LB での挙動（重要な注意）:**
  - Public LB +1σ 改善が Private LB に **反映されないことが多い**
  - BoB の Private LB ≈ 親 blend の平均 になる場合が多い (50% 線形結合のため)
  - **BoB を Public LB ベストとして Final 1 に採用するのは AI 指針 #19/#20 違反のリスク**

  **Final 2 候補にする際の注意:**
  - BoB は親 blend を 50% 含むため、Final 2 で親 blend を hedge にすると **共倒れリスク**
  - 推奨構成（優先順）:
    1. **Final 2 = (親 blend A, 親 blend B)**: 純粋多様性、最も安全（推奨デフォルト）
    2. **Final 2 = (BoB, 別 family blend)**: BoB を採用するなら family が完全に独立な blend と組む
    3. ⚠️ **避けるべき: Final 2 = (BoB, 親 blend)** → 共倒れリスク高、25%/75% 不均衡で多様性低

  > **教訓**: BoB が Public LB を +1σ 改善したが、Private LB は親 blend と同等（差なし）の事例あり。Public 微改善 = Private 改善とは限らない

**STEP 6 への重要追記 — Multi-seed averaging のデフォルト化:**

  実証的に、tree モデル (LGB/XGB/CB) の multi-seed avg5 は **+0.00010-0.00020 OOF** の安定的改善を提供する。
  Stage 6 移行前の base model 構築時、production blend に投入する model は **multi-seed=5 を default** とする。

  ```python
  SEEDS = [42, 0, 1, 7, 2026]  # default の 5 seeds
  for seed in SEEDS:
      train_with_seed(seed) → save oof_{exp}_s{seed}.npy, test_{exp}_s{seed}.npy
  avg5_oof = np.mean(all_seed_oofs, axis=0)
  ```

  - n_ens 内蔵モデル (RealMLP) は internal ensemble で代替可
  - TabM は GPU 必須で multi-seed コスト高 → single seed で OK
  - **CB は特に multi-seed 効果が高い**（過去事例で Δ=+2σ レベルの OOF 改善）
```

**アンサンブル棄却分析（「効かなかった」で終わらせない）:**

| 棄却パターン | なぜ効かなかったか | 次に試せること |
|---|---|---|
| **A: 相関 ≥ 0.998（同一予測）** | 同じ特徴量・同じCV分割・同じアルゴリズムファミリーは予測が収束する | ①異なるCV戦略（fold数・seed変更）②異なる特徴量サブセット③全く異なるアルゴリズム（NN・RF・XGB deep）を試す |
| **B: OOFは高いが blend で改善なし** | 既存モデルと同じエラーパターンを持つ（誤差の方向が同じ） | OOFスコアだけでなく**誤差の相関**を確認する（高OOFでも誤差が相関していれば多様性なし） |
| **C: Greedy HC で全モデル改善なし** | 保有モデル群の多様性が飽和している | ①FEに戻り新しいシグナルを探す ②Pseudo-labeling（STEP 5）を試す ③問題の性質上アンサンブルの伸びしろが小さい可能性 |
| **D: Stacking が Simple Blend を下回る** | ベースモデルの予測が相関しすぎてメタ学習できない | ①ベースモデルの多様性を高めてから再挑戦 ②メタ特徴量に生の特徴量を追加 |
| **E: Pseudo-labeling で OOF↑ LB↓** | leakage 確定。pseudo 源泉（外部予測・train高確信度）に問題がある | リーク診断テストで源泉を特定し、より独立な源泉（train fold内）に変更する |
| **F: EoS Aux で LB 改善なし** | Aux の OOF が最低閾値（差 -0.002）を下回っているか、系統的独立性がない | Aux OOF を確認し閾値以上の別候補を探す。変化行数が多い=精度不足のサイン |
| **G: 外部予測 voting で安定しない** | 外部予測の安定ピークを超えて補正ルールが複雑化している | 3 シグナルで安定ピークを確認し、シンプルな補正ルールの実験に戻る |

> **棄却は終わりではなく、次の探索方向を示すシグナル。**
> 各 STEP で「なぜ効かなかったか」を1文で記録してから次に進む。

**FE の採用・棄却判断は ΔOOF だけで行わない（importance との併用）:**

ΔOOF（greedy な逐次追加）は、既存特徴量と相関が高い列の貢献を過小評価する。
`feature_study.py` で新列を追加したとき、ΔOOF が小さくても以下の手順で二重確認する:

```
1. ΔOOF を確認する（目安: +0.0003 以上 = 明確な改善）
2. 追加後モデルの feature importance (gain) を確認する
   → 新列の importance が BASE 既存列の中位以上なら「情報は持っているが既存列と重複」
   → 新列の importance が BASE 最下位を大幅に下回るなら「真に情報なし」
```

判断マトリクス:
| ΔOOF | Importance | 判断 |
|---|---|---|
| ≥ +0.0003 | — | ✅ 採用 |
| < +0.0003 | BASE 中位以上 | 🔶 保留: 既存列と競合。どの列と重複しているか分析する |
| < +0.0003 | BASE 最下位未満 | ❌ 棄却: 真に情報なし |
| マイナス | — | ❌ 棄却: ノイズ追加 |

> **「ΔOOF < 0.0003 → 即棄却」は誤り。importance が中位以上なら既存列の代替候補として記録する。**

**FE の有効性はアーキテクチャに依存する（LGB 棄却 ≠ 全アーキテクチャで棄却）:**

あるアーキテクチャで ΔOOF < 閾値だった特徴量が、別アーキテクチャでは有効なケースがある。
これは特徴量の表現力（線形 vs 非線形）とアーキテクチャの相性による。

```
棄却の意味を正しく解釈する:
  × 「この特徴量は無効」      ← 誤り
  ○ 「主軸アーキテクチャ（LGB等）ではこの FE が効かなかった」
```

**Stage 4 棄却記録への追記義務:**

FE_HYPOTHESES.md の棄却エントリには「棄却したアーキテクチャ」を必ず明記する:
```
- 棄却: LGB で ΔOOF=+0.00010（閾値未満）
- 未評価: RealMLP, CatBoost（別アーキテクチャでの効果は不明）
- 再試行条件: Stage 1.5 で RealMLP が主軸になった場合は再評価する
```

**Stage 4 → Stage 6 移行時のアーキテクチャ間 FE 移植:**

FE 確定後、Stage 1.5 で「副軸候補」にリストされた全アーキテクチャへ同一 FE セットを移植して再評価する。
LGB で棄却された FE でも、副軸アーキテクチャ（例: RealMLP）に対しては効果が異なる場合がある。

> **教訓 (s6e6 事例)**: LGB で棄却した複数の特徴量が RealMLP では有効だったが、
> 「LGB 棄却 = 不採用」と判断して移植せずに提出してしまった。アーキテクチャ乗り換え時は FE の棄却リストを再検討する。

**FE 仮説の棄却記録には「再試行条件」を必ず書く:**

FE_HYPOTHESES.md の棄却エントリには以下を記録する:
```
- 棄却理由: なぜ効かなかったか（メカニズムレベルで）
- 再試行条件: どう変えれば効く可能性があるか（「不明」も可）
```
改良版を実装する前に、「棄却理由」が「再試行条件」で本当に解決されるかを確認してから着手する。
（例: 硬確率→棄却理由「0/1ノイズ」→再試行条件「ソフトな連続値に変換」→改良案「leaf_id + TargetEncoder」）

> **ステージを飛ばさない。**
> - Stage 1 を省くと CV/LB乖離に気づくのが遅れる
> - Stage 3 を省くと Stage 4 のΔAUC計測がノイズに埋もれる
> - Stage 4 で `scripts/feature_study.py` を使わず複数列を一度に追加すると、どの特徴量が効いたか分からなくなる
> - Stage 5 は Stage 4 完了後でないと最適HPが変わるため意味が薄い
> - Stage 6 の STEP 1（相関確認）を省くと、実装・学習コストをかけてから「重みゼロ」と判明する

### Kaggle提出ルール

- 提出は必ず `/kaggle-submit` スキル経由（直接CLIは禁止）
- 提出前: git working tree がcleanであること
- 提出後: `submit_score`・`lb_rank`・`learning` を log.csv に記録すること

**⚠️ 提出前確認は AI の義務（省略禁止）:**

ユーザーが「提出します」「試します」と言った場合でも、**AIは必ず以下を表示してからコマンドを実行する:**

```
`sub_NNN_model_0.XXXXX_YYYYMMDD_HHMM.csv` を提出します。
本日 X/5 回目の提出になります。よろしいですか？
```

- 「提出します」= ユーザーの **意思表示**。AIの **実行許可** ではない
- 対象ファイル名・本日の提出回数・残り回数を明示してから「よろしいですか？」と確認する
- ユーザーの「OK」「はい」「どうぞ」を受けてから初めて実行する
- **この確認を省略した場合はテンプレート違反とみなす**

**`/kaggle-submit` スキルが実行不可の場合（`disable-model-invocation` エラー等）:**

スキル経由が不可でも、以下のチェックリストを手動で実施してから CLI 提出する:

```
□ git status がcleanか確認
□ 提出ファイルは submission_path() 生成のものか確認
□ 提出後: kaggle competitions submissions | head -3 でLBスコアを確認
□ log.csv の submit_score 列を更新
□ log.csv の oof_lb_gap 列を計算・更新（= oof_score - submit_score）
□ SESSION.md のスコアテーブルを更新（OOF-LB乖離列を必ず記入）
□ SESSION.md に「本日の提出数（N/10）」を記録
□ git commit でLB結果を記録
```

スキルが提供するフローをAIが代替する。チェックリストの省略は禁止。

### 提出枠の管理方針

**基本方針: 残り枠は使い切る。未使用の提出枠はゼロ価値。**

`/kaggle-submit` 実行のたびに以下を確認・提示する:
- 本日の使用済み回数 / 上限（通常5回）
- コンペ締め切りまでの残り日数
- 推定残り総提出枠（本日の残り + 残り日数 × 日次上限）
- `data/output/submissions/` 内の未提出候補ファイル一覧
- **OOF-LB 乖離**（今回提出の OOF tuned − LB）を SESSION.md のスコアテーブルに追記する

> 「何をSubmitするか」は実験の進行状況と残り枠を見て毎回判断する。
> ステージごとの固定配分ではなく、「今この瞬間に最も価値のある1本」を選ぶ。

**最終選択（Final Submission Selection）の 2 本ルール:**

Kaggle の最終選択 2 本は「Public 最高を 2 つ」より「Private LB shakedown 防御」を最優先。
Public LB は全テストの 30% 程度で評価されるため、Public LB の小さな差（±0.00005）は統計的にノイズ範囲内。

**選定原則（優先順）:**

1. **候補プール拡張**: Public Top のみのスクリーニング禁止（AI 指針 #19）
2. **Public 最高 1 本 + 構造的多様性を持つ hedge 1 本**
3. **Shakedown 防御**: 両者が同じ family の場合、片方コケると共倒れ → 親が独立な 2 blend を取る
4. **Persona 投票**: 主観バイアスを排除するため複数視点で評価

**候補プール構築（Persona 投票の前に必須実施 - AI 指針 #19）:**

Public LB ベースだけのスクリーニングは Public 過適合候補を優先しがち。以下の和集合をプールに含める:

- **Public LB Top-10**: 標準的な選定基準
- **OOF Top-10**: Private LB の predictor として尊重（AI 指針 #18）
- **重複除去で 10-15 個**: Persona 投票の対象

各候補のプロファイルを以下のテーブルで整理:

| 候補 | OOF rank | Public LB rank | OOF-Public gap | 分類 | 注目度 |
|---|---|---|---|---|---|
| sub_A | #1 | #1 | 標準 (例: +0.0007 for AUC) | Public + OOF 両 Top | 標準候補 |
| sub_B | #2 | #25 | 大 (例: +0.0010) | **OOF only Top** | ⭐ Private で勝つ可能性 |
| sub_C | #25 | #2 | 大 (例: +0.0004) | **Public only Top** | ⚠️ Public 過適合可能性 |
| sub_D | (例) BoB | #3 (Public 最高 +0.00001) | 標準 | Public 微改善 | ⚠️ ノイズ床近辺、要 #17 適用 |

**注目度の判断:**
- ⭐ OOF only Top: Public sampling で過小評価された真の高品質候補。Final 2 候補として **必ず検討対象に**
- ⚠️ Public only Top: OOF 平凡なのに Public 高 → Public test sample への過適合疑い。**hedge を必ず付ける**
- ⚠️ Public +1σ 改善: AI 指針 #17 のノイズ床。「突破」と呼ばず、Private 確認まで保留扱い

**Persona チェックリスト（拡張プールに対して実施、最終日に必ず実施）:**

以下の 9 ペルソナの視点で Final 2 を評価し、多数決で選定する:

| Persona | 主張 |
|---|---|
| **Kaggle Grandmaster** (経験派) | "Public LB +0.00001 はノイズ。100回中70回はノイズ。**親 blend を取れ**" |
| **Statistical Theorist** (理論派) | "Public LB AUC 差 ±0.00005 以内は統計的区別不能。**Variance minimization で構造的に異なる 2 つ**" |
| **Risk Management** (守り派) | "共倒れ防止が最優先。**独立な 2 blend** を取れ" |
| **Pragmatic Engineer** (実践派) | "実証された Public 最高を **捨てるな**" |
| **Newcomer** (素朴視点) | "Blend of Blends は親の 50% 平均。**親をそのまま使えばいい**" |
| **Domain Expert** | "ドメイン的に最適な model を必ず 1 本入れる" |
| **ML Researcher** | "Bias 差が最大の **異なる philosophy のペア**を取る" |
| **External Reviewer** | "Family が同じ 2 つは hedge にならない" |
| **Behavioral Economist** | "**Hindsight bias / Loss aversion** を排除、データに基づけ" |

**投票ルール:** 多数派の意見に従う。同数の場合は **Risk Management の意見を優先**（shakedown 回避を最優先）。

**典型的 Final 2 構成パターン:**

| パターン | 1 本目 | 2 本目 | 適用条件 |
|---|---|---|---|
| **A. 親ペア** (推奨デフォルト) | Greedy HC blend | Equal weight blend | 両者 Public LB 同等の時 |
| **B. Public 最高 + 安定ピーク** | Public LB best | 外部 voting 安定ピーク代表 | 外部 voting 系列の時 |
| **C. 自前 + 外部** | 自前 best | 外部 best (rv3 系) | 外部 voting 有効と確認後 |
| **D. Blend of Blends + 別 family** | Blend of Blends (Public 最高) | 別 family blend | BoB 親に含まれない blend がある時 |

**重要な警告:**

- **Blend of Blends を Final 2 に入れる場合の罠**: BoB は親 blend を 50% 含むため、Final 2 で親 blend を hedge にすると **共倒れリスク** (例: BoB + 親 A は実効重み 75% 親 A / 25% 親 B で多様性低)
- **Public 最高への過度な執着**: Public LB +0.00001 は Public test 56,450 行のサンプリングノイズ範囲内
- **OOF-LB gap が一定なら**: OOF 同等 = Private LB 期待値も同等。Public LB 微差は誤差

**確保のタイミング:**
- 安定ピーク確認と同時に「2 本目候補」をメモ
- 終盤に判断すると Public 最高への執着で見逃しやすい
- **コンペ前日までに Final 2 候補を 3-4 個に絞り、最終日は ペルソナ投票のみ実施**

> **教訓 (過去事例)**: 9-persona vote で多数決により「親ペア (greedy HC + equal weight)」(パターン A) を選定。BoB は親 blend を 50% 含むため hedge 不適と判断し見送り → 結果的に Public LB 1σ 改善を放棄したが Private LB shakedown を回避

### ブランチ管理

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

### コミット規約

**コミットのタイミング（3つのルール + 並行実行ルール）:**

1. **学習完了直後にコミットする** — OOFスコアが判明した直後 **5 分以内**。時間を置かない
2. **1実験 = 1コミット** — 複数の変更を一度のコミットにまとめない。何が効いたか追跡できなくなる
3. **`/kaggle-submit` の前にコミット済みであること** — `git status` がcleanでなければ提出しない

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

### テンプレート改善プロトコル

コンペ作業中に改善点を発見したら `/template-update <説明>` を実行する。

スキルが「汎用プロセス / 技術インフラ / コンペ固有」を峻別して記録先を判断する。
**コンペ固有の知見をそのままテンプレートに入れない**こと。

**TODO_TEMPLATE.md → CLAUDE.md 反映サイクル（重要）:**

TODO_TEMPLATE.md への記録は「改善の予約」に過ぎない。**記録したことを CLAUDE.md に反映して初めて完了**。

反映タイミング:
- コンペの区切り（LB新ベスト更新・ステージ移行・セッション開始時の `/ds-resume`）
- TODO_TEMPLATE.md に TODO 項目が3件以上蓄積したとき
- ユーザーから「振り返りをしましょう」と指示されたとき

反映手順:
1. TODO_TEMPLATE.md の `TODO` 項目を読む
2. CLAUDE.md の対応セクションに実際に追記・修正する
3. TODO_TEMPLATE.md の `状態` を `DONE` に更新する
4. `IN PROGRESS` 項目も同様に処理する

**「TODO_TEMPLATE.mdに書いた」= 改善完了ではない。CLAUDE.mdが更新されるまで未完了。**

mainマージ前チェックリスト:
- [ ] コンペ名・ターゲット列のハードコードを `src/config.py` の変数に置換
- [ ] 回帰・分類の両方に対応（またはどちらか明記）
- [ ] 新依存関係を `pyproject.toml` に追加済み
- [ ] カスタマイズ箇所を `# TODO:` コメントで明示
