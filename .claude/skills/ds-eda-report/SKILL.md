---
name: ds-eda-report
description: 廃止済み。/ds-eda-visual と FEATURE_REPORT.md に機能を統合した。このスキルを呼んだ場合は移行先を案内する。
---

# /ds-eda-report スキル（廃止）

このスキルは廃止されました。機能は以下に統合されています。

## 移行先

| 旧機能 | 新しい場所 |
|---|---|
| データ概要・統計の自動収集 | `/ds-eda-visual` フェーズ2（データ概要の把握） |
| EDA_SUMMARY.md の生成 | `/ds-eda-visual` フェーズ7（セッションの記録） |
| 生変数の欠損・分布・ターゲット相関 | `FEATURE_REPORT.md` の「生変数サマリー」セクション |
| 可視化 | `scripts/visualize.py` → `data/output/plots/` に画像保存 |

## このスキルを呼んだ場合の対応

「`/ds-eda-report` は廃止されました。`/ds-eda-visual` を実行してください。」とユーザーに伝え、
`/ds-eda-visual` の実行を促す。
