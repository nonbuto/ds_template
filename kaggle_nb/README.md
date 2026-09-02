# kaggle_nb/ — Kaggle Notebook 実行用ファイルの置き場

**このディレクトリの中身は git 管理しない**（`.ipynb`・`output/`・`push*/` は `.gitignore` 済み）。
コンペごとの Notebook と実行結果はコンペブランチの作業ツリーに残り、テンプレート本体には入らない。

> 2026-09-02 の精査で、過去コンペの Notebook と予測 `.npy` **41 ファイル・106MB** が
> テンプレート本体（`main`）に混入していたことが判明したため、追跡対象から外した。

## 使い方

手順は `PLAYBOOK.md#kaggle-gpu-ワークフローcsv提出コンペ` にある。要点だけ:

- **方式(A) 自己完結 Notebook**（推奨・AI 実行可）—— 必要な処理を 1 ファイルに閉じて
  `.ipynb` 化し、`.ipynb` + `kernel-metadata.json` の 2 ファイルだけを push する
  （`dataset_sources: []`）。前コンペはこの方式で 19 本すべてを完走した
- **方式(B) Dataset 同期** —— `rsync` による一括コピーは AI の実行環境でブロックされるため、
  **ユーザー自身のターミナルで実行**してもらう

変換は `uv run python -m scripts.to_kaggle_nb`、回収は `kaggle kernels output`。
回収した OOF / test 予測は `data/output/oof/` へ移してから使う。
