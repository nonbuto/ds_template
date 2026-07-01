"""
実験スクリプト → Kaggle Notebook (.ipynb) 変換ツール

実験スクリプト (experiments/runs/exp***.py または scripts/*.py) を
Kaggle Notebook で実行可能な .ipynb ファイルに変換する。

変換方式:
  1. marimo 形式スクリプト (import marimo as mo を含む):
       marimo export ipynb で変換
  2. 通常のPythonスクリプト:
       セットアップセル + コード本体セルの2セル構成に変換

生成する .ipynb は:
  - /kaggle/input/<dataset-name>/ からコードを参照する
  - /kaggle/input/<competition>/ からデータを読み込む
  - /kaggle/working/ に成果物（OOF .npy, submission.csv）を保存する

使い方:
    # 実験スクリプトから変換（Notebookで実行する .ipynb を生成）
    uv run python scripts/to_kaggle_nb.py experiments/runs/exp108_s6_realmlp.py

    # 出力ファイル名を指定
    uv run python scripts/to_kaggle_nb.py experiments/runs/exp108_s6_realmlp.py -o kaggle_nb/exp108.ipynb

    # Notebook提出コンペ用（submission.csv を output に保存するセルを追加）
    uv run python scripts/to_kaggle_nb.py experiments/runs/exp108.py --submission-mode

    # Dataset名を明示指定（デフォルト: src/config.py の COMPETITION を使用）
    uv run python scripts/to_kaggle_nb.py experiments/runs/exp108.py --dataset-name my-ds-template
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import COMPETITION, IS_KAGGLE


# ──────────────────────────────────────────────────────────
# Kaggle セットアップセルのテンプレート
# ──────────────────────────────────────────────────────────

SETUP_CELL_TEMPLATE = """\
# ============================================================
# Kaggle Notebook セットアップ
# (to_kaggle_nb.py によって自動生成)
# ============================================================
import sys
import subprocess
from pathlib import Path

# --- ds_template をパスに追加 ---
# API push実行: /kaggle/input/datasets/<user>/<name>/
# UIノートブック: /kaggle/input/<name>/
DATASET_NAME = "{dataset_name}"
DATASET_USER = "{dataset_user}"
_api_path = Path(f"/kaggle/input/datasets/{{DATASET_USER}}/{{DATASET_NAME}}")
_ui_path  = Path(f"/kaggle/input/{{DATASET_NAME}}")
TEMPLATE_DIR = _api_path if _api_path.exists() else _ui_path
if TEMPLATE_DIR.exists():
    sys.path.insert(0, str(TEMPLATE_DIR))
else:
    raise FileNotFoundError(
        f"Dataset '{{DATASET_NAME}}' が見つかりません。\\n"
        f"確認したパス: {{_api_path}}, {{_ui_path}}\\n"
        f"Kaggle Notebook に ds_template Dataset を追加してください。"
    )

# --- コンペ設定を注入（Datasetのconfig.pyのプレースホルダーを上書き）---
import src.config as _cfg
_cfg.COMPETITION = "{competition}"
_cfg.TARGET_COL  = "{target_col}"
_cfg.RAW_DATA_DIR = _cfg._KAGGLE_COMP_ROOT / "{competition}"
_cfg.EXPERIMENT_NAME = f"{{_cfg.COMPETITION}}_baseline"

# --- 設定確認 ---
from src.config import IS_KAGGLE, RAW_DATA_DIR, OOF_DIR, SUBMISSIONS_DIR, COMPETITION
print(f"IS_KAGGLE      = {{IS_KAGGLE}}")
print(f"COMPETITION    = {{COMPETITION}}")
print(f"RAW_DATA_DIR   = {{RAW_DATA_DIR}}")
print(f"OOF_DIR        = {{OOF_DIR}}")
print(f"SUBMISSIONS_DIR= {{SUBMISSIONS_DIR}}")

# --- コンペデータの存在確認 ---
if not RAW_DATA_DIR.exists():
    print(f"\\n⚠️  コンペデータが見つかりません: {{RAW_DATA_DIR}}")
    print(f"   Kaggle Notebook にコンペデータを追加してください。")
    print(f"   → 右パネル: Add Data > Competition Data > {competition}")
else:
    data_files = list(RAW_DATA_DIR.glob("*.csv"))
    print(f"\\n✅ コンペデータ: {{len(data_files)}} CSV ファイル")
    for f in data_files[:5]:
        print(f"   {{f.name}}: {{f.stat().st_size / 1024:.1f}} KB")
"""

SUBMISSION_MODE_CELL = """\
# ============================================================
# 提出ファイルの確認と最終出力
# Notebook提出コンペ用: /kaggle/working/submission.csv に保存
# ============================================================
import shutil

# submission.csv を /kaggle/working/submission.csv にコピー
# (Notebookコンペでは output/submission.csv が自動的に提出物になる)
from src.config import SUBMISSIONS_DIR
import glob

sub_files = sorted(glob.glob(str(SUBMISSIONS_DIR / "sub_*.csv")))
if sub_files:
    latest = sub_files[-1]
    output_path = Path("/kaggle/working/submission.csv")
    shutil.copy(latest, output_path)
    print(f"✅ 提出ファイルを出力しました: {{output_path}}")
    print(f"   ソース: {{Path(latest).name}}")
else:
    print("⚠️  提出ファイルが見つかりません。スクリプトのエラーを確認してください。")
"""


def is_marimo_notebook(py_path: Path) -> bool:
    """marimo形式のノートブックかどうかを判定する。"""
    content = py_path.read_text(encoding="utf-8")
    return "import marimo" in content or "import marimo as mo" in content


def convert_via_marimo(py_path: Path, out_path: Path) -> bool:
    """marimo export ipynb を使って変換する。"""
    try:
        result = subprocess.run(
            ["uv", "run", "marimo", "export", "ipynb", str(py_path), "-o", str(out_path), "-f"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"marimo export エラー:\n{result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("uv または marimo が見つかりません。")
        return False


def build_ipynb_from_script(
    py_path: Path,
    out_path: Path,
    dataset_name: str,
    dataset_user: str,
    competition: str,
    target_col: str,
    submission_mode: bool,
) -> None:
    """通常の.pyスクリプトからKaggle実行用.ipynbを生成する。"""
    try:
        import nbformat
    except ImportError:
        print("nbformat が必要です: uv add nbformat")
        sys.exit(1)

    code = py_path.read_text(encoding="utf-8")

    # セットアップセル（コンペ名・ユーザー名・ターゲット列を埋め込む）
    setup_code = SETUP_CELL_TEMPLATE.format(
        dataset_name=dataset_name,
        dataset_user=dataset_user,
        competition=competition,
        target_col=target_col,
    )

    # 実験コードセル: sys.path.insert を除去し、from src.config import 後に設定値を注入
    cleaned_code = _clean_script_for_kaggle(code, py_path, dataset_name, competition, target_col)

    cells = [
        nbformat.v4.new_markdown_cell(
            f"# {py_path.stem}\n\n"
            f"**生成元**: `{py_path.relative_to(ROOT)}`  \n"
            f"**変換**: `scripts/to_kaggle_nb.py`  \n\n"
            f"> このノートブックは `to_kaggle_nb.py` によって自動生成されました。  \n"
            f"> 直接編集せず、元の `.py` ファイルを修正してから再変換してください。"
        ),
        nbformat.v4.new_code_cell(setup_code),
        nbformat.v4.new_code_cell(cleaned_code),
    ]

    if submission_mode:
        cells.append(nbformat.v4.new_code_cell(SUBMISSION_MODE_CELL))

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def _clean_script_for_kaggle(
    code: str, py_path: Path, dataset_name: str, competition: str, target_col: str
) -> str:
    """実験スクリプトをKaggle環境向けにクリーニングし、設定値を注入する。"""
    lines = code.splitlines()
    cleaned = []
    in_src_config_import = False
    injected = False

    for line in lines:
        # sys.path.insert でローカルパスを追加している行を除去
        if "sys.path.insert" in line and ("__file__" in line or "parent" in line):
            cleaned.append(f"# [Kaggle対応済み] {line}")
            continue
        # __file__ を Kaggle上の実際のパスに置換
        if "__file__" in line:
            kaggle_path = f"/kaggle/input/{dataset_name}/{py_path.relative_to(Path(__file__).resolve().parent.parent)}"
            line = line.replace("__file__", f'"{kaggle_path}"')

        # src.config の import ブロック検出
        if "from src.config import" in line:
            in_src_config_import = True
        cleaned.append(line)

        # import ブロック終了後に TARGET_COL / RAW_DATA_DIR を注入
        if in_src_config_import and not injected:
            is_block_end = ")" in line or ("from src.config import" in line and "(" not in line)
            if is_block_end:
                in_src_config_import = False
                injected = True
                cleaned.append(f'# [Kaggle注入] コンペ固有の設定値を上書き')
                cleaned.append(f'from pathlib import Path as _KagglePath')
                cleaned.append(f'TARGET_COL = "{target_col}"')
                cleaned.append(f'RAW_DATA_DIR = _KagglePath("/kaggle/input/competitions/{competition}")')

    return "\n".join(cleaned)


_GPU_INDICATORS = {
    "torch": "PyTorch を使用",
    "realmlp": "RealMLP を使用",
    "tabnet": "TabNet を使用",
    "lightning": "PyTorch Lightning を使用",
    "tensorflow": "TensorFlow を使用",
}
_GPU_OPTIONAL = {
    "xgboost": "XGBoost (GPU で高速化可能)",
    "cuml": "cuML を使用",
}


def detect_gpu_need(py_path: Path) -> tuple[bool, str]:
    """スクリプトの import からGPU必要性を自動判定する。

    Returns:
        (recommended, reason): 推奨値と理由
    """
    content = py_path.read_text(encoding="utf-8").lower()
    for keyword, reason in _GPU_INDICATORS.items():
        if keyword in content:
            return True, f"{reason} → GPU 推奨"
    for keyword, reason in _GPU_OPTIONAL.items():
        if keyword in content:
            return True, f"{reason} → GPU 推奨（任意）"
    return False, "GPU依存ライブラリなし → CPU で十分"


def generate_kernel_metadata(
    py_path: Path,
    out_dir: Path,
    competition: str,
    dataset_name: str,
    dataset_user: str,
    enable_gpu: bool,
) -> None:
    """kaggle kernels push 用の kernel-metadata.json を生成する。"""
    slug = py_path.stem.replace("_", "-").lower()
    metadata = {
        "id": f"{dataset_user}/{slug}",
        "title": py_path.stem,
        "code_file": f"{py_path.stem}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{dataset_user}/{dataset_name}"],
        "competition_sources": [competition],
        "kernel_sources": [],
    }
    meta_path = out_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"✅ kernel-metadata.json: {meta_path}")
    print(f"   → kaggle kernels push -p {out_dir} で Kaggle に送信できます。")


# ──────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="実験スクリプト → Kaggle Notebook (.ipynb) 変換")
    parser.add_argument("script", help="変換する .py ファイル")
    parser.add_argument("-o", "--output", default="",
                        help="出力 .ipynb パス（デフォルト: kaggle_nb/<script_name>.ipynb）")
    parser.add_argument("--dataset-name", default="",
                        help="Kaggle Dataset 名（デフォルト: src/config.py の COMPETITION）")
    parser.add_argument("--competition", default=COMPETITION,
                        help="コンペスラッグ（デフォルト: src/config.py の COMPETITION）")
    parser.add_argument("--submission-mode", action="store_true",
                        help="Notebook提出コンペ用: submission.csv を /kaggle/working/ に保存するセルを追加")
    parser.add_argument("--target-col", default="",
                        help="ターゲット列名（デフォルト: src/config.py の TARGET_COL）")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", action="store_true", default=None,
                           help="GPU を強制的に有効化する")
    gpu_group.add_argument("--no-gpu", action="store_true",
                           help="GPU を強制的に無効化する（自動判定を上書き）")
    parser.add_argument("--push", action="store_true",
                        help="変換後に kaggle kernels push を実行する")
    args = parser.parse_args()

    py_path = Path(args.script).resolve()
    if not py_path.exists():
        print(f"❌ ファイルが見つかりません: {py_path}")
        sys.exit(1)

    dataset_name = args.dataset_name or f"ds-template-{args.competition}"
    out_path = Path(args.output) if args.output else ROOT / "kaggle_nb" / f"{py_path.stem}.ipynb"

    # ターゲット列: 引数 > config.py の値
    from src.config import TARGET_COL as _default_target
    target_col = args.target_col or _default_target

    # Kaggle ユーザー名を kaggle config から自動取得
    try:
        _cfg_out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True)
        dataset_user = next(
            (l.split(":")[1].strip() for l in _cfg_out.stdout.splitlines() if "username" in l), ""
        )
    except Exception:
        dataset_user = ""
    if not dataset_user:
        dataset_user = input("Kaggle ユーザー名を入力してください: ").strip()

    # GPU 判定（自動 → 確認 → 上書き可能）
    gpu_recommended, gpu_reason = detect_gpu_need(py_path)
    if args.no_gpu:
        enable_gpu = False
        gpu_decision = "無効（--no-gpu で上書き）"
    elif args.gpu:
        enable_gpu = True
        gpu_decision = "有効（--gpu で上書き）"
    else:
        # 自動判定結果を表示して確認を求める
        print(f"\n[GPU 自動判定] {gpu_reason}")
        prompt = f"  GPU を{'有効' if gpu_recommended else '無効'}にします。変更しますか？ [y=変更/Enter=そのまま]: "
        answer = input(prompt).strip().lower()
        if answer == "y":
            enable_gpu = not gpu_recommended
        else:
            enable_gpu = gpu_recommended
        gpu_decision = f"{'有効' if enable_gpu else '無効'}（自動判定{'→変更' if answer == 'y' else ''}）"

    print(f"\n変換: {py_path.relative_to(ROOT)} → {out_path.relative_to(ROOT)}")
    print(f"  Dataset    : {dataset_user}/{dataset_name}")
    print(f"  Competition: {args.competition}")
    print(f"  Target col : {target_col}")
    print(f"  GPU        : {gpu_decision}")
    print(f"  Submission mode: {args.submission_mode}")

    # marimo形式かどうかで変換方法を切り替える
    if is_marimo_notebook(py_path):
        print("  形式: marimo notebook → marimo export ipynb を使用")
        success = convert_via_marimo(py_path, out_path)
        if not success:
            print("⚠️  marimo export 失敗。通常変換にフォールバックします。")
            build_ipynb_from_script(
                py_path, out_path, dataset_name, dataset_user, args.competition, target_col, args.submission_mode
            )
    else:
        print("  形式: 通常スクリプト → nbformat で変換")
        build_ipynb_from_script(
            py_path, out_path, dataset_name, dataset_user, args.competition, target_col, args.submission_mode
        )

    print(f"\n✅ 変換完了: {out_path}")

    # kernel-metadata.json の生成
    generate_kernel_metadata(py_path, out_path.parent, args.competition, dataset_name, dataset_user, enable_gpu)

    # 必要に応じて kaggle kernels push
    if args.push:
        print("\n🚀 Kaggle に push します...")
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(out_path.parent)],
            capture_output=False,
        )
        if result.returncode == 0:
            print("✅ Push 成功")
            print(f"   実行状況: kaggle kernels status {{username}}/{py_path.stem.replace('_', '-')}")
            print(f"   出力取得: kaggle kernels output {{username}}/{py_path.stem.replace('_', '-')} -p kaggle_nb/output/")
        else:
            print("❌ Push 失敗。kaggle API キーの設定を確認してください。")


if __name__ == "__main__":
    main()
