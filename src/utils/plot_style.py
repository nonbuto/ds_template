"""matplotlib の共通スタイル設定（日本語フォント・保存パス）。

`uv` 管理の venv には CJK フォントが含まれないため、matplotlib のデフォルト設定のままだと
日本語のラベル・タイトルが tofu（□□□）に化ける。可視化スクリプトの冒頭で
`setup_japanese_font()` を呼ぶこと（→ `CONVENTIONS.md#可視化の規約`）。

使い方:
    import matplotlib
    matplotlib.use("Agg")
    from src.utils.plot_style import setup_japanese_font, plot_path

    setup_japanese_font()
    fig.savefig(plot_path(exp_id="042", seq=1, name="importance_top30"))
"""

from __future__ import annotations

import warnings
from pathlib import Path

# 探索順: 明示インストール(japanize-matplotlib) → OS 同梱の CJK フォント
_CJK_CANDIDATES = [
    "IPAexGothic",
    "Noto Sans CJK JP",
    "Hiragino Sans",          # macOS 標準
    "Hiragino Maru Gothic Pro",
    "Yu Gothic",              # Windows 標準
    "Meiryo",
    "TakaoGothic",            # Linux (ipafont-gothic)
]

_configured = False


def setup_japanese_font(verbose: bool = False) -> str | None:
    """日本語を描画できるフォントを matplotlib に設定する。

    Returns:
        設定できたフォント名。見つからなければ None（警告を出して英語表記を推奨）。
    """
    global _configured
    import matplotlib
    from matplotlib import font_manager

    if _configured:
        return matplotlib.rcParams["font.family"][0]

    # japanize-matplotlib が入っていれば import するだけで設定が完了する
    try:
        import japanize_matplotlib  # noqa: F401

        _configured = True
        if verbose:
            print("[plot_style] japanize-matplotlib を使用")
        return matplotlib.rcParams["font.family"][0]
    except ImportError:
        pass

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.family"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の tofu を防ぐ
            _configured = True
            if verbose:
                print(f"[plot_style] 日本語フォント: {name}")
            return name

    warnings.warn(
        "日本語フォントが見つかりませんでした。図のラベルは英語で書くか、"
        "`uv add japanize-matplotlib` を実行してください。",
        stacklevel=2,
    )
    return None


def plot_path(name: str, exp_id: str | None = None, seq: int | None = None) -> Path:
    """`CONVENTIONS.md#可視化の規約` の命名規則に従った保存パスを返す。

    Args:
        name: 図の内容（例: "importance_top30"）
        exp_id: 実験 ID 3 桁（例: "042"）。EDA 段階など未確定なら省略する
        seq: その実験内の通し番号。省略時は 1

    Returns:
        `{exp_id}_{seq:02d}_{name}.png`（exp_id 省略時は `eda_{name}.png`）のフルパス
    """
    from src.config import PLOTS_DIR

    if exp_id is None:
        filename = f"eda_{name}.png"
    else:
        filename = f"{exp_id}_{(seq or 1):02d}_{name}.png"
    return PLOTS_DIR / filename
