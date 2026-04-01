"""
特徴量レジストリ — 段階的特徴量追加実験のための管理モジュール

各特徴量を種別（numerical/categorical/engineered）とステップで管理し、
累積特徴量セットの構築と実験シーケンスの生成をサポートする。

ステップ定義:
  1: 数値カラム（ベースライン1列）
  2: 数値カラム（追加）
  3: カテゴリカラム
  4: 数値×数値 FE
  5: 数値×カテゴリ FE
  6: カテゴリ×カテゴリ FE
"""

from dataclasses import dataclass, field
from typing import Literal

FeatureType = Literal["numerical", "categorical", "num_x_num", "num_x_cat", "cat_x_cat", "other"]

STEP_LABELS = {
    1: "Step1: 数値ベースライン",
    2: "Step2: 数値追加",
    3: "Step3: カテゴリ追加",
    4: "Step4: 数値×数値 FE",
    5: "Step5: 数値×カテゴリ FE",
    6: "Step6: カテゴリ×カテゴリ FE",
}


@dataclass
class FeatureEntry:
    name: str
    feature_type: FeatureType
    step: int
    description: str = ""


class FeatureRegistry:
    """特徴量レジストリ。登録順が実験の実行順序になる。"""

    def __init__(self) -> None:
        self._entries: list[FeatureEntry] = []

    def register(
        self,
        name: str,
        feature_type: FeatureType,
        step: int,
        description: str = "",
    ) -> "FeatureRegistry":
        """特徴量を1件登録する。メソッドチェーン対応。"""
        self._entries.append(
            FeatureEntry(name=name, feature_type=feature_type, step=step, description=description)
        )
        return self

    def register_many(
        self,
        names: list[str],
        feature_type: FeatureType,
        step: int,
    ) -> "FeatureRegistry":
        """複数特徴量を同一ステップ・種別で一括登録する。"""
        for name in names:
            self.register(name, feature_type, step)
        return self

    def build_incremental_sequence(self) -> list[tuple[str, list[str], FeatureType]]:
        """
        1列ずつ追加する実験シーケンスを構築する。

        Returns:
            list of (feature_name, cumulative_feature_list, feature_type)
            cumulative_feature_list はその時点までの全特徴量を含む。
        """
        sequence: list[tuple[str, list[str], FeatureType]] = []
        cumulative: list[str] = []
        for entry in self._entries:
            cumulative.append(entry.name)
            sequence.append((entry.name, list(cumulative), entry.feature_type))
        return sequence

    def get_step(self, step: int) -> list[str]:
        """指定ステップの特徴量名リストを返す。"""
        return [e.name for e in self._entries if e.step == step]

    def get_cumulative(self, max_step: int) -> list[str]:
        """ステップ1〜max_stepまでの累積特徴量リストを返す。"""
        return [e.name for e in self._entries if e.step <= max_step]

    def summary(self) -> "pd.DataFrame":
        """登録済み特徴量の一覧をDataFrameで返す。"""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "step": e.step,
                    "step_label": STEP_LABELS.get(e.step, f"Step{e.step}"),
                    "name": e.name,
                    "type": e.feature_type,
                    "description": e.description,
                }
                for e in self._entries
            ]
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._entries)} features)"
