"""
データバリデーションモジュール

パイプラインの各ステージで呼び出し、データ品質を検証する。
サイレントなデータバグ（リーク・型不一致・予期せぬ欠損）を早期発見する。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """バリデーション結果。passed=False の場合はメッセージを確認すること。"""

    passed: bool
    messages: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [status]
        for msg in self.messages:
            lines.append(f"  - {msg}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """バリデーション失敗時に例外を発生させる。"""
        if not self.passed:
            raise ValueError(f"データバリデーション失敗:\n{self}")


def validate_schema(
    df: pd.DataFrame,
    expected_columns: list[str],
    expected_dtypes: Optional[dict[str, str]] = None,
) -> ValidationResult:
    """カラム存在確認とデータ型チェック。

    Args:
        df: 検証対象のDataFrame
        expected_columns: 必須カラムのリスト
        expected_dtypes: {カラム名: dtype文字列} の辞書（任意）

    Returns:
        ValidationResult
    """
    messages: list[str] = []
    passed = True

    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        messages.append(f"欠損カラム: {missing_cols}")
        passed = False

    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if expected_dtype not in actual:
                messages.append(f"{col}: dtype={actual}（期待値: {expected_dtype}）")
                passed = False

    if passed:
        messages.append(f"スキーマ検証OK: {len(expected_columns)}カラム確認")

    return ValidationResult(passed=passed, messages=messages)


def validate_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> ValidationResult:
    """テストデータにターゲット情報がリークしていないことを検証。

    Args:
        train_df: 訓練データ
        test_df: テストデータ
        target_col: ターゲットカラム名

    Returns:
        ValidationResult
    """
    messages: list[str] = []
    passed = True

    if target_col in test_df.columns:
        messages.append(f"テストデータにターゲットカラム '{target_col}' が含まれています（リーク可能性）")
        passed = False

    train_cols = set(train_df.columns) - {target_col}
    test_cols = set(test_df.columns)
    only_in_train = train_cols - test_cols
    if only_in_train:
        messages.append(f"訓練データにのみ存在するカラム（特徴量エンジニアリングの一貫性を確認）: {only_in_train}")

    if passed:
        messages.append("リーク検証OK: テストデータにターゲットカラムなし")

    return ValidationResult(passed=passed, messages=messages)


def validate_missing_values(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.5,
    columns: Optional[list[str]] = None,
) -> ValidationResult:
    """欠損値の割合が閾値を超えていないことを検証。

    Args:
        df: 検証対象のDataFrame
        max_missing_ratio: 許容する最大欠損率（デフォルト: 50%）
        columns: 検証対象カラム（Noneの場合は全カラム）

    Returns:
        ValidationResult
    """
    messages: list[str] = []
    passed = True

    target_cols = columns if columns else df.columns.tolist()
    missing_ratios = df[target_cols].isnull().mean()
    over_threshold = missing_ratios[missing_ratios > max_missing_ratio]

    if len(over_threshold) > 0:
        for col, ratio in over_threshold.items():
            messages.append(f"{col}: 欠損率={ratio:.1%}（閾値: {max_missing_ratio:.0%}）")
        passed = False
    else:
        messages.append(f"欠損値検証OK: 全カラムで欠損率 < {max_missing_ratio:.0%}")

    return ValidationResult(passed=passed, messages=messages)


def validate_class_balance(
    df: pd.DataFrame,
    target_col: str,
    min_minority_ratio: float = 0.05,
) -> ValidationResult:
    """クラスバランスが極端に偏っていないことを検証（分類タスク用）。

    Args:
        df: 検証対象のDataFrame
        target_col: ターゲットカラム名
        min_minority_ratio: 最小クラスの最低比率（デフォルト: 5%）

    Returns:
        ValidationResult
    """
    messages: list[str] = []
    passed = True

    if target_col not in df.columns:
        return ValidationResult(passed=False, messages=[f"カラム '{target_col}' が存在しません"])

    value_counts = df[target_col].value_counts(normalize=True)
    minority_ratio = value_counts.min()

    class_dist = {str(k): f"{v:.1%}" for k, v in value_counts.items()}
    messages.append(f"クラス分布: {class_dist}")

    if minority_ratio < min_minority_ratio:
        messages.append(
            f"少数クラスの比率が低すぎます: {minority_ratio:.1%}（閾値: {min_minority_ratio:.0%}）"
        )
        passed = False
    else:
        messages.append(f"クラスバランス検証OK: 最小クラス比率={minority_ratio:.1%}")

    return ValidationResult(passed=passed, messages=messages)


def validate_feature_ranges(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> ValidationResult:
    """特徴量の値域が期待範囲内であることを検証。

    Args:
        df: 検証対象のDataFrame
        bounds: {カラム名: (min値, max値)} の辞書

    Returns:
        ValidationResult
    """
    messages: list[str] = []
    passed = True

    for col, (expected_min, expected_max) in bounds.items():
        if col not in df.columns:
            messages.append(f"カラム '{col}' が存在しません")
            passed = False
            continue

        actual_min = df[col].min()
        actual_max = df[col].max()

        if actual_min < expected_min or actual_max > expected_max:
            messages.append(
                f"{col}: 値域=[{actual_min}, {actual_max}]（期待値: [{expected_min}, {expected_max}]）"
            )
            passed = False

    if passed:
        messages.append(f"値域検証OK: {len(bounds)}カラム確認")

    return ValidationResult(passed=passed, messages=messages)


def run_all_validations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    expected_columns: list[str],
    raise_on_failure: bool = False,
) -> dict[str, ValidationResult]:
    """標準バリデーションをまとめて実行する。

    Args:
        train_df: 訓練データ
        test_df: テストデータ
        target_col: ターゲットカラム名
        expected_columns: 期待するカラムリスト
        raise_on_failure: Trueの場合、失敗時に例外を発生させる

    Returns:
        {バリデーション名: ValidationResult} の辞書
    """
    results: dict[str, ValidationResult] = {}

    results["schema_train"] = validate_schema(train_df, expected_columns)
    results["schema_test"] = validate_schema(test_df, [c for c in expected_columns if c != target_col])
    results["no_leakage"] = validate_no_leakage(train_df, test_df, target_col)
    results["missing_train"] = validate_missing_values(train_df)
    results["missing_test"] = validate_missing_values(test_df)

    print("=== データバリデーション結果 ===")
    for name, result in results.items():
        status = "✅" if result.passed else "❌"
        print(f"{status} {name}")
        for msg in result.messages:
            print(f"   {msg}")

    if raise_on_failure:
        for result in results.values():
            result.raise_if_failed()

    return results
