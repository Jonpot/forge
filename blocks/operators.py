from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    InsufficientInputs,
    block_param,
)


def _parse_columns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip().strip("'").strip('"') for part in text.split(",")]
        return [part for part in parts if part]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _normalize_null_handling(value: Any) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    if text == "ignore":
        return "ignore"
    if text == "drop":
        return "drop"
    if text in {"treat as 0", "treat as zero"}:
        return "treat_as_0"
    raise BlockValidationError(
        "null_handling must be one of: Drop, Ignore, Treat as 0."
    )


def _is_scalar_dataframe(frame: pd.DataFrame) -> bool:
    return frame.shape == (1, 1)


def _resolve_operand(
    df: pd.DataFrame,
    column_name: Any,
    *,
    input_number: int,
    result_index: pd.Index,
) -> tuple[list[pd.Series], list[str], str]:
    # Scalar mode: a 1x1 DataFrame acts as a constant for all result rows.
    if _is_scalar_dataframe(df):
        scalar_value = df.iloc[0, 0]
        scalar_col = str(df.columns[0])
        return (
            [pd.Series([scalar_value] * len(result_index), index=result_index)],
            [scalar_col],
            "scalar",
        )

    cols = _parse_columns(column_name)
    if not cols:
        if df.shape[1] == 1:
            cols = [str(df.columns[0])]
        else:
            raise BlockValidationError(
                f"input_{input_number}_column_name is required when Input {input_number} has multiple columns."
            )
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise BlockValidationError(f"Input {input_number} missing columns: {missing}")

    series_list: list[pd.Series] = []
    for col in cols:
        series = df[col].copy().reset_index(drop=True)
        series.index = result_index
        series_list.append(series)
    return series_list, cols, "dataframe"


class _BinaryColumnOperatorMixin:
    category = "Operator"
    n_inputs = 2
    input_labels = ["Input 1 DataFrame or Scalar", "Input 2 DataFrame or Scalar"]
    output_labels = ["DataFrame"]

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs(f"{self.__class__.__name__} requires both inputs.")

    param_descriptions = {
        "input_1_column_name": "One or more Input 1 columns (comma-separated), optional when Input 1 is scalar or has one column.",
        "input_2_column_name": "One or more Input 2 columns (comma-separated), optional when Input 2 is scalar or has one column.",
        "null_handling": "One of: Drop | Ignore | Treat as 0.",
        "output_column": "Single output column name, comma-separated output names, or prefix/template for multi-output (use '{column}' placeholder).",
    }

    class Params(BlockParams):
        input_1_column_name: str = ""
        input_2_column_name: str = ""
        null_handling: str = "Ignore"
        output_column: str = "result"

    def _operate(self, left: pd.Series, right: pd.Series) -> pd.Series:
        raise NotImplementedError

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError(f"{self.__class__.__name__} requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                f"{self.__class__.__name__} expects [input_1_df, input_2_df]."
            )
        left_df, right_df = data
        if not isinstance(left_df, pd.DataFrame) or not isinstance(
            right_df, pd.DataFrame
        ):
            raise BlockValidationError("Both inputs must be DataFrames.")

        left_is_scalar = _is_scalar_dataframe(left_df)
        right_is_scalar = _is_scalar_dataframe(right_df)
        if not left_is_scalar and not right_is_scalar and len(left_df) != len(right_df):
            raise BlockValidationError(
                "Input 2 must be either a 1x1 scalar DataFrame or have the same number of rows as Input 1."
            )

        if not left_is_scalar:
            result_index = left_df.index
        elif not right_is_scalar:
            result_index = right_df.index
        else:
            result_index = left_df.index

        output_cols_raw = _parse_columns(params.output_column)
        output_text = str(params.output_column).strip()
        if not output_cols_raw:
            raise BlockValidationError("output_column is required.")

        left_series_list, input_1_cols, input_1_mode = _resolve_operand(
            left_df,
            params.input_1_column_name,
            input_number=1,
            result_index=result_index,
        )
        right_series_list, input_2_cols, input_2_mode = _resolve_operand(
            right_df,
            params.input_2_column_name,
            input_number=2,
            result_index=result_index,
        )

        n_left = len(left_series_list)
        n_right = len(right_series_list)
        pairs: list[tuple[pd.Series, pd.Series]] = []
        varying_labels: list[str] = []
        if input_1_mode == "scalar" and input_2_mode == "scalar":
            pairs = [(left_series_list[0], right_series_list[0])]
            varying_labels = input_1_cols
            mapping_mode = "1_to_1"
        elif input_2_mode == "scalar":
            scalar_series = right_series_list[0]
            pairs = [(left_series, scalar_series) for left_series in left_series_list]
            varying_labels = input_1_cols
            mapping_mode = "n_to_1"
        elif input_1_mode == "scalar":
            scalar_series = left_series_list[0]
            pairs = [
                (scalar_series, right_series) for right_series in right_series_list
            ]
            varying_labels = input_2_cols
            mapping_mode = "1_to_n" if n_right > 1 else "1_to_1"
        elif n_left == n_right:
            pairs = list(zip(left_series_list, right_series_list))
            varying_labels = input_1_cols
            mapping_mode = "n_to_n" if n_left > 1 else "1_to_1"
        elif n_left == 1 and n_right > 1:
            pairs = [(left_series_list[0], s) for s in right_series_list]
            varying_labels = input_2_cols
            mapping_mode = "1_to_n"
        elif n_left > 1 and n_right == 1:
            pairs = [
                (left_series, right_series_list[0]) for left_series in left_series_list
            ]
            varying_labels = input_1_cols
            mapping_mode = "n_to_1"
        else:
            raise BlockValidationError(
                "Unsupported column mapping. Use n-to-1, 1-to-n, or n-to-n."
            )

        n_pairs = len(pairs)
        if n_pairs == 1:
            if len(output_cols_raw) != 1:
                raise BlockValidationError(
                    "Single-column operation requires exactly one output column name."
                )
            output_columns = [output_cols_raw[0]]
        elif len(output_cols_raw) == n_pairs:
            output_columns = output_cols_raw
        elif len(output_cols_raw) == 1:
            pattern = output_cols_raw[0]
            if "{column}" in pattern:
                output_columns = [
                    pattern.format(column=label) for label in varying_labels
                ]
            else:
                output_columns = [f"{pattern}{label}" for label in varying_labels]
        else:
            raise BlockValidationError(
                f"output_column must provide 1 or {n_pairs} names for this mapping."
            )

        null_handling = _normalize_null_handling(params.null_handling)
        keep_mask = pd.Series(True, index=result_index)
        computed: dict[str, pd.Series] = {}
        for (left_raw, right_raw), out_col in zip(pairs, output_columns):
            left = pd.to_numeric(left_raw, errors="coerce")
            right = pd.to_numeric(right_raw, errors="coerce")

            if null_handling == "drop":
                keep_mask = keep_mask & left.notna() & right.notna()
            elif null_handling == "treat_as_0":
                left = left.fillna(0.0)
                right = right.fillna(0.0)

            with np.errstate(divide="ignore", invalid="ignore"):
                result = self._operate(left, right)
            result_series = pd.Series(result, index=result_index, dtype="float64")
            result_series = result_series.where(np.isfinite(result_series), np.nan)
            computed[out_col] = result_series

        if input_1_mode != "scalar":
            out = left_df.copy()
        elif input_2_mode != "scalar":
            out = right_df.copy()
        else:
            out = left_df.copy()
        if null_handling == "drop":
            out = out.loc[keep_mask].copy()
            for out_col, series in computed.items():
                out[out_col] = series.loc[keep_mask]
        else:
            for out_col, series in computed.items():
                out[out_col] = series

        return BlockOutput(
            data=out,
            metadata={
                "operation": self.__class__.__name__,
                "input_1_column_name": str(params.input_1_column_name),
                "input_1_columns": input_1_cols,
                "input_1_mode": input_1_mode,
                "input_2_column_name": str(params.input_2_column_name),
                "input_2_columns": input_2_cols,
                "input_2_mode": input_2_mode,
                "mapping_mode": mapping_mode,
                "null_handling": null_handling,
                "output_column": output_text,
                "output_columns": output_columns,
            },
        )


class AddColumns(_BinaryColumnOperatorMixin, BaseBlock):
    name = "Add Columns"
    version = "1.0.1"
    description = "Element-wise addition: Input 1 column + Input 2 column or scalar."

    def _operate(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left + right


class SubtractColumns(_BinaryColumnOperatorMixin, BaseBlock):
    name = "Subtract Columns"
    version = "1.0.1"
    description = "Element-wise subtraction: Input 1 column - Input 2 column or scalar."

    def _operate(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left - right


class MultiplyColumnsOperator(_BinaryColumnOperatorMixin, BaseBlock):
    name = "Multiply Columns"
    version = "1.0.1"
    description = (
        "Element-wise multiplication: Input 1 column * Input 2 column or scalar."
    )

    def _operate(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left * right


class AbsoluteValueColumn(BaseBlock):
    name = "Absolute Value Column"
    version = "1.0.0"
    category = "Operator"
    description = "Compute the absolute value of one numeric column."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    usage_notes = [
        "The source column is coerced to numeric before taking the absolute value.",
        "Rows are preserved; values that cannot be coerced become NaN in the output column.",
        "Set output_column equal to source_column to overwrite the original values.",
    ]

    class Params(BlockParams):
        source_column: str = block_param(
            description="Numeric column whose absolute value will be computed.",
            example="coefficient",
        )
        output_column: str = block_param(
            "abs_value",
            description="Output column that will receive the absolute values.",
            example="abs_coefficient",
        )

    def validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise BlockValidationError("AbsoluteValueColumn expects a DataFrame input.")

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("AbsoluteValueColumn requires params.")
        self.validate(data)
        if params.source_column not in data.columns:
            raise BlockValidationError(
                f"AbsoluteValueColumn missing column: '{params.source_column}'."
            )

        frame = data.copy()
        frame[params.output_column] = pd.to_numeric(
            frame[params.source_column], errors="coerce"
        ).abs()
        return BlockOutput(
            data=frame,
            metadata={
                "source_column": params.source_column,
                "output_column": params.output_column,
            },
        )


class MultiplyColumns(BaseBlock):
    name = "Multiply Many Columns"
    version = "1.1.0"
    category = "Operator"
    description = "Multiply a set of numeric columns row-wise into a new output column."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        columns: str = "factor_a,factor_b"
        output_column: str = "product_value"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MultiplyColumns requires params.")
        cols = _parse_columns(params.columns)
        if not cols:
            raise BlockValidationError("MultiplyColumns requires at least one column.")
        missing = [col for col in cols if col not in data.columns]
        if missing:
            raise BlockValidationError(f"MultiplyColumns missing columns: {missing}")
        frame = data.copy()
        matrix = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(1.0)
        frame[params.output_column] = matrix.prod(axis=1)
        return BlockOutput(data=frame)


class MultiplyDataFrames(BaseBlock):
    name = "Multiply DataFrames"
    version = "1.0.0"
    category = "Operator"
    description = (
        "Multiply two numeric DataFrames element-wise, aligning by index and columns."
    )
    n_inputs = 2
    input_labels = ["DataFrame 1", "DataFrame 2"]
    output_labels = ["Product DataFrame"]

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("MultiplyDataFrames requires both inputs.")

    def execute(self, data: list[pd.DataFrame], params: None = None) -> BlockOutput:
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError("MultiplyDataFrames expects exactly 2 inputs.")
        df1, df2 = data
        aligned1, aligned2 = df1.align(df2, join="outer", axis=None)
        product = aligned1.multiply(aligned2)
        return BlockOutput(data=product)


class DivideColumns(_BinaryColumnOperatorMixin, BaseBlock):
    name = "Divide Columns"
    version = "1.0.1"
    description = (
        "Element-wise division: Input 1 column or scalar / Input 2 column or scalar."
    )

    def _operate(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left / right


class LogColumns(_BinaryColumnOperatorMixin, BaseBlock):
    name = "Log Columns"
    version = "1.0.0"
    description = "Element-wise logarithm: log(Input 1) in base (Input 2)."

    def _operate(self, left: pd.Series, right: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.log(left) / np.log(right)
        return pd.Series(result, index=left.index, dtype="float64")
