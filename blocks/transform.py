from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import model_validator

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    InsufficientInputs,
    ProgressBar,
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


def _parse_list_cell(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)

    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("value is null")

    text = str(value).strip()
    if not text:
        raise ValueError("value is empty")

    parsed: Any
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple, np.ndarray, pd.Series)):
        return list(parsed)

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [part.strip() for part in inner.split(",")]

    if "," in text:
        return [part.strip() for part in text.split(",")]

    raise ValueError("value is not a list-like sequence")


_NULL_TOKENS = {"", "na", "nan", "null", "none", "n/a", "<na>"}


def _normalize_nullable_scalar(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NULL_TOKENS:
            return np.nan
        return text
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    return value


_EMBEDDED_INT_PATTERN = re.compile(r"(\d+)")


def _true_numeric_sort_key(value: Any) -> tuple[tuple[int, Any], ...]:
    text = str(value)
    key: list[tuple[int, Any]] = []
    for part in _EMBEDDED_INT_PATTERN.split(text):
        if not part:
            continue
        if part.isdigit():
            key.append((1, int(part)))
            continue
        key.append((0, part.lower()))

    # Preserve a deterministic order for labels with equivalent numeric tokens.
    key.append((2, text))
    return tuple(key)


class TransposeMatrix(BaseBlock):
    name = "Transpose Matrix"
    version = "1.0.0"
    category = "Transform"
    description = "Transpose the input matrix (swap rows and columns)."
    input_labels = ["Matrix"]
    output_labels = ["Transposed Matrix"]

    class Params(BlockParams):
        pass

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        transposed = data.transpose()
        transposed.index = transposed.index.astype(str)
        transposed.columns = transposed.columns.astype(str)
        return BlockOutput(data=transposed)


class ReorderColumns(BaseBlock):
    name = "Reorder Columns"
    version = "1.0.0"
    category = "Transform"
    description = "Reorder columns according to a specified list of column names; columns not in the list are appended at the end in original order."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        column_order: str = ""

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("ReorderColumns requires params.")
        desired_order = _parse_columns(params.column_order)
        missing = [col for col in desired_order if col not in data.columns]
        if missing:
            raise BlockValidationError(f"ReorderColumns missing columns: {missing}")
        remaining = [col for col in data.columns if col not in desired_order]
        new_order = desired_order + remaining
        reordered = data[new_order].copy()
        return BlockOutput(data=reordered)


class MedianCenterRows(BaseBlock):
    name = "Median Center Rows"
    version = "1.0.0"
    category = "Transform"
    description = "Subtract each row's median from its numeric values."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        pass

    def validate(self, data: pd.DataFrame) -> None:
        numeric_cols = data.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise BlockValidationError(
                "MedianCenterRows requires at least one numeric column."
            )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        numeric = data.select_dtypes(include="number")
        medians = numeric.median(axis=1)
        centered = numeric.sub(medians, axis=0)
        result = data.copy()
        result[numeric.columns] = centered
        return BlockOutput(data=result)


class SortRows(BaseBlock):
    name = "Sort Rows"
    version = "1.0.0"
    category = "Transform"
    description = "Sort rows by values in a specified column."
    input_labels = ["DataFrame"]
    output_labels = ["Sorted DataFrame"]

    class Params(BlockParams):
        column: str
        ascending: bool = True

    def validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise BlockValidationError("SortRows requires DataFrame input.")

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("SortRows requires params.")
        if params.column not in data.columns:
            raise BlockValidationError(f"Column '{params.column}' does not exist.")
        sorted_df = data.sort_values(by=params.column, ascending=bool(params.ascending))
        return BlockOutput(data=sorted_df)


class FilterRows(BaseBlock):
    name = "Filter Rows"
    version = "1.0.1"
    category = "Transform"
    description = "Keep rows that satisfy a column comparison condition."
    input_labels = ["DataFrame"]
    output_labels = ["Filtered Rows"]
    param_descriptions = {
        "column": "Column to test for each row.",
        "operator": "Comparison operator. One of: eq, ne, gt, gte, lt, lte.",
        "value": "Value to compare against. Numeric-looking values are coerced when the selected column is numeric.",
    }

    class Params(BlockParams):
        column: str
        operator: str = "eq"
        value: Any = None

    _ops: dict[str, Callable[[Any, Any], Any]] = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
    }

    def validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise BlockValidationError("FilterRows requires DataFrame input.")

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("FilterRows requires params.")
        if params.column not in data.columns:
            raise BlockValidationError(f"Column '{params.column}' does not exist.")
        if params.operator not in self._ops:
            raise BlockValidationError(
                f"Unsupported operator '{params.operator}'. Valid: {sorted(self._ops)}"
            )

        column = data[params.column]
        compare_value = params.value

        # Cast comparison value to numeric only for numeric columns.
        if pd.api.types.is_numeric_dtype(column):
            try:
                compare_value = float(params.value)
            except (TypeError, ValueError):
                raise BlockValidationError(
                    f"Cannot convert value '{params.value}' to numeric for column '{params.column}'."
                )

        op = self._ops[params.operator]
        mask = op(column, compare_value)
        if isinstance(mask, pd.Series):
            mask = mask.fillna(False)
        filtered = data[mask].copy()
        return BlockOutput(data=filtered)


class DropNullRows(BaseBlock):
    name = "Drop Null Rows"
    version = "1.0.0"
    category = "Transform"
    description = "Drop rows when specified columns contain null values."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    usage_notes = [
        "When columns is blank, the block inspects every input column.",
        "Set how='any' to drop a row when any inspected column is null, or how='all' to drop only rows where every inspected column is null.",
    ]

    class Params(BlockParams):
        columns: str = block_param(
            "",
            description="Comma-separated columns to inspect. Leave blank to inspect all columns.",
            example="value_x,value_y",
        )
        how: str = block_param(
            "any",
            description="Drop mode. Use 'any' to drop rows with at least one null in the inspected columns, or 'all' to drop rows only when every inspected column is null.",
            example="any",
        )

    def validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise BlockValidationError("DropNullRows requires DataFrame input.")

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("DropNullRows requires params.")

        columns = _parse_columns(params.columns)
        if not columns:
            columns = [str(col) for col in data.columns]
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise BlockValidationError(f"Columns not found for null drop: {missing}")

        how = str(params.how or "any").strip().lower()
        if how not in {"any", "all"}:
            raise BlockValidationError("how must be one of: any, all.")

        null_mask = data[columns].isna()
        rows_to_drop = null_mask.any(axis=1) if how == "any" else null_mask.all(axis=1)
        filtered = data.loc[~rows_to_drop].copy()
        return BlockOutput(
            data=filtered,
            metadata={
                "columns_checked": columns,
                "how": how,
                "n_rows_dropped": int(rows_to_drop.sum()),
            },
        )


class DeduplicateRows(BaseBlock):
    name = "Deduplicate Rows"
    version = "1.0.0"
    category = "Transform"
    description = (
        "Drop duplicate rows based on key columns while preserving input order."
    )
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    usage_notes = [
        "Use SortRows upstream when the choice between duplicate rows should follow a deterministic ordering rule.",
        "The block preserves the current row order after duplicate removal.",
    ]

    class Params(BlockParams):
        key_columns: str = block_param(
            description="Comma-separated columns used to identify duplicate rows.",
            example="record_id,category",
        )
        keep: str = block_param(
            "first",
            description="Which duplicate row to keep. One of: first, last.",
            example="first",
        )

    def validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise BlockValidationError("DeduplicateRows requires DataFrame input.")

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("DeduplicateRows requires params.")

        key_columns = _parse_columns(params.key_columns)
        if not key_columns:
            raise BlockValidationError("key_columns is required.")
        missing = [col for col in key_columns if col not in data.columns]
        if missing:
            raise BlockValidationError(
                f"Columns not found for deduplication: {missing}"
            )

        keep = str(params.keep or "first").strip().lower()
        if keep not in {"first", "last"}:
            raise BlockValidationError("keep must be one of: first, last.")

        deduped = data.drop_duplicates(subset=key_columns, keep=keep).copy()  # pyright: ignore[reportArgumentType]
        return BlockOutput(
            data=deduped,
            metadata={
                "key_columns": key_columns,
                "keep": keep,
                "n_rows_removed": int(data.shape[0] - deduped.shape[0]),
            },
        )


class FilterColumns(BaseBlock):
    name = "Filter Columns"
    version = "1.0.0"
    category = "Transform"
    description = "Keep columns that satisfy a row comparison condition."
    input_labels = ["DataFrame"]
    output_labels = ["Filtered Columns"]
    param_descriptions = {
        "row_index": "Zero-based row index to inspect when filtering columns.",
        "operator": "Comparison operator. One of: eq, ne, gt, gte, lt, lte.",
        "value": "Value to compare against for the selected row.",
    }

    class Params(BlockParams):
        row_index: int = 0
        operator: str = "eq"
        value: Any = None

    _ops: dict[str, Callable[[Any, Any], Any]] = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
    }

    def validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise BlockValidationError("FilterColumns requires DataFrame input.")
        if data.empty:
            raise BlockValidationError(
                "FilterColumns requires non-empty DataFrame input."
            )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("FilterColumns requires params.")
        if params.row_index < 0 or params.row_index >= len(data):
            raise BlockValidationError(
                f"row_index {params.row_index} is out of bounds."
            )
        if params.operator not in self._ops:
            raise BlockValidationError(
                f"Unsupported operator '{params.operator}'. Valid: {sorted(self._ops)}"
            )

        op = self._ops[params.operator]
        row = data.iloc[params.row_index]
        mask = op(row, params.value)
        filtered = data.loc[:, mask].copy()
        return BlockOutput(data=filtered)  # type: ignore


class FilterColumnsByCoverage(BaseBlock):
    name = "Filter Columns By Coverage"
    version = "1.0.0"
    category = "Transform"
    description = (
        "Keep columns whose non-null fraction is at least the configured threshold."
    )
    input_labels = ["DataFrame"]
    output_labels = ["Filtered Columns"]

    class Params(BlockParams):
        min_fraction: float = 0.2
        numeric_only: bool = True

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("FilterColumnsByCoverage requires params.")

        threshold = float(params.min_fraction)
        if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
            raise BlockValidationError("min_fraction must be between 0 and 1.")

        source = (
            data.select_dtypes(include="number") if bool(params.numeric_only) else data
        )
        if source.shape[1] == 0:
            raise BlockValidationError("No columns available to evaluate coverage.")

        coverage = source.notna().mean(axis=0)
        keep = [str(col) for col in source.columns if float(coverage[col]) >= threshold]
        if not keep:
            raise BlockValidationError(
                f"No columns met min_fraction={threshold} coverage."
            )

        filtered = data.loc[:, keep].copy()
        return BlockOutput(
            data=filtered,
            metadata={
                "min_fraction": threshold,
                "n_columns_in": int(source.shape[1]),
                "n_columns_out": int(len(keep)),
            },
        )


class SelectColumnsByReference(BaseBlock):
    name = "Select Columns By Reference"
    version = "1.0.0"
    category = "Transform"
    description = "Keep columns from the input DataFrame that also exist in a reference DataFrame."
    n_inputs = 2
    input_labels = ["DataFrame", "Reference DataFrame"]
    output_labels = ["Selected DataFrame"]
    usage_notes = [
        "Input 0 provides the rows; Input 1 is used only to define which columns to keep.",
        "Use `include_columns` to preserve identifier or annotation columns that are not part of the reference feature set.",
        "When `preserve_input_order` is true, the output follows Input 0 column order; otherwise reference-column order is used after include_columns.",
    ]

    class Params(BlockParams):
        include_columns: str = block_param(
            "",
            description="Comma-separated columns from Input 0 to always keep.",
            example="foo,bar",
        )
        preserve_input_order: bool = block_param(
            True,
            description="If true, keep selected columns in Input 0 order. If false, append reference-matched columns in Input 1 order after include_columns.",
            example=True,
        )

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("SelectColumnsByReference requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("SelectColumnsByReference requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                "SelectColumnsByReference expects [data_df, reference_df]."
            )

        frame, reference = data
        if not isinstance(frame, pd.DataFrame) or not isinstance(
            reference, pd.DataFrame
        ):
            raise BlockValidationError("Both inputs must be DataFrames.")

        include_columns = _parse_columns(params.include_columns)
        missing_include = [col for col in include_columns if col not in frame.columns]
        if missing_include:
            raise BlockValidationError(
                f"Input 0 missing include_columns: {missing_include}"
            )

        reference_matches = [
            str(col)
            for col in reference.columns
            if str(col) in frame.columns and str(col) not in include_columns
        ]
        if not reference_matches:
            raise BlockValidationError(
                "No overlapping columns found between Input 0 and the reference DataFrame."
            )

        if bool(params.preserve_input_order):
            keep_set = set(include_columns) | set(reference_matches)
            keep_columns = [str(col) for col in frame.columns if str(col) in keep_set]
        else:
            keep_columns = include_columns + [
                col for col in reference_matches if col not in include_columns
            ]

        selected = frame.loc[:, keep_columns].copy()
        return BlockOutput(
            data=selected,
            metadata={
                "n_columns_in": int(frame.shape[1]),
                "n_reference_columns": int(reference.shape[1]),
                "n_reference_matches": int(len(reference_matches)),
                "n_columns_out": int(selected.shape[1]),
            },
        )


class ImputeMissingValues(BaseBlock):
    name = "Impute Missing Values"
    version = "1.0.0"
    category = "Transform"
    description = "Impute missing values in selected numeric columns using mean, median, or a constant."
    input_labels = ["DataFrame"]
    output_labels = ["Imputed DataFrame"]
    usage_notes = [
        "When `columns` is empty, the block imputes every numeric column in the input frame.",
        "Non-selected columns pass through unchanged.",
        "For `mean` and `median`, fully null columns fall back to `fill_value`.",
    ]

    class Params(BlockParams):
        columns: str = block_param(
            "",
            description="Comma-separated numeric columns to impute. Leave empty to use all numeric columns.",
            example="feature_a,feature_b",
        )
        strategy: str = block_param(
            "median",
            description="Imputation strategy. One of: mean, median, constant.",
            example="median",
        )
        fill_value: float = block_param(
            0.0,
            description="Fill value used for `constant` strategy and as fallback for fully null columns.",
            example=0.0,
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("ImputeMissingValues requires params.")

        strategy = str(params.strategy).strip().lower()
        if strategy not in {"mean", "median", "constant"}:
            raise BlockValidationError(
                "strategy must be one of: mean, median, constant."
            )

        selected_cols = _parse_columns(params.columns)
        if not selected_cols:
            selected_cols = data.select_dtypes(include="number").columns.tolist()
        if not selected_cols:
            raise BlockValidationError("No columns available to impute.")

        missing = [col for col in selected_cols if col not in data.columns]
        if missing:
            raise BlockValidationError(f"Impute columns not found: {missing}")

        result = data.copy()
        n_cells_filled = 0
        fill_values: dict[str, float] = {}
        fallback = float(params.fill_value)

        for col in selected_cols:
            original = result[col]
            numeric = pd.to_numeric(original, errors="coerce")
            if original.notna().any() and numeric.notna().sum() == 0:
                raise BlockValidationError(
                    f"Column '{col}' is not numeric and cannot be imputed."
                )

            fill_value = fallback
            if numeric.notna().any():
                if strategy == "mean":
                    candidate = float(numeric.mean(skipna=True))
                    if np.isfinite(candidate):
                        fill_value = candidate
                elif strategy == "median":
                    candidate = float(numeric.median(skipna=True))
                    if np.isfinite(candidate):
                        fill_value = candidate

            missing_mask = numeric.isna()
            n_cells_filled += int(missing_mask.sum())
            fill_values[str(col)] = float(fill_value)
            result[col] = numeric.fillna(fill_value)

        return BlockOutput(
            data=result,
            metadata={
                "strategy": strategy,
                "fill_value": fallback,
                "n_columns_imputed": int(len(selected_cols)),
                "n_cells_filled": int(n_cells_filled),
                "column_fill_values": fill_values,
            },
        )


class SelectColumns(BaseBlock):
    name = "Select Columns"
    version = "1.0.0"
    category = "Transform"
    description = "Select a subset of columns by name."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    presets = [
        {
            "id": "three_features",
            "label": "Three Features",
            "description": "Keep a compact trio of feature columns in output order.",
            "params": {"columns": "feature_a,feature_b,feature_c"},
        }
    ]

    class Params(BlockParams):
        columns: str = block_param(
            description="Comma-separated column names to keep, in output order.",
            example="feature_a,feature_b,feature_c",
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("SelectColumns requires params.")
        cols = _parse_columns(params.columns)
        missing = [col for col in cols if col not in data.columns]
        if missing:
            raise BlockValidationError(f"Columns not found: {missing}")
        selected = data[cols].copy()
        return BlockOutput(data=selected)


class SelectRows(BaseBlock):
    name = "Select Rows"
    version = "1.0.0"
    category = "Transform"
    description = "Select a subset of rows by index."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        row_indices: str = ""

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("SelectRows requires params.")
        indices = _parse_columns(params.row_indices)
        missing = [idx for idx in indices if idx not in data.index.astype(str)]
        if missing:
            raise BlockValidationError(f"Row indices not found: {missing}")
        selected = data.loc[indices].copy()
        return BlockOutput(data=selected)


class ResetIndex(BaseBlock):
    name = "Reset Index"
    version = "1.0.0"
    category = "Transform"
    description = "Reset DataFrame index into a column (or drop it)."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        drop: bool = False

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("ResetIndex requires params.")
        return BlockOutput(data=data.reset_index(drop=bool(params.drop)))


class MeltColumns(BaseBlock):
    name = "Melt Columns"
    version = "1.1.0"
    category = "Transform"
    description = "Unpivot selected value columns into a long-form table."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    usage_notes = [
        "When value_columns is blank, all columns not listed in id_columns are melted in input order.",
    ]

    class Params(BlockParams):
        id_columns: str = block_param(
            description="Comma-separated columns to preserve as identifier columns."
        )
        value_columns: str = block_param(
            "",
            description="Comma-separated columns to unpivot. Leave blank to melt all columns not listed in id_columns.",
        )
        variable_column: str = block_param(
            "variable",
            description="Output column name containing former column names.",
        )
        value_column: str = block_param(
            "value",
            description="Output column name containing melted values.",
        )
        drop_null_values: bool = block_param(
            False,
            description="If true, drop rows where melted value is null.",
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MeltColumns requires params.")

        id_columns = _parse_columns(params.id_columns)
        if not id_columns:
            raise BlockValidationError("id_columns is required.")

        value_columns = _parse_columns(params.value_columns)
        if not value_columns:
            value_columns = [col for col in data.columns if col not in id_columns]
        if not value_columns:
            raise BlockValidationError("value_columns resolved to no columns.")

        missing = [col for col in id_columns + value_columns if col not in data.columns]
        if missing:
            raise BlockValidationError(f"Columns not found for melt: {missing}")

        var_name = str(params.variable_column).strip() or "variable"
        val_name = str(params.value_column).strip() or "value"
        if var_name in data.columns and var_name not in value_columns:
            raise BlockValidationError(
                f"variable_column '{var_name}' conflicts with existing column."
            )
        if val_name in data.columns and val_name not in value_columns:
            raise BlockValidationError(
                f"value_column '{val_name}' conflicts with existing column."
            )

        melted = data.melt(
            id_vars=id_columns,
            value_vars=value_columns,
            var_name=var_name,
            value_name=val_name,
        )
        if bool(params.drop_null_values):
            melted = melted[melted[val_name].notna()].copy()

        return BlockOutput(
            data=melted,
            metadata={
                "id_columns": id_columns,
                "value_columns": value_columns,
                "variable_column": var_name,
                "value_column": val_name,
                "n_rows_out": int(melted.shape[0]),
            },
        )


class CastColumns(BaseBlock):
    name = "Cast Columns"
    version = "1.0.0"
    category = "Transform"
    description = "Cast selected columns to string or numeric dtypes."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        string_columns: str = ""
        numeric_columns: str = ""
        numeric_errors: str = "coerce"
        drop_invalid_numeric: bool = False

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("CastColumns requires params.")
        result = data.copy()

        string_cols = _parse_columns(params.string_columns)
        numeric_cols = _parse_columns(params.numeric_columns)

        missing = [
            col for col in string_cols + numeric_cols if col not in result.columns
        ]
        if missing:
            raise BlockValidationError(f"Columns not found for casting: {missing}")

        for col in ProgressBar(
            string_cols,
            label="Casting string columns",
            throttle_seconds=0.2,
        ):
            result[col] = result[col].astype(str)

        invalid_mask = pd.Series(False, index=result.index)
        for col in ProgressBar(
            numeric_cols,
            label="Casting numeric columns",
            throttle_seconds=0.2,
        ):
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise BlockValidationError(f"Column '{col}' is not numeric.")
            converted = pd.to_numeric(result[col], errors=params.numeric_errors)  # pyright: ignore[reportArgumentType, reportCallIssue]
            invalid_mask = invalid_mask | (converted.isna() & result[col].notna())
            result[col] = converted

        if params.drop_invalid_numeric and invalid_mask.any():
            result = result.loc[~invalid_mask].copy()
        return BlockOutput(data=result)


class SplitListColumn(BaseBlock):
    name = "Split List Column"
    version = "1.0.0"
    category = "Transform"
    description = "Split a list-like column (for example: '[0,1,4,2,3]') into multiple numeric columns."
    param_descriptions = {
        "column_name": "Name of the input column containing list-like values.",
        "column_name_prefix": "Prefix for generated output columns.",
        "starting_index": "Starting suffix index for generated columns.",
    }
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        column_name: str = block_param(
            description="Name of the input column containing list-like values.",
            example="embedding",
        )
        column_name_prefix: str = ""
        starting_index: int = 0

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("SplitListColumn requires params.")

        column_name = str(params.column_name).strip()
        if not column_name:
            raise BlockValidationError("column_name is required.")
        if column_name not in data.columns:
            raise BlockValidationError(f"Column '{column_name}' not found.")

        try:
            start_idx = int(params.starting_index)
        except Exception as exc:
            raise BlockValidationError("starting_index must be an integer.") from exc

        parsed_rows: list[list[Any]] = []
        expected_length: int | None = None
        for row_idx, value in enumerate(
            ProgressBar(
                data[column_name].tolist(),
                label=f"Splitting '{column_name}'",
                throttle_seconds=0.2,
            )
        ):
            try:
                row_values = _parse_list_cell(value)
            except Exception as exc:
                raise BlockValidationError(
                    f"Failed to parse row {row_idx} in column '{column_name}': {exc}."
                ) from exc

            if expected_length is None:
                expected_length = len(row_values)
                if expected_length <= 0:
                    raise BlockValidationError(
                        f"Column '{column_name}' contains empty list values."
                    )
            elif len(row_values) != expected_length:
                raise BlockValidationError(
                    f"Inconsistent list length in column '{column_name}' at row {row_idx}: "
                    f"expected {expected_length}, got {len(row_values)}."
                )

            parsed_rows.append(row_values)

        if expected_length is None:
            raise BlockValidationError(f"Column '{column_name}' is empty.")

        prefix = str(params.column_name_prefix)
        new_column_names = [f"{prefix}{start_idx + i}" for i in range(expected_length)]
        collisions = [name for name in new_column_names if name in data.columns]
        if collisions:
            raise BlockValidationError(
                f"Generated columns already exist: {collisions}."
            )

        split_df = pd.DataFrame(parsed_rows, index=data.index, columns=new_column_names)
        for col in split_df.columns:
            normalized = split_df[col].map(_normalize_nullable_scalar)
            converted = pd.to_numeric(normalized, errors="coerce")
            invalid = converted.isna() & normalized.notna()
            if invalid.any():
                first_bad = int(np.flatnonzero(invalid.to_numpy())[0])
                raise BlockValidationError(
                    f"Non-numeric value found in '{column_name}' for generated column '{col}' at row {first_bad}."
                )
            split_df[col] = converted

        result = data.copy()
        for col in new_column_names:
            result[col] = split_df[col]
        return BlockOutput(
            data=result,
            metadata={
                "source_column": column_name,
                "n_columns_added": int(expected_length),
                "column_name_prefix": prefix,
                "starting_index": int(start_idx),
            },
        )


class MaskOutliersMAD(BaseBlock):
    name = "Mask Outliers MAD"
    version = "1.0.0"
    category = "Transform"
    description = "Mask extreme values using robust MAD-based z-scores."
    input_labels = ["Matrix"]
    output_labels = ["Masked Matrix"]

    class Params(BlockParams):
        z_thresh: float = 10.0
        min_mad: float = 1e-6

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MaskOutliersMAD requires params.")
        med = data.median(axis=0, skipna=True)
        mad = (
            (data.sub(med, axis=1))
            .abs()
            .median(axis=0, skipna=True)
            .clip(lower=params.min_mad)
        )
        robust_z = (data.sub(med, axis=1)).abs().div(1.4826 * mad, axis=1)
        mask = robust_z > params.z_thresh
        masked = data.mask(mask)
        n_masked = int(np.nansum(mask.to_numpy(dtype=float)))
        return BlockOutput(
            data=masked,
            metadata={
                "masked_cells": n_masked,
                "mask_fraction": n_masked / max(masked.size, 1),
            },
        )


class ColumnMedianCenter(BaseBlock):
    name = "Column Median Center"
    version = "1.0.0"
    category = "Transform"
    description = "Subtract each column median from that column."
    input_labels = ["Matrix"]
    output_labels = ["Centered Matrix"]

    class Params(BlockParams):
        pass

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        medians = data.median(axis=0, skipna=True)
        centered = data.sub(medians, axis=1)
        return BlockOutput(
            data=centered, metadata={"column_medians": medians.to_dict()}
        )


class RowMeanCenter(BaseBlock):
    name = "Mean Center Rows"
    version = "1.0.0"
    category = "Transform"
    description = "Subtract each row mean from that row."
    input_labels = ["Matrix"]
    output_labels = ["Centered Matrix"]

    class Params(BlockParams):
        pass

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        row_mean = data.mean(axis=1, skipna=True)
        centered = data.sub(row_mean, axis=0)
        return BlockOutput(data=centered, metadata={"row_mean": row_mean.to_dict()})


class ZScoreNormalize(BaseBlock):
    name = "Z-Score Normalize"
    version = "1.0.0"
    category = "Transform"
    description = "Apply z-score normalization by columns (axis=0) or rows (axis=1)."
    input_labels = ["Matrix"]
    output_labels = ["Normalized Matrix"]
    param_descriptions = {
        "axis": "Axis to normalize across. Use 0 for column-wise normalization or 1 for row-wise normalization.",
    }

    class Params(BlockParams):
        axis: int = 0

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("ZScoreNormalize requires params.")
        axis = int(params.axis)
        if axis not in (0, 1):
            raise BlockValidationError("axis must be 0 (columns) or 1 (rows).")

        if axis == 0:
            normalized = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, 1.0)
        else:
            normalized = data.sub(data.mean(axis=1), axis=0).div(
                data.std(axis=1).replace(0, 1.0), axis=0
            )
        return BlockOutput(data=normalized)


class PivotTable(BaseBlock):
    name = "Pivot Table"
    version = "1.1.0"
    category = "Transform"
    description = "Create a pivot table from index/columns/value fields with an aggregation function."
    input_labels = ["DataFrame"]
    output_labels = ["Pivoted DataFrame"]
    param_descriptions = {
        "index": "Single column name to use for the output row index.",
        "columns": "Single column name whose values become output column labels.",
        "values": "Single column name providing the cell values.",
        "aggfunc": "Aggregation function used when multiple rows map to the same output cell.",
        "true_numeric_sorting": "When enabled, embedded numbers in output column labels are sorted numerically instead of lexicographically.",
    }
    usage_notes = [
        "Enable true_numeric_sorting to place labels like Group1__Step2 before Group1__Step10.",
    ]

    class Params(BlockParams):
        index: str = block_param(
            description="Single column name to use for the output row index."
        )
        columns: str = block_param(
            description="Single column name whose values become output column labels."
        )
        values: str = block_param(
            description="Single column name providing the cell values."
        )
        aggfunc: str = "mean"
        true_numeric_sorting: bool = False

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("PivotTable requires params.")
        index_cols = _parse_columns(params.index)
        column_cols = _parse_columns(params.columns)
        value_cols = _parse_columns(params.values)
        if len(index_cols) != 1:
            raise BlockValidationError(
                "PivotTable currently supports exactly one index column."
            )
        if len(column_cols) != 1:
            raise BlockValidationError(
                "PivotTable currently supports exactly one columns field."
            )
        if len(value_cols) != 1:
            raise BlockValidationError(
                "PivotTable currently supports exactly one values field."
            )

        required = {index_cols[0], column_cols[0], value_cols[0]}
        missing = required - set(data.columns)
        if missing:
            raise BlockValidationError(
                f"PivotTable missing required columns: {sorted(missing)}"
            )

        pivoted = data.pivot_table(
            index=index_cols[0],
            columns=column_cols[0],
            values=value_cols[0],
            aggfunc=params.aggfunc,  # pyright: ignore[reportArgumentType]
        )
        if pivoted is None or pivoted.empty:
            raise BlockValidationError("PivotTable produced an empty result.")
        pivoted.index = pivoted.index.astype(str)
        pivoted = pivoted.sort_index(axis=0)
        pivoted.columns = [str(col) for col in pivoted.columns]
        true_numeric_sorting = bool(getattr(params, "true_numeric_sorting", False))
        column_order = sorted(
            pivoted.columns,
            key=_true_numeric_sort_key if true_numeric_sorting else None,
        )
        pivoted = pivoted.reindex(column_order, axis=1)
        return BlockOutput(data=pivoted)


class FilterByLookupValues(BaseBlock):
    name = "Filter By Lookup Values"
    version = "1.1.0"
    category = "Transform"
    description = (
        "Filter rows in one table by whether key values exist in a second lookup table."
    )
    n_inputs = 2
    input_labels = ["Data", "Lookup Table"]
    output_labels = ["Filtered Data"]

    class Params(BlockParams):
        data_key: str = block_param(
            description="Column in the data input to match against lookup values.",
            example="foo",
        )
        lookup_key: str = block_param(
            description="Column in the lookup input that supplies allowed or excluded values.",
            example="bar",
        )
        lookup_filter_column: str | None = block_param(
            None,
            description="Optional lookup-table column used to pre-filter lookup rows before matching.",
            example="coverage",
        )
        lookup_filter_operator: str | None = block_param(
            None,
            description="Comparison operator for lookup_filter_column. Required only when lookup_filter_column is set.",
            example="gte",
        )
        lookup_filter_value: float | None = block_param(
            0.2,
            description="Comparison value for lookup_filter_column. Used only when lookup_filter_column is set.",
            example=0.2,
        )
        keep_matches: bool = block_param(
            True,
            description="If true, keep data rows whose key exists in the lookup set. If false, remove matching rows.",
            example=True,
        )

        @model_validator(mode="before")
        @classmethod
        def normalize_optional_lookup_filter(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data

            payload = dict(data)
            lookup_filter_column = str(
                payload.get("lookup_filter_column") or ""
            ).strip()
            lookup_filter_operator = str(
                payload.get("lookup_filter_operator") or ""
            ).strip()

            if not lookup_filter_column:
                payload["lookup_filter_column"] = None
                payload["lookup_filter_operator"] = None
                return payload

            payload["lookup_filter_column"] = lookup_filter_column
            if not lookup_filter_operator:
                raise ValueError(
                    "lookup_filter_operator is required when lookup_filter_column is set."
                )
            if payload.get("lookup_filter_value") is None:
                raise ValueError(
                    "lookup_filter_value is required when lookup_filter_column is set."
                )

            payload["lookup_filter_operator"] = lookup_filter_operator
            return payload

    _ops: dict[str, Callable[[Any, Any], Any]] = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
    }

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("FilterByLookupValues requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("FilterByLookupValues requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                "FilterByLookupValues expects [data_df, lookup_df]."
            )
        data_df, lookup_df = data
        if params.data_key not in data_df.columns:
            raise BlockValidationError(f"Data key '{params.data_key}' not found.")
        if params.lookup_key not in lookup_df.columns:
            raise BlockValidationError(f"Lookup key '{params.lookup_key}' not found.")

        filtered_lookup = lookup_df.copy()
        lookup_filter_column = params.lookup_filter_column

        if lookup_filter_column is not None:
            if lookup_filter_column not in filtered_lookup.columns:
                raise BlockValidationError(
                    f"Lookup filter column '{lookup_filter_column}' not found."
                )
            if params.lookup_filter_operator is None:
                raise BlockValidationError(
                    "lookup_filter_operator is required when lookup_filter_column is set."
                )
            operator_key = params.lookup_filter_operator
            if operator_key not in self._ops:
                raise BlockValidationError(
                    f"Unsupported lookup_filter_operator '{operator_key}'."
                )
            if params.lookup_filter_value is None:
                raise BlockValidationError(
                    "lookup_filter_value is required when lookup_filter_column is set."
                )
            op = self._ops[operator_key]
            mask = op(filtered_lookup[lookup_filter_column], params.lookup_filter_value)
            filtered_lookup = filtered_lookup[mask]

        values = set(filtered_lookup[params.lookup_key].dropna().tolist())
        if not values:
            out = data_df.iloc[0:0].copy() if params.keep_matches else data_df.copy()
            return BlockOutput(data=out, metadata={"lookup_value_count": 0})

        series = data_df[params.data_key]
        mask = series.isin(values)
        out = data_df[mask].copy() if params.keep_matches else data_df[~mask].copy()
        return BlockOutput(data=out, metadata={"lookup_value_count": int(len(values))})
