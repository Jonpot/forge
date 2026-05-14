from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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


def _exp_penalty(series: pd.Series, clip_hi: float, scale: float) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(x)
    if not np.isfinite(med):
        med = 0.0
    x = x.fillna(med).clip(lower=0.0, upper=clip_hi).astype(float).to_numpy()
    return np.exp(-((x / max(scale, 1e-12)) ** 2))


def _parse_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    out: list[float] = []
    for part in parts:
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    return out


def _resolve_numeric_columns(
    data: pd.DataFrame,
    raw_columns: Any,
    *,
    block_name: str,
) -> list[str]:
    cols = _parse_columns(raw_columns)
    if cols:
        missing = [col for col in cols if col not in data.columns]
        if missing:
            raise BlockValidationError(f"{block_name} missing columns: {missing}")
        return cols

    numeric_columns = [
        str(col) for col in data.select_dtypes(include=[np.number]).columns.tolist()
    ]
    if not numeric_columns:
        raise BlockValidationError(
            f"{block_name} requires at least one numeric column when columns is blank."
        )
    return numeric_columns


def _parse_json_spec_list(
    value: Any,
    *,
    block_name: str,
    param_name: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        specs: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise BlockValidationError(
                    f"{block_name} requires every {param_name} item to be an object."
                )
            specs.append(dict(item))
        return specs

    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise BlockValidationError(
            f"{block_name} could not parse {param_name} as JSON."
        ) from exc

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return [dict(item) for item in parsed]
    raise BlockValidationError(
        f"{block_name} expects {param_name} to be a JSON object or list of objects."
    )


def _weighted_corr_and_pvalue(
    x: pd.Series,
    y: pd.Series,
    w: pd.Series | None = None,
) -> tuple[float, float, float]:
    x_vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    if w is None:
        w_vals = np.ones_like(x_vals, dtype=float)
    else:
        w_vals = pd.to_numeric(w, errors="coerce").to_numpy(dtype=float)

    mask = (
        np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(w_vals) & (w_vals > 0)
    )
    if int(mask.sum()) < 4:
        return float("nan"), float("nan"), float(mask.sum())

    xv = x_vals[mask]
    yv = y_vals[mask]
    wv = w_vals[mask]
    sum_w = float(np.sum(wv))
    if not np.isfinite(sum_w) or sum_w <= 0:
        return float("nan"), float("nan"), float(mask.sum())

    mx = float(np.sum(wv * xv) / sum_w)
    my = float(np.sum(wv * yv) / sum_w)
    dx = xv - mx
    dy = yv - my
    vx = float(np.sum(wv * dx * dx) / sum_w)
    vy = float(np.sum(wv * dy * dy) / sum_w)
    if vx <= 0 or vy <= 0:
        return float("nan"), float("nan"), float(mask.sum())

    cov = float(np.sum(wv * dx * dy) / sum_w)
    corr = float(cov / math.sqrt(vx * vy))
    corr = float(np.clip(corr, -0.999999, 0.999999))

    n_eff = (sum_w * sum_w) / max(float(np.sum(wv * wv)), 1e-12)
    if n_eff <= 3:
        return corr, float("nan"), n_eff
    fisher_z = (
        0.5 * math.log((1.0 + corr) / (1.0 - corr)) * math.sqrt(max(n_eff - 3.0, 1e-12))
    )
    p_value = math.erfc(abs(fisher_z) / math.sqrt(2.0))
    return corr, float(p_value), float(n_eff)


class GroupAggregate(BaseBlock):
    name = "Group Aggregate"
    version = "1.0.0"
    category = "Statistics"
    description = (
        "Group rows by key columns and compute one or more summary aggregations."
    )
    input_labels = ["DataFrame"]
    output_labels = ["Aggregated DataFrame"]
    usage_notes = [
        "Aggregation specs are provided as JSON objects with keys: source, agg, output.",
        "Supported agg values are: size, count, nunique, mean, std, min, max.",
        "Use source='*' or leave source blank when agg='size'.",
        "Numeric aggregations coerce the source column to numeric and ignore non-numeric values as nulls.",
    ]

    class Params(BlockParams):
        group_columns: str = block_param(
            description="Comma-separated columns used to define each group.",
            example="segment",
        )
        aggregations: str = block_param(
            description="JSON object or JSON list of objects describing each aggregation. Each object must contain agg and output, and usually source.",
            example='[{"source":"*","agg":"size","output":"row_count"},{"source":"score","agg":"std","output":"score_std"}]',
        )

    _allowed_aggs = {"size", "count", "nunique", "mean", "std", "min", "max"}
    _numeric_aggs = {"mean", "std", "min", "max"}

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("GroupAggregate requires params.")

        group_columns = _parse_columns(params.group_columns)
        if not group_columns:
            raise BlockValidationError("group_columns is required.")
        missing_group_columns = [
            col for col in group_columns if col not in data.columns
        ]
        if missing_group_columns:
            raise BlockValidationError(
                f"GroupAggregate missing group columns: {missing_group_columns}"
            )

        specs = _parse_json_spec_list(
            params.aggregations,
            block_name="GroupAggregate",
            param_name="aggregations",
        )
        if not specs:
            raise BlockValidationError("aggregations is required.")

        grouped = data.groupby(group_columns, dropna=False, sort=False)
        result = grouped.size().reset_index(name="__group_size__")
        result = result.drop(columns=["__group_size__"])

        normalized_specs: list[dict[str, str]] = []
        for spec in specs:
            agg = str(spec.get("agg", "") or "").strip().lower()
            source = str(
                spec.get("source", spec.get("column", spec.get("source_column", "")))
                or ""
            ).strip()
            output = str(
                spec.get("output", spec.get("output_column", "")) or ""
            ).strip()
            if agg not in self._allowed_aggs:
                raise BlockValidationError(
                    f"GroupAggregate unsupported agg '{agg}'. Valid: {sorted(self._allowed_aggs)}"
                )
            if not output:
                raise BlockValidationError(
                    "GroupAggregate requires every aggregation spec to define output."
                )
            if agg != "size" and not source:
                raise BlockValidationError(
                    f"GroupAggregate aggregation '{agg}' requires source."
                )
            if source and source != "*" and source not in data.columns:
                raise BlockValidationError(
                    f"GroupAggregate missing source column '{source}'."
                )

            if agg == "size":
                series_df = grouped.size().reset_index(name=output)
            elif agg in self._numeric_aggs:
                working = data[group_columns].copy()
                working["__value__"] = pd.to_numeric(data[source], errors="coerce")
                series_df = (
                    working.groupby(group_columns, dropna=False, sort=False)[
                        "__value__"
                    ]
                    .agg(agg)
                    .reset_index(name=output)
                )
            elif agg == "count":
                series_df = grouped[source].count().reset_index(name=output)
            else:  # nunique
                series_df = (
                    grouped[source].nunique(dropna=True).reset_index(name=output)
                )

            result = result.merge(series_df, on=group_columns, how="left")
            normalized_specs.append(
                {
                    "source": source or "*",
                    "agg": agg,
                    "output": output,
                }
            )

        return BlockOutput(
            data=result,
            metadata={
                "group_columns": group_columns,
                "aggregations": normalized_specs,
                "n_groups": int(result.shape[0]),
            },
        )


class GroupPairMetrics(BaseBlock):
    name = "Group Pair Metrics"
    version = "1.0.0"
    category = "Statistics"
    description = "Group rows by key columns and compute pairwise metrics between two numeric columns."
    input_labels = ["DataFrame"]
    output_labels = ["Metric Table"]
    usage_notes = [
        "Rows with null or non-numeric values in x_column or y_column are dropped independently within each group.",
        "r2 uses sklearn's r2_score with x_column as y_true and y_column as y_pred.",
        "Supported metrics are: r2, spearman.",
    ]

    class Params(BlockParams):
        group_columns: str = block_param(
            description="Comma-separated columns used to define each group.",
            example="segment",
        )
        x_column: str = block_param(
            description="Numeric x column used in each pairwise metric.",
            example="baseline_value",
        )
        y_column: str = block_param(
            description="Numeric y column used in each pairwise metric.",
            example="comparison_value",
        )
        metrics: str = block_param(
            "r2,spearman",
            description="Comma-separated metrics to compute. Supported values: r2, spearman.",
            example="r2,spearman",
        )
        output_prefix: str = block_param(
            "",
            description="Optional prefix applied to every metric output column.",
            example="pair_",
        )

    _allowed_metrics = {"r2", "spearman"}

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("GroupPairMetrics requires params.")

        group_columns = _parse_columns(params.group_columns)
        if not group_columns:
            raise BlockValidationError("group_columns is required.")
        missing_group_columns = [
            col for col in group_columns if col not in data.columns
        ]
        if missing_group_columns:
            raise BlockValidationError(
                f"GroupPairMetrics missing group columns: {missing_group_columns}"
            )
        if params.x_column not in data.columns:
            raise BlockValidationError(
                f"GroupPairMetrics missing x column '{params.x_column}'."
            )
        if params.y_column not in data.columns:
            raise BlockValidationError(
                f"GroupPairMetrics missing y column '{params.y_column}'."
            )

        metrics = [metric.lower() for metric in _parse_columns(params.metrics)]
        if not metrics:
            raise BlockValidationError("metrics is required.")
        unsupported = [
            metric for metric in metrics if metric not in self._allowed_metrics
        ]
        if unsupported:
            raise BlockValidationError(
                f"GroupPairMetrics unsupported metrics: {unsupported}. Valid: {sorted(self._allowed_metrics)}"
            )

        output_prefix = str(params.output_prefix or "")
        rows: list[dict[str, Any]] = []
        for group_value, group_df in data.groupby(
            group_columns, dropna=False, sort=False
        ):
            if not isinstance(group_value, tuple):
                group_tuple = (group_value,)
            else:
                group_tuple = group_value

            pair_df = pd.DataFrame(
                {
                    "__x__": pd.to_numeric(group_df[params.x_column], errors="coerce"),
                    "__y__": pd.to_numeric(group_df[params.y_column], errors="coerce"),
                }
            ).dropna(subset=["__x__", "__y__"])

            row = {
                group_columns[index]: group_tuple[index]
                for index in range(len(group_columns))
            }
            if pair_df.shape[0] < 2:
                for metric in metrics:
                    row[f"{output_prefix}{metric}"] = float("nan")
                rows.append(row)
                continue

            x_values = pair_df["__x__"]
            y_values = pair_df["__y__"]
            if "r2" in metrics:
                try:
                    row[f"{output_prefix}r2"] = float(r2_score(x_values, y_values))
                except Exception:
                    row[f"{output_prefix}r2"] = float("nan")
            if "spearman" in metrics:
                row[f"{output_prefix}spearman"] = float(
                    x_values.corr(y_values, method="spearman")
                )
            rows.append(row)

        return BlockOutput(
            data=pd.DataFrame(rows),
            metadata={
                "group_columns": group_columns,
                "x_column": str(params.x_column),
                "y_column": str(params.y_column),
                "metrics": metrics,
                "output_prefix": output_prefix,
            },
        )


class GroupMeanByAssignments(BaseBlock):
    name = "Group Mean By Assignments"
    version = "1.0.0"
    category = "Statistics"
    description = (
        "Group rows by assignment labels and compute mean feature values per group."
    )
    n_inputs = 2
    input_labels = ["Feature Matrix", "Assignments"]
    output_labels = ["Grouped Means"]
    usage_notes = [
        "Input 0 is the feature matrix and input 1 is the assignments table.",
        "Both inputs are aligned by their row index after converting index values to strings.",
        "The output contains one row per assignment label with numeric mean values only.",
    ]

    class Params(BlockParams):
        cluster_column: str = block_param(
            "cluster",
            description="Column in the assignments input used to define each group.",
            example="cluster_id",
        )

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("GroupMeanByAssignments requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("GroupMeanByAssignments requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                "GroupMeanByAssignments expects [feature_df, assignments_df]."
            )

        feature_df, assignments_df = data
        if params.cluster_column not in assignments_df.columns:
            raise BlockValidationError(
                f"Missing cluster column '{params.cluster_column}'."
            )

        features = feature_df.copy()
        features.index = features.index.astype(str)
        assignments = assignments_df[[params.cluster_column]].copy()
        assignments.index = assignments.index.astype(str)

        common = features.index.intersection(assignments.index)
        if len(common) == 0:
            raise BlockValidationError(
                "No overlap between feature index and assignments index."
            )

        merged = features.loc[common].copy()
        merged[params.cluster_column] = (
            assignments.loc[common, params.cluster_column].astype(int).values
        )
        grouped = merged.groupby(params.cluster_column).mean(numeric_only=True)
        return BlockOutput(data=grouped)


class CoverageByGroup(BaseBlock):
    name = "Coverage By Group"
    version = "1.0.0"
    category = "Statistics"
    description = "Compute per-group entity coverage fraction: unique entities in group / total unique entities."
    input_labels = ["DataFrame"]
    output_labels = ["Coverage Table"]

    class Params(BlockParams):
        group_col: str = block_param()
        entity_col: str = block_param()
        output_count_col: str = "entity_count"
        output_fraction_col: str = "coverage"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("CoverageByGroup requires params.")
        required = {params.group_col, params.entity_col}
        missing = required - set(data.columns)
        if missing:
            raise BlockValidationError(
                f"CoverageByGroup missing columns: {sorted(missing)}"
            )

        frame = data.copy()
        frame[params.entity_col] = frame[params.entity_col].astype(str)
        total_entities = max(int(frame[params.entity_col].nunique()), 1)
        grouped = (
            frame.groupby(params.group_col)[params.entity_col]
            .nunique()
            .reset_index(name=params.output_count_col)
        )
        grouped[params.output_fraction_col] = grouped[params.output_count_col] / float(
            total_entities
        )
        grouped = grouped.sort_values(
            params.output_fraction_col, ascending=False
        ).reset_index(drop=True)
        return BlockOutput(data=grouped, metadata={"total_entities": total_entities})


class ExponentialPenaltyWeight(BaseBlock):
    name = "Exponential Penalty Weight"
    version = "1.1.0"
    category = "Statistics"
    description = "Create a 0..1 weight column using exp(-(x/scale)^2) after clipping."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        source_column: str = "metric_value"
        output_column: str = "penalty_weight"
        clip_hi: float = 1.5
        scale: float = 0.35

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("ExponentialPenaltyWeight requires params.")
        if params.source_column not in data.columns:
            raise BlockValidationError(
                f"Source column '{params.source_column}' not found."
            )
        frame = data.copy()
        frame[params.output_column] = _exp_penalty(
            frame[params.source_column], clip_hi=params.clip_hi, scale=params.scale
        )
        return BlockOutput(data=frame)


class LinearScaledWeight(BaseBlock):
    name = "Linear Scaled Weight"
    version = "1.1.0"
    category = "Statistics"
    description = "Linearly scale a numeric column to a bounded weight range."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        source_column: str = "count_value"
        output_column: str = "scaled_weight"
        min_value: float = 1.0
        max_value: float = 3.0
        denominator: float = 3.0

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("LinearScaledWeight requires params.")
        if params.source_column not in data.columns:
            raise BlockValidationError(
                f"Source column '{params.source_column}' not found."
            )
        frame = data.copy()
        values = (
            pd.to_numeric(frame[params.source_column], errors="coerce")
            .fillna(params.min_value)
            .clip(lower=params.min_value, upper=params.max_value)
            .to_numpy(dtype=float)
        )
        denom = max(float(params.denominator), 1e-12)
        frame[params.output_column] = values / denom
        return BlockOutput(data=frame)


class AlignToReferenceMatrix(BaseBlock):
    name = "Align To Reference Matrix"
    version = "1.0.0"
    category = "Statistics"
    description = "Reindex a matrix to match a reference matrix index/columns."
    n_inputs = 2
    input_labels = ["Matrix", "Reference Matrix"]
    output_labels = ["Aligned Matrix"]

    class Params(BlockParams):
        fill_value: float = 0.0

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("AlignToReferenceMatrix requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("AlignToReferenceMatrix requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                "AlignToReferenceMatrix expects [matrix_df, reference_df]."
            )
        matrix_df, reference_df = data
        ref = reference_df.copy()
        ref.index = ref.index.astype(str)
        matrix = matrix_df.copy()
        matrix.index = matrix.index.astype(str)
        aligned = matrix.reindex(index=ref.index, columns=ref.columns).fillna(
            params.fill_value
        )
        return BlockOutput(data=aligned)


class MaskByReferenceObserved(BaseBlock):
    name = "Mask By Reference Observed"
    version = "1.0.0"
    category = "Statistics"
    description = (
        "Set matrix values to a fill value where reference matrix entries are missing."
    )
    n_inputs = 2
    input_labels = ["Matrix", "Reference Matrix"]
    output_labels = ["Masked Matrix"]

    class Params(BlockParams):
        fill_value: float = 0.0

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("MaskByReferenceObserved requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MaskByReferenceObserved requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                "MaskByReferenceObserved expects [matrix_df, reference_df]."
            )
        matrix_df, reference_df = data
        ref = reference_df.copy()
        ref.index = ref.index.astype(str)
        matrix = matrix_df.copy()
        matrix.index = matrix.index.astype(str)
        matrix = matrix.reindex(index=ref.index, columns=ref.columns).fillna(
            params.fill_value
        )
        values = matrix.to_numpy(dtype=float, copy=True)
        obs_mask = ref.notna().to_numpy(dtype=bool)
        values[~obs_mask] = float(params.fill_value)
        out = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
        return BlockOutput(data=out)


class MinimumValueAcrossColumns(BaseBlock):
    name = "Minimum Across Columns"
    version = "1.1.0"
    category = "Statistics"
    description = (
        "Compute the minimum value across specified columns into a new output column."
    )
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    n_inputs = 1

    class Params(BlockParams):
        columns: str = "value_a,value_b"
        output_column: str = "minimum_value"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MinimumValueAcrossColumns requires params.")
        cols = _parse_columns(params.columns)
        if not cols:
            raise BlockValidationError(
                "MinimumValueAcrossColumns requires at least one column."
            )
        missing = [col for col in cols if col not in data.columns]
        if missing:
            raise BlockValidationError(
                f"MinimumValueAcrossColumns missing columns: {missing}"
            )
        frame = data.copy()
        matrix = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(float("inf"))
        frame[params.output_column] = matrix.min(axis=1)
        return BlockOutput(data=frame)


class MeanAcrossColumns(BaseBlock):
    name = "Mean Across Columns"
    version = "1.2.0"
    category = "Statistics"
    description = (
        "Compute the mean value across specified columns into a new output column."
    )
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    n_inputs = 1
    usage_notes = [
        "When columns is blank, the block averages across all numeric columns in input order.",
    ]

    class Params(BlockParams):
        columns: str = block_param(
            "",
            description="Comma-separated column names to average. Leave blank to use all numeric columns.",
            example="value_a,value_b",
        )
        output_column: str = block_param(
            "mean_value",
            description="Output column name for the row-wise mean.",
            example="mean_value",
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MeanAcrossColumns requires params.")
        cols = _resolve_numeric_columns(
            data,
            params.columns,
            block_name="MeanAcrossColumns",
        )
        frame = data.copy()
        matrix = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(float("nan"))
        frame[params.output_column] = matrix.mean(axis=1)
        return BlockOutput(
            data=frame,
            metadata={
                "columns_averaged": cols,
                "output_column": str(params.output_column),
            },
        )


class CountNonNullAcrossColumns(BaseBlock):
    name = "Count Non-Null Across Columns"
    version = "1.1.0"
    category = "Statistics"
    description = "Count non-null values across specified columns for each row into a new output column."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    n_inputs = 1
    usage_notes = [
        "When columns is blank, the block counts non-null values across all numeric columns in input order.",
    ]

    class Params(BlockParams):
        columns: str = block_param(
            "",
            description="Comma-separated columns to inspect. Leave blank to use all numeric columns.",
            example="a,b,c",
        )
        output_column: str = block_param(
            "non_null_count",
            description="Output column name for the per-row non-null count.",
            example="non_null_count",
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("CountNonNullAcrossColumns requires params.")
        cols = _resolve_numeric_columns(
            data,
            params.columns,
            block_name="CountNonNullAcrossColumns",
        )

        frame = data.copy()
        frame[params.output_column] = frame[cols].notna().sum(axis=1).astype(int)
        return BlockOutput(
            data=frame,
            metadata={
                "columns_counted": cols,
                "output_column": str(params.output_column),
            },
        )


class AssignTierByThresholds(BaseBlock):
    name = "Assign Tier By Thresholds"
    version = "1.2.0"
    category = "Transform"
    description = (
        "Assign categorical tiers from a numeric column using descending thresholds."
    )
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    n_inputs = 1
    usage_notes = [
        "Configure exactly one cutoff mode: explicit thresholds or percentiles.",
        "Percentiles are values in [0, 100].",
        "When group_column is set, percentile cutoffs are computed independently within each group.",
        "Labels are ordered from highest tier to lowest tier and must contain one more value than the number of cutoffs.",
    ]

    class Params(BlockParams):
        source_column: str = block_param(
            "non_null_patients",
            description="Numeric column used to assign tiers.",
            example="non_null_patients",
        )
        thresholds: str = block_param(
            "",
            description="Comma-separated explicit cutoffs applied directly to the source column values.",
            example="15,5",
        )
        percentiles: str = block_param(
            "",
            description="Comma-separated percentiles in [0, 100] used to derive cutoffs from the source column distribution.",
            example="90,50",
        )
        labels: str = block_param(
            "Tier 1,Tier 2,Tier 3",
            description="Comma-separated tier labels ordered from highest tier to lowest tier.",
            example="Tier 1,Tier 2,Tier 3",
        )
        output_label_column: str = block_param(
            "coverage_tier",
            description="Output column that stores the assigned tier label.",
            example="coverage_tier",
        )
        output_rank_column: str = block_param(
            "tier_rank",
            description="Output column that stores the assigned tier rank, where 1 is the highest tier.",
            example="tier_rank",
        )
        group_column: str = block_param(
            "",
            description="Optional column used to compute cutoffs independently within each group.",
            example="program_id",
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("AssignTierByThresholds requires params.")
        if params.source_column not in data.columns:
            raise BlockValidationError(
                f"Source column '{params.source_column}' not found."
            )
        group_column = str(getattr(params, "group_column", "") or "").strip()
        if group_column and group_column not in data.columns:
            raise BlockValidationError(f"Group column '{group_column}' not found.")

        values = pd.to_numeric(data[params.source_column], errors="coerce")
        explicit_thresholds = _parse_float_list(getattr(params, "thresholds", ""))
        percentiles = _parse_float_list(getattr(params, "percentiles", ""))
        configured_modes = [
            mode_name
            for mode_name, configured in (
                ("thresholds", explicit_thresholds),
                ("percentiles", percentiles),
            )
            if configured
        ]
        if not configured_modes:
            raise BlockValidationError(
                "Provide at least one cutoff mode via thresholds or percentiles."
            )
        if len(configured_modes) > 1:
            raise BlockValidationError(
                "AssignTierByThresholds accepts only one cutoff mode at a time: thresholds or percentiles."
            )

        cutoff_mode = configured_modes[0]
        cutoff_inputs: list[float]
        if cutoff_mode == "thresholds":
            cutoff_inputs = sorted(
                [float(t) for t in explicit_thresholds], reverse=True
            )
        else:
            cutoff_inputs = sorted([float(v) for v in percentiles], reverse=True)
            if any((value < 0.0 or value > 100.0) for value in cutoff_inputs):
                raise BlockValidationError("percentiles must be between 0 and 100.")

        labels = _parse_columns(params.labels)
        expected_labels = len(cutoff_inputs) + 1
        if len(labels) != expected_labels:
            raise BlockValidationError(
                f"labels must contain {expected_labels} values for {len(cutoff_inputs)} thresholds."
            )

        def _compute_thresholds(
            series: pd.Series,
            *,
            group_value: Any | None = None,
        ) -> list[float]:
            if cutoff_mode == "thresholds":
                return list(cutoff_inputs)

            finite_values = series[np.isfinite(series)]
            if finite_values.empty:
                context = f"Source column '{params.source_column}'"
                if group_value is not None:
                    context += f" in group {group_value!r}"
                raise BlockValidationError(
                    f"{context} has no finite numeric values for percentile-based cutoffs."
                )
            normalized_cutoffs = [value / 100.0 for value in cutoff_inputs]
            return [
                float(finite_values.quantile(quantile))
                for quantile in normalized_cutoffs
            ]

        def _assign(value: float, thresholds: list[float]) -> tuple[str, int]:
            if not np.isfinite(value):
                return labels[-1], expected_labels
            for idx, threshold in enumerate(thresholds):
                if float(value) >= float(threshold):
                    return labels[idx], idx + 1
            return labels[-1], expected_labels

        result = data.copy()
        if group_column:
            label_series = pd.Series(index=result.index, dtype="object")
            rank_series = pd.Series(index=result.index, dtype="float64")
            group_thresholds: dict[str, list[float]] = {}
            group_series = data[group_column]
            for group_value, group_index in group_series.groupby(
                group_series, dropna=False, sort=False
            ).groups.items():
                group_index = pd.Index(group_index)
                thresholds = _compute_thresholds(
                    values.loc[group_index],
                    group_value=group_value,
                )
                group_key = "<NA>" if pd.isna(group_value) else str(group_value)  # pyright: ignore[reportArgumentType, reportCallIssue]
                group_thresholds[group_key] = thresholds
                tiers = values.loc[group_index].apply(
                    lambda value, resolved=thresholds: _assign(value, resolved)
                )
                label_series.loc[group_index] = tiers.map(lambda pair: pair[0])  # pyright: ignore[reportIndexIssue]
                rank_series.loc[group_index] = tiers.map(lambda pair: pair[1])  # pyright: ignore[reportIndexIssue]
            result[params.output_label_column] = label_series
            result[params.output_rank_column] = rank_series.astype(int)
            thresholds_metadata: list[float] | None = None
        else:
            thresholds = _compute_thresholds(values)
            tiers = values.apply(lambda value: _assign(value, thresholds))
            result[params.output_label_column] = tiers.map(lambda pair: pair[0])  # pyright: ignore[reportIndexIssue]
            result[params.output_rank_column] = tiers.map(lambda pair: pair[1]).astype(  # pyright: ignore[reportIndexIssue]
                int
            )
            thresholds_metadata = thresholds
            group_thresholds = {}
        return BlockOutput(
            data=result,
            metadata={
                "source_column": str(params.source_column),
                "group_column": group_column or None,
                "cutoff_mode": cutoff_mode,
                "cutoff_inputs": cutoff_inputs,
                "thresholds": thresholds_metadata,
                "group_thresholds": group_thresholds,
                "labels": labels,
            },
        )
