"""End-to-end tests for the ``"index"`` sentinel that lets any column-key
field refer to the DataFrame index instead of a named column.
"""

from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from backend.block import BlockValidationError
from blocks.clustering import KMeansClustering
from blocks.combine import MergeDatasets
from blocks.operators import AbsoluteValueColumn, MultiplyColumns
from blocks.statistics import GroupAggregate, GroupPairMetrics
from blocks.transform import (
    DeduplicateRows,
    DropNullRows,
    FilterByLookupValues,
    FilterRows,
    MeltColumns,
    PivotTable,
    SelectColumns,
    SortRows,
)
from blocks.visualization import MatrixScatterPlot


# ── Combine ───────────────────────────────────────────────────────────────────


def test_merge_datasets_joins_on_index_when_on_is_index_sentinel() -> None:
    left = pd.DataFrame({"x": [10, 20, 30]}, index=["a", "b", "c"])
    right = pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    out = MergeDatasets().execute(
        [left, right],
        MergeDatasets.Params(on="index", how="inner"),
    )
    assert list(out.data.columns) == ["x", "y"]
    assert list(out.data.index) == ["a", "b", "c"]
    assert out.data.loc["b", "y"] == 2.0


def test_merge_datasets_index_sentinel_with_non_overlapping_indexes() -> None:
    left = pd.DataFrame({"x": [10, 20]}, index=["a", "b"])
    right = pd.DataFrame({"y": [9.0, 8.0]}, index=["b", "c"])
    out = MergeDatasets().execute(
        [left, right],
        MergeDatasets.Params(on="index", how="outer"),
    )
    # outer join keeps a, b, c
    assert sorted(out.data.index.tolist()) == ["a", "b", "c"]


def test_merge_datasets_index_sentinel_wins_over_real_index_column() -> None:
    # Even if a column literally named "index" exists, the sentinel always
    # joins on the DataFrame index — never the column.
    left = pd.DataFrame(
        {"index": [99, 99, 99], "x": [1, 2, 3]}, index=["a", "b", "c"]
    )
    right = pd.DataFrame({"y": [7.0, 8.0, 9.0]}, index=["a", "b", "c"])
    out = MergeDatasets().execute(
        [left, right],
        MergeDatasets.Params(on="index", how="inner"),
    )
    # If we'd joined on the literal "index" column (all 99s), every row of
    # left would cross with every row of right → 9 rows. The sentinel joins
    # on the actual index → 3 rows.
    assert out.data.shape[0] == 3


# ── Transform: single-column key fields ──────────────────────────────────────


def test_sort_rows_supports_index_sentinel() -> None:
    df = pd.DataFrame({"v": [10, 20, 30]}, index=["c", "a", "b"])
    out = SortRows().execute(
        df, SimpleNamespace(column="index", ascending=True)  # type: ignore
    )
    assert list(out.data.index) == ["a", "b", "c"]
    assert list(out.data["v"]) == [20, 30, 10]


def test_filter_rows_filters_on_index_sentinel() -> None:
    df = pd.DataFrame({"v": [1, 2, 3, 4]}, index=[1, 5, 3, 9])
    out = FilterRows().execute(
        df,
        SimpleNamespace(column="index", operator="gt", value=3),  # type: ignore
    )
    # Index values > 3 → indexes 5 and 9.
    assert sorted(out.data.index.tolist()) == [5, 9]


# ── Transform: comma-separated list fields ───────────────────────────────────


def test_deduplicate_rows_supports_index_in_key_columns() -> None:
    df = pd.DataFrame({"v": [1, 1, 1, 1]}, index=["a", "a", "b", "c"])
    out = DeduplicateRows().execute(
        df, SimpleNamespace(key_columns="index", keep="first")  # type: ignore
    )
    assert list(out.data.index) == ["a", "b", "c"]


def test_drop_null_rows_supports_index_in_columns() -> None:
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[1.0, float("nan"), 3.0])
    out = DropNullRows().execute(
        df, SimpleNamespace(columns="index", how="any")  # type: ignore
    )
    # The NaN-indexed row drops.
    assert len(out.data) == 2


def test_select_columns_materializes_index_when_sentinel_is_requested() -> None:
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]}, index=["row_a", "row_b"])
    out = SelectColumns().execute(
        df, SimpleNamespace(columns="index,x")  # type: ignore
    )
    assert list(out.data.columns) == ["index", "x"]
    assert list(out.data["index"]) == ["row_a", "row_b"]


def test_melt_columns_supports_index_in_id_columns() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["row1", "row2"])
    out = MeltColumns().execute(
        df,
        SimpleNamespace(  # type: ignore
            id_columns="index",
            value_columns="a,b",
            variable_column="variable",
            value_column="value",
            drop_null_values=False,
        ),
    )
    assert set(out.data.columns) == {"index", "variable", "value"}
    assert set(out.data["index"]) == {"row1", "row2"}


def test_pivot_table_supports_index_sentinel_in_index_field() -> None:
    df = pd.DataFrame(
        {"col": ["x", "y", "x", "y"], "val": [1.0, 2.0, 3.0, 4.0]},
        index=["r1", "r1", "r2", "r2"],
    )
    out = PivotTable().execute(
        df,
        SimpleNamespace(  # type: ignore
            index="index",
            columns="col",
            values="val",
            aggfunc="mean",
            true_numeric_sorting=False,
        ),
    )
    assert sorted(out.data.index.tolist()) == ["r1", "r2"]
    assert sorted(out.data.columns.tolist()) == ["x", "y"]


def test_filter_by_lookup_values_supports_index_keys() -> None:
    data_df = pd.DataFrame({"v": [10, 20, 30]}, index=["a", "b", "c"])
    lookup_df = pd.DataFrame({"keep": [True, True]}, index=["a", "c"])
    out = FilterByLookupValues().execute(
        [data_df, lookup_df],
        SimpleNamespace(  # type: ignore
            data_key="index",
            lookup_key="index",
            lookup_filter_column=None,
            lookup_filter_operator=None,
            lookup_filter_value=None,
            keep_matches=True,
        ),
    )
    assert sorted(out.data.index.tolist()) == ["a", "c"]


# ── Statistics ───────────────────────────────────────────────────────────────


def test_group_aggregate_groups_by_index_sentinel() -> None:
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=["a", "a", "b", "b"])
    out = GroupAggregate().execute(
        df,
        SimpleNamespace(  # type: ignore
            group_columns="index",
            aggregations='[{"source":"v","agg":"mean","output":"v_mean"}]',
        ),
    )
    by_group = dict(zip(out.data["index"], out.data["v_mean"]))
    assert by_group == {"a": 1.5, "b": 3.5}


def test_group_pair_metrics_supports_index_as_group() -> None:
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 6.0, 8.0]},
        index=["g1", "g1", "g2", "g2"],
    )
    out = GroupPairMetrics().execute(
        df,
        SimpleNamespace(  # type: ignore
            group_columns="index",
            x_column="x",
            y_column="y",
            metrics="spearman",
            output_prefix="",
        ),
    )
    assert "index" in out.data.columns
    assert sorted(out.data["index"]) == ["g1", "g2"]


# ── Operators ────────────────────────────────────────────────────────────────


def test_absolute_value_column_supports_index_source() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=[-3.0, 5.0])
    out = AbsoluteValueColumn().execute(
        df,
        SimpleNamespace(source_column="index", output_column="abs_idx"),  # type: ignore
    )
    assert list(out.data["abs_idx"]) == [3.0, 5.0]


def test_multiply_many_columns_supports_index_in_columns() -> None:
    df = pd.DataFrame({"x": [2.0, 3.0]}, index=[5, 7])
    out = MultiplyColumns().execute(
        df,
        SimpleNamespace(columns="index,x", output_column="product"),  # type: ignore
    )
    assert list(out.data["product"]) == [10.0, 21.0]


# ── Clustering ───────────────────────────────────────────────────────────────


def test_kmeans_clustering_accepts_index_in_columns() -> None:
    sk = pytest.importorskip("sklearn")
    assert sk is not None
    df = pd.DataFrame(
        {"a": [1.0, 1.0, 10.0, 10.0]},
        index=[1.0, 1.5, 100.0, 100.5],
    )
    out = KMeansClustering().execute(
        df,
        SimpleNamespace(  # type: ignore
            n_clusters=2,
            random_state=0,
            columns="index,a",
            column_prefix="",
            standardize=False,
            output_column="cluster_id",
        ),
    )
    clusters = out.data["cluster_id"].tolist()
    # Each pair (low-index, high-index) lands in the same cluster.
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


# ── Visualization ────────────────────────────────────────────────────────────


def test_scatter_plot_supports_index_for_x_column() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    out = MatrixScatterPlot().execute(
        df,
        MatrixScatterPlot.Params(
            x_column="index",
            y_column="y",
            title="Index vs Y",
            interactive=False,
        ),
    )
    assert out.metadata["x_column"] == "index"
    assert out.metadata["n_rows_plotted"] == 3


# ── Sentinel validation ──────────────────────────────────────────────────────


def test_unknown_column_still_errors() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(BlockValidationError):
        SortRows().execute(df, SimpleNamespace(column="nope", ascending=True))  # type: ignore


def test_index_sentinel_is_case_sensitive_strict() -> None:
    # "Index" (capitalized) is NOT the sentinel — it's treated as a column name.
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(BlockValidationError):
        SortRows().execute(df, SimpleNamespace(column="Index", ascending=True))  # type: ignore
