from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from types import SimpleNamespace

from backend.block import BlockValidationError
from blocks.factorization import WeightedALSFactorization
from blocks.io import Constant, LoadCSV
from blocks.operators import (
    AbsoluteValueColumn,
    AddColumns,
    DivideColumns,
    MultiplyColumnsOperator,
    SubtractColumns,
)
from blocks.statistics import (
    AssignTierByThresholds,
    CountNonNullAcrossColumns,
    GroupAggregate,
    GroupPairMetrics,
    MeanAcrossColumns,
)
from blocks.transform import (
    CastColumns,
    ColumnMedianCenter,
    DeduplicateRows,
    DropNullRows,
    FilterColumnsByCoverage,
    FilterRows,
    ImputeMissingValues,
    MaskOutliersMAD,
    MeltColumns,
    PivotTable,
    ResetIndex,
    SelectColumnsByReference,
    SplitListColumn,
)
from blocks.visualization import (
    AnnotatePlotWithArrows,
    ClusterHeatmap,
    FacetedScatterPlot,
    HighlightedBarChart,
    HighlightedScatterPlot,
    Matrix3DScatterPlot,
    MatrixBarChart,
    MatrixHeatmap,
    MatrixHistogram,
    MatrixLineChart,
    MatrixScatterPlot,
)

def _scalar_df(value: object) -> pd.DataFrame:
    return pd.DataFrame({"value": [value]})


def test_load_csv_supports_index_col(tmp_path) -> None:  # type: ignore
    csv_path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "ROW_ID": ["A1", "A2"],
            "FeatureA": [1.0, 2.0],
            "FeatureB": [3.0, 4.0],
        }
    ).to_csv(csv_path, index=False)

    out = LoadCSV().execute(
        None,
        SimpleNamespace(
            filepath=str(csv_path),
            sep=",",
            encoding="utf-8",
            index_col="ROW_ID",
        ),  # type: ignore
    )
    assert out.data.index.tolist() == ["A1", "A2"]
    assert "ROW_ID" not in out.data.columns


def test_reset_index_block() -> None:
    frame = pd.DataFrame(
        {"FeatureA": [1.0, 2.0], "FeatureB": [3.0, 4.0]},
        index=pd.Index(["A1", "A2"], name="ROW_ID"),
    )
    out = ResetIndex().execute(frame, SimpleNamespace(drop=False))  # type: ignore
    assert "ROW_ID" in out.data.columns
    assert out.data["ROW_ID"].tolist() == ["A1", "A2"]


def test_assign_tier_by_thresholds_block() -> None:
    frame = pd.DataFrame(
        {
            "item": ["A", "B", "C"],
            "row_count": [25, 8, 3],
        }
    )
    out = AssignTierByThresholds().execute(
        frame,
        SimpleNamespace(
            source_column="row_count",
            thresholds="15,5",
            labels="Tier 1,Tier 2,Tier 3",
            output_label_column="tier",
            output_rank_column="tier_rank",
        ),  # type: ignore
    )
    assert out.data["tier"].tolist() == ["Tier 1", "Tier 2", "Tier 3"]
    assert out.data["tier_rank"].tolist() == [1, 2, 3]
    assert out.metadata["cutoff_mode"] == "thresholds"
    assert out.metadata["cutoff_inputs"] == [15.0, 5.0]


def test_assign_tier_by_percentiles_block() -> None:
    frame = pd.DataFrame(
        {
            "item": ["A", "B", "C", "D", "E"],
            "row_count": [100, 80, 60, 40, 20],
        }
    )
    out = AssignTierByThresholds().execute(
        frame,
        SimpleNamespace(
            source_column="row_count",
            thresholds="",
            percentiles="50",
            labels="High,Low",
            output_label_column="tier",
            output_rank_column="tier_rank",
        ),  # type: ignore
    )
    assert out.data["tier"].tolist() == ["High", "High", "High", "Low", "Low"]
    assert out.data["tier_rank"].tolist() == [1, 1, 1, 2, 2]
    assert out.metadata["cutoff_mode"] == "percentiles"
    assert out.metadata["cutoff_inputs"] == [50.0]
    assert out.metadata["thresholds"] == [60.0]


def test_assign_tier_by_percentiles_with_group_column() -> None:
    frame = pd.DataFrame(
        {
            "program": ["A", "A", "A", "B", "B", "B"],
            "score": [100.0, 60.0, 20.0, 10.0, 6.0, 2.0],
        }
    )
    out = AssignTierByThresholds().execute(
        frame,
        SimpleNamespace(
            source_column="score",
            thresholds="",
            percentiles="50",
            labels="High,Low",
            output_label_column="tier",
            output_rank_column="tier_rank",
            group_column="program",
        ),  # type: ignore
    )
    assert out.data["tier"].tolist() == ["High", "High", "Low", "High", "High", "Low"]
    assert out.data["tier_rank"].tolist() == [1, 1, 2, 1, 1, 2]
    assert out.metadata["group_column"] == "program"
    assert out.metadata["group_thresholds"] == {"A": [60.0], "B": [6.0]}


def test_assign_tier_by_thresholds_rejects_multiple_cutoff_modes() -> None:
    frame = pd.DataFrame(
        {
            "item": ["A", "B", "C"],
            "row_count": [25, 8, 3],
        }
    )
    with pytest.raises(BlockValidationError, match="only one cutoff mode"):
        AssignTierByThresholds().execute(
            frame,
            SimpleNamespace(
                source_column="row_count",
                thresholds="15,5",
                percentiles="50",
                labels="Tier 1,Tier 2,Tier 3",
                output_label_column="tier",
                output_rank_column="tier_rank",
            ),  # type: ignore
        )


def test_impute_missing_values_block_uses_median_and_fallback() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["A1", "A2", "A3"],
            "feature_a": [1.0, np.nan, 5.0],
            "feature_b": [np.nan, np.nan, np.nan],
            "note": ["x", "y", "z"],
        }
    )

    out = ImputeMissingValues().execute(
        frame,
        SimpleNamespace(
            columns="feature_a,feature_b",
            strategy="median",
            fill_value=-1.0,
        ),  # type: ignore
    )

    assert out.data["feature_a"].tolist() == [1.0, 3.0, 5.0]
    assert out.data["feature_b"].tolist() == [-1.0, -1.0, -1.0]
    assert out.data["note"].tolist() == ["x", "y", "z"]
    assert out.metadata["strategy"] == "median"
    assert out.metadata["n_columns_imputed"] == 2
    assert out.metadata["n_cells_filled"] == 4
    assert out.metadata["column_fill_values"]["feature_a"] == 3.0
    assert out.metadata["column_fill_values"]["feature_b"] == -1.0


def test_select_columns_by_reference_block_keeps_input_rows_and_order() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["A1", "A2"],
            "dataset": [0, 2],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
            "feature_c": [5.0, 6.0],
        }
    )
    reference = pd.DataFrame({"feature_c": [9.0], "feature_a": [10.0]})

    out = SelectColumnsByReference().execute(
        [frame, reference],
        SimpleNamespace(
            include_columns="record_id,dataset",
            preserve_input_order=True,
        ),  # type: ignore
    )

    assert list(out.data.columns) == ["record_id", "dataset", "feature_a", "feature_c"]
    assert out.data["record_id"].tolist() == ["A1", "A2"]
    assert out.metadata["n_reference_matches"] == 2
    assert out.metadata["n_columns_out"] == 4


def test_select_columns_by_reference_block_requires_overlap() -> None:
    frame = pd.DataFrame({"record_id": ["A1"], "feature_a": [1.0]})
    reference = pd.DataFrame({"feature_b": [2.0]})

    with pytest.raises(Exception) as excinfo:
        SelectColumnsByReference().execute(
            [frame, reference],
            SimpleNamespace(include_columns="record_id"),  # type: ignore
        )

    assert "No overlapping columns" in str(excinfo.value)


def test_drop_null_rows_block_supports_any_and_all_modes() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [1.0, 2.0, np.nan, np.nan],
            "c": ["keep", "keep", "keep", "drop"],
        }
    )

    any_out = DropNullRows().execute(
        frame,
        SimpleNamespace(columns="a,b", how="any"),  # type: ignore
    )
    assert any_out.data.index.tolist() == [0]
    assert any_out.metadata["n_rows_dropped"] == 3

    all_out = DropNullRows().execute(
        frame,
        SimpleNamespace(columns="a,b", how="all"),  # type: ignore
    )
    assert all_out.data.index.tolist() == [0, 1, 2]
    assert all_out.metadata["n_rows_dropped"] == 1


def test_deduplicate_rows_block_preserves_input_order() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["A", "A", "B", "A"],
            "category": ["Group1", "Group1", "Group2", "Group3"],
            "value": [5.0, 3.0, 4.0, 6.0],
        }
    )

    first_out = DeduplicateRows().execute(
        frame,
        SimpleNamespace(
            key_columns="record_id,category",
            keep="first",
        ),  # type: ignore
    )
    assert first_out.data["value"].tolist() == [5.0, 4.0, 6.0]
    assert first_out.metadata["n_rows_removed"] == 1

    last_out = DeduplicateRows().execute(
        frame,
        SimpleNamespace(
            key_columns="record_id,category",
            keep="last",
        ),  # type: ignore
    )
    assert last_out.data["value"].tolist() == [3.0, 4.0, 6.0]


def test_group_aggregate_block_computes_multiple_metrics() -> None:
    frame = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2", "g2", "g2"],
            "entity": ["A", "A", "B", "C", None],
            "value": [1.0, 3.0, 10.0, np.nan, 4.0],
        }
    )

    out = GroupAggregate().execute(
        frame,
        SimpleNamespace(
            group_columns="group",
            aggregations='[{"source":"*","agg":"size","output":"n_rows"},{"source":"entity","agg":"nunique","output":"n_entities"},{"source":"value","agg":"mean","output":"mean_value"},{"source":"value","agg":"std","output":"std_value"}]',
        ),  # type: ignore
    )

    expected = pd.DataFrame(
        {
            "group": ["g1", "g2"],
            "n_rows": [2, 3],
            "n_entities": [1, 2],
            "mean_value": [2.0, 7.0],
            "std_value": [np.sqrt(2.0), np.sqrt(18.0)],
        }
    )
    pd.testing.assert_frame_equal(out.data.reset_index(drop=True), expected)
    assert out.metadata["n_groups"] == 2


def test_group_pair_metrics_block_supports_r2_and_spearman() -> None:
    frame = pd.DataFrame(
        {
            "group": ["aligned", "aligned", "aligned", "reversed", "reversed", "reversed"],
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )

    out = GroupPairMetrics().execute(
        frame,
        SimpleNamespace(
            group_columns="group",
            x_column="x",
            y_column="y",
            metrics="r2,spearman",
            output_prefix="pair_",
        ),  # type: ignore
    )

    aligned = out.data[out.data["group"] == "aligned"].iloc[0]
    assert float(aligned["pair_r2"]) == pytest.approx(1.0)
    assert float(aligned["pair_spearman"]) == pytest.approx(1.0)

    reversed_row = out.data[out.data["group"] == "reversed"].iloc[0]
    assert float(reversed_row["pair_r2"]) == pytest.approx(-3.0)
    assert float(reversed_row["pair_spearman"]) == pytest.approx(-1.0)


def test_split_list_column_block() -> None:
    frame = pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "value_series": ["[0,1,4,2,3]", "[5,6,7,8,9]", "[10,11,12,13,14]"],
        }
    )

    out = SplitListColumn().execute(
        frame,
        SimpleNamespace(
            column_name="value_series",
            column_name_prefix="value_part_",
            starting_index=0,
        ),  # type: ignore
    )

    cols = [f"value_part_{i}" for i in range(5)]
    assert all(col in out.data.columns for col in cols)
    assert out.data.loc[0, "value_part_0"] == 0
    assert out.data.loc[0, "value_part_2"] == 4
    assert out.data.loc[2, "value_part_4"] == 14
    assert out.metadata["n_columns_added"] == 5
    assert out.metadata["source_column"] == "value_series"


def test_split_list_column_rejects_inconsistent_lengths() -> None:
    frame = pd.DataFrame(
        {
            "value_series": ["[0,1,2]", "[3,4]"],
        }
    )

    with pytest.raises(Exception) as excinfo:
        SplitListColumn().execute(
            frame,
            SimpleNamespace(
                column_name="value_series",
                column_name_prefix="value_part_",
                starting_index=0,
            ),  # type: ignore
        )
    assert "Inconsistent list length" in str(excinfo.value)


def test_split_list_column_accepts_null_tokens() -> None:
    frame = pd.DataFrame(
        {
            "value_series": ["[0,NA,4,Null]", "[1,2,NaN,3]", "[4,None,5,n/a]"],
        }
    )

    out = SplitListColumn().execute(
        frame,
        SimpleNamespace(
            column_name="value_series",
            column_name_prefix="value_part_",
            starting_index=10,
        ),  # type: ignore
    )

    assert out.metadata["n_columns_added"] == 4
    assert out.data.loc[0, "value_part_10"] == 0
    assert pd.isna(out.data.loc[0, "value_part_11"])
    assert pd.isna(out.data.loc[0, "value_part_13"])
    assert pd.isna(out.data.loc[1, "value_part_12"])
    assert pd.isna(out.data.loc[2, "value_part_11"])


def test_melt_columns_uses_all_non_id_columns_when_value_columns_blank() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["A", "B"],
            "dataset": ["X", "Y"],
            "program_1": [1.0, 2.0],
            "program_2": [3.0, 4.0],
        }
    )

    out = MeltColumns().execute(
        frame,
        SimpleNamespace(
            id_columns="ID,dataset",
            value_columns="",
            variable_column="program",
            value_column="score",
            drop_null_values=False,
        ),  # type: ignore
    )

    assert out.data["program"].tolist() == [
        "program_1",
        "program_1",
        "program_2",
        "program_2",
    ]
    assert out.data["score"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out.metadata["value_columns"] == ["program_1", "program_2"]


def test_pivot_table_defaults_to_lexicographic_column_sorting() -> None:
    frame = pd.DataFrame(
        {
            "sample": ["A", "A", "A", "A", "A", "A"],
            "col_key": [
                "Group1__Step1",
                "Group1__Step10",
                "Group1__Step2",
                "Group2__Step1",
                "Group2__Step10",
                "Group2__Step2",
            ],
            "value": [1.0, 10.0, 2.0, 3.0, 30.0, 4.0],
        }
    )

    pivoted = (
        PivotTable()
        .execute(
            frame,
            SimpleNamespace(
                index="sample",
                columns="col_key",
                values="value",
                aggfunc="first",
            ),  # type: ignore
        )
        .data
    )

    assert list(pivoted.columns) == [
        "Group1__Step1",
        "Group1__Step10",
        "Group1__Step2",
        "Group2__Step1",
        "Group2__Step10",
        "Group2__Step2",
    ]


def test_pivot_table_true_numeric_sorting_orders_embedded_numbers_naturally() -> None:
    frame = pd.DataFrame(
        {
            "sample": ["A", "A", "A", "A", "A", "A"],
            "col_key": [
                "Group1__Step1",
                "Group1__Step10",
                "Group1__Step2",
                "Group2__Step1",
                "Group2__Step10",
                "Group2__Step2",
            ],
            "value": [1.0, 10.0, 2.0, 3.0, 30.0, 4.0],
        }
    )

    pivoted = (
        PivotTable()
        .execute(
            frame,
            SimpleNamespace(
                index="sample",
                columns="col_key",
                values="value",
                aggfunc="first",
                true_numeric_sorting=True,
            ),  # type: ignore
        )
        .data
    )

    assert list(pivoted.columns) == [
        "Group1__Step1",
        "Group1__Step2",
        "Group1__Step10",
        "Group2__Step1",
        "Group2__Step2",
        "Group2__Step10",
    ]


def test_count_non_null_across_columns_block() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, None, 4.0],
            "b": [np.nan, 2.0, 3.0, None],
            "c": [7.0, 8.0, np.nan, None],
        }
    )

    out = CountNonNullAcrossColumns().execute(
        frame,
        SimpleNamespace(columns="a,b,c", output_column="present_count"),  # type: ignore
    )
    assert out.data["present_count"].tolist() == [2, 2, 1, 1]
    assert out.metadata["output_column"] == "present_count"


def test_mean_across_columns_uses_all_numeric_columns_when_columns_blank() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 5.0],
            "b": [3.0, 7.0, np.nan],
            "label": ["x", "y", "z"],
        }
    )

    out = MeanAcrossColumns().execute(
        frame,
        SimpleNamespace(columns="", output_column="mean_score"),  # type: ignore
    )

    assert out.data["mean_score"].tolist() == [2.0, 7.0, 5.0]
    assert out.metadata["columns_averaged"] == ["a", "b"]
    assert out.metadata["output_column"] == "mean_score"


def test_count_non_null_across_columns_uses_all_numeric_columns_when_columns_blank() -> (
    None
):
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 5.0],
            "b": [3.0, 7.0, np.nan],
            "label": ["x", "y", "z"],
        }
    )

    out = CountNonNullAcrossColumns().execute(
        frame,
        SimpleNamespace(columns="", output_column="present_count"),  # type: ignore
    )

    assert out.data["present_count"].tolist() == [2, 1, 1]
    assert out.metadata["columns_counted"] == ["a", "b"]
    assert out.metadata["output_column"] == "present_count"


def test_count_non_null_across_columns_missing_column_errors() -> None:
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(Exception) as excinfo:
        CountNonNullAcrossColumns().execute(
            frame,
            SimpleNamespace(columns="a,c", output_column="present_count"),  # type: ignore
        )
    assert "missing columns" in str(excinfo.value).lower()


def test_operator_blocks_with_scalar_and_dataframe_inputs() -> None:
    left = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})
    left_scalar = _scalar_df(10.0)
    right_scalar = pd.DataFrame({"value": [2.0]})
    right_df = pd.DataFrame({"y": [0.5, 1.0, np.nan, 0.0]})

    add_out = AddColumns().execute(
        [left, right_scalar],
        SimpleNamespace(
            input_1_column_name="x",
            input_2_column_name="",
            null_handling="Ignore",
            output_column="sum_x",
        ),  # type: ignore
    )
    np.testing.assert_allclose(
        add_out.data["sum_x"].to_numpy(dtype=float),
        np.array([3.0, np.nan, 5.0, 6.0]),
        equal_nan=True,
    )

    scalar_left_out = SubtractColumns().execute(
        [left_scalar, right_df],
        SimpleNamespace(
            input_1_column_name="",
            input_2_column_name="y",
            null_handling="Treat as 0",
            output_column="ten_minus_y",
        ),  # type: ignore
    )
    assert scalar_left_out.metadata["input_1_mode"] == "scalar"
    assert scalar_left_out.metadata["mapping_mode"] == "1_to_1"
    np.testing.assert_allclose(
        scalar_left_out.data["y"].to_numpy(dtype=float),
        np.array([0.5, 1.0, np.nan, 0.0]),
        equal_nan=True,
    )
    assert scalar_left_out.data["ten_minus_y"].tolist() == [9.5, 9.0, 10.0, 10.0]

    sub_out = SubtractColumns().execute(
        [left, right_df],
        SimpleNamespace(
            input_1_column_name="x",
            input_2_column_name="y",
            null_handling="Treat as 0",
            output_column="diff_x",
        ),  # type: ignore
    )
    assert sub_out.data["diff_x"].tolist() == [0.5, -1.0, 3.0, 4.0]

    mul_out = MultiplyColumnsOperator().execute(
        [left, right_df],
        SimpleNamespace(
            input_1_column_name="x",
            input_2_column_name="y",
            null_handling="Drop",
            output_column="prod_x",
        ),  # type: ignore
    )
    assert list(mul_out.data.index) == [0, 3]
    assert mul_out.data["prod_x"].tolist() == [0.5, 0.0]

    div_out = DivideColumns().execute(
        [left, right_df[["y"]]],
        SimpleNamespace(
            input_1_column_name="x",
            input_2_column_name="",
            null_handling="Ignore",
            output_column="div_x",
        ),  # type: ignore
    )
    assert div_out.data["div_x"].iloc[0] == 2.0
    assert np.isnan(div_out.data["div_x"].iloc[1])
    assert np.isnan(div_out.data["div_x"].iloc[2])
    assert np.isnan(div_out.data["div_x"].iloc[3])  # divide by zero -> NaN


def test_operator_blocks_require_input2_column_for_multicolumn_dataframe() -> None:
    left = pd.DataFrame({"x": [1.0, 2.0]})
    right = pd.DataFrame({"a": [3.0, 4.0], "b": [5.0, 6.0]})

    with pytest.raises(Exception) as excinfo:
        AddColumns().execute(
            [left, right],
            SimpleNamespace(
                input_1_column_name="x",
                input_2_column_name="",
                null_handling="Ignore",
                output_column="sum_x",
            ),  # type: ignore
        )
    assert "input_2_column_name is required" in str(excinfo.value)


def test_absolute_value_column_appends_new_column() -> None:
    frame = pd.DataFrame(
        {
            "feature": ["program_1", "program_2", "program_3"],
            "coefficient": [-2.5, 0.0, 3.25],
        }
    )

    out = AbsoluteValueColumn().execute(
        frame,
        SimpleNamespace(
            source_column="coefficient",
            output_column="abs_coefficient",
        ),  # type: ignore
    )

    assert out.data["coefficient"].tolist() == [-2.5, 0.0, 3.25]
    assert out.data["abs_coefficient"].tolist() == [2.5, 0.0, 3.25]
    assert out.metadata["source_column"] == "coefficient"
    assert out.metadata["output_column"] == "abs_coefficient"


def test_absolute_value_column_supports_overwrite_and_numeric_coercion() -> None:
    frame = pd.DataFrame({"coefficient": ["-4.5", "bad", "2"]})

    out = AbsoluteValueColumn().execute(
        frame,
        SimpleNamespace(
            source_column="coefficient",
            output_column="coefficient",
        ),  # type: ignore
    )

    np.testing.assert_allclose(
        out.data["coefficient"].to_numpy(dtype=float),
        np.array([4.5, np.nan, 2.0]),
        equal_nan=True,
    )


def test_absolute_value_column_missing_column_errors() -> None:
    frame = pd.DataFrame({"score": [1.0, -2.0]})

    with pytest.raises(BlockValidationError) as excinfo:
        AbsoluteValueColumn().execute(
            frame,
            SimpleNamespace(
                source_column="coefficient",
                output_column="abs_coefficient",
            ),  # type: ignore
        )

    assert "missing column" in str(excinfo.value).lower()


def test_operator_blocks_support_n_to_1_1_to_n_and_n_to_n() -> None:
    left = pd.DataFrame(
        {
            "Foo": [2.0, 4.0, np.nan],
            "Bar": [3.0, 5.0, 7.0],
            "Baz": [1.0, np.nan, 9.0],
            "One": [1.0, 1.0, 1.0],
        }
    )
    left_scalar = _scalar_df(1.0)
    right_scalar = pd.DataFrame({"value": [-1.0]})
    right_df = pd.DataFrame(
        {
            "R1": [10.0, 20.0, 30.0],
            "R2": [1.0, 2.0, 3.0],
            "R3": [5.0, 6.0, 7.0],
        }
    )

    # n-to-1 (scalar): multiply three columns by -1
    neg = (
        MultiplyColumnsOperator()
        .execute(
            [left, right_scalar],
            SimpleNamespace(
                input_1_column_name="Foo,Bar,Baz",
                input_2_column_name="",
                null_handling="Ignore",
                output_column="neg_{column}",
            ),  # type: ignore
        )
        .data
    )
    np.testing.assert_allclose(
        neg["neg_Foo"].to_numpy(dtype=float),
        np.array([-2.0, -4.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        neg["neg_Bar"].to_numpy(dtype=float),
        np.array([-3.0, -5.0, -7.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        neg["neg_Baz"].to_numpy(dtype=float),
        np.array([-1.0, np.nan, -9.0]),
        equal_nan=True,
    )

    # scalar-to-n: 1 - Foo/Bar/Baz
    scalar_one_minus = (
        SubtractColumns()
        .execute(
            [left_scalar, left[["Foo", "Bar", "Baz"]]],
            SimpleNamespace(
                input_1_column_name="",
                input_2_column_name="Foo,Bar,Baz",
                null_handling="Ignore",
                output_column="scalar_minus_",
            ),  # type: ignore
        )
        .data
    )
    np.testing.assert_allclose(
        scalar_one_minus["scalar_minus_Foo"].to_numpy(dtype=float),
        np.array([-1.0, -3.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        scalar_one_minus["scalar_minus_Bar"].to_numpy(dtype=float),
        np.array([-2.0, -4.0, -6.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        scalar_one_minus["scalar_minus_Baz"].to_numpy(dtype=float),
        np.array([0.0, np.nan, -8.0]),
        equal_nan=True,
    )

    # 1-to-n: 1 - Foo/Bar/Baz
    one_minus = (
        SubtractColumns()
        .execute(
            [left, left[["Foo", "Bar", "Baz"]]],
            SimpleNamespace(
                input_1_column_name="One",
                input_2_column_name="Foo,Bar,Baz",
                null_handling="Ignore",
                output_column="one_minus_",
            ),  # type: ignore
        )
        .data
    )
    np.testing.assert_allclose(
        one_minus["one_minus_Foo"].to_numpy(dtype=float),
        np.array([-1.0, -3.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        one_minus["one_minus_Bar"].to_numpy(dtype=float),
        np.array([-2.0, -4.0, -6.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        one_minus["one_minus_Baz"].to_numpy(dtype=float),
        np.array([0.0, np.nan, -8.0]),
        equal_nan=True,
    )

    # n-to-n: pairwise add with explicit output names
    pair = (
        AddColumns()
        .execute(
            [left, right_df],
            SimpleNamespace(
                input_1_column_name="Foo,Bar,Baz",
                input_2_column_name="R1,R2,R3",
                null_handling="Ignore",
                output_column="sum1,sum2,sum3",
            ),  # type: ignore
        )
        .data
    )
    np.testing.assert_allclose(
        pair["sum1"].to_numpy(dtype=float),
        np.array([12.0, 24.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        pair["sum2"].to_numpy(dtype=float), np.array([4.0, 7.0, 10.0]), equal_nan=True
    )
    np.testing.assert_allclose(
        pair["sum3"].to_numpy(dtype=float),
        np.array([6.0, np.nan, 16.0]),
        equal_nan=True,
    )


def test_operator_blocks_reject_unsupported_column_mapping() -> None:
    left = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    right = pd.DataFrame({"x": [4.0], "y": [5.0]})
    with pytest.raises(Exception) as excinfo:
        AddColumns().execute(
            [left, right],
            SimpleNamespace(
                input_1_column_name="a,b,c",
                input_2_column_name="x,y",
                null_handling="Ignore",
                output_column="out_",
            ),  # type: ignore
        )
    assert "n-to-1, 1-to-n, or n-to-n" in str(excinfo.value)


def test_weighted_als_factorization_emits_multiple_outputs() -> None:
    matrix = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
        index=["A1", "A2"],
        columns=["T1", "T2", "T3"],
    )
    weights = pd.DataFrame(
        np.ones((2, 3), dtype=float),
        index=matrix.index,
        columns=matrix.columns,
    )

    out = WeightedALSFactorization().execute(
        [matrix, weights],
        SimpleNamespace(
            n_components=2,
            lambda_value=1.0,
            n_iters=3,
            seed=0,
            output_prefix="program_",
        ),  # type: ignore
    )

    assert "output_0" in out.outputs
    assert "output_1" in out.outputs
    assert out.data.equals(out.outputs["output_0"])
    assert out.outputs["output_0"].shape == (2, 2)
    assert out.outputs["output_1"].shape == (3, 2)


def test_constant_block_parses_and_emits_typed_values() -> None:
    block = Constant()

    out_int = block.execute(
        None,
        SimpleNamespace(value="7", value_type="int"),  # type: ignore
    )
    assert out_int.data.iloc[0]["value"] == 7

    out_float = block.execute(
        None,
        SimpleNamespace(value="3.5", value_type="float"),  # type: ignore
    )
    assert float(out_float.data.iloc[0]["value"]) == 3.5

    out_json = block.execute(
        None,
        SimpleNamespace(value='{"k": 2, "lambda": 1.0}', value_type="json"),  # type: ignore
    )
    assert isinstance(out_json.data.iloc[0]["value"], dict)
    assert out_json.data.iloc[0]["value"]["k"] == 2

    out_auto = block.execute(
        None,
        SimpleNamespace(value="hello", value_type="auto"),  # type: ignore
    )
    assert out_auto.data.iloc[0]["value"] == "hello"


def test_highlighted_scatter_plot_supports_export_guides_and_legend(tmp_path) -> None:  # type: ignore
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.2, 1.1, 2.3, 2.8, 4.2],
            "category": ["background", "A", "background", "B", "A"],
        }
    )
    export_dir = tmp_path / "exports"

    out = HighlightedScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            highlight_column="category",
            highlight_values="A,B",
            highlight_colors="#d62728,#1f77b4",
            label_column="category",
            size_column="",
            marker_size=45.0,
            background_color="lightgray",
            background_alpha=0.3,
            highlight_alpha=0.9,
            show_legend=True,
            label_highlights=False,
            label_fontsize=10.0,
            x_label="Reference",
            y_label="Comparison",
            identity_line=True,
            identity_line_color="0.7",
            identity_line_style="--",
            guide_lines='[{"axis":"x","value":2.5,"color":"black","linestyle":":","linewidth":1.0,"label":"cutoff"}]',
            guide_bands='[{"axis":"x","start":-1.0,"end":0.5,"color":"#eeeeee","alpha":0.2}]',
            annotations='[{"x":0.1,"y":4.0,"text":"note","color":"black"}]',
            figsize=[6.0, 4.0],
            title="Highlighted Scatter",
            export_enabled=True,
            export_dir=str(export_dir),
            export_basename="custom_scatter",
        ),  # type: ignore
    )

    expected_path = export_dir / "custom_scatter.png"
    assert expected_path.exists()
    assert out.metadata["exported_path"] == str(expected_path)
    assert out.metadata["n_background_points"] == 2
    assert out.metadata["n_highlighted_points"] == 3
    assert out.metadata["n_highlight_groups_drawn"] == 2
    assert out.metadata["n_guide_lines"] == 1
    assert out.metadata["n_guide_bands"] == 1
    assert out.metadata["n_annotations"] == 1
    fig = out.images[0]
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["cutoff", "A", "B"]


def test_highlighted_scatter_plot_can_hide_highlight_legend_entries() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "category": ["A", "B", "background"],
        }
    )

    out = HighlightedScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            highlight_column="category",
            highlight_values="A,B",
            highlight_colors="#d62728,#1f77b4",
            size_column="",
            marker_size=35.0,
            background_color="lightgray",
            background_alpha=0.3,
            highlight_alpha=0.9,
            show_legend=True,
            show_highlight_legend=False,
            label_highlights=False,
            label_fontsize=10.0,
            x_label="x",
            y_label="y",
            identity_line=False,
            guide_lines='[{"axis":"x","value":0.5,"color":"black","linestyle":"--","linewidth":1.0,"label":"guide"}]',
            guide_bands="",
            annotations="",
            figsize=[5.0, 4.0],
            title="Scatter",
            export_enabled=False,
            export_dir=None,
            export_basename="",
        ),  # type: ignore
    )

    legend = out.images[0].axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["guide"]


def test_highlighted_bar_chart_draws_reference_line_and_highlight_labels() -> None:
    frame = pd.DataFrame(
        {
            "label": ["A", "B", "C", "D"],
            "ratio": [0.2, 0.6, 0.4, 1.2],
        }
    )

    out = HighlightedBarChart().execute(
        frame,
        SimpleNamespace(
            category_column="label",
            value_column="ratio",
            highlight_values="B,D",
            highlight_colors="#2ca02c,#ff7f0e",
            top_n=0,
            sort_by_value=True,
            ascending=False,
            default_color="lightgray",
            label_highlights=True,
            label_fontsize=10.0,
            show_all_tick_labels=False,
            x_label="ratio",
            reference_lines='[{"axis":"x","value":0.5,"color":"#ff8b8b","linestyle":"--","linewidth":1.5,"label":"50% compression"}]',
            figsize=[7.0, 4.0],
            title="Highlighted Bar",
        ),  # type: ignore
    )

    assert out.metadata["n_rows_plotted"] == 4
    assert out.metadata["n_highlighted_bars"] == 2
    assert out.metadata["n_reference_lines"] == 1
    fig = out.images[0]
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["50% compression"]
    assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["", "", "", ""]


def test_faceted_scatter_plot_formats_titles_from_context() -> None:
    frame = pd.DataFrame(
        {
            "panel": ["A", "A", "B", "B"],
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": [1.1, 1.9, 2.0, 1.0],
            "item_count": [2, 2, 2, 2],
            "metric_a": [0.9, 0.9, -1.0, -1.0],
            "metric_b": [1.0, 1.0, -1.0, -1.0],
        }
    )

    out = FacetedScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            facet_column="panel",
            facet_values="A,B",
            facet_color_map='{"A":"#d62728","B":"#1f77b4"}',
            facet_status_map='{"A":"Primary","B":"Secondary"}',
            facet_title_template="{facet_value}\\nn={item_count}, status={status_label}",
            n_cols=2,
            marker_size=30.0,
            alpha=0.9,
            identity_line=True,
            identity_line_color="0.75",
            identity_line_style="--",
            x_label="Reference",
            y_label="Comparison",
            figsize=[8.0, 4.0],
            title="Facet Grid",
        ),  # type: ignore
    )

    assert out.metadata["n_facets_drawn"] == 2
    fig = out.images[0]
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Facet Grid"
    titles = [ax.get_title() for ax in fig.axes[:2]]
    assert titles == ["A\nn=2, status=Primary", "B\nn=2, status=Secondary"]


def test_faceted_scatter_plot_supports_facet_specific_title_templates() -> None:
    frame = pd.DataFrame(
        {
            "panel": ["A", "A", "B", "B"],
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": [1.1, 1.9, 2.0, 1.0],
            "item_count": [2, 2, 3, 3],
            "metric_a": [0.9, 0.9, -1.0, -1.0],
            "metric_b": [1.0, 1.0, -1.0, -1.0],
        }
    )

    out = FacetedScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            facet_column="panel",
            facet_values="A,B",
            facet_color_map='{"A":"#d62728","B":"#1f77b4"}',
            facet_status_map='{"A":"Primary","B":"Secondary"}',
            facet_title_map='{"A":"A title","B":"B\\nn={item_count}"}',
            facet_title_template="{facet_value}\\nstatus={status_label}",
            n_cols=2,
            marker_size=30.0,
            alpha=0.9,
            identity_line=True,
            identity_line_color="0.75",
            identity_line_style="--",
            x_label="Reference",
            y_label="Comparison",
            figsize=[8.0, 4.0],
            title="Facet Grid",
        ),  # type: ignore
    )

    titles = [ax.get_title() for ax in out.images[0].axes[:2]]
    assert titles == ["A title", "B\nn=3"]


def test_visualization_blocks_share_export_params() -> None:
    visualization_blocks = [
        ClusterHeatmap,
        FacetedScatterPlot,
        HighlightedBarChart,
        HighlightedScatterPlot,
        MatrixHeatmap,
        MatrixHistogram,
        MatrixBarChart,
        MatrixLineChart,
        MatrixScatterPlot,
        Matrix3DScatterPlot,
    ]

    for block_cls in visualization_blocks:
        assert block_cls.Params.model_fields["export_enabled"].default is False
        assert block_cls.Params.model_fields["export_dir"].default is None


def test_basic_visualization_chart_blocks() -> None:
    frame = pd.DataFrame(
        {
            "sample": ["A", "B", "C", "D"],
            "timepoint": [1, 2, 3, 4],
            "value": [0.5, 1.25, 0.9, 1.6],
            "value_2": [2.1, 1.8, 1.9, 2.5],
            "group": ["g1", "g1", "g2", "g2"],
            "size": [5, 8, 10, 7],
        }
    )

    bar = MatrixBarChart().execute(
        frame,
        SimpleNamespace(
            x_column="sample",
            y_column="value",
            top_n=3,
            sort_by_y=True,
            ascending=False,
            figsize=[10.0, 5.0],
            title="bar",
            color="steelblue",
            rotation=30.0,
        ),  # type: ignore
    )
    assert len(bar.images) == 1
    assert bar.metadata["n_rows_plotted"] == 3

    line = MatrixLineChart().execute(
        frame,
        SimpleNamespace(
            x_column="timepoint",
            y_column="value",
            group_column="group",
            sort_by_x=True,
            marker="o",
            figsize=[10.0, 5.0],
            title="line",
        ),  # type: ignore
    )
    assert len(line.images) == 1
    assert line.metadata["group_column"] == "group"

    scatter = MatrixScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="value",
            y_column="value_2",
            color_column="group",
            size_column="size",
            marker_size=40.0,
            alpha=0.7,
            cmap="viridis",
            figsize=[10.0, 5.0],
            title="scatter",
        ),  # type: ignore
    )
    assert len(scatter.images) == 1
    assert scatter.metadata["n_rows_plotted"] == frame.shape[0]

    plotly = pytest.importorskip("plotly")
    assert plotly is not None
    scatter3d = Matrix3DScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="timepoint",
            y_column="value",
            z_column="value_2",
            color_column="group",
            size_column="size",
            marker_size=6.0,
            opacity=0.8,
            title="scatter3d",
        ),  # type: ignore
    )
    assert len(scatter3d.images) == 1
    assert scatter3d.metadata["render_backend"] == "plotly"


def test_matplotlib_visualization_exports_png_when_enabled(tmp_path) -> None:  # type: ignore
    frame = pd.DataFrame({"value": [0.5, 1.25, 0.9, 1.6]})
    export_dir = tmp_path / "exports"

    out = MatrixHistogram().execute(
        frame,
        SimpleNamespace(
            column_name="value",
            bucket_size=None,
            skip_nulls=True,
            plot_title="Exported Histogram",
            export_enabled=True,
            export_dir=str(export_dir),
        ),  # type: ignore
    )

    expected_path = export_dir / "Exported Histogram.png"
    assert expected_path.exists()
    assert out.metadata["plot_title"] == "Exported Histogram"
    assert out.metadata["exported_path"] == str(expected_path)


def test_matplotlib_visualization_resolves_relative_export_dir_from_workspace(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore
    frame = pd.DataFrame({"value": [0.5, 1.25, 0.9, 1.6]})
    relative_export_dir = Path("outputs") / "ensemble_expansion_pipeline" / "viz"
    monkeypatch.setenv("FORGE_WORKSPACE_DIR", str(tmp_path))

    out = MatrixHistogram().execute(
        frame,
        SimpleNamespace(
            column_name="value",
            bucket_size=None,
            skip_nulls=True,
            plot_title="Exported Histogram",
            export_enabled=True,
            export_dir=relative_export_dir.as_posix(),
        ),  # type: ignore
    )

    expected_path = tmp_path / relative_export_dir / "Exported Histogram.png"
    assert expected_path.exists()
    assert out.metadata["plot_title"] == "Exported Histogram"
    assert out.metadata["exported_path"] == str(expected_path)


def test_plotly_visualization_exports_html_when_enabled(tmp_path) -> None:  # type: ignore
    plotly = pytest.importorskip("plotly")
    assert plotly is not None

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [1.0, 0.5, 1.5],
            "z": [2.0, 2.5, 3.0],
        }
    )
    export_dir = tmp_path / "exports"

    out = Matrix3DScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            z_column="z",
            color_column="",
            color_mode="auto",
            size_column="",
            marker_size=6.0,
            opacity=0.8,
            title="Exported 3D Scatter",
            export_enabled=True,
            export_dir=str(export_dir),
        ),  # type: ignore
    )

    expected_path = export_dir / "Exported 3D Scatter.html"
    assert expected_path.exists()
    assert out.metadata["plot_title"] == "Exported 3D Scatter"
    assert out.metadata["exported_path"] == str(expected_path)


def test_matrix_line_chart_uses_equidistant_index_when_x_column_missing() -> None:
    frame = pd.DataFrame(
        {
            "value": [10.0, 20.0, 30.0],
            "group": ["g1", "g1", "g1"],
        },
        index=[5, 1, 3],
    )

    out = MatrixLineChart().execute(
        frame,
        SimpleNamespace(
            x_column="",
            y_column="value",
            group_column="",
            sort_by_x=True,
            marker="o",
            figsize=[8.0, 4.0],
            title="",
        ),  # type: ignore
    )

    assert len(out.images) == 1
    assert out.metadata["x_column"] is None
    assert out.metadata["x_mode"] == "index_position"
    assert out.metadata["n_rows_plotted"] == 3

    fig = out.images[0]
    line = fig.axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(line.get_ydata(), np.array([20.0, 30.0, 10.0]))


def test_matrix_scatter_numeric_color_handles_negative_and_null_values() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 0.0, -1.0, 2.0, -2.0],
            "color_metric": [-2.5, -1.0, 0.0, 1.5, np.nan],
            "size": [5.0, 8.0, 10.0, 7.0, 9.0],
        }
    )

    scatter = MatrixScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            color_column="color_metric",
            size_column="size",
            marker_size=30.0,
            alpha=0.7,
            cmap="viridis",
            figsize=[8.0, 5.0],
            title="scatter-negative-color",
        ),  # type: ignore
    )
    assert len(scatter.images) == 1
    assert scatter.metadata["color_mode_used"] == "numeric"
    assert scatter.metadata["n_rows_plotted"] == frame.shape[0]


def test_matrix_scatter_string_color_uses_categorical_legend_in_auto_mode() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [1.0, 0.0, -1.0, 2.0],
            "cluster_id": pd.Series(["0", "0", "1", "1"], dtype="string"),
        }
    )

    scatter = MatrixScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            color_column="cluster_id",
            color_mode="auto",
            size_column="",
            marker_size=80.0,
            alpha=0.9,
            cmap="tab10",
            figsize=[8.0, 5.0],
            title="scatter-categorical-auto",
        ),  # type: ignore
    )

    assert len(scatter.images) == 1
    assert scatter.metadata["color_mode_used"] == "categorical"
    fig = scatter.images[0]
    assert len(fig.axes) == 1
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["0", "1"]


def test_matrix_scatter_categorical_draws_null_group_first() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [1.0, 0.0, -1.0, 2.0],
            "cluster_id": [None, "alpha", "beta", None],
        }
    )

    scatter = MatrixScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            color_column="cluster_id",
            color_mode="categorical",
            size_column="",
            marker_size=80.0,
            alpha=0.9,
            cmap="tab10",
            figsize=[8.0, 5.0],
            title="scatter-categorical-null-first",
        ),  # type: ignore
    )

    fig = scatter.images[0]
    collection_labels = [
        collection.get_label() for collection in fig.axes[0].collections
    ]
    assert collection_labels == ["null", "alpha", "beta"]

    null_offsets = fig.axes[0].collections[0].get_offsets()
    np.testing.assert_allclose(null_offsets, np.array([[0.0, 1.0], [3.0, 2.0]]))

    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == ["null", "alpha", "beta"]


def test_annotate_plot_with_arrows_draws_expected_arrow_count() -> None:
    nodes = pd.DataFrame(
        {
            "node_id": ["A", "B", "C", "D"],
            "umap_x": [0.0, 1.0, 2.0, 1.5],
            "umap_y": [0.0, 1.0, 0.5, 2.0],
            "cluster": [None, "alpha", "beta", "alpha"],
            "point_scale": [30.0, 45.0, np.nan, 60.0],
        }
    )
    edges = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "dest_1": ["B", "C", None],
            "dest_2": ["C", "D", None],
        }
    )

    out = AnnotatePlotWithArrows().execute(
        [nodes, edges],
        SimpleNamespace(
            x_column="umap_x",
            y_column="umap_y",
            label_column="node_id",
            color_column="cluster",
            color_mode="categorical",
            size_column="point_scale",
            marker_size=55.0,
            alpha=0.85,
            cmap="tab10",
            point_color="steelblue",
            show_node_labels=False,
            label_fontsize=9.0,
            arrow_color="black",
            arrow_alpha=0.7,
            arrow_linewidth=1.2,
            arrow_style="->",
            figsize=[8.0, 5.0],
            title="Annotated Graph",
        ),  # type: ignore
    )

    assert len(out.images) == 1
    assert out.data.equals(nodes)
    assert out.metadata["color_column"] == "cluster"
    assert out.metadata["color_mode_used"] == "categorical"
    assert out.metadata["label_column"] == "node_id"
    assert out.metadata["size_column"] == "point_scale"
    assert out.metadata["n_nodes_plotted"] == 4
    assert out.metadata["n_edge_rows_processed"] == 2
    assert out.metadata["n_arrows_drawn"] == 4

    fig = out.images[0]
    collection_labels = [
        collection.get_label() for collection in fig.axes[0].collections
    ]
    assert collection_labels == ["null", "alpha", "beta"]
    annotation_zorders = [
        text.arrow_patch.get_zorder()
        for text in fig.axes[0].texts
        if getattr(text, "arrow_patch", None) is not None
    ]
    assert annotation_zorders == [3, 3, 3, 3]
    assert all(
        zorder > collection.get_zorder()
        for zorder in annotation_zorders
        for collection in fig.axes[0].collections
    )
    np.testing.assert_allclose(
        fig.axes[0].collections[0].get_offsets(),
        np.array([[0.0, 0.0]]),
    )
    np.testing.assert_allclose(fig.axes[0].collections[0].get_sizes(), np.array([30.0]))
    np.testing.assert_allclose(
        fig.axes[0].collections[1].get_sizes(),
        np.array([45.0, 60.0]),
    )
    np.testing.assert_allclose(fig.axes[0].collections[2].get_sizes(), np.array([55.0]))


def test_annotate_plot_with_arrows_rejects_unknown_edge_labels() -> None:
    nodes = pd.DataFrame(
        {
            "node_id": ["A", "B"],
            "umap_x": [0.0, 1.0],
            "umap_y": [0.0, 1.0],
        }
    )
    edges = pd.DataFrame({"source": ["A"], "dest_1": ["missing"]})

    with pytest.raises(BlockValidationError, match="could not resolve"):
        AnnotatePlotWithArrows().execute(
            [nodes, edges],
            SimpleNamespace(
                x_column="umap_x",
                y_column="umap_y",
                label_column="node_id",
                color_column="",
                color_mode="auto",
                size_column="",
                marker_size=45.0,
                alpha=0.9,
                cmap="viridis",
                point_color="steelblue",
                show_node_labels=False,
                label_fontsize=9.0,
                arrow_color="black",
                arrow_alpha=0.7,
                arrow_linewidth=1.2,
                arrow_style="->",
                figsize=[8.0, 5.0],
                title="Annotated Graph",
            ),  # type: ignore
        )


def test_matrix_scatter_and_3d_accept_index_color_column() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [1.0, 0.0, -1.0, 2.0],
            "z": [0.5, 0.6, 0.7, 0.8],
            "size": [5.0, 8.0, 10.0, 7.0],
        },
        index=[101, 104, 109, 120],
    )

    scatter = MatrixScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            color_column="index",
            size_column="size",
            marker_size=30.0,
            alpha=0.7,
            cmap="viridis",
            figsize=[8.0, 5.0],
            title="scatter-index-color",
        ),  # type: ignore
    )
    assert len(scatter.images) == 1
    assert scatter.metadata["color_column"] == "index"
    assert scatter.metadata["color_mode_used"] == "numeric"
    assert scatter.metadata["n_rows_plotted"] == frame.shape[0]

    plotly = pytest.importorskip("plotly")
    assert plotly is not None
    scatter3d = Matrix3DScatterPlot().execute(
        frame,
        SimpleNamespace(
            x_column="x",
            y_column="y",
            z_column="z",
            color_column="index",
            size_column="size",
            marker_size=6.0,
            opacity=0.8,
            title="scatter3d-index-color",
        ),  # type: ignore
    )
    assert len(scatter3d.images) == 1
    assert scatter3d.metadata["color_column"] == "index"
    assert scatter3d.metadata["color_mode_used"] == "numeric"
    assert scatter3d.metadata["render_backend"] == "plotly"
