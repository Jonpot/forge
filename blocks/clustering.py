from __future__ import annotations

from typing import Any

import pandas as pd

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    block_param,
)
from blocks._columns import column_values, is_index_key


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


class KMeansClustering(BaseBlock):
    name = "K-Means Clustering"
    version = "1.0.0"
    category = "Clustering"
    description = "Cluster rows with K-Means and append a cluster assignment column."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame + Cluster"]
    usage_notes = [
        "When `columns` is empty, the block uses every numeric column in the input frame.",
        "Matching `column_prefix` columns are added to the explicit column list before fitting.",
        "The output preserves the input rows and appends one cluster label column.",
    ]
    presets = [
        {
            "id": "three_clusters",
            "label": "Three Clusters",
            "description": "Small deterministic clustering setup for quick exploration.",
            "params": {
                "n_clusters": 3,
                "random_state": 0,
                "output_column": "cluster_id",
            },
        }
    ]

    class Params(BlockParams):
        n_clusters: int = block_param(
            6,
            description="Number of clusters to fit.",
            example=3,
        )
        random_state: int = block_param(
            0,
            description="Random seed for reproducibility.",
            example=0,
        )
        columns: str | None = block_param(
            None,
            description="Comma-separated numeric columns to use. Leave empty to use all numeric columns.",
            example="feature_a,feature_b,feature_c",
        )
        column_prefix: str = block_param(
            "",
            description="Optional prefix used to include every matching column in the clustering input.",
            example="feature_",
        )
        standardize: bool = block_param(
            False,
            description="Standardize selected columns before fitting K-Means.",
            example=False,
        )
        output_column: str = block_param(
            "cluster_id",
            description="Name of the cluster assignment column to append.",
            example="cluster_id",
        )

    def validate(self, data: pd.DataFrame) -> None:
        numeric_cols = data.select_dtypes(include="number").columns
        if len(numeric_cols) < 1:
            raise BlockValidationError("KMeansClustering requires numeric columns.")

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("KMeansClustering requires params.")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        selected_cols = _parse_columns(params.columns)
        if params.column_prefix:
            prefixed = [col for col in data.columns if str(col).startswith(params.column_prefix)]
            selected_cols = sorted(set(selected_cols + prefixed))
        if not selected_cols:
            selected_cols = numeric_cols
        selected_cols = [
            col for col in selected_cols if is_index_key(col) or col in data.columns
        ]
        if len(selected_cols) == 0:
            raise BlockValidationError("No columns selected for clustering.")
        if params.n_clusters <= 0:
            raise BlockValidationError("n_clusters must be > 0.")

        feature_series = [
            pd.to_numeric(column_values(data, c), errors="coerce") for c in selected_cols
        ]
        matrix = pd.concat(feature_series, axis=1).to_numpy(dtype=float)
        if params.standardize:
            matrix = StandardScaler(with_mean=True, with_std=True).fit_transform(matrix)

        km = KMeans(
            n_clusters=params.n_clusters,
            random_state=params.random_state,
            n_init=20,
        )
        labels = km.fit_predict(matrix)
        result = data.copy()
        result[params.output_column] = labels
        return BlockOutput(
            data=result,
            metadata={"inertia": float(km.inertia_), "columns_used": selected_cols},
        )
