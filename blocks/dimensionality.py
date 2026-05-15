from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    ProgressBar,
    block_param,
)
from blocks._columns import column_values, require_columns


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


class UMAPEmbed(BaseBlock):
    name = "UMAP Embed"
    version = "1.0.0"
    category = "Dimensionality"
    description = "Compute UMAP embeddings from selected numeric columns."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame + Embedding"]
    usage_notes = [
        "When `columns` is empty, the block embeds every numeric column in the input frame.",
        "The output preserves the input rows and appends one embedding column per component.",
        "UMAP is scale-sensitive, so upstream normalization is often helpful before this block.",
    ]

    class Params(BlockParams):
        columns: str = block_param(
            "",
            description="Comma-separated numeric columns to embed. Leave empty to use all numeric columns.",
            example="feature_a,feature_b,feature_c",
        )
        n_components: int = block_param(
            2,
            description="Number of embedding dimensions to generate.",
            example=2,
        )
        n_neighbors: int = block_param(
            15,
            description="Number of neighbors used when constructing the local UMAP graph.",
            example=15,
        )
        min_dist: float = block_param(
            0.1,
            description="Minimum embedded distance between nearby points.",
            example=0.1,
        )
        random_state: int = block_param(
            0,
            description="Random seed for reproducibility.",
            example=0,
        )
        prefix: str = block_param(
            "UMAP_",
            description="Prefix used when naming generated embedding columns.",
            example="UMAP_",
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("UMAPEmbed requires params.")
        try:
            import umap  # type: ignore[import-not-found]
        except Exception as exc:
            raise BlockValidationError(
                "UMAPEmbed requires 'umap-learn'. Install it in your environment."
            ) from exc

        raw_cols = _parse_columns(params.columns)
        if not raw_cols:
            selected_cols = data.select_dtypes(include="number").columns.tolist()
        else:
            selected_cols = require_columns(data, raw_cols, "UMAPEmbed", "columns")
        if not selected_cols:
            raise BlockValidationError("No numeric columns available for UMAP.")

        reducer = umap.UMAP(
            n_neighbors=int(params.n_neighbors),
            n_components=int(params.n_components),
            min_dist=float(params.min_dist),
            random_state=int(params.random_state),
        )
        feature_matrix = pd.concat(
            [pd.to_numeric(column_values(data, c), errors="coerce") for c in selected_cols],
            axis=1,
        ).to_numpy(dtype=float)
        emb = reducer.fit_transform(feature_matrix)

        result = data.copy()
        if not isinstance(emb, np.ndarray) or emb.shape[1] != params.n_components:
            raise BlockValidationError("UMAP embedding failed to compute.")
        for idx in ProgressBar(
            range(emb.shape[1]),
            total=int(emb.shape[1]),
            label="Appending embedding columns",
            throttle_seconds=0.2,
        ):
            result[f"{params.prefix}{idx}"] = emb[:, idx]
        return BlockOutput(data=result)
