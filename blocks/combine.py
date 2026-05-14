from __future__ import annotations

import pandas as pd

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    InsufficientInputs,
    block_param,
)


class AppendDatasets(BaseBlock):
    name = "Append Datasets"
    version = "1.0.0"
    category = "Operator"
    description = "Append two DataFrames row-wise."
    n_inputs = 2
    input_labels = ["Left", "Right"]
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        ignore_index: bool = block_param(True, description="Whether to reindex appended rows from 0..n-1.")

    def validate(self, data) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("AppendDatasets requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("AppendDatasets requires params.")
        appended = pd.concat([data[0], data[1]], ignore_index=params.ignore_index)
        return BlockOutput(data=appended)


class MergeDatasets(BaseBlock):
    name = "Merge Datasets"
    version = "1.0.0"
    category = "Operator"
    description = "Merge two DataFrames using pandas merge semantics."
    n_inputs = 2
    input_labels = ["Left", "Right"]
    output_labels = ["DataFrame"]
    usage_notes = [
        "Input 0 is treated as the left table and input 1 as the right table.",
        "Rows are matched by exact equality on the configured join key column.",
        "Columns from both inputs are preserved using pandas merge semantics.",
    ]

    class Params(BlockParams):
        on: str = block_param(
            description="Join key column present in both inputs.",
            example="id",
        )
        how: str = block_param(
            "inner",
            description="Join mode. One of: inner, left, right, outer.",
            example="inner",
        )

    def validate(self, data) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("MergeDatasets requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MergeDatasets requires params.")
        merged = data[0].merge(data[1], on=params.on, how=params.how)  # pyright: ignore[reportArgumentType]
        return BlockOutput(data=merged)
