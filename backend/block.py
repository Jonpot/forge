from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, get_origin

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import PydanticUndefined  # pyright: ignore[reportPrivateImportUsage]
from backend.progress import ProgressBar, progress_iter


@dataclass(slots=True)
class BlockOutput:
    data: pd.DataFrame
    outputs: dict[str, pd.DataFrame] = field(default_factory=dict)
    images: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized = {str(key): value for key, value in self.outputs.items()}
        if "output_0" not in normalized:
            normalized["output_0"] = self.data
        self.outputs = normalized
        self.data = normalized["output_0"]


class BlockValidationError(Exception):
    pass


BrowseMode = Literal["open_file", "save_file", "directory"]


def _annotation_allows_none(annotation: Any) -> bool:
    if annotation is Any or annotation is None or annotation is type(None):
        return True

    origin = get_origin(annotation)
    if origin is None:
        return False

    return any(_annotation_allows_none(arg) for arg in get_args(annotation))


class BlockParams(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_defaulted_nulls(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        for key, payload_field in cls.model_fields.items():
            if key not in payload or payload[key] is not None:
                continue
            if payload_field.is_required():
                continue
            if _annotation_allows_none(payload_field.annotation):
                continue
            payload[key] = deepcopy(
                payload_field.get_default(call_default_factory=True)
            )

        return payload


def block_param(
    default: Any = PydanticUndefined,
    *,
    description: str | None = None,
    example: Any = PydanticUndefined,
    browse_mode: BrowseMode | None = None,
) -> Any:
    field_kwargs: dict[str, Any] = {}
    if description is not None and description.strip():
        field_kwargs["description"] = description.strip()
    if example is not PydanticUndefined:
        field_kwargs["examples"] = [deepcopy(example)]
    if browse_mode is not None:
        field_kwargs["json_schema_extra"] = {"browse_mode": browse_mode}
    if default is PydanticUndefined:
        return Field(**field_kwargs)
    return Field(default=deepcopy(default), **field_kwargs)


class BaseBlock(ABC):
    name: str
    version: str
    category: str
    description: str = ""
    param_descriptions: dict[str, str] = {}
    usage_notes: list[str] | str = []
    presets: list[dict[str, Any]] = []
    n_inputs: int = 1
    input_labels: list[str] = []
    output_labels: list[str] = ["output"]
    always_execute: bool = False
    # When True, the block's BlockOutput.namespace is persisted alongside the
    # parquet checkpoint and made available to namespace-consuming descendants.
    produces_namespace: bool = False
    # When True, _resolve_input_data passes parent namespaces (synthesizing
    # {"data": parent_df} for non-namespace-producing parents) instead of
    # raw DataFrames.
    consumes_namespace: bool = False
    # If set, the param name that drives n_inputs at runtime (frontend uses this
    # to recompute handle count when the user edits the param).
    arity_input_param: str | None = None
    # If set, the param name that drives n_outputs at runtime.
    arity_output_param: str | None = None

    @abstractmethod
    def execute(self, data: Any, params: Any | None = None) -> BlockOutput:
        """Execute block logic and return a BlockOutput."""

    def validate(self, data: Any) -> None:
        """Override to raise BlockValidationError when preconditions fail."""

    @classmethod
    def normalize_params_payload(
        cls, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if params is None:
            return {}
        return deepcopy(params)

    @classmethod
    def should_force_execute(cls, params: dict[str, Any] | None = None) -> bool:
        return bool(getattr(cls, "always_execute", False))

    @classmethod
    def resolve_n_inputs(cls, params: dict[str, Any] | None = None) -> int:
        del params
        return int(getattr(cls, "n_inputs", 1))

    @classmethod
    def resolve_input_labels(
        cls, params: dict[str, Any] | None = None
    ) -> list[str]:
        del params
        labels = getattr(cls, "input_labels", [])
        if isinstance(labels, (list, tuple)):
            return [str(item) for item in labels]
        return []

    @classmethod
    def resolve_output_labels(
        cls, params: dict[str, Any] | None = None
    ) -> list[str]:
        del params
        labels = getattr(cls, "output_labels", ["output"])
        if not isinstance(labels, (list, tuple)) or not labels:
            return ["output"]
        normalized = [str(item) for item in labels if str(item).strip()]
        return normalized or ["output"]


__all__ = [
    "BaseBlock",
    "BlockParams",
    "BlockOutput",
    "BlockValidationError",
    "BrowseMode",
    "ProgressBar",
    "block_param",
    "progress_iter",
]
