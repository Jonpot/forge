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

    def __post_init__(self) -> None:
        normalized = {str(key): value for key, value in self.outputs.items()}
        if "output_0" not in normalized:
            normalized["output_0"] = self.data
        self.outputs = normalized
        self.data = normalized["output_0"]


class BlockValidationError(Exception):
    pass


class InsufficientInputs(Exception):
    """Raised by a block when it cannot run because required inputs are not connected.

    The engine catches this and silently skips the block, exactly as if it had
    never been reached in the topological order.  Raise this from validate() or
    execute() instead of BlockValidationError so the pipeline is not marked as
    failed — it simply leaves this node unrun.
    """
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

    @abstractmethod
    def execute(self, data: Any, params: Any | None = None) -> BlockOutput:
        """Execute block logic and return a BlockOutput."""

    def validate(self, data: Any) -> None:
        """Override to raise InsufficientInputs when inputs are missing, or BlockValidationError for data issues.

        The default implementation raises InsufficientInputs when data is None
        (for n_inputs=1 blocks) so that an unconnected block is skipped rather
        than crashing.  Multi-input blocks should override this to check each
        slot individually.
        """
        n = getattr(self.__class__, "n_inputs", 1)
        if n == 1 and data is None:
            raise InsufficientInputs(
                f"'{getattr(self.__class__, 'name', self.__class__.__name__)}' "
                "requires an input but none is connected."
            )

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


__all__ = [
    "BaseBlock",
    "BlockParams",
    "BlockOutput",
    "BlockValidationError",
    "InsufficientInputs",
    "BrowseMode",
    "ProgressBar",
    "block_param",
    "progress_iter",
]
