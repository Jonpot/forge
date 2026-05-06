from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    block_param,
)


_MAX_SLOTS = 8

_DEFAULT_CODE = """\
# `data` is the parent's DataFrame (or the variable named `data` from the parent's namespace).
# Define output_0..output_N for downstream regular blocks.
# All variables you define here are visible to downstream freeform blocks.
output_0 = data.copy()
"""


class _SlotProxy:
    """Read-only namespace proxy. Attribute access maps to dict keys.

    Used in FreeformCode cells to expose parent-cell namespaces under stable
    names like `in_0`, `in_1`, … . Reading is direct (`in_0.x`); writing is
    intentionally blocked because the parent's state is upstream and must not
    diverge from its checkpoint.
    """

    __slots__ = ("_slot_name", "_namespace")

    def __init__(self, slot_name: str, namespace: dict[str, Any]) -> None:
        object.__setattr__(self, "_slot_name", slot_name)
        object.__setattr__(self, "_namespace", namespace)

    def __getattr__(self, key: str) -> Any:
        ns = object.__getattribute__(self, "_namespace")
        if key in ns:
            return ns[key]
        slot_name = object.__getattribute__(self, "_slot_name")
        raise AttributeError(
            f"{slot_name} has no variable {key!r} (available: {sorted(ns.keys())})"
        )

    def __setattr__(self, key: str, value: Any) -> None:
        slot_name = object.__getattribute__(self, "_slot_name")
        raise AttributeError(
            f"{slot_name} is read-only — copy a value out before mutating it."
        )

    def __dir__(self) -> list[str]:
        return sorted(object.__getattribute__(self, "_namespace").keys())

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_namespace")

    def __repr__(self) -> str:
        ns = object.__getattribute__(self, "_namespace")
        slot_name = object.__getattribute__(self, "_slot_name")
        keys = list(ns.keys())[:8]
        more = "" if len(ns) <= 8 else f", … +{len(ns) - 8}"
        return f"<{slot_name} keys=[{', '.join(repr(k) for k in keys)}{more}]>"


def _coerce_slot_count(value: Any, *, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n < low:
        return low
    if n > high:
        return high
    return n


class FreeformCode(BaseBlock):
    name = "Freeform Code"
    version = "2.0.0"
    category = "Custom"
    description = (
        "Execute arbitrary Python in a node, like a Jupyter cell. "
        "Inputs and outputs are configurable. The cell's full namespace (every "
        "variable you define) flows to downstream Freeform Code cells."
    )
    usage_notes = [
        "Set `n_inputs` to wire 0+ parents.",
        "Set `n_outputs` to expose 1+ output handles.",
        "Inside the cell: `pd` and `np` are imported. With 1 input, the parent's namespace is spread into your scope (e.g. `data` available directly). With 2+ inputs, access each parent via `in_0`, `in_1`, … (read-only).",
        "Assign `output_0`, `output_1`, … to expose values on the corresponding output handles. Non-DataFrame values go through to namespace consumers but show as empty DataFrames to regular consumers.",
    ]
    n_inputs = 1
    output_labels = ["output_0"]
    produces_namespace = True
    consumes_namespace = True
    arity_input_param = "n_inputs"
    arity_output_param = "n_outputs"

    class Params(BlockParams):
        n_inputs: int = block_param(
            1,
            description=f"Number of input handles (0–{_MAX_SLOTS}).",
            example=2,
        )
        n_outputs: int = block_param(
            1,
            description=f"Number of output handles (1–{_MAX_SLOTS}).",
            example=2,
        )
        code: str = block_param(
            _DEFAULT_CODE,
            description="Python code defining the cell body.",
        )

    @classmethod
    def resolve_n_inputs(cls, params: dict[str, Any] | None = None) -> int:
        params = params or {}
        return _coerce_slot_count(params.get("n_inputs"), default=1, low=0, high=_MAX_SLOTS)

    @classmethod
    def resolve_output_labels(cls, params: dict[str, Any] | None = None) -> list[str]:
        params = params or {}
        n = _coerce_slot_count(params.get("n_outputs"), default=1, low=1, high=_MAX_SLOTS)
        return [f"output_{i}" for i in range(n)]

    @classmethod
    def resolve_input_labels(cls, params: dict[str, Any] | None = None) -> list[str]:
        n = cls.resolve_n_inputs(params)
        return [f"in_{i}" for i in range(n)]

    def execute(self, data: Any, params: Params) -> BlockOutput:
        n_inputs = type(self).resolve_n_inputs(params.model_dump())
        n_outputs = len(type(self).resolve_output_labels(params.model_dump()))

        # Normalize `data` into a list of parent namespaces.
        parent_namespaces: list[dict[str, Any]] = []
        if n_inputs == 1:
            if not isinstance(data, dict):
                raise BlockValidationError(
                    "FreeformCode expected a parent namespace dict for n_inputs=1; "
                    f"received {type(data).__name__}. (consumes_namespace synthesizes "
                    "{'data': <df>} for non-namespace parents.)"
                )
            parent_namespaces = [data]
        elif n_inputs >= 2:
            if not isinstance(data, list):
                raise BlockValidationError(
                    f"FreeformCode expected a list of parent namespaces for n_inputs={n_inputs}; "
                    f"received {type(data).__name__}."
                )
            parent_namespaces = list(data)

        # Build the execution environment. `harness_keys` are the names we
        # injected as scaffolding (pd, np, slot proxies) — these are stripped
        # from the outgoing namespace. Spread parent-namespace keys are NOT
        # harness keys: the cell may have read, modified, or replaced them, and
        # downstream cells should see the cell's final value.
        env: dict[str, Any] = {"pd": pd, "np": np}
        harness_keys: set[str] = {"pd", "np"}

        if n_inputs == 1:
            # Single-input ergonomics: spread the parent namespace into env so the
            # cell reads like a continuation (Jupyter-style). Slot proxy still
            # exposed as `in_0` for explicit access.
            ns0 = parent_namespaces[0]
            env["in_0"] = _SlotProxy("in_0", ns0)
            harness_keys.add("in_0")
            for key, value in ns0.items():
                env[key] = value
        else:
            for idx, ns in enumerate(parent_namespaces):
                slot_name = f"in_{idx}"
                env[slot_name] = _SlotProxy(slot_name, ns)
                harness_keys.add(slot_name)

        code_text = params.code or ""
        try:
            exec(compile(code_text, "<FreeformCode>", "exec"), env)
        except Exception as exc:
            raise BlockValidationError(f"FreeformCode execution failed: {exc}") from exc

        # Collect output frames from declared output_i variables.
        outputs: dict[str, pd.DataFrame] = {}
        for i in range(n_outputs):
            handle = f"output_{i}"
            value = env.get(handle)
            if isinstance(value, pd.DataFrame):
                outputs[handle] = value
            else:
                # Placeholder so regular downstream consumers don't break, while
                # the real value still travels in the namespace.
                outputs[handle] = pd.DataFrame()

        # Capture the full user namespace: everything in env except harness
        # scaffolding and dunder/builtin noise.
        user_namespace: dict[str, Any] = {
            key: value
            for key, value in env.items()
            if key not in harness_keys and not key.startswith("__")
        }

        return BlockOutput(
            data=outputs["output_0"],
            outputs=outputs,
            namespace=user_namespace,
        )
