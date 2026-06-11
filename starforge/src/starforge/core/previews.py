"""Cropped, JSON-safe output previews, computed at checkpoint-write time.

Previews are precomputed artifacts stored inside ``provenance.json`` — the
kernel serves them by reading a file, never by deserializing data (it stays
stdlib-only and instant). Because they're built while the value is in the
worker's hands, even EPHEMERAL outputs get a preview of their last run.

Everything emitted here must survive strict JSON.parse on the TypeScript
side: NaN/Infinity are stringified, containers are size-capped, and unknown
objects fall back to repr.
"""

from __future__ import annotations

import json
from typing import Any

MAX_ROWS = 8
MAX_COLS = 10
MAX_ITEMS = 50
MAX_DEPTH = 5
MAX_CELL_CHARS = 120
MAX_TEXT_CHARS = 600
MAX_VALUE_CHARS = 2000


def _cell(value: Any) -> Any:
    """One scalar table/array cell, strict-JSON safe."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    text = value if isinstance(value, str) else repr(value)
    return text[:MAX_CELL_CHARS] + ("…" if len(text) > MAX_CELL_CHARS else "")


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth >= MAX_DEPTH:
        return _cell(value)
    if isinstance(value, dict):
        items = list(value.items())[:MAX_ITEMS]
        out = {str(k)[:MAX_CELL_CHARS]: _sanitize(v, depth + 1) for k, v in items}
        if len(value) > MAX_ITEMS:
            out["…"] = f"+{len(value) - MAX_ITEMS} more"
        return out
    if isinstance(value, (list, tuple)):
        out = [_sanitize(v, depth + 1) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            out.append(f"… +{len(value) - MAX_ITEMS} more")
        return out
    return _cell(value)


def _root_type_module(value: Any) -> str:
    return type(value).__module__.split(".")[0]


def build_preview(value: Any) -> dict[str, Any]:
    if _root_type_module(value) == "pandas":
        import pandas as pd

        frame = value.to_frame() if isinstance(value, pd.Series) else value
        if isinstance(frame, pd.DataFrame):
            columns = [str(c) for c in frame.columns[:MAX_COLS]]
            head = frame.iloc[:MAX_ROWS, :MAX_COLS]
            return {
                "kind": "table",
                "shape": [int(frame.shape[0]), int(frame.shape[1])],
                "columns": columns,
                "columns_truncated": frame.shape[1] > MAX_COLS,
                "index": [_cell(i) for i in head.index.tolist()],
                "rows": [[_cell(v) for v in row] for row in head.itertuples(index=False, name=None)],
            }

    if _root_type_module(value) == "numpy":
        import numpy as np

        if isinstance(value, np.ndarray):
            corner = value
            if corner.ndim == 0:
                corner_list: Any = _cell(corner.item())
            else:
                slicer = tuple(slice(0, MAX_ROWS) for _ in range(corner.ndim))
                corner_list = _sanitize(corner[slicer].tolist())
            return {
                "kind": "array",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "corner": corner_list,
            }
        if isinstance(value, np.generic):
            return {"kind": "value", "value": _cell(value.item())}

    if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
        sanitized = _sanitize(value)
        try:
            encoded = json.dumps(sanitized, allow_nan=False)
        except (TypeError, ValueError):
            encoded = None
        if encoded is not None:
            if len(encoded) > MAX_VALUE_CHARS:
                return {"kind": "text", "text": encoded[:MAX_VALUE_CHARS] + "…"}
            return {"kind": "value", "value": sanitized}

    # Arbitrary objects: an honest repr, marked as text rather than data.
    text = repr(value)
    return {"kind": "text", "text": text[:MAX_TEXT_CHARS] + ("…" if len(text) > MAX_TEXT_CHARS else "")}
