"""Typed serializer registry for checkpoint outputs (DESIGN.md §8).

Probed in order per value: parquet (DataFrame/Series) → npy (ndarray) → json
(plain data) → pickle (opt-in, default OFF) → ephemeral.

Ephemeral values flow normally to downstream nodes within the same run but
are not persisted, so future runs recompute them on demand. That keeps the
cost of unserializable types localized instead of poisoning the store.

Import discipline: this module never imports pandas/numpy itself. If a value
IS a DataFrame, pandas is by definition already imported in this process —
we detect by the type's module name first, then import for free.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

EPHEMERAL = "ephemeral"


class EphemeralValueError(RuntimeError):
    """Raised when loading an output that was never persisted."""


def _root_type_module(value: Any) -> str:
    return type(value).__module__.split(".")[0]


def _meta(value: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and all(isinstance(d, int) for d in shape):
        meta["shape"] = list(shape)
    elif isinstance(value, (list, dict, str)):
        meta["len"] = len(value)
    return meta


def save_value(
    value: Any,
    outputs_dir: Path,
    name: str,
    pickle_enabled: bool = False,
) -> dict[str, Any]:
    """Persist one output value; returns its manifest entry."""
    entry: dict[str, Any] = {"name": name, "meta": _meta(value)}
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if _root_type_module(value) == "pandas":
        import pandas as pd

        frame = value
        if isinstance(value, pd.Series):
            frame = value.to_frame(name=value.name if value.name is not None else "value")
            entry["meta"]["series"] = True
        if isinstance(frame, pd.DataFrame):
            try:
                filename = f"{name}.parquet"
                frame.to_parquet(outputs_dir / filename)
                entry.update(serializer="parquet", file=filename)
                return entry
            except (ImportError, ValueError, OSError):
                pass  # no pyarrow, or unserializable dtypes — fall through

    if _root_type_module(value) == "numpy":
        import numpy as np

        if isinstance(value, np.ndarray):
            try:
                filename = f"{name}.npy"
                np.save(outputs_dir / filename, value, allow_pickle=False)
                entry.update(serializer="npy", file=filename)
                return entry
            except (ValueError, OSError):
                pass

    try:
        text = json.dumps(value)
        filename = f"{name}.json"
        (outputs_dir / filename).write_text(text, encoding="utf-8")
        entry.update(serializer="json", file=filename)
        return entry
    except (TypeError, ValueError, OSError):
        pass

    # Figures can't round-trip from their rendered form, so the VALUE is
    # ephemeral (flows within a run; downstream recomputes later via the
    # cascade) — but it leaves a rendered artifact behind for display.
    from starforge.core import figures

    try:
        artifact = figures.render_figure(value, outputs_dir, name)
    except Exception:
        artifact = None
    if artifact is not None:
        entry.update(serializer=EPHEMERAL, file=None, artifact=artifact)
        return entry

    if pickle_enabled:
        try:
            filename = f"{name}.pkl"
            with (outputs_dir / filename).open("wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            entry.update(serializer="pickle", file=filename)
            return entry
        except Exception:
            pass

    entry.update(serializer=EPHEMERAL, file=None)
    return entry


def load_value(outputs_dir: Path, entry: dict[str, Any]) -> Any:
    serializer = entry.get("serializer")
    name = entry.get("name")
    if serializer == EPHEMERAL:
        raise EphemeralValueError(
            f"output '{name}' was not persisted (ephemeral); the producing node must re-run"
        )
    path = outputs_dir / entry["file"]
    if serializer == "parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        if entry.get("meta", {}).get("series"):
            return frame.iloc[:, 0]
        return frame
    if serializer == "npy":
        import numpy as np

        return np.load(path, allow_pickle=False)
    if serializer == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if serializer == "pickle":
        with path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"unknown serializer {serializer!r} for output '{name}'")
