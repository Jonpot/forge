"""Content-addressed object store for large values inside checkpoint
namespaces (DataFrames, ndarrays).

Phase 5b motivation: a freeform cell's namespace.pkl frequently inherits the
same large DataFrame from a shared parent. Naive dill-pickling re-serializes
the entire DataFrame in every child's pickle. Routing those values through
this store dedupes by content hash — one parquet/npy on disk, many namespaces
that reference it.

A side benefit: every reload from the store yields a fresh deserialized object,
so sibling cells can't leak in-place mutations to one another.

Inline pickling is preserved for everything else (small objects, sklearn
models, dicts, …) by returning None from persistent_id.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

# Below this size (in bytes), inline pickling is cheaper than parquet/npy round-trip
# plus the disk read. Above it, content-addressing wins on dedup and forks.
_MIN_DATAFRAME_BYTES = 256 * 1024  # 256 KB
_MIN_NDARRAY_BYTES = 256 * 1024


class ObjectStore:
    """File-backed content-addressed store. Lives next to checkpoints — same
    parent dir, separate subtree (so checkpoint gc doesn't sweep it)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── DataFrames ──────────────────────────────────────────────────────────

    @staticmethod
    def should_intercept_dataframe(df: pd.DataFrame) -> bool:
        try:
            nbytes = int(df.memory_usage(deep=True).sum())
        except Exception:
            return False
        return nbytes >= _MIN_DATAFRAME_BYTES

    def put_dataframe(self, df: pd.DataFrame) -> str:
        buf = io.BytesIO()
        # index=True so row identity round-trips; matches CheckpointStore.save.
        df.to_parquet(buf, index=True)
        data = buf.getvalue()
        content_hash = hashlib.sha256(data).hexdigest()
        path = self._dataframe_path(content_hash)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return content_hash

    def get_dataframe(self, content_hash: str) -> pd.DataFrame:
        path = self._dataframe_path(content_hash)
        if not path.exists():
            raise FileNotFoundError(
                f"ObjectStore missing dataframe with content hash {content_hash}."
            )
        return pd.read_parquet(path)

    def _dataframe_path(self, content_hash: str) -> Path:
        return self.root / "dataframe" / f"{content_hash}.parquet"

    # ── ndarrays ────────────────────────────────────────────────────────────

    @staticmethod
    def should_intercept_ndarray(arr: np.ndarray) -> bool:
        try:
            return int(arr.nbytes) >= _MIN_NDARRAY_BYTES
        except Exception:
            return False

    def put_ndarray(self, arr: np.ndarray) -> str:
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        data = buf.getvalue()
        content_hash = hashlib.sha256(data).hexdigest()
        path = self._ndarray_path(content_hash)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return content_hash

    def get_ndarray(self, content_hash: str) -> np.ndarray:
        path = self._ndarray_path(content_hash)
        if not path.exists():
            raise FileNotFoundError(
                f"ObjectStore missing ndarray with content hash {content_hash}."
            )
        return np.load(path, allow_pickle=False)

    def _ndarray_path(self, content_hash: str) -> Path:
        return self.root / "ndarray" / f"{content_hash}.npy"

    # ── housekeeping ────────────────────────────────────────────────────────

    def stored_hashes(self) -> dict[str, set[str]]:
        """Return {kind: {hash, hash, …}} for inspection / future GC."""
        result: dict[str, set[str]] = {"dataframe": set(), "ndarray": set()}
        for kind, ext in (("dataframe", ".parquet"), ("ndarray", ".npy")):
            kind_dir = self.root / kind
            if not kind_dir.exists():
                continue
            for path in kind_dir.iterdir():
                if path.suffix == ext:
                    result[kind].add(path.stem)
        return result
