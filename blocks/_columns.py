"""Shared helpers for column-key params.

Blocks that take a DataFrame column name as a parameter accept the literal
sentinel ``"index"`` to mean the DataFrame's index instead of a named column.
The sentinel is strict (case-sensitive, trimmed) and always wins, even if a
column literally named ``index`` exists on the frame — users with such a
column should rename or reset_index first.

This module is leading-underscore so the block registry's auto-discovery
skips it.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from backend.block import BlockValidationError


INDEX_KEY = "index"


def is_index_key(key: Any) -> bool:
    """True when *key* is the index sentinel after str-coercion + strip."""
    if key is None:
        return False
    return str(key).strip() == INDEX_KEY


def column_exists(df: pd.DataFrame, key: str) -> bool:
    """True if *key* is the index sentinel or names a column on *df*."""
    if is_index_key(key):
        return True
    return key in df.columns


def require_column(
    df: pd.DataFrame,
    key: Any,
    block_name: str,
    label: str,
) -> str:
    """Validate *key* refers to a usable column or the index sentinel.

    Returns the cleaned key (the canonical ``"index"`` sentinel or the
    column name). Raises ``BlockValidationError`` on missing/empty input
    or an unknown column name.
    """
    clean = str(key or "").strip()
    if not clean:
        raise BlockValidationError(f"{block_name}: {label} is required.")
    if is_index_key(clean):
        return INDEX_KEY
    if clean not in df.columns:
        raise BlockValidationError(
            f"{block_name}: column '{clean}' not found."
        )
    return clean


def require_columns(
    df: pd.DataFrame,
    keys: Iterable[Any],
    block_name: str,
    label: str,
) -> list[str]:
    """Validate every key in *keys* (skipping blanks). Returns the cleaned
    list preserving input order; duplicates are kept."""
    cleaned: list[str] = []
    missing: list[str] = []
    for raw in keys:
        text = str(raw or "").strip()
        if not text:
            continue
        if is_index_key(text):
            cleaned.append(INDEX_KEY)
            continue
        if text not in df.columns:
            missing.append(text)
            continue
        cleaned.append(text)
    if missing:
        raise BlockValidationError(
            f"{block_name}: missing columns for {label}: {missing}"
        )
    return cleaned


def column_values(df: pd.DataFrame, key: str) -> pd.Series:
    """Return the Series for *key*. For the sentinel, returns the DataFrame
    index materialized as a Series with name=``"index"``."""
    if is_index_key(key):
        return pd.Series(df.index, index=df.index, name=INDEX_KEY)
    return df[key]


def groupby_key(df: pd.DataFrame, key: str) -> Any:
    """Return a value usable as a single grouping key in ``df.groupby(...)``.
    For the sentinel, returns ``df.index`` (which pandas accepts directly)."""
    if is_index_key(key):
        return df.index
    return key


def groupby_keys(df: pd.DataFrame, keys: list[str]) -> list[Any]:
    """Translate a list of keys for ``df.groupby(...)``. The index sentinel
    becomes ``df.index`` in the returned list."""
    return [groupby_key(df, k) for k in keys]


def select_columns_with_index(
    df: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Return a DataFrame containing the requested columns, materializing
    the DataFrame index as an ``"index"`` column when the sentinel appears
    in *keys*. The returned frame is always a copy."""
    if not any(is_index_key(k) for k in keys):
        return df[keys].copy()
    out = pd.DataFrame(index=df.index)
    for k in keys:
        out[INDEX_KEY if is_index_key(k) else k] = column_values(df, k)
    return out


def with_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow copy of *df* with an extra ``"index"`` column holding
    the original row index. Useful for handing off to pandas APIs that only
    accept column names (e.g. ``melt(id_vars=...)``). Uses positional ndarray
    assignment so that frames with duplicate index values are handled correctly.
    """
    out = df.copy()
    out[INDEX_KEY] = df.index.to_numpy()
    return out


__all__ = [
    "INDEX_KEY",
    "column_exists",
    "column_values",
    "groupby_key",
    "groupby_keys",
    "is_index_key",
    "require_column",
    "require_columns",
    "select_columns_with_index",
    "with_index_column",
]
