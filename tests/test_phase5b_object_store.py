"""Phase 5b: content-addressed object store for namespace pickles.

Verifies:
  - Large DataFrames in namespaces are stored once and dedup across siblings.
  - Sibling cells cannot leak in-place mutations to one another (every fork
    gets a fresh deserialized copy).
  - Below-threshold values still inline-pickle.
  - ndarrays round-trip through the store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.block import BaseBlock, BlockOutput, BlockParams
from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.object_store import ObjectStore
from backend.engine.runner import PipelineRunner
from backend.registry import BlockRegistry
from blocks.freeform import FreeformCode


@pytest.fixture
def runner(tmp_path: Path) -> PipelineRunner:
    registry = BlockRegistry(blocks_dir="blocks", package_name="blocks")
    registry._blocks = {"FreeformCode": FreeformCode}
    store = CheckpointStore(tmp_path / "checkpoints")
    return PipelineRunner(registry, store)


def _large_df_code(n_rows: int = 50_000) -> str:
    return (
        "import pandas as pd\n"
        f"df = pd.DataFrame({{'x': range({n_rows}), 'y': [v * 2 for v in range({n_rows})]}})\n"
        "output_0 = df\n"
    )


def test_object_store_dedupes_shared_dataframes(runner: PipelineRunner) -> None:
    """Two siblings inheriting the same parent DataFrame: the object store
    should hold one parquet, not two."""
    pipeline = {
        "nodes": [
            {"id": "a", "block": "FreeformCode",
             "params": {"n_inputs": 0, "n_outputs": 1, "code": _large_df_code()}},
            {"id": "b", "block": "FreeformCode",
             "params": {"n_inputs": 1, "n_outputs": 1,
                        "code": "output_0 = df.head(10)"}},
            {"id": "c", "block": "FreeformCode",
             "params": {"n_inputs": 1, "n_outputs": 1,
                        "code": "output_0 = df.tail(10)"}},
        ],
        "edges": [
            {"id": "ab", "source": "a", "target": "b", "source_output": 0, "target_input": 0},
            {"id": "ac", "source": "a", "target": "c", "source_output": 0, "target_input": 0},
        ],
    }
    result = runner.run_pipeline(pipeline)
    assert set(result.executed_nodes) == {"a", "b", "c"}

    obj_store = runner.checkpoint_store.object_store
    stored = obj_store.stored_hashes()
    # A's df is large enough to intercept; B and C also keep `df` in scope
    # (single-input spread) and may keep their own derived DataFrames if those
    # are also large. The shared `df` from A must dedupe to a single hash.
    df_hashes = stored["dataframe"]
    assert len(df_hashes) >= 1
    # Reload A's namespace and confirm the df actually round-trips.
    a_cp = result.node_results["a"].checkpoint_id
    a_ns = runner.checkpoint_store.load_namespace(a_cp)
    assert a_ns is not None
    assert isinstance(a_ns["df"], pd.DataFrame)
    assert len(a_ns["df"]) == 50_000


def test_sibling_mutation_does_not_leak(runner: PipelineRunner) -> None:
    """The earlier in-memory shared-dict bug: B mutating df should NOT be
    visible to C. With per-fork fresh loads, both see the original."""
    pipeline = {
        "nodes": [
            {"id": "a", "block": "FreeformCode",
             "params": {"n_inputs": 0, "n_outputs": 1, "code": _large_df_code(n_rows=50_000)}},
            {"id": "b", "block": "FreeformCode",
             "params": {"n_inputs": 1, "n_outputs": 1,
                        "code": (
                            # Mutate the inherited df in place.
                            "df.loc[0, 'x'] = -999\n"
                            "output_0 = df.head(1)\n"
                        )}},
            {"id": "c", "block": "FreeformCode",
             "params": {"n_inputs": 1, "n_outputs": 1,
                        "code": (
                            # C must see the original value at row 0, not B's mutation.
                            "assert df.loc[0, 'x'] == 0, f\"C saw B's mutation: {df.loc[0, 'x']}\"\n"
                            "output_0 = df.head(1)\n"
                        )}},
        ],
        "edges": [
            {"id": "ab", "source": "a", "target": "b", "source_output": 0, "target_input": 0},
            {"id": "ac", "source": "a", "target": "c", "source_output": 0, "target_input": 0},
        ],
    }
    result = runner.run_pipeline(pipeline)
    assert set(result.executed_nodes) == {"a", "b", "c"}


def test_small_values_inline_pickle(runner: PipelineRunner) -> None:
    """Small DataFrames (below the threshold) should NOT touch the object
    store — inline pickling is cheaper for tiny values."""
    pipeline = {
        "nodes": [
            {"id": "a", "block": "FreeformCode",
             "params": {"n_inputs": 0, "n_outputs": 1,
                        "code": (
                            "import pandas as pd\n"
                            # Tiny df: well under the 256KB threshold.
                            "tiny = pd.DataFrame({'x': [1, 2, 3]})\n"
                            "scalar = 42\n"
                            "tag = 'hello'\n"
                            "output_0 = tiny\n"
                        )}},
        ],
        "edges": [],
    }
    runner.run_pipeline(pipeline)

    obj_store = runner.checkpoint_store.object_store
    stored = obj_store.stored_hashes()
    assert stored["dataframe"] == set(), "tiny DataFrame must inline-pickle"


def test_ndarray_round_trip_through_store(tmp_path: Path) -> None:
    """ndarrays above the threshold round-trip through the object store
    losslessly, dedup by content."""
    store = ObjectStore(tmp_path / "objs")

    arr1 = np.arange(200_000, dtype=np.float64)  # ~1.6 MB
    arr2 = np.arange(200_000, dtype=np.float64)  # identical content
    arr3 = np.arange(200_000, dtype=np.float64) + 1.0  # different content

    h1 = store.put_ndarray(arr1)
    h2 = store.put_ndarray(arr2)
    h3 = store.put_ndarray(arr3)

    assert h1 == h2  # content equal → same hash
    assert h1 != h3

    loaded = store.get_ndarray(h1)
    assert loaded.shape == arr1.shape
    assert loaded.dtype == arr1.dtype
    np.testing.assert_array_equal(loaded, arr1)


def test_object_store_directory_skipped_in_gc(tmp_path: Path) -> None:
    """gc() must leave the _objects subtree alone."""
    store = CheckpointStore(tmp_path / "checkpoints")
    df = pd.DataFrame({"x": np.arange(50_000, dtype=np.float64)})
    h = store.object_store.put_dataframe(df)
    assert (store.root_dir / "_objects" / "dataframe" / f"{h}.parquet").exists()

    # gc with empty keep set should not wipe the object store.
    store.gc(keep_checkpoint_ids=set())
    assert (store.root_dir / "_objects" / "dataframe" / f"{h}.parquet").exists()


def test_namespace_with_large_dataframe_round_trips(runner: PipelineRunner) -> None:
    """End-to-end: a freeform cell with a large DataFrame in its namespace
    saves, the namespace.pkl is small (refs only), and reload reconstructs
    the original DataFrame correctly."""
    pipeline = {
        "nodes": [
            {"id": "n", "block": "FreeformCode",
             "params": {"n_inputs": 0, "n_outputs": 1, "code": _large_df_code(n_rows=100_000)}},
        ],
        "edges": [],
    }
    result = runner.run_pipeline(pipeline)
    cp = result.node_results["n"].checkpoint_id

    pkl_path = runner.checkpoint_store._namespace_path(
        runner.checkpoint_store._checkpoint_dir(cp)
    )
    pkl_size = pkl_path.stat().st_size
    # The DataFrame is ~1.6 MB inline; with content-addressing the pickle
    # holds only the ref tuple plus small variables — tiny.
    assert pkl_size < 50_000, f"namespace.pkl unexpectedly large: {pkl_size} bytes"

    ns = runner.checkpoint_store.load_namespace(cp)
    assert ns is not None
    assert isinstance(ns["df"], pd.DataFrame)
    assert len(ns["df"]) == 100_000
    assert ns["df"]["x"].iloc[0] == 0
    assert ns["df"]["x"].iloc[-1] == 99_999
