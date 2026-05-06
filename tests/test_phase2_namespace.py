"""Phase 2 smoke tests for namespace plumbing through PipelineRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backend.block import BaseBlock, BlockOutput, BlockParams
from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.runner import PipelineRunner
from backend.registry import BlockRegistry


class _NamespaceProducer(BaseBlock):
    name = "Namespace Producer"
    version = "1.0.0"
    category = "Test"
    description = "Produces a namespace alongside a DataFrame."
    n_inputs = 0
    output_labels = ["DataFrame"]
    produces_namespace = True

    class Params(BlockParams):
        pass

    def execute(self, data: Any, params: Any) -> BlockOutput:
        df = pd.DataFrame({"x": [1, 2, 3]})
        return BlockOutput(
            data=df,
            namespace={"answer": 42, "df": df, "tag": "from-producer"},
        )


class _NamespaceConsumer(BaseBlock):
    name = "Namespace Consumer"
    version = "1.0.0"
    category = "Test"
    description = "Consumes parent namespace, surfaces a marker into output metadata."
    n_inputs = 1
    output_labels = ["DataFrame"]
    consumes_namespace = True

    class Params(BlockParams):
        pass

    def execute(self, data: Any, params: Any) -> BlockOutput:
        # `data` here should be the parent namespace dict (Phase 2 contract).
        assert isinstance(data, dict), f"Expected namespace dict, got {type(data)!r}"
        assert data.get("answer") == 42
        assert data.get("tag") == "from-producer"
        df = data["df"].copy()
        df["seen_answer"] = data["answer"]
        return BlockOutput(data=df, metadata={"saw_namespace": True})


class _DataFrameToNamespaceConsumer(BaseBlock):
    """Consumes a DataFrame parent (no namespace) — should see synthesized {'data': df}."""

    name = "DataFrame-to-Namespace Consumer"
    version = "1.0.0"
    category = "Test"
    description = "Verifies non-namespace parents get a synthesized namespace shim."
    n_inputs = 1
    output_labels = ["DataFrame"]
    consumes_namespace = True

    class Params(BlockParams):
        pass

    def execute(self, data: Any, params: Any) -> BlockOutput:
        assert isinstance(data, dict)
        assert "data" in data
        assert isinstance(data["data"], pd.DataFrame)
        return BlockOutput(data=data["data"].copy())


@pytest.fixture
def runner_with_test_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineRunner:
    registry = BlockRegistry(blocks_dir="blocks", package_name="blocks")
    # Replace internal registry with our test classes only; skip filesystem discovery.
    registry._blocks = {
        "_NamespaceProducer": _NamespaceProducer,
        "_NamespaceConsumer": _NamespaceConsumer,
        "_DataFrameToNamespaceConsumer": _DataFrameToNamespaceConsumer,
    }
    store = CheckpointStore(tmp_path / "checkpoints")
    return PipelineRunner(registry, store)


def test_namespace_round_trips_through_checkpoint(runner_with_test_blocks: PipelineRunner) -> None:
    pipeline = {
        "nodes": [
            {"id": "n1", "block": "_NamespaceProducer", "params": {}},
            {"id": "n2", "block": "_NamespaceConsumer", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2", "source_output": 0, "target_input": 0}
        ],
    }
    result = runner_with_test_blocks.run_pipeline(pipeline)

    assert result.executed_nodes == ["n1", "n2"]

    n1_checkpoint = result.node_results["n1"].checkpoint_id
    ns_path = runner_with_test_blocks.checkpoint_store._namespace_path(
        runner_with_test_blocks.checkpoint_store._checkpoint_dir(n1_checkpoint)
    )
    assert ns_path.exists(), "namespace.pkl must be written for produces_namespace=True"

    loaded = runner_with_test_blocks.checkpoint_store.load_namespace(n1_checkpoint)
    assert loaded is not None
    assert loaded["answer"] == 42
    assert loaded["tag"] == "from-producer"


def test_namespace_reused_on_second_run(runner_with_test_blocks: PipelineRunner) -> None:
    pipeline = {
        "nodes": [
            {"id": "n1", "block": "_NamespaceProducer", "params": {}},
            {"id": "n2", "block": "_NamespaceConsumer", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2", "source_output": 0, "target_input": 0}
        ],
    }
    runner_with_test_blocks.run_pipeline(pipeline)
    second = runner_with_test_blocks.run_pipeline(pipeline)
    # Both nodes should reuse — n1's namespace is loaded from disk and supplied to n2.
    assert second.reused_nodes == ["n1", "n2"]
    assert second.executed_nodes == []


def test_dataframe_parent_synthesizes_namespace_for_consumer(
    runner_with_test_blocks: PipelineRunner,
) -> None:
    """A non-namespace producer feeding a namespace consumer must work via the
    {'data': <df>} shim — that's the backwards-compat contract."""

    class _DataFrameProducer(BaseBlock):
        name = "DataFrame Producer"
        version = "1.0.0"
        category = "Test"
        description = "Plain DataFrame source, no namespace."
        n_inputs = 0
        output_labels = ["DataFrame"]

        class Params(BlockParams):
            pass

        def execute(self, data: Any, params: Any) -> BlockOutput:
            return BlockOutput(data=pd.DataFrame({"y": [10, 20]}))

    runner_with_test_blocks.registry._blocks["_DataFrameProducer"] = _DataFrameProducer

    pipeline = {
        "nodes": [
            {"id": "n1", "block": "_DataFrameProducer", "params": {}},
            {"id": "n2", "block": "_DataFrameToNamespaceConsumer", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2", "source_output": 0, "target_input": 0}
        ],
    }
    result = runner_with_test_blocks.run_pipeline(pipeline)
    assert result.executed_nodes == ["n1", "n2"]


def test_static_blocks_unaffected_by_namespace_plumbing(
    runner_with_test_blocks: PipelineRunner,
) -> None:
    """A regular DataFrame-only pipeline must behave identically — no namespace.pkl."""

    class _PlainSource(BaseBlock):
        name = "Plain Source"
        version = "1.0.0"
        category = "Test"
        description = "Plain DataFrame source."
        n_inputs = 0
        output_labels = ["DataFrame"]

        class Params(BlockParams):
            pass

        def execute(self, data: Any, params: Any) -> BlockOutput:
            return BlockOutput(data=pd.DataFrame({"a": [1, 2]}))

    class _PlainConsumer(BaseBlock):
        name = "Plain Consumer"
        version = "1.0.0"
        category = "Test"
        description = "Plain DataFrame transform."
        n_inputs = 1
        output_labels = ["DataFrame"]

        class Params(BlockParams):
            pass

        def execute(self, data: Any, params: Any) -> BlockOutput:
            assert isinstance(data, pd.DataFrame), f"Expected DataFrame, got {type(data)!r}"
            return BlockOutput(data=data.copy())

    runner_with_test_blocks.registry._blocks["_PlainSource"] = _PlainSource
    runner_with_test_blocks.registry._blocks["_PlainConsumer"] = _PlainConsumer

    pipeline = {
        "nodes": [
            {"id": "n1", "block": "_PlainSource", "params": {}},
            {"id": "n2", "block": "_PlainConsumer", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2", "source_output": 0, "target_input": 0}
        ],
    }
    result = runner_with_test_blocks.run_pipeline(pipeline)
    assert result.executed_nodes == ["n1", "n2"]

    n1_checkpoint = result.node_results["n1"].checkpoint_id
    ns_path = runner_with_test_blocks.checkpoint_store._namespace_path(
        runner_with_test_blocks.checkpoint_store._checkpoint_dir(n1_checkpoint)
    )
    assert not ns_path.exists(), (
        "namespace.pkl must NOT be written for blocks without produces_namespace"
    )
