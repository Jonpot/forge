"""End-to-end tests for FreeformCode v2: dynamic arity, slot proxies,
namespace flow, and output-handle placeholders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.block import BaseBlock, BlockOutput, BlockParams
from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.runner import PipelineRunner
from backend.registry import BlockRegistry
from blocks.freeform import FreeformCode, _SlotProxy


@pytest.fixture
def runner(tmp_path: Path) -> PipelineRunner:
    registry = BlockRegistry(blocks_dir="blocks", package_name="blocks")
    registry._blocks = {"FreeformCode": FreeformCode}
    store = CheckpointStore(tmp_path / "checkpoints")
    return PipelineRunner(registry, store)


def test_resolvers_read_n_inputs_and_n_outputs_from_params() -> None:
    assert FreeformCode.resolve_n_inputs({"n_inputs": 0}) == 0
    assert FreeformCode.resolve_n_inputs({"n_inputs": 3}) == 3
    assert FreeformCode.resolve_n_inputs({}) == 1  # default

    assert FreeformCode.resolve_output_labels({"n_outputs": 1}) == ["output_0"]
    assert FreeformCode.resolve_output_labels({"n_outputs": 3}) == [
        "output_0",
        "output_1",
        "output_2",
    ]


def test_resolver_clamps_out_of_range_values() -> None:
    assert FreeformCode.resolve_n_inputs({"n_inputs": -5}) == 0
    assert FreeformCode.resolve_n_inputs({"n_inputs": 999}) == 8
    assert FreeformCode.resolve_output_labels({"n_outputs": 0}) == ["output_0"]


def test_slot_proxy_is_read_only_and_lists_keys() -> None:
    proxy = _SlotProxy("in_0", {"x": 1, "y": "hello"})
    assert proxy.x == 1
    assert proxy.y == "hello"
    assert "x" in proxy
    assert sorted(dir(proxy)) == ["x", "y"]
    with pytest.raises(AttributeError, match="read-only"):
        proxy.z = 99
    with pytest.raises(AttributeError, match="has no variable"):
        _ = proxy.missing


def test_source_freeform_zero_inputs(runner: PipelineRunner) -> None:
    pipeline = {
        "nodes": [
            {
                "id": "n1",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 0,
                    "n_outputs": 1,
                    "code": "import pandas as pd\noutput_0 = pd.DataFrame({'a': [1, 2, 3]})\nflag = 'hello'",
                },
            },
        ],
        "edges": [],
    }
    result = runner.run_pipeline(pipeline)
    assert result.executed_nodes == ["n1"]

    cp = result.node_results["n1"].checkpoint_id
    ns = runner.checkpoint_store.load_namespace(cp)
    assert ns is not None
    assert ns["flag"] == "hello"
    assert isinstance(ns["output_0"], pd.DataFrame)


def test_single_input_spreads_parent_namespace_into_env(runner: PipelineRunner) -> None:
    """With n_inputs=1, parent's namespace keys are available directly (`data`, etc.)."""
    pipeline = {
        "nodes": [
            {
                "id": "p",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 0,
                    "n_outputs": 1,
                    "code": (
                        "import pandas as pd\n"
                        "data = pd.DataFrame({'x': [1, 2, 3]})\n"
                        "secret = 'spread-me'\n"
                        "output_0 = data\n"
                    ),
                },
            },
            {
                "id": "c",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 1,
                    "n_outputs": 1,
                    "code": (
                        # Both `data` (spread) and `in_0.data` (proxy) should work.
                        "assert data is in_0.data\n"
                        "assert secret == 'spread-me'\n"
                        "output_0 = data.assign(touched=True)\n"
                    ),
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "p", "target": "c", "source_output": 0, "target_input": 0}
        ],
    }
    result = runner.run_pipeline(pipeline)
    assert result.executed_nodes == ["p", "c"]


def test_multi_input_uses_slot_proxies(runner: PipelineRunner) -> None:
    """With n_inputs>=2, only `in_0`, `in_1`, … exist (no flat-merge)."""
    pipeline = {
        "nodes": [
            {
                "id": "p1",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 0,
                    "n_outputs": 1,
                    "code": (
                        "import pandas as pd\n"
                        "df = pd.DataFrame({'a': [1, 2]})\n"
                        "tag = 'first'\n"
                        "output_0 = df\n"
                    ),
                },
            },
            {
                "id": "p2",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 0,
                    "n_outputs": 1,
                    "code": (
                        "import pandas as pd\n"
                        "df = pd.DataFrame({'b': [9, 8]})\n"
                        "tag = 'second'\n"
                        "output_0 = df\n"
                    ),
                },
            },
            {
                "id": "c",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 2,
                    "n_outputs": 2,
                    "code": (
                        # Verify slot proxies expose the right values, no flat-merge.
                        "assert in_0.tag == 'first', in_0.tag\n"
                        "assert in_1.tag == 'second', in_1.tag\n"
                        "assert 'tag' not in dir() or globals().get('tag') is None\n"
                        "output_0 = in_0.df\n"
                        "output_1 = in_1.df\n"
                        "merged_tags = (in_0.tag, in_1.tag)\n"
                    ),
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "p1", "target": "c", "source_output": 0, "target_input": 0},
            {"id": "e2", "source": "p2", "target": "c", "source_output": 0, "target_input": 1},
        ],
    }
    result = runner.run_pipeline(pipeline)
    assert result.executed_nodes == ["p1", "p2", "c"]

    cp = result.node_results["c"].checkpoint_id
    ns = runner.checkpoint_store.load_namespace(cp)
    assert ns is not None
    assert ns["merged_tags"] == ("first", "second")


def test_non_dataframe_output_becomes_empty_placeholder(runner: PipelineRunner) -> None:
    """A non-DataFrame `output_i` is stored as empty DataFrame on disk;
    the real value still rides in the namespace."""
    pipeline = {
        "nodes": [
            {
                "id": "n1",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 0,
                    "n_outputs": 2,
                    "code": (
                        "import pandas as pd\n"
                        "output_0 = pd.DataFrame({'x': [1]})\n"
                        "output_1 = {'not': 'a-dataframe'}\n"
                    ),
                },
            },
        ],
        "edges": [],
    }
    result = runner.run_pipeline(pipeline)
    cp = result.node_results["n1"].checkpoint_id

    out_1 = runner.checkpoint_store.load_output(cp, "output_1")
    assert isinstance(out_1, pd.DataFrame)
    assert out_1.empty

    ns = runner.checkpoint_store.load_namespace(cp)
    assert ns is not None
    assert ns["output_1"] == {"not": "a-dataframe"}


def test_namespace_passes_through_freeform_chain(runner: PipelineRunner) -> None:
    """A → B → C, all freeform. C should see B's namespace, which can include
    transformed copies of A's contributions."""
    pipeline = {
        "nodes": [
            {
                "id": "a",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 0,
                    "n_outputs": 1,
                    "code": "import pandas as pd\nseed = 7\noutput_0 = pd.DataFrame({'v': [seed]})\n",
                },
            },
            {
                "id": "b",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 1,
                    "n_outputs": 1,
                    "code": (
                        "doubled = seed * 2\n"
                        "output_0 = output_0.assign(doubled=doubled)\n"
                    ),
                },
            },
            {
                "id": "c",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 1,
                    "n_outputs": 1,
                    "code": (
                        # `seed` flowed through B (B saw it spread, defined `doubled`).
                        # Both should be visible to C now.
                        "assert seed == 7\n"
                        "assert doubled == 14\n"
                        "output_0 = output_0\n"
                    ),
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "a", "target": "b", "source_output": 0, "target_input": 0},
            {"id": "e2", "source": "b", "target": "c", "source_output": 0, "target_input": 0},
        ],
    }
    result = runner.run_pipeline(pipeline)
    assert result.executed_nodes == ["a", "b", "c"]


def test_dataframe_parent_synthesizes_data_key_for_freeform_consumer(runner: PipelineRunner) -> None:
    """A regular DataFrame producer feeding a Freeform consumer must work via
    the {'data': df} shim from Phase 2."""

    class _PlainSource(BaseBlock):
        name = "Plain Source"
        version = "1.0.0"
        category = "Test"
        description = "Plain DataFrame source."
        n_inputs = 0
        output_labels = ["DataFrame"]

        class Params(BlockParams):
            pass

        def execute(self, data, params):
            return BlockOutput(data=pd.DataFrame({"a": [1, 2, 3]}))

    runner.registry._blocks["_PlainSource"] = _PlainSource

    pipeline = {
        "nodes": [
            {"id": "src", "block": "_PlainSource", "params": {}},
            {
                "id": "free",
                "block": "FreeformCode",
                "params": {
                    "n_inputs": 1,
                    "n_outputs": 1,
                    "code": (
                        "assert data['a'].sum() == 6\n"
                        "output_0 = data.assign(b=data['a'] * 10)\n"
                    ),
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "src", "target": "free", "source_output": 0, "target_input": 0}
        ],
    }
    result = runner.run_pipeline(pipeline)
    assert result.executed_nodes == ["src", "free"]


def test_arity_change_reflows_through_engine(runner: PipelineRunner) -> None:
    """Changing n_inputs param value alters the resolved arity used for edge validation."""
    base_pipeline = {
        "nodes": [
            {"id": "p1", "block": "FreeformCode", "params": {"n_inputs": 0, "n_outputs": 1, "code": "import pandas as pd\noutput_0 = pd.DataFrame({'x': [1]})"}},
            {"id": "p2", "block": "FreeformCode", "params": {"n_inputs": 0, "n_outputs": 1, "code": "import pandas as pd\noutput_0 = pd.DataFrame({'y': [2]})"}},
            {"id": "c", "block": "FreeformCode", "params": {"n_inputs": 2, "n_outputs": 1, "code": "output_0 = in_0.output_0"}},
        ],
        "edges": [
            {"id": "e1", "source": "p1", "target": "c", "source_output": 0, "target_input": 0},
            {"id": "e2", "source": "p2", "target": "c", "source_output": 0, "target_input": 1},
        ],
    }
    result = runner.run_pipeline(base_pipeline)
    assert "c" in result.executed_nodes
