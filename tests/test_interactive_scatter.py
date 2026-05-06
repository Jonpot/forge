"""Smoke tests for InteractiveMatrixScatterPlot + the .html checkpoint path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.runner import PipelineRunner
from backend.registry import BlockRegistry
from blocks.visualization import InteractiveMatrixScatterPlot


@pytest.fixture
def runner(tmp_path: Path) -> PipelineRunner:
    registry = BlockRegistry(blocks_dir="blocks", package_name="blocks")
    registry._blocks = {"InteractiveMatrixScatterPlot": InteractiveMatrixScatterPlot}
    store = CheckpointStore(tmp_path / "checkpoints")
    return PipelineRunner(registry, store)


def _seed_pipeline_with_inline_data(rows: int = 50) -> dict:
    return {
        "nodes": [
            {
                "id": "src",
                "block": "_TestSource",
                "params": {"rows": rows},
            },
            {
                "id": "viz",
                "block": "InteractiveMatrixScatterPlot",
                "params": {
                    "x_column": "a",
                    "y_column": "b",
                    "color_column": "group",
                    "title": "Test Plot",
                },
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "src",
                "target": "viz",
                "source_output": 0,
                "target_input": 0,
            }
        ],
    }


def test_interactive_scatter_writes_html_checkpoint(runner: PipelineRunner) -> None:
    """The new block must persist its Plotly figure as .html (interactive)
    in the checkpoint's images dir, not as .png."""
    from backend.block import BaseBlock, BlockOutput, BlockParams, block_param

    class _TestSource(BaseBlock):
        name = "Test Source"
        version = "1.0.0"
        category = "Test"
        description = "Inline DataFrame for tests."
        n_inputs = 0
        output_labels = ["DataFrame"]

        class Params(BlockParams):
            rows: int = block_param(50, description="Row count")

        def execute(self, data, params):
            n = int(params.rows)
            return BlockOutput(
                data=pd.DataFrame(
                    {
                        "a": list(range(n)),
                        "b": [v * 2 for v in range(n)],
                        "group": [f"g{v % 3}" for v in range(n)],
                    }
                )
            )

    runner.registry._blocks["_TestSource"] = _TestSource
    pipeline = _seed_pipeline_with_inline_data(rows=50)
    result = runner.run_pipeline(pipeline)

    assert "viz" in result.executed_nodes
    cp = result.node_results["viz"].checkpoint_id
    image_dir = runner.checkpoint_store._checkpoint_dir(cp) / "images"
    files = sorted(image_dir.iterdir()) if image_dir.exists() else []
    assert len(files) == 1, f"expected one artifact, got {[f.name for f in files]}"
    artifact = files[0]
    assert artifact.suffix == ".html", f"expected .html, got {artifact.name}"

    html = artifact.read_text(encoding="utf-8")
    # Plotly's full HTML emits a <script ... plotly></script> block referencing
    # the Plotly bundle. Light sanity check that we got an interactive payload.
    assert "plotly" in html.lower()
    assert "<script" in html.lower()


def test_interactive_scatter_metadata_marks_interactive(runner: PipelineRunner) -> None:
    """The block stamps metadata so downstream code (or future MCP inspect calls)
    can tell it produced an interactive artifact."""
    block = InteractiveMatrixScatterPlot()
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
    )
    params = InteractiveMatrixScatterPlot.Params(
        x_column="a", y_column="b", title="Plot"
    )
    output = block.execute(df, params)
    assert output.metadata.get("interactive") is True
    assert output.metadata.get("x_column") == "a"
    assert output.metadata.get("y_column") == "b"
