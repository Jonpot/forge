"""Smoke tests for the ``interactive`` toggle on scatter plot blocks
and the corresponding checkpoint-store HTML path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.runner import PipelineRunner
from backend.registry import BlockRegistry
from blocks.visualization import Matrix3DScatterPlot, MatrixScatterPlot


@pytest.fixture
def runner(tmp_path: Path) -> PipelineRunner:
    registry = BlockRegistry(blocks_dir="blocks", package_name="blocks")
    registry._blocks = {"MatrixScatterPlot": MatrixScatterPlot}
    store = CheckpointStore(tmp_path / "checkpoints")
    return PipelineRunner(registry, store)


def _seed_pipeline_with_inline_data(*, interactive: bool, rows: int = 50) -> dict:
    return {
        "nodes": [
            {
                "id": "src",
                "block": "_TestSource",
                "params": {"rows": rows},
            },
            {
                "id": "viz",
                "block": "MatrixScatterPlot",
                "params": {
                    "x_column": "a",
                    "y_column": "b",
                    "color_column": "group",
                    "title": "Test Plot",
                    "interactive": interactive,
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


def _register_test_source(runner: PipelineRunner) -> None:
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


def test_interactive_scatter_writes_html_checkpoint(runner: PipelineRunner) -> None:
    """``interactive=True`` must persist the figure as .html (Plotly), not .png."""
    pytest.importorskip("plotly")
    _register_test_source(runner)

    result = runner.run_pipeline(
        _seed_pipeline_with_inline_data(interactive=True, rows=50)
    )

    cp = result.node_results["viz"].checkpoint_id
    image_dir = runner.checkpoint_store._checkpoint_dir(cp) / "images"
    files = sorted(image_dir.iterdir()) if image_dir.exists() else []
    html_files = [f for f in files if f.suffix == ".html"]
    assert len(html_files) == 1, (
        f"expected one .html artifact, got {[f.name for f in files]}"
    )

    html = html_files[0].read_text(encoding="utf-8")
    # Plotly's "directory" mode references plotly.min.js relative to the HTML.
    assert "plotly.min.js" in html
    # Sibling bundle exists and is non-empty.
    bundle = html_files[0].parent / "plotly.min.js"
    assert bundle.exists() and bundle.stat().st_size > 0


def test_static_scatter_writes_png_checkpoint_by_default(
    runner: PipelineRunner,
) -> None:
    """Default ``interactive=False`` keeps the matplotlib PNG path."""
    _register_test_source(runner)

    result = runner.run_pipeline(
        _seed_pipeline_with_inline_data(interactive=False, rows=20)
    )

    cp = result.node_results["viz"].checkpoint_id
    image_dir = runner.checkpoint_store._checkpoint_dir(cp) / "images"
    files = sorted(image_dir.iterdir()) if image_dir.exists() else []
    assert len(files) == 1, f"expected one artifact, got {[f.name for f in files]}"
    assert files[0].suffix == ".png", f"expected .png, got {files[0].name}"


def test_interactive_scatter_metadata_marks_interactive() -> None:
    """The block stamps ``interactive`` metadata so downstream code can tell
    which artifact type it produced."""
    pytest.importorskip("plotly")

    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
    )
    interactive_out = MatrixScatterPlot().execute(
        df,
        MatrixScatterPlot.Params(
            x_column="a", y_column="b", title="Plot", interactive=True
        ),
    )
    assert interactive_out.metadata["interactive"] is True
    assert interactive_out.metadata["x_column"] == "a"

    static_out = MatrixScatterPlot().execute(
        df,
        MatrixScatterPlot.Params(
            x_column="a", y_column="b", title="Plot", interactive=False
        ),
    )
    assert static_out.metadata["interactive"] is False


def test_3d_scatter_respects_interactive_toggle() -> None:
    """Matrix3DScatterPlot also honors the toggle: True returns a plotly figure
    (write_html), False returns a static-PNG adapter (savefig, no write_html)."""
    pytest.importorskip("plotly")

    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [1.0, 0.5, 1.5, 2.0],
            "z": [2.0, 2.5, 3.0, 3.5],
        }
    )

    interactive_out = Matrix3DScatterPlot().execute(
        df,
        Matrix3DScatterPlot.Params(
            x_column="x", y_column="y", z_column="z", interactive=True
        ),
    )
    assert interactive_out.metadata["interactive"] is True
    artifact = interactive_out.images[0]
    assert hasattr(artifact, "write_html")

    static_out = Matrix3DScatterPlot().execute(
        df,
        Matrix3DScatterPlot.Params(
            x_column="x", y_column="y", z_column="z", interactive=False
        ),
    )
    assert static_out.metadata["interactive"] is False
    static_artifact = static_out.images[0]
    assert hasattr(static_artifact, "savefig")
    assert not hasattr(static_artifact, "write_html")
