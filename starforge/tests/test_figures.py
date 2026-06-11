import pytest

from conftest import three_node_doc  # noqa: F401  (workspace fixture import side)

from starforge.core import figures
from starforge.core.checkpoints import CheckpointStore
from test_runner_end_to_end import execute, names


def test_capture_sweeps_shown_and_unshown_figures():
    plt = pytest.importorskip("matplotlib.pyplot")
    with figures.capture() as captured:
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2], [3, 4])
        plt.show()  # no-op under Agg; figure stays open for the sweep
        fig2, _ = plt.subplots()
    assert set(map(id, captured.matplotlib)) == {id(fig1), id(fig2)}
    figures.close_figures(captured.all_objects())
    assert plt.get_fignums() == []


def test_capture_only_sees_new_figures():
    plt = pytest.importorskip("matplotlib.pyplot")
    pre_existing, _ = plt.subplots()
    try:
        with figures.capture() as captured:
            fresh, _ = plt.subplots()
        assert [id(f) for f in captured.matplotlib] == [id(fresh)]
    finally:
        plt.close("all")


def test_render_figure_handles_figures_and_axes(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots()
    try:
        entry = figures.render_figure(fig, tmp_path, "direct")
        assert entry == {"file": "direct.png", "kind": "image"}
        assert (tmp_path / "direct.png").stat().st_size > 0

        # Axes (what sns.heatmap & friends return) render their parent figure.
        entry = figures.render_figure(ax, tmp_path, "via_axes")
        assert entry == {"file": "via_axes.png", "kind": "image"}

        assert figures.render_figure({"not": "a figure"}, tmp_path, "nope") is None
    finally:
        plt.close(fig)


def test_render_plotly_figure_to_html(tmp_path):
    go = pytest.importorskip("plotly.graph_objects")
    fig = go.Figure(data=[go.Bar(x=["a", "b"], y=[1, 2])])
    entry = figures.render_figure(fig, tmp_path, "chart")
    assert entry == {"file": "chart.html", "kind": "html"}
    html = (tmp_path / "chart.html").read_text(encoding="utf-8")
    assert "plotly" in html.lower()


def test_block_with_plt_show_attaches_figure_artifacts(workspace):
    pytest.importorskip("matplotlib")
    workspace.write(
        "pipeline_lib/plots.py",
        "import matplotlib.pyplot as plt\n"
        "from starforge import block\n\n"
        "@block\n"
        "def plot_values(data: dict) -> dict:\n"
        "    fig, ax = plt.subplots()\n"
        "    ax.plot(data['values'])\n"
        "    plt.show()\n"
        "    return data\n",
    )
    doc = {
        "schema": "starforge/1",
        "name": "viz",
        "nodes": [
            {"id": "c1", "block": "builtin:constant", "params": {"value": {"values": [1, 3, 2]}}},
            {"id": "p1", "block": "pipeline_lib.plots:plot_values", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "c1", "source_output": "output", "target": "p1", "target_param": "data"},
        ],
    }
    events, states, _ = execute(workspace, doc)
    assert set(names(events, "node_completed")) == {"c1", "p1"}

    store = CheckpointStore(str(workspace.root))
    provenance = store.read_provenance(states["p1"].history_hash)
    [artifact] = provenance["figures"]
    assert artifact["kind"] == "image"
    png = workspace.root / provenance["dir"] / "outputs" / artifact["file"]
    assert png.stat().st_size > 0

    # The data output is untouched by the figure side effect.
    assert store.load_output(states["p1"].history_hash, "output") == {"values": [1, 3, 2]}

    # Figures were closed after the checkpoint write — no canvas leaks.
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


def test_returned_figure_is_ephemeral_with_artifact_and_flows_downstream(workspace):
    pytest.importorskip("matplotlib")
    workspace.write(
        "pipeline_lib/figmaker.py",
        "import matplotlib.pyplot as plt\n"
        "from starforge import block\n\n"
        "@block\n"
        "def make_fig(data: dict) -> object:\n"
        "    fig, ax = plt.subplots()\n"
        "    ax.plot(data['values'])\n"
        "    return fig\n",
    )
    workspace.write(
        "pipeline_lib/figconsumer.py",
        "from starforge import block\n\n"
        "@block\n"
        "def count_axes(fig: object) -> int:\n"
        "    return len(fig.get_axes())\n",
    )
    doc = {
        "schema": "starforge/1",
        "name": "figflow",
        "nodes": [
            {"id": "c1", "block": "builtin:constant", "params": {"value": {"values": [1, 2]}}},
            {"id": "f1", "block": "pipeline_lib.figmaker:make_fig", "params": {}},
            {"id": "k1", "block": "pipeline_lib.figconsumer:count_axes", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "c1", "source_output": "output", "target": "f1", "target_param": "data"},
            {"id": "e2", "source": "f1", "source_output": "output", "target": "k1", "target_param": "fig"},
        ],
    }
    events, states, _ = execute(workspace, doc)
    assert set(names(events, "node_completed")) == {"c1", "f1", "k1"}

    store = CheckpointStore(str(workspace.root))
    assert store.load_output(states["k1"].history_hash, "output") == 1  # flowed in-run

    provenance = store.read_provenance(states["f1"].history_hash)
    [entry] = provenance["outputs"]
    assert entry["serializer"] == "ephemeral"  # can't round-trip from a PNG
    assert entry["artifact"]["kind"] == "image"
    assert entry["preview"]["kind"] == "figure"
    # Returned figures are NOT duplicated into the side-figure list.
    assert provenance["figures"] == []
