"""The M0 demo criterion, headless (DESIGN.md §12):

decorate functions → draw edges → run → rerun is instant (all checkpoints
reused) → edit one function → only it and its descendants rerun.

Runs the engine in-process for speed. Real *Forge runs use a fresh worker
process per run (which makes module reloading a non-issue); here we emulate
that isolation by purging workspace modules from sys.modules between runs.
"""

from __future__ import annotations

import sys

from conftest import three_node_doc

from starforge.core.checkpoints import CheckpointStore
from starforge.core.provenance import compute_states, env_fingerprint
from starforge.core.runner import run_pipeline
from starforge.core.spec import PipelineDoc
from starforge.index import scan_workspace


def _purge_workspace_modules(workspace_root: str) -> None:
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin and origin.startswith(workspace_root):
            del sys.modules[name]


def execute(workspace, doc_dict, cache=None, pickle_enabled=False):
    """One simulated worker run; returns (events, states, cache)."""
    root = str(workspace.root)
    index, cache = scan_workspace(workspace.root, cache)
    doc = PipelineDoc.from_dict(doc_dict)
    store = CheckpointStore(root)
    store.ensure_layout()
    states = compute_states(doc, index, env_fingerprint(root), store.exists)
    blocks = {
        b.block_id: {
            "module": b.module,
            "qualname": b.qualname,
            "label": b.label,
            "outputs": b.outputs,
            "source_hash": b.source_hash,
            "optional_params": [p.name for p in b.params if p.optional and not p.has_default],
        }
        for b in index.blocks.values()
    }
    events: list[dict] = []
    sys.path.insert(0, root)
    _purge_workspace_modules(root)
    try:
        run_pipeline(
            doc,
            blocks,
            {nid: s.to_dict() for nid, s in states.items()},
            store,
            events.append,
            pickle_enabled=pickle_enabled,
        )
    finally:
        sys.path.remove(root)
        _purge_workspace_modules(root)
    return events, states, cache


def names(events, kind):
    return [e["node"] for e in events if e["event"] == kind]


def test_m0_demo_loop(workspace):
    doc = three_node_doc(n=4, factor=3.0)
    store = CheckpointStore(str(workspace.root))

    # Run 1: everything executes, results are correct.
    events, states, cache = execute(workspace, doc)
    assert names(events, "node_completed") == ["n1", "n2", "n3"]
    assert events[-1] == {"event": "run_finished", "status": "completed"}
    assert store.load_output(states["n2"].history_hash, "output") == {"values": [3, 6, 9, 12]}
    assert store.load_output(states["n3"].history_hash, "total") == 30
    assert store.load_output(states["n3"].history_hash, "count") == 4

    # Run 2: instant — every node reused, nothing executes.
    events, _, cache = execute(workspace, doc, cache)
    assert names(events, "node_completed") == []
    assert names(events, "node_skipped") == ["n1", "n2", "n3"]

    # Edit a helper (not even a block!): Tier 2 staleness reruns only the
    # importing subgraph, and the recomputed values reflect the edit.
    workspace.write("pipeline_lib/helpers.py", "def scale_factor():\n    return 10\n")
    events, states, cache = execute(workspace, doc, cache)
    assert names(events, "node_skipped") == ["n1"]
    assert names(events, "node_completed") == ["n2", "n3"]
    assert store.load_output(states["n3"].history_hash, "total") == 300

    # Reverting the edit hits the ORIGINAL checkpoints — nothing reruns.
    workspace.write("pipeline_lib/helpers.py", "def scale_factor():\n    return 1\n")
    events, states, cache = execute(workspace, doc, cache)
    assert names(events, "node_completed") == []
    assert store.load_output(states["n3"].history_hash, "total") == 30


def test_param_change_reruns_only_affected_subgraph(workspace):
    events, _, cache = execute(workspace, three_node_doc(n=4, factor=3.0))
    assert len(names(events, "node_completed")) == 3

    events, states, cache = execute(workspace, three_node_doc(n=4, factor=5.0), cache)
    assert names(events, "node_skipped") == ["n1"]
    assert names(events, "node_completed") == ["n2", "n3"]
    store = CheckpointStore(str(workspace.root))
    assert store.load_output(states["n3"].history_hash, "total") == 50


def test_failure_blocks_descendants_but_not_independent_branches(workspace):
    workspace.write(
        "pipeline_lib/flaky.py",
        "from starforge import block\n\n"
        "@block\n"
        "def explode(data: dict) -> dict:\n"
        "    raise RuntimeError('boom')\n",
    )
    doc = three_node_doc()
    doc["nodes"].insert(2, {"id": "nx", "block": "pipeline_lib.flaky:explode", "params": {}})
    doc["edges"] = [
        {"id": "e1", "source": "n1", "source_output": "output", "target": "nx", "target_param": "data"},
        {"id": "e2", "source": "nx", "source_output": "output", "target": "n2", "target_param": "data"},
        {"id": "e3", "source": "n2", "source_output": "output", "target": "n3", "target_param": "data"},
    ]
    events, _, _ = execute(workspace, doc)
    assert names(events, "node_completed") == ["n1"]  # independent work still ran
    assert names(events, "node_failed") == ["nx"]
    assert set(names(events, "node_blocked")) == {"n2", "n3"}
    assert events[-1]["status"] == "failed"
    [failure] = [e for e in events if e["event"] == "node_failed"]
    assert "boom" in failure["traceback"]


def test_builtin_constant_feeds_pipeline_and_carries_previews(workspace):
    doc = {
        "schema": "starforge/1",
        "name": "const",
        "nodes": [
            {"id": "c1", "block": "builtin:constant", "params": {"value": {"values": [5, 10]}}},
            {"id": "n2", "block": "pipeline_lib.transforms:scale_values", "params": {"factor": 2.0}},
        ],
        "edges": [
            {"id": "e1", "source": "c1", "source_output": "output", "target": "n2", "target_param": "data"},
        ],
    }
    events, states, cache = execute(workspace, doc)
    assert set(names(events, "node_completed")) == {"c1", "n2"}
    store = CheckpointStore(str(workspace.root))
    assert store.load_output(states["n2"].history_hash, "output") == {"values": [10.0, 20.0]}

    # Previews are stored in provenance for every output.
    provenance = store.read_provenance(states["n2"].history_hash)
    [entry] = provenance["outputs"]
    assert entry["preview"]["kind"] == "value"
    assert entry["preview"]["value"] == {"values": [10.0, 20.0]}

    # Same value → reused; changed value → constant and descendants rerun.
    events, _, cache = execute(workspace, doc, cache)
    assert names(events, "node_completed") == []
    doc["nodes"][0]["params"]["value"] = {"values": [7]}
    events, _, _ = execute(workspace, doc, cache)
    assert set(names(events, "node_completed")) == {"c1", "n2"}


def test_optional_params_inject_none_when_unfilled(workspace):
    """`T | None` with no default is optional by convention: the worker
    injects None when the param is unconnected and has no literal."""
    workspace.write(
        "pipeline_lib/optionals.py",
        "from starforge import block\n\n"
        "@block\n"
        "def greet(name: str = 'world', title: str | None = None, suffix: 'str | None' = None) -> str:\n"
        "    return f\"{title or 'hi'} {name}{suffix or ''}\"\n\n"
        "@block\n"
        "def must_inject(tag: str | None) -> str:\n"
        "    return 'none' if tag is None else tag\n",
    )
    doc = {
        "schema": "starforge/1",
        "name": "opt",
        "nodes": [{"id": "m1", "block": "pipeline_lib.optionals:must_inject", "params": {}}],
        "edges": [],
    }
    events, states, _ = execute(workspace, doc)
    assert names(events, "node_completed") == ["m1"]
    store = CheckpointStore(str(workspace.root))
    assert store.load_output(states["m1"].history_hash, "output") == "none"


def test_constant_rejects_inputs(workspace):
    doc = three_node_doc()
    doc["nodes"].append({"id": "c1", "block": "builtin:constant", "params": {"value": 1}})
    doc["edges"].append(
        {"id": "ex", "source": "n1", "source_output": "output", "target": "c1", "target_param": "value"}
    )
    events, states, _ = execute(workspace, doc)
    assert any("takes no inputs" in p for p in states["c1"].problems)
    assert "c1" in names(events, "node_blocked")


def test_ephemeral_output_reexecutes_fresh_parent_when_child_needs_it(workspace):
    # Parent and child live in SEPARATE modules with no import between them,
    # so editing the child cannot touch the parent's hash via Tier 2 — the
    # ephemeral cascade is the only mechanism that can re-execute the parent.
    workspace.write(
        "pipeline_lib/opaque_parent.py",
        "from starforge import block\n\n"
        "class Model:\n"
        "    def __init__(self):\n"
        "        self.tag = 'fitted'\n\n"
        "@block\n"
        "def fit() -> object:\n"
        "    return Model()\n",
    )
    workspace.write(
        "pipeline_lib/opaque_child.py",
        "from starforge import block\n\n"
        "@block\n"
        "def describe(model: object) -> str:\n"
        "    return model.tag\n",
    )
    doc = {
        "schema": "starforge/1",
        "name": "ephemeral",
        "nodes": [
            {"id": "fit", "block": "pipeline_lib.opaque_parent:fit", "params": {}},
            {"id": "desc", "block": "pipeline_lib.opaque_child:describe", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "fit", "source_output": "output", "target": "desc", "target_param": "model"},
        ],
    }

    # Run 1: both execute; fit's output can't serialize (pickle off) → ephemeral.
    events, states, cache = execute(workspace, doc)
    assert names(events, "node_completed") == ["fit", "desc"]
    store = CheckpointStore(str(workspace.root))
    assert store.is_ephemeral(states["fit"].history_hash, "output")
    assert store.load_output(states["desc"].history_hash, "output") == "fitted"

    # Edit ONLY the child. The parent is fresh, but its output was never
    # persisted — the planner must pull it back into execution.
    original = (workspace.root / "pipeline_lib/opaque_child.py").read_text(encoding="utf-8")
    workspace.write("pipeline_lib/opaque_child.py", original.replace("model.tag", "model.tag + '!'"))
    events, states, cache = execute(workspace, doc, cache)
    [plan] = [e for e in events if e["event"] == "run_plan"]
    assert set(plan["execute"]) == {"fit", "desc"}
    assert set(names(events, "node_completed")) == {"fit", "desc"}
    assert store.load_output(states["desc"].history_hash, "output") == "fitted!"
