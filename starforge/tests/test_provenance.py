from conftest import three_node_doc

from starforge.core.provenance import compute_states, env_fingerprint, toposort
from starforge.core.spec import PipelineDoc
from starforge.index import scan_workspace

NEVER = lambda h: False  # noqa: E731 — every node reads as stale


def _states(workspace, doc_dict, cache=None):
    index, cache = scan_workspace(workspace.root, cache)
    doc = PipelineDoc.from_dict(doc_dict)
    return compute_states(doc, index, env_fingerprint(workspace.root), NEVER), cache


def test_all_nodes_hash_and_start_stale(workspace):
    states, _ = _states(workspace, three_node_doc())
    assert set(states) == {"n1", "n2", "n3"}
    for state in states.values():
        assert state.history_hash is not None
        assert state.stale
        assert state.problems == []


def test_param_change_invalidates_node_and_descendants_only(workspace):
    before, _ = _states(workspace, three_node_doc(n=4))
    after, _ = _states(workspace, three_node_doc(n=6))
    assert before["n1"].history_hash != after["n1"].history_hash
    assert before["n2"].history_hash != after["n2"].history_hash
    assert before["n3"].history_hash != after["n3"].history_hash

    # Downstream-only change leaves upstream untouched.
    factor_change, _ = _states(workspace, three_node_doc(n=4, factor=9.0))
    assert factor_change["n1"].history_hash == before["n1"].history_hash
    assert factor_change["n2"].history_hash != before["n2"].history_hash


def test_helper_edit_cascades_via_import_closure(workspace):
    """Tier 2: editing helpers.py (no blocks!) marks blocks in modules that
    import it stale, but not unrelated blocks."""
    before, cache = _states(workspace, three_node_doc())
    workspace.write("pipeline_lib/helpers.py", "def scale_factor():\n    return 10\n")
    after, _ = _states(workspace, three_node_doc(), cache)

    assert after["n1"].history_hash == before["n1"].history_hash  # sources.py untouched
    assert after["n2"].history_hash != before["n2"].history_hash  # transforms imports helpers
    assert after["n3"].history_hash != before["n3"].history_hash  # descendant of n2


def test_function_body_edit_cascades_downstream(workspace):
    before, cache = _states(workspace, three_node_doc())
    original = (workspace.root / "pipeline_lib/transforms.py").read_text(encoding="utf-8")
    workspace.write("pipeline_lib/transforms.py", original.replace("* factor", "* factor * 1"))
    after, _ = _states(workspace, three_node_doc(), cache)

    assert after["n1"].history_hash == before["n1"].history_hash
    assert after["n2"].history_hash != before["n2"].history_hash
    assert after["n3"].history_hash != before["n3"].history_hash


def test_missing_block_reports_problem_and_blocks_descendants(workspace):
    doc = three_node_doc()
    doc["nodes"][1]["block"] = "pipeline_lib.transforms:deleted_function"
    states, _ = _states(workspace, doc)
    assert states["n1"].history_hash is not None
    assert any("not found" in p for p in states["n2"].problems)
    assert states["n2"].history_hash is None
    assert any("unresolvable" in p for p in states["n3"].problems)


def test_edge_to_unknown_param_is_a_problem(workspace):
    doc = three_node_doc()
    doc["edges"][0]["target_param"] = "nonexistent"
    states, _ = _states(workspace, doc)
    assert any("unknown parameter" in p for p in states["n2"].problems)


def test_cycle_detection(workspace):
    doc = three_node_doc()
    doc["edges"].append(
        {"id": "e3", "source": "n3", "source_output": "total", "target": "n1", "target_param": "n"}
    )
    pipeline = PipelineDoc.from_dict(doc)
    order, cyclic = toposort(pipeline)
    assert cyclic == {"n1", "n2", "n3"}
    assert order == []
