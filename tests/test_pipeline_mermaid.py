from __future__ import annotations

from backend.pipeline_mermaid import inspect_group, render_mermaid


def _sample_pipeline() -> dict:
    return {
        "nodes": [
            {"id": "load", "block": "LoadCSV", "group_ids": ["group_load"]},
            {"id": "clean", "block": "SelectColumns", "group_ids": ["group_clean"]},
            {"id": "helper", "block": "NoOp", "group_ids": []},
            {"id": "train", "block": "GroupReduce", "group_ids": ["group_process"]},
            {"id": "bridge", "block": "NoOp", "group_ids": []},
            {"id": "export", "block": "ExportCSV", "group_ids": ["group_export"]},
        ],
        "edges": [
            {"id": "e1", "source": "load", "target": "clean"},
            {"id": "e2", "source": "clean", "target": "helper"},
            {"id": "e3", "source": "helper", "target": "train"},
            {"id": "e4", "source": "train", "target": "bridge"},
            {"id": "e5", "source": "bridge", "target": "export"},
        ],
        "groups": [
            {"id": "group_load", "name": "Load Inputs", "comment_id": "comment_load"},
            {"id": "group_process", "name": "Process", "comment_id": "comment_process"},
            {"id": "group_clean", "name": "Cleaning Step", "comment_id": "comment_clean"},
            {"id": "group_export", "name": "Export", "comment_id": "comment_export"},
        ],
        "comments": [
            {"id": "comment_load", "position": {"x": 0, "y": 0}, "width": 180, "height": 140},
            {"id": "comment_process", "position": {"x": 240, "y": 0}, "width": 420, "height": 260},
            {"id": "comment_clean", "position": {"x": 300, "y": 40}, "width": 200, "height": 120},
            {"id": "comment_export", "position": {"x": 720, "y": 0}, "width": 180, "height": 140},
        ],
    }


def test_render_mermaid_returns_top_level_only() -> None:
    rendered = render_mermaid(_sample_pipeline())

    assert set(rendered) == {"mermaid"}
    assert rendered["mermaid"].startswith("graph TD")
    assert 'group_load["Load Inputs\\n1 nodes"]' in rendered["mermaid"]
    assert 'group_process["Process\\n3 nodes\\n+1 adopted orphans"]' in rendered["mermaid"]
    assert 'group_load --> group_process' in rendered["mermaid"]
    assert 'group_process --> orphan_' in rendered["mermaid"]


def test_inspect_group_returns_next_level_scope_and_mermaid() -> None:
    inspected = inspect_group(_sample_pipeline(), target_group="group_process")
    rendered = render_mermaid(_sample_pipeline(), target_group="group_process")

    assert set(inspected) == {"id", "name", "kind", "node_count", "children", "mermaid"}
    assert inspected["id"] == "group_process"
    assert inspected["name"] == "Process"
    assert inspected["kind"] == "group"
    assert inspected["node_count"] == 3
    assert inspected["children"] == [
        {
            "id": "group_clean",
            "name": "Cleaning Step",
            "kind": "group",
            "node_count": 1,
            "inspect_with": "inspect_group",
        },
        {
            "id": "group_process__direct",
            "name": "Direct Work + Orphans",
            "kind": "direct",
            "node_count": 2,
            "inspect_with": "inspect_group",
        },
    ]
    assert inspected["mermaid"] == rendered["mermaid"]
    assert 'group_clean["Cleaning Step\\n(1 nodes)"]' in inspected["mermaid"]
    assert 'group_process__direct["Direct Work + Orphans\\n(2 nodes)"]' in inspected["mermaid"]
    assert "group_clean --> group_process__direct" in inspected["mermaid"]


def test_inspect_group_supports_recursive_direct_and_leaf_drilldown() -> None:
    direct = inspect_group(_sample_pipeline(), target_group="group_process__direct")
    leaf = inspect_group(_sample_pipeline(), target_group="group_clean")

    assert direct["children"] == [
        {
            "id": "helper",
            "name": "NoOp [helper]",
            "kind": "node",
            "node_count": 1,
            "inspect_with": "inspect_block",
        },
        {
            "id": "train",
            "name": "GroupReduce [train]",
            "kind": "node",
            "node_count": 1,
            "inspect_with": "inspect_block",
        },
    ]
    assert "helper --> train" in direct["mermaid"]

    assert leaf["children"] == [
        {
            "id": "clean",
            "name": "SelectColumns [clean]",
            "kind": "node",
            "node_count": 1,
            "inspect_with": "inspect_block",
        }
    ]
    assert 'clean["SelectColumns [clean]"]' in leaf["mermaid"]
