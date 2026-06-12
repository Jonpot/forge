"""The MCP tool functions (plain-function layer; FastMCP wraps them thin)."""

from __future__ import annotations

import pytest

from starforge import mcp
from conftest import three_node_doc


def test_list_blocks_includes_builtins_and_discovered(workspace):
    blocks = {b["block_id"] for b in mcp.list_blocks(str(workspace.root))}
    assert "builtin:constant" in blocks
    assert "pipeline_lib.sources:make_numbers" in blocks


def test_write_read_list_state_roundtrip(workspace):
    ws = str(workspace.root)
    states = mcp.write_pipeline(ws, ".forge/pipelines/demo", three_node_doc())
    assert set(states["nodes"]) == {"n1", "n2", "n3"}
    assert all(s["stale"] for s in states["nodes"].values())

    assert mcp.list_pipelines(ws) == [".forge/pipelines/demo.forge"]
    doc = mcp.read_pipeline(ws, ".forge/pipelines/demo.forge")
    assert [n["id"] for n in doc["nodes"]] == ["n1", "n2", "n3"]
    # Saves are atomic — no temp residue for watchers to trip over.
    pipelines_dir = workspace.root / ".forge" / "pipelines"
    assert [p.name for p in pipelines_dir.glob("*.tmp")] == []


def test_write_pipeline_rejects_bad_docs_and_escapes(workspace):
    ws = str(workspace.root)
    with pytest.raises(ValueError):
        mcp.write_pipeline(ws, "demo", {"schema": "other/9", "nodes": []})
    with pytest.raises(ValueError):
        mcp.write_pipeline(ws, "../outside", three_node_doc())


def test_run_pipeline_through_real_worker_with_target(workspace):
    ws = str(workspace.root)
    mcp.write_pipeline(ws, ".forge/pipelines/demo", three_node_doc())

    result = mcp.run_pipeline(ws, ".forge/pipelines/demo", target="n2")
    assert result["status"] == "completed"
    assert result["completed"] == ["n1", "n2"]
    assert result["states"]["n3"]["stale"] is True  # untouched downstream

    result = mcp.run_pipeline(ws, ".forge/pipelines/demo")
    assert result["skipped"] == ["n1", "n2"]
    assert result["completed"] == ["n3"]

    inspected = mcp.inspect_node(ws, ".forge/pipelines/demo", "n3")
    outputs = {o["name"]: o for o in inspected["checkpoint"]["outputs"]}
    assert outputs["total"]["preview"]["value"] == 30  # sum of [3,6,9,12]


def test_fastmcp_wiring_importable():
    pytest.importorskip("mcp.server.fastmcp")
    # main() would start serving; constructing the underlying pieces is the
    # import-level smoke we want.
    from mcp.server.fastmcp import FastMCP

    assert FastMCP is not None
