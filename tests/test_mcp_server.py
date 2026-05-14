from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.mcp_server import build_mcp_server
from backend.services import build_services
from backend.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        pipeline_dir=str(tmp_path / "pipelines"),
        blocks_dir="blocks",
        cors_origins=["http://localhost:5173"],
    )


def _services(tmp_path: Path):
    return build_services(_settings(tmp_path))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _seed_basic_draft(tmp_path: Path, *, client_id: str = "client-a"):
    services = _services(tmp_path)
    document_service = services.document_service
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        pd.DataFrame(
            [
                {"a": 1, "b": 10},
                {"a": 2, "b": 20},
                {"a": 3, "b": 30},
            ]
        ),
    )
    draft = document_service.create_draft(name="Draft Pipeline", client_id=client_id)
    document_service.add_block(
        block_key="LoadCSV",
        node_id="load",
        params={"filepath": str(csv_path)},
        draft_id=draft.draft_id,
        client_id=client_id,
    )
    document_service.add_block(
        block_key="MedianCenterRows",
        node_id="center",
        draft_id=draft.draft_id,
        client_id=client_id,
    )
    document_service.add_edge(
        source_node_id="load",
        target_node_id="center",
        draft_id=draft.draft_id,
        client_id=client_id,
    )
    return services, draft


def _poll_until_terminal(document_service, *, run_id: str, client_id: str) -> dict:
    terminal = {"completed", "error", "cancelled", "timed_out"}
    latest: dict = {}
    for _ in range(120):
        latest = document_service.poll_run(
            run_id=run_id,
            client_id=client_id,
            wait_seconds=0.05,
        )
        if latest["status"] in terminal:
            return latest
    raise AssertionError(f"Run did not reach a terminal state: {latest}")


def _seed_histogram_draft(tmp_path: Path, *, client_id: str = "client-hist"):
    services = _services(tmp_path)
    document_service = services.document_service
    csv_path = tmp_path / "histogram.csv"
    _write_csv(
        csv_path,
        pd.DataFrame(
            [
                {"feature_a": 1.0, "feature_b": 2.0},
                {"feature_a": 2.0, "feature_b": 3.0},
                {"feature_a": 3.0, "feature_b": 4.0},
                {"feature_a": 4.0, "feature_b": 5.0},
                {"feature_a": 5.0, "feature_b": 6.0},
                {"feature_a": 6.0, "feature_b": 7.0},
            ]
        ),
    )
    draft = document_service.create_draft(name="Histogram Draft", client_id=client_id)
    document_service.add_block(
        block_key="LoadCSV",
        node_id="load",
        params={"filepath": str(csv_path)},
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.add_block(
        block_key="MatrixHistogram",
        node_id="hist",
        params={"column_name": "feature_a", "plot_title": "Feature A"},
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.add_edge(
        source_node_id="load",
        target_node_id="hist",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    return services, draft


def test_run_pipeline_saves_new_draft_before_execution(tmp_path: Path) -> None:
    services, draft = _seed_basic_draft(tmp_path)
    started = services.document_service.run_pipeline(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        timeout_seconds=10,
    )
    result = _poll_until_terminal(
        services.document_service,
        run_id=started["run_id"],
        client_id=draft.client_id,
    )

    refreshed = services.document_service.get_draft(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    assert result["status"] == "completed"
    assert refreshed.pipeline_id is not None
    assert refreshed.dirty is False
    assert (tmp_path / "pipelines" / f"{refreshed.pipeline_id}.json").exists()


def test_mcp_stdio_run_pipeline_returns_promptly(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, pd.DataFrame([{"a": 1, "b": 10}, {"a": 2, "b": 20}]))

    async def _exercise_stdio() -> tuple[float, dict]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "Forge",
                "mcp",
                "--blocks-dir",
                str(repo_root / "blocks"),
                "--pipeline-dir",
                str(tmp_path / "pipelines"),
                "--checkpoint-dir",
                str(tmp_path / "checkpoints"),
            ],
            cwd=str(repo_root),
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                created = await session.call_tool(
                    "create_pipeline", {"name": "Stdio Repro"}
                )
                assert created is not None
                assert created.structuredContent is not None
                draft_id = created.structuredContent["draft_id"]
                await session.call_tool(
                    "add_block",
                    {
                        "draft_id": draft_id,
                        "block_key": "LoadCSV",
                        "node_id": "load",
                        "params": {"filepath": str(csv_path)},
                    },
                )
                await session.call_tool(
                    "add_block",
                    {
                        "draft_id": draft_id,
                        "block_key": "MedianCenterRows",
                        "node_id": "center",
                    },
                )
                await session.call_tool(
                    "add_edge",
                    {
                        "draft_id": draft_id,
                        "source_node_id": "load",
                        "target_node_id": "center",
                    },
                )
                started_at = time.monotonic()
                started = await session.call_tool(
                    "run_pipeline",
                    {"draft_id": draft_id, "timeout_seconds": 10},
                )
                elapsed = time.monotonic() - started_at
                payload = started.structuredContent
                assert payload is not None
                assert payload["status"] == "running"
                for _ in range(120):
                    polled = await session.call_tool(
                        "poll_run",
                        {"run_id": payload["run_id"], "wait_seconds": 0.05},
                    )
                    terminal = polled.structuredContent
                    assert terminal is not None
                    if terminal["status"] in {
                        "completed",
                        "error",
                        "cancelled",
                        "timed_out",
                    }:
                        return elapsed, terminal
        raise AssertionError("Stdio MCP run did not reach a terminal state.")

    elapsed, terminal = asyncio.run(_exercise_stdio())
    assert elapsed < 2.0, f"run_pipeline blocked for {elapsed:.2f} seconds"
    assert terminal["status"] == "completed"


def test_run_pipeline_aborts_when_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services, draft = _seed_basic_draft(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(services.document_service, "save_draft", _boom)

    with pytest.raises(RuntimeError, match="save failed"):
        services.document_service.run_pipeline(
            draft_id=draft.draft_id,
            client_id=draft.client_id,
            timeout_seconds=5,
        )

    assert services.execution_manager._runs_by_id == {}


def test_add_and_remove_edge_by_id_and_tuple(tmp_path: Path) -> None:
    services, draft = _seed_basic_draft(tmp_path)
    document_service = services.document_service

    inspect_before = document_service.inspect_pipeline(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    existing_edge_id = inspect_before["edges"][0]["id"]
    removed_by_id = document_service.remove_edge(
        edge_id=existing_edge_id,
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    assert removed_by_id["removed_edge"]["id"] == existing_edge_id
    assert (
        document_service.inspect_pipeline(
            draft_id=draft.draft_id,
            client_id=draft.client_id,
        )["edges"]
        == []
    )

    added = document_service.add_edge(
        source_node_id="load",
        target_node_id="center",
        source_output=0,
        target_input=0,
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    removed_by_tuple = document_service.remove_edge(
        source_node_id="load",
        target_node_id="center",
        source_output=0,
        target_input=0,
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    assert removed_by_tuple["removed_edge"]["id"] == added["id"]


def test_group_crud_and_prettify_do_not_duplicate_managed_comments(
    tmp_path: Path,
) -> None:
    services, draft = _seed_basic_draft(tmp_path)
    document_service = services.document_service

    group = document_service.create_group(
        name="QC",
        description="Quality control blocks",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.add_block_to_group(
        node_id="load",
        group_id=group["id"],
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.add_block_to_group(
        node_id="center",
        group_id=group["id"],
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )

    document_service.prettify(draft_id=draft.draft_id, client_id=draft.client_id)
    live_draft = document_service.get_draft(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    live_draft.pipeline["comments"][0]["color"] = "#14b8a6"
    live_draft.pipeline = document_service._normalize_and_validate(live_draft.pipeline)
    document_service.prettify(draft_id=draft.draft_id, client_id=draft.client_id)

    pipeline = document_service.get_draft(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    ).pipeline
    managed_comments = [
        comment
        for comment in pipeline["comments"]
        if comment.get("managed") and comment.get("group_id") == group["id"]
    ]
    assert len(managed_comments) == 1
    assert pipeline["groups"][0]["comment_id"] == managed_comments[0]["id"]
    assert managed_comments[0]["color"] == "#14b8a6"


def test_add_comment_persists_custom_color(tmp_path: Path) -> None:
    services, draft = _seed_basic_draft(tmp_path)
    document_service = services.document_service

    created = document_service.add_comment(
        title="Focus",
        description="Manual note",
        color="#06b6d4",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )

    pipeline = document_service.get_draft(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    ).pipeline
    comment = next(
        item for item in pipeline["comments"] if item["id"] == created["id"]
    )
    assert created["color"] == "#06b6d4"
    assert comment["color"] == "#06b6d4"


def test_inspect_results_truncates_and_reports_images(tmp_path: Path) -> None:
    services = _services(tmp_path)
    document_service = services.document_service
    server = build_mcp_server(services)
    csv_path = tmp_path / "wide.csv"
    wide_frame = pd.DataFrame(
        [{f"col_{col}": row + col for col in range(12)} for row in range(6)]
    )
    _write_csv(csv_path, wide_frame)

    draft = document_service.create_draft(
        name="Histogram Pipeline", client_id="client-b"
    )
    document_service.add_block(
        block_key="LoadCSV",
        node_id="load",
        params={"filepath": str(csv_path)},
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.add_block(
        block_key="MatrixHistogram",
        node_id="hist",
        params={"column_name": "col_0", "plot_title": "Histogram"},
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.add_edge(
        source_node_id="load",
        target_node_id="hist",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )

    run_result = document_service.run_pipeline(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        timeout_seconds=20,
    )
    run_result = _poll_until_terminal(
        document_service,
        run_id=run_result["run_id"],
        client_id=draft.client_id,
    )
    assert run_result["status"] == "completed"

    inspection = document_service.inspect_results(
        node_ids=["hist"],
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    hist_result = inspection["results"]["hist"]
    output_0 = hist_result["outputs"]["output_0"]
    assert hist_result["available"] is True
    assert "Rows truncated" in str(output_0["warning"])
    assert "Columns truncated" in str(output_0["warning"])
    assert len(hist_result["images"]) >= 1

    content = asyncio.run(
        server.call_tool(
            "inspect_results",
            {"node_ids": ["hist"], "draft_id": draft.draft_id},
        )
    )
    content_types = [block.type for block in content]
    assert "text" in content_types
    assert "image" in content_types


def test_inspect_results_uses_current_history_hash_after_reopen(tmp_path: Path) -> None:
    services, draft = _seed_basic_draft(tmp_path)
    run_result = services.document_service.run_pipeline(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        timeout_seconds=10,
    )
    run_result = _poll_until_terminal(
        services.document_service,
        run_id=run_result["run_id"],
        client_id=draft.client_id,
    )
    assert run_result["status"] == "completed"

    reopened_services = _services(tmp_path)
    reopened_draft = reopened_services.document_service.open_draft(
        run_result["pipeline_id"],
        client_id="client-c",
    )
    inspection = reopened_services.document_service.inspect_results(
        node_ids=["center"],
        draft_id=reopened_draft.draft_id,
        client_id=reopened_draft.client_id,
    )
    assert inspection["results"]["center"]["available"] is True


def test_pipeline_api_preserves_mcp_metadata_fields(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    payload = {
        "name": "Metadata Pipeline",
        "nodes": [
            {
                "id": "load",
                "block": "LoadCSV",
                "params": {"filepath": "pipelines/sample_input.csv"},
                "notes": "seed node",
                "group_ids": ["group_1"],
            },
            {"id": "center", "block": "MedianCenterRows", "params": {}},
        ],
        "edges": [
            {
                "id": "edge_load_center",
                "source": "load",
                "target": "center",
                "source_output": 0,
                "target_input": 0,
            }
        ],
        "groups": [
            {
                "id": "group_1",
                "name": "Primary",
                "description": "Main flow",
                "comment_id": "comment_1",
            }
        ],
        "comments": [
            {
                "id": "comment_1",
                "title": "Primary",
                "description": "Main flow",
                "color": "#14b8a6",
                "position": {"x": 10, "y": 20},
                "width": 300,
                "height": 150,
                "managed": True,
                "group_id": "group_1",
            }
        ],
    }

    with TestClient(app) as client:
        create_resp = client.post("/api/pipelines", json=payload)
        assert create_resp.status_code == 201
        pipeline_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/pipelines/{pipeline_id}")
        assert get_resp.status_code == 200
        pipeline = get_resp.json()["pipeline"]
        assert pipeline["nodes"][0]["notes"] == "seed node"
        assert pipeline["nodes"][0]["group_ids"] == ["group_1"]
        assert pipeline["edges"][0]["id"] == "edge_load_center"
        assert pipeline["groups"][0]["comment_id"] == "comment_1"
    assert pipeline["comments"][0]["managed"] is True
    assert pipeline["comments"][0]["group_id"] == "group_1"
    assert pipeline["comments"][0]["color"] == "#14b8a6"


def test_apply_pipeline_spec_upserts_graph_in_one_call(tmp_path: Path) -> None:
    services = _services(tmp_path)
    document_service = services.document_service
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}]))
    draft = document_service.create_draft(name="Spec Draft", client_id="client-spec")

    applied = document_service.apply_pipeline_spec(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        spec={
            "groups": [
                {
                    "id": "group_load",
                    "name": "Loading",
                    "member_node_ids": ["load"],
                }
            ],
            "nodes": [
                {
                    "id": "load",
                    "block": "LoadCSV",
                    "params": {"filepath": str(csv_path)},
                },
                {
                    "id": "center",
                    "block": "MedianCenterRows",
                },
            ],
            "edges": [{"source": "load", "target": "center"}],
        },
    )

    pipeline = applied["pipeline"]
    assert pipeline["topological_order"] == ["load", "center"]
    assert pipeline["groups"][0]["member_node_ids"] == ["load"]
    load_node = next(node for node in pipeline["nodes"] if node["id"] == "load")
    assert load_node["group_ids"] == ["group_load"]
    assert len(pipeline["edges"]) == 1


def test_set_groups_and_batch_group_membership_update_multiple_nodes(
    tmp_path: Path,
) -> None:
    services, draft = _seed_basic_draft(tmp_path)
    document_service = services.document_service
    document_service.create_group(
        name="Inputs",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.create_group(
        name="Transforms",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )

    set_result = document_service.set_groups(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        assignments=[
            {"node_id": "load", "group_ids": ["inputs_1"]},
            {"node_id": "center", "group_ids": ["transforms_1"]},
        ],
    )
    assert set_result["updated_nodes"] == ["load", "center"]

    batch_result = document_service.batch_group_membership(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        assignments=[
            {"node_id": "center", "add": ["inputs_1"]},
            {"node_id": "load", "remove": ["inputs_1"], "add": ["transforms_1"]},
        ],
    )
    pipeline = batch_result["pipeline"]
    load_node = next(node for node in pipeline["nodes"] if node["id"] == "load")
    center_node = next(node for node in pipeline["nodes"] if node["id"] == "center")
    assert load_node["group_ids"] == ["transforms_1"]
    assert center_node["group_ids"] == ["transforms_1", "inputs_1"]


def test_run_pipeline_and_wait_with_result_assets_and_rendered_image(
    tmp_path: Path,
) -> None:
    services, draft = _seed_histogram_draft(tmp_path)
    document_service = services.document_service
    server = build_mcp_server(services)

    run_result = document_service.run_pipeline_and_wait(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        timeout_seconds=20,
        poll_interval_seconds=0.05,
    )
    assert run_result["status"] == "completed"

    image_asset = document_service.get_result_asset(
        node_id="hist",
        asset_type="image",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    assert image_asset["asset_type"] == "image"
    assert Path(image_asset["path"]).exists()

    data_asset = document_service.get_result_asset(
        node_id="hist",
        asset_type="data",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    assert data_asset["asset_type"] == "data"
    assert Path(data_asset["path"]).exists()

    content = asyncio.run(
        server.call_tool(
            "render_result_image",
            {"node_id": "hist", "draft_id": draft.draft_id},
        )
    )
    content_types = [block.type for block in content]
    assert "text" in content_types
    assert "image" in content_types


def test_validate_draft_reports_missing_params_and_missing_files(
    tmp_path: Path,
) -> None:
    services = _services(tmp_path)
    document_service = services.document_service
    draft = document_service.create_draft(
        name="Invalid Draft", client_id="client-invalid"
    )
    document_service.add_block(
        block_key="LoadCSV",
        node_id="load",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    document_service.add_block(
        block_key="SelectColumns",
        node_id="select",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )

    validation = document_service.validate_draft(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    assert validation["valid"] is False
    assert any(
        "missing required param 'filepath'" in error for error in validation["errors"]
    )
    assert any("expects 1 inputs, found 0" in error for error in validation["errors"])


def test_add_block_and_apply_pipeline_spec_accept_string_payload_forms(
    tmp_path: Path,
) -> None:
    services = _services(tmp_path)
    document_service = services.document_service
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, pd.DataFrame([{"id": 1, "value": 2.0}]))
    draft = document_service.create_draft(
        name="String Payload Draft", client_id="client-strings"
    )
    group = document_service.create_group(
        name="Inputs",
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )

    added = document_service.add_block(
        block_key="LoadCSV",
        node_id="load",
        params=json.dumps({"filepath": str(csv_path)}),
        group_ids=group["id"],
        draft_id=draft.draft_id,
        client_id=draft.client_id,
    )
    assert added["params"]["filepath"] == str(csv_path)
    assert added["group_ids"] == [group["id"]]

    applied = document_service.apply_pipeline_spec(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        spec={
            "groups": [
                {
                    "id": "group_viz",
                    "name": "Visuals",
                    "member_node_ids": "scatter",
                }
            ],
            "nodes": [
                {
                    "id": "scatter",
                    "block": "MatrixScatterPlot",
                    "params": json.dumps(
                        {
                            "x_column": "value",
                            "y_column": "value",
                            "color_column": "id",
                            "color_mode": "categorical",
                        }
                    ),
                    "group_ids": "group_viz",
                }
            ],
            "edges": [{"source": "load", "target": "scatter"}],
        },
    )
    scatter_node = next(
        node for node in applied["pipeline"]["nodes"] if node["id"] == "scatter"
    )
    assert scatter_node["group_ids"] == ["group_viz"]


def test_describe_block_type_and_list_block_presets_include_richer_schema(
    tmp_path: Path,
) -> None:
    services = _services(tmp_path)
    scatter_schema = services.document_service.describe_block_type("MatrixScatterPlot")
    scatter3d_schema = services.document_service.describe_block_type(
        "Matrix3DScatterPlot"
    )
    merge_schema = services.document_service.describe_block_type("MergeDatasets")
    group_mean_schema = services.document_service.describe_block_type(
        "GroupMeanByAssignments"
    )
    assign_tier_schema = services.document_service.describe_block_type(
        "AssignTierByThresholds"
    )
    melt_schema = services.document_service.describe_block_type("MeltColumns")
    mean_schema = services.document_service.describe_block_type("MeanAcrossColumns")
    count_schema = services.document_service.describe_block_type(
        "CountNonNullAcrossColumns"
    )
    presets = services.document_service.list_block_presets("MatrixScatterPlot")
    pipeline_spec = services.document_service.describe_pipeline_spec()

    assert scatter_schema["required_params"] == ["x_column", "y_column"]
    assert scatter_schema["param_examples"]["color_column"] == "cluster_id"
    assert scatter_schema["params"]["color_mode"] == "auto"
    assert scatter_schema["params"]["export_enabled"] is False
    assert scatter_schema["params"]["export_dir"] is None
    assert scatter_schema["usage_notes"]
    assert any(
        preset["id"] == "cluster_scatter" for preset in scatter_schema["presets"]
    )
    assert scatter3d_schema["required_params"] == ["x_column", "y_column", "z_column"]
    assert (
        merge_schema["param_descriptions"]["on"]
        == "Join key column present in both inputs."
    )
    assert any(
        "aligned by their row index" in note
        for note in group_mean_schema["usage_notes"]
    )
    assert assign_tier_schema["params"]["group_column"] == ""
    assert any(
        field["key"] == "group_column"
        and "independently within each group" in field.get("description", "")
        for field in assign_tier_schema["param_schema"]
    )
    assert any(
        "within each group" in note for note in assign_tier_schema["usage_notes"]
    )
    assert melt_schema["params"]["value_columns"] == ""
    assert any(
        field["key"] == "value_columns"
        and "Leave blank" in field.get("description", "")
        for field in melt_schema["param_schema"]
    )
    assert mean_schema["params"]["columns"] == ""
    assert count_schema["params"]["columns"] == ""
    assert presets["block_key"] == "MatrixScatterPlot"
    assert any(preset["id"] == "default" for preset in presets["presets"])
    assert "add_block" in pipeline_spec
    assert pipeline_spec["apply_pipeline_spec"]["edge_spec"]["notes"]


def test_inspect_results_tools_accept_comma_delimited_node_ids(tmp_path: Path) -> None:
    services, draft = _seed_basic_draft(tmp_path)
    server = build_mcp_server(services)
    run_result = services.document_service.run_pipeline(
        draft_id=draft.draft_id,
        client_id=draft.client_id,
        timeout_seconds=10,
    )
    run_result = _poll_until_terminal(
        services.document_service,
        run_id=run_result["run_id"],
        client_id=draft.client_id,
    )
    assert run_result["status"] == "completed"

    inspect_one = asyncio.run(
        server.call_tool(
            "inspect_results",
            {"node_ids": "load,center", "draft_id": draft.draft_id},
        )
    )
    inspect_many = asyncio.run(
        server.call_tool(
            "inspect_results_many",
            {"node_ids": "load,center", "draft_id": draft.draft_id},
        )
    )
    one_payload = json.loads(
        next(block.text for block in inspect_one if block.type == "text")
    )
    many_payload = json.loads(
        next(block.text for block in inspect_many if block.type == "text")
    )
    assert set(one_payload["results"]) == {"load", "center"}
    assert set(many_payload["results"]) == {"load", "center"}


def test_mcp_server_registers_required_tools_and_prompt(tmp_path: Path) -> None:
    services = _services(tmp_path)
    server = build_mcp_server(services)

    tool_names = {tool.name for tool in asyncio.run(server.list_tools())}
    prompt_names = {prompt.name for prompt in asyncio.run(server.list_prompts())}

    assert {
        "list_pipelines",
        "list_blocks",
        "describe_block_type",
        "list_block_presets",
        "describe_pipeline_spec",
        "create_pipeline",
        "open_pipeline",
        "save_pipeline",
        "add_block",
        "apply_pipeline_spec",
        "batch_upsert_graph",
        "remove_block",
        "add_edge",
        "remove_edge",
        "inspect_pipeline",
        "render_pipeline_mermaid",
        "inspect_group",
        "inspect_block",
        "create_group",
        "delete_group",
        "add_block_to_group",
        "remove_block_from_group",
        "set_groups",
        "batch_group_membership",
        "prettify",
        "run_pipeline",
        "run_pipeline_and_wait",
        "poll_run",
        "inspect_results",
        "inspect_results_many",
        "get_result_asset",
        "render_result_image",
        "validate_draft",
        "create_new_block",
    }.issubset(tool_names)
    assert "forge_create_block" in prompt_names
