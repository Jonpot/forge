from __future__ import annotations

from functools import update_wrapper
import json
from typing import Any

import anyio
from mcp.server.fastmcp import Context, FastMCP, Image

from backend.block_authoring import (
    forge_block_skill_dir,
    load_forge_block_skill,
    render_block_author_prompt,
)
from backend.services import AppServices


class _GenericCallableCompat:
    def __init__(self, func: Any) -> None:
        self._func = func
        update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __getitem__(self, _item: Any) -> Any:
        return self._func


def _patch_anyio_compat() -> None:
    if hasattr(anyio.create_memory_object_stream, "__getitem__"):
        return
    anyio.create_memory_object_stream = _GenericCallableCompat(
        anyio.create_memory_object_stream
    )  # type: ignore[assignment]


_patch_anyio_compat()


def build_mcp_server(services: AppServices) -> FastMCP:
    document_service = services.document_service
    server = FastMCP(
        name="Forge",
        instructions=(
            "Manipulate Forge pipeline drafts, save them, execute them, and inspect block results. "
            "Use draft_id when working with multiple drafts; otherwise the active client draft is used."
        ),
        log_level=services.settings.log_level,  # pyright: ignore[reportArgumentType]
        streamable_http_path="/",
    )

    def client_id(ctx: Context | None) -> str | None:
        if ctx is None:
            return None
        try:
            return ctx.client_id
        except ValueError:
            return None

    def normalize_node_ids(node_ids: str | list[str] | None) -> list[str] | None:
        if node_ids is None:
            return None
        if isinstance(node_ids, str):
            items = [item.strip() for item in node_ids.split(",")]
            normalized = [item for item in items if item]
            return normalized or None
        return [str(item).strip() for item in node_ids if str(item).strip()]

    def pack_result_inspection(payload: dict[str, Any]) -> list[Any]:
        images = [Image(path=path) for path in payload.pop("image_paths", [])]
        return [json.dumps(payload, indent=2), *images]

    @server.tool(description="List saved Forge pipelines.", structured_output=True)
    def list_pipelines() -> list[dict[str, Any]]:
        return document_service.list_pipelines()

    @server.tool(
        description="List available Forge block types.", structured_output=True
    )
    def list_blocks(compact: bool = False) -> list[dict[str, Any]]:
        return document_service.list_blocks(compact=compact)

    @server.tool(
        description="Describe one Forge block type, including field-backed param schema, defaults, docs, usage notes, and input/output labels.",
        structured_output=True,
    )
    def describe_block_type(block_key: str) -> dict[str, Any]:
        return document_service.describe_block_type(block_key)

    @server.tool(
        description="List reusable parameter presets for one Forge block type.",
        structured_output=True,
    )
    def list_block_presets(block_key: str) -> dict[str, Any]:
        return document_service.list_block_presets(block_key)

    @server.tool(
        description=(
            "Describe the accepted payload shape for add_block and apply_pipeline_spec, "
            "including examples for groups, nodes, and edges."
        ),
        structured_output=True,
    )
    def describe_pipeline_spec() -> dict[str, Any]:
        return document_service.describe_pipeline_spec()

    @server.tool(
        description="Create a new active draft pipeline.", structured_output=True
    )
    def create_pipeline(
        name: str = "Untitled Pipeline", ctx: Context | None = None
    ) -> dict[str, Any]:
        draft = document_service.create_draft(name=name, client_id=client_id(ctx))
        return document_service.inspect_pipeline(
            draft_id=draft.draft_id,
            client_id=draft.client_id,
        )

    @server.tool(
        description="Open an existing pipeline into a new active draft.",
        structured_output=True,
    )
    def open_pipeline(pipeline_id: str, ctx: Context | None = None) -> dict[str, Any]:
        draft = document_service.open_draft(pipeline_id, client_id=client_id(ctx))
        return document_service.inspect_pipeline(
            draft_id=draft.draft_id,
            client_id=draft.client_id,
        )

    @server.tool(
        description="Save the active or specified draft pipeline.",
        structured_output=True,
    )
    def save_pipeline(
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        draft = document_service.save_draft(draft_id=draft_id, client_id=client_id(ctx))
        return document_service.inspect_pipeline(
            draft_id=draft.draft_id,
            client_id=draft.client_id,
        )

    @server.tool(
        description="Inspect a draft pipeline using a compact graph summary.",
        structured_output=True,
    )
    def inspect_pipeline(
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.inspect_pipeline(
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Render a Mermaid graph TD structural summary of the pipeline. "
            "When `target_group` is omitted, this returns the highest-level chunk DAG only. "
            "When `target_group` is provided, it returns only the next-level Mermaid view inside that specific chunk/group."
        ),
        structured_output=True,
    )
    def render_pipeline_mermaid(
        target_group: str | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.render_pipeline_mermaid(
            target_group=target_group,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Inspect one structural chunk/group of the current draft. "
            "Returns a small child list plus the next-level Mermaid view for that specific target. "
            "Use the returned child ids to drill deeper with `inspect_group`, or call `inspect_block` for child entries "
            "whose `inspect_with` value is `inspect_block`."
        ),
        structured_output=True,
    )
    def inspect_group(
        target_group: str,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.inspect_group(
            target_group=target_group,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Create a manual (non-managed) comment block. "
            "Pass `member_ids` as a list of node IDs and/or existing comment IDs: the tool computes the "
            "bounding box of all specified elements and positions the comment around them with standard padding. "
            "Raw `x`, `y`, `width`, `height` can be supplied as a fallback when no member_ids are given. "
            "Optionally pass `color` as a hex string such as `#14b8a6`."
        ),
        structured_output=True,
    )
    def add_comment(
        title: str,
        description: str = "",
        color: str | None = None,
        member_ids: list[str] | None = None,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.add_comment(
            title=title,
            description=description,
            color=color,
            member_ids=member_ids,
            x=x,
            y=y,
            width=width,
            height=height,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Add a block node to the active or specified draft. "
            "`params` accepts an object or JSON string; `group_ids` accepts a list, JSON string list, or comma-delimited string."
        ),
        structured_output=True,
    )
    def add_block(
        block_key: str,
        params: dict[str, Any] | str | None = None,
        node_id: str | None = None,
        notes: str | None = None,
        group_ids: list[str] | str | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.add_block(
            block_key=block_key,
            params=params,
            node_id=node_id,
            notes=notes,
            group_ids=group_ids,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Apply a declarative graph spec to the active draft in one call. "
            "The spec upserts groups, nodes, and edges without deleting unspecified items. "
            "Call describe_pipeline_spec for the exact payload shape."
        ),
        structured_output=True,
    )
    def apply_pipeline_spec(
        spec: dict[str, Any],
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.apply_pipeline_spec(
            spec=spec,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Batch upsert groups, nodes, and edges on the active draft from one declarative spec payload."
        ),
        structured_output=True,
    )
    def batch_upsert_graph(
        spec: dict[str, Any],
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.batch_upsert_graph(
            spec=spec,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Remove a block and all of its incident edges from a draft.",
        structured_output=True,
    )
    def remove_block(
        node_id: str,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.remove_block(
            node_id=node_id,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Add a connection between two block nodes.", structured_output=True
    )
    def add_edge(
        source_node_id: str,
        target_node_id: str,
        source_output: int | str | None = None,
        target_input: int | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.add_edge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_output=source_output,
            target_input=target_input,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Remove an edge by ID or source/output/target/input tuple.",
        structured_output=True,
    )
    def remove_edge(
        edge_id: str | None = None,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        source_output: int | str | None = None,
        target_input: int | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.remove_edge(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_output=source_output,
            target_input=target_input,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Inspect one block node in the current draft.",
        structured_output=True,
    )
    def inspect_block(
        node_id: str,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.inspect_block(
            node_id=node_id,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Create a logical group used for block organization and prettify.",
        structured_output=True,
    )
    def create_group(
        name: str,
        description: str = "",
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.create_group(
            name=name,
            description=description,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Delete a group and remove all group memberships.",
        structured_output=True,
    )
    def delete_group(
        group_id: str,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.delete_group(
            group_id=group_id,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(description="Add a block node to a group.", structured_output=True)
    def add_block_to_group(
        node_id: str,
        group_id: str,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.add_block_to_group(
            node_id=node_id,
            group_id=group_id,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Remove a block node from a group.", structured_output=True
    )
    def remove_block_from_group(
        node_id: str,
        group_id: str,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.remove_block_from_group(
            node_id=node_id,
            group_id=group_id,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Replace the group memberships for multiple nodes in one call.",
        structured_output=True,
    )
    def set_groups(
        assignments: list[dict[str, Any]],
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.set_groups(
            assignments=assignments,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Apply add, remove, or set group membership operations for multiple nodes in one call."
        ),
        structured_output=True,
    )
    def batch_group_membership(
        assignments: list[dict[str, Any]],
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.batch_group_membership(
            assignments=assignments,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Lay out the current draft in deterministic DAG order and update managed group comments.",
        structured_output=True,
    )
    def prettify(
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.prettify(
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Save the current draft and start execution in the background. "
            "Use poll_run with the returned run_id until completion."
        ),
        structured_output=True,
    )
    def run_pipeline(
        timeout_seconds: float | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.run_pipeline(
            timeout_seconds=timeout_seconds,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Save the current draft, start execution, and block until the run reaches a terminal state."
        ),
        structured_output=True,
    )
    def run_pipeline_and_wait(
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.25,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.run_pipeline_and_wait(
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Poll a background pipeline run started by run_pipeline. "
            "Returns running, completed, error, cancelled, or timed_out status."
        ),
        structured_output=True,
    )
    def poll_run(
        run_id: str,
        wait_seconds: float = 0.25,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.poll_run(
            run_id=run_id,
            wait_seconds=wait_seconds,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Inspect checkpoint-backed outputs for one node or a list of nodes. "
            "Returns preview metadata plus inline images when available."
        ),
        structured_output=False,
    )
    def inspect_results(
        node_ids: str | list[str] | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> list[Any]:
        payload = document_service.inspect_results(
            node_ids=normalize_node_ids(node_ids),
            draft_id=draft_id,
            client_id=client_id(ctx),
        )
        return pack_result_inspection(payload)

    @server.tool(
        description=(
            "Inspect checkpoint-backed outputs for multiple nodes. "
            "node_ids accepts a list or a comma-delimited string."
        ),
        structured_output=False,
    )
    def inspect_results_many(
        node_ids: str | list[str],
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> list[Any]:
        payload = document_service.inspect_results_many(
            node_ids=normalize_node_ids(node_ids) or [],
            draft_id=draft_id,
            client_id=client_id(ctx),
        )
        return pack_result_inspection(payload)

    @server.tool(
        description=(
            "Resolve one checkpoint-backed node asset, such as an image, output parquet, or data parquet."
        ),
        structured_output=True,
    )
    def get_result_asset(
        node_id: str,
        asset_type: str = "image",
        asset_name: str | None = None,
        index: int = 0,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.get_result_asset(
            node_id=node_id,
            asset_type=asset_type,
            asset_name=asset_name,
            index=index,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description="Render one checkpoint-backed image asset for a node result.",
        structured_output=False,
    )
    def render_result_image(
        node_id: str,
        image_index: int = 0,
        image_name: str | None = None,
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> list[Any]:
        payload = document_service.render_result_image(
            node_id=node_id,
            image_index=image_index,
            image_name=image_name,
            draft_id=draft_id,
            client_id=client_id(ctx),
        )
        return [json.dumps(payload, indent=2), Image(path=payload["path"])]

    @server.tool(
        description=(
            "Validate the current draft for structural errors, missing required params, missing root files, "
            "and incomplete wiring before execution."
        ),
        structured_output=True,
    )
    def validate_draft(
        draft_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return document_service.validate_draft(
            draft_id=draft_id,
            client_id=client_id(ctx),
        )

    @server.tool(
        description=(
            "Return the Forge block-authoring skill path and a ready-to-use prompt for creating a new atomic block."
        ),
        structured_output=True,
    )
    def create_new_block(
        block_name: str,
        description: str,
        category: str = "Custom",
        n_inputs: int = 1,
        output_count: int = 1,
    ) -> dict[str, Any]:
        return {
            "skill_path": str(forge_block_skill_dir()),
            "skill_contents": load_forge_block_skill(),
            "prompt": render_block_author_prompt(
                block_name=block_name,
                description=description,
                category=category,
                n_inputs=n_inputs,
                output_count=output_count,
            ),
        }

    @server.prompt(
        name="forge_create_block",
        description="Prompt template for authoring a new Forge block using the repo-bundled skill.",
    )
    def forge_create_block_prompt(
        block_name: str,
        description: str,
        category: str = "Custom",
        n_inputs: int = 1,
        output_count: int = 1,
    ) -> str:
        return render_block_author_prompt(
            block_name=block_name,
            description=description,
            category=category,
            n_inputs=n_inputs,
            output_count=output_count,
        )

    return server


def run_mcp_stdio(services: AppServices) -> None:
    build_mcp_server(services).run(transport="stdio")
