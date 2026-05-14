from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import re
from time import monotonic
from typing import Any
import uuid

import pandas as pd

from backend.block_catalog import list_block_presets
from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.execution_manager import ActiveExecution, ExecutionManager
from backend.engine.runner import PipelineRunner
from backend.pipeline_graph import (
    build_adjacency,
    clone_pipeline_payload,
    collect_used_group_ids,
    collect_used_node_ids,
    edge_source_output_handle,
    edge_target_input,
    expected_output_handles,
    find_edge_by_id,
    match_edge_tuple,
    next_identifier,
    node_map,
    normalize_output_handle,
    topological_order,
)
from backend.pipeline_layout import (
    COMMENT_PADDING_BOTTOM,
    COMMENT_PADDING_TOP,
    COMMENT_PADDING_X,
    DEFAULT_NODE_HEIGHT,
    DEFAULT_NODE_WIDTH,
    START_X,
    prettify_pipeline_layout,
)
from backend.pipeline_mermaid import inspect_group as inspect_pipeline_group
from backend.pipeline_mermaid import render_mermaid
from backend.pipeline_store import PipelineStore
from backend.registry import BlockRegistry, BlockSpec
from backend.schemas import DEFAULT_COMMENT_COLOR, normalize_pipeline_payload
from backend.settings import Settings


@dataclass(slots=True)
class PipelineDraft:
    draft_id: str
    client_id: str
    pipeline_id: str | None
    pipeline: dict[str, Any]
    dirty: bool = False


@dataclass(slots=True)
class PipelineRun:
    run_id: str
    draft_id: str
    client_id: str
    pipeline_id: str
    started_at: float
    timeout_seconds: float | None
    deadline: float | None
    active_run: ActiveExecution | None
    status: str = "running"
    message: str | None = None
    topological_order: list[str] = field(default_factory=list)
    executed_nodes: list[str] = field(default_factory=list)
    reused_nodes: list[str] = field(default_factory=list)
    node_results: dict[str, Any] = field(default_factory=dict)
    node_statuses: dict[str, dict[str, Any]] = field(default_factory=dict)
    observed_events: list[dict[str, Any]] = field(default_factory=list)
    finished_at: float | None = None
    finalized: bool = False


class DraftService:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: BlockRegistry,
        checkpoint_store: CheckpointStore,
        pipeline_store: PipelineStore,
        runner: PipelineRunner,
        execution_manager: ExecutionManager,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.checkpoint_store = checkpoint_store
        self.pipeline_store = pipeline_store
        self.runner = runner
        self.execution_manager = execution_manager
        self._drafts: dict[str, PipelineDraft] = {}
        self._active_by_client: dict[str, str] = {}
        self._runs: dict[str, PipelineRun] = {}

    def list_pipelines(self) -> list[dict[str, Any]]:
        return self.pipeline_store.list()

    def create_draft(
        self,
        *,
        name: str = "Untitled Pipeline",
        client_id: str | None = None,
    ) -> PipelineDraft:
        client_key = self._client_key(client_id)
        draft = PipelineDraft(
            draft_id=uuid.uuid4().hex,
            client_id=client_key,
            pipeline_id=None,
            pipeline=normalize_pipeline_payload(
                {"name": name, "nodes": [], "edges": [], "comments": [], "groups": []}
            ),
            dirty=True,
        )
        self._drafts[draft.draft_id] = draft
        self._active_by_client[client_key] = draft.draft_id
        return draft

    def open_draft(
        self,
        pipeline_id: str,
        *,
        client_id: str | None = None,
    ) -> PipelineDraft:
        client_key = self._client_key(client_id)
        payload = clone_pipeline_payload(self.pipeline_store.read(pipeline_id))
        draft = PipelineDraft(
            draft_id=uuid.uuid4().hex,
            client_id=client_key,
            pipeline_id=pipeline_id,
            pipeline=payload,
            dirty=False,
        )
        self._drafts[draft.draft_id] = draft
        self._active_by_client[client_key] = draft.draft_id
        return draft

    def get_draft(
        self,
        *,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> PipelineDraft:
        client_key = self._client_key(client_id)
        resolved_id = draft_id or self._active_by_client.get(client_key)
        if resolved_id is None:
            raise KeyError("No active pipeline draft for this client.")
        draft = self._drafts.get(resolved_id)
        if draft is None:
            raise KeyError(f"Unknown draft: {resolved_id}")
        self._active_by_client[client_key] = draft.draft_id
        return draft

    def save_draft(
        self,
        *,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> PipelineDraft:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        payload = self._normalize_and_validate(draft.pipeline)
        if draft.pipeline_id is None:
            pipeline_id, saved_payload = self.pipeline_store.create(payload)
            draft.pipeline_id = pipeline_id
            draft.pipeline = saved_payload
        else:
            draft.pipeline = self.pipeline_store.update(draft.pipeline_id, payload)
        draft.dirty = False
        return draft

    def list_blocks(self, *, compact: bool = False) -> list[dict[str, Any]]:
        if compact:
            return [
                {
                    "key": spec.key,
                    "category": spec.category,
                    "description": spec.description,
                }
                for spec in self.registry.all_specs()
            ]
        return [
            {
                "key": spec.key,
                "name": spec.display_name,
                "category": spec.category,
                "n_inputs": spec.n_inputs,
                "output_labels": spec.output_labels,
                "description": spec.description,
            }
            for spec in self.registry.all_specs()
        ]

    def describe_block_type(self, block_key: str) -> dict[str, Any]:
        block_cls = self.registry.get(block_key)
        spec = self._spec_for_class(block_cls)
        full_doc = (
            re.sub(r"\s+\n", "\n", block_cls.__doc__ or "").strip()
            or spec.description
        )
        usage_notes = self._normalize_usage_notes(getattr(block_cls, "usage_notes", []))
        return {
            "key": spec.key,
            "name": spec.display_name,
            "aliases": spec.aliases,
            "version": spec.version,
            "category": spec.category,
            "description": spec.description,
            "docstring": full_doc,
            "n_inputs": spec.n_inputs,
            "input_labels": spec.input_labels,
            "output_labels": spec.output_labels,
            "param_schema": self._serialize_param_schema(spec),
            "params": spec.params,
            "param_types": spec.param_types,
            "param_descriptions": spec.param_descriptions,
            "required_params": spec.required_params,
            "param_examples": spec.param_examples,
            "presets": list_block_presets(spec, block_cls),
            "usage_notes": usage_notes,
            "always_execute": bool(getattr(block_cls, "always_execute", False)),
        }

    def list_block_presets(self, block_key: str) -> dict[str, Any]:
        block_cls = self.registry.get(block_key)
        spec = self._spec_for_class(block_cls)
        return {
            "block_key": spec.key,
            "block_name": spec.display_name,
            "presets": list_block_presets(spec, block_cls),
        }

    def describe_pipeline_spec(self) -> dict[str, Any]:
        return {
            "add_block": {
                "summary": (
                    "Append one node to the draft. `params` are merged over block defaults "
                    "and `group_ids` must reference existing groups."
                ),
                "fields": {
                    "block_key": {"required": True, "type": "str"},
                    "node_id": {"required": False, "type": "str"},
                    "params": {
                        "required": False,
                        "accepted_forms": ["object", "JSON string"],
                    },
                    "notes": {"required": False, "type": "str"},
                    "group_ids": {
                        "required": False,
                        "accepted_forms": [
                            "list[str]",
                            "comma-delimited string",
                            "JSON string list",
                        ],
                    },
                    "draft_id": {"required": False, "type": "str"},
                },
                "example": {
                    "block_key": "LoadCSV",
                    "node_id": "load_sales",
                    "params": {"filepath": "C:\\Users\\you\\sales.csv"},
                    "group_ids": ["group_load"],
                },
            },
            "apply_pipeline_spec": {
                "summary": (
                    "Upsert groups, nodes, and edges without deleting unspecified items. "
                    "Use stable IDs when editing an existing graph."
                ),
                "top_level_keys": {
                    "name": "Optional pipeline name override.",
                    "groups": "Optional list of group specs.",
                    "nodes": "Optional list of node specs.",
                    "edges": "Optional list of edge specs.",
                },
                "group_spec": {
                    "required_keys": ["id or name"],
                    "optional_keys": ["description", "member_node_ids"],
                    "member_node_ids_forms": [
                        "list[str]",
                        "comma-delimited string",
                        "JSON string list",
                    ],
                },
                "node_spec": {
                    "required_keys": ["id", "block or block_key"],
                    "optional_keys": ["params", "notes", "group_ids"],
                    "params_forms": ["object", "JSON string"],
                    "group_ids_forms": [
                        "list[str]",
                        "comma-delimited string",
                        "JSON string list",
                    ],
                },
                "edge_spec": {
                    "required_keys": ["source/source_node_id", "target/target_node_id"],
                    "optional_keys": ["id", "source_output", "target_input"],
                    "notes": [
                        "Use `target_input` explicitly for multi-input blocks.",
                        "If `target_input` is omitted, Forge picks the first open slot.",
                    ],
                },
                "example": {
                    "groups": [
                        {
                            "id": "group_load",
                            "name": "Loading",
                            "member_node_ids": ["load_sales"],
                        }
                    ],
                    "nodes": [
                        {
                            "id": "load_sales",
                            "block": "LoadCSV",
                            "params": {"filepath": "C:\\Users\\you\\sales.csv"},
                            "group_ids": ["group_load"],
                        },
                        {
                            "id": "scatter",
                            "block": "MatrixScatterPlot",
                            "params": {
                                "x_column": "revenue",
                                "y_column": "profit",
                                "color_column": "segment",
                                "color_mode": "categorical",
                            },
                        },
                    ],
                    "edges": [{"source": "load_sales", "target": "scatter"}],
                },
                "notes": [
                    "Node and edge IDs are durable metadata and should be reused when editing.",
                    "Layout, notes, comments, and groups do not affect history hashes.",
                ],
            },
        }

    def add_block(
        self,
        *,
        block_key: str,
        params: dict[str, Any] | str | None = None,
        node_id: str | None = None,
        notes: str | None = None,
        group_ids: list[str] | str | None = None,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        spec = self.describe_block_type(block_key)
        pipeline = clone_pipeline_payload(draft.pipeline)
        resolved_params = self._normalize_mapping_payload(params, field_name="params")
        resolved_group_ids = self._normalize_string_list(
            group_ids,
            field_name="group_ids",
        )
        self._ensure_groups_exist(pipeline, resolved_group_ids)

        actual_node_id = node_id or self._next_node_id(pipeline, spec["key"])
        if actual_node_id in collect_used_node_ids(pipeline):
            raise ValueError(f"Node '{actual_node_id}' already exists.")

        pipeline["nodes"].append(
            {
                "id": actual_node_id,
                "block": spec["key"],
                "params": {**spec["params"], **resolved_params},
                "notes": notes,
                "group_ids": resolved_group_ids,
            }
        )
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return self.inspect_block(
            node_id=actual_node_id,
            draft_id=draft.draft_id,
            client_id=draft.client_id,
        )

    def remove_block(
        self,
        *,
        node_id: str,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        nodes = pipeline.get("nodes", [])
        if node_id not in {str(node["id"]) for node in nodes}:
            raise KeyError(f"Node not found: {node_id}")
        pipeline["nodes"] = [node for node in nodes if str(node["id"]) != node_id]
        pipeline["edges"] = [
            edge
            for edge in pipeline.get("edges", [])
            if str(edge["source"]) != node_id and str(edge["target"]) != node_id
        ]
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "removed_node_id": node_id,
        }

    def add_edge(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
        source_output: int | str | None = None,
        target_input: int | None = None,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        nodes = node_map(pipeline)
        source_node = nodes.get(source_node_id)
        target_node = nodes.get(target_node_id)
        if source_node is None:
            raise KeyError(f"Unknown source node: {source_node_id}")
        if target_node is None:
            raise KeyError(f"Unknown target node: {target_node_id}")

        source_cls = self.registry.get(str(source_node["block"]))
        valid_outputs = expected_output_handles(source_cls)
        source_output_handle = self._resolve_source_output_handle(source_output)
        if source_output_handle not in valid_outputs:
            raise ValueError(
                f"Node '{source_node_id}' does not provide output handle '{source_output_handle}'."
            )

        resolved_target_input = self._resolve_target_input(
            pipeline=pipeline,
            target_node=target_node,
            requested_target_input=target_input,
        )
        edge_id = next_identifier(
            [str(edge["id"]) for edge in pipeline.get("edges", [])],
            f"edge_{source_node_id}_{target_node_id}",
        )
        pipeline["edges"].append(
            {
                "id": edge_id,
                "source": source_node_id,
                "target": target_node_id,
                "source_output": int(source_output_handle.removeprefix("output_")),
                "sourceHandle": source_output_handle,
                "target_input": resolved_target_input,
                "targetHandle": f"input_{resolved_target_input}",
            }
        )
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return self._edge_summary(find_edge_by_id(draft.pipeline, edge_id))

    def remove_edge(
        self,
        *,
        edge_id: str | None = None,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        source_output: int | str | None = None,
        target_input: int | None = None,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        matches: list[dict[str, Any]] = []

        if edge_id:
            edge = find_edge_by_id(pipeline, edge_id)
            if edge is None:
                raise KeyError(f"Edge not found: {edge_id}")
            matches = [edge]
        else:
            if not source_node_id or not target_node_id:
                raise ValueError(
                    "Removing an edge by tuple requires source_node_id and target_node_id."
                )
            source_output_handle = (
                self._resolve_source_output_handle(source_output)
                if source_output is not None
                else None
            )
            matches = [
                edge
                for edge in pipeline.get("edges", [])
                if match_edge_tuple(
                    edge,
                    source=source_node_id,
                    target=target_node_id,
                    source_output_handle=source_output_handle,
                    target_input=target_input,
                )
            ]
            if not matches:
                raise KeyError("No edge matched the requested source/target tuple.")
            if len(matches) > 1:
                raise ValueError(
                    "The provided source/target tuple matched multiple edges. "
                    "Provide edge_id or a more specific tuple."
                )

        matched_edge = matches[0]
        pipeline["edges"] = [
            edge
            for edge in pipeline.get("edges", [])
            if str(edge["id"]) != str(matched_edge["id"])
        ]
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return {"removed_edge": self._edge_summary(matched_edge)}

    def apply_pipeline_spec(
        self,
        *,
        spec: dict[str, Any],
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)

        if "name" in spec:
            pipeline["name"] = str(spec.get("name") or pipeline.get("name") or "Untitled Pipeline")

        resolved_group_ids: list[str] = []
        for group_spec in spec.get("groups", []):
            group_id = self._upsert_group_payload(pipeline, group_spec)
            resolved_group_ids.append(group_id)

        updated_node_ids: list[str] = []
        for node_spec in spec.get("nodes", []):
            node_id = self._upsert_node_payload(pipeline, node_spec)
            updated_node_ids.append(node_id)

        for group_spec in spec.get("groups", []):
            member_node_ids = self._normalize_string_list(
                group_spec.get("member_node_ids"),
                field_name="member_node_ids",
            )
            if not member_node_ids:
                continue
            group_id = self._resolve_group_identifier(pipeline, group_spec)
            for node_id in member_node_ids:
                self._apply_group_membership(
                    pipeline,
                    node_id=node_id,
                    group_ids=[group_id],
                    replace=False,
                )

        updated_edge_ids: list[str] = []
        for edge_spec in spec.get("edges", []):
            edge_id = self._upsert_edge_payload(pipeline, edge_spec)
            updated_edge_ids.append(edge_id)

        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        inspection = self.inspect_pipeline(draft_id=draft.draft_id, client_id=draft.client_id)
        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "applied": {
                "group_ids": resolved_group_ids,
                "node_ids": updated_node_ids,
                "edge_ids": updated_edge_ids,
            },
            "pipeline": inspection,
        }

    def batch_upsert_graph(
        self,
        *,
        spec: dict[str, Any],
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        return self.apply_pipeline_spec(
            spec=spec,
            draft_id=draft_id,
            client_id=client_id,
        )

    def create_group(
        self,
        *,
        name: str,
        description: str = "",
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        group_id = self._next_group_id(pipeline, name)
        pipeline.setdefault("groups", []).append(
            {"id": group_id, "name": name, "description": description, "comment_id": None}
        )
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return {"id": group_id, "name": name, "description": description}

    def delete_group(
        self,
        *,
        group_id: str,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        if group_id not in collect_used_group_ids(pipeline):
            raise KeyError(f"Group not found: {group_id}")
        pipeline["groups"] = [
            group for group in pipeline.get("groups", []) if str(group["id"]) != group_id
        ]
        for node in pipeline.get("nodes", []):
            node["group_ids"] = [
                str(item) for item in node.get("group_ids", []) if str(item) != group_id
            ]
        pipeline["comments"] = [
            comment
            for comment in pipeline.get("comments", [])
            if str(comment.get("group_id") or "") != group_id
        ]
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return {"deleted_group_id": group_id}

    def add_block_to_group(
        self,
        *,
        node_id: str,
        group_id: str,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        self._ensure_groups_exist(pipeline, [group_id])
        node = self._get_node_payload(pipeline, node_id)
        group_ids = [str(item) for item in node.get("group_ids", [])]
        if group_id not in group_ids:
            group_ids.append(group_id)
        node["group_ids"] = group_ids
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return self.inspect_block(
            node_id=node_id,
            draft_id=draft.draft_id,
            client_id=draft.client_id,
        )

    def remove_block_from_group(
        self,
        *,
        node_id: str,
        group_id: str,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        node = self._get_node_payload(pipeline, node_id)
        node["group_ids"] = [
            str(item) for item in node.get("group_ids", []) if str(item) != group_id
        ]
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return self.inspect_block(
            node_id=node_id,
            draft_id=draft.draft_id,
            client_id=draft.client_id,
        )

    def set_groups(
        self,
        *,
        assignments: list[dict[str, Any]],
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        updated_nodes: list[str] = []
        for assignment in assignments:
            node_id = str(assignment["node_id"])
            group_ids = [str(group_id) for group_id in assignment.get("group_ids", [])]
            self._apply_group_membership(
                pipeline,
                node_id=node_id,
                group_ids=group_ids,
                replace=True,
            )
            updated_nodes.append(node_id)
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "updated_nodes": updated_nodes,
            "pipeline": self.inspect_pipeline(draft_id=draft.draft_id, client_id=draft.client_id),
        }

    def batch_group_membership(
        self,
        *,
        assignments: list[dict[str, Any]],
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        updated_nodes: list[str] = []
        for assignment in assignments:
            node_id = str(assignment["node_id"])
            node = self._get_node_payload(pipeline, node_id)
            current = [str(group_id) for group_id in node.get("group_ids", [])]
            if "set" in assignment and assignment.get("set") is not None:
                next_group_ids = [str(group_id) for group_id in assignment.get("set", [])]
            else:
                next_group_ids = list(current)
                for group_id in assignment.get("add", []):
                    text = str(group_id)
                    if text not in next_group_ids:
                        next_group_ids.append(text)
                removals = {str(group_id) for group_id in assignment.get("remove", [])}
                next_group_ids = [
                    group_id for group_id in next_group_ids if group_id not in removals
                ]
            self._apply_group_membership(
                pipeline,
                node_id=node_id,
                group_ids=next_group_ids,
                replace=True,
            )
            updated_nodes.append(node_id)
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "updated_nodes": updated_nodes,
            "pipeline": self.inspect_pipeline(draft_id=draft.draft_id, client_id=draft.client_id),
        }

    def prettify(
        self,
        *,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        draft.pipeline = self._normalize_and_validate(prettify_pipeline_layout(draft.pipeline))
        draft.dirty = True
        return self.inspect_pipeline(draft_id=draft.draft_id, client_id=draft.client_id)

    def inspect_pipeline(
        self,
        *,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = draft.pipeline
        nodes = node_map(pipeline)
        order = topological_order(pipeline) if pipeline.get("nodes") else []
        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "name": str(pipeline.get("name", "")),
            "dirty": draft.dirty,
            "topological_order": order,
            "nodes": [
                {
                    "id": node_id,
                    "block_key": str(node["block"]),
                    "block_name": self.describe_block_type(str(node["block"]))["name"],
                    "notes_present": bool(str(node.get("notes") or "").strip()),
                    "group_ids": [str(item) for item in node.get("group_ids", [])],
                }
                for node_id, node in nodes.items()
            ],
            "edges": [self._edge_summary(edge) for edge in pipeline.get("edges", [])],
            "groups": [
                {
                    "id": str(group["id"]),
                    "name": str(group.get("name", "")),
                    "description": str(group.get("description", "")),
                    "comment_id": group.get("comment_id"),
                    "member_node_ids": [
                        str(node["id"])
                        for node in pipeline.get("nodes", [])
                        if str(group["id"]) in [str(item) for item in node.get("group_ids", [])]
                    ],
                }
                for group in pipeline.get("groups", [])
            ],
        }

    def inspect_block(
        self,
        *,
        node_id: str,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = draft.pipeline
        nodes = node_map(pipeline)
        node = nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node not found: {node_id}")
        incoming, outgoing = build_adjacency(pipeline)
        history_hashes = self._safe_history_hashes(pipeline)
        history_hash = history_hashes.get(node_id)
        checkpoint_id = (
            self.checkpoint_store.get_checkpoint_id_by_hash(history_hash)
            if history_hash is not None
            else None
        )
        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "node_id": node_id,
            "block_key": str(node["block"]),
            "block_name": self.describe_block_type(str(node["block"]))["name"],
            "params": node.get("params", {}),
            "notes": node.get("notes"),
            "group_ids": [str(item) for item in node.get("group_ids", [])],
            "inputs": [
                {
                    "edge_id": str(edge["id"]),
                    "from_node_id": str(edge["source"]),
                    "from_block_key": str(nodes[str(edge["source"])]["block"]),
                    "source_output_handle": edge_source_output_handle(edge),
                    "target_input": edge_target_input(edge),
                }
                for edge in incoming.get(node_id, [])
            ],
            "outputs": [
                {
                    "edge_id": str(edge["id"]),
                    "to_node_id": str(edge["target"]),
                    "to_block_key": str(nodes[str(edge["target"])]["block"]),
                    "source_output_handle": edge_source_output_handle(edge),
                    "target_input": edge_target_input(edge),
                }
                for edge in outgoing.get(node_id, [])
            ],
            "current_history_hash": history_hash,
            "checkpoint_id": checkpoint_id,
            "has_results": checkpoint_id is not None,
        }

    def inspect_results(
        self,
        *,
        node_ids: list[str] | None = None,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = draft.pipeline
        nodes = node_map(pipeline)
        history_hashes = self._safe_history_hashes(pipeline)
        requested_nodes = node_ids or list(nodes.keys())
        results: dict[str, Any] = {}
        image_paths: list[str] = []

        for node_id in requested_nodes:
            if node_id not in nodes:
                raise KeyError(f"Node not found: {node_id}")
            history_hash = history_hashes.get(node_id)
            if history_hash is None:
                results[node_id] = {
                    "available": False,
                    "reason": "Current node history hash could not be computed.",
                }
                continue

            checkpoint_id = self.checkpoint_store.get_checkpoint_id_by_hash(history_hash)
            if checkpoint_id is None:
                results[node_id] = {
                    "available": False,
                    "history_hash": history_hash,
                    "reason": "No checkpoint exists for the current node history hash.",
                }
                continue

            block_cls = self.registry.get(str(nodes[node_id]["block"]))
            provenance = self.checkpoint_store.load_provenance(checkpoint_id)
            outputs: dict[str, Any] = {}
            for output_handle in expected_output_handles(block_cls):
                frame = self.checkpoint_store.load_output(checkpoint_id, output_handle)
                outputs[output_handle] = self._preview_dataframe(frame)

            checkpoint_image_paths = [
                str(
                    (
                        Path(self.settings.checkpoint_dir)
                        / checkpoint_id
                        / "images"
                        / image_name
                    ).resolve()
                )
                for image_name in provenance.images
            ]
            image_paths.extend(checkpoint_image_paths)
            results[node_id] = {
                "available": True,
                "checkpoint_id": checkpoint_id,
                "history_hash": history_hash,
                "shape": {
                    "rows": int(provenance.output_shape[0]) if provenance.output_shape else None,
                    "columns": int(provenance.output_shape[1]) if provenance.output_shape else None,
                },
                "outputs": outputs,
                "images": [
                    {"filename": image_name, "path": image_path}
                    for image_name, image_path in zip(provenance.images, checkpoint_image_paths, strict=False)
                ],
            }

        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "results": results,
            "image_paths": image_paths,
        }

    def inspect_results_many(
        self,
        *,
        node_ids: list[str],
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        return self.inspect_results(
            node_ids=node_ids,
            draft_id=draft_id,
            client_id=client_id,
        )

    def get_result_asset(
        self,
        *,
        node_id: str,
        asset_type: str = "image",
        asset_name: str | None = None,
        index: int = 0,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = draft.pipeline
        nodes = node_map(pipeline)
        if node_id not in nodes:
            raise KeyError(f"Node not found: {node_id}")

        history_hashes = self._safe_history_hashes(pipeline)
        history_hash = history_hashes.get(node_id)
        if history_hash is None:
            raise KeyError(f"No current history hash available for node: {node_id}")

        checkpoint_id = self.checkpoint_store.get_checkpoint_id_by_hash(history_hash)
        if checkpoint_id is None:
            raise KeyError(f"No checkpoint exists for node: {node_id}")

        normalized_type = str(asset_type or "image").strip().lower()
        checkpoint_dir = Path(self.settings.checkpoint_dir) / checkpoint_id
        provenance = self.checkpoint_store.load_provenance(checkpoint_id)

        if normalized_type == "image":
            images = list(provenance.images)
            if not images:
                raise FileNotFoundError(f"Node '{node_id}' has no image assets.")
            resolved_index = self._resolve_asset_index(
                items=images,
                index=index,
                asset_name=asset_name,
                item_label="image",
            )
            filename = images[resolved_index]
            path = (checkpoint_dir / "images" / filename).resolve()
            return {
                "draft_id": draft.draft_id,
                "pipeline_id": draft.pipeline_id,
                "node_id": node_id,
                "checkpoint_id": checkpoint_id,
                "history_hash": history_hash,
                "asset_type": "image",
                "asset_name": filename,
                "path": str(path),
                "mime_type": "image/png",
            }

        if normalized_type == "data":
            path = (checkpoint_dir / "data.parquet").resolve()
            return {
                "draft_id": draft.draft_id,
                "pipeline_id": draft.pipeline_id,
                "node_id": node_id,
                "checkpoint_id": checkpoint_id,
                "history_hash": history_hash,
                "asset_type": "data",
                "asset_name": "data.parquet",
                "path": str(path),
                "mime_type": "application/x-parquet",
            }

        if normalized_type == "output":
            output_handle = str(asset_name or "output_0")
            if output_handle == "output_0":
                path = (checkpoint_dir / "data.parquet").resolve()
            else:
                path = self.checkpoint_store._output_data_path(checkpoint_dir, output_handle).resolve()
            return {
                "draft_id": draft.draft_id,
                "pipeline_id": draft.pipeline_id,
                "node_id": node_id,
                "checkpoint_id": checkpoint_id,
                "history_hash": history_hash,
                "asset_type": "output",
                "asset_name": output_handle,
                "path": str(path),
                "mime_type": "application/x-parquet",
            }

        raise ValueError("asset_type must be one of: image, data, output.")

    def render_result_image(
        self,
        *,
        node_id: str,
        image_index: int = 0,
        image_name: str | None = None,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        return self.get_result_asset(
            node_id=node_id,
            asset_type="image",
            asset_name=image_name,
            index=image_index,
            draft_id=draft_id,
            client_id=client_id,
        )

    def run_pipeline(
        self,
        *,
        draft_id: str | None = None,
        client_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        draft = self.save_draft(draft_id=draft_id, client_id=client_id)
        if draft.pipeline_id is None:
            raise RuntimeError("Saved draft is missing a pipeline_id.")

        active_run = self.execution_manager.start_execution(
            draft.pipeline_id,
            clone_pipeline_payload(draft.pipeline),
        )
        started_at = monotonic()
        pipeline_run = PipelineRun(
            run_id=active_run.run_id,
            draft_id=draft.draft_id,
            client_id=draft.client_id,
            pipeline_id=draft.pipeline_id,
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            deadline=(started_at + timeout_seconds) if timeout_seconds is not None else None,
            active_run=active_run,
        )
        self._runs[pipeline_run.run_id] = pipeline_run
        return self._serialize_run(pipeline_run)

    def run_pipeline_and_wait(
        self,
        *,
        draft_id: str | None = None,
        client_id: str | None = None,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.25,
    ) -> dict[str, Any]:
        result = self.run_pipeline(
            draft_id=draft_id,
            client_id=client_id,
            timeout_seconds=timeout_seconds,
        )
        while result["status"] not in {"completed", "error", "cancelled", "timed_out"}:
            result = self.poll_run(
                run_id=result["run_id"],
                client_id=client_id,
                wait_seconds=max(poll_interval_seconds, 0.0),
            )
        return result

    def poll_run(
        self,
        *,
        run_id: str,
        client_id: str | None = None,
        wait_seconds: float = 0.25,
    ) -> dict[str, Any]:
        pipeline_run = self._get_pipeline_run(run_id, client_id=client_id)
        self._advance_run(pipeline_run, wait_seconds=max(wait_seconds, 0.0))
        return self._serialize_run(pipeline_run)

    def validate_draft(
        self,
        *,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)
        errors: list[str] = []
        warnings: list[str] = []

        try:
            normalized = self._normalize_and_validate(pipeline)
        except Exception as exc:
            normalized = normalize_pipeline_payload(pipeline)
            errors.append(str(exc))

        nodes = node_map(normalized)
        incoming, outgoing = build_adjacency(normalized)
        order = topological_order(normalized) if not errors else []

        for node_id, node in nodes.items():
            block_key = str(node["block"])
            block_cls = self.registry.get(block_key)
            schema = self.describe_block_type(block_key)
            params = node.get("params", {}) or {}
            known_params = set(schema.get("params", {}))

            for param_name in sorted(str(key) for key in params if key not in known_params):
                errors.append(
                    f"Node '{node_id}' has unknown param '{param_name}' for block '{block_key}'."
                )

            for param_name in schema.get("required_params", []):
                value = params.get(param_name)
                if value is None or (isinstance(value, str) and not value.strip()):
                    errors.append(
                        f"Node '{node_id}' is missing required param '{param_name}'."
                    )

            expected_inputs = int(getattr(block_cls, "n_inputs", 1))
            actual_inputs = len(incoming.get(node_id, []))
            if actual_inputs < expected_inputs:
                errors.append(
                    f"Node '{node_id}' expects {expected_inputs} inputs, found {actual_inputs}."
                )
            if expected_inputs == 0 and actual_inputs > 0:
                errors.append(
                    f"Node '{node_id}' expects no inputs, found {actual_inputs}."
                )
            if actual_inputs == 0 and expected_inputs == 0 and not outgoing.get(node_id):
                warnings.append(f"Node '{node_id}' is disconnected from the rest of the graph.")

            if block_key == "LoadCSV":
                filepath = params.get("filepath")
                if filepath is None or not str(filepath).strip():
                    errors.append(f"Node '{node_id}' is missing a CSV filepath.")
                elif not Path(str(filepath)).exists():
                    errors.append(f"Node '{node_id}' references a missing file: {filepath}")

        return {
            "draft_id": draft.draft_id,
            "pipeline_id": draft.pipeline_id,
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "topological_order": order,
            "summary": {
                "node_count": len(normalized.get("nodes", [])),
                "edge_count": len(normalized.get("edges", [])),
                "group_count": len(normalized.get("groups", [])),
            },
        }

    def _get_pipeline_run(
        self,
        run_id: str,
        *,
        client_id: str | None = None,
    ) -> PipelineRun:
        pipeline_run = self._runs.get(run_id)
        if pipeline_run is None:
            raise KeyError(f"Run not found: {run_id}")
        if client_id is not None and pipeline_run.client_id != self._client_key(client_id):
            raise KeyError(f"Run not found for this client: {run_id}")
        return pipeline_run

    def _advance_run(self, pipeline_run: PipelineRun, *, wait_seconds: float = 0.0) -> None:
        active_run = pipeline_run.active_run
        if active_run is None:
            return

        stop_at = monotonic() + max(wait_seconds, 0.0)
        while True:
            self._enforce_run_timeout(pipeline_run)
            active_run = pipeline_run.active_run
            if active_run is None:
                break

            item: dict[str, Any] | None = None
            remaining = max(0.0, stop_at - monotonic())
            try:
                if wait_seconds > 0 and remaining > 0:
                    item = active_run.event_queue.get(timeout=min(0.05, remaining))
                else:
                    item = active_run.event_queue.get(block=False)
            except queue.Empty:
                item = None

            if item is not None:
                self._process_run_event(pipeline_run, item)
                if pipeline_run.status in {"completed", "error", "cancelled", "timed_out"}:
                    self._finalize_pipeline_run(pipeline_run)
                    break
                continue

            if not active_run.process.is_alive():
                if pipeline_run.status == "running":
                    if self.execution_manager.is_cancel_requested(active_run.run_id):
                        pipeline_run.status = "cancelled"
                        pipeline_run.message = "Execution cancelled."
                    else:
                        pipeline_run.status = "error"
                        pipeline_run.message = "Execution process exited unexpectedly."
                self._finalize_pipeline_run(pipeline_run)
                break

            if monotonic() >= stop_at:
                break

    def _enforce_run_timeout(self, pipeline_run: PipelineRun) -> None:
        active_run = pipeline_run.active_run
        if active_run is None:
            return
        if pipeline_run.deadline is None or pipeline_run.status != "running":
            return
        if monotonic() < pipeline_run.deadline:
            return
        self.execution_manager.cancel_run(active_run.run_id)
        pipeline_run.status = "timed_out"
        pipeline_run.message = (
            f"Execution timed out after {pipeline_run.timeout_seconds} seconds."
        )

    def _process_run_event(self, pipeline_run: PipelineRun, item: dict[str, Any]) -> None:
        pipeline_run.observed_events.append(item)
        kind = str(item.get("kind", ""))
        if kind == "event":
            payload = item.get("payload")
            if not isinstance(payload, dict):
                return
            if payload.get("type") == "run_status":
                pipeline_run.topological_order = [
                    str(node_id) for node_id in payload.get("topological_order", [])
                ]
            if payload.get("type") == "node_status":
                node_id = str(payload.get("node_id", ""))
                if node_id:
                    pipeline_run.node_statuses[node_id] = dict(payload)
            return

        if kind == "result":
            payload = item.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("Execution returned an invalid result payload.")
            pipeline_run.status = "completed"
            pipeline_run.topological_order = [
                str(node_id) for node_id in payload.get("topological_order", [])
            ]
            pipeline_run.executed_nodes = [
                str(node_id) for node_id in payload.get("executed_nodes", [])
            ]
            pipeline_run.reused_nodes = [
                str(node_id) for node_id in payload.get("reused_nodes", [])
            ]
            pipeline_run.node_results = dict(payload.get("node_results", {}))
            return

        if kind == "error":
            pipeline_run.status = "error"
            pipeline_run.message = str(item.get("message", "Execution failed"))

    def _finalize_pipeline_run(self, pipeline_run: PipelineRun) -> None:
        if pipeline_run.finalized:
            return
        active_run = pipeline_run.active_run
        if active_run is not None:
            self.execution_manager.finalize_run(active_run.run_id)
        pipeline_run.active_run = None
        pipeline_run.finalized = True
        pipeline_run.finished_at = monotonic()

    def _serialize_run(self, pipeline_run: PipelineRun) -> dict[str, Any]:
        is_terminal = pipeline_run.status in {"completed", "error", "cancelled", "timed_out"}
        return {
            "run_id": pipeline_run.run_id,
            "status": pipeline_run.status,
            "message": pipeline_run.message,
            "draft_id": pipeline_run.draft_id,
            "pipeline_id": pipeline_run.pipeline_id,
            "topological_order": pipeline_run.topological_order,
            "executed_nodes": pipeline_run.executed_nodes,
            "reused_nodes": pipeline_run.reused_nodes,
            "node_results": pipeline_run.node_results,
            "node_statuses": pipeline_run.node_statuses,
            "observed_events": pipeline_run.observed_events,
            "started_at_monotonic": pipeline_run.started_at,
            "finished_at_monotonic": pipeline_run.finished_at,
            "active": not is_terminal,
        }

    def _spec_for_class(self, block_cls: type[Any]) -> BlockSpec:
        for spec in self.registry.all_specs():
            if spec.key == block_cls.__name__:
                return spec
        raise KeyError(f"Unknown block class: {block_cls.__name__}")

    def _normalize_and_validate(self, pipeline: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_pipeline_payload(pipeline)
        self._validate_pipeline(normalized)
        return normalized

    def _validate_pipeline(self, pipeline: dict[str, Any]) -> None:
        nodes = pipeline.get("nodes", [])
        node_ids = [str(node["id"]) for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Pipeline contains duplicate node IDs.")

        groups = pipeline.get("groups", [])
        group_ids = [str(group["id"]) for group in groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Pipeline contains duplicate group IDs.")

        comments = pipeline.get("comments", [])
        comment_ids = [str(comment["id"]) for comment in comments]
        if len(comment_ids) != len(set(comment_ids)):
            raise ValueError("Pipeline contains duplicate comment IDs.")

        group_id_set = set(group_ids)
        for node in nodes:
            self.registry.get(str(node["block"]))
            for group_id in node.get("group_ids", []):
                if str(group_id) not in group_id_set:
                    raise ValueError(
                        f"Node '{node['id']}' references unknown group '{group_id}'."
                    )

        incoming, _ = build_adjacency(pipeline)
        for node in nodes:
            block_cls = self.registry.get(str(node["block"]))
            expected_inputs = int(getattr(block_cls, "n_inputs", 1))
            node_incoming = incoming[str(node["id"])]
            if len(node_incoming) > expected_inputs:
                raise ValueError(
                    f"Node '{node['id']}' expects {expected_inputs} inputs, "
                    f"found {len(node_incoming)}."
                )
            self._resolve_incoming_slots(node_incoming, expected_inputs)
            for edge in node_incoming:
                source_node = self._get_node_payload(pipeline, str(edge["source"]))
                source_cls = self.registry.get(str(source_node["block"]))
                source_outputs = set(expected_output_handles(source_cls))
                output_handle = edge_source_output_handle(edge)
                if output_handle not in source_outputs:
                    raise ValueError(
                        f"Edge '{edge['id']}' references unknown source output '{output_handle}'."
                    )

        for comment in comments:
            group_id = comment.get("group_id")
            if group_id is not None and str(group_id) not in group_id_set:
                raise ValueError(
                    f"Comment '{comment['id']}' references unknown group '{group_id}'."
                )
        topological_order(pipeline)

    def _resolve_group_identifier(
        self,
        pipeline: dict[str, Any],
        group_spec: dict[str, Any],
    ) -> str:
        requested_id = str(group_spec.get("id") or "").strip()
        if requested_id:
            for group in pipeline.get("groups", []):
                if str(group["id"]) == requested_id:
                    return requested_id
        requested_name = str(group_spec.get("name") or "").strip()
        if requested_name:
            for group in pipeline.get("groups", []):
                if str(group.get("name", "")) == requested_name:
                    return str(group["id"])
        raise KeyError("Group could not be resolved from the provided spec.")

    def _upsert_group_payload(
        self,
        pipeline: dict[str, Any],
        group_spec: dict[str, Any],
    ) -> str:
        requested_id = str(group_spec.get("id") or "").strip()
        requested_name = str(group_spec.get("name") or "").strip()
        description = str(group_spec.get("description") or "")
        if not requested_id and not requested_name:
            raise ValueError("Group spec requires an id or name.")

        groups = pipeline.setdefault("groups", [])
        existing: dict[str, Any] | None = None
        if requested_id:
            existing = next(
                (group for group in groups if str(group["id"]) == requested_id),
                None,
            )
        if existing is None and requested_name:
            existing = next(
                (group for group in groups if str(group.get("name", "")) == requested_name),
                None,
            )

        if existing is None:
            resolved_id = requested_id or self._next_group_id(pipeline, requested_name)
            groups.append(
                {
                    "id": resolved_id,
                    "name": requested_name or resolved_id,
                    "description": description,
                    "comment_id": None,
                }
            )
            return resolved_id

        if requested_name:
            existing["name"] = requested_name
        if "description" in group_spec:
            existing["description"] = description
        return str(existing["id"])

    def _upsert_node_payload(
        self,
        pipeline: dict[str, Any],
        node_spec: dict[str, Any],
    ) -> str:
        node_id = str(node_spec.get("id") or "").strip()
        if not node_id:
            raise ValueError("Node spec requires a non-empty id.")

        block_key = node_spec.get("block") or node_spec.get("block_key")
        if not block_key:
            raise ValueError(f"Node '{node_id}' is missing block or block_key.")

        schema = self.describe_block_type(str(block_key))
        nodes = node_map(pipeline)
        existing = nodes.get(node_id)

        if existing is None:
            params_payload = dict(schema["params"])
            params_payload.update(
                self._normalize_mapping_payload(
                    node_spec.get("params"),
                    field_name=f"node '{node_id}' params",
                )
            )
            group_ids = self._normalize_string_list(
                node_spec.get("group_ids"),
                field_name=f"node '{node_id}' group_ids",
            )
            self._ensure_groups_exist(pipeline, group_ids)
            pipeline.setdefault("nodes", []).append(
                {
                    "id": node_id,
                    "block": schema["key"],
                    "params": params_payload,
                    "notes": node_spec.get("notes"),
                    "group_ids": group_ids,
                }
            )
            return node_id

        if str(existing["block"]) == schema["key"]:
            params_payload = {**schema["params"], **(existing.get("params") or {})}
        else:
            params_payload = dict(schema["params"])
        if "params" in node_spec:
            params_payload.update(
                self._normalize_mapping_payload(
                    node_spec.get("params"),
                    field_name=f"node '{node_id}' params",
                )
            )

        existing["block"] = schema["key"]
        existing["params"] = params_payload
        if "notes" in node_spec:
            existing["notes"] = node_spec.get("notes")
        if "group_ids" in node_spec:
            group_ids = self._normalize_string_list(
                node_spec.get("group_ids"),
                field_name=f"node '{node_id}' group_ids",
            )
            self._ensure_groups_exist(pipeline, group_ids)
            existing["group_ids"] = group_ids
        return node_id

    def _apply_group_membership(
        self,
        pipeline: dict[str, Any],
        *,
        node_id: str,
        group_ids: list[str],
        replace: bool,
    ) -> None:
        normalized_group_ids = [str(group_id) for group_id in group_ids]
        self._ensure_groups_exist(pipeline, normalized_group_ids)
        node = self._get_node_payload(pipeline, node_id)
        current = [str(group_id) for group_id in node.get("group_ids", [])]
        if replace:
            next_group_ids = normalized_group_ids
        else:
            next_group_ids = list(current)
            for group_id in normalized_group_ids:
                if group_id not in next_group_ids:
                    next_group_ids.append(group_id)
        node["group_ids"] = next_group_ids

    def _upsert_edge_payload(
        self,
        pipeline: dict[str, Any],
        edge_spec: dict[str, Any],
    ) -> str:
        source_node_id = str(
            edge_spec.get("source_node_id", edge_spec.get("source", ""))
        ).strip()
        target_node_id = str(
            edge_spec.get("target_node_id", edge_spec.get("target", ""))
        ).strip()
        if not source_node_id or not target_node_id:
            raise ValueError("Edge spec requires source/source_node_id and target/target_node_id.")

        self._get_node_payload(pipeline, source_node_id)
        target_node = self._get_node_payload(pipeline, target_node_id)

        requested_source_output = edge_spec.get("source_output")
        source_output_handle = self._resolve_source_output_handle(requested_source_output)
        requested_target_input = edge_spec.get("target_input")
        requested_edge_id = str(edge_spec.get("id") or "").strip() or None

        matched_edge: dict[str, Any] | None = None
        if requested_edge_id is not None:
            matched_edge = find_edge_by_id(pipeline, requested_edge_id)
        if matched_edge is None:
            exact_matches = [
                edge
                for edge in pipeline.get("edges", [])
                if match_edge_tuple(
                    edge,
                    source=source_node_id,
                    target=target_node_id,
                    source_output_handle=source_output_handle,
                    target_input=requested_target_input,
                )
            ]
            if len(exact_matches) > 1:
                raise ValueError("Edge spec matched multiple existing edges.")
            if exact_matches:
                matched_edge = exact_matches[0]
        if matched_edge is None and requested_target_input is None:
            pair_matches = [
                edge
                for edge in pipeline.get("edges", [])
                if str(edge["source"]) == source_node_id
                and str(edge["target"]) == target_node_id
                and edge_source_output_handle(edge) == source_output_handle
            ]
            if len(pair_matches) == 1:
                matched_edge = pair_matches[0]

        if matched_edge is None:
            resolved_target_input = self._resolve_target_input(
                pipeline=pipeline,
                target_node=target_node,
                requested_target_input=requested_target_input,
            )
            edge_id = requested_edge_id or next_identifier(
                [str(edge["id"]) for edge in pipeline.get("edges", [])],
                f"edge_{source_node_id}_{target_node_id}",
            )
            pipeline.setdefault("edges", []).append(
                {
                    "id": edge_id,
                    "source": source_node_id,
                    "target": target_node_id,
                    "source_output": int(source_output_handle.removeprefix("output_")),
                    "sourceHandle": source_output_handle,
                    "target_input": resolved_target_input,
                    "targetHandle": f"input_{resolved_target_input}",
                }
            )
            return edge_id

        resolved_target_input = (
            int(requested_target_input)
            if requested_target_input is not None
            else edge_target_input(matched_edge)
        )
        if resolved_target_input is None:
            resolved_target_input = self._resolve_target_input(
                pipeline=pipeline,
                target_node=target_node,
                requested_target_input=None,
            )
        matched_edge["source"] = source_node_id
        matched_edge["target"] = target_node_id
        matched_edge["source_output"] = int(source_output_handle.removeprefix("output_"))
        matched_edge["sourceHandle"] = source_output_handle
        matched_edge["target_input"] = resolved_target_input
        matched_edge["targetHandle"] = f"input_{resolved_target_input}"
        return str(matched_edge["id"])

    def _resolve_asset_index(
        self,
        *,
        items: list[str],
        index: int,
        asset_name: str | None,
        item_label: str,
    ) -> int:
        if asset_name is not None:
            try:
                return items.index(asset_name)
            except ValueError as exc:
                raise FileNotFoundError(f"{item_label.title()} '{asset_name}' not found.") from exc
        if index < 0 or index >= len(items):
            raise IndexError(f"{item_label.title()} index out of range: {index}")
        return index

    def _resolve_target_input(
        self,
        *,
        pipeline: dict[str, Any],
        target_node: dict[str, Any],
        requested_target_input: int | None,
    ) -> int:
        block_cls = self.registry.get(str(target_node["block"]))
        expected_inputs = int(getattr(block_cls, "n_inputs", 1))
        incoming, _ = build_adjacency(pipeline)
        target_node_id = str(target_node["id"])
        current_incoming = list(incoming[target_node_id])

        if requested_target_input is not None:
            if requested_target_input < 0 or requested_target_input >= expected_inputs:
                raise ValueError(
                    f"Node '{target_node_id}' has no input slot {requested_target_input}."
                )
            resolved = self._resolve_incoming_slots(current_incoming, expected_inputs)
            if requested_target_input in resolved:
                raise ValueError(
                    f"Node '{target_node_id}' already has an edge targeting input {requested_target_input}."
                )
            return requested_target_input

        resolved = self._resolve_incoming_slots(current_incoming, expected_inputs)
        for input_index in range(expected_inputs):
            if input_index not in resolved:
                return input_index
        raise ValueError(f"Node '{target_node_id}' has no open input slots.")

    def _resolve_incoming_slots(
        self,
        incoming_edges: list[dict[str, Any]],
        expected_inputs: int,
    ) -> dict[int, dict[str, Any]]:
        slots: dict[int, dict[str, Any]] = {}
        unresolved: list[dict[str, Any]] = []
        for edge in incoming_edges:
            target_input = edge_target_input(edge)
            if target_input is None:
                unresolved.append(edge)
                continue
            if target_input < 0 or target_input >= expected_inputs:
                raise ValueError(
                    f"Edge '{edge['id']}' targets invalid input slot {target_input}."
                )
            if target_input in slots:
                raise ValueError(f"Multiple edges target input slot {target_input}.")
            slots[target_input] = edge

        for edge in unresolved:
            for input_index in range(expected_inputs):
                if input_index not in slots:
                    slots[input_index] = edge
                    break
            else:
                raise ValueError("Incoming edges exceed the target block input arity.")
        return slots

    def _ensure_groups_exist(self, pipeline: dict[str, Any], group_ids: list[str]) -> None:
        available = collect_used_group_ids(pipeline)
        missing = [group_id for group_id in group_ids if group_id not in available]
        if missing:
            raise ValueError(f"Unknown group IDs: {', '.join(missing)}")

    def _normalize_mapping_payload(
        self,
        value: dict[str, Any] | str | None,
        *,
        field_name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field_name} must be a mapping or valid JSON object string.") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{field_name} must decode to a JSON object.")
            return dict(parsed)
        raise ValueError(f"{field_name} must be a mapping or JSON object string.")

    def _normalize_string_list(
        self,
        value: list[str] | str | None,
        *,
        field_name: str,
    ) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return [
                        str(item).strip()
                        for item in parsed
                        if str(item).strip()
                    ]
            return [
                item.strip().strip("'").strip('"')
                for item in text.split(",")
                if item.strip().strip("'").strip('"')
            ]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError(f"{field_name} must be a list of strings or a comma-delimited string.")

    def _normalize_usage_notes(self, value: Any) -> list[str]:
        if isinstance(value, str):
            note = value.strip()
            return [note] if note else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _next_node_id(self, pipeline: dict[str, Any], block_key: str) -> str:
        prefix = _slugify(block_key or "node")
        return next_identifier(collect_used_node_ids(pipeline), prefix)

    def _next_group_id(self, pipeline: dict[str, Any], name: str) -> str:
        prefix = _slugify(name or "group")
        return next_identifier(collect_used_group_ids(pipeline), prefix)

    def _get_node_payload(self, pipeline: dict[str, Any], node_id: str) -> dict[str, Any]:
        nodes = node_map(pipeline)
        node = nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node not found: {node_id}")
        return node

    def _resolve_source_output_handle(self, source_output: int | str | None) -> str:
        if source_output is None:
            return "output_0"
        if isinstance(source_output, int):
            if source_output < 0:
                raise ValueError("source_output must be >= 0.")
            return f"output_{source_output}"
        return normalize_output_handle(source_output)

    def _edge_summary(self, edge: dict[str, Any] | None) -> dict[str, Any]:
        if edge is None:
            raise KeyError("Edge not found.")
        return {
            "id": str(edge["id"]),
            "source": str(edge["source"]),
            "target": str(edge["target"]),
            "source_output_handle": edge_source_output_handle(edge),
            "target_input": edge_target_input(edge),
        }

    def _safe_history_hashes(self, pipeline: dict[str, Any]) -> dict[str, str]:
        try:
            return self.runner.compute_history_hashes(pipeline)
        except Exception:
            return {}

    def _serialize_param_schema(self, spec: BlockSpec) -> list[dict[str, Any]]:
        return [
            {
                "key": param.key,
                "type": param.type,
                "default": param.default,
                "required": param.required,
                "description": param.description,
                "example": param.example,
                "browse_mode": param.browse_mode,
            }
            for param in spec.param_schema
        ]

    def _preview_dataframe(self, data: pd.DataFrame) -> dict[str, Any]:
        preview = data.head(5)
        index_labels = _normalize_index_labels(preview.index, list(preview.columns))
        preview_with_index = preview.reset_index()
        renamed = list(preview_with_index.columns)
        for index, label in enumerate(index_labels):
            renamed[index] = label
        preview_with_index.columns = renamed

        warning_parts: list[str] = []
        if int(data.shape[0]) > len(preview):
            warning_parts.append("Rows truncated to first 5.")
        if len(preview_with_index.columns) > 10:
            warning_parts.append("Columns truncated to first 10.")
            preview_with_index = preview_with_index.iloc[:, :10]

        return {
            "rows": preview_with_index.to_dict(orient="records"),
            "columns": list(preview_with_index.columns),
            "dtypes": {
                str(column): str(dtype)
                for column, dtype in preview_with_index.dtypes.items()
            },
            "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
            "warning": " ".join(warning_parts) if warning_parts else None,
        }

    def render_pipeline_mermaid(
        self,
        *,
        target_group: str | None = None,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = draft.pipeline
        nodes = pipeline.get("nodes", [])
        block_names = {}
        for node in nodes:
            nid = str(node["id"])
            try:
                block_names[nid] = self.describe_block_type(str(node["block"]))["name"]
            except Exception:
                block_names[nid] = str(node["block"])
        return render_mermaid(
            pipeline,
            target_group=target_group,
            block_names=block_names,
        )

    def inspect_group(
        self,
        *,
        target_group: str,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = draft.pipeline
        nodes = pipeline.get("nodes", [])
        block_names = {}
        for node in nodes:
            nid = str(node["id"])
            try:
                block_names[nid] = self.describe_block_type(str(node["block"]))["name"]
            except Exception:
                block_names[nid] = str(node["block"])
        return inspect_pipeline_group(
            pipeline,
            target_group=target_group,
            block_names=block_names,
        )

    def add_comment(
        self,
        *,
        title: str,
        description: str = "",
        color: str | None = None,
        member_ids: list[str] | None = None,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        draft_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id=draft_id, client_id=client_id)
        pipeline = clone_pipeline_payload(draft.pipeline)

        if member_ids:
            nodes_by_id = {str(n["id"]): n for n in pipeline.get("nodes", [])}
            comments_by_id = {str(c["id"]): c for c in pipeline.get("comments", [])}
            rects: list[tuple[float, float, float, float]] = []
            for mid in member_ids:
                if mid in nodes_by_id:
                    node = nodes_by_id[mid]
                    pos = node.get("position") or {}
                    rects.append((
                        float(pos.get("x", 0)),
                        float(pos.get("y", 0)),
                        float(node.get("width") or DEFAULT_NODE_WIDTH),
                        float(node.get("height") or DEFAULT_NODE_HEIGHT),
                    ))
                elif mid in comments_by_id:
                    c = comments_by_id[mid]
                    pos = c.get("position") or {}
                    rects.append((
                        float(pos.get("x", 0)),
                        float(pos.get("y", 0)),
                        float(c.get("width", 300)),
                        float(c.get("height", 150)),
                    ))
            if rects:
                min_x = min(r[0] for r in rects)
                min_y = min(r[1] for r in rects)
                max_x = max(r[0] + r[2] for r in rects)
                max_y = max(r[1] + r[3] for r in rects)
                x = min_x - COMMENT_PADDING_X
                y = min_y - COMMENT_PADDING_TOP
                width = max(max_x - min_x + COMMENT_PADDING_X * 2, 280.0)
                height = max(max_y - min_y + COMMENT_PADDING_TOP + COMMENT_PADDING_BOTTOM, 140.0)

        if x is None or y is None:
            # Fall back to below all existing content
            max_y_existing = START_X
            for node in pipeline.get("nodes", []):
                pos = node.get("position") or {}
                ny = float(pos.get("y", START_X))
                nh = float(node.get("height") or DEFAULT_NODE_HEIGHT)
                max_y_existing = max(max_y_existing, ny + nh)
            x = x if x is not None else START_X
            y = y if y is not None else max_y_existing + 120.0

        width = width if width is not None else 300.0
        height = height if height is not None else 150.0

        used_ids = {str(c["id"]) for c in pipeline.get("comments", [])}
        comment_id = next_identifier(used_ids, "comment")
        pipeline.setdefault("comments", []).append({
            "id": comment_id,
            "title": title,
            "description": description,
            "color": str(color or DEFAULT_COMMENT_COLOR),
            "position": {"x": x, "y": y},
            "width": width,
            "height": height,
            "managed": False,
            "group_id": None,
        })
        draft.pipeline = self._normalize_and_validate(pipeline)
        draft.dirty = True
        return {
            "id": comment_id,
            "title": title,
            "description": description,
            "color": str(color or DEFAULT_COMMENT_COLOR),
            "position": {"x": x, "y": y},
            "width": width,
            "height": height,
        }

    def _client_key(self, client_id: str | None) -> str:
        return client_id or "__default__"


def _normalize_index_labels(index: pd.Index, existing_columns: list[str]) -> list[str]:
    if isinstance(index, pd.MultiIndex):
        raw = [
            str(name) if name is not None and str(name).strip() else f"index_{i}"
            for i, name in enumerate(index.names)
        ]
    else:
        raw_name = index.name
        raw = [str(raw_name) if raw_name is not None and str(raw_name).strip() else "index"]

    taken = {str(column) for column in existing_columns}
    labels: list[str] = []
    for base in raw:
        candidate = base
        if candidate in taken or candidate in labels:
            suffix = 1
            while f"{base}_{suffix}" in taken or f"{base}_{suffix}" in labels:
                suffix += 1
            candidate = f"{base}_{suffix}"
        labels.append(candidate)
        taken.add(candidate)
    return labels


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "item"
