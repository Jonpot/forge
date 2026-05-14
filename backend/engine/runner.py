from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import re

import pandas as pd
from pydantic import ValidationError

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    InsufficientInputs,
)
from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.provenance import (
    Provenance,
    canonical_json,
    combine_parent_history_hashes,
    compute_history_hash,
    compute_initial_data_signature,
    sha256_text,
)
from backend.progress import reset_progress_context, set_progress_context
from backend.registry import BlockRegistry


@dataclass(slots=True)
class NodeExecutionResult:
    node_id: str
    checkpoint_id: str
    history_hash: str
    status: str  # executed | reused


@dataclass(slots=True)
class PipelineRunResult:
    topological_order: list[str]
    node_results: dict[str, NodeExecutionResult]
    executed_nodes: list[str]
    reused_nodes: list[str]


@dataclass(slots=True)
class ParentInputRef:
    edge_index: int
    target_input: int | None
    source_node_id: str
    source_output_handle: str


class PipelineRunner:
    def __init__(
        self, registry: BlockRegistry, checkpoint_store: CheckpointStore
    ) -> None:
        self.registry = registry
        self.checkpoint_store = checkpoint_store

    def run_pipeline(
        self,
        pipeline: dict[str, Any],
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PipelineRunResult:
        node_map, incoming, topo_order = self._prepare_graph(pipeline)
        node_results: dict[str, NodeExecutionResult] = {}
        node_outputs: dict[str, dict[str, pd.DataFrame]] = {}
        node_checkpoint_ids: dict[str, str] = {}
        computed_hashes: dict[str, str] = {}
        executed_nodes: list[str] = []
        reused_nodes: list[str] = []

        if event_callback is not None:
            event_callback(
                {
                    "type": "run_status",
                    "status": "started",
                    "topological_order": topo_order,
                }
            )

        for node_id in topo_order:
            node = node_map[node_id]
            block_cls = self.registry.get(node["block"])
            block = block_cls()
            params_payload = self._params_payload(node, block_cls)
            params_obj = self._instantiate_params(block_cls, params_payload)

            expected_num_inputs = getattr(block_cls, "n_inputs", 1)
            parent_refs = self._sorted_parent_refs(
                node_id, incoming, expected_slots=expected_num_inputs
            )
            connected_refs = [ref for ref in parent_refs if ref is not None]

            # Skip source-less non-source blocks before emitting any events.
            # Blocks with at least one connected input proceed and decide for
            # themselves (via InsufficientInputs) whether they can run.
            if expected_num_inputs > 0 and not connected_refs:
                continue

            # Skip if any connected parent didn't produce output (was itself skipped).
            if connected_refs and not all(
                ref.source_node_id in node_outputs for ref in connected_refs
            ):
                continue

            parent_history_hash = self._parent_history_hash(
                node, parent_refs, computed_hashes
            )
            history_hash = compute_history_hash(
                parent_history_hash,
                block_cls.name,
                block_cls.version,
                params_payload,
            )
            computed_hashes[node_id] = history_hash

            if event_callback is not None:
                event_callback(
                    {"type": "node_status", "node_id": node_id, "status": "running"}
                )

            try:
                existing_checkpoint_id = (
                    self.checkpoint_store.get_checkpoint_id_by_hash(history_hash)
                )
                should_force_execute = bool(
                    block_cls.should_force_execute(params_payload)
                )
                if existing_checkpoint_id is not None and not should_force_execute:
                    output_handles = self._expected_output_handles(block_cls)
                    node_outputs[node_id] = self.checkpoint_store.load_outputs(
                        existing_checkpoint_id, output_handles
                    )
                    node_checkpoint_ids[node_id] = existing_checkpoint_id
                    node_results[node_id] = NodeExecutionResult(
                        node_id=node_id,
                        checkpoint_id=existing_checkpoint_id,
                        history_hash=history_hash,
                        status="reused",
                    )
                    reused_nodes.append(node_id)
                    if event_callback is not None:
                        event_callback(
                            {
                                "type": "node_status",
                                "node_id": node_id,
                                "status": "complete",
                                "mode": "reused",
                                "checkpoint_id": existing_checkpoint_id,
                            }
                        )
                    continue

                input_data = self._resolve_input_data(
                    block_cls, parent_refs, node_outputs
                )
                block.validate(input_data)

                started = perf_counter()
                progress_token = None
                if event_callback is not None:
                    progress_token = set_progress_context(node_id, event_callback)
                try:
                    output = block.execute(input_data, params_obj)
                finally:
                    if progress_token is not None:
                        reset_progress_context(progress_token)
                duration = perf_counter() - started

                if not isinstance(output, BlockOutput):
                    raise TypeError(
                        f"Block {block_cls.__name__} returned {type(output)!r}; expected BlockOutput."
                    )

                output_frames = self._normalize_output_frames(block_cls, output)
                output_df = output_frames["output_0"]

                provenance = Provenance(
                    checkpoint_id="",
                    block_name=block_cls.name,
                    block_version=block_cls.version,
                    params=params_payload,
                    parent_checkpoint_ids=[
                        node_checkpoint_ids[parent.source_node_id]
                        for parent in parent_refs
                        if parent is not None
                    ],
                    initial_data_signature=parent_history_hash
                    if not parent_refs
                    else None,
                    history_hash=history_hash,
                    timestamp=datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    execution_time_seconds=round(duration, 6),
                    output_shape=[int(output_df.shape[0]), int(output_df.shape[1])],
                    images=[],
                )
                checkpoint_id = self.checkpoint_store.save(
                    data=output_df,
                    provenance=provenance,
                    outputs=output_frames,
                    images=output.images,
                )

                node_outputs[node_id] = output_frames
                node_checkpoint_ids[node_id] = checkpoint_id
                node_results[node_id] = NodeExecutionResult(
                    node_id=node_id,
                    checkpoint_id=checkpoint_id,
                    history_hash=history_hash,
                    status="executed",
                )
                executed_nodes.append(node_id)
                if event_callback is not None:
                    event_callback(
                        {
                            "type": "node_status",
                            "node_id": node_id,
                            "status": "complete",
                            "mode": "executed",
                            "checkpoint_id": checkpoint_id,
                        }
                    )
            except InsufficientInputs:
                # Block signalled that its inputs aren't ready — skip silently.
                if event_callback is not None:
                    event_callback(
                        {
                            "type": "node_status",
                            "node_id": node_id,
                            "status": "skipped",
                        }
                    )
                continue
            except Exception as exc:
                if event_callback is not None:
                    event_callback(
                        {
                            "type": "node_status",
                            "node_id": node_id,
                            "status": "error",
                            "message": str(exc),
                        }
                    )
                raise

        result = PipelineRunResult(
            topological_order=topo_order,
            node_results=node_results,
            executed_nodes=executed_nodes,
            reused_nodes=reused_nodes,
        )
        return result

    def compute_staleness(self, pipeline: dict[str, Any]) -> dict[str, bool]:
        hashes = self.compute_history_hashes(pipeline)
        return {
            node_id: not self.checkpoint_store.exists_by_hash(history_hash)
            for node_id, history_hash in hashes.items()
        }

    def compute_history_hashes(self, pipeline: dict[str, Any]) -> dict[str, str]:
        node_map, incoming, topo_order = self._prepare_graph(pipeline)
        computed_hashes: dict[str, str] = {}
        for node_id in topo_order:
            node = node_map[node_id]
            block_cls = self.registry.get(node["block"])
            expected_inputs = getattr(block_cls, "n_inputs", 1)
            parent_refs = self._sorted_parent_refs(
                node_id,
                incoming,
                expected_slots=expected_inputs,
            )
            connected_refs = [ref for ref in parent_refs if ref is not None]
            if expected_inputs > 0 and not connected_refs:
                continue
            if connected_refs and not all(
                parent.source_node_id in computed_hashes for parent in connected_refs
            ):
                continue
            parent_history_hash = self._parent_history_hash(
                node, parent_refs, computed_hashes
            )
            computed_hashes[node_id] = compute_history_hash(
                parent_history_hash,
                block_cls.name,
                block_cls.version,
                self._params_payload(node, block_cls),
            )
        return computed_hashes

    def _parent_history_hash(
        self,
        node: dict[str, Any],
        parent_refs: list[ParentInputRef | None],
        computed_hashes: dict[str, str],
    ) -> str:
        connected = [ref for ref in parent_refs if ref is not None]
        if not connected:
            return self._compute_initial_signature(node)
        return combine_parent_history_hashes(
            [
                self._parent_dependency_hash(
                    computed_hashes[parent.source_node_id],
                    parent.source_output_handle,
                )
                for parent in connected
            ]
        )

    def _resolve_input_data(
        self,
        block_cls: type[BaseBlock],
        parent_refs: list[ParentInputRef | None],
        outputs: dict[str, dict[str, pd.DataFrame]],
    ) -> Any:
        n_inputs = getattr(block_cls, "n_inputs", 1)
        if n_inputs == 0:
            return None
        if n_inputs == 1:
            ref = parent_refs[0] if parent_refs else None
            return (
                self._resolve_parent_output(ref, outputs) if ref is not None else None
            )
        # Multi-input: return a list of length n_inputs with None for disconnected slots.
        return [
            self._resolve_parent_output(ref, outputs) if ref is not None else None
            for ref in parent_refs
        ]

    def _validate_input_arity(
        self,
        block_cls: type[BaseBlock],
        parent_refs: list[ParentInputRef],
        node_id: str,
    ) -> None:
        expected = getattr(block_cls, "n_inputs", 1)
        if len(parent_refs) != expected:
            raise ValueError(
                f"Node {node_id} ({block_cls.__name__}) expects {expected} inputs, "
                f"found {len(parent_refs)}."
            )

    def _compute_initial_signature(self, node: dict[str, Any]) -> str:
        block_cls = self.registry.get(str(node.get("block")))
        params = self._params_payload(node, block_cls)
        filepath = params.get("filepath")
        if filepath:
            path = Path(filepath)
            if not path.is_absolute():
                workspace_dir = os.environ.get("FORGE_WORKSPACE_DIR", "")
                if workspace_dir:
                    path = Path(workspace_dir) / path
            if path.exists():
                return compute_initial_data_signature(path)

        # Fallback for root blocks that do not load file inputs.
        root_payload = {
            "block": node.get("block"),
            "params": params,
        }
        return sha256_text(canonical_json(root_payload))

    def _params_payload(
        self,
        node: dict[str, Any],
        block_cls: type[BaseBlock] | None = None,
    ) -> dict[str, Any]:
        params = node.get("params", {})
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise TypeError(f"Node params must be an object; got {type(params)!r}")
        if block_cls is None:
            return params
        return block_cls.normalize_params_payload(params)

    def _instantiate_params(
        self, block_cls: type[BaseBlock], params: dict[str, Any]
    ) -> Any:
        params_cls = getattr(block_cls, "Params", None)
        if params_cls is None:
            return None
        if not issubclass(params_cls, BlockParams):
            raise TypeError(
                f"{block_cls.__name__}.Params must inherit from BlockParams."
            )
        try:
            return params_cls.model_validate(params)
        except ValidationError as exc:
            raise BlockValidationError(
                f"{block_cls.__name__} received invalid params: {exc}"
            ) from exc

    def _prepare_graph(
        self, pipeline: dict[str, Any]
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, list[ParentInputRef]],
        list[str],
    ]:
        nodes = pipeline.get("nodes", [])
        edges = pipeline.get("edges", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise TypeError("Pipeline requires 'nodes' and 'edges' lists.")

        node_map: dict[str, dict[str, Any]] = {}
        node_order: dict[str, int] = {}
        for index, node in enumerate(nodes):
            node_id = node["id"]
            node_map[node_id] = node
            node_order[node_id] = index

        incoming: dict[str, list[ParentInputRef]] = {
            node_id: [] for node_id in node_map
        }
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_map}

        for edge_idx, edge in enumerate(edges):
            source = edge["source"]
            target = edge["target"]
            if source not in node_map or target not in node_map:
                raise ValueError(f"Edge references unknown node: {edge}")

            target_input = self._edge_target_input(edge)
            source_output_handle = self._edge_source_output_handle(edge)
            incoming[target].append(
                ParentInputRef(
                    edge_index=edge_idx,
                    target_input=target_input,
                    source_node_id=source,
                    source_output_handle=source_output_handle,
                )
            )
            outgoing[source].append(target)

        indegree = {node_id: len(incoming[node_id]) for node_id in node_map}
        queue = sorted(
            [node_id for node_id, degree in indegree.items() if degree == 0],
            key=lambda node_id: node_order[node_id],
        )
        topo: list[str] = []
        while queue:
            node_id = queue.pop(0)
            topo.append(node_id)
            for child in outgoing[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
                    queue.sort(key=lambda candidate: node_order[candidate])

        if len(topo) != len(node_map):
            raise ValueError("Pipeline graph contains at least one cycle.")
        return node_map, incoming, topo

    def _sorted_parent_refs(
        self,
        node_id: str,
        incoming: dict[str, list[ParentInputRef]],
        *,
        expected_slots: int | None = None,
    ) -> list[ParentInputRef | None]:
        """Return an ordered list of parent refs, one entry per input slot.

        When expected_slots > len(edges), trailing slots are None — those
        represent optional inputs that are not currently connected.  Callers
        should filter out None entries before checking whether parent nodes
        have run, and should pass None to the block so it can decide via
        InsufficientInputs whether it can proceed.
        """
        edges = sorted(incoming[node_id], key=lambda item: item.edge_index)

        slot_count = max(len(edges), expected_slots or 0)
        if slot_count == 0:
            return []

        slots: list[ParentInputRef | None] = [None] * slot_count
        unresolved: list[ParentInputRef] = []

        for edge in edges:
            if edge.target_input is None:
                unresolved.append(edge)
                continue

            if edge.target_input < 0 or edge.target_input >= slot_count:
                raise ValueError(
                    f"Node {node_id} has edge from '{edge.source_node_id}' targeting input {edge.target_input}, "
                    f"but only {slot_count} slot(s) exist."
                )
            if slots[edge.target_input] is not None:
                raise ValueError(
                    f"Node {node_id} has multiple edges targeting input {edge.target_input}."
                )
            slots[edge.target_input] = edge

        next_open = 0
        for edge in unresolved:
            while next_open < slot_count and slots[next_open] is not None:
                next_open += 1
            if next_open >= slot_count:
                raise ValueError(
                    f"Node {node_id} could not resolve input ordering for source '{edge.source_node_id}'."
                )
            slots[next_open] = edge
            next_open += 1

        # None entries beyond the connected edge count are intentional — they
        # represent unconnected optional input ports.
        return slots

    def _edge_target_input(self, edge: dict[str, Any]) -> int | None:
        if "target_input" in edge and isinstance(edge["target_input"], int):
            return edge["target_input"]

        target_handle = edge.get("targetHandle")
        if isinstance(target_handle, str):
            match = re.search(r"(\d+)$", target_handle)
            if match is not None:
                return int(match.group(1))
        return None

    def _edge_source_output_handle(self, edge: dict[str, Any]) -> str:
        source_output = edge.get("source_output")
        if isinstance(source_output, int) and source_output >= 0:
            return f"output_{source_output}"
        if isinstance(source_output, str):
            parsed = self._parse_output_handle(source_output)
            if parsed is not None:
                return parsed

        source_handle = edge.get("sourceHandle")
        if isinstance(source_handle, str):
            parsed = self._parse_output_handle(source_handle)
            if parsed is not None:
                return parsed
        return "output_0"

    def _parse_output_handle(self, value: str) -> str | None:
        text = value.strip()
        if not text:
            return None
        if text.startswith("output_"):
            return text
        match = re.search(r"(\d+)$", text)
        if match is None:
            return None
        return f"output_{int(match.group(1))}"

    def _expected_output_handles(self, block_cls: type[BaseBlock]) -> list[str]:
        output_labels = getattr(block_cls, "output_labels", ["output"])
        if isinstance(output_labels, (list, tuple)):
            count = max(len(output_labels), 1)
        else:
            count = 1
        return [f"output_{idx}" for idx in range(count)]

    def _normalize_output_frames(
        self,
        block_cls: type[BaseBlock],
        output: BlockOutput,
    ) -> dict[str, pd.DataFrame]:
        expected_handles = self._expected_output_handles(block_cls)
        for handle in expected_handles:
            if handle not in output.outputs:
                raise TypeError(
                    f"Block {block_cls.__name__} did not provide required handle '{handle}'."
                )

        normalized: dict[str, pd.DataFrame] = {}
        for handle, frame in output.outputs.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"Block {block_cls.__name__} output handle '{handle}' has data type {type(frame)!r}; expected DataFrame."
                )
            normalized[handle] = frame
        return normalized

    def _resolve_parent_output(
        self,
        parent_ref: ParentInputRef,
        outputs: dict[str, dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        parent_outputs = outputs.get(parent_ref.source_node_id)
        if parent_outputs is None:
            raise KeyError(
                f"Missing outputs for parent node '{parent_ref.source_node_id}'."
            )
        if parent_ref.source_output_handle not in parent_outputs:
            raise KeyError(
                f"Parent node '{parent_ref.source_node_id}' does not provide output handle '{parent_ref.source_output_handle}'."
            )
        return parent_outputs[parent_ref.source_output_handle]

    def _parent_dependency_hash(self, parent_hash: str, output_handle: str) -> str:
        if output_handle == "output_0":
            return parent_hash
        return sha256_text(f"{parent_hash}|{output_handle}")
