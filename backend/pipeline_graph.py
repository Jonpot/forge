from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import json
import re
from typing import Any

from backend.schemas import normalize_pipeline_payload


def clone_pipeline_payload(pipeline: dict[str, Any]) -> dict[str, Any]:
    return normalize_pipeline_payload(json.loads(json.dumps(pipeline)))


def node_map(pipeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in pipeline.get("nodes", [])}


def group_map(pipeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(group["id"]): group for group in pipeline.get("groups", [])}


def _parse_handle_suffix(value: str | None) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+)$", value.strip())
    if match is None:
        return None
    return int(match.group(1))


def edge_target_input(edge: dict[str, Any]) -> int | None:
    if isinstance(edge.get("target_input"), int):
        return int(edge["target_input"])
    return _parse_handle_suffix(edge.get("targetHandle"))


def edge_source_output_handle(edge: dict[str, Any]) -> str:
    source_output = edge.get("source_output")
    if isinstance(source_output, int) and source_output >= 0:
        return f"output_{source_output}"
    if isinstance(source_output, str) and source_output.strip():
        return normalize_output_handle(source_output)

    source_handle = edge.get("sourceHandle")
    if isinstance(source_handle, str) and source_handle.strip():
        return normalize_output_handle(source_handle)
    return "output_0"


def normalize_output_handle(value: str) -> str:
    text = value.strip()
    if text.startswith("output_"):
        return text
    parsed = _parse_handle_suffix(text)
    if parsed is None:
        return text
    return f"output_{parsed}"


def expected_output_handles(
    block_cls: type[Any], params: dict[str, Any] | None = None
) -> list[str]:
    resolver = getattr(block_cls, "resolve_output_labels", None)
    if callable(resolver):
        labels = resolver(params)
    else:
        labels = getattr(block_cls, "output_labels", ["output"])
    if isinstance(labels, (list, tuple)):
        count = max(len(labels), 1)
    else:
        count = 1
    return [f"output_{index}" for index in range(count)]


def build_adjacency(
    pipeline: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    nodes = node_map(pipeline)
    incoming = {node_id: [] for node_id in nodes}
    outgoing = {node_id: [] for node_id in nodes}
    for edge in pipeline.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in nodes or target not in nodes:
            raise ValueError(f"Edge references unknown node: {edge}")
        incoming[target].append(edge)
        outgoing[source].append(edge)
    return incoming, outgoing


def topological_order(pipeline: dict[str, Any]) -> list[str]:
    nodes = node_map(pipeline)
    incoming, outgoing = build_adjacency(pipeline)
    indegree = {node_id: len(edges) for node_id, edges in incoming.items()}
    ordered_node_ids = [str(node["id"]) for node in pipeline.get("nodes", [])]
    queue = deque(
        node_id
        for node_id in ordered_node_ids
        if indegree.get(node_id, 0) == 0
    )
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for edge in outgoing.get(node_id, []):
            child = str(edge["target"])
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise ValueError("Pipeline graph contains at least one cycle.")
    return order


def compute_layers(pipeline: dict[str, Any]) -> dict[str, int]:
    order = topological_order(pipeline)
    incoming, _ = build_adjacency(pipeline)
    layers: dict[str, int] = {}
    for node_id in order:
        parent_layers = [layers[str(edge["source"])] for edge in incoming[node_id]]
        layers[node_id] = (max(parent_layers) + 1) if parent_layers else 0
    return layers


def collect_used_node_ids(pipeline: dict[str, Any]) -> set[str]:
    return {str(node["id"]) for node in pipeline.get("nodes", [])}


def collect_used_group_ids(pipeline: dict[str, Any]) -> set[str]:
    return {str(group["id"]) for group in pipeline.get("groups", [])}


def find_edge_by_id(pipeline: dict[str, Any], edge_id: str) -> dict[str, Any] | None:
    for edge in pipeline.get("edges", []):
        if str(edge.get("id")) == edge_id:
            return edge
    return None


def match_edge_tuple(
    edge: dict[str, Any],
    *,
    source: str,
    target: str,
    source_output_handle: str | None = None,
    target_input: int | None = None,
) -> bool:
    if str(edge["source"]) != source or str(edge["target"]) != target:
        return False
    if source_output_handle is not None and edge_source_output_handle(edge) != source_output_handle:
        return False
    if target_input is not None and edge_target_input(edge) != target_input:
        return False
    return True


def next_identifier(existing_ids: Iterable[str], prefix: str) -> str:
    taken = {item for item in existing_ids if item}
    index = 1
    while f"{prefix}_{index}" in taken:
        index += 1
    return f"{prefix}_{index}"
