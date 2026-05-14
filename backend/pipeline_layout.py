from __future__ import annotations

from typing import Any

from backend.pipeline_graph import (
    clone_pipeline_payload,
    compute_layers,
    next_identifier,
    topological_order,
)
from backend.schemas import DEFAULT_COMMENT_COLOR

DEFAULT_NODE_WIDTH = 240.0
DEFAULT_NODE_HEIGHT = 150.0
COMMENT_PADDING_X = 36.0
COMMENT_PADDING_TOP = 44.0
COMMENT_PADDING_BOTTOM = 28.0
LAYER_GAP_X = 340.0
NODE_GAP_Y = 72.0
GROUP_GAP_Y = 36.0
START_X = 80.0
START_Y = 120.0


def prettify_pipeline_layout(pipeline: dict[str, Any]) -> dict[str, Any]:
    normalized = clone_pipeline_payload(pipeline)
    nodes = normalized.get("nodes", [])
    if not nodes:
        return normalized

    topo_order = topological_order(normalized)
    topo_index = {node_id: index for index, node_id in enumerate(topo_order)}
    layers = compute_layers(normalized)
    group_order = {
        str(group["id"]): index
        for index, group in enumerate(normalized.get("groups", []))
    }

    nodes_by_layer: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        layer = layers[str(node["id"])]
        nodes_by_layer.setdefault(layer, []).append(node)

    for layer, layer_nodes in nodes_by_layer.items():
        layer_nodes.sort(
            key=lambda node: (
                _primary_group_rank(node, group_order),
                topo_index[str(node["id"])],
                str(node["id"]),
            )
        )
        current_y = START_Y
        last_primary_group: str | None = None
        for node in layer_nodes:
            width = float(node.get("width") or DEFAULT_NODE_WIDTH)
            height = float(node.get("height") or DEFAULT_NODE_HEIGHT)
            primary_group = _primary_group_id(node)
            if last_primary_group is not None and primary_group != last_primary_group:
                current_y += GROUP_GAP_Y
            node["position"] = {"x": START_X + (layer * LAYER_GAP_X), "y": current_y}
            node.setdefault("width", width)
            node.setdefault("height", height)
            current_y += height + NODE_GAP_Y
            last_primary_group = primary_group

    _update_group_comments(normalized)
    return normalized


def _primary_group_id(node: dict[str, Any]) -> str | None:
    group_ids = node.get("group_ids", [])
    if not isinstance(group_ids, list) or not group_ids:
        return None
    return str(group_ids[0])


def _primary_group_rank(node: dict[str, Any], group_order: dict[str, int]) -> tuple[int, str]:
    primary_group = _primary_group_id(node)
    if primary_group is None:
        return (len(group_order), str(node["id"]))
    return (group_order.get(primary_group, len(group_order)), primary_group)


def _update_group_comments(pipeline: dict[str, Any]) -> None:
    existing_comments = list(pipeline.get("comments", []))
    manual_comments = [
        comment
        for comment in existing_comments
        if not bool(comment.get("managed")) or not comment.get("group_id")
    ]
    existing_managed = {
        str(comment.get("group_id")): comment
        for comment in existing_comments
        if bool(comment.get("managed")) and comment.get("group_id")
    }
    used_comment_ids = {str(comment["id"]) for comment in manual_comments}

    nodes = {str(node["id"]): node for node in pipeline.get("nodes", [])}
    managed_comments: list[dict[str, Any]] = []
    fallback_y = _max_graph_y(pipeline) + 120.0

    for group in pipeline.get("groups", []):
        group_id = str(group["id"])
        member_nodes = [
            node
            for node in nodes.values()
            if group_id in [str(item) for item in node.get("group_ids", [])]
        ]
        existing_comment = existing_managed.get(group_id)
        comment_id = str(group.get("comment_id") or "")
        if not comment_id:
            comment_id = str(existing_comment["id"]) if existing_comment is not None else ""
        if not comment_id or comment_id in used_comment_ids:
            comment_id = next_identifier(used_comment_ids, "comment")
        used_comment_ids.add(comment_id)
        group["comment_id"] = comment_id

        if member_nodes:
            min_x = min(float(node["position"]["x"]) for node in member_nodes)
            min_y = min(float(node["position"]["y"]) for node in member_nodes)
            max_x = max(
                float(node["position"]["x"]) + float(node.get("width") or DEFAULT_NODE_WIDTH)
                for node in member_nodes
            )
            max_y = max(
                float(node["position"]["y"]) + float(node.get("height") or DEFAULT_NODE_HEIGHT)
                for node in member_nodes
            )
            position = {"x": min_x - COMMENT_PADDING_X, "y": min_y - COMMENT_PADDING_TOP}
            width = max((max_x - min_x) + (COMMENT_PADDING_X * 2), 280.0)
            height = max(
                (max_y - min_y) + COMMENT_PADDING_TOP + COMMENT_PADDING_BOTTOM,
                140.0,
            )
        else:
            prior_position = existing_comment.get("position") if existing_comment else None
            position = prior_position or {"x": START_X, "y": fallback_y}
            width = float(existing_comment.get("width") or 300.0) if existing_comment else 300.0
            height = float(existing_comment.get("height") or 150.0) if existing_comment else 150.0
            fallback_y += height + 40.0

        managed_comments.append(
            {
                "id": comment_id,
                "title": str(group.get("name", "")),
                "description": str(group.get("description", "")),
                "color": str(existing_comment.get("color") or DEFAULT_COMMENT_COLOR)
                if existing_comment is not None
                else DEFAULT_COMMENT_COLOR,
                "position": position,
                "width": width,
                "height": height,
                "managed": True,
                "group_id": group_id,
            }
        )

    pipeline["comments"] = manual_comments + managed_comments


def _max_graph_y(pipeline: dict[str, Any]) -> float:
    max_y = START_Y
    for node in pipeline.get("nodes", []):
        position = node.get("position") or {}
        node_y = float(position.get("y") or START_Y)
        height = float(node.get("height") or DEFAULT_NODE_HEIGHT)
        max_y = max(max_y, node_y + height)
    return max_y
