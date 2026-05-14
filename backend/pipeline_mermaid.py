from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import heapq
import re
from typing import Any


PASS_THROUGH_BLOCK_KEYS = {"noop", "no-op", "no_op"}
LEAF_NODE_THRESHOLD = 8


def _safe_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)


def _esc(text: str) -> str:
    return text.replace('"', "&quot;").replace("\n", " ")


def _edge_line(src: str, tgt: str, count: int) -> str:
    if count > 1:
        return f"    {_safe_id(src)} -->|{count}| {_safe_id(tgt)}"
    return f"    {_safe_id(src)} --> {_safe_id(tgt)}"


def _build_group_hierarchy(
    pipeline: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str | None],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    """
    Returns:
        groups          group_id -> group dict
        parent_of       group_id -> parent group_id (None = root), inferred from comment containment
        children_of     group_id -> [child group_ids]
        nodes_in_group  group_id -> [node_ids whose *primary* valid group is this group]
    """
    groups = {str(group["id"]): group for group in pipeline.get("groups", [])}
    nodes = {str(node["id"]): node for node in pipeline.get("nodes", [])}
    comments = {str(comment["id"]): comment for comment in pipeline.get("comments", [])}

    def _group_rect(group_id: str) -> tuple[float, float, float, float] | None:
        comment_id = str(groups[group_id].get("comment_id") or "")
        comment = comments.get(comment_id)
        if not comment:
            return None
        position = comment.get("position") or {}
        return (
            float(position.get("x", 0)),
            float(position.get("y", 0)),
            float(comment.get("width", 300)),
            float(comment.get("height", 150)),
        )

    def _contains(
        outer: tuple[float, float, float, float],
        inner: tuple[float, float, float, float],
    ) -> bool:
        outer_x, outer_y, outer_width, outer_height = outer
        inner_x, inner_y, inner_width, inner_height = inner
        return (
            outer_x <= inner_x
            and outer_y <= inner_y
            and (outer_x + outer_width) >= (inner_x + inner_width)
            and (outer_y + outer_height) >= (inner_y + inner_height)
        )

    parent_of: dict[str, str | None] = {}
    for group_id in groups:
        inner_rect = _group_rect(group_id)
        if inner_rect is None:
            parent_of[group_id] = None
            continue
        best_parent: str | None = None
        best_area = float("inf")
        for other_group_id in groups:
            if other_group_id == group_id:
                continue
            outer_rect = _group_rect(other_group_id)
            if outer_rect is None:
                continue
            if _contains(outer_rect, inner_rect):
                area = outer_rect[2] * outer_rect[3]
                if area < best_area:
                    best_area = area
                    best_parent = other_group_id
        parent_of[group_id] = best_parent

    children_of: dict[str, list[str]] = {group_id: [] for group_id in groups}
    for group_id, parent_group_id in parent_of.items():
        if parent_group_id:
            children_of[parent_group_id].append(group_id)

    nodes_in_group: dict[str, list[str]] = {group_id: [] for group_id in groups}
    for node_id, node in nodes.items():
        group_ids = [str(group_id) for group_id in node.get("group_ids", []) if str(group_id) in groups]
        if group_ids:
            nodes_in_group.setdefault(group_ids[0], []).append(node_id)

    return groups, parent_of, children_of, nodes_in_group


def _group_ancestor_path(group_id: str, parent_of: dict[str, str | None]) -> list[str]:
    """[group_id, parent, ..., root_group] (group_id-to-root order)"""
    path = [group_id]
    current = group_id
    while parent_of.get(current):
        current = parent_of[current]
        path.append(current)
    return path


def _node_group_path(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    parent_of: dict[str, str | None],
) -> list[str]:
    """Path from node's primary valid group to root. Empty list when ungrouped."""
    group_ids = [str(group_id) for group_id in nodes.get(node_id, {}).get("group_ids", []) if str(group_id) in groups]
    if not group_ids:
        return []
    return _group_ancestor_path(group_ids[0], parent_of)


def _topological_order(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[str]:
    order_hint = {node_id: index for index, node_id in enumerate(nodes)}
    indegree = {node_id: 0 for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}

    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in nodes or target not in nodes:
            continue
        outgoing[source].append(target)
        indegree[target] += 1

    heap: list[tuple[int, str]] = []
    for node_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(heap, (order_hint[node_id], node_id))

    ordered: list[str] = []
    while heap:
        _, node_id = heapq.heappop(heap)
        ordered.append(node_id)
        for child in sorted(outgoing[node_id], key=lambda candidate: (order_hint[candidate], candidate)):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, (order_hint[child], child))

    remaining = [node_id for node_id in nodes if node_id not in set(ordered)]
    remaining.sort(key=lambda node_id: (order_hint[node_id], node_id))
    ordered.extend(remaining)
    return ordered


def _is_pass_through(node: dict[str, Any], block_label: str) -> bool:
    block_key = str(node.get("block") or "").strip().lower()
    if block_key in PASS_THROUGH_BLOCK_KEYS:
        return True
    return block_label.strip().lower() in PASS_THROUGH_BLOCK_KEYS


@dataclass
class TopLevelChunk:
    id: str
    name: str
    kind: str
    node_ids: list[str]
    source_group_id: str | None = None
    adopted_orphan_node_ids: list[str] = field(default_factory=list)


@dataclass
class InspectChild:
    id: str
    name: str
    kind: str
    node_ids: list[str]
    inspect_with: str


@dataclass
class InspectScope:
    id: str | None
    name: str
    kind: str
    node_ids: list[str]
    children: list[InspectChild] = field(default_factory=list)
    child_edges: list[tuple[str, str, int]] = field(default_factory=list)


@dataclass
class _RenderContext:
    pipeline: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    groups: dict[str, dict[str, Any]]
    parent_of: dict[str, str | None]
    children_of: dict[str, list[str]]
    nodes_in_group: dict[str, list[str]]
    block_labels: dict[str, str]
    node_rank: dict[str, int]
    descendant_cache: dict[str, list[str]] = field(default_factory=dict)
    top_level_chunks: list[TopLevelChunk] = field(default_factory=list)
    top_level_edges: list[tuple[str, str, int]] = field(default_factory=list)
    top_level_chunk_map: dict[str, TopLevelChunk] = field(default_factory=dict)


def _build_context(
    pipeline: dict[str, Any],
    block_names: dict[str, str] | None = None,
) -> _RenderContext:
    nodes = {str(node["id"]): node for node in pipeline.get("nodes", [])}
    edges = pipeline.get("edges", [])
    groups, parent_of, children_of, nodes_in_group = _build_group_hierarchy(pipeline)
    topo = _topological_order(nodes, edges)
    block_labels = {
        node_id: str((block_names or {}).get(node_id) or nodes[node_id].get("block") or node_id)
        for node_id in nodes
    }
    node_rank = {node_id: index for index, node_id in enumerate(topo)}
    context = _RenderContext(
        pipeline=pipeline,
        nodes=nodes,
        edges=edges,
        groups=groups,
        parent_of=parent_of,
        children_of=children_of,
        nodes_in_group=nodes_in_group,
        block_labels=block_labels,
        node_rank=node_rank,
    )
    context.top_level_chunks, context.top_level_edges = _build_top_level_chunks(context)
    context.top_level_chunk_map = {chunk.id: chunk for chunk in context.top_level_chunks}
    return context


def _descendant_node_ids(context: _RenderContext, group_id: str) -> list[str]:
    if group_id in context.descendant_cache:
        return context.descendant_cache[group_id]

    collected = list(context.nodes_in_group.get(group_id, []))
    for child_group_id in context.children_of.get(group_id, []):
        collected.extend(_descendant_node_ids(context, child_group_id))

    collected.sort(key=lambda node_id: (context.node_rank.get(node_id, 10**9), node_id))
    context.descendant_cache[group_id] = collected
    return collected


def _group_sort_key(context: _RenderContext, group_id: str) -> tuple[int, str, str]:
    member_ids = _descendant_node_ids(context, group_id)
    min_rank = min((context.node_rank[node_id] for node_id in member_ids), default=10**9)
    name = str(context.groups[group_id].get("name") or group_id)
    return (min_rank, name.lower(), group_id)


def _build_top_level_chunks(
    context: _RenderContext,
) -> tuple[list[TopLevelChunk], list[tuple[str, str, int]]]:
    group_node_count = {
        group_id: len(_descendant_node_ids(context, group_id))
        for group_id in context.groups
    }

    seed_group_ids = [
        group_id
        for group_id in context.groups
        if group_node_count[group_id] > 0
        and all(
            group_node_count.get(ancestor_group_id, 0) == 0
            for ancestor_group_id in _group_ancestor_path(group_id, context.parent_of)[1:]
        )
    ]
    seed_group_ids.sort(key=lambda group_id: _group_sort_key(context, group_id))
    seed_group_set = set(seed_group_ids)

    preassigned_chunk_of_node: dict[str, str] = {}
    grouped_chunk_nodes: dict[str, list[str]] = {group_id: [] for group_id in seed_group_ids}
    orphan_nodes: set[str] = set()

    for node_id in sorted(context.nodes, key=lambda candidate: (context.node_rank[candidate], candidate)):
        group_path = _node_group_path(node_id, context.nodes, context.groups, context.parent_of)
        assigned_seed = next((group_id for group_id in group_path if group_id in seed_group_set), None)
        if assigned_seed is not None:
            preassigned_chunk_of_node[node_id] = assigned_seed
            grouped_chunk_nodes[assigned_seed].append(node_id)
        else:
            orphan_nodes.add(node_id)

    orphan_adjacency: dict[str, set[str]] = {node_id: set() for node_id in orphan_nodes}
    for edge in context.edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in orphan_nodes and target in orphan_nodes:
            orphan_adjacency[source].add(target)
            orphan_adjacency[target].add(source)

    orphan_components: list[list[str]] = []
    visited: set[str] = set()
    for node_id in sorted(orphan_nodes, key=lambda candidate: (context.node_rank[candidate], candidate)):
        if node_id in visited:
            continue
        queue = deque([node_id])
        visited.add(node_id)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(
                orphan_adjacency.get(current, set()),
                key=lambda candidate: (context.node_rank[candidate], candidate),
            ):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        component.sort(key=lambda candidate: (context.node_rank[candidate], candidate))
        orphan_components.append(component)

    adopted_orphans: dict[str, list[str]] = defaultdict(list)
    orphan_chunks: list[TopLevelChunk] = []
    seed_names = {
        group_id: str(context.groups[group_id].get("name") or group_id)
        for group_id in seed_group_ids
    }

    for index, component in enumerate(orphan_components, start=1):
        connectivity: Counter[str] = Counter()
        component_set = set(component)
        for edge in context.edges:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if source in component_set and target in preassigned_chunk_of_node:
                connectivity[preassigned_chunk_of_node[target]] += 1
            if target in component_set and source in preassigned_chunk_of_node:
                connectivity[preassigned_chunk_of_node[source]] += 1

        if connectivity:
            ranked_neighbors = sorted(
                connectivity.items(),
                key=lambda item: (-item[1], seed_group_ids.index(item[0]), item[0]),
            )
            if len(ranked_neighbors) == 1 or ranked_neighbors[0][1] > ranked_neighbors[1][1]:
                winner = ranked_neighbors[0][0]
                grouped_chunk_nodes[winner].extend(component)
                adopted_orphans[winner].extend(component)
                continue
        else:
            ranked_neighbors = []

        all_pass_through = all(
            _is_pass_through(context.nodes[node_id], context.block_labels[node_id])
            for node_id in component
        )
        if ranked_neighbors:
            ranked_names = [seed_names[group_id] for group_id, _count in ranked_neighbors[:2]]
            prefix = "Scaffold Bridge" if all_pass_through else "Ungrouped Bridge"
            orphan_name = f"{prefix}: {' / '.join(ranked_names)}"
        elif len(component) == 1:
            only_node_id = component[0]
            prefix = "Scaffolding" if all_pass_through else "Ungrouped"
            orphan_name = f"{prefix}: {only_node_id}"
        else:
            prefix = "Scaffold Component" if all_pass_through else "Ungrouped Component"
            orphan_name = f"{prefix} {index}"

        orphan_chunks.append(
            TopLevelChunk(
                id=f"orphan_{index}",
                name=orphan_name,
                kind="orphan",
                node_ids=list(component),
            )
        )

    chunks: list[TopLevelChunk] = []
    for group_id in seed_group_ids:
        node_ids = sorted(grouped_chunk_nodes[group_id], key=lambda node_id: (context.node_rank[node_id], node_id))
        chunks.append(
            TopLevelChunk(
                id=group_id,
                name=str(context.groups[group_id].get("name") or group_id),
                kind="group",
                node_ids=node_ids,
                source_group_id=group_id,
                adopted_orphan_node_ids=sorted(
                    adopted_orphans.get(group_id, []),
                    key=lambda node_id: (context.node_rank[node_id], node_id),
                ),
            )
        )
    chunks.extend(orphan_chunks)
    chunks.sort(
        key=lambda chunk: (
            min((context.node_rank[node_id] for node_id in chunk.node_ids), default=10**9),
            chunk.name.lower(),
            chunk.id,
        )
    )

    node_to_chunk: dict[str, str] = {}
    for chunk in chunks:
        for node_id in chunk.node_ids:
            node_to_chunk[node_id] = chunk.id

    edge_counts: Counter[tuple[str, str]] = Counter()
    for edge in context.edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        source_chunk_id = node_to_chunk.get(source)
        target_chunk_id = node_to_chunk.get(target)
        if source_chunk_id and target_chunk_id and source_chunk_id != target_chunk_id:
            edge_counts[(source_chunk_id, target_chunk_id)] += 1

    chunk_order = {chunk.id: index for index, chunk in enumerate(chunks)}
    chunk_edges = [
        (source, target, count)
        for (source, target), count in sorted(
            edge_counts.items(),
            key=lambda item: (
                chunk_order[item[0][0]],
                chunk_order[item[0][1]],
                item[0][0],
                item[0][1],
            ),
        )
    ]
    return chunks, chunk_edges


def _frontier_groups(context: _RenderContext, parent_group_id: str) -> list[str]:
    frontier: list[str] = []

    def _walk(current_group_id: str) -> None:
        for child_group_id in sorted(
            context.children_of.get(current_group_id, []),
            key=lambda group_id: _group_sort_key(context, group_id),
        ):
            if _descendant_node_ids(context, child_group_id):
                frontier.append(child_group_id)
            else:
                _walk(child_group_id)

    _walk(parent_group_id)
    return frontier


def _direct_bucket_name(
    context: _RenderContext,
    node_ids: list[str],
    adopted_orphan_node_ids: list[str] | None = None,
) -> str:
    adopted_count = len(adopted_orphan_node_ids or [])
    if node_ids and all(
        _is_pass_through(context.nodes[node_id], context.block_labels[node_id])
        for node_id in node_ids
    ):
        return "Scaffolding"
    if adopted_count and adopted_count == len(node_ids):
        return "Adopted Orphans"
    if adopted_count:
        return "Direct Work + Orphans"
    return "Direct Work"


def _build_child_edges(
    context: _RenderContext,
    scope_node_ids: list[str],
    node_to_child: dict[str, str],
    child_order: dict[str, int],
) -> list[tuple[str, str, int]]:
    scope_node_set = set(scope_node_ids)
    edge_counts: Counter[tuple[str, str]] = Counter()
    for edge in context.edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in scope_node_set or target not in scope_node_set:
            continue
        source_child = node_to_child.get(source)
        target_child = node_to_child.get(target)
        if source_child and target_child and source_child != target_child:
            edge_counts[(source_child, target_child)] += 1

    return [
        (source, target, count)
        for (source, target), count in sorted(
            edge_counts.items(),
            key=lambda item: (
                child_order[item[0][0]],
                child_order[item[0][1]],
                item[0][0],
                item[0][1],
            ),
        )
    ]


def _build_group_scope(
    context: _RenderContext,
    *,
    scope_id: str,
    scope_name: str,
    group_id: str,
    allowed_node_ids: list[str] | None = None,
    adopted_orphan_node_ids: list[str] | None = None,
) -> InspectScope:
    scope_node_ids = list(allowed_node_ids or _descendant_node_ids(context, group_id))
    scope_node_ids.sort(key=lambda node_id: (context.node_rank[node_id], node_id))
    scope_node_set = set(scope_node_ids)

    frontier_group_ids = [
        child_group_id
        for child_group_id in _frontier_groups(context, group_id)
        if any(node_id in scope_node_set for node_id in _descendant_node_ids(context, child_group_id))
    ]

    children: list[InspectChild] = []
    node_to_child: dict[str, str] = {}
    frontier_node_ids: set[str] = set()

    for child_group_id in frontier_group_ids:
        child_node_ids = [
            node_id
            for node_id in _descendant_node_ids(context, child_group_id)
            if node_id in scope_node_set
        ]
        if not child_node_ids:
            continue
        children.append(
            InspectChild(
                id=child_group_id,
                name=str(context.groups[child_group_id].get("name") or child_group_id),
                kind="group",
                node_ids=child_node_ids,
                inspect_with="inspect_group",
            )
        )
        for node_id in child_node_ids:
            node_to_child[node_id] = child_group_id
            frontier_node_ids.add(node_id)

    direct_node_ids = [
        node_id for node_id in scope_node_ids
        if node_id not in frontier_node_ids
    ]
    if direct_node_ids:
        if not frontier_group_ids and len(direct_node_ids) <= LEAF_NODE_THRESHOLD:
            for node_id in direct_node_ids:
                children.append(
                    InspectChild(
                        id=node_id,
                        name=f'{context.block_labels[node_id]} [{node_id}]',
                        kind="node",
                        node_ids=[node_id],
                        inspect_with="inspect_block",
                    )
                )
                node_to_child[node_id] = node_id
        else:
            direct_child_id = f"{scope_id}__direct"
            children.append(
                InspectChild(
                    id=direct_child_id,
                    name=_direct_bucket_name(
                        context,
                        direct_node_ids,
                        adopted_orphan_node_ids=adopted_orphan_node_ids,
                    ),
                    kind="direct",
                    node_ids=direct_node_ids,
                    inspect_with="inspect_group",
                )
            )
            for node_id in direct_node_ids:
                node_to_child[node_id] = direct_child_id

    children.sort(
        key=lambda child: (
            min((context.node_rank[node_id] for node_id in child.node_ids), default=10**9),
            child.name.lower(),
            child.id,
        )
    )
    child_order = {child.id: index for index, child in enumerate(children)}
    child_edges = _build_child_edges(context, scope_node_ids, node_to_child, child_order)
    return InspectScope(
        id=scope_id,
        name=scope_name,
        kind="group",
        node_ids=scope_node_ids,
        children=children,
        child_edges=child_edges,
    )


def _build_direct_scope(
    context: _RenderContext,
    *,
    scope_id: str,
    scope_name: str,
    node_ids: list[str],
) -> InspectScope:
    sorted_node_ids = sorted(node_ids, key=lambda node_id: (context.node_rank[node_id], node_id))
    children = [
        InspectChild(
            id=node_id,
            name=f'{context.block_labels[node_id]} [{node_id}]',
            kind="node",
            node_ids=[node_id],
            inspect_with="inspect_block",
        )
        for node_id in sorted_node_ids
    ]
    child_order = {child.id: index for index, child in enumerate(children)}
    node_to_child = {node_id: node_id for node_id in sorted_node_ids}
    child_edges = _build_child_edges(context, sorted_node_ids, node_to_child, child_order)
    return InspectScope(
        id=scope_id,
        name=scope_name,
        kind="direct",
        node_ids=sorted_node_ids,
        children=children,
        child_edges=child_edges,
    )


def _build_orphan_scope(context: _RenderContext, chunk: TopLevelChunk) -> InspectScope:
    return _build_direct_scope(
        context,
        scope_id=chunk.id,
        scope_name=chunk.name,
        node_ids=chunk.node_ids,
    )


def _build_root_scope(context: _RenderContext) -> InspectScope:
    children = [
        InspectChild(
            id=chunk.id,
            name=chunk.name,
            kind=chunk.kind,
            node_ids=chunk.node_ids,
            inspect_with="inspect_group",
        )
        for chunk in context.top_level_chunks
    ]
    return InspectScope(
        id=None,
        name="Pipeline",
        kind="root",
        node_ids=sorted(context.nodes, key=lambda node_id: (context.node_rank[node_id], node_id)),
        children=children,
        child_edges=list(context.top_level_edges),
    )


def _resolve_direct_scope(context: _RenderContext, direct_scope_id: str) -> InspectScope:
    parent_scope_id = direct_scope_id[: -len("__direct")]
    parent_scope = _resolve_scope(context, parent_scope_id)
    direct_child = next(
        (
            child
            for child in parent_scope.children
            if child.id == direct_scope_id and child.kind == "direct"
        ),
        None,
    )
    if direct_child is None:
        raise ValueError(f"Unknown target_group: {direct_scope_id}")
    return _build_direct_scope(
        context,
        scope_id=direct_child.id,
        scope_name=direct_child.name,
        node_ids=direct_child.node_ids,
    )


def _resolve_scope(
    context: _RenderContext,
    target_group: str | None,
) -> InspectScope:
    if target_group is None:
        return _build_root_scope(context)

    if target_group.endswith("__direct"):
        return _resolve_direct_scope(context, target_group)

    top_level_chunk = context.top_level_chunk_map.get(target_group)
    if top_level_chunk is not None:
        if top_level_chunk.kind == "group" and top_level_chunk.source_group_id is not None:
            return _build_group_scope(
                context,
                scope_id=top_level_chunk.id,
                scope_name=top_level_chunk.name,
                group_id=top_level_chunk.source_group_id,
                allowed_node_ids=top_level_chunk.node_ids,
                adopted_orphan_node_ids=top_level_chunk.adopted_orphan_node_ids,
            )
        return _build_orphan_scope(context, top_level_chunk)

    if target_group in context.groups:
        return _build_group_scope(
            context,
            scope_id=target_group,
            scope_name=str(context.groups[target_group].get("name") or target_group),
            group_id=target_group,
        )

    raise ValueError(f"Unknown target_group: {target_group}")


def _root_chunk_label(context: _RenderContext, child: InspectChild) -> str:
    chunk = context.top_level_chunk_map.get(child.id)
    label_parts = [child.name, f"{len(child.node_ids)} nodes"]
    if chunk is not None and chunk.adopted_orphan_node_ids:
        label_parts.append(f"+{len(chunk.adopted_orphan_node_ids)} adopted orphans")
    return "\\n".join(label_parts)


def _scope_child_label(
    context: _RenderContext,
    scope: InspectScope,
    child: InspectChild,
) -> str:
    if scope.kind == "root":
        return _root_chunk_label(context, child)
    if child.kind == "node":
        return child.name
    return f"{child.name}\\n({len(child.node_ids)} nodes)"


def _render_scope_mermaid(
    context: _RenderContext,
    scope: InspectScope,
) -> str:
    lines = ["graph TD"]
    for child in scope.children:
        label = _scope_child_label(context, scope, child)
        lines.append(f'    {_safe_id(child.id)}["{_esc(label)}"]')
    for source, target, count in scope.child_edges:
        lines.append(_edge_line(source, target, count))
    return "\n".join(lines)


def render_mermaid(
    pipeline: dict[str, Any],
    target_group: str | None = None,
    block_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Render a Mermaid ``graph TD`` structural summary of the pipeline.

    target_group=None
        Render the highest-level chunk DAG only.

    target_group=<group id>
        Render only the next-level structure inside that group/chunk.
    """
    context = _build_context(pipeline, block_names=block_names)
    scope = _resolve_scope(context, target_group)
    return {"mermaid": _render_scope_mermaid(context, scope)}


def inspect_group(
    pipeline: dict[str, Any],
    *,
    target_group: str,
    block_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    context = _build_context(pipeline, block_names=block_names)
    scope = _resolve_scope(context, target_group)
    if scope.id is None:
        raise ValueError("target_group is required")
    return {
        "id": scope.id,
        "name": scope.name,
        "kind": scope.kind,
        "node_count": len(scope.node_ids),
        "children": [
            {
                "id": child.id,
                "name": child.name,
                "kind": child.kind,
                "node_count": len(child.node_ids),
                "inspect_with": child.inspect_with,
            }
            for child in scope.children
        ],
        "mermaid": _render_scope_mermaid(context, scope),
    }
