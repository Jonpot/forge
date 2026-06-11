"""History-hash computation — the Tier 2 staleness recipe from DESIGN.md §7.

history_hash = sha256(canonical_json({
    fn:      source hash of the decorated function (AST-normalized),
    closure: hash of the defining module's repo import-closure,
    env:     environment fingerprint (python version + dependency files),
    params:  literal params for UNCONNECTED parameters only,
    inputs:  {param_name: [parent_history_hash, source_output]},
}))

A node is stale iff no checkpoint exists for its computed hash. Everything
here is pure stdlib and cheap enough to recompute on every document edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

from starforge.core.spec import PipelineDoc
from starforge.index.scanner import WorkspaceIndex

#: Doc-native nodes that execute without importing user code. Constants are
#: the first; the snippet node (DESIGN.md §10) will join this namespace.
BUILTIN_PREFIX = "builtin:"
BUILTINS = {"builtin:constant"}

#: Dependency manifests folded into the environment fingerprint. pyproject is
#: deliberately excluded — version bumps would invalidate every checkpoint.
ENV_FILES = (
    "requirements.txt",
    "requirements.lock",
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def env_fingerprint(workspace: str | Path) -> str:
    workspace = Path(workspace)
    parts = [f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"]
    for name in ENV_FILES:
        path = workspace / name
        if path.is_file():
            try:
                parts.append(f"{name}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
            except OSError:
                continue
    return _sha("\n".join(parts))


@dataclass
class NodeState:
    history_hash: str | None = None
    stale: bool = True
    #: Human-readable reasons the node cannot hash or run (missing block,
    #: cycle membership, bad edge target, ...). Empty means healthy.
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"history_hash": self.history_hash, "stale": self.stale, "problems": self.problems}


def toposort(doc: PipelineDoc) -> tuple[list[str], set[str]]:
    """Kahn's algorithm. Returns (ordered_ids, ids_stuck_in_cycles)."""
    indegree = {n.id: 0 for n in doc.nodes}
    children: dict[str, list[str]] = {n.id: [] for n in doc.nodes}
    for edge in doc.edges:
        if edge.source in indegree and edge.target in indegree:
            indegree[edge.target] += 1
            children[edge.source].append(edge.target)
    ready = sorted(nid for nid, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for child in children[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()  # determinism beats micro-speed at canvas scale
    return order, {nid for nid, deg in indegree.items() if deg > 0}


def compute_states(
    doc: PipelineDoc,
    index: WorkspaceIndex,
    env_fp: str,
    checkpoint_exists: Callable[[str], bool],
) -> dict[str, NodeState]:
    """Hash every node and decide staleness. Never raises on a sick document —
    problems are reported per-node so the canvas can render them in place."""
    blocks = index.blocks
    states: dict[str, NodeState] = {n.id: NodeState() for n in doc.nodes}
    order, cyclic = toposort(doc)
    for nid in cyclic:
        states[nid].problems.append("part of a dependency cycle")

    for nid in order:
        node = doc.node(nid)
        state = states[nid]

        if node.block.startswith(BUILTIN_PREFIX):
            if node.block not in BUILTINS:
                state.problems.append(f"unknown builtin '{node.block}'")
                continue
            if doc.in_edges(nid):
                state.problems.append("Constant is a source node and takes no inputs")
                continue
            # Deliberately excludes env/closure: a constant's identity is its
            # value, so checkpoints survive dependency upgrades and edits.
            state.history_hash = _sha(canonical_json({"builtin": node.block, "params": node.params}))
            state.stale = not checkpoint_exists(state.history_hash)
            continue

        info = blocks.get(node.block)
        if info is None:
            state.problems.append(f"block '{node.block}' not found in workspace (decorator removed or file deleted?)")
            continue

        param_names = {p.name for p in info.params}
        inputs: dict[str, list[str]] = {}
        broken = False
        for edge in doc.in_edges(nid):
            parent_state = states.get(edge.source)
            if parent_state is None or parent_state.history_hash is None:
                state.problems.append(f"input '{edge.target_param}' depends on unresolvable node '{edge.source}'")
                broken = True
                continue
            if edge.target_param not in param_names:
                state.problems.append(f"edge targets unknown parameter '{edge.target_param}'")
                broken = True
                continue
            if edge.target_param in inputs:
                state.problems.append(f"parameter '{edge.target_param}' has multiple incoming edges")
                broken = True
                continue
            parent_block = blocks.get(doc.node(edge.source).block)
            if parent_block is not None and edge.source_output not in parent_block.outputs:
                state.problems.append(
                    f"edge expects output '{edge.source_output}' but '{parent_block.label}' produces {parent_block.outputs}"
                )
                broken = True
                continue
            inputs[edge.target_param] = [parent_state.history_hash, edge.source_output]
        if broken:
            continue

        literals = {k: v for k, v in node.params.items() if k not in inputs}
        state.history_hash = _sha(
            canonical_json(
                {
                    "fn": info.source_hash,
                    "closure": index.closure_hash(info.module),
                    "env": env_fp,
                    "params": literals,
                    "inputs": inputs,
                }
            )
        )
        state.stale = not checkpoint_exists(state.history_hash)

    return states
