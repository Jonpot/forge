"""The ``.forge`` pipeline document model.

Documents are plain JSON text — the VS Code custom editor is text-based so
undo/redo/diff/merge come for free. Layout and notes are durable metadata and
never participate in provenance hashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

SCHEMA = "starforge/1"


@dataclass
class Node:
    id: str
    block: str  # "dotted.module:qualname"
    params: dict[str, Any] = field(default_factory=dict)  # literals for unconnected params
    position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "block": self.block,
            "params": self.params,
            "position": self.position,
            "notes": self.notes,
        }


@dataclass
class Edge:
    id: str
    source: str
    target: str
    target_param: str  # parameter NAME on the target function — never a slot index
    source_output: str = "output"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "source_output": self.source_output,
            "target": self.target,
            "target_param": self.target_param,
        }


@dataclass
class PipelineDoc:
    name: str = "Untitled"
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def in_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineDoc":
        schema = d.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"unsupported .forge schema {schema!r} (expected {SCHEMA!r})")
        nodes = [
            Node(
                id=n["id"],
                block=n["block"],
                params=dict(n.get("params", {})),
                position=dict(n.get("position", {"x": 0.0, "y": 0.0})),
                notes=n.get("notes", ""),
            )
            for n in d.get("nodes", [])
        ]
        edges = [
            Edge(
                id=e["id"],
                source=e["source"],
                source_output=e.get("source_output", "output"),
                target=e["target"],
                target_param=e["target_param"],
            )
            for e in d.get("edges", [])
        ]
        return cls(name=d.get("name", "Untitled"), nodes=nodes, edges=edges)

    @classmethod
    def from_json(cls, text: str) -> "PipelineDoc":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "PipelineDoc":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")
