"""*Forge MCP server — agents author and run pipelines in the workspace.

Run over stdio:  python -m starforge.mcp  [workspace]

Tools wrap the same core engine the VS Code extension uses, file-based and
stateless: pipelines are the `.forge` JSON documents under
``.forge/pipelines/``, blocks come from the static AST index, and runs
execute in the standard per-run worker subprocess. The `mcp` dependency is
optional — ``pip install starforge-kernel[mcp]``.

The plain functions below are the implementation; FastMCP registration is a
thin veneer so everything stays unit-testable without an MCP client.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from starforge.core.checkpoints import CheckpointStore
from starforge.core.provenance import compute_states, env_fingerprint
from starforge.core.spec import PipelineDoc
from starforge.index import scan_workspace
from starforge.kernel.server import BUILTIN_PALETTE


def _workspace(path: str) -> Path:
    workspace = Path(path).resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace '{path}' is not a directory")
    return workspace


def _pipeline_path(workspace: Path, name: str) -> Path:
    candidate = (workspace / name).resolve()
    if candidate.suffix != ".forge":
        candidate = candidate.with_suffix(".forge")
    if workspace not in candidate.parents and candidate != workspace:
        raise ValueError("pipeline path escapes the workspace")
    return candidate


def list_blocks(workspace: str = ".") -> list[dict[str, Any]]:
    """Every block available in the workspace: builtins plus discovered
    @block functions, with params/outputs/annotations/docstrings."""
    ws = _workspace(workspace)
    index, _ = scan_workspace(ws)
    return list(BUILTIN_PALETTE) + [b.to_dict() for b in sorted(index.blocks.values(), key=lambda b: b.block_id)]


def list_pipelines(workspace: str = ".") -> list[str]:
    """Workspace-relative paths of saved pipelines."""
    ws = _workspace(workspace)
    root = ws / ".forge" / "pipelines"
    if not root.is_dir():
        return []
    return sorted(p.relative_to(ws).as_posix() for p in root.glob("*.forge"))


def read_pipeline(workspace: str, path: str) -> dict[str, Any]:
    """The pipeline document (nodes, edges, comments) as JSON."""
    ws = _workspace(workspace)
    return PipelineDoc.load(_pipeline_path(ws, path)).to_dict()


def write_pipeline(workspace: str, path: str, doc: dict[str, Any]) -> dict[str, Any]:
    """Validate and save a pipeline document; returns its node states so the
    caller immediately sees problems and staleness."""
    ws = _workspace(workspace)
    pipeline = PipelineDoc.from_dict(doc)  # validates the schema
    target = _pipeline_path(ws, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(target)
    return pipeline_state(workspace, path)


def pipeline_state(workspace: str, path: str) -> dict[str, Any]:
    """Per-node history hash, staleness, and problems for a saved pipeline."""
    ws = _workspace(workspace)
    doc = PipelineDoc.load(_pipeline_path(ws, path))
    index, _ = scan_workspace(ws)
    store = CheckpointStore(ws)
    states = compute_states(doc, index, env_fingerprint(ws), store.exists)
    return {
        "path": _pipeline_path(ws, path).relative_to(ws).as_posix(),
        "nodes": {nid: state.to_dict() for nid, state in states.items()},
    }


def run_pipeline(workspace: str, path: str, target: str | None = None, timeout_seconds: float = 600) -> dict[str, Any]:
    """Run a saved pipeline (optionally only up to `target` node) in the
    standard worker subprocess; blocks until done. Returns the event stream
    summary and terminal status."""
    ws = _workspace(workspace)
    doc = PipelineDoc.load(_pipeline_path(ws, path))
    index, _ = scan_workspace(ws)
    store = CheckpointStore(ws)
    store.ensure_layout()
    states = compute_states(doc, index, env_fingerprint(ws), store.exists)
    blocks = {
        b.block_id: {
            "module": b.module,
            "qualname": b.qualname,
            "label": b.label,
            "outputs": b.outputs,
            "source_hash": b.source_hash,
            "optional_params": [p.name for p in b.params if p.optional and not p.has_default],
        }
        for b in index.blocks.values()
    }
    spec = {
        "workspace": str(ws),
        "doc": doc.to_dict(),
        "blocks": blocks,
        "states": {nid: s.to_dict() for nid, s in states.items()},
        "pickle_enabled": False,
        "target": target,
    }
    runs_dir = ws / ".forge" / "cache" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = runs_dir / "mcp_run.json"
    spec_path.write_text(json.dumps(spec, default=repr), encoding="utf-8")

    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "starforge.kernel.worker", str(spec_path)],
            cwd=str(ws),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_seconds,
        )
    finally:
        spec_path.unlink(missing_ok=True)

    events = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    status = next(
        (e.get("status") for e in reversed(events) if e.get("event") == "run_finished"), "failed"
    )
    summary = {
        "completed": [e["node"] for e in events if e.get("event") == "node_completed"],
        "skipped": [e["node"] for e in events if e.get("event") == "node_skipped"],
        "blocked": [e["node"] for e in events if e.get("event") == "node_blocked"],
        "failed": {
            e["node"]: e.get("traceback", "").strip().splitlines()[-1:]
            for e in events
            if e.get("event") == "node_failed"
        },
    }
    return {"status": status, **summary, "states": pipeline_state(workspace, path)["nodes"]}


def inspect_node(workspace: str, path: str, node_id: str) -> dict[str, Any]:
    """Checkpoint provenance for one node: outputs with previews, figures,
    timing — or its staleness/problems when no checkpoint exists."""
    ws = _workspace(workspace)
    state = pipeline_state(workspace, path)["nodes"].get(node_id)
    if state is None:
        raise ValueError(f"node '{node_id}' not found in {path}")
    store = CheckpointStore(ws)
    history_hash = state.get("history_hash")
    if not history_hash or not store.exists(history_hash):
        return {"node": node_id, "state": state, "checkpoint": None}
    return {"node": node_id, "state": state, "checkpoint": store.read_provenance(history_hash)}


TOOL_NAMES = (
    "starforge_list_blocks",
    "starforge_list_pipelines",
    "starforge_read_pipeline",
    "starforge_write_pipeline",
    "starforge_pipeline_state",
    "starforge_run_pipeline",
    "starforge_inspect_node",
)


def main() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        return
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The *Forge MCP server needs the 'mcp' package: pip install starforge-kernel[mcp]"
        ) from exc

    default_workspace = sys.argv[1] if len(sys.argv) > 1 else "."

    # stdout is the MCP protocol channel; everything human goes to stderr.
    # Without this banner, running the server by hand looks like a hang.
    print(
        "\n".join(
            [
                "*Forge MCP server",
                f"  workspace : {Path(default_workspace).resolve()}",
                "  transport : stdio (waiting for an MCP client on stdin)",
                f"  tools     : {', '.join(TOOL_NAMES)}",
                "",
                "Register with an agent via \"*Forge: Set Up MCP for Agents\" in VS Code, or:",
                '  { "command": "python", "args": ["-m", "starforge.mcp"] }   (cwd = repo root)',
                "",
                "Ctrl+C to exit.",
            ]
        ),
        file=sys.stderr,
        flush=True,
    )
    server = FastMCP(
        "starforge",
        instructions=(
            "Author and run *Forge pipelines in this workspace. Blocks are @block-decorated "
            "functions in the repo (see list_blocks for ids/params/outputs); the decorator is "
            "behavior-neutral, so blocks remain plain callables you can invoke directly when "
            "testing logic outside a pipeline. New @block functions are picked up automatically "
            "on save — no registration step. Pipelines are .forge JSON documents shaped like: "
            '{"schema": "starforge/1", "name": "demo", '
            '"nodes": [{"id": "n1", "block": "module:function", "params": {"x": 1}, '
            '"position": {"x": 80, "y": 80}}], '
            '"edges": [{"id": "e1", "source": "n1", "source_output": "output", '
            '"target": "n2", "target_param": "data"}]}. '
            "Canvas annotation boxes live in a top-level \"comments\" array, each shaped "
            '{"id": "c1", "title": "...", "description": "...", "position": {"x": 0, "y": 0}, '
            '"width": 280, "height": 150, "color": "#6366f1"} — these exact field names; '
            "comments never affect execution. "
            "Edges target parameter NAMES; unwired params use doc literals or signature "
            "defaults; `T | None` params with no default receive None. write_pipeline "
            "validates and returns per-node staleness; run_pipeline executes only stale nodes "
            "(pass `target` to run one node's ancestor cone); inspect_node returns checkpoint "
            f"previews. Default workspace: {Path(default_workspace).resolve()}"
        ),
    )

    def _ws(workspace: str | None) -> str:
        return workspace or default_workspace

    @server.tool()
    def starforge_list_blocks(workspace: str | None = None) -> list[dict[str, Any]]:
        """List every available block with params, outputs, types, and docs."""
        return list_blocks(_ws(workspace))

    @server.tool()
    def starforge_list_pipelines(workspace: str | None = None) -> list[str]:
        """List saved .forge pipelines in the workspace."""
        return list_pipelines(_ws(workspace))

    @server.tool()
    def starforge_read_pipeline(path: str, workspace: str | None = None) -> dict[str, Any]:
        """Read a pipeline document (nodes, edges, comments)."""
        return read_pipeline(_ws(workspace), path)

    @server.tool()
    def starforge_write_pipeline(path: str, doc: dict[str, Any], workspace: str | None = None) -> dict[str, Any]:
        """Validate and save a pipeline document; returns per-node states (problems, staleness)."""
        return write_pipeline(_ws(workspace), path, doc)

    @server.tool()
    def starforge_pipeline_state(path: str, workspace: str | None = None) -> dict[str, Any]:
        """Per-node staleness and problems without running anything."""
        return pipeline_state(_ws(workspace), path)

    @server.tool()
    def starforge_run_pipeline(
        path: str, target: str | None = None, workspace: str | None = None
    ) -> dict[str, Any]:
        """Run a saved pipeline (stale nodes only; `target` limits to one node's
        ancestor cone). Blocks until the run finishes."""
        return run_pipeline(_ws(workspace), path, target=target)

    @server.tool()
    def starforge_inspect_node(path: str, node_id: str, workspace: str | None = None) -> dict[str, Any]:
        """Checkpoint provenance for a node: output previews, figures, timing."""
        return inspect_node(_ws(workspace), path, node_id)

    server.run()


if __name__ == "__main__":
    main()
