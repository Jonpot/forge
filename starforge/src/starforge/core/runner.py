"""Pipeline execution. Runs INSIDE the per-run worker process — the kernel
never imports user code (DESIGN.md §3, process-per-run).

Planning rules:
- stale nodes (no checkpoint for their history hash) execute;
- a fresh node is pulled back into execution if an executing descendant needs
  one of its outputs and that output was ephemeral (never persisted) — this
  cascades upward until every needed value is either loadable or recomputed;
- nodes with hashing problems, and every node downstream of a problem or a
  failure, are blocked (independent branches keep running).
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import importlib
import sys
import time
import traceback
from typing import Any, Callable

import starforge
from starforge.core import figures as figmod
from starforge.core.checkpoints import CheckpointStore
from starforge.core.provenance import BUILTIN_PREFIX, toposort
from starforge.core.serializers import EphemeralValueError
from starforge.core.spec import PipelineDoc, Node

#: Minimum seconds between forwarded progress events per node — blocks may
#: call progress() in tight loops; the canvas needs ~12fps at most.
PROGRESS_THROTTLE_S = 0.08


def _progress_hook(node_id: str, emit: Emit) -> Callable[[Any, Any, Any], None]:
    last = {"at": 0.0}

    def hook(current: Any, total: Any, label: Any) -> None:
        now = time.monotonic()
        final = total is not None and current == total
        if not final and now - last["at"] < PROGRESS_THROTTLE_S:
            return
        last["at"] = now
        event: dict[str, Any] = {"event": "node_progress", "node": node_id}
        if current is not None:
            event["current"] = current
        if total is not None:
            event["total"] = total
        if label is not None:
            event["label"] = str(label)
        if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total:
            event["percent"] = max(0.0, min(1.0, current / total))
        emit(event)

    return hook

Emit = Callable[[dict[str, Any]], None]


@dataclass
class _NodePlanInfo:
    history_hash: str | None
    problems: list[str]


def plan_execution(
    doc: PipelineDoc,
    states: dict[str, dict[str, Any]],
    store: CheckpointStore,
) -> tuple[list[str], set[str], set[str], set[str]]:
    """Returns (topo_order, execute, reuse, blocked)."""
    order, cyclic = toposort(doc)
    blocked: set[str] = set(cyclic)
    execute: set[str] = set()

    for nid in order:
        state = states.get(nid, {})
        if state.get("problems") or state.get("history_hash") is None:
            blocked.add(nid)
        elif state.get("stale", True):
            execute.add(nid)

    # Block everything downstream of a blocked node.
    for nid in order:
        if nid in blocked:
            continue
        if any(edge.source in blocked for edge in doc.in_edges(nid)):
            blocked.add(nid)
            execute.discard(nid)

    # Ephemeral cascade: pull fresh parents back in when an executing child
    # needs an output that was never persisted.
    changed = True
    while changed:
        changed = False
        for edge in doc.edges:
            if edge.target not in execute or edge.source in execute or edge.source in blocked:
                continue
            parent_hash = states.get(edge.source, {}).get("history_hash")
            if parent_hash is None:
                continue
            if store.is_ephemeral(parent_hash, edge.source_output):
                execute.add(edge.source)
                changed = True

    reuse = {nid for nid in order if nid not in execute and nid not in blocked}
    return order, execute, reuse, blocked


def _run_builtin(node: Node) -> dict[str, Any]:
    if node.block == "builtin:constant":
        return {"output": node.params.get("value")}
    raise ValueError(f"unknown builtin '{node.block}'")


def _import_function(module: str, qualname: str) -> Callable[..., Any]:
    mod = importlib.import_module(module)
    fn = getattr(mod, qualname, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"function '{qualname}' not found in module '{module}'")
    return fn


def _map_outputs(result: Any, output_names: list[str], block_label: str) -> dict[str, Any]:
    if len(output_names) == 1:
        return {output_names[0]: result}
    if not isinstance(result, (tuple, list)) or len(result) != len(output_names):
        raise TypeError(
            f"'{block_label}' declares outputs {output_names} but returned "
            f"{type(result).__name__} (expected a {len(output_names)}-tuple)"
        )
    return dict(zip(output_names, result))


def run_pipeline(
    doc: PipelineDoc,
    blocks: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    store: CheckpointStore,
    emit: Emit,
    pickle_enabled: bool = False,
) -> str:
    """Execute the plan; returns terminal status: 'completed' or 'failed'."""
    store.ensure_layout()
    order, execute, reuse, blocked = plan_execution(doc, states, store)
    emit(
        {
            "event": "run_plan",
            "execute": sorted(execute),
            "reuse": sorted(reuse),
            "blocked": sorted(blocked),
        }
    )

    # In-memory values for this run: (node_id, output_name) -> value.
    memory: dict[tuple[str, str], Any] = {}
    failed: set[str] = set()
    any_failed = False

    for nid in order:
        state = states.get(nid, {})
        node_hash = state.get("history_hash")

        if nid in blocked or any(e.source in failed or e.source in blocked for e in doc.in_edges(nid)):
            blocked.add(nid)
            emit({"event": "node_blocked", "node": nid, "problems": state.get("problems", [])})
            continue

        if nid in reuse:
            store.touch(node_hash)  # LRU signal for checkpoint GC
            emit({"event": "node_skipped", "node": nid, "history_hash": node_hash})
            continue

        node = doc.node(nid)
        emit({"event": "node_started", "node": nid, "block": node.block})
        started = time.time()
        try:
            connected: dict[str, dict[str, str]] = {}
            side_figures: list[Any] = []
            if node.block.startswith(BUILTIN_PREFIX):
                outputs = _run_builtin(node)
                label = node.block.removeprefix(BUILTIN_PREFIX).title()
                source_hash = node.block
            else:
                info = blocks[node.block]
                kwargs: dict[str, Any] = {}
                for edge in doc.in_edges(nid):
                    key = (edge.source, edge.source_output)
                    if key not in memory:
                        parent_hash = states[edge.source]["history_hash"]
                        try:
                            memory[key] = store.load_output(parent_hash, edge.source_output)
                        except EphemeralValueError:
                            # plan_execution should have prevented this; surface loudly.
                            raise RuntimeError(
                                f"input '{edge.target_param}' of node '{nid}' is ephemeral and its "
                                f"producer was not scheduled — planner bug, please report"
                            )
                    kwargs[edge.target_param] = memory[key]
                    connected[edge.target_param] = {
                        "node": edge.source,
                        "output": edge.source_output,
                        "history_hash": states[edge.source]["history_hash"],
                    }
                for name, value in node.params.items():
                    if name not in kwargs:
                        kwargs[name] = value
                # `T | None` params with no signature default are optional by
                # *Forge convention: inject None when nothing else fills them.
                for name in info.get("optional_params", ()):
                    kwargs.setdefault(name, None)

                # capture() wraps the import too: module-level figure code
                # and figures created/shown inside the call are both swept.
                starforge._progress_hook = _progress_hook(nid, emit)
                try:
                    with figmod.capture() as captured:
                        fn = _import_function(info["module"], info["qualname"])
                        result = fn(**kwargs)
                finally:
                    starforge._progress_hook = None
                outputs = _map_outputs(result, list(info["outputs"]), info.get("label", node.block))
                side_figures = [
                    fig for fig in captured.all_objects()
                    if all(fig is not value for value in outputs.values())
                ]
                label = info.get("label", node.block)
                source_hash = info.get("source_hash")

            duration = time.time() - started
            manifest = store.write(
                node_hash,
                {
                    "block_id": node.block,
                    "label": label,
                    "source_hash": source_hash,
                    "params": {k: v for k, v in node.params.items() if k not in connected},
                    "inputs": connected,
                    "duration_seconds": round(duration, 6),
                    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
                outputs,
                pickle_enabled=pickle_enabled,
                side_figures=side_figures,
            )
            figmod.close_figures(side_figures)
            for name, value in outputs.items():
                memory[(nid, name)] = value
            emit(
                {
                    "event": "node_completed",
                    "node": nid,
                    "history_hash": node_hash,
                    "duration_seconds": round(duration, 6),
                    "outputs": manifest,
                }
            )
        except Exception:
            any_failed = True
            failed.add(nid)
            emit(
                {
                    "event": "node_failed",
                    "node": nid,
                    "duration_seconds": round(time.time() - started, 6),
                    "traceback": traceback.format_exc(),
                }
            )

    status = "failed" if any_failed else "completed"
    emit({"event": "run_finished", "status": status})
    return status
