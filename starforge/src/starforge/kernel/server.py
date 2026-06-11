"""The *Forge kernel: NDJSON JSON-RPC over stdio (DESIGN.md §9).

Spawned by the VS Code extension inside the workspace venv as
``python -m starforge.kernel``. Stdlib-only by design: pandas and friends
exist only inside run workers, so an idle kernel stays small and an idle
workspace costs nothing at all (the extension kills us when unused).

Requests:  {"id": .., "method": "..", "params": {..}}
Responses: {"id": .., "result": {..}} | {"id": .., "error": {"message": ..}}
Notifications (kernel → client): {"method": "run/event", "params": {..}}
                                  {"method": "log",       "params": {..}}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, BinaryIO
import uuid

import starforge
from starforge.core.checkpoints import CheckpointStore
from starforge.core.provenance import BUILTIN_PREFIX, compute_states, env_fingerprint
from starforge.core.spec import PipelineDoc
from starforge.index import WorkspaceIndex, scan_workspace

#: Doc-native nodes, served to the palette alongside discovered functions.
BUILTIN_PALETTE = [
    {
        "block_id": "builtin:constant",
        "module": "builtin",
        "qualname": "constant",
        "file": "",
        "lineno": 0,
        "label": "Constant",
        "category": "Built-in",
        "params": [
            {
                "name": "value",
                "annotation": None,
                "default_repr": "null",
                "has_default": True,
                "keyword_only": False,
            }
        ],
        "outputs": ["output"],
        "returns": None,
        "doc": "Inject a literal JSON value (number, string, list, object) as a source node.",
        "source_hash": "builtin:constant",
    }
]


class Kernel:
    def __init__(self, stdin: BinaryIO, stdout: BinaryIO) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._write_lock = threading.Lock()
        self.workspace: Path | None = None
        self.settings: dict[str, Any] = {}
        self.index: WorkspaceIndex | None = None
        self._scan_cache: dict[str, Any] | None = None
        self._runs: dict[str, subprocess.Popen] = {}

    # ---------------------------------------------------------------- wire

    def _send(self, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload, default=repr) + "\n").encode("utf-8")
        with self._write_lock:
            self._stdout.write(data)
            self._stdout.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def serve_forever(self) -> None:
        for raw in self._stdin:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                self._send({"id": None, "error": {"message": f"bad frame: {exc}"}})
                continue
            request_id = request.get("id")
            method = str(request.get("method", ""))
            params = request.get("params") or {}
            if method == "shutdown":
                self._send({"id": request_id, "result": {}})
                break
            handler = getattr(self, "rpc_" + method.replace("/", "_"), None)
            if handler is None:
                self._send({"id": request_id, "error": {"message": f"unknown method '{method}'"}})
                continue
            try:
                self._send({"id": request_id, "result": handler(params)})
            except Exception as exc:  # noqa: BLE001 — every failure must answer the request
                self._send({"id": request_id, "error": {"message": f"{type(exc).__name__}: {exc}"}})
        self._terminate_runs()

    # ------------------------------------------------------------- helpers

    def _require_workspace(self) -> Path:
        if self.workspace is None:
            raise RuntimeError("kernel not initialized — call 'initialize' first")
        return self.workspace

    def _cache_path(self) -> Path:
        return self._require_workspace() / ".forge" / "cache" / "index.json"

    def _store(self) -> CheckpointStore:
        return CheckpointStore(self._require_workspace())

    def _ensure_index(self) -> WorkspaceIndex:
        if self.index is None:
            self.rpc_index_scan({})
        assert self.index is not None
        return self.index

    def _palette(self, index: WorkspaceIndex) -> dict[str, Any]:
        discovered = [b.to_dict() for b in sorted(index.blocks.values(), key=lambda b: b.block_id)]
        return {
            "blocks": list(BUILTIN_PALETTE) + discovered,
            "errors": index.errors(),
        }

    def _terminate_runs(self) -> None:
        for proc in self._runs.values():
            if proc.poll() is None:
                proc.terminate()

    # ------------------------------------------------------------- methods

    def rpc_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        self.workspace = Path(params["workspace"]).resolve()
        self.settings = dict(params.get("settings") or {})
        store = self._store()
        store.ensure_layout()
        cache_path = self._cache_path()
        if cache_path.is_file():
            try:
                self._scan_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._scan_cache = None
        return {
            "kernel_version": starforge.__version__,
            "python": sys.version.split()[0],
            "env_fingerprint": env_fingerprint(self.workspace),
        }

    def rpc_index_scan(self, params: dict[str, Any]) -> dict[str, Any]:
        workspace = self._require_workspace()
        index, cache = scan_workspace(workspace, self._scan_cache)
        self.index, self._scan_cache = index, cache
        cache_path = self._cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass  # cache is an optimization; never fail a scan over it
        return self._palette(index)

    def rpc_pipeline_hashes(self, params: dict[str, Any]) -> dict[str, Any]:
        workspace = self._require_workspace()
        doc = PipelineDoc.from_dict(params["doc"])
        index = self._ensure_index()
        states = compute_states(doc, index, env_fingerprint(workspace), self._store().exists)
        return {"nodes": {nid: state.to_dict() for nid, state in states.items()}}

    def rpc_run_start(self, params: dict[str, Any]) -> dict[str, Any]:
        workspace = self._require_workspace()
        doc = PipelineDoc.from_dict(params["doc"])
        index = self._ensure_index()
        states = compute_states(doc, index, env_fingerprint(workspace), self._store().exists)
        blocks = index.blocks
        referenced = {
            node.block: {
                "module": blocks[node.block].module,
                "qualname": blocks[node.block].qualname,
                "label": blocks[node.block].label,
                "outputs": blocks[node.block].outputs,
                "source_hash": blocks[node.block].source_hash,
                "optional_params": [
                    p.name for p in blocks[node.block].params if p.optional and not p.has_default
                ],
            }
            for node in doc.nodes
            if node.block in blocks and not node.block.startswith(BUILTIN_PREFIX)
        }
        run_id = params.get("run_id") or uuid.uuid4().hex[:12]
        spec = {
            "workspace": str(workspace),
            "doc": doc.to_dict(),
            "blocks": referenced,
            "states": {nid: state.to_dict() for nid, state in states.items()},
            "pickle_enabled": bool(self.settings.get("pickle_enabled", False)),
        }
        runs_dir = workspace / ".forge" / "cache" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        spec_path = runs_dir / f"{run_id}.json"
        spec_path.write_text(json.dumps(spec, default=repr), encoding="utf-8")

        env = dict(os.environ)
        package_root = str(Path(starforge.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [sys.executable, "-m", "starforge.kernel.worker", str(spec_path)],
            cwd=str(workspace),
            env=env,
            # DEVNULL is load-bearing: inheriting our stdin (the protocol
            # pipe) deadlocks the child's interpreter bootstrap on Windows
            # while our main thread is blocked reading that same handle —
            # and user code calling input() must never eat protocol frames.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._runs[run_id] = proc
        threading.Thread(target=self._pump_events, args=(run_id, proc), daemon=True).start()
        threading.Thread(target=self._pump_logs, args=(run_id, proc), daemon=True).start()
        return {"run_id": run_id}

    def _pump_events(self, run_id: str, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        saw_terminal = False
        for raw in proc.stdout:
            try:
                event = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            saw_terminal = saw_terminal or event.get("event") == "run_finished"
            self.notify("run/event", {"run_id": run_id, **event})
        code = proc.wait()
        if not saw_terminal:
            # rpc_run_cancel pops the run before terminating it; a missing
            # entry with no terminal event means the worker was killed by us.
            status = "cancelled" if run_id not in self._runs else "failed"
            self.notify(
                "run/event",
                {"run_id": run_id, "event": "run_finished", "status": status, "exit_code": code},
            )
        self._runs.pop(run_id, None)

    def _pump_logs(self, run_id: str, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            text = raw.decode("utf-8", errors="replace").rstrip()
            if text:
                self.notify("log", {"run_id": run_id, "text": text})

    def rpc_run_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        run_id = params["run_id"]
        proc = self._runs.pop(run_id, None)
        if proc is None or proc.poll() is not None:
            return {"cancelled": False, "reason": "run not active"}
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        return {"cancelled": True}

    def rpc_results_manifest(self, params: dict[str, Any]) -> dict[str, Any]:
        store = self._store()
        history_hash = params["history_hash"]
        if not store.exists(history_hash):
            raise FileNotFoundError(f"no checkpoint for hash {history_hash[:16]}")
        return store.read_provenance(history_hash)

    def rpc_results_figures(self, params: dict[str, Any]) -> dict[str, Any]:
        """Batch lookup of display artifacts for node thumbnails: for each
        existing checkpoint, its dir plus side-figures and output artifacts."""
        store = self._store()
        results: dict[str, Any] = {}
        for history_hash in params.get("history_hashes", []):
            if not store.exists(history_hash):
                continue
            try:
                provenance = store.read_provenance(history_hash)
            except (OSError, json.JSONDecodeError):
                continue
            artifacts = list(provenance.get("figures", []))
            artifacts.extend(
                entry["artifact"]
                for entry in provenance.get("outputs", [])
                if entry.get("artifact")
            )
            if artifacts:
                results[history_hash] = {"dir": provenance.get("dir"), "artifacts": artifacts}
        return {"checkpoints": results}


def main() -> None:
    Kernel(sys.stdin.buffer, sys.stdout.buffer).serve_forever()


if __name__ == "__main__":
    main()
