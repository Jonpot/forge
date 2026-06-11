"""Exercises the real seam: kernel subprocess over stdio NDJSON, which itself
spawns a worker subprocess for the run — exactly what the VS Code extension
drives."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import pytest

from conftest import three_node_doc

SRC = str(Path(__file__).resolve().parents[1] / "src")


class KernelClient:
    def __init__(self, cwd: str) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "starforge.kernel"],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._frames: queue.Queue = queue.Queue()
        self._next_id = 0
        self.notifications: list[dict] = []
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                self._frames.put(json.loads(raw.decode("utf-8")))
            except json.JSONDecodeError:
                pass
        self._frames.put(None)  # EOF sentinel

    def request(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        self._next_id += 1
        request_id = self._next_id
        frame = json.dumps({"id": request_id, "method": method, "params": params or {}}) + "\n"
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame.encode("utf-8"))
        self.proc.stdin.flush()
        while True:
            message = self._frames.get(timeout=timeout)
            if message is None:
                raise RuntimeError(f"kernel died during '{method}': {self.stderr()}")
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']['message']}")
                return message["result"]
            if "method" in message:
                self.notifications.append(message)

    def wait_for_event(self, predicate, timeout: float = 90) -> dict:
        for note in self.notifications:
            if note["method"] == "run/event" and predicate(note["params"]):
                return note["params"]
        while True:
            message = self._frames.get(timeout=timeout)
            if message is None:
                raise RuntimeError(f"kernel died while waiting for event: {self.stderr()}")
            if "method" in message:
                self.notifications.append(message)
                if message["method"] == "run/event" and predicate(message["params"]):
                    return message["params"]

    def stderr(self) -> str:
        if self.proc.poll() is None:
            return "(kernel still running)"
        assert self.proc.stderr is not None
        return self.proc.stderr.read().decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            self.request("shutdown", timeout=5)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture
def kernel(workspace, tmp_path):
    client = KernelClient(cwd=str(tmp_path))
    try:
        init = client.request("initialize", {"workspace": str(workspace.root), "settings": {}})
        assert init["kernel_version"]
        yield client
    finally:
        client.close()


def test_full_protocol_loop(workspace, kernel):
    palette = kernel.request("index/scan")
    assert {b["block_id"] for b in palette["blocks"]} == {
        "builtin:constant",
        "pipeline_lib.sources:make_numbers",
        "pipeline_lib.transforms:scale_values",
        "pipeline_lib.transforms:summarize",
    }
    assert palette["errors"] == {}

    doc = three_node_doc(n=4, factor=3.0)
    hashes = kernel.request("pipeline/hashes", {"doc": doc})
    assert all(node["stale"] for node in hashes["nodes"].values())
    assert all(node["problems"] == [] for node in hashes["nodes"].values())

    started = kernel.request("run/start", {"doc": doc})
    run_id = started["run_id"]
    finished = kernel.wait_for_event(
        lambda p: p.get("run_id") == run_id and p.get("event") == "run_finished"
    )
    assert finished["status"] == "completed"

    completed = [
        n["params"]["node"]
        for n in kernel.notifications
        if n["method"] == "run/event" and n["params"].get("event") == "node_completed"
    ]
    assert completed == ["n1", "n2", "n3"]

    # The same doc is now fully fresh through the protocol as well.
    hashes = kernel.request("pipeline/hashes", {"doc": doc})
    assert not any(node["stale"] for node in hashes["nodes"].values())

    manifest = kernel.request(
        "results/manifest", {"history_hash": hashes["nodes"]["n3"]["history_hash"]}
    )
    assert {entry["name"] for entry in manifest["outputs"]} == {"total", "count"}
    assert manifest["block_id"] == "pipeline_lib.transforms:summarize"


def test_results_figures_roundtrip_through_real_worker(workspace, kernel):
    pytest.importorskip("matplotlib")
    workspace.write(
        "plots.py",
        "import matplotlib.pyplot as plt\n"
        "from starforge import block\n\n"
        "@block\n"
        "def quick_plot(n: int = 3) -> int:\n"
        "    plt.plot(range(n))\n"
        "    plt.show()\n"
        "    return n\n",
    )
    kernel.request("index/scan")
    doc = {
        "schema": "starforge/1",
        "name": "figs",
        "nodes": [{"id": "q1", "block": "plots:quick_plot", "params": {}}],
        "edges": [],
    }
    started = kernel.request("run/start", {"doc": doc})
    finished = kernel.wait_for_event(
        lambda p: p.get("run_id") == started["run_id"] and p.get("event") == "run_finished"
    )
    assert finished["status"] == "completed"

    hashes = kernel.request("pipeline/hashes", {"doc": doc})
    history_hash = hashes["nodes"]["q1"]["history_hash"]
    figures = kernel.request("results/figures", {"history_hashes": [history_hash]})
    checkpoint = figures["checkpoints"][history_hash]
    [artifact] = checkpoint["artifacts"]
    assert artifact["kind"] == "image"
    png = Path(str(workspace.root)) / checkpoint["dir"] / "outputs" / artifact["file"]
    assert png.stat().st_size > 0


def test_user_prints_become_logs_not_protocol_corruption(workspace, kernel):
    workspace.write(
        "noisy.py",
        "from starforge import block\n\n"
        "@block\n"
        "def shout(n: int = 1) -> int:\n"
        "    print('LOUD NOISES')\n"
        "    return n\n",
    )
    kernel.request("index/scan")
    doc = {
        "schema": "starforge/1",
        "name": "noisy",
        "nodes": [{"id": "s1", "block": "noisy:shout", "params": {}}],
        "edges": [],
    }
    started = kernel.request("run/start", {"doc": doc})
    finished = kernel.wait_for_event(
        lambda p: p.get("run_id") == started["run_id"] and p.get("event") == "run_finished"
    )
    assert finished["status"] == "completed"
    logs = [n for n in kernel.notifications if n["method"] == "log"]
    assert any("LOUD NOISES" in n["params"]["text"] for n in logs)
