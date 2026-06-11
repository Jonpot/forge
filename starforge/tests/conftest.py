from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pytest

# In-process runner tests execute user code in this interpreter; force the
# headless backend before anything can import matplotlib (the real worker
# does the same in its entrypoint).
os.environ.setdefault("MPLBACKEND", "Agg")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SOURCES_PY = '''
from starforge import block


@block(category="IO")
def make_numbers(n: int = 5) -> dict:
    """Produce the first n positive integers."""
    return {"values": list(range(1, n + 1))}
'''

HELPERS_PY = '''
def scale_factor():
    return 1
'''

TRANSFORMS_PY = '''
from starforge import block

from pipeline_lib.helpers import scale_factor


@block
def scale_values(data: dict, factor: float = 2.0) -> dict:
    """Multiply every value by factor (and the helper's scale factor)."""
    return {"values": [v * factor * scale_factor() for v in data["values"]]}


@block(outputs=("total", "count"))
def summarize(data: dict) -> tuple[float, int]:
    return sum(data["values"]), len(data["values"])
'''


class Workspace:
    """A fake user repo with @block-decorated functions."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, relpath: str, text: str) -> None:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        # Guarantee a fresh mtime so the indexer's fast path can't reuse a
        # stale cache entry when tests rewrite files back-to-back.
        bumped = time.time_ns() + int(1e9)
        os.utime(path, ns=(bumped, bumped))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "repo")
    ws.write("pipeline_lib/__init__.py", "")
    ws.write("pipeline_lib/sources.py", SOURCES_PY)
    ws.write("pipeline_lib/helpers.py", HELPERS_PY)
    ws.write("pipeline_lib/transforms.py", TRANSFORMS_PY)
    return ws


def three_node_doc(n: int = 4, factor: float = 3.0) -> dict:
    """make_numbers -> scale_values -> summarize, as a .forge document dict."""
    return {
        "schema": "starforge/1",
        "name": "test pipeline",
        "nodes": [
            {"id": "n1", "block": "pipeline_lib.sources:make_numbers", "params": {"n": n}},
            {"id": "n2", "block": "pipeline_lib.transforms:scale_values", "params": {"factor": factor}},
            {"id": "n3", "block": "pipeline_lib.transforms:summarize", "params": {}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "source_output": "output", "target": "n2", "target_param": "data"},
            {"id": "e2", "source": "n2", "source_output": "output", "target": "n3", "target_param": "data"},
        ],
    }
