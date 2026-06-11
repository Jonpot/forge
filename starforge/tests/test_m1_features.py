"""Progress streaming, checkpoint GC, comments round-trip, tier selection."""

from __future__ import annotations

import os
import time

import starforge
from starforge.core.checkpoints import CheckpointStore
from starforge.core.provenance import compute_states, env_fingerprint
from starforge.core.spec import PipelineDoc
from starforge.index import scan_workspace
from conftest import three_node_doc
from test_runner_end_to_end import execute, names

NEVER = lambda h: False  # noqa: E731


# ------------------------------------------------------------------ progress


def test_progress_is_noop_outside_forge():
    assert starforge._progress_hook is None
    starforge.progress(1, 10, "should do nothing")  # must not raise


def test_progress_events_flow_with_throttle_and_final(workspace):
    workspace.write(
        "pipeline_lib/progressive.py",
        "import starforge\n"
        "from starforge import block\n\n"
        "@block\n"
        "def chunked(n: int = 50) -> int:\n"
        "    for i in range(n):\n"
        "        starforge.progress(i + 1, n, 'crunching')\n"
        "    return n\n",
    )
    doc = {
        "schema": "starforge/1",
        "name": "prog",
        "nodes": [{"id": "p1", "block": "pipeline_lib.progressive:chunked", "params": {}}],
        "edges": [],
    }
    events, _, _ = execute(workspace, doc)
    progress = [e for e in events if e["event"] == "node_progress"]
    assert progress, "expected progress events"
    # Throttled: a 50-iteration tight loop must not emit 50 events…
    assert len(progress) < 50
    # …but the terminal 100% event always lands.
    final = progress[-1]
    assert final["current"] == 50 and final["total"] == 50
    assert final["percent"] == 1.0
    assert final["label"] == "crunching"
    # The hook is uninstalled after the block call.
    assert starforge._progress_hook is None
    assert names(events, "node_completed") == ["p1"]


# ------------------------------------------------------------------ gc


def _fake_checkpoint(store: CheckpointStore, name: str, size: int, age_seconds: float) -> None:
    directory = store.base / name
    (directory / "outputs").mkdir(parents=True)
    (directory / "outputs" / "data.bin").write_bytes(b"x" * size)
    (directory / "provenance.json").write_text("{}", encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(directory, (stamp, stamp))


def test_gc_evicts_oldest_until_under_budget(tmp_path):
    store = CheckpointStore(tmp_path)
    store.ensure_layout()
    _fake_checkpoint(store, "a" * 32, size=1000, age_seconds=300)  # oldest
    _fake_checkpoint(store, "b" * 32, size=1000, age_seconds=200)
    _fake_checkpoint(store, "c" * 32, size=1000, age_seconds=100)  # newest

    stats = store.gc(max_bytes=2100)
    assert stats["deleted"] == 1
    assert not (store.base / ("a" * 32)).exists()
    assert (store.base / ("c" * 32)).exists()

    stats = store.gc(max_bytes=0)
    assert stats["deleted"] == 2
    assert stats["remaining_bytes"] == 0


def test_gc_within_budget_deletes_nothing(tmp_path):
    store = CheckpointStore(tmp_path)
    store.ensure_layout()
    _fake_checkpoint(store, "a" * 32, size=100, age_seconds=10)
    assert store.gc(max_bytes=10_000_000) == {"freed_bytes": 0, "deleted": 0, "remaining_bytes": 100 + 2}


def test_clean_run_specs(tmp_path):
    store = CheckpointStore(tmp_path)
    store.ensure_layout()
    runs = store.forge_dir / "cache" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    old = runs / "old.json"
    old.write_text("{}")
    stamp = time.time() - 100000
    os.utime(old, (stamp, stamp))
    fresh = runs / "fresh.json"
    fresh.write_text("{}")
    store.clean_run_specs(max_age_seconds=86400)
    assert not old.exists()
    assert fresh.exists()


# ------------------------------------------------------------------ comments


def test_comments_round_trip_and_never_affect_hashes(workspace):
    doc_dict = three_node_doc()
    doc_dict["comments"] = [
        {
            "id": "cm1",
            "title": "QC",
            "description": "quality control steps",
            "position": {"x": 50, "y": 40},
            "width": 400,
            "height": 220,
            "color": "#6366f1",
        }
    ]
    doc = PipelineDoc.from_dict(doc_dict)
    assert doc.to_dict()["comments"] == doc_dict["comments"]

    index, _ = scan_workspace(workspace.root)
    env = env_fingerprint(workspace.root)
    with_comments = compute_states(doc, index, env, NEVER)
    plain = compute_states(PipelineDoc.from_dict(three_node_doc()), index, env, NEVER)
    for nid in ("n1", "n2", "n3"):
        assert with_comments[nid].history_hash == plain[nid].history_hash


# ------------------------------------------------------------------ tiers


def test_tier_zero_ignores_helper_edits_but_sees_body_edits(workspace):
    index1, cache = scan_workspace(workspace.root)
    env = env_fingerprint(workspace.root)
    doc = PipelineDoc.from_dict(three_node_doc())
    before = compute_states(doc, index1, env, NEVER, tier="T0")

    workspace.write("pipeline_lib/helpers.py", "def scale_factor():\n    return 99\n")
    index2, cache = scan_workspace(workspace.root, cache)
    after_helper = compute_states(doc, index2, env, NEVER, tier="T0")
    assert after_helper["n2"].history_hash == before["n2"].history_hash  # T0 blind spot

    original = (workspace.root / "pipeline_lib/transforms.py").read_text(encoding="utf-8")
    workspace.write("pipeline_lib/transforms.py", original.replace("* factor", "* factor * 3"))
    index3, _ = scan_workspace(workspace.root, cache)
    after_body = compute_states(doc, index3, env, NEVER, tier="T0")
    assert after_body["n2"].history_hash != before["n2"].history_hash


def test_tier_one_sees_same_module_but_not_cross_module(workspace):
    index1, cache = scan_workspace(workspace.root)
    env = env_fingerprint(workspace.root)
    doc = PipelineDoc.from_dict(three_node_doc())
    before = compute_states(doc, index1, env, NEVER, tier="T1")

    # Cross-module helper edit: invisible at T1.
    workspace.write("pipeline_lib/helpers.py", "def scale_factor():\n    return 99\n")
    index2, cache = scan_workspace(workspace.root, cache)
    after_helper = compute_states(doc, index2, env, NEVER, tier="T1")
    assert after_helper["n2"].history_hash == before["n2"].history_hash

    # Same-module edit (another function in transforms.py): visible at T1.
    original = (workspace.root / "pipeline_lib/transforms.py").read_text(encoding="utf-8")
    workspace.write(
        "pipeline_lib/transforms.py", original + "\n\ndef unrelated_helper():\n    return 1\n"
    )
    index3, _ = scan_workspace(workspace.root, cache)
    after_module = compute_states(doc, index3, env, NEVER, tier="T1")
    assert after_module["n2"].history_hash != before["n2"].history_hash
    # …while T0 stays blind to it.
    t0_before = compute_states(doc, index1, env, NEVER, tier="T0")
    t0_after = compute_states(doc, index3, env, NEVER, tier="T0")
    assert t0_after["n2"].history_hash == t0_before["n2"].history_hash
