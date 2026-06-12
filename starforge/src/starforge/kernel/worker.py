"""Per-run worker subprocess: ``python -m starforge.kernel.worker <spec.json>``.

This is the ONLY *Forge process that imports user code. It lives for exactly
one run, so cancellation is a process kill, leaked memory returns to the OS,
and just-edited functions are picked up by fresh imports — no reload hacks.

Protocol: NDJSON events on the real stdout. ``sys.stdout`` is rebound to
stderr before any user code runs, so stray ``print()`` calls in user blocks
become log lines instead of corrupting the event stream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m starforge.kernel.worker <run-spec.json>", file=sys.stderr)
        return 2

    # Before any user import: headless matplotlib (no GUI windows, no Tk
    # errors) — plt.show() becomes a no-op and the figure sweep collects
    # what the user "showed".
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))

    events_out = sys.stdout
    sys.stdout = sys.stderr  # user prints become logs, never protocol frames

    def emit(event: dict) -> None:
        events_out.write(json.dumps(event, default=repr) + "\n")
        events_out.flush()

    emit({"event": "worker_started", "python": sys.version.split()[0]})

    workspace = spec["workspace"]
    sys.path.insert(0, workspace)

    from starforge.core.checkpoints import CheckpointStore
    from starforge.core.runner import run_pipeline
    from starforge.core.spec import PipelineDoc

    try:
        status = run_pipeline(
            doc=PipelineDoc.from_dict(spec["doc"]),
            blocks=spec["blocks"],
            states=spec["states"],
            store=CheckpointStore(workspace),
            emit=emit,
            pickle_enabled=bool(spec.get("pickle_enabled", False)),
            target=spec.get("target"),
        )
    except Exception:
        import traceback

        emit({"event": "run_finished", "status": "failed", "traceback": traceback.format_exc()})
        return 1
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
