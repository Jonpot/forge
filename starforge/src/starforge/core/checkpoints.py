"""Checkpoint store under ``<workspace>/.forge/checkpoints/``.

One directory per history hash (truncated to 32 hex chars — 128 bits — to
stay friendly to Windows path limits):

    .forge/checkpoints/<hash32>/
    ├── provenance.json     # written LAST: its presence marks completeness
    └── outputs/<name>.<ext per serializer>

The store also owns ``.forge/.gitignore`` so checkpoints and caches never
land in the user's repo history while ``pipelines/`` remains committable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starforge.core import figures as figmod
from starforge.core import previews, serializers

FORGE_DIR = ".forge"
GITIGNORE_BODY = "checkpoints/\ncache/\n"


class CheckpointStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.forge_dir = self.workspace / FORGE_DIR
        self.base = self.forge_dir / "checkpoints"

    def ensure_layout(self) -> None:
        (self.forge_dir / "pipelines").mkdir(parents=True, exist_ok=True)
        (self.forge_dir / "cache").mkdir(parents=True, exist_ok=True)
        self.base.mkdir(parents=True, exist_ok=True)
        gitignore = self.forge_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(GITIGNORE_BODY, encoding="utf-8")

    def dir_for(self, history_hash: str) -> Path:
        return self.base / history_hash[:32]

    def exists(self, history_hash: str) -> bool:
        return (self.dir_for(history_hash) / "provenance.json").is_file()

    def read_provenance(self, history_hash: str) -> dict[str, Any]:
        path = self.dir_for(history_hash) / "provenance.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def write(
        self,
        history_hash: str,
        provenance: dict[str, Any],
        outputs: dict[str, Any],
        pickle_enabled: bool = False,
        side_figures: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Persist outputs then provenance (in that order, for atomicity).
        Returns the output manifest, including ephemeral entries.

        ``side_figures`` are figures the block created or showed without
        returning them (plt.show() and friends); they render to artifacts
        recorded under the provenance ``figures`` key."""
        directory = self.dir_for(history_hash)
        outputs_dir = directory / "outputs"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = []
        for name, value in outputs.items():
            entry = serializers.save_value(value, outputs_dir, name, pickle_enabled=pickle_enabled)
            try:
                # Previews ride inside provenance.json so the stdlib-only
                # kernel can serve them without deserializing data. Computed
                # for ephemeral outputs too — their only window is right now.
                if entry.get("artifact"):
                    entry["preview"] = {
                        "kind": "figure",
                        "file": entry["artifact"]["file"],
                        "format": entry["artifact"]["kind"],
                    }
                else:
                    entry["preview"] = previews.build_preview(value)
            except Exception:
                entry["preview"] = {"kind": "text", "text": f"<preview failed for {type(value).__name__}>"}
            manifest.append(entry)

        rendered_figures: list[dict[str, Any]] = []
        for i, fig in enumerate(side_figures or []):
            try:
                artifact = figmod.render_figure(fig, outputs_dir, f"figure_{i}")
            except Exception:
                artifact = None
            if artifact is not None:
                rendered_figures.append(artifact)

        record = dict(provenance)
        record["history_hash"] = history_hash
        record["outputs"] = manifest
        record["figures"] = rendered_figures
        record["dir"] = directory.relative_to(self.workspace).as_posix()
        path = directory / "provenance.json"
        tmp = directory / "provenance.json.tmp"
        tmp.write_text(json.dumps(record, indent=2, default=repr), encoding="utf-8")
        tmp.replace(path)
        return manifest

    def output_entry(self, history_hash: str, name: str) -> dict[str, Any]:
        for entry in self.read_provenance(history_hash).get("outputs", []):
            if entry.get("name") == name:
                return entry
        raise KeyError(f"checkpoint {history_hash[:12]} has no output named '{name}'")

    def load_output(self, history_hash: str, name: str) -> Any:
        """Raises serializers.EphemeralValueError for non-persisted outputs."""
        entry = self.output_entry(history_hash, name)
        return serializers.load_value(self.dir_for(history_hash) / "outputs", entry)

    def is_ephemeral(self, history_hash: str, name: str) -> bool:
        try:
            entry = self.output_entry(history_hash, name)
        except (KeyError, FileNotFoundError, json.JSONDecodeError):
            return True
        return entry.get("serializer") == serializers.EPHEMERAL
