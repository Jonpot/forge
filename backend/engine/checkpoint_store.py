from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any

import pandas as pd

from backend.engine.provenance import Provenance, canonical_json


@dataclass(slots=True)
class CheckpointRecord:
    checkpoint_id: str
    path: Path
    provenance: Provenance
    data: pd.DataFrame | None = None


class CheckpointStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root_dir / "index.json"

    def _load_index(self) -> dict[str, str]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _save_index(self, payload: dict[str, str]) -> None:
        self._index_path.write_text(canonical_json(payload), encoding="utf-8")

    def _checkpoint_dir(self, checkpoint_id: str) -> Path:
        return self.root_dir / checkpoint_id

    def _safe_output_handle(self, output_handle: str) -> str:
        text = str(output_handle).strip()
        if not text:
            raise ValueError("Output handle cannot be empty.")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
        return safe or "output"

    def _output_data_path(self, checkpoint_dir: Path, output_handle: str) -> Path:
        safe_handle = self._safe_output_handle(output_handle)
        return checkpoint_dir / "outputs" / f"{safe_handle}.parquet"

    def get_checkpoint_id_by_hash(self, history_hash: str) -> str | None:
        return self._load_index().get(history_hash)

    def exists_by_hash(self, history_hash: str) -> bool:
        checkpoint_id = self.get_checkpoint_id_by_hash(history_hash)
        if checkpoint_id is None:
            return False
        return (self._checkpoint_dir(checkpoint_id) / "provenance.json").exists()

    def save(
        self,
        data: pd.DataFrame,
        provenance: Provenance,
        outputs: dict[str, pd.DataFrame] | None = None,
        images: list[Any] | None = None,
    ) -> str:
        checkpoint_id = provenance.history_hash.replace("sha256:", "")
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        normalized_outputs = {str(key): value for key, value in (outputs or {}).items()}
        if "output_0" not in normalized_outputs:
            normalized_outputs["output_0"] = data

        data_path = checkpoint_dir / "data.parquet"
        if not data_path.exists():
            # Preserve index so row identity survives checkpoint reuse.
            normalized_outputs["output_0"].to_parquet(data_path, index=True)

        for output_handle, output_data in normalized_outputs.items():
            if output_handle == "output_0":
                continue
            output_path = self._output_data_path(checkpoint_dir, output_handle)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_path.exists():
                output_data.to_parquet(output_path, index=True)

        image_names = self._save_images(checkpoint_dir, checkpoint_id, images or [])

        provenance.checkpoint_id = checkpoint_id
        provenance.images = image_names
        provenance_path = checkpoint_dir / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        index_payload = self._load_index()
        index_payload[provenance.history_hash] = checkpoint_id
        self._save_index(index_payload)
        return checkpoint_id

    def _save_images(
        self,
        checkpoint_dir: Path,
        checkpoint_id: str,
        images: list[Any],
    ) -> list[str]:
        if not images:
            return []

        image_dir = checkpoint_dir / "images"
        image_dir.mkdir(exist_ok=True)
        names: list[str] = []
        for idx, image in enumerate(images):
            base = f"image_{idx}_{checkpoint_id[:8]}"

            if hasattr(image, "savefig"):
                # matplotlib Figure
                name = f"{base}.png"
                image.savefig(image_dir / name, bbox_inches="tight")
            elif hasattr(image, "write_html"):
                # Plotly Figure — persist as interactive HTML so the frontend
                # can embed it in an iframe with full hover/pan/zoom support.
                name = f"{base}.html"
                image.write_html(  # type: ignore[call-arg]
                    str(image_dir / name),
                    full_html=True,
                    include_plotlyjs=True,
                )
            elif hasattr(image, "write_image"):
                # Non-Plotly image emitter (rare). Fall back to PNG via kaleido.
                name = f"{base}.png"
                try:
                    image.write_image(str(image_dir / name), format="png")  # type: ignore[call-arg]
                except Exception as exc:
                    raise TypeError(
                        "Failed to render image artifact. Ensure 'kaleido' is installed."
                    ) from exc
            elif isinstance(image, (str, Path)):
                src = Path(image)
                name = f"{base}{src.suffix or '.png'}"
                shutil.copy2(src, image_dir / name)
            else:
                raise TypeError(f"Unsupported image artifact type: {type(image)!r}")
            names.append(name)
        return names

    def load(self, checkpoint_id: str, with_data: bool = True) -> CheckpointRecord:
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        provenance = self.load_provenance(checkpoint_id)
        data = self.load_data(checkpoint_id) if with_data else None
        return CheckpointRecord(
            checkpoint_id=checkpoint_id,
            path=checkpoint_dir,
            provenance=provenance,
            data=data,
        )

    def load_provenance(self, checkpoint_id: str) -> Provenance:
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        provenance_path = checkpoint_dir / "provenance.json"
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        return Provenance.from_dict(payload)

    def load_data(self, checkpoint_id: str) -> pd.DataFrame:
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        data_path = checkpoint_dir / "data.parquet"
        return pd.read_parquet(data_path)

    def load_output(self, checkpoint_id: str, output_handle: str) -> pd.DataFrame:
        handle = str(output_handle)
        if handle == "output_0":
            return self.load_data(checkpoint_id)

        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        output_path = self._output_data_path(checkpoint_dir, handle)
        if not output_path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_id}' missing output handle '{output_handle}'."
            )
        return pd.read_parquet(output_path)

    def load_outputs(
        self,
        checkpoint_id: str,
        output_handles: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        handles = output_handles or ["output_0"]
        return {handle: self.load_output(checkpoint_id, handle) for handle in handles}

    def get_by_history_hash(
        self,
        history_hash: str,
        with_data: bool = True,
    ) -> CheckpointRecord | None:
        checkpoint_id = self.get_checkpoint_id_by_hash(history_hash)
        if checkpoint_id is None:
            return None
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        if not checkpoint_dir.exists():
            return None
        return self.load(checkpoint_id, with_data=with_data)

    def gc(self, keep_checkpoint_ids: set[str]) -> list[str]:
        removed: list[str] = []
        for path in self.root_dir.iterdir():
            if not path.is_dir():
                continue
            if path.name in keep_checkpoint_ids:
                continue
            shutil.rmtree(path)
            removed.append(path.name)

        index_payload = self._load_index()
        filtered = {
            history_hash: checkpoint_id
            for history_hash, checkpoint_id in index_payload.items()
            if checkpoint_id not in removed
        }
        self._save_index(filtered)
        return sorted(removed)
