from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any

import dill
import numpy as np
import pandas as pd

from backend.engine.object_store import ObjectStore
from backend.engine.provenance import Provenance, canonical_json


_OBJECT_STORE_DIRNAME = "_objects"


class _ContentAddressedPickler(dill.Pickler):
    """Pickler that diverts large DataFrames / ndarrays into an ObjectStore.

    persistent_id() is the standard pickle hook for "instead of serializing
    this object inline, here's a stable identifier — go look it up on
    reconstruction." Returning None means fall through to default pickling.
    """

    def __init__(self, file: Any, *, store: ObjectStore) -> None:
        super().__init__(file, recurse=True)
        self._store = store

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame) and ObjectStore.should_intercept_dataframe(obj):
            return ("dataframe", self._store.put_dataframe(obj))
        if isinstance(obj, np.ndarray) and ObjectStore.should_intercept_ndarray(obj):
            return ("ndarray", self._store.put_ndarray(obj))
        return None


class _ContentAddressedUnpickler(dill.Unpickler):
    def __init__(self, file: Any, *, store: ObjectStore) -> None:
        super().__init__(file)
        self._store = store

    def persistent_load(self, pid: Any) -> Any:
        if not isinstance(pid, tuple) or len(pid) != 2:
            raise dill.UnpicklingError(f"Unrecognized persistent id: {pid!r}")
        kind, content_hash = pid
        if kind == "dataframe":
            return self._store.get_dataframe(str(content_hash))
        if kind == "ndarray":
            return self._store.get_ndarray(str(content_hash))
        raise dill.UnpicklingError(f"Unknown persistent id kind: {kind!r}")


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
        # Content-addressed store for large values inside namespace pickles.
        # Lives next to the per-checkpoint dirs but is named with a leading
        # underscore so checkpoint IDs (sha256 hex) can never collide.
        self._object_store = ObjectStore(self.root_dir / _OBJECT_STORE_DIRNAME)

    @property
    def object_store(self) -> ObjectStore:
        return self._object_store

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
        namespace: dict[str, Any] | None = None,
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

        if namespace is not None:
            self._save_namespace(checkpoint_dir, namespace)

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

    def _namespace_path(self, checkpoint_dir: Path) -> Path:
        return checkpoint_dir / "namespace.pkl"

    def _save_namespace(
        self,
        checkpoint_dir: Path,
        namespace: dict[str, Any],
    ) -> None:
        path = self._namespace_path(checkpoint_dir)
        if path.exists():
            return
        with path.open("wb") as fh:
            pickler = _ContentAddressedPickler(fh, store=self._object_store)
            pickler.dump(namespace)

    def has_namespace(self, checkpoint_id: str) -> bool:
        return self._namespace_path(self._checkpoint_dir(checkpoint_id)).exists()

    def load_namespace(self, checkpoint_id: str) -> dict[str, Any] | None:
        path = self._namespace_path(self._checkpoint_dir(checkpoint_id))
        if not path.exists():
            return None
        with path.open("rb") as fh:
            unpickler = _ContentAddressedUnpickler(fh, store=self._object_store)
            obj = unpickler.load()
        if not isinstance(obj, dict):
            raise TypeError(
                f"Checkpoint '{checkpoint_id}' namespace.pkl did not deserialize to dict."
            )
        return obj

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
            name = f"image_{idx}_{checkpoint_id[:8]}.png"
            image_path = image_dir / name

            if hasattr(image, "savefig"):
                image.savefig(image_path, bbox_inches="tight")
            elif hasattr(image, "write_image"):
                try:
                    image.write_image(str(image_path), format="png")  # type: ignore[call-arg]
                except Exception as exc:
                    raise TypeError(
                        "Failed to render Plotly image artifact. Ensure 'kaleido' is installed."
                    ) from exc
            elif isinstance(image, (str, Path)):
                shutil.copy2(Path(image), image_path)
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
            if path.name == _OBJECT_STORE_DIRNAME:
                # Content-addressed object store is shared across checkpoints;
                # not a checkpoint dir itself. TODO: prune by reachability from
                # surviving namespaces in a separate pass.
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
