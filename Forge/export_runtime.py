from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from backend.block import BlockOutput
from backend.engine.checkpoint_store import CheckpointStore
from backend.engine.provenance import (
    Provenance,
    combine_parent_history_hashes,
    compute_history_hash,
    sha256_text,
)
from backend.engine.runner import PipelineRunner
from backend.registry import BlockRegistry


@dataclass(frozen=True, slots=True)
class OutputRef:
    result: "ExecutedBlockResult"
    handle: str = "output_0"

    @property
    def frame(self) -> pd.DataFrame:
        return self.result.outputs[self.handle]

    @property
    def dependency_hash(self) -> str:
        return self.result.dependency_hash(self.handle)


@dataclass(slots=True)
class ExecutedBlockResult:
    node_id: str
    block_key: str
    checkpoint_id: str
    history_hash: str
    status: str
    outputs: dict[str, pd.DataFrame]
    metadata: dict[str, Any] = field(default_factory=dict)
    image_paths: list[str] = field(default_factory=list)
    namespace: dict[str, Any] | None = None

    @property
    def data(self) -> pd.DataFrame:
        return self.outputs["output_0"]

    def output(self, handle: str = "output_0") -> OutputRef:
        return OutputRef(result=self, handle=handle)

    def dependency_hash(self, handle: str = "output_0") -> str:
        if handle == "output_0":
            return self.history_hash
        return sha256_text(f"{self.history_hash}|{handle}")

    def to_summary(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "block_key": self.block_key,
            "checkpoint_id": self.checkpoint_id,
            "history_hash": self.history_hash,
            "status": self.status,
            "image_paths": self.image_paths,
        }


class ExportRuntime:
    def __init__(self, *, root_dir: str | Path, pipeline_name: str) -> None:
        self.root_dir = Path(root_dir)
        self.pipeline_name = pipeline_name
        self.outputs_dir = self.root_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.outputs_dir / "checkpoints"
        self.checkpoint_store = CheckpointStore(self.checkpoint_dir)
        self.registry = BlockRegistry(
            blocks_dir=self.root_dir / "blocks",
            package_name="blocks",
        )
        self.registry.discover(force_reload=False)
        self.runner = PipelineRunner(self.registry, self.checkpoint_store)
        self.results: dict[str, ExecutedBlockResult] = {}
        self.executed_nodes: list[str] = []
        self.reused_nodes: list[str] = []

    def run_block(
        self,
        *,
        node_id: str,
        block_key: str,
        params: dict[str, Any] | None = None,
        inputs: list[OutputRef] | None = None,
    ) -> ExecutedBlockResult:
        block_cls = self.registry.get(block_key)
        block = block_cls()
        params_payload = block_cls.normalize_params_payload(params or {})
        params_obj = self.runner._instantiate_params(block_cls, params_payload)
        input_refs = list(inputs or [])
        expected_inputs = int(block_cls.resolve_n_inputs(params_payload))
        if len(input_refs) != expected_inputs:
            raise ValueError(
                f"Node {node_id} ({block_cls.__name__}) expects {expected_inputs} inputs, "
                f"received {len(input_refs)}."
            )

        parent_history_hash = self._parent_history_hash(
            node_id=node_id,
            block_key=block_key,
            params_payload=params_payload,
            input_refs=input_refs,
        )
        history_hash = compute_history_hash(
            parent_history_hash,
            block_cls.name,
            block_cls.version,
            params_payload,
        )

        existing_checkpoint_id = self.checkpoint_store.get_checkpoint_id_by_hash(history_hash)
        should_force_execute = bool(block_cls.should_force_execute(params_payload))
        if existing_checkpoint_id is not None and not should_force_execute:
            result = self._load_existing_result(
                node_id=node_id,
                block_key=block_key,
                block_cls=block_cls,
                checkpoint_id=existing_checkpoint_id,
                history_hash=history_hash,
                params_payload=params_payload,
            )
            self.results[node_id] = result
            self.reused_nodes.append(node_id)
            return result

        input_data = self._resolve_input_data(block_cls, input_refs, params_payload)
        block.validate(input_data)

        started = perf_counter()
        output = block.execute(input_data, params_obj)
        duration = perf_counter() - started
        if not isinstance(output, BlockOutput):
            raise TypeError(
                f"Block {block_cls.__name__} returned {type(output)!r}; expected BlockOutput."
            )

        output_frames = self.runner._normalize_output_frames(block_cls, output, params_payload)
        output_df = output_frames["output_0"]
        provenance = Provenance(
            checkpoint_id="",
            block_name=block_cls.name,
            block_version=block_cls.version,
            params=params_payload,
            parent_checkpoint_ids=[ref.result.checkpoint_id for ref in input_refs],
            initial_data_signature=parent_history_hash if not input_refs else None,
            history_hash=history_hash,
            execution_time_seconds=round(duration, 6),
            output_shape=[int(output_df.shape[0]), int(output_df.shape[1])],
            images=[],
        )
        output_namespace: dict[str, Any] | None = None
        if getattr(block_cls, "produces_namespace", False):
            output_namespace = output.namespace if output.namespace is not None else {}

        checkpoint_id = self.checkpoint_store.save(
            data=output_df,
            provenance=provenance,
            outputs=output_frames,
            images=output.images,
            namespace=output_namespace,
        )
        result = ExecutedBlockResult(
            node_id=node_id,
            block_key=block_key,
            checkpoint_id=checkpoint_id,
            history_hash=history_hash,
            status="executed",
            outputs=output_frames,
            metadata=dict(output.metadata),
            image_paths=self._image_paths(checkpoint_id),
            namespace=output_namespace,
        )
        self.results[node_id] = result
        self.executed_nodes.append(node_id)
        return result

    def finish(self) -> dict[str, Any]:
        summary = {
            "pipeline_name": self.pipeline_name,
            "executed_nodes": list(self.executed_nodes),
            "reused_nodes": list(self.reused_nodes),
            "node_results": {
                node_id: result.to_summary() for node_id, result in self.results.items()
            },
        }
        summary_path = self.outputs_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    def _resolve_input_data(
        self,
        block_cls: type[Any],
        input_refs: list[OutputRef],
        params: dict[str, Any] | None = None,
    ) -> Any:
        n_inputs = int(block_cls.resolve_n_inputs(params))
        if n_inputs == 0:
            return None

        if getattr(block_cls, "consumes_namespace", False):
            resolved = [self._resolve_parent_namespace(ref) for ref in input_refs]
            return resolved[0] if n_inputs == 1 else resolved

        if n_inputs == 1:
            return input_refs[0].frame
        return [ref.frame for ref in input_refs]

    def _resolve_parent_namespace(self, ref: OutputRef) -> dict[str, Any]:
        ns = ref.result.namespace
        if ns is not None:
            return ns
        return {"data": ref.frame}

    def _parent_history_hash(
        self,
        *,
        node_id: str,
        block_key: str,
        params_payload: dict[str, Any],
        input_refs: list[OutputRef],
    ) -> str:
        if not input_refs:
            return self.runner._compute_initial_signature(
                {"id": node_id, "block": block_key, "params": params_payload}
            )
        return combine_parent_history_hashes(
            [ref.dependency_hash for ref in input_refs]
        )

    def _load_existing_result(
        self,
        *,
        node_id: str,
        block_key: str,
        block_cls: type[Any],
        checkpoint_id: str,
        history_hash: str,
        params_payload: dict[str, Any] | None = None,
    ) -> ExecutedBlockResult:
        output_handles = self.runner._expected_output_handles(block_cls, params_payload)
        outputs = self.checkpoint_store.load_outputs(checkpoint_id, output_handles)
        loaded_namespace: dict[str, Any] | None = None
        if getattr(block_cls, "produces_namespace", False):
            loaded_namespace = self.checkpoint_store.load_namespace(checkpoint_id)
        return ExecutedBlockResult(
            node_id=node_id,
            block_key=block_key,
            checkpoint_id=checkpoint_id,
            history_hash=history_hash,
            status="reused",
            outputs=outputs,
            metadata={},
            image_paths=self._image_paths(checkpoint_id),
            namespace=loaded_namespace,
        )

    def _image_paths(self, checkpoint_id: str) -> list[str]:
        provenance = self.checkpoint_store.load_provenance(checkpoint_id)
        image_dir = self.checkpoint_dir / checkpoint_id / "images"
        return [
            str((image_dir / name).resolve())
            for name in provenance.images
        ]
