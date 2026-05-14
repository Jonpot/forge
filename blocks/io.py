from __future__ import annotations

import os
from pathlib import Path
import json

import pandas as pd

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    block_param,
)

def _resolve_path(filepath: str) -> Path:
    """Resolve a filepath, expanding relative paths against FORGE_WORKSPACE_DIR."""
    path = Path(filepath)
    if not path.is_absolute():
        workspace_dir = os.environ.get("FORGE_WORKSPACE_DIR", "")
        if workspace_dir:
            path = Path(workspace_dir) / path
    return path



def _resolve_path(filepath: str) -> Path:
    """Resolve a filepath, expanding relative paths against FORGE_WORKSPACE_DIR."""
    path = Path(filepath)
    if not path.is_absolute():
        workspace_dir = os.environ.get("FORGE_WORKSPACE_DIR", "")
        if workspace_dir:
            path = Path(workspace_dir) / path
    return path


class LoadCSV(BaseBlock):
    name = "Load CSV"
    version = "1.0.0"
    category = "IO"
    description = "Load a CSV file into a DataFrame."
    n_inputs = 0
    output_labels = ["DataFrame"]
    presets = [
        {
            "id": "utf8_csv",
            "label": "UTF-8 CSV",
            "description": "Standard CSV input with comma delimiter and UTF-8 encoding.",
            "params": {
                "filepath": "C:\\Users\\you\\data.csv",
                "sep": ",",
                "encoding": "utf-8",
                "index_col": None,
            },
        }
    ]

    class Params(BlockParams):
        filepath: str = block_param(
            description="CSV file to load.",
            example="C:\\Users\\you\\data.csv",
            browse_mode="open_file",
        )
        sep: str = block_param(
            ",",
            description="Delimiter used in the CSV file.",
            example=",",
        )
        encoding: str = block_param(
            "utf-8",
            description="Text encoding for the input file, for example utf-8.",
            example="utf-8",
        )
        index_col: str | int | None = block_param(
            None,
            description="Optional column to use as the DataFrame index. Use null to keep the default integer index.",
            example=None,
        )

    def validate(self, data) -> None:
        if data is not None:
            raise BlockValidationError("LoadCSV expects no input data.")

    def execute(self, data, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("LoadCSV requires params.")
        filepath = _resolve_path(params.filepath)
        if not filepath.exists():
            raise BlockValidationError(f"CSV file not found: {filepath}")
        index_col = params.index_col
        if isinstance(index_col, str):
            index_col = index_col.strip() or None
        frame = pd.read_csv(
            filepath,
            sep=params.sep,
            encoding=params.encoding,
            index_col=index_col,
        )
        if index_col is not None:
            frame.index = frame.index.astype(str)
        return BlockOutput(data=frame)


class ExportCSV(BaseBlock):
    name = "Export CSV"
    version = "1.0.0"
    category = "IO"
    description = "Write the incoming DataFrame to a CSV file and pass data through."
    always_execute = True
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]
    presets = [
        {
            "id": "csv_no_index",
            "label": "CSV Without Index",
            "description": "Write a plain CSV without the DataFrame index column.",
            "params": {
                "filepath": "C:\\Users\\you\\outputs\\result.csv",
                "index": False,
            },
        }
    ]

    class Params(BlockParams):
        filepath: str = block_param(
            description="Destination CSV path to write.",
            example="C:\\Users\\you\\outputs\\result.csv",
            browse_mode="save_file",
        )
        index: bool = block_param(
            True,
            description="Whether to include the DataFrame index in the written CSV.",
            example=False,
        )

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("ExportCSV requires params.")
        path = _resolve_path(params.filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=bool(params.index))
        return BlockOutput(data=data, metadata={"exported_path": str(path)})


class Constant(BaseBlock):
    name = "Constant"
    version = "1.0.0"
    category = "IO"
    description = "Emit a typed constant value as a one-cell DataFrame."
    n_inputs = 0
    output_labels = ["Value"]

    class Params(BlockParams):
        value: str = block_param(
            "0",
            description="Literal value (interpreted according to value_type).",
            example="42",
        )
        value_type: str = block_param(
            "auto",
            description="How to interpret value: auto, int, float, string, or json.",
            example="auto",
        )

    def validate(self, data) -> None:
        if data is not None:
            raise BlockValidationError("Constant expects no input data.")

    def _parse_value(self, raw_value: object, value_type: str) -> object:
        mode = str(value_type or "auto").strip().lower()

        if mode == "string":
            return "" if raw_value is None else str(raw_value)

        if mode == "int":
            try:
                return int(raw_value)  # pyright: ignore[reportArgumentType]
            except Exception as exc:
                raise BlockValidationError(
                    f"Constant failed to parse int from value {raw_value!r}."
                ) from exc

        if mode == "float":
            try:
                return float(raw_value)  # pyright: ignore[reportArgumentType]
            except Exception as exc:
                raise BlockValidationError(
                    f"Constant failed to parse float from value {raw_value!r}."
                ) from exc

        if mode == "json":
            if isinstance(raw_value, str):
                text = raw_value.strip()
                if text == "":
                    raise BlockValidationError(
                        "Constant json mode requires a non-empty JSON string."
                    )
                try:
                    return json.loads(text)
                except Exception as exc:
                    raise BlockValidationError(
                        f"Constant failed to parse JSON from value {raw_value!r}."
                    ) from exc
            return raw_value

        if mode != "auto":
            raise BlockValidationError(
                "Constant value_type must be one of: auto, int, float, string, json."
            )

        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float, bool, dict, list)):
            return raw_value
        text = str(raw_value).strip()
        if text == "":
            return ""
        try:
            return int(text)
        except Exception:
            pass
        try:
            return float(text)
        except Exception:
            pass
        try:
            return json.loads(text)
        except Exception:
            return text

    def execute(self, data, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("Constant requires params.")
        parsed = self._parse_value(params.value, params.value_type)
        payload = pd.DataFrame({"value": [parsed]})
        return BlockOutput(
            data=payload,
            metadata={"value_type": type(parsed).__name__},
        )


class NoOp(BaseBlock):
    name = "No-Op"
    version = "1.0.0"
    category = "IO"
    description = (
        "Pass data through without modification. Generally used for anchoring edges."
    )
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame"]

    def execute(self, data: pd.DataFrame, params: None = None) -> BlockOutput:
        return BlockOutput(data=data.copy(deep=True))
