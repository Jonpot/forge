from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Any

from pydantic.fields import FieldInfo

from backend.block import BaseBlock, BlockParams


def _annotation_to_text(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return str(annotation).replace("typing.", "")


def _humanize_param_key(key: str) -> str:
    words = [part for part in key.split("_") if part]
    if not words:
        return key
    return " ".join(words)


def _auto_param_description(
    key: str,
    default: Any,
    block_cls: type[BaseBlock] | None = None,
) -> str:
    lowered = key.lower()
    if lowered == "operator" or lowered.endswith("_operator"):
        options = getattr(block_cls, "_ops", None)
        if isinstance(options, dict) and options:
            allowed = ", ".join(sorted(str(name) for name in options))
            return f"Comparison operator. One of: {allowed}."
        return "Comparison operator."
    if lowered == "how":
        return "Join mode. One of: inner, left, right, outer."
    if lowered == "axis":
        return "Axis selection. Use 0 for columns and 1 for rows."
    if lowered == "sep":
        return "Delimiter used when reading or writing delimited text."
    if lowered == "encoding":
        return "Text encoding such as utf-8."
    if lowered == "index_col":
        return "Optional column to use as the row index. Use null to keep the default index."
    if lowered == "columns":
        return "Comma-separated column names. Leave empty to use the block default selection."
    if lowered.endswith("_columns") or lowered.endswith("_cols"):
        return "Comma-separated column names."
    if lowered.endswith("_column") or lowered.endswith("_col"):
        return "Column name."
    if lowered == "column_prefix":
        return "Optional prefix used to include every column whose name starts with this text."
    if lowered == "output_column":
        return "Name of the output column to create."
    if lowered == "standardize":
        return "Standardize numeric columns before fitting the model."
    if lowered == "random_state":
        return "Random seed for reproducibility."
    if lowered == "n_components":
        return "Number of output dimensions or latent components to compute."
    if lowered == "n_neighbors":
        return "Number of nearest neighbors used to build the local neighborhood graph."
    if lowered == "min_dist":
        return "UMAP minimum distance controlling how tightly points may cluster."
    if lowered == "prefix":
        return "Text prefix used when naming generated output columns."
    if lowered == "figsize":
        return "Figure size as [width, height] in inches."
    if lowered == "cmap":
        return "Matplotlib color map name."
    if lowered == "alpha":
        return "Opacity value between 0 and 1."
    if lowered == "rotation":
        return "Axis label rotation in degrees."
    if lowered == "value_type":
        return "How to interpret the literal value. One of: auto, int, float, string, json."
    if lowered in {"filepath", "path"}:
        return "Filesystem path."
    if "separator" in lowered:
        return "Separator string used when splitting or composing keys."
    if lowered.startswith("n_") or lowered in {"n_iters", "n_repeats", "n_clusters"}:
        return f"Count for {_humanize_param_key(key)}."
    if "seed" in lowered:
        return "Random seed for reproducibility."
    if "lambda" in lowered:
        return "Regularization strength."
    if isinstance(default, bool):
        return f"Enable or disable {_humanize_param_key(key)}."
    if isinstance(default, int | float):
        return f"Numeric value for {_humanize_param_key(key)}."
    if isinstance(default, str):
        return f"Value for {_humanize_param_key(key)}."
    return f"Parameter for {_humanize_param_key(key)}."


def _params_model_for_block(block_cls: type[BaseBlock]) -> type[BlockParams] | None:
    params_cls = getattr(block_cls, "Params", None)
    if params_cls is None:
        return None
    if not inspect.isclass(params_cls) or not issubclass(params_cls, BlockParams):
        raise TypeError(
            f"{block_cls.__name__}.Params must inherit from BlockParams."
        )
    return params_cls


def _field_default(field: FieldInfo) -> Any:
    if field.is_required():
        return None
    return deepcopy(field.get_default(call_default_factory=True))


def _field_example(field: FieldInfo, default: Any) -> Any:
    if field.examples:
        return deepcopy(field.examples[0])
    if isinstance(field.json_schema_extra, dict) and "example" in field.json_schema_extra:
        return deepcopy(field.json_schema_extra["example"])
    return deepcopy(default)


def _field_browse_mode(field: FieldInfo) -> str | None:
    if not isinstance(field.json_schema_extra, dict):
        return None
    browse_mode = field.json_schema_extra.get("browse_mode")
    return browse_mode if isinstance(browse_mode, str) else None


@dataclass(slots=True)
class BlockParamSpec:
    key: str
    type: str
    default: Any
    required: bool
    description: str
    example: Any
    browse_mode: str | None


def _extract_param_specs(block_cls: type[BaseBlock]) -> list[BlockParamSpec]:
    params_cls = _params_model_for_block(block_cls)
    if params_cls is None:
        return []

    explicit_descriptions = getattr(block_cls, "param_descriptions", {})
    if not isinstance(explicit_descriptions, dict):
        explicit_descriptions = {}

    specs: list[BlockParamSpec] = []
    for key, field in params_cls.model_fields.items():
        default = _field_default(field)
        description = (field.description or "").strip()
        if not description:
            description = str(explicit_descriptions.get(key, "") or "").strip()
        if not description:
            description = _auto_param_description(key, default, block_cls)
        specs.append(
            BlockParamSpec(
                key=key,
                type=_annotation_to_text(field.annotation),
                default=default,
                required=field.is_required(),
                description=description,
                example=_field_example(field, default),
                browse_mode=_field_browse_mode(field),
            )
        )
    return specs


@dataclass(slots=True)
class BlockSpec:
    key: str
    display_name: str
    aliases: list[str]
    version: str
    category: str
    description: str
    n_inputs: int
    input_labels: list[str]
    output_labels: list[str]
    param_schema: list[BlockParamSpec]
    params: dict[str, Any]
    param_types: dict[str, str]
    param_descriptions: dict[str, str]
    required_params: list[str]
    param_examples: dict[str, Any]
    is_custom: bool = False
    custom_filename: str | None = None
    arity_input_param: str | None = None
    arity_output_param: str | None = None


class BlockRegistry:
    def __init__(
        self,
        blocks_dir: str | Path = "blocks",
        package_name: str = "blocks",
        custom_blocks_dir: str | Path | None = None,
    ) -> None:
        self.blocks_dir = Path(blocks_dir)
        self.package_name = package_name
        self.custom_blocks_dir: Path | None = (
            Path(custom_blocks_dir) if custom_blocks_dir else None
        )
        self._blocks: dict[str, type[BaseBlock]] = {}
        # Track which block keys came from custom files (filename stem → set of keys)
        self._custom_block_keys: dict[str, set[str]] = {}

    def discover(self, force_reload: bool = False) -> None:
        if not self.blocks_dir.exists():
            # In production (installed via pip), blocks_dir may be relative to cwd
            # which doesn't contain the blocks. Fall back to the installed package location.
            try:
                package = importlib.import_module(self.package_name)
                if package.__file__:
                    pkg_dir = Path(package.__file__).parent
                    if pkg_dir.exists():
                        self.blocks_dir = pkg_dir
            except ImportError:
                pass
        if not self.blocks_dir.exists():
            raise FileNotFoundError(f"Blocks directory not found: {self.blocks_dir}")

        self._blocks = {}
        self._custom_block_keys = {}
        package = importlib.import_module(self.package_name)
        if package.__file__ is None:
            raise ImportError(
                f"Cannot determine file path for package: {self.package_name}"
            )
        package_path = Path(package.__file__).parent

        for module_info in pkgutil.iter_modules([str(package_path)]):
            if module_info.name.startswith("_"):
                continue
            module_name = f"{self.package_name}.{module_info.name}"
            module = importlib.import_module(module_name)
            if force_reload:
                module = importlib.reload(module)

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    not issubclass(cls, BaseBlock)
                    or cls is BaseBlock
                    or inspect.isabstract(cls)
                ):
                    continue
                self._validate_block_metadata(cls)
                self._blocks[cls.__name__] = cls

        # Load custom blocks from the user directory
        if self.custom_blocks_dir and self.custom_blocks_dir.exists():
            self._discover_custom_blocks(force_reload=force_reload)

    def _discover_custom_blocks(self, force_reload: bool = False) -> None:
        """Load BaseBlock subclasses from loose .py files in custom_blocks_dir."""
        import types as _types

        assert self.custom_blocks_dir is not None
        for py_file in sorted(self.custom_blocks_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            stem = py_file.stem
            module_name = f"_forge_custom_{stem}"

            try:
                if module_name in sys.modules and force_reload:
                    del sys.modules[module_name]

                if module_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec is None or spec.loader is None:
                        continue
                    module = _types.ModuleType(module_name)
                    module.__file__ = str(py_file)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                else:
                    module = sys.modules[module_name]

                keys_for_file: set[str] = set()
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    if (
                        not issubclass(cls, BaseBlock)
                        or cls is BaseBlock
                        or inspect.isabstract(cls)
                    ):
                        continue
                    try:
                        self._validate_block_metadata(cls)
                    except (ValueError, TypeError):
                        continue
                    # Mark as custom
                    cls._is_custom = True  # type: ignore[attr-defined]
                    cls._custom_filename = py_file.name  # type: ignore[attr-defined]
                    self._blocks[cls.__name__] = cls
                    keys_for_file.add(cls.__name__)

                self._custom_block_keys[stem] = keys_for_file

            except Exception:
                # Don't crash the whole registry on a broken custom block
                import traceback
                traceback.print_exc()
                continue

    def reload_custom_blocks(self) -> None:
        """Hot-reload all custom blocks (call after installing a new one)."""
        # Remove custom block entries from _blocks
        for keys in self._custom_block_keys.values():
            for key in keys:
                self._blocks.pop(key, None)
        self._custom_block_keys = {}

        if self.custom_blocks_dir and self.custom_blocks_dir.exists():
            self._discover_custom_blocks(force_reload=True)

    def is_custom(self, block_key: str) -> bool:
        """Return True if block_key refers to a user-installed custom block."""
        cls = self._blocks.get(block_key)
        return bool(cls and getattr(cls, "_is_custom", False))

    def custom_filename(self, block_key: str) -> str | None:
        """Return the .py filename for a custom block, or None."""
        cls = self._blocks.get(block_key)
        if cls is None:
            return None
        return getattr(cls, "_custom_filename", None)

    def _validate_block_metadata(self, block_cls: type[BaseBlock]) -> None:
        required = ("name", "version", "category")
        missing = [field for field in required if not getattr(block_cls, field, None)]
        if missing:
            raise ValueError(
                f"Block {block_cls.__name__} missing required metadata: {missing}"
            )
        _params_model_for_block(block_cls)

    def get(self, block_key: str) -> type[BaseBlock]:
        block_cls = self._blocks.get(block_key)
        if block_cls is not None:
            return block_cls

        for candidate in self._blocks.values():
            if candidate.name == block_key:
                return candidate
            aliases = getattr(candidate, "aliases", None)
            if isinstance(aliases, (list, tuple, set)) and block_key in {
                str(alias) for alias in aliases
            }:
                return candidate
        raise KeyError(f"Unknown block: {block_key}")

    def all_specs(self) -> list[BlockSpec]:
        specs: list[BlockSpec] = []
        for key, block_cls in sorted(
            self._blocks.items(), key=lambda item: item[0].lower()
        ):
            n_inputs = int(getattr(block_cls, "n_inputs", 1))
            input_labels = self._normalize_input_labels(
                getattr(block_cls, "input_labels", []), n_inputs
            )
            output_labels = self._normalize_output_labels(
                getattr(block_cls, "output_labels", ["output"])
            )
            description = self._description_for_block(block_cls)
            param_schema = _extract_param_specs(block_cls)
            is_custom = bool(getattr(block_cls, "_is_custom", False))
            custom_fn = getattr(block_cls, "_custom_filename", None)
            specs.append(
                BlockSpec(
                    key=key,
                    display_name=block_cls.name,
                    aliases=self._normalize_aliases(getattr(block_cls, "aliases", [])),
                    version=block_cls.version,
                    category=block_cls.category,
                    description=description,
                    n_inputs=n_inputs,
                    input_labels=input_labels,
                    output_labels=output_labels,
                    param_schema=param_schema,
                    params={item.key: deepcopy(item.default) for item in param_schema},
                    param_types={item.key: item.type for item in param_schema},
                    param_descriptions={
                        item.key: item.description for item in param_schema
                    },
                    required_params=[
                        item.key for item in param_schema if item.required
                    ],
                    param_examples={
                        item.key: deepcopy(item.example) for item in param_schema
                    },
                    is_custom=is_custom,
                    custom_filename=custom_fn if isinstance(custom_fn, str) else None,
                    arity_input_param=getattr(block_cls, "arity_input_param", None),
                    arity_output_param=getattr(block_cls, "arity_output_param", None),
                )
            )
        return specs

    def _description_for_block(self, block_cls: type[BaseBlock]) -> str:
        description = str(getattr(block_cls, "description", "") or "").strip()
        if description:
            return description
        doc = inspect.getdoc(block_cls)
        if not doc:
            return f"{block_cls.name} block."
        return doc.splitlines()[0].strip()

    def _normalize_input_labels(self, labels: Any, n_inputs: int) -> list[str]:
        if n_inputs <= 0:
            return []
        if isinstance(labels, (list, tuple)):
            normalized = [str(item).strip() for item in labels]
        else:
            normalized = []
        if len(normalized) < n_inputs:
            normalized.extend(
                [f"Input {i + 1}" for i in range(len(normalized), n_inputs)]
            )
        return normalized[:n_inputs]

    def _normalize_output_labels(self, labels: Any) -> list[str]:
        if isinstance(labels, (list, tuple)):
            normalized = [str(item).strip() for item in labels if str(item).strip()]
            if normalized:
                return normalized
        return ["output"]

    def _normalize_aliases(self, aliases: Any) -> list[str]:
        if not isinstance(aliases, (list, tuple, set)):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            text = str(alias).strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return normalized
