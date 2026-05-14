"""
Custom block management: install, list, delete, export, and template generation.

Custom blocks live in a user-accessible directory (e.g. ~/Documents/Forge/custom_blocks/)
so they survive app reinstalls and are easy to back up or share.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Block template ─────────────────────────────────────────────────────────────

BLOCK_TEMPLATE = '''\
"""
Custom Forge block: {block_name}

Drop this file onto the Forge canvas (or into the Plugins menu) to install it.
Another Forge user can install it the same way.

ANATOMY OF A BLOCK
------------------
1.  Class attributes  — name, version, category, description tell Forge how to
    display the block in the palette.  Change `category` to any string you like;
    Forge will create that category automatically.

2.  class Params      — a Pydantic model that describes every user-editable
    parameter.  Use `block_param()` to attach a description, default value,
    example, and (optionally) a file-picker browse mode.

3.  def validate()    — optional pre-flight check run before execute().
    Raise InsufficientInputs when a required input slot is not connected —
    the engine will silently skip the block rather than failing the pipeline.
    Raise BlockValidationError for data-quality issues (wrong columns, empty
    frame, etc.) that should surface as an error in the UI.

4.  def execute()     — the core logic.  Receives the upstream DataFrame (or
    None for source blocks) and the validated Params object.  Must return a
    BlockOutput.  Put matplotlib figures in `images=` to display them in the
    node inspector.

PACKAGE DEPENDENCIES
--------------------
If your block imports third-party packages, list them in REQUIREMENTS below.
Forge will pip-install them automatically when this file is first dropped in.

PLUGIN METADATA
---------------
Each .py file is treated as a plugin file. Optionally declare PLUGIN_TITLE and
PLUGIN_DESCRIPTION to control how the plugin appears in the Manage Plugins UI.

USAGE
-----
* Drag this .py file onto the Forge canvas to install.
* Right-click the block in the palette → "Export Plugin Source" to share it.
"""

# List any extra pip packages your block needs.
# Example: REQUIREMENTS = ["scipy>=1.12", "statsmodels"]
REQUIREMENTS: list[str] = []

# Optional plugin metadata shown in the Manage Plugins window.
PLUGIN_TITLE = "{block_name}"
PLUGIN_DESCRIPTION = "Describe what this plugin file provides."

from backend.block import BaseBlock, BlockOutput, BlockParams, BlockValidationError, InsufficientInputs, block_param
import pandas as pd


class {class_name}(BaseBlock):
    # ── Metadata ──────────────────────────────────────────────────────────────
    name = "{block_name}"
    version = "1.0.0"
    # Change to any category string — Forge will create it automatically.
    category = "Custom"
    description = "A short description shown in the palette tooltip."

    # How many input ports does this block expose in the UI?
    # 0 = source block (no inputs), 1 = single required input (default),
    # 2+ = multi-input.  Use validate() to decide which slots are truly
    # required: raise InsufficientInputs for missing required inputs so the
    # block is skipped rather than crashing the pipeline.
    n_inputs = 1
    input_labels = ["DataFrame"]
    output_labels = ["Result"]

    # Optional: list bullet-point tips shown in the node inspector.
    usage_notes: list[str] = [
        "Tip 1: what this block expects.",
        "Tip 2: what the output looks like.",
    ]

    # ── Parameters ────────────────────────────────────────────────────────────
    class Params(BlockParams):
        # Each field becomes an editable field in the node inspector.
        threshold: float = block_param(
            0.5,
            description="Example numeric parameter.",
            example=0.75,
        )
        column: str = block_param(
            "",
            description="Column to operate on. Leave blank to use the first column.",
        )
        # Uncomment for a file-picker:
        # output_path: str = block_param(
        #     "",
        #     description="Where to write output.",
        #     browse_mode="save_file",
        # )

    # ── Validation ────────────────────────────────────────────────────────────
    def validate(self, data: pd.DataFrame | None) -> None:
        # Raise InsufficientInputs when a required input isn't connected yet —
        # the pipeline will skip this block silently instead of failing.
        if data is None:
            raise InsufficientInputs("Input is not connected.")
        if data.empty:
            raise BlockValidationError("Input DataFrame is empty.")

    # ── Execution ─────────────────────────────────────────────────────────────
    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("Params are required.")

        col = params.column or data.columns[0]
        if col not in data.columns:
            raise BlockValidationError(f"Column \\"{col}\\" not found in DataFrame.")

        # --- YOUR LOGIC HERE ---
        result = data[data[col] > params.threshold].copy()
        # -----------------------

        # Optionally produce a matplotlib figure (shown in the node inspector):
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.hist(result[col], bins=20)
        # return BlockOutput(data=result, images=[fig])

        return BlockOutput(data=result)
'''


def _make_class_name(filename_stem: str) -> str:
    """Convert a filename stem to a valid PascalCase Python class name."""
    words = re.split(r"[\s_\-]+", filename_stem)
    return "".join(w.capitalize() for w in words if w)


def get_template(block_name: str = "My Custom Block") -> str:
    """Return a filled-in block template for the given display name."""
    class_name = _make_class_name(block_name.replace(" ", "_"))
    # Use simple replacement instead of .format() so code in the template
    # (f-strings, dicts, etc.) with curly braces is preserved as-is.
    return (
        BLOCK_TEMPLATE
        .replace("{block_name}", block_name)
        .replace("{class_name}", class_name)
    )


# ── Dependency detection ───────────────────────────────────────────────────────

def _extract_requirements(source: str) -> list[str]:
    """
    Parse a block source file and extract the REQUIREMENTS list literal.
    Returns the list of requirement strings, or [] if not found / not parseable.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "REQUIREMENTS"
            and isinstance(node.value, ast.List)
        ):
            reqs: list[str] = []
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    reqs.append(elt.value)
            return reqs
    return []


def _extract_imports(source: str) -> list[str]:
    """
    Walk the AST and collect top-level third-party package names that are
    imported but are NOT part of the standard library or known Forge internals.
    Only used as a fallback when REQUIREMENTS is absent.
    """
    _STDLIB = frozenset(sys.stdlib_module_names)  # Python 3.10+
    _FORGE_INTERNAL = frozenset({"backend", "blocks"})

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _STDLIB and top not in _FORGE_INTERNAL:
                    packages.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _STDLIB and top not in _FORGE_INTERNAL:
                    packages.add(top)

    # Filter out packages already present in the venv
    missing: list[str] = []
    for pkg in sorted(packages):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    return missing


def _iter_module_assignments(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    assignments: list[tuple[str, ast.AST]] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments.append((target.id, node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            assignments.append((node.target.id, node.value))
    return assignments


def _humanize_plugin_title(filename_stem: str) -> str:
    parts = [part for part in re.split(r"[\s_\-]+", filename_stem) if part]
    if not parts:
        return filename_stem or "Custom Plugin"
    return " ".join(part[:1].upper() + part[1:] for part in parts)


def _default_plugin_description(filename: str) -> str:
    return f"Custom block plugin installed from {filename}."


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    title: str
    description: str


def _extract_plugin_metadata(source: str, filename: str) -> PluginMetadata:
    stem = Path(filename).stem
    title: str | None = None
    description: str | None = None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return PluginMetadata(
            title=_humanize_plugin_title(stem),
            description=_default_plugin_description(filename),
        )

    for key, value_node in _iter_module_assignments(tree):
        try:
            value = ast.literal_eval(value_node)
        except Exception:
            continue

        if key == "PLUGIN_METADATA" and isinstance(value, dict):
            meta_title = value.get("title")
            meta_description = value.get("description")
            if title is None and isinstance(meta_title, str) and meta_title.strip():
                title = meta_title.strip()
            if (
                description is None
                and isinstance(meta_description, str)
                and meta_description.strip()
            ):
                description = meta_description.strip()
        elif key == "PLUGIN_TITLE" and isinstance(value, str) and value.strip():
            title = value.strip()
        elif (
            key == "PLUGIN_DESCRIPTION"
            and isinstance(value, str)
            and value.strip()
        ):
            description = value.strip()

    return PluginMetadata(
        title=title or _humanize_plugin_title(stem),
        description=description or _default_plugin_description(filename),
    )


# ── Installation result ────────────────────────────────────────────────────────

@dataclass
class InstallResult:
    success: bool
    block_name: str
    filename: str
    installed_packages: list[str]
    skipped_packages: list[str]
    errors: list[str]
    message: str
    # Conflict detection — set when an existing file would be overwritten and
    # no conflict_resolution was provided.
    conflict: bool = False
    suggested_filename: str | None = None


@dataclass(frozen=True, slots=True)
class CustomBlockFileInfo:
    filename: str
    stem: str
    path: str
    requirements: list[str]
    title: str
    description: str


# ── Manager ────────────────────────────────────────────────────────────────────

class CustomBlockManager:
    def __init__(self, custom_blocks_dir: str | Path) -> None:
        self.custom_blocks_dir = Path(custom_blocks_dir)

    def ensure_dir(self) -> None:
        self.custom_blocks_dir.mkdir(parents=True, exist_ok=True)

    def _suggest_filename(self, filename: str) -> str:
        """Return the next free filename: foo_2.py, foo_3.py, …"""
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        i = 2
        while (self.custom_blocks_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        return f"{stem}_{i}{suffix}"

    # ── List ──────────────────────────────────────────────────────────────────

    def list_blocks(self) -> list[CustomBlockFileInfo]:
        """Return metadata about all installed custom block files."""
        if not self.custom_blocks_dir.exists():
            return []
        result: list[CustomBlockFileInfo] = []
        for path in sorted(self.custom_blocks_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            source = path.read_text(encoding="utf-8")
            metadata = _extract_plugin_metadata(source, path.name)
            result.append(
                CustomBlockFileInfo(
                    filename=path.name,
                    stem=path.stem,
                    path=str(path),
                    requirements=_extract_requirements(source),
                    title=metadata.title,
                    description=metadata.description,
                )
            )
        return result

    # ── Install ───────────────────────────────────────────────────────────────

    def install(
        self,
        filename: str,
        content: bytes,
        conflict_resolution: str | None = None,
    ) -> InstallResult:
        """
        Save a .py block file to the custom blocks directory, detect its
        dependencies, install missing ones via pip, and return the result.

        conflict_resolution:
          None      — If a file with this name already exists, return a
                      conflict result instead of overwriting (safe default).
          "overwrite" — Replace the existing file unconditionally.
          "rename"  — Auto-rename the incoming file to the next free slot
                      (e.g. foo_2.py) so both files are kept.
        """
        self.ensure_dir()

        if not filename.endswith(".py"):
            return InstallResult(
                success=False,
                block_name=filename,
                filename=filename,
                installed_packages=[],
                skipped_packages=[],
                errors=["File must be a .py file."],
                message="File must be a .py file.",
            )

        source = content.decode("utf-8")

        # Validate parseable Python
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return InstallResult(
                success=False,
                block_name=filename,
                filename=filename,
                installed_packages=[],
                skipped_packages=[],
                errors=[f"Syntax error: {exc}"],
                message=f"Syntax error in block file: {exc}",
            )

        # Use plugin metadata for user-facing install messages when available.
        plugin_metadata = _extract_plugin_metadata(source, filename)
        block_name = plugin_metadata.title or _extract_block_name(source) or Path(filename).stem

        # ── Conflict check ────────────────────────────────────────────────────
        dest = self.custom_blocks_dir / filename
        if dest.exists() and conflict_resolution is None:
            # Caller must explicitly choose a resolution; return without writing.
            return InstallResult(
                success=False,
                conflict=True,
                block_name=block_name,
                filename=filename,
                suggested_filename=self._suggest_filename(filename),
                installed_packages=[],
                skipped_packages=[],
                errors=[],
                message=f"A file named '{filename}' is already installed.",
            )
        if conflict_resolution == "rename":
            filename = self._suggest_filename(filename)
            dest = self.custom_blocks_dir / filename
        # "overwrite" or no pre-existing file → dest already set correctly above
        elif not dest.exists():
            pass  # dest is correct; no conflict
        # else: conflict_resolution == "overwrite" → dest is the existing path, will overwrite

        # Determine requirements
        reqs = _extract_requirements(source)
        if not reqs:
            # Fallback: scan for missing imports
            reqs = _extract_imports(source)

        # Install requirements
        installed: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []

        for req in reqs:
            pkg_name = re.split(r"[>=<!@\[]", req)[0].strip()
            if importlib.util.find_spec(pkg_name) is not None:
                skipped.append(req)
                continue
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", req],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                installed.append(req)
            except subprocess.CalledProcessError as exc:
                errors.append(f"Failed to install {req!r}: {exc.stderr.strip()}")

        # Write the file (dest and filename already resolved above)
        dest.write_bytes(content)

        success = len(errors) == 0
        parts: list[str] = [f"Installed '{block_name}'."]
        if installed:
            parts.append(f"Installed packages: {', '.join(installed)}.")
        if errors:
            parts.append(f"Package errors: {'; '.join(errors)}.")

        return InstallResult(
            success=success,
            block_name=block_name,
            filename=filename,
            installed_packages=installed,
            skipped_packages=skipped,
            errors=errors,
            message=" ".join(parts),
        )

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, filename: str) -> bool:
        """Remove a custom block file. Returns True if deleted, False if not found."""
        path = self.custom_blocks_dir / filename
        if not path.exists():
            return False
        path.unlink()
        return True

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, filename: str) -> bytes:
        """Return the raw bytes of a custom block file."""
        path = self.custom_blocks_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Custom block not found: {filename}")
        return path.read_bytes()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_block_name(source: str) -> str | None:
    """Try to extract the `name = "..."` class attribute from a block source."""
    match = re.search(r'\bname\s*=\s*["\']([^"\']+)["\']', source)
    return match.group(1) if match else None
