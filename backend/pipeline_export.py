from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
from typing import Any, Literal
import zipfile

from backend.pipeline_graph import group_map

ExportFormat = Literal["python", "notebook"]


def _slugify(name: str, *, fallback: str = "pipeline") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name or "").strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


@dataclass(frozen=True, slots=True)
class CodeExpr:
    source: str


@dataclass(frozen=True, slots=True)
class CopiedInputFile:
    source: Path
    relative_path: Path


@dataclass(frozen=True, slots=True)
class GeneratedBundle:
    archive_name: str
    content: bytes


def build_pipeline_export_archive(
    *,
    pipeline_id: str,
    pipeline: dict[str, Any],
    settings: Any,
    registry: Any,
    runner: Any,
    export_format: ExportFormat,
) -> GeneratedBundle:
    if export_format not in {"python", "notebook"}:
        raise ValueError(f"Unsupported export format: {export_format}")

    package_name = _slugify(pipeline.get("name", pipeline_id), fallback=pipeline_id)
    archive_root = f"{package_name}_{export_format}"
    export_basename = package_name
    exported_pipeline, copied_inputs = _rewrite_pipeline_paths(
        pipeline=pipeline,
        registry=registry,
        runner=runner,
    )

    script_name = f"{export_basename}.py"
    notebook_name = f"{export_basename}.ipynb"
    pipeline_json_name = f"{export_basename}.pipeline.json"
    readme_text = _build_readme(
        pipeline_name=str(pipeline.get("name", pipeline_id)),
        script_name=script_name,
        notebook_name=notebook_name,
        pipeline_json_name=pipeline_json_name,
        export_format=export_format,
    )

    if export_format == "python":
        primary_name = script_name
        primary_content = _build_python_script(
            pipeline=exported_pipeline,
            registry=registry,
            runner=runner,
            script_name=script_name,
        )
    else:
        primary_name = notebook_name
        primary_content = _build_notebook(
            pipeline=exported_pipeline,
            registry=registry,
            runner=runner,
        )

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{archive_root}/{pipeline_json_name}",
            json.dumps(exported_pipeline, indent=2, sort_keys=False),
        )
        archive.writestr(f"{archive_root}/{primary_name}", primary_content)
        archive.writestr(f"{archive_root}/README.md", readme_text)
        archive.writestr(
            f"{archive_root}/outputs/.gitkeep",
            "",
        )
        requirements_path = _repo_root() / "requirements.txt"
        if requirements_path.exists():
            archive.writestr(
                f"{archive_root}/requirements.txt",
                requirements_path.read_text(encoding="utf-8"),
            )

        pyproject_path = _repo_root() / "pyproject.toml"
        if pyproject_path.exists():
            archive.writestr(
                f"{archive_root}/pyproject.toml",
                pyproject_path.read_text(encoding="utf-8"),
            )

        for copied in copied_inputs:
            archive.write(
                copied.source,
                arcname=f"{archive_root}/{copied.relative_path.as_posix()}",
            )

        _write_tree_to_archive(
            archive,
            source_root=_repo_root() / "Forge",
            archive_root=Path(archive_root) / "Forge",
        )
        _write_tree_to_archive(
            archive,
            source_root=_repo_root() / "backend",
            archive_root=Path(archive_root) / "backend",
        )
        _write_tree_to_archive(
            archive,
            source_root=_resolve_repo_path(settings.blocks_dir),
            archive_root=Path(archive_root) / "blocks",
        )

    return GeneratedBundle(
        archive_name=f"{archive_root}.zip",
        content=bundle.getvalue(),
    )


def _rewrite_pipeline_paths(
    *,
    pipeline: dict[str, Any],
    registry: Any,
    runner: Any,
) -> tuple[dict[str, Any], list[CopiedInputFile]]:
    payload = json.loads(json.dumps(pipeline))
    node_map, incoming, _ = runner._prepare_graph(payload)
    copied_inputs: list[CopiedInputFile] = []

    for node_id, node in node_map.items():
        block_cls = registry.get(str(node["block"]))
        params = dict(node.get("params", {}) or {})
        params_payload = runner._params_payload(node, block_cls)
        is_root = len(incoming[node_id]) == 0

        if is_root and not block_cls.should_force_execute(params_payload):
            filepath = str(params_payload.get("filepath") or "").strip()
            if filepath:
                source_path = Path(filepath)
                if not source_path.exists() or not source_path.is_file():
                    raise FileNotFoundError(
                        f"Cannot export pipeline input for node '{node_id}': {source_path}"
                    )
                relative_path = Path("inputs") / _slugify(node_id, fallback="input") / source_path.name
                params["filepath"] = relative_path.as_posix()
                copied_inputs.append(
                    CopiedInputFile(
                        source=source_path,
                        relative_path=relative_path,
                    )
                )

        if block_cls.should_force_execute(params_payload):
            filepath = str(params_payload.get("filepath") or "").strip()
            if filepath:
                target_name = Path(filepath).name or f"{_slugify(node_id)}.csv"
                params["filepath"] = (
                    Path("outputs")
                    / "exports"
                    / _slugify(node_id, fallback="node")
                    / target_name
                ).as_posix()

            export_dir = params_payload.get("export_dir")
            if isinstance(export_dir, str) and export_dir.strip():
                params["export_dir"] = (
                    Path("outputs")
                    / "exports"
                    / _slugify(node_id, fallback="node")
                ).as_posix()

        node["params"] = params

    return payload, copied_inputs


def _write_tree_to_archive(
    archive: zipfile.ZipFile,
    *,
    source_root: Path,
    archive_root: Path,
) -> None:
    if not source_root.exists():
        return

    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if any(part == "__pycache__" for part in relative.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        archive.write(path, arcname=(archive_root / relative).as_posix())


def _build_readme(
    *,
    pipeline_name: str,
    script_name: str,
    notebook_name: str,
    pipeline_json_name: str,
    export_format: ExportFormat,
) -> str:
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    primary_artifact = script_name if export_format == "python" else notebook_name
    return "\n".join(
        [
            f"# {pipeline_name}",
            "",
            "Generated by Forge.",
            f"Generated at: `{generated_at}`",
            "",
            "## Contents",
            "",
            f"- `{primary_artifact}`: the exported runnable pipeline.",
            f"- `{pipeline_json_name}`: the same pipeline definition with local `inputs/` and `outputs/` paths.",
            "- `inputs/`: copied root input files used by the pipeline.",
            "- `outputs/`: created at runtime. Checkpoints go under `outputs/checkpoints` and side-effect exports go under `outputs/exports`.",
            "- `Forge/`, `backend/`, `blocks/`: the bundled Forge runtime and block implementations.",
            "",
            "## Run",
            "",
            "1. Install dependencies:",
            "   `pip install -r requirements.txt`",
            "",
            "2. Run the exported artifact:",
            f"   - Script: `python {script_name}`",
            f"   - Notebook: open `{notebook_name}` and run all cells",
            "",
            "3. Optional: run the bundled CLI directly against the exported pipeline JSON:",
            f"   `python -m Forge run {pipeline_json_name} --checkpoint-dir ./outputs/checkpoints --blocks-dir ./blocks`",
            "",
        ]
    )


def _comment_anchor_map(
    *,
    pipeline: dict[str, Any],
    topo_order: list[str],
) -> dict[str, list[dict[str, str]]]:
    groups = group_map(pipeline)
    first_node_index = {node_id: index for index, node_id in enumerate(topo_order)}
    anchors: list[tuple[int, float, float, int, dict[str, str]]] = []

    for comment_index, comment in enumerate(pipeline.get("comments", [])):
        group_id = str(comment.get("group_id") or comment.get("id") or "")
        if not group_id:
            continue
        node_ids = [
            str(node["id"])
            for node in pipeline.get("nodes", [])
            if group_id in [str(item) for item in node.get("group_ids", [])]
        ]
        if not node_ids:
            continue

        anchor_node_id = min(
            node_ids,
            key=lambda node_id: first_node_index.get(node_id, 10**9),
        )
        title = str(comment.get("title") or groups.get(group_id, {}).get("name") or "").strip()
        description = str(
            comment.get("description") or groups.get(group_id, {}).get("description") or ""
        ).strip()
        if not title and not description:
            continue

        width = float(comment.get("width") or 0)
        height = float(comment.get("height") or 0)
        area = width * height
        position = comment.get("position", {}) or {}
        anchors.append(
            (
                first_node_index.get(anchor_node_id, 10**9),
                -area,
                float(position.get("y") or 0),
                comment_index,
                {
                    "anchor_node_id": anchor_node_id,
                    "title": title,
                    "description": description,
                },
            )
        )

    ordered: dict[str, list[dict[str, str]]] = {}
    for _, _, _, _, payload in sorted(anchors):
        ordered.setdefault(payload["anchor_node_id"], []).append(payload)
    return ordered


def _sanitize_identifier(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    text = text.strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    return text


def _node_variable_names(node_ids: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    used: set[str] = set()
    for node_id in node_ids:
        base = _sanitize_identifier(node_id, fallback="node")
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        names[node_id] = candidate
        used.add(candidate)
    return names


def _path_expr(relative_path: str) -> CodeExpr:
    path = Path(relative_path)
    parts = [json.dumps(part) for part in path.parts]
    expr = "ROOT_DIR"
    for part in parts:
        expr = f"{expr} / {part}"
    return CodeExpr(f"str({expr})")


def _render_python_value(value: Any, *, indent: int = 0) -> str:
    if isinstance(value, CodeExpr):
        return value.source
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = [
            " " * (indent + 4) + _render_python_value(item, indent=indent + 4)
            for item in value
        ]
        return "[\n" + ",\n".join(inner) + ",\n" + " " * indent + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = []
        for key, item in value.items():
            rendered = _render_python_value(item, indent=indent + 4)
            inner.append(
                " " * (indent + 4) + f"{json.dumps(str(key))}: {rendered}"
            )
        return "{\n" + ",\n".join(inner) + ",\n" + " " * indent + "}"
    return repr(value)


def _render_comment_lines(title: str, description: str) -> list[str]:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
    if description:
        for raw_line in description.splitlines():
            text = raw_line.strip()
            lines.append(f"# {text}" if text else "#")
    return lines


def _skip_step(node_id: str, block_name: str, reason: str) -> list[str]:
    return [
        f"# Skipped {node_id}: {block_name}",
        f"# {reason}",
    ]


def _script_steps(
    *,
    pipeline: dict[str, Any],
    registry: Any,
    runner: Any,
) -> list[list[str]]:
    node_map, incoming, topo_order = runner._prepare_graph(pipeline)
    topo_order = [str(node_id) for node_id in topo_order]
    comment_map = _comment_anchor_map(pipeline=pipeline, topo_order=topo_order)
    variable_names = _node_variable_names(topo_order)
    steps: list[list[str]] = []
    emitted_nodes: set[str] = set()

    for node_id in topo_order:
        node = node_map[node_id]
        block_cls = registry.get(str(node["block"]))
        params = dict(node.get("params", {}) or {})
        if "filepath" in params and isinstance(params["filepath"], str) and not Path(params["filepath"]).is_absolute():
            params["filepath"] = _path_expr(params["filepath"])
        if "export_dir" in params and isinstance(params["export_dir"], str) and not Path(params["export_dir"]).is_absolute():
            params["export_dir"] = _path_expr(params["export_dir"])

        lines: list[str] = []
        for comment in comment_map.get(node_id, []):
            lines.extend(_render_comment_lines(comment["title"], comment["description"]))
            lines.append("")

        lines.append(f"# {node_id}: {block_cls.name}")
        notes = str(node.get("notes") or "").strip()
        if notes:
            for note_line in notes.splitlines():
                text = note_line.strip()
                lines.append(f"# Note: {text}" if text else "#")

        expected_inputs = int(block_cls.resolve_n_inputs(node.get("params") or {}))
        if len(incoming[node_id]) < expected_inputs:
            lines.extend(
                _skip_step(
                    node_id,
                    block_cls.name,
                    (
                        f"Native Forge skips this node because it expects {expected_inputs} "
                        f"input(s) and only {len(incoming[node_id])} edge(s) are connected."
                    ),
                )
            )
            steps.append(lines)
            continue

        try:
            parent_refs = runner._sorted_parent_refs(
                node_id,
                incoming,
                expected_slots=expected_inputs,
            )
        except ValueError as exc:
            lines.extend(
                _skip_step(
                    node_id,
                    block_cls.name,
                    f"Native Forge skips this node because its input ordering is unresolved: {exc}",
                )
            )
            steps.append(lines)
            continue

        missing_parents = [
            parent_ref.source_node_id
            for parent_ref in parent_refs
            if parent_ref.source_node_id not in emitted_nodes
        ]
        if missing_parents:
            lines.extend(
                _skip_step(
                    node_id,
                    block_cls.name,
                    "Native Forge skips this node because required upstream outputs "
                    f"were not produced: {', '.join(missing_parents)}.",
                )
            )
            steps.append(lines)
            continue

        input_exprs: list[str] = []
        for parent_ref in parent_refs:
            parent_var = variable_names[parent_ref.source_node_id]
            if parent_ref.source_output_handle == "output_0":
                input_exprs.append(f"{parent_var}.output()")
            else:
                input_exprs.append(
                    f'{parent_var}.output("{parent_ref.source_output_handle}")'
                )

        call_lines = [
            f'{variable_names[node_id]} = runtime.run_block(',
            f'    node_id={json.dumps(node_id)},',
            f'    block_key={json.dumps(str(node["block"]))},',
            f'    params={_render_python_value(params, indent=4)},',
        ]
        if input_exprs:
            rendered_inputs = "[\n" + "".join(
                f"        {expr},\n" for expr in input_exprs
            ) + "    ]"
            call_lines.append(f"    inputs={rendered_inputs},")
        call_lines.append(")")
        lines.extend(call_lines)
        steps.append(lines)
        emitted_nodes.add(node_id)

    return steps


def _build_python_script(
    *,
    pipeline: dict[str, Any],
    registry: Any,
    runner: Any,
    script_name: str,
) -> str:
    pipeline_name = str(pipeline.get("name", Path(script_name).stem))
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    header_lines = [
        '"""',
        f"Generated by Forge for pipeline: {pipeline_name}",
        "",
        "Run from this directory after installing requirements:",
        f"    python {script_name}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "ROOT_DIR = Path(__file__).resolve().parent",
        "if str(ROOT_DIR) not in sys.path:",
        "    sys.path.insert(0, str(ROOT_DIR))",
        "",
        "from Forge.export_runtime import ExportRuntime",
        "",
        f'PIPELINE_NAME = {json.dumps(pipeline_name)}',
        f'GENERATED_AT = {json.dumps(generated_at)}',
        "",
        "",
        "def main() -> dict[str, object]:",
        "    runtime = ExportRuntime(root_dir=ROOT_DIR, pipeline_name=PIPELINE_NAME)",
        "",
    ]

    body_lines: list[str] = []
    for step in _script_steps(pipeline=pipeline, registry=registry, runner=runner):
        if body_lines:
            body_lines.append("")
        body_lines.extend(f"    {line}" if line else "" for line in step)

    footer_lines = [
        "",
        "    summary = runtime.finish()",
        '    print(f"Pipeline: {summary[\'pipeline_name\']}")',
        '    print(f"Executed nodes: {len(summary[\'executed_nodes\'])}")',
        '    print(f"Reused nodes: {len(summary[\'reused_nodes\'])}")',
        '    print(f"Outputs: {ROOT_DIR / \'outputs\'}")',
        "    return summary",
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ]
    return "\n".join(header_lines + body_lines + footer_lines)


def _build_notebook(
    *,
    pipeline: dict[str, Any],
    registry: Any,
    runner: Any,
) -> str:
    pipeline_name = str(pipeline.get("name", "Forge Export"))
    setup_cell = [
        "from __future__ import annotations",
        "",
        "import sys",
        "from pathlib import Path",
        "",
        'ROOT_DIR = Path.cwd()',
        "if str(ROOT_DIR) not in sys.path:",
        "    sys.path.insert(0, str(ROOT_DIR))",
        "",
        "from Forge.export_runtime import ExportRuntime",
        "",
        f'PIPELINE_NAME = {json.dumps(pipeline_name)}',
        "runtime = ExportRuntime(root_dir=ROOT_DIR, pipeline_name=PIPELINE_NAME)",
    ]

    cells = [
        _markdown_cell(
            [
                f"# {pipeline_name}",
                "",
                "Generated by Forge. Run the setup cell first, then run the pipeline cells in order.",
            ]
        ),
        _code_cell(setup_cell),
    ]

    for step in _script_steps(pipeline=pipeline, registry=registry, runner=runner):
        cells.append(_code_cell(step))

    cells.append(
        _code_cell(
            [
                "summary = runtime.finish()",
                "summary",
            ]
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2)


def _markdown_cell(lines: list[str]) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }


def _code_cell(lines: list[str]) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }
