from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Iterator

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    InsufficientInputs,
    block_param,
)


_WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _coerce_figsize(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            width = float(value[0])
            height = float(value[1])
            if width > 0 and height > 0:
                return (width, height)
        except Exception:
            pass
    return default


def _require_non_empty_frame(data: pd.DataFrame, block_name: str) -> None:
    if data.empty:
        raise BlockValidationError(f"{block_name} received empty data.")


def _require_column(
    data: pd.DataFrame,
    column_name: str,
    block_name: str,
    label: str,
) -> str:
    clean = str(column_name).strip()
    if not clean:
        raise BlockValidationError(f"{label} is required.")
    if clean not in data.columns:
        raise BlockValidationError(f"Column '{clean}' not found.")
    return clean


def _normalize_color_mode(value: Any, block_name: str) -> str:
    mode = str(value or "auto").strip().lower() or "auto"
    if mode not in {"auto", "numeric", "categorical"}:
        raise BlockValidationError(
            f"{block_name} color_mode must be one of: auto, numeric, categorical."
        )
    return mode


def _resolve_color_series(
    plot_df: pd.DataFrame,
    color_column: str,
) -> tuple[str, pd.Series]:
    if color_column == "index":
        plot_df["_color"] = pd.Series(plot_df.index, index=plot_df.index)
        return "_color", plot_df["_color"]
    return color_column, plot_df[color_column]


def _iter_groups_with_null_first(
    plot_df: pd.DataFrame,
    group_key: str,
) -> Iterator[tuple[Any, pd.DataFrame]]:
    null_mask = plot_df[group_key].isna()
    if bool(null_mask.any()):
        yield np.nan, plot_df.loc[null_mask]

    non_null_df = plot_df.loc[~null_mask]
    if non_null_df.empty:
        return

    yield from non_null_df.groupby(group_key, dropna=False)


def _wants_numeric_color_scale(series: pd.Series, color_mode: str) -> bool:
    if color_mode == "numeric":
        return True
    if color_mode == "categorical":
        return False
    return bool(is_numeric_dtype(series))


def _normalize_plot_label(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _parse_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            text = text[1:-1]
        parts = [part.strip().strip("'").strip('"') for part in text.split(",")]
        return [part for part in parts if part]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _parse_json_object_list(
    value: Any,
    *,
    block_name: str,
    param_name: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        specs: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise BlockValidationError(
                    f"{block_name} requires every {param_name} item to be an object."
                )
            specs.append(dict(item))
        return specs

    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise BlockValidationError(
            f"{block_name} could not parse {param_name} as JSON."
        ) from exc

    if isinstance(parsed, dict):
        return [dict(parsed)]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return [dict(item) for item in parsed]
    raise BlockValidationError(
        f"{block_name} expects {param_name} to be a JSON object or list of objects."
    )


def _parse_json_mapping(
    value: Any,
    *,
    block_name: str,
    param_name: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}

    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise BlockValidationError(
            f"{block_name} could not parse {param_name} as JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise BlockValidationError(
            f"{block_name} expects {param_name} to be a JSON object."
        )
    return {str(key): item for key, item in parsed.items()}


def _draw_reference_bands(
    ax: Any,
    specs: list[dict[str, Any]],
    *,
    block_name: str,
) -> int:
    bands_drawn = 0
    for spec in specs:
        axis = str(spec.get("axis", "x") or "x").strip().lower()
        if axis not in {"x", "y"}:
            raise BlockValidationError(f"{block_name} band axis must be 'x' or 'y'.")
        try:
            start = float(spec["start"])
            end = float(spec["end"])
        except Exception as exc:
            raise BlockValidationError(
                f"{block_name} band specs require numeric start and end."
            ) from exc
        color = str(spec.get("color", "#dddddd") or "#dddddd")
        alpha = float(spec.get("alpha", 0.2))
        zorder = float(spec.get("zorder", 0.5))
        label = str(spec.get("label", "") or "").strip() or None
        if axis == "x":
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=zorder, label=label)
        else:
            ax.axhspan(start, end, color=color, alpha=alpha, zorder=zorder, label=label)
        bands_drawn += 1
    return bands_drawn


def _draw_reference_lines(
    ax: Any,
    specs: list[dict[str, Any]],
    *,
    block_name: str,
) -> int:
    lines_drawn = 0
    for spec in specs:
        axis = str(spec.get("axis", "x") or "x").strip().lower()
        if axis not in {"x", "y"}:
            raise BlockValidationError(f"{block_name} line axis must be 'x' or 'y'.")
        try:
            value = float(spec["value"])
        except Exception as exc:
            raise BlockValidationError(
                f"{block_name} line specs require a numeric value."
            ) from exc
        color = str(spec.get("color", "0.6") or "0.6")
        linestyle = str(spec.get("linestyle", "--") or "--")
        linewidth = float(spec.get("linewidth", 1.5))
        alpha = float(spec.get("alpha", 1.0))
        zorder = float(spec.get("zorder", 1.5))
        label = str(spec.get("label", "") or "").strip() or None
        if axis == "x":
            ax.axvline(
                value,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                label=label,
            )
        else:
            ax.axhline(
                value,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                label=label,
            )
        lines_drawn += 1
    return lines_drawn


def _draw_text_annotations(
    ax: Any,
    specs: list[dict[str, Any]],
    *,
    block_name: str,
) -> int:
    annotations_drawn = 0
    for spec in specs:
        try:
            x = float(spec["x"])
            y = float(spec["y"])
        except Exception as exc:
            raise BlockValidationError(
                f"{block_name} annotation specs require numeric x and y."
            ) from exc
        text = str(spec.get("text", "") or "").strip()
        if not text:
            raise BlockValidationError(
                f"{block_name} annotation specs require non-empty text."
            )
        ax.text(
            x,
            y,
            text,
            color=str(spec.get("color", "black") or "black"),
            fontsize=float(spec.get("fontsize", 10.0)),
            ha=str(spec.get("ha", "left") or "left"),
            va=str(spec.get("va", "center") or "center"),
            zorder=float(spec.get("zorder", 5.0)),
        )
        annotations_drawn += 1
    return annotations_drawn


def _format_text_template(
    template: Any,
    context: dict[str, Any],
    *,
    block_name: str,
    param_name: str,
) -> str:
    text = str(template or "").strip()
    if not text:
        return ""
    try:
        rendered = text.format(**context)
    except Exception as exc:
        raise BlockValidationError(
            f"{block_name} could not format {param_name} with the available context."
        ) from exc
    return rendered.replace("\\n", "\n").replace("\\t", "\t")


def _require_two_dataframes(
    data: Any,
    block_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise BlockValidationError(f"{block_name} expects [nodes_df, edges_df].")

    nodes_df, edges_df = data
    if not isinstance(nodes_df, pd.DataFrame) or not isinstance(edges_df, pd.DataFrame):
        raise BlockValidationError(
            f"{block_name} expects both inputs to be DataFrames."
        )
    return nodes_df, edges_df


def _prepare_scatter_plot_df(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    size_column: str,
    marker_size: float,
    block_name: str,
) -> pd.DataFrame:
    plot_df = data.copy()
    plot_df["_x"] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df["_y"] = pd.to_numeric(plot_df[y_column], errors="coerce")
    if size_column:
        size_values = pd.to_numeric(plot_df[size_column], errors="coerce")
        size_values = size_values.fillna(float(marker_size))
        plot_df["_size"] = np.clip(
            size_values.to_numpy(dtype=float), a_min=1.0, a_max=None
        )
    else:
        plot_df["_size"] = float(marker_size)
    plot_df = plot_df.dropna(subset=["_x", "_y"])
    if plot_df.empty:
        raise BlockValidationError(
            f"{block_name} found no numeric rows for '{x_column}' and '{y_column}'."
        )
    return plot_df


def _render_scatter_points(
    fig: Any,
    ax: Any,
    plot_df: pd.DataFrame,
    *,
    color_column: str,
    color_mode: str,
    cmap: str,
    alpha: float,
    block_name: str,
    default_color: str | None = None,
    zorder: float | None = None,
) -> str | None:
    x_values = plot_df["_x"].to_numpy(dtype=float)
    y_values = plot_df["_y"].to_numpy(dtype=float)
    size_values = plot_df["_size"].to_numpy(dtype=float)

    scatter_kwargs: dict[str, Any] = {}
    if zorder is not None:
        scatter_kwargs["zorder"] = zorder

    color_mode_used: str | None = None
    if color_column:
        normalized_color_mode = _normalize_color_mode(color_mode, block_name)
        color_group_key, color_values_raw = _resolve_color_series(plot_df, color_column)
        color_values_raw = plot_df[color_group_key]
        numeric_color = pd.to_numeric(color_values_raw, errors="coerce")
        non_null_color = color_values_raw.notna()
        numeric_for_non_null = numeric_color[non_null_color]
        can_use_numeric_scale = bool(non_null_color.any()) and bool(
            numeric_for_non_null.notna().all()
        )
        wants_numeric_scale = _wants_numeric_color_scale(
            color_values_raw,
            normalized_color_mode,
        )
        if normalized_color_mode == "numeric" and not can_use_numeric_scale:
            raise BlockValidationError(
                f"{block_name} could not interpret '{color_column}' as numeric colors."
            )

        if wants_numeric_scale and can_use_numeric_scale:
            color_mode_used = "numeric"
            color_values = numeric_color.to_numpy(dtype=float)
            finite_color_mask = np.isfinite(color_values)
            if np.any(finite_color_mask):
                scatter = ax.scatter(
                    x_values[finite_color_mask],
                    y_values[finite_color_mask],
                    c=color_values[finite_color_mask],
                    s=size_values[finite_color_mask],
                    alpha=alpha,
                    cmap=cmap,
                    **scatter_kwargs,
                )
                fig.colorbar(scatter, ax=ax, label=color_column)
            if np.any(~finite_color_mask):
                ax.scatter(
                    x_values[~finite_color_mask],
                    y_values[~finite_color_mask],
                    s=size_values[~finite_color_mask],
                    alpha=alpha,
                    color="lightgray",
                    label=f"{color_column}: null/non-finite",
                    **scatter_kwargs,
                )
                ax.legend()
        else:
            color_mode_used = "categorical"
            n_groups = int(plot_df[color_group_key].nunique(dropna=False))
            if n_groups > 40:
                categories = plot_df[color_group_key].astype("string").fillna("null")
                codes, _ = pd.factorize(categories, sort=True)
                scatter = ax.scatter(
                    x_values,
                    y_values,
                    c=codes.astype(float),
                    s=size_values,
                    alpha=alpha,
                    cmap=cmap,
                    **scatter_kwargs,
                )
                colorbar = fig.colorbar(scatter, ax=ax)
                colorbar.set_label(f"{color_column} (category code)")
            else:
                for group_value, group_df in _iter_groups_with_null_first(
                    plot_df,
                    color_group_key,
                ):
                    label = "null" if pd.isna(group_value) else str(group_value)
                    ax.scatter(
                        group_df["_x"].to_numpy(dtype=float),
                        group_df["_y"].to_numpy(dtype=float),
                        s=group_df["_size"].to_numpy(dtype=float),
                        alpha=alpha,
                        label=label,
                        **scatter_kwargs,
                    )
                ax.legend(title=color_column)
    else:
        if default_color is not None:
            scatter_kwargs["color"] = default_color
        ax.scatter(
            x_values,
            y_values,
            s=size_values,
            alpha=alpha,
            **scatter_kwargs,
        )

    return color_mode_used


def _normalize_plot_title(value: Any, fallback: str = "plot") -> str:
    text = str(value or "").strip()
    return text or fallback


def _safe_plot_filename(plot_title: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", plot_title).strip()
    safe = re.sub(r"\s+", " ", safe).rstrip(". ")
    if not safe:
        safe = "plot"
    if safe.upper() in _WINDOWS_RESERVED_FILENAMES:
        safe = f"_{safe}"
    return safe


def _resolve_export_dir(export_dir: str) -> Path:
    """Resolve a visualization export directory against FORGE_WORKSPACE_DIR."""
    directory = Path(export_dir)
    if not directory.is_absolute():
        workspace_dir = os.environ.get("FORGE_WORKSPACE_DIR", "").strip()
        if workspace_dir:
            directory = Path(workspace_dir) / directory
    return directory


def _export_visualization(plot: Any, params: Any, plot_title: str) -> str | None:
    if not bool(getattr(params, "export_enabled", False)):
        return None

    export_dir = str(getattr(params, "export_dir", "") or "").strip()
    if not export_dir:
        raise BlockValidationError(
            "export_dir is required when export_enabled is true."
        )

    directory = _resolve_export_dir(export_dir)
    directory.mkdir(parents=True, exist_ok=True)
    export_basename = str(getattr(params, "export_basename", "") or "").strip()
    filename = _safe_plot_filename(export_basename or plot_title)

    if hasattr(plot, "write_html"):
        path = directory / f"{filename}.html"
        try:
            plot.write_html(
                str(path),
                full_html=True,
                include_plotlyjs=True,
            )  # type: ignore[call-arg]
        except Exception as exc:
            raise BlockValidationError(
                f"Failed to export Plotly visualization to '{path}'."
            ) from exc
        return str(path)

    if hasattr(plot, "savefig"):
        path = directory / f"{filename}.png"
        try:
            plot.savefig(path, bbox_inches="tight")
        except Exception as exc:
            raise BlockValidationError(
                f"Failed to export visualization image to '{path}'."
            ) from exc
        return str(path)

    raise BlockValidationError(
        f"Unsupported visualization artifact type: {type(plot)!r}"
    )


def _visual_output(
    data: pd.DataFrame,
    plot: Any,
    params: Any,
    plot_title: str,
    metadata: dict[str, Any] | None = None,
) -> BlockOutput:
    normalized_title = _normalize_plot_title(plot_title)
    next_metadata = dict(metadata or {})
    next_metadata["plot_title"] = normalized_title
    exported_path = _export_visualization(plot, params, normalized_title)
    if exported_path is not None:
        next_metadata["exported_path"] = exported_path
    return BlockOutput(data=data, images=[plot], metadata=next_metadata)


class VisualizationParams(BlockParams):
    export_enabled: bool = block_param(
        False,
        description="Whether to export the rendered visualization to a file every time this node executes.",
        example=True,
    )
    export_dir: str | None = block_param(
        None,
        description="Destination directory for exported visuals. Relative paths resolve from the Forge workspace, and Forge writes `{Plot Title}.{extension}` into this folder when export is enabled.",
        example="C:\\Users\\you\\exports",
        browse_mode="directory",
    )


class VisualizationBlock(BaseBlock):
    @classmethod
    def should_force_execute(cls, params: dict[str, Any] | None = None) -> bool:
        return super().should_force_execute(params) or bool(
            isinstance(params, dict) and params.get("export_enabled")
        )


class ScatterPlotParams(VisualizationParams):
    x_column: str = block_param(
        description="Numeric x-axis column.",
        example="feature_a",
    )
    y_column: str = block_param(
        description="Numeric y-axis column.",
        example="feature_b",
    )
    color_column: str = block_param(
        "",
        description="Optional column used for color mapping or categories. Use 'index' to color by row index.",
        example="cluster_id",
    )
    color_mode: str = block_param(
        "auto",
        description="How to interpret color_column. Use auto, numeric, or categorical.",
        example="categorical",
    )
    size_column: str = block_param(
        "",
        description="Optional numeric column for point sizes.",
        example="feature_c",
    )
    marker_size: float = block_param(
        40.0,
        description="Fallback marker size used when size_column is blank or contains nulls.",
        example=40.0,
    )
    alpha: float = block_param(
        0.75,
        description="Opacity for plotted points.",
        example=0.75,
    )
    cmap: str = block_param(
        "viridis",
        description="Matplotlib color map used for numeric colors and dense categorical fallback.",
        example="tab10",
    )
    figsize: list[float] = [10.0, 6.0]
    title: str = "Scatter Plot"


class ClusterHeatmap(VisualizationBlock):
    name = "Cluster Profile Heatmap"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a heatmap of cluster-level numeric profiles."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame + Image"]
    usage_notes = [
        "The input should contain row-level numeric features plus a cluster label column.",
        "This block computes per-cluster means internally before plotting the heatmap.",
        "The incoming DataFrame passes through unchanged; the image is a side artifact.",
    ]

    class Params(VisualizationParams):
        cluster_column: str = "cluster_id"
        cmap: str = "RdBu_r"
        z_score: bool = True
        figsize: list[float] = [12.0, 6.0]

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("ClusterHeatmap requires params.")
        if params.cluster_column not in data.columns:
            raise BlockValidationError(f"Column '{params.cluster_column}' not found.")

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric_cols = data.select_dtypes(include="number").columns.drop(
            params.cluster_column,
            errors="ignore",
        )
        if len(numeric_cols) == 0:
            raise BlockValidationError("No numeric columns available for heatmap.")

        profiles = data.groupby(params.cluster_column)[numeric_cols].mean()
        if params.z_score:
            profiles = (profiles - profiles.mean()) / profiles.std().replace(0, 1.0)

        figsize = (
            tuple(params.figsize) if isinstance(params.figsize, list) else (12.0, 6.0)
        )
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            profiles, cmap=params.cmap, center=0 if params.z_score else None, ax=ax
        )
        plot_title = "Cluster Profiles"
        ax.set_title(plot_title)
        fig.tight_layout()

        return _visual_output(data, fig, params, plot_title)


class MatrixHeatmap(VisualizationBlock):
    name = "Matrix Heatmap"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a heatmap for the incoming numeric matrix."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]

    class Params(VisualizationParams):
        cmap: str = "vlag"
        center: float | None = 0.0
        figsize: list[float] = [16.0, 6.0]
        title: str = "Matrix Heatmap"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MatrixHeatmap requires params.")
        if data.empty:
            raise BlockValidationError("MatrixHeatmap received empty data.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=tuple(params.figsize))
        sns.heatmap(
            data.select_dtypes(include="number"),
            cmap=params.cmap,
            center=params.center,
            ax=ax,
        )
        plot_title = _normalize_plot_title(params.title, "Matrix Heatmap")
        ax.set_title(plot_title)
        fig.tight_layout()
        return _visual_output(data, fig, params, plot_title)


class MatrixHistogram(VisualizationBlock):
    name = "Matrix Histogram"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a histogram for one numeric column from the incoming matrix."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]

    class Params(VisualizationParams):
        column_name: str = block_param(
            description="Column name to plot.",
            example="feature_a",
        )
        bucket_size: float | None = block_param(
            None,
            description="Histogram bucket width. Leave unset to auto-target about 25 buckets.",
            example=0.5,
        )
        skip_nulls: bool = block_param(
            True,
            description="Whether to ignore null values when determining bucket ranges.",
            example=True,
        )
        plot_title: str = block_param(
            "",
            description="Title for the histogram plot.",
            example="Preview Plot",
        )

    def _coerce_bucket_size(self, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            value = text
        try:
            bucket_size = float(value)  # pyright: ignore[reportArgumentType]
        except Exception as exc:
            raise BlockValidationError(
                "bucket_size must be a numeric value when provided."
            ) from exc
        if not np.isfinite(bucket_size) or bucket_size <= 0:
            raise BlockValidationError("bucket_size must be > 0 when provided.")
        return bucket_size

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MatrixHistogram requires params.")
        if data.empty:
            raise BlockValidationError("MatrixHistogram received empty data.")

        column_name = str(getattr(params, "column_name", "")).strip()
        if not column_name:
            raise BlockValidationError("column_name is required.")
        if column_name not in data.columns:
            raise BlockValidationError(f"Column '{column_name}' not found.")

        values = pd.to_numeric(data[column_name], errors="coerce")
        if bool(getattr(params, "skip_nulls", True)):
            values = values.dropna()
        # Always drop non-finite values (inf/-inf); they cause NaN/inf in bin calculations
        values = values[np.isfinite(values)]
        if values.empty:
            raise BlockValidationError(
                f"Column '{column_name}' has no finite numeric values after dropping null/non-numeric entries."
            )

        min_value = float(values.min())
        max_value = float(values.max())
        value_range = max_value - min_value

        bucket_size = self._coerce_bucket_size(getattr(params, "bucket_size", None))
        auto_bucket = bucket_size is None
        if auto_bucket:
            if value_range <= 0:
                bucket_size = 1.0
            else:
                target_buckets = 25
                unique_count = int(values.nunique(dropna=True))
                n_buckets = max(1, min(target_buckets, unique_count))
                bucket_size = value_range / max(n_buckets, 1)

        if value_range <= 0 or bucket_size is None or bucket_size <= 0:
            # Edge case: all values are the same or bucket_size invalid
            bin_edges = np.array([min_value - 0.5, max_value + 0.5], dtype=float)
        else:
            # Ensure bucket_size is not too small to avoid arange errors
            n_bins = int(np.ceil(value_range / bucket_size))
            if n_bins < 1:
                n_bins = 1
            bin_edges = np.linspace(min_value, max_value, n_bins + 1, dtype=float)
            if bin_edges.size < 2:
                bin_edges = np.array([min_value, max_value], dtype=float)
            if bin_edges[-1] < max_value:
                bin_edges = np.append(bin_edges, max_value)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        ax.hist(
            values.to_numpy(dtype=float),
            bins=bin_edges,  # pyright: ignore[reportArgumentType]
            edgecolor="black",
            alpha=0.75,
        )
        plot_title = _normalize_plot_title(
            getattr(params, "plot_title", ""),
            f"Histogram of {column_name}",
        )
        ax.set_title(plot_title)
        ax.set_xlabel(column_name)
        ax.set_ylabel("Count")
        fig.tight_layout()

        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "column_name": column_name,
                "bucket_size": float(bucket_size) if bucket_size is not None else None,
                "n_buckets": int(max(len(bin_edges) - 1, 1)),
                "auto_bucket_size": auto_bucket,
            },
        )


class MatrixBarChart(VisualizationBlock):
    name = "Matrix Bar Chart"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a bar chart from one x column and one numeric y column."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]

    class Params(VisualizationParams):
        x_column: str = block_param(description="Column for x-axis labels.")
        y_column: str = block_param(description="Numeric column for bar heights.")
        top_n: int = 30
        sort_by_y: bool = True
        ascending: bool = False
        figsize: list[float] = [12.0, 6.0]
        title: str = "Bar Chart"
        color: str = "steelblue"
        rotation: float = 45.0

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MatrixBarChart requires params.")
        _require_non_empty_frame(data, "MatrixBarChart")

        x_column = _require_column(data, params.x_column, "MatrixBarChart", "x_column")
        y_column = _require_column(data, params.y_column, "MatrixBarChart", "y_column")

        y_values = pd.to_numeric(data[y_column], errors="coerce")
        plot_df = pd.DataFrame({"_x": data[x_column].astype(str), "_y": y_values})
        plot_df = plot_df.dropna(subset=["_y"])
        if plot_df.empty:
            raise BlockValidationError(
                f"Column '{y_column}' has no numeric values after dropping null/non-numeric entries."
            )

        if bool(params.sort_by_y):
            plot_df = plot_df.sort_values("_y", ascending=bool(params.ascending))
        top_n = int(getattr(params, "top_n", 0))
        if top_n > 0:
            plot_df = plot_df.head(top_n)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=_coerce_figsize(params.figsize, (12.0, 6.0)))
        ax.bar(plot_df["_x"], plot_df["_y"], color=str(params.color))
        plot_title = _normalize_plot_title(params.title, f"{y_column} by {x_column}")
        ax.set_title(plot_title)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.tick_params(axis="x", rotation=float(params.rotation))
        fig.tight_layout()
        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "n_rows_plotted": int(plot_df.shape[0]),
            },
        )


class MatrixLineChart(VisualizationBlock):
    name = "Matrix Line Chart"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a line chart from one x column and one numeric y column."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]

    class Params(VisualizationParams):
        x_column: str = ""
        y_column: str = block_param(description="Numeric column for y-axis values.")
        group_column: str = ""
        sort_by_x: bool = True
        marker: str = "o"
        figsize: list[float] = [12.0, 6.0]
        title: str = "Line Chart"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MatrixLineChart requires params.")
        _require_non_empty_frame(data, "MatrixLineChart")

        x_column_raw = str(params.x_column).strip()
        x_column = x_column_raw if x_column_raw else None
        y_column = _require_column(data, params.y_column, "MatrixLineChart", "y_column")
        group_column = str(params.group_column).strip()
        if group_column and group_column not in data.columns:
            raise BlockValidationError(f"Column '{group_column}' not found.")
        if x_column and x_column not in data.columns:
            raise BlockValidationError(f"Column '{x_column}' not found.")

        plot_df = data.copy()
        plot_df["_y"] = pd.to_numeric(plot_df[y_column], errors="coerce")
        if x_column:
            plot_df = plot_df.dropna(subset=[x_column, "_y"])
            if bool(params.sort_by_x):
                plot_df = plot_df.sort_values(x_column)
            x_label = x_column
            x_key = x_column
        else:
            # Missing x_column: use ascending index order and equidistant x positions.
            plot_df = plot_df.sort_index().dropna(subset=["_y"])
            plot_df["_x"] = np.arange(plot_df.shape[0], dtype=float)
            x_label = "index"
            x_key = "_x"

        if plot_df.empty:
            raise BlockValidationError(
                f"MatrixLineChart found no plottable rows after cleaning '{y_column}'."
            )

        marker = str(params.marker).strip() or None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=_coerce_figsize(params.figsize, (12.0, 6.0)))
        if group_column:
            for group_value, group_df in plot_df.groupby(group_column, dropna=False):
                series_df = group_df
                if x_column and bool(params.sort_by_x):
                    series_df = group_df.sort_values(x_column)
                label = "null" if pd.isna(group_value) else str(group_value)
                ax.plot(
                    series_df[x_key].tolist(),
                    series_df["_y"].to_numpy(dtype=float),
                    marker=marker,
                    label=label,
                )
            ax.legend(title=group_column)
        else:
            ax.plot(
                plot_df[x_key].tolist(),
                plot_df["_y"].to_numpy(dtype=float),
                marker=marker,
            )

        plot_title = _normalize_plot_title(params.title, f"{y_column} over {x_label}")
        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_column)
        fig.tight_layout()
        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "group_column": group_column or None,
                "x_mode": "column" if x_column else "index_position",
                "n_rows_plotted": int(plot_df.shape[0]),
            },
        )


class MatrixScatterPlot(VisualizationBlock):
    name = "Matrix Scatter Plot"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a 2D scatter plot using numeric x and y columns."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]
    usage_notes = [
        "`color_mode='auto'` treats numeric dtype columns as continuous and string/object columns as categorical.",
        "Set `color_mode='categorical'` when cluster IDs are numeric labels but should render as a discrete legend.",
        "When there are more than 40 categories, Forge collapses the legend into a category-code colorbar.",
    ]
    presets = [
        {
            "id": "cluster_scatter",
            "label": "Cluster Scatter",
            "description": "Scatter plot with cluster assignments encoded by color.",
            "params": {
                "x_column": "feature_a",
                "y_column": "feature_b",
                "color_column": "cluster_id",
                "color_mode": "categorical",
                "title": "Cluster Scatter Plot",
            },
        }
    ]

    class Params(ScatterPlotParams):
        pass

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("MatrixScatterPlot requires params.")
        _require_non_empty_frame(data, "MatrixScatterPlot")

        x_column = _require_column(
            data, params.x_column, "MatrixScatterPlot", "x_column"
        )
        y_column = _require_column(
            data, params.y_column, "MatrixScatterPlot", "y_column"
        )
        color_column = str(params.color_column).strip()
        size_column = str(params.size_column).strip()
        if (
            color_column
            and color_column != "index"
            and color_column not in data.columns
        ):
            raise BlockValidationError(f"Column '{color_column}' not found.")
        if size_column and size_column not in data.columns:
            raise BlockValidationError(f"Column '{size_column}' not found.")

        plot_df = _prepare_scatter_plot_df(
            data,
            x_column,
            y_column,
            size_column,
            float(params.marker_size),
            "MatrixScatterPlot",
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=_coerce_figsize(params.figsize, (10.0, 6.0)))
        color_mode_used = _render_scatter_points(
            fig,
            ax,
            plot_df,
            color_column=color_column,
            color_mode=str(getattr(params, "color_mode", "auto")),
            cmap=str(params.cmap),
            alpha=float(params.alpha),
            block_name="MatrixScatterPlot",
        )

        plot_title = _normalize_plot_title(params.title, f"{y_column} vs {x_column}")
        ax.set_title(plot_title)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        fig.tight_layout()
        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column or None,
                "color_mode_used": color_mode_used,
                "size_column": size_column or None,
                "n_rows_plotted": int(plot_df.shape[0]),
            },
        )


class HighlightedScatterPlot(VisualizationBlock):
    name = "Highlighted Scatter Plot"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a scatter plot with muted background points and highlighted named subsets."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]
    usage_notes = [
        "Rows whose highlight_column value is not listed in highlight_values are rendered as muted background points.",
        "guide_lines, guide_bands, and annotations accept JSON objects or JSON lists of objects for advanced plot decoration.",
        "Set label_highlights=true to draw text labels for every highlighted point using label_column or highlight_column.",
    ]

    class Params(VisualizationParams):
        export_basename: str = block_param(
            "",
            description="Optional export filename stem used instead of the plot title when export is enabled.",
            example="highlighted_scatter",
        )
        x_column: str = block_param(
            description="Numeric x-axis column.",
            example="x_value",
        )
        y_column: str = block_param(
            description="Numeric y-axis column.",
            example="y_value",
        )
        highlight_column: str = block_param(
            description="Column whose values determine which rows are highlighted.",
            example="category",
        )
        highlight_values: str = block_param(
            description="Comma-separated values, or a JSON string list, defining the highlighted subsets in display order.",
            example="Group A,Group B",
        )
        highlight_colors: str = block_param(
            "",
            description="Optional comma-separated colors, or a JSON string list of colors, aligned to highlight_values.",
            example="#d62728,#ff7f0e",
        )
        label_column: str = block_param(
            "",
            description="Optional text column used when label_highlights is true. Defaults to highlight_column.",
            example="label",
        )
        size_column: str = block_param(
            "",
            description="Optional numeric column used for marker sizes.",
            example="point_size",
        )
        marker_size: float = block_param(
            40.0,
            description="Fallback marker size used when size_column is blank or contains nulls.",
            example=40.0,
        )
        background_color: str = block_param(
            "lightgray",
            description="Color used for non-highlighted background points.",
            example="lightgray",
        )
        background_alpha: float = block_param(
            0.35,
            description="Opacity for non-highlighted background points.",
            example=0.35,
        )
        highlight_alpha: float = block_param(
            0.9,
            description="Opacity for highlighted points.",
            example=0.9,
        )
        show_legend: bool = block_param(
            True,
            description="Whether to show a legend for highlighted groups and labeled guide lines.",
            example=True,
        )
        show_highlight_legend: bool = block_param(
            True,
            description="Whether highlighted groups should contribute legend entries when show_legend is true.",
            example=True,
        )
        label_highlights: bool = block_param(
            False,
            description="Whether to draw text labels for each highlighted point.",
            example=True,
        )
        label_fontsize: float = block_param(
            10.0,
            description="Font size used when label_highlights is true.",
            example=10.0,
        )
        x_label: str = block_param(
            "",
            description="Optional x-axis label override. Leave blank to use x_column.",
            example="Reference Value",
        )
        y_label: str = block_param(
            "",
            description="Optional y-axis label override. Leave blank to use y_column.",
            example="Comparison Value",
        )
        identity_line: bool = block_param(
            False,
            description="Whether to draw an x=y diagonal reference line spanning the plotted extent.",
            example=True,
        )
        identity_line_color: str = block_param(
            "0.7",
            description="Color used for the x=y diagonal reference line.",
            example="0.7",
        )
        identity_line_style: str = block_param(
            "--",
            description="Line style used for the x=y diagonal reference line.",
            example="--",
        )
        guide_lines: str = block_param(
            "",
            description="Optional JSON object or JSON list of objects describing reference lines with keys such as axis, value, color, linestyle, linewidth, alpha, and label.",
            example='[{"axis":"x","value":0.0,"color":"0.8","linestyle":"-","linewidth":1.5}]',
        )
        guide_bands: str = block_param(
            "",
            description="Optional JSON object or JSON list of objects describing shaded guide bands with keys such as axis, start, end, color, alpha, and label.",
            example='[{"axis":"x","start":-1.0,"end":0.0,"color":"#fde7e7","alpha":0.35}]',
        )
        annotations: str = block_param(
            "",
            description="Optional JSON object or JSON list of objects describing free-text annotations with keys such as x, y, text, color, fontsize, ha, and va.",
            example='[{"x":-0.9,"y":3.9,"text":"Reference note","color":"#ff6b6b"}]',
        )
        figsize: list[float] = [10.0, 6.0]
        title: str = "Highlighted Scatter Plot"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("HighlightedScatterPlot requires params.")
        _require_non_empty_frame(data, "HighlightedScatterPlot")

        x_column = _require_column(
            data, params.x_column, "HighlightedScatterPlot", "x_column"
        )
        y_column = _require_column(
            data, params.y_column, "HighlightedScatterPlot", "y_column"
        )
        highlight_column = _require_column(
            data,
            params.highlight_column,
            "HighlightedScatterPlot",
            "highlight_column",
        )
        size_column = str(params.size_column).strip()
        if size_column and size_column not in data.columns:
            raise BlockValidationError(f"Column '{size_column}' not found.")

        label_column = (
            str(getattr(params, "label_column", "") or "").strip() or highlight_column
        )
        if bool(params.label_highlights) and label_column not in data.columns:
            raise BlockValidationError(f"Column '{label_column}' not found.")

        highlight_values = _parse_text_list(params.highlight_values)
        if not highlight_values:
            raise BlockValidationError("highlight_values is required.")

        plot_df = _prepare_scatter_plot_df(
            data,
            x_column,
            y_column,
            size_column,
            float(params.marker_size),
            "HighlightedScatterPlot",
        ).copy()
        plot_df["_highlight_key"] = plot_df[highlight_column].map(_normalize_plot_label)
        highlight_set = set(highlight_values)
        highlight_mask = plot_df["_highlight_key"].isin(highlight_set)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=_coerce_figsize(params.figsize, (10.0, 6.0)))
        default_cycle = list(plt.get_cmap("tab10").colors)  # pyright: ignore[reportAttributeAccessIssue]
        highlight_colors = _parse_text_list(params.highlight_colors)
        if highlight_colors and len(highlight_colors) != len(highlight_values):
            raise BlockValidationError(
                "highlight_colors must be blank or aligned to highlight_values."
            )
        color_map = {
            value: (
                highlight_colors[index]
                if highlight_colors
                else default_cycle[index % len(default_cycle)]
            )
            for index, value in enumerate(highlight_values)
        }

        guide_bands = _parse_json_object_list(
            params.guide_bands,
            block_name="HighlightedScatterPlot",
            param_name="guide_bands",
        )
        guide_lines = _parse_json_object_list(
            params.guide_lines,
            block_name="HighlightedScatterPlot",
            param_name="guide_lines",
        )
        annotations = _parse_json_object_list(
            params.annotations,
            block_name="HighlightedScatterPlot",
            param_name="annotations",
        )
        n_bands = _draw_reference_bands(
            ax,
            guide_bands,
            block_name="HighlightedScatterPlot",
        )

        x_values = plot_df["_x"].to_numpy(dtype=float)
        y_values = plot_df["_y"].to_numpy(dtype=float)
        if bool(params.identity_line):
            lower = float(min(np.nanmin(x_values), np.nanmin(y_values)))
            upper = float(max(np.nanmax(x_values), np.nanmax(y_values)))
            ax.plot(
                [lower, upper],
                [lower, upper],
                color=str(params.identity_line_color),
                linestyle=str(params.identity_line_style),
                linewidth=1.25,
                zorder=1.25,
            )

        n_lines = _draw_reference_lines(
            ax,
            guide_lines,
            block_name="HighlightedScatterPlot",
        )

        background_df = plot_df.loc[~highlight_mask]
        if not background_df.empty:
            ax.scatter(
                background_df["_x"].to_numpy(dtype=float),
                background_df["_y"].to_numpy(dtype=float),
                s=background_df["_size"].to_numpy(dtype=float),
                alpha=float(params.background_alpha),
                color=str(params.background_color),
                zorder=2.0,
            )

        n_highlighted_points = 0
        highlight_groups_drawn = 0
        for highlight_value in highlight_values:
            subset = plot_df.loc[plot_df["_highlight_key"] == highlight_value]
            if subset.empty:
                continue
            ax.scatter(
                subset["_x"].to_numpy(dtype=float),
                subset["_y"].to_numpy(dtype=float),
                s=subset["_size"].to_numpy(dtype=float),
                alpha=float(params.highlight_alpha),
                color=color_map[highlight_value],
                label=highlight_value
                if bool(getattr(params, "show_highlight_legend", True))
                else None,
                zorder=3.0,
            )
            n_highlighted_points += int(subset.shape[0])
            highlight_groups_drawn += 1
            if bool(params.label_highlights):
                for _, row in subset.iterrows():
                    ax.annotate(
                        str(row[label_column]),
                        xy=(float(row["_x"]), float(row["_y"])),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=float(params.label_fontsize),
                        zorder=4.0,
                    )

        n_annotations = _draw_text_annotations(
            ax,
            annotations,
            block_name="HighlightedScatterPlot",
        )

        if bool(params.show_legend):
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend()

        plot_title = _normalize_plot_title(
            params.title,
            f"{y_column} vs {x_column}",
        )
        x_label = str(params.x_label or "").strip() or x_column
        y_label = str(params.y_label or "").strip() or y_column
        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "highlight_column": highlight_column,
                "highlight_values": highlight_values,
                "size_column": size_column or None,
                "n_rows_plotted": int(plot_df.shape[0]),
                "n_background_points": int(background_df.shape[0]),
                "n_highlighted_points": int(n_highlighted_points),
                "n_highlight_groups_drawn": int(highlight_groups_drawn),
                "n_guide_bands": int(n_bands),
                "n_guide_lines": int(n_lines),
                "n_annotations": int(n_annotations),
            },
        )


class HighlightedBarChart(VisualizationBlock):
    name = "Highlighted Bar Chart"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a horizontal bar chart with highlighted named subsets and optional reference lines."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]
    usage_notes = [
        "Bars whose category_column value is not listed in highlight_values are rendered with the default color.",
        "reference_lines accepts a JSON object or JSON list of objects describing reference lines.",
        "Set show_all_tick_labels=false and label_highlights=true to annotate only the highlighted bars.",
    ]

    class Params(VisualizationParams):
        export_basename: str = block_param(
            "",
            description="Optional export filename stem used instead of the plot title when export is enabled.",
            example="highlighted_bar",
        )
        category_column: str = block_param(
            description="Column providing the bar labels.",
            example="label",
        )
        value_column: str = block_param(
            description="Numeric column providing the bar values.",
            example="score",
        )
        highlight_values: str = block_param(
            "",
            description="Optional comma-separated values, or a JSON string list, defining the highlighted bars in display order.",
            example="Group A,Group B,Group C",
        )
        highlight_colors: str = block_param(
            "",
            description="Optional comma-separated colors, or a JSON string list of colors, aligned to highlight_values.",
            example="#17becf,#2ca02c,#1f77b4",
        )
        top_n: int = block_param(
            0,
            description="Maximum number of rows to plot after sorting. Use 0 to keep all rows.",
            example=57,
        )
        sort_by_value: bool = block_param(
            True,
            description="Whether to sort bars by the numeric value before plotting.",
            example=True,
        )
        ascending: bool = block_param(
            False,
            description="Sort direction when sort_by_value is true.",
            example=False,
        )
        default_color: str = block_param(
            "lightgray",
            description="Color used for non-highlighted bars.",
            example="lightgray",
        )
        label_highlights: bool = block_param(
            True,
            description="Whether to add text labels for the highlighted bars.",
            example=True,
        )
        label_fontsize: float = block_param(
            10.0,
            description="Font size used when label_highlights is true.",
            example=10.0,
        )
        show_all_tick_labels: bool = block_param(
            False,
            description="Whether to show y-axis tick labels for every bar instead of only annotating highlighted bars.",
            example=False,
        )
        x_label: str = block_param(
            "",
            description="Optional x-axis label override. Leave blank to use value_column.",
            example="Score",
        )
        reference_lines: str = block_param(
            "",
            description="Optional JSON object or JSON list of objects describing reference lines with keys such as axis, value, color, linestyle, linewidth, alpha, and label.",
            example='[{"axis":"x","value":0.5,"color":"#ff8b8b","linestyle":"--","linewidth":2.0,"label":"threshold"}]',
        )
        figsize: list[float] = [14.0, 8.0]
        title: str = "Highlighted Bar Chart"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("HighlightedBarChart requires params.")
        _require_non_empty_frame(data, "HighlightedBarChart")

        category_column = _require_column(
            data,
            params.category_column,
            "HighlightedBarChart",
            "category_column",
        )
        value_column = _require_column(
            data,
            params.value_column,
            "HighlightedBarChart",
            "value_column",
        )

        plot_df = pd.DataFrame(
            {
                "_label": data[category_column].astype(str),
                "_key": data[category_column].map(_normalize_plot_label),
                "_value": pd.to_numeric(data[value_column], errors="coerce"),
            }
        ).dropna(subset=["_value"])
        if plot_df.empty:
            raise BlockValidationError(
                f"Column '{value_column}' has no numeric values after dropping null/non-numeric entries."
            )

        if bool(params.sort_by_value):
            plot_df = plot_df.sort_values("_value", ascending=bool(params.ascending))
        top_n = int(getattr(params, "top_n", 0))
        if top_n > 0:
            plot_df = plot_df.head(top_n)
        plot_df = plot_df.reset_index(drop=True)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=_coerce_figsize(params.figsize, (14.0, 8.0)))
        default_cycle = list(plt.get_cmap("tab10").colors)  # pyright: ignore[reportAttributeAccessIssue]
        highlight_values = _parse_text_list(params.highlight_values)
        highlight_colors = _parse_text_list(params.highlight_colors)
        if highlight_colors and len(highlight_colors) != len(highlight_values):
            raise BlockValidationError(
                "highlight_colors must be blank or aligned to highlight_values."
            )
        color_map = {
            value: (
                highlight_colors[index]
                if highlight_colors
                else default_cycle[index % len(default_cycle)]
            )
            for index, value in enumerate(highlight_values)
        }
        plot_df["_is_highlight"] = plot_df["_key"].isin(set(highlight_values))
        plot_df["_color"] = (
            plot_df["_key"].map(color_map).fillna(str(params.default_color))
        )

        positions = np.arange(plot_df.shape[0], dtype=float)
        ax.barh(
            positions,
            plot_df["_value"].to_numpy(dtype=float),
            color=plot_df["_color"].tolist(),
        )
        ax.set_yticks(positions.tolist())
        if bool(params.show_all_tick_labels):
            ax.set_yticklabels(plot_df["_label"].tolist())
        else:
            ax.set_yticklabels([""] * plot_df.shape[0])
        ax.invert_yaxis()

        reference_lines = _parse_json_object_list(
            params.reference_lines,
            block_name="HighlightedBarChart",
            param_name="reference_lines",
        )
        n_reference_lines = _draw_reference_lines(
            ax,
            reference_lines,
            block_name="HighlightedBarChart",
        )

        if bool(params.label_highlights):
            values = plot_df["_value"].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            scale = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
            offset = max(scale * 0.01, 0.02)
            for position, row in zip(
                positions.tolist(),
                plot_df.to_dict(orient="records"),
            ):
                if not bool(row["_is_highlight"]):
                    continue
                bar_value = float(row["_value"])
                ax.text(
                    bar_value + offset if bar_value >= 0 else bar_value - offset,
                    float(position),
                    str(row["_label"]),
                    ha="left" if bar_value >= 0 else "right",
                    va="center",
                    fontsize=float(params.label_fontsize),
                )

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()

        plot_title = _normalize_plot_title(
            params.title,
            f"{value_column} by {category_column}",
        )
        x_label = str(params.x_label or "").strip() or value_column
        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(category_column)
        fig.tight_layout()
        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "category_column": category_column,
                "value_column": value_column,
                "highlight_values": highlight_values,
                "n_rows_plotted": int(plot_df.shape[0]),
                "n_highlighted_bars": int(plot_df["_is_highlight"].sum()),
                "n_reference_lines": int(n_reference_lines),
            },
        )


class FacetedScatterPlot(VisualizationBlock):
    name = "Faceted Scatter Plot"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a grid of scatter plots split by a facet column and ordered by explicit facet values."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]
    usage_notes = [
        "Only facet values listed in facet_values are rendered, and they appear in that exact order.",
        "facet_title_template is formatted using the first row of each facet plus facet_value and status_label.",
        "facet_color_map and facet_status_map accept JSON objects keyed by facet value.",
    ]

    class Params(VisualizationParams):
        export_basename: str = block_param(
            "",
            description="Optional export filename stem used instead of the plot title when export is enabled.",
            example="faceted_scatter",
        )
        x_column: str = block_param(
            description="Numeric x-axis column.",
            example="x_value",
        )
        y_column: str = block_param(
            description="Numeric y-axis column.",
            example="y_value",
        )
        facet_column: str = block_param(
            description="Column whose values define each facet panel.",
            example="category",
        )
        facet_values: str = block_param(
            description="Comma-separated values, or a JSON string list, defining the facets to render in order.",
            example="Group A,Group B,Group C",
        )
        facet_color_map: str = block_param(
            "",
            description="Optional JSON object mapping facet values to point colors.",
            example='{"Group A":"#d62728","Group B":"#ff7f0e"}',
        )
        facet_status_map: str = block_param(
            "",
            description="Optional JSON object mapping facet values to a status label that can be referenced in facet_title_template as {status_label}.",
            example='{"Group A":"Primary","Group B":"Secondary"}',
        )
        facet_title_map: str = block_param(
            "",
            description="Optional JSON object mapping facet values to facet-specific Python format strings or literal titles. When provided, entries override facet_title_template for matching facets.",
            example='{"Group C":"Group C\\nn={item_count:.0f}, r={metric_a:.2f}, s={metric_b:.2f}"}',
        )
        facet_title_template: str = block_param(
            "{facet_value}",
            description="Python format string used for each facet title. Available keys include facet_value, status_label, and the first row's columns for that facet.",
            example="{facet_value}\\nn={item_count:.0f}, r={metric_a:.2f}, s={metric_b:.2f}\\n[{status_label}]",
        )
        n_cols: int = block_param(
            5,
            description="Number of facet columns in the subplot grid.",
            example=5,
        )
        marker_size: float = block_param(
            35.0,
            description="Marker size used for every facet point.",
            example=35.0,
        )
        alpha: float = block_param(
            0.9,
            description="Opacity for every facet point.",
            example=0.9,
        )
        identity_line: bool = block_param(
            True,
            description="Whether to draw an x=y diagonal reference line within each facet.",
            example=True,
        )
        identity_line_color: str = block_param(
            "0.75",
            description="Color used for each facet's x=y diagonal reference line.",
            example="0.75",
        )
        identity_line_style: str = block_param(
            "--",
            description="Line style used for each facet's x=y diagonal reference line.",
            example="--",
        )
        x_label: str = block_param(
            "",
            description="Optional global x-axis label override. Leave blank to use x_column.",
            example="Reference Value",
        )
        y_label: str = block_param(
            "",
            description="Optional global y-axis label override. Leave blank to use y_column.",
            example="Comparison Value",
        )
        figsize: list[float] = [18.0, 10.0]
        title: str = "Faceted Scatter Plot"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("FacetedScatterPlot requires params.")
        _require_non_empty_frame(data, "FacetedScatterPlot")

        x_column = _require_column(
            data, params.x_column, "FacetedScatterPlot", "x_column"
        )
        y_column = _require_column(
            data, params.y_column, "FacetedScatterPlot", "y_column"
        )
        facet_column = _require_column(
            data, params.facet_column, "FacetedScatterPlot", "facet_column"
        )

        facet_values = _parse_text_list(params.facet_values)
        if not facet_values:
            raise BlockValidationError("facet_values is required.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cols = max(int(params.n_cols), 1)
        n_rows = int(np.ceil(len(facet_values) / float(n_cols)))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=_coerce_figsize(params.figsize, (18.0, 10.0)),
        )
        axes_array = np.atleast_1d(axes).ravel()
        default_cycle = list(plt.get_cmap("tab10").colors)  # pyright: ignore[reportAttributeAccessIssue]
        facet_color_map = _parse_json_mapping(
            params.facet_color_map,
            block_name="FacetedScatterPlot",
            param_name="facet_color_map",
        )
        facet_status_map = _parse_json_mapping(
            params.facet_status_map,
            block_name="FacetedScatterPlot",
            param_name="facet_status_map",
        )
        facet_title_map = _parse_json_mapping(
            getattr(params, "facet_title_map", ""),
            block_name="FacetedScatterPlot",
            param_name="facet_title_map",
        )

        working = data.copy()
        working["_facet_key"] = working[facet_column].map(_normalize_plot_label)
        facets_drawn = 0
        for index, facet_value in enumerate(facet_values):
            ax = axes_array[index]
            facet_df = working.loc[working["_facet_key"] == facet_value].copy()
            if facet_df.empty:
                raise BlockValidationError(
                    f"FacetedScatterPlot found no rows for facet '{facet_value}'."
                )

            plot_df = _prepare_scatter_plot_df(
                facet_df,
                x_column,
                y_column,
                "",
                float(params.marker_size),
                "FacetedScatterPlot",
            )
            color = str(
                facet_color_map.get(
                    facet_value,
                    default_cycle[index % len(default_cycle)],
                )
            )
            ax.scatter(
                plot_df["_x"].to_numpy(dtype=float),
                plot_df["_y"].to_numpy(dtype=float),
                s=float(params.marker_size),
                alpha=float(params.alpha),
                color=color,
            )
            if bool(params.identity_line):
                x_vals = plot_df["_x"].to_numpy(dtype=float)
                y_vals = plot_df["_y"].to_numpy(dtype=float)
                lower = float(min(np.nanmin(x_vals), np.nanmin(y_vals)))
                upper = float(max(np.nanmax(x_vals), np.nanmax(y_vals)))
                ax.plot(
                    [lower, upper],
                    [lower, upper],
                    color=str(params.identity_line_color),
                    linestyle=str(params.identity_line_style),
                    linewidth=1.0,
                    zorder=1.0,
                )

            context = {key: value for key, value in facet_df.iloc[0].to_dict().items()}
            context["facet_value"] = facet_value
            context["status_label"] = str(facet_status_map.get(facet_value, "") or "")
            title_template = str(
                facet_title_map.get(facet_value, params.facet_title_template)
            )
            title_text = _format_text_template(
                title_template,
                context,  # pyright: ignore[reportArgumentType]
                block_name="FacetedScatterPlot",
                param_name="facet_title_map"
                if facet_value in facet_title_map
                else "facet_title_template",
            )
            ax.set_title(title_text or facet_value, fontsize=11.0, fontweight="bold")
            facets_drawn += 1

        for index in range(len(facet_values), len(axes_array)):
            axes_array[index].axis("off")

        x_label = str(params.x_label or "").strip() or x_column
        y_label = str(params.y_label or "").strip() or y_column
        try:
            fig.supxlabel(x_label)
            fig.supylabel(y_label)
        except Exception:
            if axes_array.size:
                axes_array[0].set_xlabel(x_label)
                axes_array[0].set_ylabel(y_label)

        plot_title = _normalize_plot_title(params.title, "Faceted Scatter Plot")
        if plot_title:
            fig.suptitle(plot_title)
            fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
        else:
            fig.tight_layout()
        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "facet_column": facet_column,
                "facet_values": facet_values,
                "n_facets_drawn": int(facets_drawn),
                "n_cols": int(n_cols),
            },
        )


class AnnotatePlotWithArrows(VisualizationBlock):
    name = "Annotate Plot With Arrows"
    version = "1.0.0"
    category = "Visualization"
    description = (
        "Render a node scatter plot and draw directional arrows from source labels "
        "to destination labels listed in a separate edges matrix."
    )
    n_inputs = 2
    input_labels = ["Nodes Matrix", "Edges Matrix"]
    output_labels = ["Nodes Matrix + Image"]
    usage_notes = [
        "Input 1 must contain one row per node with numeric x/y coordinates plus a unique label column.",
        "Input 2 uses its first column as the source label and every remaining column as an optional destination label.",
        "Scatter styling and color behavior match Matrix Scatter Plot, including categorical legends and numeric colorbars.",
        "Blank destination cells are ignored; the nodes DataFrame passes through unchanged and the image is a side artifact.",
    ]

    class Params(ScatterPlotParams):
        label_column: str = block_param(
            description="Column in the nodes input used to match edge labels to points.",
            example="compound_id",
        )
        point_color: str = block_param(
            "steelblue",
            description="Fallback marker color when color_column is blank.",
            example="black",
        )
        show_node_labels: bool = block_param(
            False,
            description="Whether to draw each node label next to its point.",
            example=True,
        )
        label_fontsize: float = block_param(
            9.0,
            description="Font size used when show_node_labels is true.",
            example=10.0,
        )
        arrow_color: str = block_param(
            "black",
            description="Matplotlib color for directional arrows.",
            example="darkred",
        )
        arrow_alpha: float = block_param(
            0.7,
            description="Opacity for directional arrows.",
            example=0.75,
        )
        arrow_linewidth: float = block_param(
            1.2,
            description="Line width for directional arrows.",
            example=1.5,
        )
        arrow_style: str = block_param(
            "->",
            description="Matplotlib arrowstyle passed to annotate arrowprops.",
            example="-|>",
        )
        figsize: list[float] = [10.0, 6.0]
        title: str = "Annotated Plot"

    def validate(self, data: Any) -> None:
        if not isinstance(data, (list, tuple)) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("AnnotatePlotWithArrows requires both inputs.")
        nodes_df, edges_df = data[:2]
        _require_non_empty_frame(nodes_df, "AnnotatePlotWithArrows")
        if edges_df.shape[1] < 2:
            raise BlockValidationError(
                "AnnotatePlotWithArrows requires the edges input to have at least "
                "2 columns: source and one destination."
            )

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("AnnotatePlotWithArrows requires params.")

        nodes_df, edges_df = _require_two_dataframes(data, "AnnotatePlotWithArrows")
        _require_non_empty_frame(nodes_df, "AnnotatePlotWithArrows")
        if edges_df.shape[1] < 2:
            raise BlockValidationError(
                "AnnotatePlotWithArrows requires the edges input to have at least "
                "2 columns: source and one destination."
            )

        x_column = _require_column(
            nodes_df, params.x_column, "AnnotatePlotWithArrows", "x_column"
        )
        y_column = _require_column(
            nodes_df, params.y_column, "AnnotatePlotWithArrows", "y_column"
        )
        color_column = str(params.color_column).strip()
        size_column = str(params.size_column).strip()
        label_column = _require_column(
            nodes_df, params.label_column, "AnnotatePlotWithArrows", "label_column"
        )
        if (
            color_column
            and color_column != "index"
            and color_column not in nodes_df.columns
        ):
            raise BlockValidationError(f"Column '{color_column}' not found.")
        if size_column and size_column not in nodes_df.columns:
            raise BlockValidationError(f"Column '{size_column}' not found.")

        plot_df = _prepare_scatter_plot_df(
            nodes_df,
            x_column,
            y_column,
            size_column,
            float(params.marker_size),
            "AnnotatePlotWithArrows",
        )
        plot_df["_label_key"] = plot_df[label_column].map(_normalize_plot_label)

        if bool(plot_df["_label_key"].isna().any()):
            raise BlockValidationError(
                f"AnnotatePlotWithArrows requires non-null, non-blank values in '{label_column}'."
            )

        duplicate_labels = sorted(
            set(
                plot_df.loc[
                    plot_df["_label_key"].duplicated(keep=False), "_label_key"
                ].tolist()
            )
        )
        if duplicate_labels:
            preview = ", ".join(repr(label) for label in duplicate_labels[:10])
            suffix = ", ..." if len(duplicate_labels) > 10 else ""
            raise BlockValidationError(
                "AnnotatePlotWithArrows requires unique node labels in "
                f"'{label_column}'. Duplicates: {preview}{suffix}"
            )

        invalid_coordinate_mask = plot_df[["_x", "_y"]].isna().any(axis=1)
        if bool(invalid_coordinate_mask.any()):
            invalid_labels = plot_df.loc[invalid_coordinate_mask, "_label_key"].tolist()
            preview = ", ".join(repr(label) for label in invalid_labels[:10])
            suffix = ", ..." if len(invalid_labels) > 10 else ""
            raise BlockValidationError(
                "AnnotatePlotWithArrows found non-numeric coordinates for labels: "
                f"{preview}{suffix}"
            )

        node_lookup = plot_df.set_index("_label_key")[["_x", "_y"]]
        parsed_edges: list[tuple[str, list[str]]] = []
        missing_labels: set[str] = set()
        for row_number, (_, edge_row) in enumerate(edges_df.iterrows(), start=1):
            values = edge_row.tolist()
            source_label = _normalize_plot_label(values[0])
            destination_labels = [
                label
                for label in (_normalize_plot_label(value) for value in values[1:])
                if label is not None
            ]

            if not destination_labels:
                continue
            if source_label is None:
                raise BlockValidationError(
                    "AnnotatePlotWithArrows found a blank source label in the edges "
                    f"input at row {row_number}."
                )

            parsed_edges.append((source_label, destination_labels))
            if source_label not in node_lookup.index:
                missing_labels.add(source_label)
            for destination_label in destination_labels:
                if destination_label not in node_lookup.index:
                    missing_labels.add(destination_label)

        if missing_labels:
            missing_list = sorted(missing_labels)
            preview = ", ".join(repr(label) for label in missing_list[:10])
            suffix = ", ..." if len(missing_list) > 10 else ""
            raise BlockValidationError(
                "AnnotatePlotWithArrows could not resolve these edge labels in "
                f"'{label_column}': {preview}{suffix}"
            )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=_coerce_figsize(params.figsize, (10.0, 6.0)))
        arrows_drawn = 0
        arrow_zorder = 3
        for source_label, destination_labels in parsed_edges:
            source_xy = node_lookup.loc[source_label]
            source = (float(source_xy["_x"]), float(source_xy["_y"]))  # pyright: ignore[reportArgumentType]
            for destination_label in destination_labels:
                destination_xy = node_lookup.loc[destination_label]
                ax.annotate(
                    "",
                    xy=(float(destination_xy["_x"]), float(destination_xy["_y"])),  # pyright: ignore[reportArgumentType]
                    xytext=source,
                    arrowprops={
                        "arrowstyle": str(params.arrow_style),
                        "color": str(params.arrow_color),
                        "alpha": float(params.arrow_alpha),
                        "linewidth": float(params.arrow_linewidth),
                        "zorder": arrow_zorder,
                    },
                    zorder=arrow_zorder,
                )
                arrows_drawn += 1

        point_color = str(getattr(params, "point_color", "") or "").strip() or None
        color_mode_used = _render_scatter_points(
            fig,
            ax,
            plot_df,
            color_column=color_column,
            color_mode=str(getattr(params, "color_mode", "auto")),
            cmap=str(params.cmap),
            alpha=float(params.alpha),
            block_name="AnnotatePlotWithArrows",
            default_color=point_color,
            zorder=2,
        )
        if bool(params.show_node_labels):
            for _, row in plot_df.iterrows():
                ax.annotate(
                    str(row[label_column]),
                    xy=(float(row["_x"]), float(row["_y"])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=float(params.label_fontsize),
                    zorder=4,
                )

        plot_title = _normalize_plot_title(
            params.title,
            f"Annotated {y_column} vs {x_column}",
        )
        ax.set_title(plot_title)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        fig.tight_layout()
        return _visual_output(
            nodes_df,
            fig,
            params,
            plot_title,
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column or None,
                "color_mode_used": color_mode_used,
                "label_column": label_column,
                "size_column": size_column or None,
                "n_nodes_plotted": int(plot_df.shape[0]),
                "n_edge_rows_processed": int(len(parsed_edges)),
                "n_arrows_drawn": int(arrows_drawn),
            },
        )


class Matrix3DScatterPlot(VisualizationBlock):
    name = "Matrix 3D Scatter Plot"
    version = "1.0.0"
    category = "Visualization"
    description = "Render a 3D scatter plot using Plotly."
    input_labels = ["Matrix"]
    output_labels = ["Matrix + Image"]
    usage_notes = [
        "`color_mode='auto'` treats numeric dtype columns as continuous and string/object columns as categorical.",
        "Set `color_mode='categorical'` when numeric-looking cluster IDs should render as discrete categories.",
        "The image is rendered through Plotly and stored as a checkpoint-backed artifact.",
    ]

    class Params(VisualizationParams):
        x_column: str = block_param(
            description="Numeric x-axis column.",
            example="feature_a",
        )
        y_column: str = block_param(
            description="Numeric y-axis column.",
            example="feature_b",
        )
        z_column: str = block_param(
            description="Numeric z-axis column.",
            example="feature_c",
        )
        color_column: str = block_param(
            "",
            description="Optional column used to color points. Use 'index' to color by row index.",
            example="cluster_id",
        )
        color_mode: str = block_param(
            "auto",
            description="How to interpret color_column. Use auto, numeric, or categorical.",
            example="categorical",
        )
        size_column: str = block_param(
            "",
            description="Optional numeric column used for marker size.",
            example="feature_d",
        )
        marker_size: float = 6.0
        opacity: float = 0.75
        title: str = "3D Scatter Plot"

    def execute(self, data: pd.DataFrame, params: Params | None = None) -> BlockOutput:
        if params is None:
            raise BlockValidationError("Matrix3DScatterPlot requires params.")
        _require_non_empty_frame(data, "Matrix3DScatterPlot")

        x_column = _require_column(
            data, params.x_column, "Matrix3DScatterPlot", "x_column"
        )
        y_column = _require_column(
            data, params.y_column, "Matrix3DScatterPlot", "y_column"
        )
        z_column = _require_column(
            data, params.z_column, "Matrix3DScatterPlot", "z_column"
        )
        color_column = str(params.color_column).strip()
        size_column = str(params.size_column).strip()
        if (
            color_column
            and color_column != "index"
            and color_column not in data.columns
        ):
            raise BlockValidationError(f"Column '{color_column}' not found.")
        if size_column and size_column not in data.columns:
            raise BlockValidationError(f"Column '{size_column}' not found.")

        plot_df = data.copy()
        plot_df["_x"] = pd.to_numeric(plot_df[x_column], errors="coerce")
        plot_df["_y"] = pd.to_numeric(plot_df[y_column], errors="coerce")
        plot_df["_z"] = pd.to_numeric(plot_df[z_column], errors="coerce")
        plot_df = plot_df.dropna(subset=["_x", "_y", "_z"])
        if plot_df.empty:
            raise BlockValidationError(
                f"Matrix3DScatterPlot found no numeric rows for '{x_column}', '{y_column}', and '{z_column}'."
            )
        if size_column:
            size_values = pd.to_numeric(plot_df[size_column], errors="coerce")
            size_values = size_values.fillna(float(params.marker_size))
            plot_df["_size"] = np.clip(
                size_values.to_numpy(dtype=float), a_min=1.0, a_max=None
            )

        try:
            import plotly.express as px
        except Exception as exc:
            raise BlockValidationError(
                "Matrix3DScatterPlot requires plotly. Install dependencies and retry."
            ) from exc

        plot_title = _normalize_plot_title(params.title, "3D Scatter Plot")
        figure_kwargs: dict[str, Any] = {
            "x": "_x",
            "y": "_y",
            "z": "_z",
            "title": plot_title,
        }
        color_mode_used: str | None = None
        if color_column:
            color_mode = _normalize_color_mode(
                getattr(params, "color_mode", "auto"),
                "Matrix3DScatterPlot",
            )
            color_group_key, color_values_raw = _resolve_color_series(
                plot_df,
                color_column,
            )
            numeric_color = pd.to_numeric(color_values_raw, errors="coerce")
            non_null_color = color_values_raw.notna()
            can_use_numeric_scale = bool(non_null_color.any()) and bool(
                numeric_color[non_null_color].notna().all()
            )
            wants_numeric_scale = _wants_numeric_color_scale(
                color_values_raw,
                color_mode,
            )
            if color_mode == "numeric" and not can_use_numeric_scale:
                raise BlockValidationError(
                    f"Matrix3DScatterPlot could not interpret '{color_column}' as numeric colors."
                )
            if wants_numeric_scale and can_use_numeric_scale:
                plot_df["_color_numeric"] = numeric_color
                figure_kwargs["color"] = "_color_numeric"
                color_mode_used = "numeric"
            else:
                plot_df["_color_display"] = color_values_raw.astype("string").fillna(
                    "null"
                )
                figure_kwargs["color"] = "_color_display"
                color_mode_used = "categorical"
        if size_column:
            figure_kwargs["size"] = "_size"

        fig = px.scatter_3d(plot_df, **figure_kwargs)
        marker_update: dict[str, Any] = {"opacity": float(params.opacity)}
        if not size_column:
            marker_update["size"] = float(params.marker_size)
        fig.update_traces(marker=marker_update)
        fig.update_layout(
            scene={
                "xaxis_title": x_column,
                "yaxis_title": y_column,
                "zaxis_title": z_column,
            }
        )

        return _visual_output(
            data,
            fig,
            params,
            plot_title,
            metadata={
                "x_column": x_column,
                "y_column": y_column,
                "z_column": z_column,
                "color_column": color_column or None,
                "color_mode_used": color_mode_used,
                "size_column": size_column or None,
                "n_rows_plotted": int(plot_df.shape[0]),
                "render_backend": "plotly",
            },
        )
