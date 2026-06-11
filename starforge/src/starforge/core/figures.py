"""Figure capture and artifact rendering.

The notebook muscle memory is ``plt.plot(...); plt.show()`` — or no show()
at all. The worker honors it with zero code changes: matplotlib runs on the
Agg backend, and :func:`capture` sweeps every figure that exists after the
block call that didn't exist before (``plt.show`` is a no-op under Agg, so
"shown" figures are still open when we sweep). Plotly's ``fig.show()`` is
intercepted by patching ``plotly.io.show`` while the block runs.

Captured and returned figures render to checkpoint artifacts — matplotlib →
PNG, plotly → self-contained HTML — and are closed afterward so a long run
never accumulates canvases.

Import discipline: stdlib-only at import time; matplotlib/plotly are only
touched when the user's process already loaded them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterator


def _pyplot() -> Any | None:
    return sys.modules.get("matplotlib.pyplot")


def _root_module(value: Any) -> str:
    return type(value).__module__.split(".")[0]


@dataclass
class CapturedFigures:
    matplotlib: list[Any] = field(default_factory=list)
    plotly: list[Any] = field(default_factory=list)

    def all_objects(self) -> list[Any]:
        return [*self.matplotlib, *self.plotly]


@contextmanager
def capture() -> Iterator[CapturedFigures]:
    """Collect figures created (matplotlib) or shown (plotly) inside the
    block call. The matplotlib sweep also catches figures created during the
    block module's first import, since the import happens inside this
    context in the runner."""
    captured = CapturedFigures()

    plt = _pyplot()
    before: set[int] = set(plt.get_fignums()) if plt is not None else set()

    pio = sys.modules.get("plotly.io")
    original_show = getattr(pio, "show", None) if pio is not None else None
    if pio is not None and original_show is not None:

        def _grab(fig: Any, *args: Any, **kwargs: Any) -> None:
            captured.plotly.append(fig)

        pio.show = _grab

    try:
        yield captured
    finally:
        if pio is not None and original_show is not None:
            pio.show = original_show
        plt = _pyplot()  # may have been imported during the call
        if plt is not None:
            for num in plt.get_fignums():
                if num not in before:
                    captured.matplotlib.append(plt.figure(num))


def as_figure(value: Any) -> Any | None:
    """Return a renderable figure for ``value``, or None.

    Accepts matplotlib Figures, matplotlib Axes (``sns.heatmap`` et al.
    return Axes — we render their parent figure), and plotly figures.
    """
    root = _root_module(value)
    if root == "matplotlib":
        if hasattr(value, "savefig"):
            return value
        parent = getattr(value, "figure", None)  # Axes and friends
        if parent is not None and hasattr(parent, "savefig"):
            return parent
    if root == "plotly" and hasattr(value, "write_html"):
        return value
    return None


def render_figure(value: Any, directory: Path, basename: str) -> dict[str, Any] | None:
    """Render to ``directory/basename.(png|html)``; returns the artifact
    entry ``{"file", "kind"}`` or None if ``value`` is not a figure."""
    fig = as_figure(value)
    if fig is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    if _root_module(fig) == "matplotlib":
        filename = f"{basename}.png"
        fig.savefig(directory / filename, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        return {"file": filename, "kind": "image"}
    filename = f"{basename}.html"
    fig.write_html(directory / filename, include_plotlyjs=True, full_html=True)
    return {"file": filename, "kind": "html"}


def close_figures(figures: list[Any]) -> None:
    plt = _pyplot()
    if plt is None:
        return
    for fig in figures:
        if _root_module(fig) == "matplotlib":
            try:
                plt.close(fig)
            except Exception:
                pass
