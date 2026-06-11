# *Forge — pipeline canvas for the repo you already have open

Visual, block-based pipelines over your own Python functions. Every node
checkpoints its output with full provenance; change a parameter or edit a
function and only the affected subgraph reruns. Revert, and the old
checkpoint is still there — instant.

## Quick start

1. Install the kernel into the Python environment your repo uses:

   ```bash
   pip install starforge-kernel
   # (or, from a Forge checkout: pip install -e <forge>/starforge)
   ```

2. Decorate a function and save:

   ```python
   from starforge import block

   @block
   def scale(data: dict, factor: float = 2.0) -> dict:
       return {"values": [v * factor for v in data["values"]]}
   ```

3. Run **“*Forge: New Pipeline”**, drag blocks from the palette, wire them
   up, press **▶ Run**. Run again — instant. Edit the function — only it and
   its descendants go stale.

## What you get

- **Zero-template blocks** — the function signature *is* the block: params
  become typed input handles and inspector fields, `T | None` means optional,
  tuple returns are multi-output. A CodeLens adds `@block` to any function.
- **Provenance-backed caching** — Tier-2 staleness tracks your repo's import
  graph; editing a helper marks its dependents stale. Configurable
  (`starforge.stalenessTier`).
- **Figures inline** — `plt.show()` renders on the node (thumbnail, carousel,
  lightbox); plotly figures open live in an editor panel.
- **Progress + cancellation** — `starforge.progress(i, n, "label")` drives a
  live bar; Cancel kills the run process cleanly.
- **Comments, copy/paste, box-select, minimap navigation** — the full canvas
  experience, in your editor.
- **Team-shareable pipelines** — `.forge/pipelines/*.forge` are plain JSON,
  made to be committed; checkpoints and caches are auto-gitignored.

## Requirements

- Python ≥ 3.10 in the workspace with `starforge-kernel` installed
  (`starforge.pythonPath` overrides interpreter discovery; the ms-python
  extension's active interpreter is used when available).
- Heavy libraries (pandas, matplotlib, plotly) are only ever loaded inside
  run workers — the kernel itself is stdlib-only and idles at zero.

Apache-2.0 · part of [Forge](https://github.com/Jonpot/forge)
