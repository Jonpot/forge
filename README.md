# Forge

A visual, block-based experimentation framework for interactive data science pipelines. Define atomic operations in Python, compose them in a drag-and-drop DAG editor, and never rerun your entire notebook again.

## Problem

Data science experimentation lives in notebooks. Notebooks are linear. When your MD asks "what if you clustered before normalizing?" you rerun 20 minutes of computation. When you try something, hate it, and revert, you rerun 20 minutes of computation. When you find a bug in your normalization function, you rerun everything downstream manually.

Forge fixes this. Every node checkpoints its output. Every checkpoint carries its full provenance. Change a parameter, and only the affected subgraph reruns. Revert a change, and the old checkpoint is still on disk. Version a block, and all downstream nodes know they're stale.

## Current Status

- Core engine, FastAPI backend (REST + WebSocket), and React/Tauri frontend are all shipped.
- Packaged as a desktop app: `.msi` installer for Windows, `.dmg` for macOS.
- 60+ blocks across 9 categories: IO, Transform, Clustering, Visualization, Operator, Dimensionality, Factorization, Statistics, and Custom.
- MCP server over stdio (`python -m Forge mcp`) and streamable HTTP (`/mcp`), with 32 tools covering the full pipeline lifecycle — create, edit, run, poll, and inspect results.
- Onboarding tour, workspace setup wizard, settings modal, and in-app file browser included.

## Adding Custom Blocks (Plugins)
Forge supports user-defined blocks as Python classes inheriting from `BaseBlock`. To add your own blocks, you can download a template from the "Plugins" dropdown in the app. Once you've created your block class, you can simply drag the file onto the canvas to add it to the palette, or use the "Import Plugin" option in the menu. The backend will auto-discover it and make it available for use in your pipelines.

You can manage your custom blocks in the "Plugins" section of the settings, where you can see all imported blocks, their source files, and options to remove them from the palette if needed.

Custom blocks can define their own custom categories- for which the colors and icons can be configured in the settings. This allows you to organize your blocks in a way that makes sense for your specific use case.

Custom blocks are identified by a star icon in the palette, making it easy to distinguish them from built-in blocks. They can be used just like any other block in your pipelines, with full support for parameters, presets, and provenance tracking.

You can right-click on a custom block to download its source code as a template for creating new blocks. This is a great way to get started with block development, as it provides a working example that you can modify to suit your needs.

### Frontend: React + React Flow + TypeScript

- **DAG Editor**: Drag blocks from a sidebar palette onto a canvas. Draw edges from output handles to input handles. Each node displays its block type, parameters, and status (stale / running / complete / error).
- **Node Inspector Panel**: Click a node to see its parameters (editable), its checkpoint provenance (full history chain), output preview (first N rows of data, or rendered image), and execution time/status.
- **Execution Controls**: "Run from here" on any node. "Run all stale" button. "Run full pipeline" button.
- **Image Gallery**: Nodes that produce images display thumbnails inline on the canvas. Click to expand.
- **In-app File Browser**: Native file picker for selecting data files and output paths directly from the canvas.
- **Settings & Workspace**: Preferences modal and guided workspace setup wizard on first launch.
- **Onboarding Tour**: Interactive walkthrough covering the palette, canvas, inspector, and execution controls.

### Backend: FastAPI + Python
*Coming Soon: Migration to Rust for faster core engine and block execution, with a Python compatibility layer for blocks.*

- **Block Registry**: Discovers and catalogs all available block classes. Serves the palette to the frontend.
- **Pipeline Engine**: Receives the DAG definition from the frontend. Topologically sorts. Determines which nodes are stale (by comparing current provenance hash against stored checkpoint provenance). Executes stale nodes in dependency order.
- **Checkpoint Store**: Manages the `checkpoints/` directory. Each checkpoint is a directory containing the data file, a `provenance.json`, and any produced images.

### Communication

- REST for CRUD operations (save/load pipelines, list blocks, get checkpoint previews).
- WebSocket for execution progress (node status updates, streaming logs).
- MCP for agent workflows over stdio (`python -m Forge mcp`) and mounted streamable HTTP (`/mcp`).

### MCP Server

To get started with LLM Agent workflows, you can copy a customized setup prompt from the "MCP" section of the settings. Paste this into your favorite coding agent's chat interface (e.g. Codex or Claude Code) and the agent should be able to finish it's own setup. You may need to reboot the agent after setup to ensure it picks up the new tools.

- Agents can list block types and inspect block docs, params, inputs, and outputs.
- Agents can create/open/save draft pipelines, add or remove blocks and edges, manage groups, and run `prettify`.
- `run_pipeline` is non-blocking for MCP clients; use `poll_run` with the returned `run_id` until the run reaches a terminal state.
- `run_pipeline_and_wait` provides a blocking one-shot alternative when a client wants a single terminal payload.
- `inspect_pipeline` returns a compact graph summary for low token usage.
- `render_pipeline_mermaid` now returns only a small top-level Mermaid chunk map (or a scoped Mermaid when `target_group` is provided), and `inspect_group` lets agents drill into one chunk at a time without paying for the full graph context up front.
- `inspect_results` and `inspect_results_many` return cropped tabular previews, full output shapes, and attached images when checkpoints exist.
- `apply_pipeline_spec` and `batch_upsert_graph` upsert groups, nodes, and edges from one declarative spec payload.
- `set_groups` and `batch_group_membership` batch-edit node memberships across existing groups.
- `list_block_presets`, field-backed `describe_block_type` metadata (`param_schema`, `required_params`, `param_examples`), `get_result_asset`, `render_result_image`, and `validate_draft` support safer agent authoring and preflight checks.
- `create_new_block` returns the repo-bundled Forge block-authoring skill and a ready-to-use prompt.

---

## Core Concepts

### Blocks

A block is a Python class that defines a single atomic data operation. All blocks inherit from `BaseBlock`.

```python
# blocks/normalize.py

from backend.block import BaseBlock, BlockOutput, BlockParams, BlockValidationError, block_param
import pandas as pd

class MedianCenterRows(BaseBlock):
    name = "Median Center Rows"
    version = "1.0.0"
    category = "Transform"
    description = "Subtract each row's median, centering response profiles."
    usage_notes = ["Input must contain at least one numeric column."]

    class Params(BlockParams):
        pass  # No parameters.

    def validate(self, data: pd.DataFrame) -> None:
        if len(data.select_dtypes(include="number").columns) == 0:
            raise BlockValidationError("No numeric columns to center.")

    def execute(self, data: pd.DataFrame, params: Params) -> BlockOutput:
        numeric = data.select_dtypes(include="number")
        centered = numeric.sub(numeric.median(axis=1), axis=0)
        result = data.copy()
        result[numeric.columns] = centered
        return BlockOutput(data=result)
```

A block with parameters, presets, and usage guidance:

```python
class KMeansClustering(BaseBlock):
    name = "K-Means Clustering"
    version = "1.0.0"
    category = "Clustering"
    description = "Cluster rows with K-Means and append a cluster assignment column."
    input_labels = ["DataFrame"]
    output_labels = ["DataFrame + Cluster"]
    usage_notes = [
        "Leave `columns` empty to use all numeric columns.",
        "Enable `standardize` if features have different scales.",
    ]
    presets = [
        {
            "id": "three_clusters",
            "label": "Three Clusters",
            "description": "Small deterministic clustering setup for quick exploration.",
            "params": {"n_clusters": 3, "random_state": 0},
        }
    ]

    class Params(BlockParams):
        n_clusters: int = block_param(6, description="Number of clusters to fit.", example=3)
        random_state: int = block_param(0, description="Random seed for reproducibility.")
        columns: str | None = block_param(
            None, description="Comma-separated columns to cluster on. Empty = all numeric."
        )
        standardize: bool = block_param(False, description="Standardize features before fitting.")
        output_column: str = block_param("cluster_id", description="Name of the output cluster label column.")

    def execute(self, data: pd.DataFrame, params: Params) -> BlockOutput:
        from sklearn.cluster import KMeans
        numeric = data.select_dtypes(include="number")
        km = KMeans(n_clusters=params.n_clusters, random_state=params.random_state)
        labels = km.fit_predict(numeric)
        result = data.copy()
        result[params.output_column] = labels
        return BlockOutput(data=result, metadata={"inertia": float(km.inertia_)})
```

A block that produces images (pass-through data, images as side effects):

```python
class ClusterHeatmap(BaseBlock):
    name = "Cluster Profile Heatmap"
    version = "1.0.0"
    category = "Visualization"
    description = "Z-scored heatmap of mean feature values per cluster."

    class Params(BlockParams):
        cluster_column: str = block_param("cluster_id", description="Column containing cluster labels.")
        cmap: str = block_param("RdBu_r", description="Matplotlib colormap name.")
        figsize_w: float = block_param(12.0, description="Figure width in inches.")
        figsize_h: float = block_param(6.0, description="Figure height in inches.")

    def execute(self, data: pd.DataFrame, params: Params) -> BlockOutput:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if params.cluster_column not in data.columns:
            raise BlockValidationError(f"Column '{params.cluster_column}' not found.")
        numeric_cols = data.select_dtypes(include="number").columns.drop(params.cluster_column, errors="ignore")
        profiles = data.groupby(params.cluster_column)[numeric_cols].mean()
        z_scored = (profiles - profiles.mean()) / profiles.std()
        fig, ax = plt.subplots(figsize=(params.figsize_w, params.figsize_h))
        sns.heatmap(z_scored, cmap=params.cmap, center=0, ax=ax)
        return BlockOutput(data=data, images=[fig])  # data passes through; image is a side effect
```

A source block (no inputs) with a file-picker parameter:

```python
class LoadCSV(BaseBlock):
    name = "Load CSV"
    version = "1.0.0"
    category = "IO"
    description = "Load a CSV file into a DataFrame."
    n_inputs = 0
    output_labels = ["DataFrame"]

    class Params(BlockParams):
        filepath: str = block_param(description="CSV file to load.", browse_mode="open_file")
        sep: str = block_param(",", description="Delimiter used in the file.")
        encoding: str = block_param("utf-8", description="File encoding.")

    def execute(self, data, params: Params) -> BlockOutput:
        df = pd.read_csv(params.filepath, sep=params.sep, encoding=params.encoding)
        return BlockOutput(data=df)
```

### Block Base Class

```python
# backend/block.py (public interface)

@dataclass
class BlockOutput:
    data: pd.DataFrame                        # primary output (also stored as outputs["output_0"])
    outputs: dict[str, pd.DataFrame] = ...   # named outputs for multi-output blocks
    images: list[Any] = ...                   # matplotlib figures; saved as PNGs in the checkpoint
    metadata: dict[str, Any] = ...           # arbitrary execution metadata

class BlockParams(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    # Null values for non-optional fields are automatically coerced to their defaults.

def block_param(default=..., *, description=None, example=..., browse_mode=None) -> Any:
    """Pydantic Field with Forge metadata.
    browse_mode: 'open_file' | 'save_file' | 'directory'  — renders a file-picker in the UI.
    """

class BaseBlock(ABC):
    name: str                          # display name shown in the palette and on canvas nodes
    version: str                       # bumping this marks all instances stale
    category: str                      # palette grouping
    description: str = ""             # tooltip shown in the palette
    usage_notes: list[str] = []       # bullet points shown in the node inspector
    presets: list[dict] = []          # one-click param templates shown in the inspector
    n_inputs: int = 1                 # 0 for source blocks, 2+ for multi-input blocks
    input_labels: list[str] = []      # per-handle labels shown on the node
    output_labels: list[str] = ["output"]
    always_execute: bool = False      # skip staleness check (use for side-effect blocks like ExportCSV)

    @abstractmethod
    def execute(self, data, params=None) -> BlockOutput: ...

    def validate(self, data) -> None:
        """Raise BlockValidationError when preconditions fail."""

class BlockValidationError(Exception):
    pass
```

### Provenance

Every checkpoint carries a `provenance.json` that fully describes how the data got to that state:

```json
{
  "checkpoint_id": "a3f8c1d2",
  "block_name": "K-Means Clustering",
  "block_version": "1.0.0",
  "params": {
    "n_clusters": 6,
    "random_state": 0,
    "output_column": "cluster_id"
  },
  "parent_checkpoint_ids": ["b7e2a4f1"],
  "initial_data_signature": "sha256:9f86d08...",
  "history_hash": "sha256:4e1c3b2...",
  "timestamp": "2026-02-23T14:30:00Z",
  "execution_time_seconds": 2.3,
  "output_shape": [26578, 20],
  "images": ["cluster_heatmap_a3f8c1d2.png"]
}
```

The `history_hash` is computed as:

```python
history_hash = sha256(
    parent_history_hash
    + block_name
    + block_version
    + canonical_json(params)
)
```

For the root "Load Data" block, the `parent_history_hash` is replaced by the `initial_data_signature` (hash of the raw input file).

This means:
- Changing a parameter at any node changes the `history_hash` for that node and all descendants.
- Bumping a block version changes the `history_hash` for all nodes using that block and all their descendants.
- The engine compares the stored `history_hash` on each checkpoint against the computed `history_hash` from the current DAG definition to determine staleness.

### Checkpoint Store

```
checkpoints/
├── a3f8c1d2/
│   ├── data.parquet
│   ├── provenance.json
│   └── images/
│       └── cluster_heatmap_a3f8c1d2.png
├── b7e2a4f1/
│   ├── data.parquet
│   └── provenance.json
└── ...
```

- Data stored as Parquet (fast, columnar, typed, smaller than CSV).
- Old checkpoints are never deleted automatically. A `Forge gc` command can prune orphaned checkpoints not referenced by any saved pipeline.
- The engine resolves "should I rerun this node?" by: compute the expected `history_hash` from the DAG → check if a checkpoint with that hash exists → if yes, skip; if no, execute.

### Pipeline Definition

Saved as JSON. This is what the frontend sends to the backend and what gets persisted to disk.

```json
{
  "name": "OVCA Target Discovery",
  "nodes": [
    {
      "id": "node_1",
      "block": "LoadCSV",
      "params": {"filepath": "data/auc_matrix.csv"},
      "notes": "Primary input matrix",
      "group_ids": ["group_ingest"],
      "position": {"x": 100, "y": 200}
    },
    {
      "id": "node_2",
      "block": "MedianCenterRows",
      "params": {},
      "group_ids": ["group_qc"],
      "position": {"x": 350, "y": 200}
    }
  ],
  "edges": [
    {
      "id": "edge_node_1_node_2",
      "source": "node_1",
      "target": "node_2",
      "source_output": 0,
      "sourceHandle": "output_0",
      "target_input": 0,
      "targetHandle": "input_0"
    }
  ],
  "groups": [
    {
      "id": "group_qc",
      "name": "QC",
      "description": "Quality-control transforms",
      "comment_id": "comment_qc"
    }
  ],
  "comments": [
    {
      "id": "comment_qc",
      "title": "QC",
      "description": "Quality-control transforms",
      "position": {"x": 60, "y": 150},
      "width": 520,
      "height": 220,
      "managed": true,
      "group_id": "group_qc"
    }
  ]
}
```

Additional notes:

- Stable edge IDs, node notes, node group memberships, root-level groups, and managed comments are durable pipeline metadata.
- Layout and organization metadata must round-trip through the frontend and MCP server, but must not affect provenance hashing or staleness.
- Multi-input execution still depends on ordered input slots; edge IDs are identifiers, not execution order.

### MCP Drafts

- `create_pipeline` and `open_pipeline` create mutable server-side drafts and mark them active for the calling client.
- Most MCP tools default to the active draft, but also accept an explicit `draft_id`.
- `save_pipeline` persists the draft without running it.
- `run_pipeline` saves first, starts execution for that exact saved snapshot, and returns a `run_id`.
- `poll_run` returns `running`, `completed`, `error`, `cancelled`, or `timed_out` along with the accumulated run payload.

---

## Staleness Propagation

When a user changes a parameter on a node:

1. Frontend sends the parameter update to the backend.
2. Backend recomputes the `history_hash` for that node.
3. Backend walks the DAG forward from that node, recomputing `history_hash` for every descendant.
4. Any node whose recomputed hash doesn't match its stored checkpoint is marked **stale**.
5. Frontend updates node status indicators (yellow border = stale).

When a user bumps a block version (edits the Python source):

1. Backend detects the version change via the block registry.
2. All nodes using that block type are marked stale.
3. Staleness propagates downstream as above.

Execution then runs only the stale subgraph, in topological order.

---

## Project Structure

```
Forge/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, CORS, lifespan
│   ├── api/
│   │   ├── __init__.py
│   │   ├── blocks.py            # GET /blocks (registry)
│   │   ├── pipelines.py         # CRUD /pipelines
│   │   ├── execution.py         # POST /execute, WebSocket /ws/execute
│   │   └── checkpoints.py       # GET /checkpoints/{id}/preview
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── runner.py            # Topological sort, staleness check, execution loop
│   │   ├── execution_manager.py # Background execution + cancellation
│   │   ├── provenance.py        # Hash computation, history chain
│   │   └── checkpoint_store.py  # Read/write checkpoints, GC
│   ├── block.py                 # BaseBlock, BlockOutput, BlockValidationError
│   ├── block_authoring.py       # Forge block-authoring prompt helpers
│   ├── document_service.py      # MCP draft lifecycle + graph mutations
│   ├── mcp_server.py            # FastMCP tool/prompt registry
│   ├── pipeline_graph.py        # Pipeline graph helpers
│   ├── pipeline_layout.py       # Deterministic prettify layout
│   ├── registry.py              # Auto-discover block classes from blocks/
│   └── schemas.py               # Shared pipeline and API models
│
├── blocks/                       # User-defined blocks (auto-discovered)
│   ├── __init__.py
│   ├── io.py                    # LoadCSV, LoadParquet, ExportCSV
│   ├── transform.py             # MedianCenterRows, MedianCenterCols, ZScore, FilterRows
│   ├── factorization.py         # ALSFactorization, NMFFactorization
│   ├── clustering.py            # KMeansClustering, SpectralClustering, HDBSCAN
│   ├── visualization.py         # ClusterHeatmap, UMAPPlot, DistributionHistogram
│   └── combine.py               # MergeDatasets, AppendDatasets, JoinOnIndex
│
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── Canvas.tsx        # React Flow canvas
│       │   ├── BlockNode.tsx     # Custom node component
│       │   ├── BlockPalette.tsx  # Sidebar with draggable blocks
│       │   ├── NodeInspector.tsx # Parameter editor + preview panel
│       │   ├── Toolbar.tsx       # Run controls, save/load
│       │   └── ImagePreview.tsx  # Lightbox for visualization outputs
│       ├── hooks/
│       │   ├── useWebSocket.ts   # Execution progress
│       │   └── usePipeline.ts    # DAG state management
│       ├── api/
│       │   └── client.ts         # REST + WS client
│       └── types/
│           └── pipeline.ts       # TypeScript types mirroring backend models
│
├── skills/
│   └── forge-block-author/      # Repo-bundled skill used by create_new_block
│
├── checkpoints/                  # Generated at runtime, gitignored
│
└── pipelines/                    # Saved pipeline JSON files (gitignored; created at runtime)
```

---

## Block Library

60+ blocks ship out of the box, organized by category.

**IO**

| Block | Inputs | Notes |
|---|---|---|
| `Load CSV` | 0 | File-picker param; configurable delimiter, encoding, index column |
| `Export CSV` | 1 | Always re-executes (side effect); pass-through |
| `Constant` | 0 | Injects a scalar or JSON value as a single-cell DataFrame |
| `No-Op` | 1 | Pass-through; useful for layout and grouping |

**Transform** (23 blocks, representative selection)

| Block | Notes |
|---|---|
| `Filter Rows` | Comparison operators: eq, ne, gt, lt, ge, le, contains, startswith |
| `Select Columns` | Keep or drop by name list |
| `Median Center Rows / Cols` | Subtract row or column medians |
| `Z-Score Normalize` | Per-column standardization |
| `Pivot Table` | index × columns × values aggregation |
| `Melt Columns` | Unpivot wide to long |
| `Impute Missing Values` | mean, median, constant, forward-fill strategies |
| `Filter Columns By Coverage` | Drop columns below a null-fraction threshold |
| `Cast Columns` | Type coercion |
| `Mask Outliers (MAD)` | Replace outliers with NaN using median absolute deviation |
| + 13 more | Reorder, Sort, Deduplicate, Reset Index, Transpose, Split List Column, … |

**Clustering**

| Block | Notes |
|---|---|
| `K-Means Clustering` | Appends cluster label column; optional column subset and standardization |

**Dimensionality**

| Block | Notes |
|---|---|
| `UMAP Embed` | Appends UMAP_0, UMAP_1, … columns |

**Factorization**

| Block | Notes |
|---|---|
| `Weighted ALS Factorization` | 2 inputs (data + weight matrix) |
| `Nuisance ALS Sweep` | 2 inputs; sweeps nuisance correction |
| `Nuisance ALS` | 4 inputs; full nuisance-corrected factorization |

**Visualization** (all pass data through; images saved as PNG artifacts)

| Block | Notes |
|---|---|
| `Cluster Profile Heatmap` | Z-scored cluster mean heatmap |
| `Matrix Heatmap` | General-purpose heatmap with annotations |
| `Matrix Histogram` | Per-column or global histograms |
| `Matrix Bar Chart` | Grouped or stacked bar charts |
| `Matrix Line Chart` | Multi-series line chart |
| `Matrix Scatter Plot` | Column-pair scatter |
| `Highlighted Scatter Plot` | Scatter with configurable highlight groups |
| `Highlighted Bar Chart` | Bar chart with highlight groups |
| `Faceted Scatter Plot` | Grid of scatter plots by facet variable |
| `Annotate Plot With Arrows` | 2 inputs; overlays arrow annotations on an existing plot |
| `Matrix 3D Scatter Plot` | Three-axis scatter |

**Operator**

| Block | Notes |
|---|---|
| `Add / Subtract / Multiply / Divide Columns` | 2 inputs; element-wise column arithmetic |
| `Log Columns` | 2 inputs; log-transform with base from second input |
| `Absolute Value Column` | Single-column abs |
| `Multiply Many Columns` | Element-wise product across multiple columns |
| `Multiply DataFrames` | 2 inputs; element-wise DataFrame product |
| `Append Datasets` | 2 inputs; row-wise concatenation |
| `Merge Datasets` | 2 inputs; join on shared key column |

**Statistics** (12 blocks)

| Block | Notes |
|---|---|
| `Group Aggregate` | groupby + agg (mean, sum, count, …) |
| `Group Pair Metrics` | Pairwise group comparison metrics |
| `Group Mean By Assignments` | 2 inputs; map group assignments to mean profiles |
| `Coverage By Group` | Non-null fraction per group |
| `Exponential / Linear Scaled Weight` | Compute observation weights from a score column |
| `Align To Reference Matrix` | 2 inputs; reindex rows/columns to match a reference |
| `Mask By Reference Observed` | 2 inputs; mask values absent in a reference |
| `Min / Mean / Count Non-Null Across Columns` | Row-wise aggregation to a new column |
| `Assign Tier By Thresholds` | Bin a numeric column into labeled tiers |

---


## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12+, FastAPI, uvicorn, Pydantic v2 |
| Data | pandas, numpy, Parquet (pyarrow) |
| ML/Stats | scikit-learn, umap-learn, hdbscan, scipy |
| Visualization | matplotlib, seaborn |
| Frontend | React 18, TypeScript, Vite |
| DAG Editor | React Flow |
| Styling | Tailwind CSS |
| Communication | REST + WebSocket + MCP |
| Linting | Ruff (backend), ESLint + Prettier (frontend) |
| Type Checking | Pylance (backend), TypeScript strict (frontend) |

---

## Key Design Decisions

1. **Parquet over CSV/SQLite for checkpoints.** Fast columnar reads, preserves dtypes, compresses well. A 26K × 20 DataFrame is ~2MB as Parquet vs ~15MB as CSV.

2. **Blocks are classes, not functions.** Classes give us `Params(BlockParams)`, `validate()`, versioning, presets, and a clean registry pattern. Functions are simpler but don't carry enough metadata.

3. **Provenance is a hash chain, not a log.** The hash chain means equality checking is O(1) — compare two hashes. If they match, the data is identical regardless of when it was computed. This is what makes "revert a change and skip recomputation" work for free.

4. **The frontend is the composition layer.** The JSON pipeline definition is the source of truth. The Python side never needs to know about node positions or UI state. This keeps the engine testable and the frontend replaceable.

5. **Multi-input blocks use ordered lists, not named inputs.** For merge/append with 2 inputs, `data[0]` is left and `data[1]` is right. The edge order in the JSON determines the mapping. Named inputs are an option for Phase 4 if this proves confusing.

6. **Images are artifacts, not data.** Visualization blocks pass the DataFrame through unchanged and produce images as side effects. This means you can always wire a visualization block in parallel with a computation block without affecting the data flow.

7. **MCP drafts mutate the same pipeline schema.** MCP adds a draft/document layer and organizational metadata, but saved pipelines remain the source of truth and execution still runs against persisted JSON snapshots.

---

## Running

### Desktop app

Download the `.msi` (Windows) or `.dmg` (macOS) from the [Releases](https://github.com/Jonpot/forge/releases) page and run the installer. The app starts the Python backend automatically.

### Development setup

```bash
# Backend
python -m venv .venv
.venv\Scripts\activate        # Windows — use source .venv/bin/activate on Unix
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload --port 40964

# MCP over stdio
python -m Forge mcp

# Frontend (Vite dev server, connects to the running backend)
cd frontend
npm install
npm run dev
```

When the FastAPI app is running, the MCP server is also mounted at `/mcp`.

