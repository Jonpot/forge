# Forge Draft Surface

Use this reference when you are authoring a pipeline from scratch or recovering from MCP payload-shape friction.

## First Stops

1. Call `describe_pipeline_spec` before building a large `apply_pipeline_spec` payload.
2. Call `describe_block_type` before setting params on any block you have not used recently.
3. Trust `usage_notes` first; only open block source when the metadata appears incomplete or wrong.

## Orienting on a Pipeline: `render_pipeline_mermaid`

Use this tool whenever you open an unfamiliar or large pipeline and need to understand its structure before editing.

**Top-level view** — highest-level chunk DAG only. Start here.

```json
{}
```

Returns a Mermaid `graph TD` where each top-level chunk is a root-most non-empty group, plus explicit orphan regions when ungrouped work cannot be attached unambiguously. Clear ungrouped scaffolding is absorbed into the strongest neighboring chunk. The response is intentionally small: Mermaid only.

**Scoped view** — render only one chunk/group.

```json
{ "target_group": "group_conf_allocate" }
```

Returns only the next-level Mermaid view inside that target. Use this when you already know which chunk you want to inspect and only need the structure under that one scope.

## Drilling In: `inspect_group`

Use this after `render_pipeline_mermaid()` when you want a small structured payload for a specific chunk or subgroup.

```json
{ "target_group": "group_conf_allocate" }
```

Returns:
- the target id, name, kind, and node count
- a small `children` list with child ids, names, kinds, node counts, and whether to recurse with `inspect_group` or switch to `inspect_block`
- a Mermaid graph for just that scope

This is the primary drill-down workflow for LLMs:
1. `render_pipeline_mermaid()` for the top-level map
2. `inspect_group(target_group=...)` for one chunk
3. recurse with `inspect_group` on child ids as needed
4. switch to `inspect_block` once the child says `inspect_with="inspect_block"`

## Manual Comment Blocks: `add_comment`

Creates a non-managed comment annotation block positioned around a set of elements.

**With `member_ids` (preferred):** pass node IDs and/or existing comment IDs; the tool computes the bounding box with standard padding automatically.

```json
{
  "title": "Outlier Handling",
  "description": "Clips and imputes before model input",
  "member_ids": ["clip_outliers", "impute_missing", "log_transform"]
}
```

**With raw coordinates (fallback):** use when no member elements exist yet or you need precise placement.

```json
{
  "title": "Note",
  "description": "Placeholder for future steps",
  "x": 80,
  "y": 600,
  "width": 400,
  "height": 150
}
```

**Managed vs manual:** `add_comment` always creates a *manual* comment (`managed: false`). It will not be repositioned when `prettify` runs. If you want a comment block that tracks a group's layout automatically, create a group and call `prettify` — it will auto-generate and maintain that group's managed comment block.

## `add_block`

- `params` may be:
  - a JSON object
  - a JSON string that decodes to an object
- `group_ids` may be:
  - a list of group IDs
  - a JSON string list
  - a comma-delimited string

Example:

```json
{
  "block_key": "LoadCSV",
  "node_id": "load_sales",
  "params": {"filepath": "C:\\Users\\you\\sales.csv"},
  "group_ids": ["group_load"]
}
```

## `apply_pipeline_spec`

Top-level keys:

- `name`
- `groups`
- `nodes`
- `edges`

Group spec:

```json
{
  "id": "group_load",
  "name": "Loading",
  "description": "Input roots",
  "member_node_ids": ["load_sales"]
}
```

Node spec:

```json
{
  "id": "scatter",
  "block": "MatrixScatterPlot",
  "params": {
    "x_column": "revenue",
    "y_column": "profit",
    "color_column": "segment",
    "color_mode": "categorical"
  },
  "group_ids": ["group_viz"]
}
```

Edge spec:

```json
{
  "id": "edge_load_scatter",
  "source": "load_sales",
  "target": "scatter"
}
```

Multi-input edge example:

```json
{
  "source": "assignments",
  "target": "group_means",
  "target_input": 1
}
```

## Behavior Notes Worth Watching

- `GroupMeanByAssignments` aligns its two inputs by row index before grouping.
- `MergeDatasets` treats input 0 as left and input 1 as right.
- `UMAPEmbed` appends embedding columns to the incoming frame.
- `KMeansClustering` appends the output cluster column to the incoming frame.
- `MatrixScatterPlot` uses a continuous color scale for numeric dtypes in `auto` mode and a discrete legend for string/object dtypes. Set `color_mode="categorical"` when numeric cluster labels should behave like categories.
