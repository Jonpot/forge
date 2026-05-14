# Forge Pipeline Quality Bar

Use this checklist when authoring or auditing a pipeline with more than 5 block nodes, multiple branches, or user-visible organization work.

## Group Coverage

- Count only block nodes. Ignore comments and group annotations.
- Compute grouped coverage as:
  `nodes with at least one group_id / total block nodes`
- Minimum bar for a finished non-trivial pipeline: `>= 90%`
- Preferred bar for new or freshly refactored pipelines: `100%`

## What Counts As A Well-Defined Group

- The name communicates a stage or outcome, not a vague bucket.
- The group contains nodes that belong to the same phase of work.
- The group boundary helps a human understand the DAG at a glance.
- The group is not so broad that unrelated transforms, modeling, and exports all land together.

Good examples:
- `Data Loading`
- `Feature Prep`
- `Clustering`
- `Viz/Export`

Weak examples:
- `Stuff`
- `Pipeline`
- `Transforms 2`
- `Misc`

## Context Checklist

Build context before editing:

1. Confirm the target draft or pipeline.
2. Inspect the current graph if it already exists. For pipelines with more than ~10 nodes, call `render_pipeline_mermaid()` first to get the top-level structural chunk DAG, then drill into the relevant chunk with `inspect_group(target_group=...)`.
3. Read block schemas for every block whose params will be set or changed.
4. Identify root inputs, side effects, terminal outputs, and visualization branches.
5. Infer stage boundaries from the user goal and expected artifacts.

## Authoring Sequence

1. Open or create the draft.
2. Inspect pipeline state. For non-trivial existing pipelines, run `render_pipeline_mermaid()` to orient on chunk topology before editing.
3. Read relevant block schemas and presets.
4. Apply graph edits, preferably through `apply_pipeline_spec` or `batch_upsert_graph`.
5. Assign groups with `set_groups` or `batch_group_membership`.
6. Run `validate_draft`.
7. Run `prettify` after structure stabilizes.
8. Optionally add manual annotations with `add_comment(member_ids=[...])` for callouts that should not be managed by prettify.
9. Save or run the draft.
10. Inspect key outputs and artifacts.

## Handoff Checklist

- Pipeline name matches the user task.
- Node IDs are stable and meaningful.
- Required params are set explicitly when defaults would be ambiguous.
- Export paths and other side-effect paths are explicit.
- Important nodes have inspected previews or assets after execution.
- If execution failed, report the exact failing node and message.

## Exceptions

- Very small drafts can use fewer groups, but the final graph should still feel intentionally organized.
- If the user is actively hand-arranging the canvas, avoid `prettify` unless requested.
- If the user wants a sketch or brainstorming graph, temporary low coverage is acceptable during exploration, but group before final delivery.
