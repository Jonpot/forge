---
name: forge-pipeline-author
description: Create, extend, reorganize, validate, run, and inspect Forge pipelines in this repository, preferably through the Forge MCP draft tools. Use when an agent needs to author or modify Forge DAGs, choose block params from schemas or presets, batch-apply nodes/edges/groups, improve group coverage and layout quality, or inspect run outputs and visualization artifacts.
---

# Forge Pipeline Author

Author Forge pipelines through the draft and MCP layer instead of ad hoc JSON edits. Build context first, prefer declarative graph updates, keep groups intentional, and leave the pipeline in a valid, inspectable state.

## Workflow

1. Build context before mutating.
   - Open or inspect the relevant draft or pipeline first.
   - On any pipeline with more than ~10 nodes or non-trivial grouping, call `render_pipeline_mermaid()` immediately after opening. This returns only the highest-level chunk DAG: root-most non-empty groups become top-level chunks, clear ungrouped scaffolding is absorbed into its strongest neighboring chunk, and ambiguous orphan regions stay explicit.
   - When you need to drill in, call `inspect_group(target_group=...)` on a returned chunk id. This gives you a small child list plus a next-level Mermaid for just that scope. Repeat recursively instead of asking for the whole graph at once.
   - Read `describe_pipeline_spec` before using `add_block`, `apply_pipeline_spec`, or `batch_upsert_graph` when the payload shape is not already obvious.
   - Read block contracts with `describe_block_type` before setting params.
   - Prefer `describe_block_type` output, especially `param_schema`, `required_params`, `param_examples`, `presets`, and `usage_notes`, over opening block source.
   - Only read `blocks/*.py` when the MCP metadata is missing, contradictory, or appears buggy.
   - Read `list_block_presets` when the block has common setup patterns.
   - Reuse existing node IDs, edge IDs, notes, groups, and comments when editing an existing pipeline.
2. Prefer declarative edits for multi-node work.
   - Use `apply_pipeline_spec` or `batch_upsert_graph` for graph creation or larger refactors.
   - Use single-node tools only for small targeted edits or repair work.
3. Organize the graph by stage, not by implementation detail.
   - Create groups for meaningful phases such as loading, transforms, clustering, reporting, or viz/export.
   - Target at least 90% grouped coverage for non-trivial pipelines.
   - Target 100% grouped coverage when authoring from scratch unless the user explicitly wants a rough sketch.
   - Use short, concrete group names. Avoid catch-all names like `misc` or `processing`.
   - Use `add_comment(title, member_ids=[...])` to place manual annotation boxes around a specific set of block nodes and/or existing comment blocks. Pass `member_ids` and the tool computes the bounding box automatically; avoid raw coordinate inputs. Use this for cross-group callouts, phased sub-annotations, or any label that should not be auto-managed by `prettify`.
4. Use context to shape the graph.
   - Infer stage boundaries from the user goal, input/output files, and expected artifacts.
   - Put side-effect blocks such as exports in terminal groups.
   - Put visualization blocks on parallel branches when they should not change downstream data semantics.
   - Keep multi-input ordering explicit with `target_input` when the block arity is greater than 1.
5. Preflight before execution.
   - Run `validate_draft` after structural edits.
   - Fix blocking issues before running: missing required params, missing root files, incomplete wiring, invalid input slots, or cycles.
   - Run `prettify` once the structure is stable, unless the user explicitly wants to preserve a manual layout.
6. Execute and inspect deliberately.
   - Use `run_pipeline_and_wait` when a blocking terminal result is acceptable.
   - Otherwise use `run_pipeline` followed by `poll_run`.
   - Use `inspect_results_many` for tabular previews.
   - Use `render_result_image` and `get_result_asset` for visualization outputs and exact artifact paths.

## Friction Guardrails

- Read [references/draft-surface.md](references/draft-surface.md) when you are authoring from scratch, composing a larger declarative spec, or recovering from a payload-shape error.
- `add_block` accepts `params` as either an object or a JSON string.
- `add_block` accepts `group_ids` as a list, a JSON string list, or a comma-delimited string.
- `describe_block_type.param_schema` is field-backed. Trust `required=true` there over any old habit of interpreting blank-string defaults.
- Use `target_input` explicitly for multi-input blocks unless you intentionally want Forge to choose the next open slot.
- For scatter plots where cluster IDs are numeric labels, set `color_mode="categorical"` instead of inventing a workaround column.
- `render_pipeline_mermaid` is intentionally minimal now: it returns only Mermaid. Use `inspect_group` to drill into one scope at a time, and `inspect_block` once you reach node-level children.
- `add_comment` with `member_ids` creates a **manual** (non-managed) comment block and will not be repositioned by `prettify`. If you want a comment block that stays anchored to a group's nodes through layout changes, create a group and let `prettify` manage its comment instead.

## Quality Bar

- Read [references/quality-bar.md](references/quality-bar.md) when authoring a pipeline with more than 5 blocks, multiple branches, or group/comment cleanup work.
- Treat these as minimum expectations unless the user asks for a temporary sketch:
  - No blocking `validate_draft` errors.
  - Stable, meaningful pipeline name and node IDs.
  - At least 90% of block nodes assigned to well-defined groups.
  - `prettify` applied after major structural changes.
  - If execution was requested, a terminal run result plus inspected outputs or artifacts for the important nodes.

## Repo Rules

- Prefer Forge MCP tools over shell scripts or backend HTTP when MCP is available.
- Preserve saved-pipeline metadata such as notes, group IDs, comments, edge IDs, and layout unless the task is specifically to reorganize them.
- Do not treat layout or organization metadata as provenance inputs.
- Save or run the draft so the persisted JSON remains the source of truth.
- If you discover yourself reading repo code to infer routine MCP payload shapes or block behavior, stop and check whether `describe_pipeline_spec` or `describe_block_type.usage_notes` already answers it.

## Pointers

- Pipeline behavior and contracts: `README.md`
- MCP tools: `backend/mcp_server.py`
- Draft mutations and validation: `backend/document_service.py`
- Reusable quality rubric: [references/quality-bar.md](references/quality-bar.md)
