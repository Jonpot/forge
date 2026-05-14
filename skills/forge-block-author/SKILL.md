---
name: forge-block-author
description: Create or update a Forge atomic block in this repository. Use when an agent needs to add a new Python-defined block under blocks/, wire it to the existing BaseBlock contract, and add matching tests.
---

# Forge Block Author

Create blocks for Forge by following the existing backend contract instead of inventing a new pattern.

## Workflow

1. Inspect similar blocks in `blocks/` for the same category and arity.
2. Implement a new `BaseBlock` subclass in `blocks/`.
3. Set required metadata: `name`, `version`, `category`.
4. Define self-describing metadata so pipeline authors do not need to read source for routine usage:
   - `description`
   - `input_labels`
   - `output_labels`
   - `usage_notes` when behavior is not obvious from the params alone
5. Add a nested `Params(BlockParams)` class with typed fields for every persisted parameter.
   - Use `block_param(...)` for required fields, descriptions, and examples.
   - Prefer field metadata over block-level `param_descriptions`.
6. Implement `validate(data)` for precondition failures.
   - Raise `InsufficientInputs` when a required input slot is not connected — the engine skips the block silently.
   - Raise `BlockValidationError` for data-quality failures (missing columns, wrong shape, etc.) that should surface as a pipeline error.
   - Multi-input blocks that require all inputs should check `any(d is None for d in data[:n])` and raise `InsufficientInputs`.
   - Blocks with optional inputs should check only the truly required slots and let `None` pass through to `execute`.
7. Implement `execute(data, params)` and return `BlockOutput`.
8. Preserve deterministic output handles:
   - Single output blocks should use the default `output_0`.
   - Multi-output blocks must populate `BlockOutput.outputs` with stable `output_{n}` keys that match `output_labels`.
9. Add focused tests in `tests/`:
   - Happy path behavior.
   - Validation or edge-case failures.
   - Multi-input or multi-output ordering when relevant.

## Repo-Specific Rules

- Put new blocks in the most appropriate existing module under `blocks/`; only create a new module if no current module fits.
- Keep block behavior atomic. Compose broader workflows as pipelines, not monolithic blocks.
- Do not bypass the existing registry or provenance model.
- If the block writes files as a side effect, mark it `always_execute = True` only when checkpoint reuse would be incorrect.
- Treat `describe_block_type` as a user-facing contract. If an agent would need to inspect the block source to understand input ordering, index alignment, appended columns, categorical vs numeric behavior, or pass-through semantics, add that detail to `usage_notes` or field metadata on `Params`.
- Requiredness must come from the `Params` model itself, not blank-string defaults or separate override tables.
- When you add enum-like or otherwise subtle params, make sure the field descriptions and examples are discoverable through `describe_block_type.param_schema`.

## Pointers

- Base contract: `backend/block.py`
- Registry-discovered metadata: `backend/registry.py`
- MCP block-schema surface: `backend/document_service.py`
- Engine output/provenance expectations: `backend/engine/runner.py`
- Existing tests to mirror: `tests/test_atomic_blocks.py`
