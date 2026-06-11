# *Forge (`starforge`) — Design

> **Status:** Locked design, M0 in progress. Last updated 2026-06-11.
> **Branding:** "*Forge" in UI and prose where the glyph renders; `starforge` for every identifier — packages, commands, marketplace IDs, file names.

*Forge brings Forge's canvas — checkpointing, provenance, stale/hydrate execution, visualization — into VS Code, operating directly on the repository the user already has open. Blocks are ordinary Python functions in the user's codebase, registered with a decorator. The user writes code in their real editor with their real tooling; *Forge is the orchestration and caching layer over it.

---

## 1. Why this exists

Forge v1 requires users to write blocks in a template format, import them into Forge, and test them there. That loop is the adoption ceiling: people see the vision (checkpointing, staleness, visualization) but won't move their code into our box.

PR #6 ("Feature/freecode") proved the demand — an Arbitrary Code block that made Forge usable as a primary IDE — and also demonstrated the cost of attacking it from inside the desktop app. That PR conflates two problems:

1. **Code identity** — *what code ran here?* Code in a text param has no file, no version, no history; provenance degrades. The PR's answer was to carry entire Python namespaces across edges and persist them, which balloons checkpoints and blurs staleness semantics.
2. **Data transport** — *what flows between blocks?* The PR's answer was "everything."

*Forge solves #1 structurally: blocks **are** functions in the repo, so code identity is the file — hashable, diffable, git-versioned. #2 is solved deliberately with a typed serializer registry (§8). PR #6 will not merge; its use cases are served by the decorator path and the snippet node (§10).

---

## 2. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Pipeline-as-document is the spine.** `.forge` documents reference repo functions by qualified name + source hash. Code → canvas is one-way resolution. | Live bidirectional code↔canvas sync means rewriting user Python from canvas edits — formatting, comments, merge conflicts. Script→pipeline conversion ships later as a **one-way importer** (M2). |
| 2 | **`@block` decorator is THE registration path.** Tiny zero-dependency package; explicit opt-in; the palette is exactly the functions the user blessed. | No heuristics, no palette noise. The "add undecorated function" gesture (CodeLens) *inserts the decorator into the source* — one mechanism, one source of truth, survives clones, shows up in PRs. |
| 3 | **Edges carry arbitrary Python objects.** No privileged DataFrame type. Runtime `TypeError`s from a bad edge are valid user errors. | DF-centrism was a v1 pain point (JSON→DataFrame adapter blocks). Static type-hint mismatch gets a *warning* squiggle, never a blocker — editors warn, runtimes err. |
| 4 | **Params and inputs are unified.** Every function parameter is simultaneously an input handle and an inspector field. Edge connected → value flows in; unconnected → inspector literal or signature default. Edges target parameter *names*, not slot indices. | Resolves Forge v1's ordered-slots design debt ("named inputs in Phase 4") for free. |
| 5 | **Source hash = automatic versioning.** Hash of the function's normalized AST replaces manual `version` bumps and slots into the existing history-hash chain design. | Edit the function → dependents go stale. No human in the loop. |
| 6 | **Staleness = Tier 2 by default** (§7): function hash + repo import-closure hash, plus an environment fingerprint. A settings slider can expose other tiers later. | Over-invalidates in the safe direction. False stale wastes a rerun; false fresh serves wrong data. |
| 7 | **Discovery is automatic and static.** AST scan, never importing user code. Event-driven: VS Code file-watcher triggers incremental re-parse; a reconcile sweep runs on window refocus. **No polling timers.** | Imports execute side effects and load heavy libraries. Events are fresher than polling and cost zero idle CPU. |
| 8 | **Slim stdio kernel, no HTTP server.** NDJSON JSON-RPC over stdio; runs inside the workspace venv; lazily spawned, idle-killed. Stdlib-only hard dependencies. | No uvicorn/FastAPI footprint, no ports, no firewall prompts, nothing imposed on the user's environment. |
| 9 | **Process-per-run execution.** Each run is a worker subprocess: cancellation = kill, memory returns to the OS, fresh imports pick up just-edited code with zero `importlib.reload` hacks. | Checkpointing makes cold processes cheap — only the stale subgraph executes anyway. |
| 10 | **`.forge/` folder in the user's repo.** `pipelines/` committed (PR-reviewable team artifacts), `checkpoints/` + `cache/` auto-gitignored. | Pipelines live next to the code they orchestrate; teammates clone and see the same canvas. |
| 11 | **Monorepo.** *Forge lives in the Forge repo under `starforge/`; the desktop app continues unchanged. | A separate repo or long-lived branch bitrots against main. See §11 for the core-engine relationship. |
| 12 | **Explicit Run button.** Auto-run-stale-on-save is a stretch goal at best. | Watch mode next to a 20-minute factorization node is a footgun. |
| 13 | **Snippet node** for one-liners: code text stored in the `.forge` doc, hashed as a param. | The legitimate heir of the freecode use case — convenience without namespace packaging. (M1.) |

---

## 3. Architecture

```
┌─ VS Code ────────────────────────────────────────────────┐
│  Extension host (TS, ~zero idle cost)                    │
│    • custom text editor for *.forge                      │
│    • .py file watcher → incremental index requests       │
│    • CodeLens (M1): "⊕ Add to palette" inserts decorator │
│    • spawns/kills kernel; NDJSON bridge                  │
│  Webview (loads only while a canvas is visible)          │
│    • React Flow canvas, palette, inspector, Run          │
└──────────────┬───────────────────────────────────────────┘
               │ NDJSON JSON-RPC over stdio
┌──────────────▼───────────────────────────────────────────┐
│  Kernel (python -m starforge.kernel, workspace venv)     │
│    • AST indexer (stdlib ast; no user-code imports)      │
│    • history-hash computation + checkpoint existence     │
│    • spawns worker per run; relays progress events       │
└──────────────┬───────────────────────────────────────────┘
               │ subprocess, NDJSON events on stdout
┌──────────────▼───────────────────────────────────────────┐
│  Worker (one per run, dies after)                        │
│    • imports user modules, executes stale subgraph       │
│    • serializer registry persists outputs to checkpoints │
└──────────────────────────────────────────────────────────┘
```

**Memory posture:** extension host does nothing until a `.forge` file opens or a command fires. The webview dies with its tab. The kernel imports only stdlib (pandas et al. exist only inside workers, which exist only during runs). Previews are cropped kernel-side; full objects never cross into the webview.

---

## 4. The decorator

```python
# pip install starforge-kernel   →   import starforge
from starforge import block

@block  # or @block(label="Clean AUC Matrix", category="QC", outputs=("clean", "stats"))
def clean_auc(raw: pd.DataFrame, min_coverage: float = 0.8) -> pd.DataFrame:
    """Drop sparse columns and median-center rows."""
    ...
```

Hard constraints — the decorator lives in *user production code*, so:

- **Zero dependencies, microsecond import.** The top-level `starforge` module contains only the decorator; kernel/core are submodules imported lazily and only by Forge itself.
- **Behavior-neutral.** Tags a metadata attribute and returns the function unchanged. Code runs identically under pytest, in CI, anywhere — Forge nowhere in sight.
- **Statically matchable.** The indexer finds it syntactically (import-aware: `@block`, `@block(...)`, `@starforge.block`, aliases). Decorator kwargs must be literals to appear in the palette.

Defaults: `label` = function name title-cased; `category` = module path. Future extension point (not M0): `@block(also_depends_on=["data/schema.json"])` to fold non-code dependencies into the staleness hash.

> **PyPI note:** the name `starforge` is squatted by a dormant Galaxy Project tool (last release 2018). Distribution name is `starforge-kernel`; **import name remains `starforge`**. A PEP 541 abandoned-name claim is worth attempting before public release. VS Code marketplace IDs are publisher-namespaced — no conflict there.

---

## 5. Block schema derivation

| Function element | Becomes |
|---|---|
| Parameter | Input handle **and** inspector field (unified — §2.4). Annotation displayed; default from signature. |
| `T \| None` / `Optional[T]` annotation | **Optional by convention**: when the param has no signature default, is unconnected, and has no literal, the worker injects `None`. Marking optionality via the annotation keeps the code template-free. |
| `*args` / `**kwargs` | Ignored in M0 (documented limitation). |
| Return annotation | Output handles: single value → `output`; `tuple[...]` → `output_0..n`; decorator `outputs=(...)` overrides with names. |
| Docstring | Block description / usage notes. |
| Type hints | Edge-compatibility *warnings* (never blockers). Untyped code: fully permitted, no warnings. |

---

## 6. `.forge` document schema

```jsonc
{
  "schema": "starforge/1",
  "name": "AUC cleanup",
  "nodes": [
    {
      "id": "n1",
      "block": "analysis.transforms:clean_auc",   // module:qualname
      "params": {"min_coverage": 0.9},             // literals for unconnected params
      "position": {"x": 120, "y": 80},
      "notes": ""
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "n1", "source_output": "output",
      "target": "n2", "target_param": "raw"        // param NAME, not slot index
    }
  ]
}
```

Layout/notes are durable metadata and never participate in hashing. Groups/comments port over from Forge v1 in M1. Documents are plain JSON text — the VS Code custom editor is text-based, so undo/redo/diff/merge work natively.

**`builtin:` namespace.** Nodes whose `block` starts with `builtin:` are doc-native and execute without importing user code. `builtin:constant` (value param, env-independent hash — its checkpoints survive dependency upgrades) ships first; the snippet node (§10) joins the same namespace in M1. Built-ins appear in the palette under "Built-in", served by the kernel rather than the indexer.

---

## 7. Provenance & staleness

Per node:

```
history_hash = sha256(canonical_json({
  fn:      sha256(ast.dump(function_def)),          // whitespace/comment-insensitive
  closure: sha256(sorted module hashes of repo import-closure of defining module),
  env:     sha256(python_version + requirements/lockfile hashes),
  params:  canonical_json(literal params),
  inputs:  {param_name: [parent_history_hash, source_output]}
}))
```

A node is **stale** iff no checkpoint exists for its computed `history_hash`. Checkpoint reuse is an optimization, never a correctness requirement.

**Tiers** (future settings slider; T2 is the default):

- **T0** — function body only. Free; misses helper edits → *false fresh* (the dangerous direction).
- **T1** — + defining module.
- **T2** — + repo import closure, module granularity. Over-invalidates in the safe direction: editing `utils.py` reruns everything that imports it. Wasted compute, never wrong data.
- **T3** — true call-graph analysis. Expensive and still not airtight (dynamic dispatch, `getattr`, runtime imports).

What no static tier sees — data files read inside functions, env vars, network — is covered by the **"force rerun from here"** affordance and, later, `also_depends_on`.

---

## 8. Serialization & previews

Edges carry arbitrary objects, so checkpoints use a **serializer registry**, probed in order per output value:

| Tier | Types | Format | Availability |
|---|---|---|---|
| parquet | DataFrame/Series | `.parquet` | lazy, iff pandas+pyarrow importable in the *user's* env |
| npy | ndarray | `.npy` | lazy, iff numpy |
| json | dict/list/str/num/bool/None | `.json` | always (stdlib) |
| pickle | anything | `.pkl` | **opt-in setting, default off** |
| ephemeral | the rest | not persisted | always |

**Ephemeral semantics:** the value flows normally to downstream nodes *within the same run* (held in worker memory), but is not checkpointed — so downstream of an ephemeral output recomputes in future runs. The cost of unserializable types stays localized instead of poisoning the store. Provenance records the serializer per output.

**Previews are precomputed artifacts** (shipped in M0): the worker builds a cropped, strict-JSON-safe preview of every output at checkpoint-write time — DataFrame → head table, ndarray → corner slice + dtype/shape, plain data → truncated value, fallback → `repr` — and stores it inside `provenance.json`. The stdlib-only kernel serves previews by reading a file, never by deserializing data, and **ephemeral outputs get previews too** (their only observable window is the run that produced them).

**Figures** (shipped in M1): the worker forces `MPLBACKEND=Agg` and sweeps matplotlib figures around each block call — `plt.show()` (a no-op under Agg) or simply leaving a figure open is enough; no code changes required. Plotly `fig.show()` is intercepted via `plotly.io.show` while the block runs. Side-effect figures render to checkpoint artifacts (matplotlib → PNG, plotly → self-contained HTML) recorded under provenance `figures`; **returned** figures (including bare Axes, which render their parent figure) become ephemeral outputs carrying an artifact — they flow on edges within a run, and future runs recompute them via the existing cascade. All figures are closed after the checkpoint write. The extension converts artifact paths to webview URIs (`results/figures` batch RPC) and nodes show desktop-style inline thumbnails with a lightbox.

---

## 9. Kernel protocol (NDJSON JSON-RPC over stdio)

| Method | → Result |
|---|---|
| `initialize {workspace, settings}` | kernel version, env fingerprint |
| `index/scan {changed_files?}` | palette (block infos), per-module errors |
| `pipeline/hashes {doc}` | per-node `history_hash`, `stale`, `missing_block` |
| `run/start {doc, run_id}` | ack; then `run/event` notifications (`node_started/completed/skipped/failed/blocked`, `run_finished`) |
| `run/cancel {run_id}` | kills worker |
| `results/manifest {history_hash}` | provenance + output manifest (shapes/types/timing) |

Line-delimited JSON (one object per line) rather than LSP `Content-Length` framing — we own both ends; revisit only if an LSP-proper integration appears.

---

## 10. What happens to PR #6 / freecode

Not merged. Its requirements map to:
- "Write code without the template" → decorator on ordinary functions; CodeLens inserts it.
- "Forge as my IDE" → *Forge *is* the IDE; code lives in real files with real tooling.
- "Quick inline expressions" → snippet node (M1): code text in the doc, hashed as a param, executed via the same worker — no namespace transport, no object store.

Nick's design doc remains the requirements reference for power-user workflows.

---

## 11. Repo layout & relationship to the desktop engine

```
forge/                      # this repo
├── backend/ blocks/ frontend/ Forge/    # desktop app — UNTOUCHED by M0
└── starforge/
    ├── DESIGN.md           # this document
    ├── pyproject.toml      # distribution: starforge-kernel; import: starforge
    ├── src/starforge/
    │   ├── __init__.py     # the decorator (zero-dep, instant import)
    │   ├── core/           # spec, hashing, staleness, serializers, checkpoints, runner
    │   ├── index/          # AST scanner, import graph, incremental cache
    │   └── kernel/         # stdio server, worker entrypoint
    ├── tests/              # headless M0 proof (pytest starforge/tests)
    └── vscode/             # TS extension + React Flow webview
```

**Honest adjustment from the design chats:** the original phrasing was "extract `forge-core` from the existing backend." On contact with the code, the new engine differs at nearly every line that touches a schema (arbitrary outputs vs privileged DataFrame, named params vs ordered slots, source-hash vs manual versions, per-output serializers vs `data.parquet`). Parameterizing the v1 engine to serve both products would risk the shipping desktop app to save little. So: **`starforge.core` is built fresh** (informed by `backend/engine`), and the desktop app migrates onto it as a deliberate post-M2 unification, at which point the old engine is deleted. One implementation eventually; zero risk to shipping users now.

---

## 12. Milestones

**M0 — prove the loop** ✅ *(shipped 2026-06-11, interactively verified)*
Decorator package · AST indexer · core engine (hashing/staleness/serializers/checkpoints/runner) · stdio kernel + worker · VS Code extension with the React Flow canvas in desktop Forge's visual language · `builtin:constant` · checkpoint-time previews · `T | None` optionals · delete/prune interactions · `.forge/` store.
**Demo criterion (met):** decorate two functions → draw an edge → Run → rerun is instant → edit one function → only it and descendants rerun.

**M1+ — priority order** *(set by Jonathan, 2026-06-11)*
1. Output rendering — `plt.show()`/figures inline on nodes (worker captures figures → checkpoint artifacts → thumbnails + lightbox).
2. Progress bars / streaming / cancellation UX.
3. Comments (canvas comments/groups parity with desktop).
4. CodeLens decorator insertion ("⊕ Add to palette" writes `@block` into the source).
5. Edge type warnings (static hint mismatch → squiggle, never a blocker).
6. Registry/cache + checkpoint hygiene — these directories grow fast; max-cache-size / max-checkpoint-size extension settings + GC.
7. Staleness tier slider (T0–T3, default T2).
8. Palette block hover cards — inputs/outputs/docstring/source hint.
9. MCP surface from the kernel (agents in VS Code author pipelines).
10. Marketplace packaging + CI.
11. Script→pipeline importer — lowest priority, likely post-v1.0.

**Post-v1 — unification**
Desktop app migrates to `starforge.core`; v1 engine deleted.

---

## 13. Open items

- PyPI distribution name: ship as `starforge-kernel`, attempt PEP 541 claim of `starforge` before public release.
- Marketplace publisher ID (suggest `predictive-oncology` or personal).
- Pickle tier default stays **off**; revisit after real-world ephemeral-tier friction is observed.
- `*args/**kwargs` parameters unsupported in M0.
- Multi-language blocks (TS/R): out of scope; the indexer/kernel split keeps the door open.
