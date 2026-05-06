  Today: edges carry a DataFrame. A node is a pure function f(inputs, params) → DataFrame.

  Proposed: edges carry a namespace (the dict of names defined by the parent's code). A freeform
  node forks the parent's namespace, runs its own code, and emits the resulting namespace.
  Children see all parent variables; mutations don't propagate upstream because each fork is a
  deep copy (effectively a pickle round-trip).

  Regular blocks (LoadCSV, KMeans, etc.) still work — their output namespace is implicitly
  {"data": <df>}, and a freeform block can read data. A freeform block that defines data can feed
   downstream regular blocks. So the existing 60+ blocks aren't disturbed.

  Provenance is unchanged in spirit: hash = sha256(parent_hash + code + params). Code text is
  just one more param.

  Open design decisions (need your call before I start)

  1. Editor library: Monaco (VS Code's, ~1MB gz, full-fat) vs CodeMirror 6 (~300KB gz, lighter,
  very good Python support). I'd pick CodeMirror 6 — lighter for the Tauri bundle and easier to
  embed in a resizable panel — but say if you want Monaco.
  2. Namespace serialization: Stdlib pickle covers most cases; dill covers lambdas, closures,
  locally-defined classes. I'd add dill as a dep and document "no open file handles, no live
  threads" as the contract. OK?
  3. Multi-parent semantics: If a freeform node has 2 parents, what does it see? My
  recommendation: each input slot gets a proxy named in_0, in_1, … — child code reads in_0.x for
  parent-0's variable x. Clean, no name collisions. Alternative is "merge into one flat
  namespace" with last-wins, which I think is a footgun.
  4. Output handles: Do all branches of a freeform cell see the same namespace (the simplest
  model — one output, fanout to N children, each gets an independent fork), or do you want
  per-handle named subsets (e.g. handle train exposes X_train, y_train; handle test exposes
  X_test, y_test)? Simple model is easier and probably fine for v1; named subsets is a nice
  future addition.
  5. Per-node n_inputs: Confirm we go with "configurable per node via a param" (Option B from
  before). Editing it on a node changes its handle count and marks it stale. Yes?

  Phased plan

  Phase 1 — Per-node arity (foundational; needed for Phases 2-3)
  - n_inputs becomes resolvable from node params, falling back to class default
  - Update the 6 backend read sites: registry.py, engine/runner.py, document_service.py (×3),
  pipeline_export.py, Forge/export_runtime.py
  - Frontend renders handles from node params, not just block spec
  - Smoke test with an existing 2-input block to make sure nothing regresses

  Phase 2 — Namespace as a checkpoint type
  - BlockOutput gains optional namespace: dict[str, Any]
  - New checkpoint format alongside parquet: namespace.pkl (dill)
  - Engine's _resolve_input_data learns to pass namespaces to freeform-style blocks
  - Backwards-compat shim: regular blocks' DataFrame output is wrapped as {"data": df} when a
  freeform child reads it; freeform output that defines data is unwrapped for regular children

  Phase 3 — FreeformCode v2
  - Rewrite blocks/freeform.py with configurable n_inputs, n_outputs, and code params
  - Multi-parent: child sees in_0, in_1, … namespace proxies
  - v1 output model: single namespace, fanout to children (Decision #4 simple version)

  Phase 4 — Frontend code editor
  - Add CodeMirror 6 with Python language support
  - Embed in NodeInspector with a resize handle (vertical resize at minimum; ideally a draggable
  corner)
  - Make the editor wide enough that it's actually pleasant — this is the one thing the user said
   could kill the project

  Phase 5 — Polish
  - Error reporting: surface tracebacks with line numbers in the inspector
  - Variable inspector: list dir() of the namespace under the editor (cheap and very Jupyter-y)
  - Optional: render DataFrame outputs inline like Jupyter does

  Risks I want you to know about

  - Pickle bloat: A namespace with a 5GB DataFrame loaded gets pickled per-node. We should reuse
  parquet within the pickle (store DataFrames as parquet bytes inside the namespace dict) before
  this hits real workflows. Phase 5.
  - Nondeterminism: Freeform code with random() and no seed will pass the hash check but produce
  different values across runs. Forge always trusts the cache after first run, so you'd see the
  first random output stick. This is fine but worth saying out loud — not a Forge bug, an
  intrinsic property of "cache by code identity."
  - Bundle size: CodeMirror Python pack adds ~300KB gz to the frontend. Not a big deal but
  visible.

  Tell me on the 5 decisions above (especially editor library, multi-parent semantics, and
  confirming Option B), and I'll start Phase 1.2