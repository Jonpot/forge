# *Forge (`starforge`)

Forge's canvas — checkpointing, provenance, stale/hydrate execution — as a VS Code
extension over the repo you already have open. Blocks are ordinary Python functions
tagged with `@block`. See [DESIGN.md](DESIGN.md) for the full design.

## Try it (M0)

```bash
# 1. Install the kernel + decorator into the venv your target repo uses
pip install -e <forge-repo>/starforge

# 2. Build the extension
cd <forge-repo>/starforge/vscode
npm install && npm run build

# 3. Open starforge/vscode in VS Code and press F5 (extension dev host).
#    In the dev-host window, open any Python repo.
```

In your repo:

```python
# analysis/blocks.py
import matplotlib.pyplot as plt
from starforge import block

@block(category="IO")
def make_numbers(n: int = 5) -> dict:
    return {"values": list(range(1, n + 1))}

@block
def scale(data: dict, factor: float = 2.0) -> dict:
    return {"values": [v * factor for v in data["values"]]}

@block
def plot(data: dict) -> dict:
    plt.plot(data["values"])
    plt.show()  # rendered inline on the canvas node
    return data
```

Save, run **“*Forge: New Pipeline”**, drag the blocks from the palette, wire
`output → data`, hit **▶ Run**. Run again — instant, everything reused. Edit
`scale`, watch it (and only it) go stale.

## Layout

| Path | What |
|---|---|
| `src/starforge/__init__.py` | the `@block` decorator — zero-dep, the only thing user code touches |
| `src/starforge/index/` | static AST indexer (discovery, import graph, incremental cache) |
| `src/starforge/core/` | doc schema, history hashing, serializers, checkpoint store, runner |
| `src/starforge/kernel/` | stdio JSON-RPC kernel + per-run worker subprocess |
| `vscode/` | the extension (TS host + React Flow webview) |
| `tests/` | headless M0 proof — `python -m pytest starforge/tests` |

State lives in the target repo under `.forge/` — `pipelines/` is committable,
`checkpoints/` and `cache/` are auto-gitignored.
