"""*Forge — pipeline canvas for the repo you already have open.

This top-level module is the entire public surface that user code touches.
It must import in microseconds and depend on nothing: the decorator lives in
production codebases and has to be free. Everything heavy (indexer, engine,
kernel) lives in submodules that only *Forge itself* imports.
"""

from __future__ import annotations

__version__ = "0.1.3"

__all__ = ["block", "progress", "BLOCK_ATTR"]

#: Attribute set on decorated functions. The AST indexer matches the decorator
#: syntactically and never imports user code; this runtime tag exists so user
#: code and future runtime introspection can also recognize blocks.
BLOCK_ATTR = "__starforge_block__"


def block(fn=None, *, label=None, category=None, outputs=None):
    """Register a function as a *Forge block.

    Usable bare or with keyword arguments::

        @block
        def clean(raw: pd.DataFrame) -> pd.DataFrame: ...

        @block(label="Clean AUC Matrix", category="QC", outputs=("clean", "stats"))
        def clean_auc(raw, min_coverage: float = 0.8): ...

    The decorated function is returned unchanged — behavior under pytest, in
    CI, or in production is identical whether or not *Forge is anywhere near.

    Args:
        label: Palette display name. Defaults to the function name, title-cased.
        category: Palette grouping. Defaults to the defining module's path.
        outputs: Names for multiple return values (function must return a tuple
            of the same length). Defaults to a single output named "output".

    Note for palette metadata: the indexer reads ``label``/``category``/
    ``outputs`` from the *source*, so they must be literals at the decoration
    site to appear in the palette.
    """

    def apply(f):
        setattr(
            f,
            BLOCK_ATTR,
            {
                "label": label,
                "category": category,
                "outputs": tuple(outputs) if outputs is not None else None,
            },
        )
        return f

    if fn is not None:
        return apply(fn)
    return apply


#: Installed by the *Forge run worker around each block call; None everywhere
#: else, which keeps progress() a guaranteed no-op in pytest/CI/production.
_progress_hook = None


def progress(current=None, total=None, label=None):
    """Report block progress to the *Forge canvas.

    Call freely inside a block::

        for i, chunk in enumerate(chunks):
            progress(i + 1, len(chunks), "fitting folds")
            ...

    Outside a *Forge run this does nothing and costs one attribute read —
    safe to leave in production code. Any combination of arguments works:
    (current, total) renders a determinate bar, label alone updates the text.
    """
    hook = _progress_hook
    if hook is not None:
        try:
            hook(current, total, label)
        except Exception:
            pass  # progress must never break user code
