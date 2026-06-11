"""Static workspace indexer.

Discovers ``@block``-decorated functions and the module import graph by
parsing source with :mod:`ast`. User code is NEVER imported here — imports
execute side effects and load heavy libraries; the indexer must stay safe to
run on every keystroke-adjacent event. Execution-time imports happen only in
the run worker.

Incrementality: callers pass the previous scan's cache back in; files whose
(mtime_ns, size) match are not even re-read, files whose content hash matches
are not re-parsed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Iterator

#: Bump when the per-file cache entry shape changes; old entries re-parse.
CACHE_VERSION = 2

#: Directories never worth scanning. Dot-directories are skipped wholesale
#: (.git, .forge, .venv, ...), these cover the common non-dotted offenders.
SKIP_DIRS = {
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    ".eggs",
}


def _sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class ParamInfo:
    name: str
    annotation: str | None = None
    default_repr: str | None = None
    has_default: bool = False
    keyword_only: bool = False
    #: ``T | None`` / ``Optional[T]`` annotations mark a parameter optional:
    #: when unconnected and given no literal, the worker injects None
    #: (DESIGN.md §5). Independent of has_default.
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "annotation": self.annotation,
            "default_repr": self.default_repr,
            "has_default": self.has_default,
            "keyword_only": self.keyword_only,
            "optional": self.optional,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParamInfo":
        return cls(**d)


@dataclass
class BlockInfo:
    block_id: str  # "dotted.module:qualname"
    module: str
    qualname: str
    file: str  # workspace-relative posix path
    lineno: int
    label: str
    category: str
    params: list[ParamInfo]
    outputs: list[str]
    returns: str | None
    doc: str | None
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "module": self.module,
            "qualname": self.qualname,
            "file": self.file,
            "lineno": self.lineno,
            "label": self.label,
            "category": self.category,
            "params": [p.to_dict() for p in self.params],
            "outputs": self.outputs,
            "returns": self.returns,
            "doc": self.doc,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlockInfo":
        d = dict(d)
        d["params"] = [ParamInfo.from_dict(p) for p in d["params"]]
        return cls(**d)


@dataclass
class ModuleInfo:
    module: str
    file: str
    file_hash: str
    #: AST-normalized content hash: whitespace/comment edits don't change it,
    #: so they don't invalidate importers via the Tier-2 closure. Falls back
    #: to file_hash for files that don't parse.
    ast_hash: str = ""
    #: Raw dotted import targets as written; resolved against the workspace
    #: module set at closure-hash time so cache entries stay valid as other
    #: files appear and disappear.
    imports: list[str] = field(default_factory=list)
    blocks: list[BlockInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class WorkspaceIndex:
    root: str
    modules: dict[str, ModuleInfo] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._closure_memo: dict[str, str] = {}

    @property
    def blocks(self) -> dict[str, BlockInfo]:
        return {b.block_id: b for m in self.modules.values() for b in m.blocks}

    def errors(self) -> dict[str, list[str]]:
        return {m.module: m.errors for m in self.modules.values() if m.errors}

    def _resolve_imports(self, module: str) -> set[str]:
        """Map a module's raw import strings to workspace-internal modules.

        Importing a package conservatively pulls in every module under it:
        over-invalidation is the safe direction for staleness (DESIGN.md §7).
        """
        info = self.modules.get(module)
        if info is None:
            return set()
        resolved: set[str] = set()
        for target in info.imports:
            if target in self.modules:
                resolved.add(target)
            prefix = target + "."
            resolved.update(m for m in self.modules if m.startswith(prefix))
        resolved.discard(module)
        return resolved

    def closure_hash(self, module: str) -> str:
        """Tier-2 staleness input: hash of this module plus everything it
        (transitively) imports inside the workspace, order-independent."""
        if module in self._closure_memo:
            return self._closure_memo[module]
        seen: set[str] = set()
        frontier = [module]
        while frontier:
            current = frontier.pop()
            if current in seen or current not in self.modules:
                continue
            seen.add(current)
            frontier.extend(self._resolve_imports(current))
        digest = _sha("\n".join(sorted(self.modules[m].ast_hash or self.modules[m].file_hash for m in seen)))
        self._closure_memo[module] = digest
        return digest


def _module_name(relpath: Path) -> str:
    parts = list(relpath.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else "__root__"


def _iter_py_files(root: Path) -> Iterator[Path]:
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                if name.startswith(".") or name in SKIP_DIRS or name.endswith(".egg-info"):
                    continue
                # Treat any directory that contains a venv marker as foreign.
                if (entry / "pyvenv.cfg").exists():
                    continue
                stack.append(entry)
            elif entry.is_file() and name.endswith(".py"):
                yield entry


class _StarforgeAliases:
    """Names under which the @block decorator is reachable in one module."""

    def __init__(self) -> None:
        self.direct: set[str] = set()  # from starforge import block [as b]
        self.modules: set[str] = set()  # import starforge [as sf]

    def collect(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "starforge":
                    self.modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module == "starforge":
                for alias in node.names:
                    if alias.name == "block":
                        self.direct.add(alias.asname or alias.name)

    def match(self, decorator: ast.expr) -> tuple[bool, dict[str, Any]]:
        """Return (is_block_decorator, literal_kwargs)."""
        target = decorator
        kwargs: dict[str, Any] = {}
        if isinstance(decorator, ast.Call):
            target = decorator.func
            for kw in decorator.keywords:
                if kw.arg is None:
                    continue
                kwargs[kw.arg] = _literal(kw.value)
        if isinstance(target, ast.Name) and target.id in self.direct:
            return True, kwargs
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "block"
            and isinstance(target.value, ast.Name)
            and target.value.id in self.modules
        ):
            return True, kwargs
        return False, {}


def _literal(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _collect_imports(tree: ast.Module, module: str) -> list[str]:
    """Raw dotted import targets, with relative imports resolved against the
    importing module's package."""
    package_parts = module.split(".")[:-1] if module != "__root__" else []
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    targets.add(node.module)
                    # `from pkg import name` may bind a submodule, not an attr.
                    for alias in node.names:
                        targets.add(f"{node.module}.{alias.name}")
            else:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                if node.module:
                    base = base + node.module.split(".")
                if base:
                    targets.add(".".join(base))
                for alias in node.names:
                    if base:
                        targets.add(".".join(base + [alias.name]))
    return sorted(targets)


def _annotation_is_optional(node: ast.expr | None) -> bool:
    """True for ``T | None``, ``None | T``, ``Optional[T]``, bare ``None``,
    and string annotations mentioning None."""
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        if node.value is None:
            return True
        return isinstance(node.value, str) and "None" in node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_is_optional(node.left) or _annotation_is_optional(node.right)
    if isinstance(node, ast.Subscript):
        base = node.value
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None)
        return name == "Optional"
    return False


def _extract_params(fn: ast.FunctionDef) -> list[ParamInfo]:
    params: list[ParamInfo] = []
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    for arg, default in zip(positional, defaults):
        params.append(
            ParamInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                default_repr=ast.unparse(default) if default is not None else None,
                has_default=default is not None,
                optional=_annotation_is_optional(arg.annotation),
            )
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            ParamInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                default_repr=ast.unparse(default) if default is not None else None,
                has_default=default is not None,
                keyword_only=True,
                optional=_annotation_is_optional(arg.annotation),
            )
        )
    # *args/**kwargs are intentionally not modeled in M0 (DESIGN.md §5).
    return params


def _infer_outputs(fn: ast.FunctionDef, decorator_outputs: Any) -> list[str]:
    if decorator_outputs:
        names = [str(n) for n in decorator_outputs]
        if names:
            return names
    returns = fn.returns
    if (
        isinstance(returns, ast.Subscript)
        and isinstance(returns.value, ast.Name)
        and returns.value.id in {"tuple", "Tuple"}
        and isinstance(returns.slice, ast.Tuple)
    ):
        elts = returns.slice.elts
        # tuple[int, ...] is variadic — fall back to a single output.
        if not any(isinstance(e, ast.Constant) and e.value is Ellipsis for e in elts):
            return [f"output_{i}" for i in range(len(elts))]
    return ["output"]


def _default_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _parse_module(
    text: str, module: str, relpath: str
) -> tuple[list[BlockInfo], list[str], list[str], str]:
    """Returns (blocks, raw_imports, errors, ast_hash) for one source file."""
    errors: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], [], [f"syntax error: line {exc.lineno}: {exc.msg}"], ""

    aliases = _StarforgeAliases()
    for node in ast.walk(tree):
        aliases.collect(node)

    blocks: list[BlockInfo] = []
    top_level = {id(n) for n in tree.body}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        matched_kwargs: dict[str, Any] | None = None
        for decorator in node.decorator_list:
            is_block, kwargs = aliases.match(decorator)
            if is_block:
                matched_kwargs = kwargs
                break
        if matched_kwargs is None:
            continue
        if isinstance(node, ast.AsyncFunctionDef):
            errors.append(f"{node.name}: async @block functions are not supported yet")
            continue
        if id(node) not in top_level:
            errors.append(
                f"{node.name}: @block only registers module-level functions "
                "(methods and nested functions are not supported yet)"
            )
            continue
        blocks.append(
            BlockInfo(
                block_id=f"{module}:{node.name}",
                module=module,
                qualname=node.name,
                file=relpath,
                lineno=node.lineno,
                label=matched_kwargs.get("label") or _default_label(node.name),
                category=matched_kwargs.get("category") or module,
                params=_extract_params(node),
                outputs=_infer_outputs(node, matched_kwargs.get("outputs")),
                returns=ast.unparse(node.returns) if node.returns else None,
                doc=ast.get_docstring(node),
                source_hash=_sha(ast.dump(node)),
            )
        )
    return blocks, _collect_imports(tree, module), errors, _sha(ast.dump(tree))


def scan_workspace(
    root: str | Path,
    cache: dict[str, Any] | None = None,
) -> tuple[WorkspaceIndex, dict[str, Any]]:
    """Scan a workspace; returns the index and a cache for the next scan.

    ``cache`` is the second return value of a previous call (typically
    persisted to ``.forge/cache/index.json``). Unchanged files are reused
    without re-reading (mtime+size fast path) or re-parsing (content hash).
    """
    root = Path(root).resolve()
    prev_files: dict[str, Any] = (cache or {}).get("files", {})
    next_files: dict[str, Any] = {}
    index = WorkspaceIndex(root=str(root))

    for path in _iter_py_files(root):
        relpath = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
        except OSError:
            continue
        entry = prev_files.get(relpath)
        # Cache-format version gate: entries from older formats are re-parsed
        # rather than trusted. Bump CACHE_VERSION when entry shape changes.
        reusable = entry and entry.get("v") == CACHE_VERSION
        if reusable and entry["mtime_ns"] == stat.st_mtime_ns and entry["size"] == stat.st_size:
            next_files[relpath] = entry
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            file_hash = _sha(text)
            if reusable and entry["sha"] == file_hash:
                entry = {**entry, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
            else:
                module = _module_name(Path(relpath))
                blocks, imports, errors, ast_hash = _parse_module(text, module, relpath)
                entry = {
                    "v": CACHE_VERSION,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "sha": file_hash,
                    "ast_sha": ast_hash,
                    "module": module,
                    "imports": imports,
                    "blocks": [b.to_dict() for b in blocks],
                    "errors": errors,
                }
            next_files[relpath] = entry

        entry = next_files[relpath]
        index.modules[entry["module"]] = ModuleInfo(
            module=entry["module"],
            file=relpath,
            file_hash=entry["sha"],
            ast_hash=entry["ast_sha"],
            imports=list(entry["imports"]),
            blocks=[BlockInfo.from_dict(b) for b in entry["blocks"]],
            errors=list(entry["errors"]),
        )

    return index, {"files": next_files}
