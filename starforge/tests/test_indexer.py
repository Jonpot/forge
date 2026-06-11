from starforge.index import scan_workspace


def test_discovers_decorated_functions(workspace):
    index, _ = scan_workspace(workspace.root)
    blocks = index.blocks
    assert set(blocks) == {
        "pipeline_lib.sources:make_numbers",
        "pipeline_lib.transforms:scale_values",
        "pipeline_lib.transforms:summarize",
    }

    make = blocks["pipeline_lib.sources:make_numbers"]
    assert make.label == "Make Numbers"  # default: title-cased name
    assert make.category == "IO"  # decorator override
    assert make.outputs == ["output"]
    assert make.doc and "first n positive integers" in make.doc
    [param] = make.params
    assert (param.name, param.annotation, param.default_repr) == ("n", "int", "5")

    scale = blocks["pipeline_lib.transforms:scale_values"]
    assert scale.category == "pipeline_lib.transforms"  # default: module path
    assert [p.name for p in scale.params] == ["data", "factor"]

    summarize = blocks["pipeline_lib.transforms:summarize"]
    assert summarize.outputs == ["total", "count"]  # decorator-named outputs


def test_import_aware_decorator_matching(workspace):
    workspace.write(
        "aliased.py",
        "import starforge as sf\n"
        "from starforge import block as blk\n\n"
        "@sf.block\n"
        "def via_module(x: int) -> int:\n    return x\n\n"
        "@blk\n"
        "def via_alias(x: int) -> int:\n    return x\n",
    )
    workspace.write(
        "impostor.py",
        "def block(fn):\n    return fn\n\n"
        "@block\n"
        "def not_a_forge_block(x):\n    return x\n",
    )
    index, _ = scan_workspace(workspace.root)
    assert "aliased:via_module" in index.blocks
    assert "aliased:via_alias" in index.blocks
    assert "impostor:not_a_forge_block" not in index.blocks


def test_optional_param_detection(workspace):
    workspace.write(
        "optionals.py",
        "from typing import Optional\n"
        "from starforge import block\n\n"
        "@block\n"
        "def f(a: int, b: float | None, c: Optional[str], d: str = 'x', e: 'int | None' = None) -> int:\n"
        "    return a\n",
    )
    index, _ = scan_workspace(workspace.root)
    by_name = {p.name: p for p in index.blocks["optionals:f"].params}
    assert not by_name["a"].optional
    assert by_name["b"].optional and not by_name["b"].has_default
    assert by_name["c"].optional
    assert not by_name["d"].optional and by_name["d"].has_default
    assert by_name["e"].optional and by_name["e"].has_default


def test_tuple_return_annotation_infers_outputs(workspace):
    workspace.write(
        "tuples.py",
        "from starforge import block\n\n"
        "@block\n"
        "def pair(x: int) -> tuple[int, str]:\n    return x, str(x)\n",
    )
    index, _ = scan_workspace(workspace.root)
    assert index.blocks["tuples:pair"].outputs == ["output_0", "output_1"]


def test_incremental_cache_and_source_hash_stability(workspace):
    index1, cache = scan_workspace(workspace.root)
    hash_before = index1.blocks["pipeline_lib.transforms:scale_values"].source_hash

    index2, cache = scan_workspace(workspace.root, cache)
    assert index2.blocks["pipeline_lib.transforms:scale_values"].source_hash == hash_before

    # Whitespace/comment-only edits must NOT change the source hash.
    original = (workspace.root / "pipeline_lib/transforms.py").read_text(encoding="utf-8")
    workspace.write("pipeline_lib/transforms.py", "# a comment\n" + original + "\n\n")
    index3, cache = scan_workspace(workspace.root, cache)
    assert index3.blocks["pipeline_lib.transforms:scale_values"].source_hash == hash_before

    # A body edit must change it.
    workspace.write("pipeline_lib/transforms.py", original.replace("* factor", "* factor * 2"))
    index4, _ = scan_workspace(workspace.root, cache)
    assert index4.blocks["pipeline_lib.transforms:scale_values"].source_hash != hash_before


def test_closure_hash_tracks_transitive_imports(workspace):
    index1, cache = scan_workspace(workspace.root)
    transforms_before = index1.closure_hash("pipeline_lib.transforms")
    sources_before = index1.closure_hash("pipeline_lib.sources")

    # Comment/whitespace-only edits to a helper must NOT invalidate importers.
    workspace.write("pipeline_lib/helpers.py", "# tweaked\ndef scale_factor():\n\n    return 1\n")
    index2, cache = scan_workspace(workspace.root, cache)
    assert index2.closure_hash("pipeline_lib.transforms") == transforms_before

    workspace.write("pipeline_lib/helpers.py", "def scale_factor():\n    return 10\n")
    index3, _ = scan_workspace(workspace.root, cache)

    # transforms imports helpers -> closure changes; sources does not import it.
    assert index3.closure_hash("pipeline_lib.transforms") != transforms_before
    assert index3.closure_hash("pipeline_lib.sources") == sources_before


def test_errors_reported_not_fatal(workspace):
    workspace.write("broken.py", "def broken(:\n")
    workspace.write(
        "nested.py",
        "from starforge import block\n\n"
        "class Holder:\n"
        "    @block\n"
        "    def method(self, x):\n        return x\n",
    )
    index, _ = scan_workspace(workspace.root)
    errors = index.errors()
    assert any("syntax error" in e for e in errors.get("broken", []))
    assert any("module-level" in e for e in errors.get("nested", []))
    # The healthy modules are still fully indexed.
    assert len(index.blocks) == 3
