import subprocess
import sys

from starforge import BLOCK_ATTR, block


def test_bare_decorator_is_behavior_neutral():
    @block
    def add(a, b=1):
        return a + b

    assert add(2, 3) == 5
    assert getattr(add, BLOCK_ATTR) == {"label": None, "category": None, "outputs": None}


def test_decorator_with_kwargs():
    @block(label="Fancy Add", category="Math", outputs=("sum", "carry"))
    def add(a, b):
        return a + b, 0

    assert add(1, 2) == (3, 0)
    meta = getattr(add, BLOCK_ATTR)
    assert meta["label"] == "Fancy Add"
    assert meta["category"] == "Math"
    assert meta["outputs"] == ("sum", "carry")


def test_import_pulls_in_nothing_heavy():
    """The decorator lives in user production code; importing starforge must
    not drag in pandas/numpy or the engine submodules."""
    code = (
        "import starforge, sys\n"
        "banned = {'pandas', 'numpy', 'starforge.core', 'starforge.kernel', 'starforge.index'}\n"
        "loaded = banned & set(sys.modules)\n"
        "assert not loaded, f'heavy imports: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src')},
    )
    assert result.returncode == 0, result.stderr
