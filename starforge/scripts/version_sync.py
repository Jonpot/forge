"""Set the *Forge version everywhere at once — kernel and extension ship in
lockstep (same convention as the desktop repo's scripts/version_sync.py).

Usage:  python starforge/scripts/version_sync.py 0.1.2

Rewrites:
  starforge/pyproject.toml            project.version
  starforge/src/starforge/__init__.py __version__
  starforge/vscode/package.json       version
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        print(__doc__)
        return 2
    version = sys.argv[1]

    pyproject = ROOT / "pyproject.toml"
    text, n = re.subn(r'(?m)^version = "[^"]+"', f'version = "{version}"', pyproject.read_text(encoding="utf-8"))
    assert n == 1, "expected exactly one version line in pyproject.toml"
    pyproject.write_text(text, encoding="utf-8")

    init = ROOT / "src" / "starforge" / "__init__.py"
    text, n = re.subn(r'(?m)^__version__ = "[^"]+"', f'__version__ = "{version}"', init.read_text(encoding="utf-8"))
    assert n == 1, "expected exactly one __version__ line in __init__.py"
    init.write_text(text, encoding="utf-8")

    package_json = ROOT / "vscode" / "package.json"
    manifest = json.loads(package_json.read_text(encoding="utf-8"))
    manifest["version"] = version
    package_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"*Forge -> {version}")  # ASCII only: Windows consoles are cp1252
    print(f"  {pyproject.relative_to(ROOT.parent)}")
    print(f"  {init.relative_to(ROOT.parent)}")
    print(f"  {package_json.relative_to(ROOT.parent)}")
    print("Next: rebuild dist (python -m build starforge) and the VSIX (npm run package).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
