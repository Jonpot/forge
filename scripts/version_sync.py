from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# Accept MAJOR.MINOR.PATCH optionally followed by a pre-release tag (-foo, -1,
# -alpha.2, etc.) and/or build metadata (+sha). Cargo and npm both accept this
# shape natively.
SEMVER_RE = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)

PYPROJECT = REPO_ROOT / "pyproject.toml"
FRONTEND_PACKAGE = REPO_ROOT / "frontend" / "package.json"
FRONTEND_LOCK = REPO_ROOT / "frontend" / "package-lock.json"
TAURI_CARGO = REPO_ROOT / "frontend" / "src-tauri" / "Cargo.toml"
TAURI_LOCK = REPO_ROOT / "frontend" / "src-tauri" / "Cargo.lock"
TAURI_CONF = REPO_ROOT / "frontend" / "src-tauri" / "tauri.conf.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def replace_one(path: Path, pattern: re.Pattern[str], replacement: str) -> bool:
    original = read_text(path)
    updated, count = pattern.subn(replacement, original, count=1)
    if count != 1:
        raise RuntimeError(f"Expected exactly one version match in {path}")
    if updated == original:
        return False
    write_text(path, updated)
    return True


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def read_pyproject_version() -> str:
    match = re.search(r'(?m)^version = "([^"]+)"\s*$', read_text(PYPROJECT))
    if not match:
        raise RuntimeError(f"Could not find project version in {PYPROJECT}")
    return match.group(1)


def read_cargo_lock_root_version() -> str:
    match = re.search(
        r'\[\[package\]\]\s+name = "forge"\s+version = "([^"]+)"',
        read_text(TAURI_LOCK),
    )
    if not match:
        raise RuntimeError(f'Could not find root package version for "forge" in {TAURI_LOCK}')
    return match.group(1)


def collect_versions() -> dict[str, str]:
    frontend_package = load_json(FRONTEND_PACKAGE)
    frontend_lock = load_json(FRONTEND_LOCK)
    tauri_conf = load_json(TAURI_CONF)

    cargo_match = re.search(r'(?m)^version = "([^"]+)"\s*$', read_text(TAURI_CARGO))
    if not cargo_match:
        raise RuntimeError(f"Could not find crate version in {TAURI_CARGO}")

    return {
        "pyproject.toml": read_pyproject_version(),
        "frontend/package.json": str(frontend_package["version"]),
        "frontend/package-lock.json": str(frontend_lock["version"]),
        'frontend/package-lock.json packages[""].version': str(frontend_lock["packages"][""]["version"]),
        "frontend/src-tauri/Cargo.toml": cargo_match.group(1),
        "frontend/src-tauri/Cargo.lock": read_cargo_lock_root_version(),
        "frontend/src-tauri/tauri.conf.json": str(tauri_conf["version"]),
    }


def check_versions() -> int:
    versions = collect_versions()
    expected = versions["pyproject.toml"]
    mismatches = {
        path: version
        for path, version in versions.items()
        if version != expected
    }

    if mismatches:
        print(f"Version mismatch detected. Expected all release surfaces to be {expected}.", file=sys.stderr)
        for path, version in mismatches.items():
            print(f"  {path}: {version}", file=sys.stderr)
        return 1

    print(f"All release version surfaces are synced at {expected}.")
    return 0


def bump_versions(version: str) -> int:
    if not SEMVER_RE.fullmatch(version):
        print(
            f'Invalid version "{version}". Expected semantic version format like 0.2.3.',
            file=sys.stderr,
        )
        return 1

    changed_paths: list[str] = []

    if replace_one(PYPROJECT, re.compile(r'(?m)^version = "[^"]+"\s*$'), f'version = "{version}"'):
        changed_paths.append("pyproject.toml")

    frontend_package = load_json(FRONTEND_PACKAGE)
    if frontend_package.get("version") != version:
        frontend_package["version"] = version
        save_json(FRONTEND_PACKAGE, frontend_package)
        changed_paths.append("frontend/package.json")

    frontend_lock = load_json(FRONTEND_LOCK)
    root_package = frontend_lock.setdefault("packages", {}).setdefault("", {})
    frontend_lock_changed = False
    if frontend_lock.get("version") != version:
        frontend_lock["version"] = version
        frontend_lock_changed = True
    if root_package.get("version") != version:
        root_package["version"] = version
        frontend_lock_changed = True
    if frontend_lock_changed:
        save_json(FRONTEND_LOCK, frontend_lock)
        changed_paths.append("frontend/package-lock.json")

    if replace_one(TAURI_CARGO, re.compile(r'(?m)^version = "[^"]+"\s*$'), f'version = "{version}"'):
        changed_paths.append("frontend/src-tauri/Cargo.toml")

    if replace_one(
        TAURI_LOCK,
        re.compile(r'(\[\[package\]\]\s+name = "forge"\s+version = ")[^"]+(")'),
        rf'\g<1>{version}\2',
    ):
        changed_paths.append("frontend/src-tauri/Cargo.lock")

    tauri_conf = load_json(TAURI_CONF)
    if tauri_conf.get("version") != version:
        tauri_conf["version"] = version
        save_json(TAURI_CONF, tauri_conf)
        changed_paths.append("frontend/src-tauri/tauri.conf.json")

    if not changed_paths:
        print(f"Release version already at {version}.")
    else:
        print(f"Bumped Forge release version to {version}:")
        for path in changed_paths:
            print(f"  {path}")

    return check_versions()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Forge release version metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bump = subparsers.add_parser("bump", help="Update all release version surfaces.")
    bump.add_argument("version", help="Semantic version to write, e.g. 0.2.3")

    subparsers.add_parser("check", help="Verify all release version surfaces match.")
    subparsers.add_parser("current", help="Print the current release version.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bump":
        return bump_versions(args.version)
    if args.command == "check":
        return check_versions()
    if args.command == "current":
        print(read_pyproject_version())
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
