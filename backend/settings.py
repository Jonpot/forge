from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_dotenv_values() -> dict[str, str]:
    # Search for .env in cwd, its parents, and the Forge data directory
    search_dirs = [Path.cwd().resolve(), *Path.cwd().resolve().parents]

    # Also check the platform-specific Forge data directory
    forge_data = _forge_data_dir()
    if forge_data and forge_data not in search_dirs:
        search_dirs.append(forge_data)

    for directory in search_dirs:
        env_path = directory / ".env"
        if not env_path.exists():
            continue

        values: dict[str, str] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = _strip_optional_quotes(raw_value.strip())
        return values
    return {}


def _forge_data_dir() -> Path | None:
    """Return the platform-specific Forge data directory."""
    import sys

    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "Forge"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Forge"
    else:
        # Linux / other Unix: follow XDG spec
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        return base / "Forge"


def env_or_default(key: str, default: str) -> str:
    dotenv_values = _load_dotenv_values()
    if key in os.environ:
        return os.environ[key]
    return dotenv_values.get(key, default)


def _default_custom_blocks_dir() -> str:
    """Return the platform-specific default directory for user-installed custom blocks."""
    import sys

    if sys.platform == "win32":
        docs = Path.home() / "Documents"
    elif sys.platform == "darwin":
        docs = Path.home() / "Documents"
    else:
        docs = Path.home() / "Documents"
    return str(docs / "Forge" / "custom_blocks")


@dataclass(slots=True)
class Settings:
    checkpoint_dir: str = "./checkpoints"
    pipeline_dir: str = "./pipelines"
    blocks_dir: str = "./blocks"
    custom_blocks_dir: str = field(default_factory=_default_custom_blocks_dir)
    default_file_path: str = ""
    workspace_dir: str = ""
    log_level: str = "INFO"
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://tauri.localhost",
            "https://tauri.localhost",
            "http://localhost:1420",
        ]
    )

    @classmethod
    def from_env(cls) -> "Settings":
        default_cors = "http://tauri.localhost,https://tauri.localhost,tauri://localhost,http://localhost:1420"
        cors_raw = env_or_default("CORS_ORIGINS", default_cors)
        cors = [item.strip() for item in cors_raw.split(",") if item.strip()]
        return cls(
            checkpoint_dir=env_or_default("CHECKPOINT_DIR", "./checkpoints"),
            pipeline_dir=env_or_default("PIPELINE_DIR", "./pipelines"),
            blocks_dir=env_or_default("BLOCKS_DIR", "./blocks"),
            custom_blocks_dir=env_or_default(
                "CUSTOM_BLOCKS_DIR", _default_custom_blocks_dir()
            ),
            default_file_path=env_or_default("DEFAULT_FILE_PATH", ""),
            workspace_dir=env_or_default("WORKSPACE_DIR", ""),
            log_level=env_or_default("LOG_LEVEL", "DEBUG"),
            cors_origins=cors or ["http://localhost:5173"],
        )
