"""Project-relative paths that work on Windows and Linux alike."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ENV_ROOT = "SHIFTGUARD_ROOT"


def project_root(start: str | Path | None = None) -> Path:
    """Locate the project root.

    The ``SHIFTGUARD_ROOT`` environment variable wins. Otherwise the first ancestor of
    ``start`` (default: the current directory) containing ``pyproject.toml`` is used,
    falling back to ``start`` itself (e.g. on Kaggle, where the package is pip-installed).
    """
    env = os.environ.get(ENV_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve ``value`` against ``root`` unless it is already absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root or project_root()) / path


def data_dir(config: Mapping[str, Any] | None = None) -> Path:
    """Directory holding downloaded and derived data (never committed to git)."""
    value = (config or {}).get("paths", {}).get("data_dir", "data")
    return resolve_path(value)


def reports_dir(config: Mapping[str, Any] | None = None) -> Path:
    """Directory receiving tables, figures and run folders."""
    value = (config or {}).get("paths", {}).get("reports_dir", "reports")
    return resolve_path(value)
