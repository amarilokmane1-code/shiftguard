"""Run provenance: every result folder records how it was produced.

A run directory ``reports/runs/<YYYY-MM-DD_HHMMSS>_<name>/`` contains the resolved
config, the seeds, the git commit (and whether the tree was dirty) and the versions of
the scientific stack, so any number in the report can be traced back to its code.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from shiftguard.config import save_config

TRACKED_PACKAGES: tuple[str, ...] = (
    "shiftguard",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "statsmodels",
    "nilearn",
    "matplotlib",
    "torch",
    "neuroHarmonize",
    "neuroCombat",
    "pymc",
    "torch-geometric",
)


def git_info(repo: str | Path | None = None) -> dict[str, Any]:
    """Return the current commit hash and dirty flag, or ``None`` values outside git."""
    cwd = Path(repo) if repo else None

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions of ``packages`` (``None`` when a package is not installed)."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _slug(name: str) -> str:
    """Make ``name`` safe for use in a directory name on every OS."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return slug or "run"


def create_run_dir(reports_dir: str | Path, name: str, *, now: datetime | None = None) -> Path:
    """Create and return a fresh ``<reports_dir>/runs/<timestamp>_<name>/`` folder.

    A numeric suffix is appended if the folder already exists, so runs never overwrite
    each other.
    """
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    base = Path(reports_dir) / "runs" / f"{stamp}_{_slug(name)}"
    candidate, counter = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{counter}")
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def write_run_metadata(
    run_dir: str | Path,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``config.yaml`` and ``metadata.json`` into ``run_dir``.

    Parameters
    ----------
    run_dir
        Target directory (created if needed).
    config
        The fully resolved configuration used by the run.
    seeds
        The random seeds used.
    extra
        Additional JSON-serialisable fields (e.g. the CLI command).

    Returns
    -------
    pathlib.Path
        Path of the written ``metadata.json``.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    save_config(config, run_path / "config.yaml")
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seeds": [int(s) for s in seeds],
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        **(dict(extra) if extra else {}),
    }
    out = run_path / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out
