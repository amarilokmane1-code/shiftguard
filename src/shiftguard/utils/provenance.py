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
    "nilearn",
    "matplotlib",
    "torch",
    "neuroHarmonize",
    "neuroCombat",
)


#: Folders the pipeline itself writes (tables, figures, manifest). Edits there are outputs
#: of earlier steps, not code, so they do not make a later run "dirty"; they are still
#: listed in ``outputs_modified``.
OUTPUT_PATHS: tuple[str, ...] = ("reports",)


def git_info(repo: str | Path | None = None) -> dict[str, Any]:
    """Return the commit hash and dirty flags, or ``None`` values outside git.

    ``dirty`` is true when anything outside :data:`OUTPUT_PATHS` differs from the commit
    (code, configs, docs, untracked files). ``outputs_modified`` lists the changed paths
    under :data:`OUTPUT_PATHS`, for example tables rewritten by an earlier pipeline step.
    """
    cwd = Path(repo) if repo else None

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.rstrip()

    commit = _git("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "dirty": None, "outputs_modified": None}
    # ":(top)" makes each pathspec relative to the repository root, not the cwd.
    outputs_spec = [f":(top){p}" for p in OUTPUT_PATHS]
    excludes = [f":(top,exclude){p}" for p in OUTPUT_PATHS]
    source = _git("status", "--porcelain", "--", ":/", *excludes)
    outputs = _git("status", "--porcelain", "--", *outputs_spec)
    return {
        "commit": commit,
        "dirty": bool(source) if source is not None else None,
        "outputs_modified": [line[3:] for line in outputs.splitlines()]
        if outputs is not None
        else None,
    }


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
    out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out
