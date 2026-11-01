"""Cross-platform task runner (Windows has no `make` by default).

Usage: python tasks.py <task>   (run without arguments to list tasks)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def install() -> None:
    """Editable install with development extras."""
    _run(PY, "-m", "pip", "install", "-e", ".[dev,harmonize]")
    _run(PY, "-m", "pre_commit", "install")


def lint() -> None:
    """Ruff lint and format check."""
    _run(PY, "-m", "ruff", "check", ".")
    _run(PY, "-m", "ruff", "format", "--check", ".")


def fmt() -> None:
    """Auto-fix lint issues and format the code."""
    _run(PY, "-m", "ruff", "check", "--fix", ".")
    _run(PY, "-m", "ruff", "format", ".")


def test() -> None:
    """Full test suite with coverage."""
    _run(PY, "-m", "pytest", "--cov=shiftguard", "--cov-report=term-missing")


def test_fast() -> None:
    """Tests without the slow marker."""
    _run(PY, "-m", "pytest", "-m", "not slow", "-q")


def pipeline() -> None:
    """Whole ABIDE pipeline (downloads data on first run)."""
    _run(PY, "-m", "shiftguard.cli", "all", "--config", "configs/experiment/loso.yaml")


def synthetic() -> None:
    """End-to-end run on simulated data (no download)."""
    _run(PY, "-m", "shiftguard.cli", "all", "--config", "configs/experiment/synthetic.yaml")


def clean() -> None:
    """Remove caches and build artefacts (never touches data/ or reports/)."""
    for pattern in ("**/__pycache__", ".pytest_cache", ".ruff_cache", "htmlcov", "build", "dist"):
        for path in ROOT.glob(pattern):
            shutil.rmtree(path, ignore_errors=True)
    for path in (ROOT / ".coverage", ROOT / "coverage.xml"):
        path.unlink(missing_ok=True)


TASKS = {
    f.__name__.replace("_", "-"): f
    for f in (install, lint, fmt, test, test_fast, pipeline, synthetic, clean)
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print(__doc__)
        for name, func in TASKS.items():
            print(f"  {name:<10} {func.__doc__}")
        sys.exit(0 if len(sys.argv) == 1 else 1)
    TASKS[sys.argv[1]]()
