"""Command-line interface: one command per pipeline stage.

``shiftguard download``  fetch ABIDE I (PCP) time series and phenotypes
``shiftguard features``  QC + functional-connectivity features
``shiftguard run``       run an experiment from a YAML config
``shiftguard report``    regenerate tables, figures and docs/RESULTS.md
``shiftguard all``       the whole pipeline
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated

import typer

from shiftguard import __version__
from shiftguard.utils.log import get_logger, setup_logging
from shiftguard.utils.provenance import git_info, package_versions

app = typer.Typer(
    name="shiftguard",
    help="ShiftGuard: does conformal abstention survive site shift?",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger(__name__)

DEFAULT_EXPERIMENT = Path("configs/experiment/loso.yaml")

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Experiment YAML config.", exists=False),
]
OverrideOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a config value, e.g. -s experiment.n_seeds=3."),
]


def _not_ready(stage: str, phase: int) -> None:
    """Exit with a clear message for stages that are implemented in a later phase."""
    log.error("`shiftguard %s` is implemented in phase %d and is not available yet.", stage, phase)
    raise typer.Exit(code=2)


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Configure logging before any command runs."""
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def info() -> None:
    """Print package, environment and git information."""
    git = git_info()
    typer.echo(f"shiftguard {__version__}")
    typer.echo(f"python     {sys.version.split()[0]} ({platform.platform()})")
    typer.echo(f"git        {git['commit'] or 'n/a'}{' (dirty)' if git['dirty'] else ''}")
    for name, ver in package_versions().items():
        if name != "shiftguard":
            typer.echo(f"  {name:<16} {ver or '-'}")


@app.command()
def download(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Download ABIDE I preprocessed ROI time series and phenotypes."""
    _not_ready("download", 3)


@app.command()
def features(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Apply QC and compute functional-connectivity features."""
    _not_ready("features", 3)


@app.command()
def run(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Run an experiment (LOSO or within-site) described by a YAML config."""
    _not_ready("run", 4)


@app.command()
def report(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Regenerate tables, figures and docs/RESULTS.md from saved results."""
    _not_ready("report", 7)


@app.command("all")
def run_all(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Run download, features, experiments and report in sequence."""
    _not_ready("all", 7)


if __name__ == "__main__":  # pragma: no cover
    app()
