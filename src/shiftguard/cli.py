"""Command-line interface: one command per pipeline stage.

``shiftguard download``  fetch ABIDE I (PCP) time series and phenotypes
``shiftguard features``  QC + functional-connectivity features
``shiftguard run``       run an experiment from a YAML config
``shiftguard summarize`` rebuild a run's tables from its saved predictions
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
def download(
    config: ConfigOption = DEFAULT_EXPERIMENT,
    overrides: OverrideOption = None,
    method: Annotated[str, typer.Option(help="auto | nilearn | s3")] = "auto",
    n_jobs: Annotated[int, typer.Option(help="Parallel HTTPS connections (s3).")] = 8,
) -> None:
    """Download ABIDE I preprocessed ROI time series and phenotypes."""
    from shiftguard.config import load_config
    from shiftguard.data.download import AbideSpec, OfflineError, download_abide
    from shiftguard.utils.paths import data_dir

    cfg = load_config(config, overrides)
    if method not in ("auto", "nilearn", "s3"):
        raise typer.BadParameter("method must be auto, nilearn or s3")
    try:
        rep = download_abide(
            data_dir(cfg),
            AbideSpec.from_config(cfg["data"]),
            method=method,  # type: ignore[arg-type]
            n_jobs=n_jobs,
        )
    except OfflineError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=3) from exc
    typer.echo(
        f"{rep.n_present}/{rep.n_expected} time series on disk, "
        f"{rep.bytes_on_disk / 1e6:.1f} MB, method: {rep.method_used}"
    )
    if rep.missing:
        log.warning("Missing after download (%d): %s", len(rep.missing), ", ".join(rep.missing))


@app.command()
def features(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Apply QC and compute functional-connectivity features."""
    from shiftguard.config import load_config
    from shiftguard.data.dataset import build_dataset

    cfg = load_config(config, overrides)
    try:
        qc = build_dataset(cfg)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=3) from exc
    totals = qc.summary.set_index("SITE_ID").loc["TOTAL"]
    typer.echo(
        f"Kept {totals['n_after']}/{totals['n_before']} subjects "
        f"(ASD {totals['n_after_ASD']}, TDC {totals['n_after_TDC']}) "
        f"from {qc.included['SITE_ID'].nunique()} sites."
    )
    for reason, count in qc.reason_counts.items():
        typer.echo(f"  excluded, {reason:<22} {count}")


@app.command()
def run(
    config: ConfigOption = DEFAULT_EXPERIMENT,
    overrides: OverrideOption = None,
    n_jobs: Annotated[int | None, typer.Option(help="Parallel fold fits (-1 = all).")] = None,
) -> None:
    """Run an experiment (LOSO and/or within-site) described by a YAML config."""
    from shiftguard.config import load_config
    from shiftguard.data.synthetic import load_any_dataset
    from shiftguard.evaluation.runner import run_experiment, write_baseline_outputs
    from shiftguard.utils.paths import reports_dir
    from shiftguard.utils.provenance import create_run_dir, write_run_metadata

    cfg = load_config(config, overrides)
    exp = cfg["experiment"]
    try:
        ds = load_any_dataset(cfg)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=3) from exc
    reports = reports_dir(cfg)
    run_dir = create_run_dir(reports, exp["name"])
    setup_logging("INFO", log_file=run_dir / "run.log")
    write_run_metadata(
        run_dir,
        cfg,
        seeds=list(range(exp["n_seeds"])),
        extra={"command": "run", "n_subjects": len(ds.labels), "n_sites": len(set(ds.sites))},
    )
    log.info("Run folder: %s (%d subjects, %d sites)", run_dir, len(ds.labels), len(set(ds.sites)))
    preds, info = run_experiment(ds, cfg, n_jobs=n_jobs)
    table = write_baseline_outputs(preds, info, cfg, run_dir, reports)
    typer.echo(
        table[
            ["protocol", "model", "auc", "auc_site_mean", "balanced_accuracy", "ece_raw", "ece_cal"]
        ].to_string(index=False)
    )
    typer.echo(f"Run folder: {run_dir}")


@app.command()
def summarize(
    run_dir: Annotated[Path, typer.Argument(help="Existing run folder (reports/runs/...).")],
) -> None:
    """Rebuild the tables and Figure 6 of an existing run from its saved predictions."""
    import pandas as pd

    from shiftguard.config import load_config
    from shiftguard.evaluation.runner import write_baseline_outputs
    from shiftguard.utils.paths import reports_dir

    needed = [run_dir / n for n in ("config.yaml", "predictions.csv.gz", "fold_info.csv")]
    missing = [p.name for p in needed if not p.is_file()]
    if missing:
        log.error("%s is not a complete run folder (missing %s)", run_dir, ", ".join(missing))
        raise typer.Exit(code=3)
    cfg = load_config(needed[0])
    preds, info = pd.read_csv(needed[1]), pd.read_csv(needed[2])
    table = write_baseline_outputs(preds, info, cfg, run_dir, reports_dir(cfg), save_raw=False)
    typer.echo(table[["protocol", "model", "auc", "ece_cal", "ece_null"]].to_string(index=False))


@app.command()
def report(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Regenerate tables, figures and docs/RESULTS.md from saved results."""
    _not_ready("report", 7)


@app.command()
def simulate(
    n_trials: Annotated[int, typer.Option(help="Trials per experiment.")] = 4000,
    alpha: Annotated[float, typer.Option(help="Miscoverage level.")] = 0.1,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    reports: Annotated[Path, typer.Option("--reports-dir", help="Output root.")] = Path("reports"),
) -> None:
    """Synthetic validity checks of the conformal core (no data needed)."""
    from shiftguard.evaluation.simulation import run_all_simulations
    from shiftguard.utils.provenance import create_run_dir, write_run_metadata
    from shiftguard.utils.tables import write_table

    log.info("Running synthetic conformal experiments (n_trials=%d, seed=%d)", n_trials, seed)
    df = run_all_simulations(n_trials=n_trials, alpha=alpha, seed=seed).frame()
    df["coverage_95ci"] = [
        f"{c:.3f} [{lo:.3f}, {hi:.3f}]"
        for c, lo, hi in zip(df.coverage, df.ci_low, df.ci_high, strict=True)
    ]
    df["p_below"] = [f"{p:.2g}" for p in df.p_below_target]
    columns = [
        "experiment",
        "method",
        "score",
        "n_classes",
        "n_calibration",
        "target",
        "coverage_95ci",
        "p_below",
        "mean_set_size",
        "abstention_rate",
        "mean_ess",
        "theory",
    ]
    title = f"Synthetic conformal validity checks ({n_trials} trials, seed {seed})"
    csv_path, md_path = write_table(
        df, reports / "tables" / "conformal_synthetic", markdown_columns=columns, title=title
    )
    run_dir = create_run_dir(reports, "simulate")
    write_table(df, run_dir / "conformal_synthetic", markdown_columns=columns, title=title)
    config = {"simulate": {"n_trials": n_trials, "alpha": alpha, "seed": seed}}
    write_run_metadata(run_dir, config, seeds=[seed], extra={"command": "simulate"})
    log.info("Wrote %s, %s and run folder %s", csv_path, md_path, run_dir)
    typer.echo(md_path.read_text(encoding="utf-8"))


@app.command("all")
def run_all(config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None) -> None:
    """Run download, features, experiments and report in sequence."""
    _not_ready("all", 7)


if __name__ == "__main__":  # pragma: no cover
    app()
