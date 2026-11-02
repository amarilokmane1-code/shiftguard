"""Command-line interface: one command per pipeline stage.

``shiftguard download``  fetch ABIDE I (PCP) time series and phenotypes
``shiftguard features``  QC + functional-connectivity features
``shiftguard run``       run an experiment from a YAML config
``shiftguard summarize`` rebuild a run's tables from its saved predictions
``shiftguard conformal`` conformal prediction under site shift (phase 5)
``shiftguard site-effect`` site effect on the features before / after ComBat
``shiftguard detect``    label-free shift detection (phase 6)
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
) -> Path:
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
    return run_dir


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
def conformal(
    baseline: Annotated[Path, typer.Argument(help="Baseline run folder (shiftguard run).")],
    combat: Annotated[
        Path | None, typer.Option(help="Baseline run refitted on ComBat features.")
    ] = None,
    n_jobs: Annotated[int, typer.Option(help="Parallel domain-classifier folds.")] = 1,
    alpha: Annotated[
        float | None,
        typer.Option(help="Override the miscoverage level (sensitivity analysis)."),
    ] = None,
) -> Path:
    """Phase 5: conformal prediction under site shift, from saved baseline predictions.

    With ``--alpha``, outputs carry an ``alpha<value>`` report tag (e.g. ``_alpha0p2``),
    so a sensitivity run never overwrites the main tables and figures.
    """
    import pandas as pd

    from shiftguard.config import load_config
    from shiftguard.data.synthetic import load_any_dataset
    from shiftguard.evaluation import conformal_report as rep
    from shiftguard.evaluation.conformal_runner import load_run_predictions, run_conformal
    from shiftguard.utils.paths import reports_dir
    from shiftguard.utils.provenance import create_run_dir, write_run_metadata

    try:
        preds = load_run_predictions(baseline)
        combat_preds = load_run_predictions(combat) if combat else None
    except FileNotFoundError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=3) from exc
    cfg = load_config(baseline / "config.yaml")
    if alpha is not None:
        if not 0 < alpha < 1:
            log.error("--alpha must be in (0, 1), got %s", alpha)
            raise typer.Exit(code=2)
        cfg["conformal"]["alpha"] = alpha
        tag = cfg["experiment"].get("report_tag", "")
        a_tag = "alpha" + f"{alpha:g}".replace(".", "p")
        cfg["experiment"]["report_tag"] = f"{tag}_{a_tag}" if tag else a_tag
    if combat_preds is not None:
        rep.check_same_splits(preds, combat_preds, "ComBat run")
    reports = reports_dir(cfg)
    run_dir = create_run_dir(reports, "conformal")
    setup_logging("INFO", log_file=run_dir / "run.log")
    names = {"baseline": baseline.name, "combat": combat.name if combat else ""}
    write_run_metadata(
        run_dir,
        cfg,
        seeds=sorted(preds["seed"].unique().tolist()),
        extra={"command": "conformal"} | names,
    )
    ds = load_any_dataset(cfg)  # raw features (never ComBat) for the domain classifier
    weights, w_info = rep.domain_weights_cached(
        ds, preds, cfg, reports / "cache" / f"domain_weights_{baseline.name}", n_jobs
    )
    w_info = w_info.assign(variant="none")
    c_weights = None
    if combat is not None and combat_preds is not None:
        c_cfg = load_config(combat / "config.yaml")
        c_ds = load_any_dataset(c_cfg)  # ComBat-harmonised features (no labels used)
        c_weights, c_info = rep.domain_weights_cached(
            c_ds, combat_preds, c_cfg, reports / "cache" / f"domain_weights_{combat.name}", n_jobs
        )
        w_info = pd.concat([w_info, c_info.assign(variant="combat")], ignore_index=True)
    outcomes, diag = run_conformal(preds, cfg, weights, combat_preds, c_weights)
    diag = diag.merge(w_info, on=["seed", "test_site", "variant"], how="left")
    head = rep.write_outputs(outcomes, diag, cfg, run_dir, reports, names)
    cols = ["protocol", "model", "method", "coverage", "worst_site_coverage", "abstention"]
    typer.echo(head[[c for c in cols if c in head]].round(3).to_string(index=False))
    typer.echo(f"Run folder: {run_dir}")
    return run_dir


@app.command("site-effect")
def site_effect(
    config: ConfigOption = DEFAULT_EXPERIMENT, overrides: OverrideOption = None
) -> Path:
    """Describe the site effect on the features, before and after ComBat (Figure 2)."""
    import copy

    from shiftguard.config import load_config
    from shiftguard.data.synthetic import load_any_dataset
    from shiftguard.evaluation.site_effect import (
        describe_site_effect,
        save_site_effect,
        write_site_effect_outputs,
    )
    from shiftguard.features.harmonize import harmonize_dataset
    from shiftguard.utils.paths import reports_dir
    from shiftguard.utils.provenance import create_run_dir, write_run_metadata

    cfg = load_config(config, overrides)
    cfg.setdefault("features", {})["harmonization"] = "none"
    reports = reports_dir(cfg)
    run_dir = create_run_dir(reports, "site_effect")
    setup_logging("INFO", log_file=run_dir / "run.log")
    seed = int(cfg["experiment"]["base_seed"])
    write_run_metadata(run_dir, cfg, seeds=[seed], extra={"command": "site-effect"})
    ds = load_any_dataset(cfg)
    c_cfg = copy.deepcopy(cfg)
    c_cfg["features"]["harmonization"] = "combat"
    ds_c = harmonize_dataset(ds, c_cfg)  # age and sex only; never the diagnosis
    cov = ds.phenotypes[["AGE_AT_SCAN", "SEX"]].assign(dx=ds.labels)
    effects = {
        name: describe_site_effect(d.features, ds.sites, cov, seed=seed)
        for name, d in (("none", ds), ("combat", ds_c))
    }
    save_site_effect(run_dir, effects, ds.sites)
    table = write_site_effect_outputs(
        effects, ds.sites, run_dir.name, reports, cfg["experiment"].get("report_tag", "")
    )
    typer.echo(table.round(3).to_string(index=False))
    typer.echo(f"Run folder: {run_dir}")
    return run_dir


@app.command()
def detect(
    conformal_run: Annotated[Path, typer.Argument(help="Run folder of `shiftguard conformal`.")],
) -> Path:
    """Phase 6: label-free signals vs the real coverage drop per deployment site."""
    import json

    import pandas as pd

    from shiftguard.config import load_config
    from shiftguard.evaluation import detection as det
    from shiftguard.utils.paths import reports_dir
    from shiftguard.utils.provenance import create_run_dir, write_run_metadata

    needed = [
        conformal_run / n for n in ("config.yaml", "fold_diagnostics.csv", "site_coverage.csv")
    ]
    missing = [p.name for p in needed if not p.is_file()]
    if missing:
        log.error(
            "%s is not a conformal run folder (missing %s)", conformal_run, ", ".join(missing)
        )
        raise typer.Exit(code=3)
    cfg = load_config(needed[0])
    alpha = float(cfg["conformal"]["alpha"])
    n_boot = int(cfg["statistics"]["n_bootstrap"])
    diag = pd.read_csv(needed[1], dtype={"test_site": str})
    sites = pd.read_csv(needed[2], dtype={"test_site": str})
    reports = reports_dir(cfg)
    run_dir = create_run_dir(reports, "detection")
    setup_logging("INFO", log_file=run_dir / "run.log")
    write_run_metadata(
        run_dir,
        cfg,
        seeds=sorted(diag["seed"].unique().tolist()),
        extra={"command": "detect", "conformal_run": conformal_run.name},
    )
    loso = diag[diag["protocol"] == "loso"]
    settings = [
        (m, v)
        for m, v in (("logreg", "none"), ("mlp", "none"), ("logreg", "combat"))
        if ((loso["model"] == m) & (loso["variant"] == v)).any()
    ]
    site_sig = pd.concat(
        [det.site_signals(diag, sites, alpha, m, v) for m, v in settings], ignore_index=True
    )
    corr = det.correlation_table(site_sig, n_boot=n_boot)
    ref = {m: det.within_site_reference(diag, m) for m in sorted(set(diag["model"]))}
    site_sig.to_csv(run_dir / "site_signals.csv", index=False, lineterminator="\n")
    corr.to_csv(run_dir / "correlations.csv", index=False, lineterminator="\n")
    (run_dir / "within_site_reference.json").write_text(
        json.dumps(ref, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    det.write_detection_outputs(site_sig, corr, ref, cfg, run_dir.name, conformal_run.name, reports)
    p = corr[(corr["model"] == "logreg") & (corr["variant"] == "none")]
    typer.echo(
        p[["signal", "target", "rho", "ci_low", "ci_high", "q_value"]]
        .round(3)
        .to_string(index=False)
    )
    typer.echo(f"Run folder: {run_dir}")
    return run_dir


@app.command()
def report(
    config: ConfigOption = DEFAULT_EXPERIMENT,
    overrides: OverrideOption = None,
    manifest: Annotated[
        Path | None,
        typer.Option(help="Run manifest (default: <reports_dir>/final_runs.yaml)."),
    ] = None,
    results: Annotated[Path, typer.Option(help="Output Markdown file.")] = Path("docs/RESULTS.md"),
    readme: Annotated[
        Path | None, typer.Option(help="README whose key-results block is refreshed.")
    ] = Path("README.md"),
) -> None:
    """Regenerate tables, figures and docs/RESULTS.md from the saved final runs."""
    from shiftguard.config import load_config
    from shiftguard.evaluation import report as rep
    from shiftguard.utils.paths import reports_dir

    cfg = load_config(config, overrides)
    reports = reports_dir(cfg)
    path = manifest or reports / rep.MANIFEST
    if not path.is_file():
        log.error("No run manifest at %s (written by `shiftguard all`)", path)
        raise typer.Exit(code=3)
    runs = rep.load_manifest(path)
    status = rep.regenerate(runs, reports)
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(rep.build_results_md(reports, runs), encoding="utf-8", newline="\n")
    for key, st in status.items():
        typer.echo(f"{key:<20} {runs[key]:<34} {st}")
    typer.echo(f"Wrote {results}")
    if readme is not None and readme.is_file():
        if rep.update_readme(readme, reports / "tables"):
            typer.echo(f"Updated the key-results block of {readme}")
        else:
            log.warning("%s has no key-results markers; not updated", readme)


@app.command()
def simulate(
    n_trials: Annotated[int, typer.Option(help="Trials per experiment.")] = 4000,
    alpha: Annotated[float, typer.Option(help="Miscoverage level.")] = 0.1,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    reports: Annotated[Path, typer.Option("--reports-dir", help="Output root.")] = Path("reports"),
) -> Path:
    """Synthetic validity checks of the conformal core (no data needed)."""
    from shiftguard.evaluation.simulation import run_all_simulations, write_simulation_tables
    from shiftguard.utils.provenance import create_run_dir, write_run_metadata

    log.info("Running synthetic conformal experiments (n_trials=%d, seed=%d)", n_trials, seed)
    df = run_all_simulations(n_trials=n_trials, alpha=alpha, seed=seed).frame()
    run_dir = create_run_dir(reports, "simulate")
    config = {"simulate": {"n_trials": n_trials, "alpha": alpha, "seed": seed}}
    write_run_metadata(run_dir, config, seeds=[seed], extra={"command": "simulate"})
    md_path = write_simulation_tables(df, reports, n_trials, seed, run_dir=run_dir)
    log.info("Wrote %s and run folder %s", md_path, run_dir)
    typer.echo(md_path.read_text(encoding="utf-8"))
    return run_dir


@app.command("all")
def run_all(
    config: ConfigOption = DEFAULT_EXPERIMENT,
    overrides: OverrideOption = None,
    n_jobs: Annotated[int | None, typer.Option(help="Parallel jobs (-1 = all).")] = None,
    sensitivity: Annotated[
        bool, typer.Option(help="Also run the alpha = 0.2 and ROI-exclusion analyses.")
    ] = False,
    simulate_trials: Annotated[
        int, typer.Option(help="Trials per synthetic validity check.")
    ] = 4000,
) -> None:
    """Run the whole pipeline, write the run manifest and the report.

    download -> features -> synthetic checks -> baselines -> ComBat refit -> conformal ->
    site effect -> detection -> manifest -> ``<reports>/RESULTS.md`` (plus, with
    --sensitivity, alpha = 0.2 and ROI-level exclusion). Synthetic configs skip download
    and features. The repository's docs/RESULTS.md and README are only refreshed by an
    explicit ``shiftguard report``.
    """
    from shiftguard.config import load_config
    from shiftguard.evaluation import report as rep
    from shiftguard.utils.paths import reports_dir

    base = list(overrides or [])
    cfg = load_config(config, base)
    reports = reports_dir(cfg)
    jobs = 1 if n_jobs is None else n_jobs
    abide = cfg.get("data", {}).get("source", "abide") != "synthetic"
    name = cfg["experiment"]["name"]
    combat_ovr = [
        "features.harmonization=combat",
        "experiment.protocols=[loso]",
        "experiment.models=[logreg]",
        f"experiment.name={name}_combat",
        "experiment.report_tag=combat",
    ]
    runs: dict[str, str] = {}
    if abide:
        download(config, base)
        features(config, base)
    runs["simulate"] = simulate(n_trials=simulate_trials, reports=reports).name
    b = run(config, base, n_jobs)
    c = run(config, base + combat_ovr, n_jobs)
    runs["baseline"], runs["baseline_combat"] = b.name, c.name
    conf = conformal(b, c, n_jobs=jobs)
    runs["conformal"] = conf.name
    if sensitivity:
        runs["conformal_alpha0p2"] = conformal(b, c, n_jobs=jobs, alpha=0.2).name
        if abide:
            drop_ovr = [
                "qc.zero_variance_policy=drop_rois",
                "experiment.protocols=[loso]",
                "experiment.models=[logreg]",
                f"experiment.name={name}_droprois",
                "experiment.report_tag=droprois",
            ]
            features(config, base + drop_ovr)
            d = run(config, base + drop_ovr, n_jobs)
            runs["baseline_droprois"] = d.name
            runs["conformal_droprois"] = conformal(d, n_jobs=jobs).name
    runs["site_effect"] = site_effect(config, base).name
    runs["detection"] = detect(conf).name
    rep.save_manifest(reports / rep.MANIFEST, runs)
    typer.echo(f"Wrote {reports / rep.MANIFEST}")
    # never touches docs/ or the README: run `shiftguard report` for that
    report(config, base, results=reports / "RESULTS.md", readme=None)


if __name__ == "__main__":  # pragma: no cover
    app()
