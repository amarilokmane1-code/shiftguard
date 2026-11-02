"""Run the baseline experiments and summarise them.

For every protocol, seed, split and model:

1. features for train / calibration / test (subject-wise features are indexed; tangent
   features are fitted on the training rows only, inside the fold);
2. the model is fitted on the training rows (nested CV inside, for logistic regression);
3. logits are computed for calibration and test rows;
4. the calibrator is fitted on the calibration rows only;
5. raw and calibrated probabilities are stored for calibration and test rows.

The per-subject predictions (``predictions.csv.gz`` in the run folder, git-ignored) are
the input of the conformal phase, so the models are trained once.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from shiftguard.data.dataset import Dataset
from shiftguard.evaluation.metrics import binary_metrics
from shiftguard.evaluation.protocols import Split, check_split, make_splits
from shiftguard.features.connectivity import fit_transform_fold
from shiftguard.models import make_model
from shiftguard.models.calibration import make_calibrator, null_ece, sigmoid
from shiftguard.utils.log import get_logger
from shiftguard.utils.seeding import derive_seed

log = get_logger(__name__)


def features_for_split(
    ds: Dataset, split: Split, features_cfg: dict[str, Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Train / calibration / test features without leakage."""
    if ds.features is not None:
        return ds.features[split.train], ds.features[split.cal], ds.features[split.test]
    X_tr, (X_cal, X_te) = fit_transform_fold(
        ds.time_series,
        split.train,
        [split.cal, split.test],
        kind=features_cfg["kind"],
        fisher_z=features_cfg.get("fisher_z", True),
        cov_estimator=features_cfg.get("cov_estimator", "ledoit_wolf"),
    )
    return X_tr, X_cal, X_te


def run_fold(
    ds: Dataset, split: Split, model_name: str, config: dict[str, Any], single_thread: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit, calibrate and predict one (split, model); return predictions and fold info."""
    if single_thread and model_name == "mlp":
        import torch

        torch.set_num_threads(1)
    t0 = time.perf_counter()
    y = ds.labels
    X_tr, X_cal, X_te = features_for_split(ds, split, config["features"])
    seed = derive_seed(
        config["experiment"]["base_seed"], model_name, split.protocol, split.seed, split.test_site
    )
    model = make_model(model_name, config["models"], seed).fit(X_tr, y[split.train])
    z_cal, z_te = model.decision_function(X_cal), model.decision_function(X_te)
    calibrator = make_calibrator(config["calibration"]["method"]).fit(z_cal, y[split.cal])

    pheno = ds.phenotypes
    frames = []
    for part, idx, z in (("cal", split.cal, z_cal), ("test", split.test, z_te)):
        frames.append(
            pd.DataFrame(
                {
                    "protocol": split.protocol,
                    "model": model_name,
                    "seed": split.seed,
                    "test_site": split.test_site,
                    "part": part,
                    "row": idx,
                    "SUB_ID": pheno["SUB_ID"].to_numpy()[idx],
                    "SITE_ID": pheno["SITE_ID"].to_numpy()[idx],
                    "label": y[idx],
                    "logit": z,
                    "p_raw": sigmoid(z),
                    "p_cal": calibrator.predict_proba(z),
                }
            )
        )
    info = {
        "protocol": split.protocol,
        "model": model_name,
        "seed": split.seed,
        "test_site": split.test_site,
        "n_train": len(split.train),
        "n_cal": len(split.cal),
        "n_test": len(split.test),
        "best_params": str(getattr(model, "best_params_", "")),
        "n_epochs": getattr(model, "n_epochs_", np.nan),
        "temperature": getattr(calibrator, "temperature_", np.nan),
        "fit_seconds": time.perf_counter() - t0,
    }
    return pd.concat(frames, ignore_index=True), info


def run_experiment(
    ds: Dataset, config: dict[str, Any], n_jobs: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every protocol x seed x split x model; return predictions and fold info."""
    exp = config["experiment"]
    ids, sites, labels = ds.phenotypes["SUB_ID"].to_numpy(), ds.sites, ds.labels
    tasks: list[tuple[Split, str]] = []
    for protocol in exp["protocols"]:
        splits = make_splits(protocol, sites, labels, exp)
        for split in splits:
            check_split(split, ids, sites)  # leakage guard on every split actually used
        tasks += [(split, model) for model in exp["models"] for split in splits]
    n_jobs = exp.get("n_jobs", 1) if n_jobs is None else n_jobs
    log.info("Running %d fold fits (n_jobs=%s)", len(tasks), n_jobs)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_fold)(ds, split, model, config, n_jobs != 1) for split, model in tasks
    )
    log.info("Fold fits finished in %.1f min", (time.perf_counter() - t0) / 60)
    preds = pd.concat([r[0] for r in results], ignore_index=True)
    info = pd.DataFrame([r[1] for r in results])
    return preds, info


# --------------------------------------------------------------------------- summaries
def fold_metrics(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Test-set metrics per (protocol, model, seed, test_site), raw and calibrated."""
    rows = []
    test = preds[preds["part"] == "test"]
    for key, g in test.groupby(["protocol", "model", "seed", "test_site"], sort=True):
        raw = binary_metrics(g["label"], g["p_raw"], n_bins)
        cal = binary_metrics(g["label"], g["p_cal"], n_bins)
        rows.append(
            dict(zip(("protocol", "model", "seed", "test_site"), key, strict=True))
            | {f"{k}_raw": v for k, v in raw.items() if k not in ("n", "prevalence")}
            | {f"{k}_cal": v for k, v in cal.items() if k not in ("n", "prevalence")}
            | {"ece_null_cal": null_ece(g["p_cal"], n_bins)}
            | {"n": raw["n"], "prevalence": raw["prevalence"]}
        )
    return pd.DataFrame(rows)


def pooled_metrics(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Per (protocol, model, seed): metrics on all test predictions pooled over sites."""
    rows = []
    test = preds[preds["part"] == "test"]
    for key, g in test.groupby(["protocol", "model", "seed"], sort=True):
        raw = binary_metrics(g["label"], g["p_raw"], n_bins)
        cal = binary_metrics(g["label"], g["p_cal"], n_bins)
        rows.append(
            dict(zip(("protocol", "model", "seed"), key, strict=True))
            | {f"{k}_raw": v for k, v in raw.items()}
            | {f"{k}_cal": v for k, v in cal.items()}
            | {"ece_null_cal": null_ece(g["p_cal"], n_bins)}
        )
    return pd.DataFrame(rows)


# metric column in the per-seed tables -> column name in the summary tables
BASELINE_METRICS = {
    "auc_cal": "auc",
    "balanced_accuracy_cal": "balanced_accuracy",
    "ece_raw": "ece_raw",
    "ece_cal": "ece_cal",
    "ece_null_cal": "ece_null",
    "cal_in_large_cal": "cal_in_large",
    "brier_cal": "brier",
}


def _mean_sd(values: Sequence[float]) -> str:
    v = np.asarray(values, dtype=float)
    return f"{np.nanmean(v):.3f} ± {np.nanstd(v, ddof=1):.3f}" if v.size > 1 else f"{v[0]:.3f}"


def baseline_table(pooled: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Headline table: mean ± SD over seeds, per protocol and model.

    ``auc_site_mean`` / ``auc_site_worst`` average / take the minimum of per-site AUC
    within each seed (LOSO only), then summarise over seeds.
    """
    rows = []
    for (protocol, model), g in pooled.groupby(["protocol", "model"], sort=False):
        row = {"protocol": protocol, "model": model, "n_seeds": g["seed"].nunique()}
        for col in BASELINE_METRICS:
            row[BASELINE_METRICS[col]] = _mean_sd(g[col])
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_sd"] = float(g[col].std(ddof=1)) if len(g) > 1 else np.nan
        f = folds[(folds["protocol"] == protocol) & (folds["model"] == model)]
        if protocol == "loso" and len(f):
            per_seed = f.groupby("seed")["auc_cal"]
            row["auc_site_mean"] = _mean_sd(per_seed.mean())
            row["auc_site_worst"] = _mean_sd(per_seed.min())
        else:
            row["auc_site_mean"] = row["auc_site_worst"] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def write_baseline_outputs(
    preds: pd.DataFrame,
    info: pd.DataFrame,
    config: dict[str, Any],
    run_dir: Path,
    reports: Path,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Save per-subject outputs in the run folder and aggregate tables/figures in reports.

    With ``save_raw=False`` the predictions and fold info are not rewritten (used by
    ``shiftguard summarize`` to rebuild the tables of an existing run without refitting).
    """
    from shiftguard.utils.tables import write_table
    from shiftguard.viz.figures import figure_reliability

    n_bins = config["calibration"]["n_bins"]
    # experiment.report_tag keeps variant runs (e.g. ComBat) from overwriting the main tables
    tag = config["experiment"].get("report_tag", "")
    sfx = f"_{tag}" if tag else ""
    if save_raw:
        preds.to_csv(run_dir / "predictions.csv.gz", index=False, lineterminator="\n")
        info.to_csv(run_dir / "fold_info.csv", index=False, lineterminator="\n")
    folds = fold_metrics(preds, n_bins)
    pooled = pooled_metrics(preds, n_bins)
    folds.to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    pooled.to_csv(run_dir / "pooled_metrics.csv", index=False, lineterminator="\n")

    table = baseline_table(pooled, folds)
    cols = [
        "protocol",
        "model",
        "n_seeds",
        "auc",
        "auc_site_mean",
        "auc_site_worst",
        "balanced_accuracy",
        "ece_raw",
        "ece_cal",
        "ece_null",
        "cal_in_large",
        "brier",
    ]
    title = (
        f"Baselines: mean ± SD over seeds (run {run_dir.name}). AUC, balanced accuracy "
        "and Brier use temperature-scaled probabilities. ece_null = expected ECE of a perfectly "
        "calibrated model with the same probabilities and test size; cal_in_large = mean "
        "P(ASD) minus observed ASD fraction."
    )
    for out in (run_dir / "baselines", reports / "tables" / f"baselines{sfx}"):
        write_table(table, out, markdown_columns=cols, title=title)
    sites = site_table(folds)
    if len(sites):
        site_cols = [
            "model",
            "test_site",
            "n",
            "pct_ASD",
            "auc",
            "balanced_accuracy",
            "ece_raw",
            "ece_cal",
            "ece_null",
            "cal_in_large",
        ]
        for out in (run_dir / "loso_sites", reports / "tables" / f"loso_sites{sfx}"):
            write_table(
                sites,
                out,
                markdown_columns=site_cols,
                digits=1,
                title=(
                    f"LOSO per deployment site: mean ± SD over seeds (run {run_dir.name}). "
                    "ece_null = size-matched ECE of a perfectly calibrated model."
                ),
            )
    figure_reliability(preds, reports / "figures" / f"fig6_reliability{sfx}", n_bins)
    return table


def site_table(folds: pd.DataFrame) -> pd.DataFrame:
    """LOSO per-site AUC, balanced accuracy and ECE: mean ± SD over seeds."""
    loso = folds[folds["protocol"] == "loso"]
    rows = []
    for (model, site), g in loso.groupby(["model", "test_site"], sort=True):
        rows.append(
            {
                "model": model,
                "test_site": site,
                "n": int(g["n"].iloc[0]),
                "pct_ASD": 100 * float(g["prevalence"].iloc[0]),
                "auc": _mean_sd(g["auc_cal"]),
                "balanced_accuracy": _mean_sd(g["balanced_accuracy_cal"]),
                "ece_raw": _mean_sd(g["ece_raw"]),
                "ece_cal": _mean_sd(g["ece_cal"]),
                "ece_null": _mean_sd(g["ece_null_cal"]),
                "cal_in_large": _mean_sd(g["cal_in_large_cal"]),
                "auc_mean": float(g["auc_cal"].mean()),
            }
        )
    return pd.DataFrame(rows)
