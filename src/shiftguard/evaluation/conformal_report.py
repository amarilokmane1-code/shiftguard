"""Run phase 5 from saved baseline runs and write its tables and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shiftguard.evaluation import conformal_runner as cr
from shiftguard.evaluation import conformal_summary as cs
from shiftguard.utils.log import get_logger
from shiftguard.utils.tables import write_table

log = get_logger(__name__)

# Methods shown in the per-site tables and figures (LAC, primary model)
MAIN_METHODS = [
    "split",
    "class_conditional",
    "weighted",
    "weighted_clip",
    "label_shift_em",
    "label_shift_oracle",
    "combat+split",
    "combat+class_conditional",
    "combat+weighted",
    "combat+weighted_clip",
]


def check_same_splits(preds: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    """Every LOSO fold of ``other`` must use exactly the same cal / test rows as ``preds``."""
    keys = ["seed", "test_site", "part"]
    a = preds[preds["protocol"] == "loso"].drop_duplicates([*keys, "row"])
    b = other[other["protocol"] == "loso"].drop_duplicates([*keys, "row"])
    ra = a.groupby(keys)["row"].apply(lambda r: tuple(sorted(r)))
    rb = b.groupby(keys)["row"].apply(lambda r: tuple(sorted(r)))
    if not ra.reindex(rb.index).equals(rb):
        raise ValueError(f"{name}: calibration / test rows differ from the baseline run")


def check_train_rows(preds: pd.DataFrame, train_rows: dict, n_subjects: int) -> None:
    """Regenerated LOSO training rows must complete the stored cal / test rows exactly."""
    loso = preds[preds["protocol"] == "loso"]
    for (seed, site), g in loso.groupby(["seed", "test_site"]):
        tr = train_rows[(int(seed), str(site))]
        used = np.r_[tr, g["row"].unique()]
        if len(used) != n_subjects or len(np.unique(used)) != n_subjects:
            raise ValueError(f"regenerated split for seed {seed}, site {site} does not match")


def domain_weights_cached(
    ds: Any, preds: pd.DataFrame, config: dict[str, Any], cache: Path, n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute (or reuse) the domain-classifier weights of a baseline run."""
    w_path, i_path = cache.with_suffix(".csv.gz"), cache.with_name(cache.name + "_info.csv")
    if w_path.is_file() and i_path.is_file():
        log.info("Reusing domain weights from %s", w_path)
        return pd.read_csv(w_path, dtype={"test_site": str}), pd.read_csv(
            i_path, dtype={"test_site": str}
        )
    train_rows = cr.train_rows_from_splits(config, ds.sites, ds.labels)
    check_train_rows(preds, train_rows, len(ds.labels))
    w, info = cr.compute_domain_weights(ds.features, preds, train_rows, config, n_jobs=n_jobs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(w_path, index=False, lineterminator="\n")
    info.to_csv(i_path, index=False, lineterminator="\n")
    return w, info


def _fmt(mean: pd.Series, sd: pd.Series | None = None, digits: int = 3) -> pd.Series:
    if sd is None:
        return mean.map(lambda v: "-" if pd.isna(v) else f"{v:.{digits}f}")
    return pd.Series(
        [
            "-"
            if pd.isna(m)
            else (f"{m:.{digits}f}" if pd.isna(s) else f"{m:.{digits}f} ± {s:.{digits}f}")
            for m, s in zip(mean, sd, strict=True)
        ],
        index=mean.index,
    )


def write_outputs(
    outcomes: pd.DataFrame,
    diag: pd.DataFrame,
    config: dict[str, Any],
    run_dir: Path,
    reports: Path,
    run_names: dict[str, str],
) -> pd.DataFrame:
    """Write the phase 5 tables (run folder + reports/tables) and figures."""
    from shiftguard.viz import conformal_figures as cf

    alpha = float(config["conformal"]["alpha"])
    n_boot = int(config["statistics"]["n_bootstrap"])
    outcomes.to_csv(run_dir / "outcomes.csv.gz", index=False, lineterminator="\n")
    diag.to_csv(run_dir / "fold_diagnostics.csv", index=False, lineterminator="\n")
    per_seed = cs.per_seed_site(outcomes)
    sites = cs.site_table(per_seed, alpha)
    head = cs.headline_table(per_seed, sites, alpha)
    comp = cs.comparisons_vs_split(per_seed, outcomes, alpha, n_boot=n_boot)
    prev = cs.prevalence_table(diag)
    corr = cs.prevalence_correlations(prev, diag)
    corr |= cs.offset_vs_class_gap(sites, prev, alpha=alpha)
    for df, name in (
        (sites, "site_coverage"),
        (head, "headline"),
        (comp, "comparisons"),
        (prev, "prevalence"),
    ):
        df.to_csv(run_dir / f"{name}.csv", index=False, lineterminator="\n")
    (run_dir / "prevalence_correlations.json").write_text(
        json.dumps(corr, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    tables = reports / "tables"
    tag = config["experiment"].get("report_tag", "")
    sfx = f"_{tag}" if tag else ""  # variant runs never overwrite the main tables
    src = f"run {run_dir.name}; baseline {run_names['baseline']}"
    if run_names.get("combat"):
        src += f"; ComBat {run_names['combat']}"

    # headline
    h = head.copy()
    for c in (
        "coverage",
        "coverage_asd",
        "coverage_tdc",
        "mean_site_coverage",
        "worst_site_coverage",
        "worst_site_class_coverage",
        "coverage_gap",
        "abstention",
        "mean_set_size",
        "selective_accuracy",
        "between_site_sd",
    ):
        if c in h:
            h[c] = _fmt(h[c])
    write_table(
        h,
        tables / f"conformal_headline{sfx}",
        markdown_columns=[
            c
            for c in (
                "protocol",
                "model",
                "method",
                "coverage",
                "coverage_asd",
                "coverage_tdc",
                "mean_site_coverage",
                "worst_site_coverage",
                "worst_site",
                "coverage_gap",
                "worst_site_class_coverage",
                "n_sites_under",
                "n_site_classes_under",
                "abstention",
                "mean_set_size",
                "selective_accuracy",
                "between_site_sd",
            )
            if c in h
        ],
        title=(
            f"Conformal prediction at alpha = {alpha} (target coverage {1 - alpha:.2f}); "
            f"{src}. Coverage pooled over sites, mean over seeds. Site metrics use "
            "seed-averaged site coverage; coverage_gap = mean over sites of "
            "max(0, target - coverage); n_*_under = BH q < 0.05 for coverage < target; "
            "abstention = set is not a single class; selective_accuracy = accuracy on the "
            "non-abstained cases; between_site_sd = empirical-Bayes SD of true site coverage."
        ),
    )
    # per-site, per-class (primary model, main methods)
    s = sites[(sites["protocol"] == "loso") & (sites["method"].isin(MAIN_METHODS))].copy()
    s["coverage_ci"] = [
        f"{c:.3f} [{lo:.3f}, {hi:.3f}]{' *' if q < 0.05 else ''}"
        for c, lo, hi, q in zip(s["coverage"], s["ci_low"], s["ci_high"], s["q_under"], strict=True)
    ]
    for cls in ("asd", "tdc"):
        s[f"coverage_{cls}_fmt"] = [
            f"{c:.3f}{' *' if q < 0.05 else ''}"
            for c, q in zip(s[f"coverage_{cls}"], s[f"q_under_{cls}"], strict=True)
        ]
    s["abstention_fmt"] = _fmt(s["abstention"])
    s["coverage_eb_fmt"] = _fmt(s["coverage_eb"])
    s["selective_accuracy_fmt"] = _fmt(s["selective_accuracy"])
    write_table(
        s,
        tables / f"conformal_sites{sfx}",
        markdown_columns=[
            "model",
            "method",
            "test_site",
            "n",
            "coverage_ci",
            "coverage_eb_fmt",
            "coverage_asd_fmt",
            "coverage_tdc_fmt",
            "abstention_fmt",
            "selective_accuracy_fmt",
        ],
        title=(
            f"LOSO coverage per deployment site and per class (ASD, TDC), mean over seeds; {src}. "
            "Brackets: Clopper-Pearson 95% CI on the seed-averaged count. "
            "* = BH q < 0.05 for coverage below target. coverage_eb = empirical-Bayes "
            "beta-binomial estimate (shrunk towards the across-site mean, most for small "
            "sites). selective_accuracy = accuracy on non-abstained cases."
        ),
    )
    # comparisons
    c = comp.copy()
    c["difference"] = [
        f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        for d, lo, hi in zip(c["diff"], c["ci_low"], c["ci_high"], strict=True)
    ]
    c["q"] = _fmt(c["q_value"])
    write_table(
        c,
        tables / f"conformal_comparisons{sfx}",
        markdown_columns=["model", "method", "metric", "n_sites", "difference", "q"],
        title=(
            f"Method minus split CP (LOSO), paired bootstrap over sites ({n_boot} resamples), "
            f"95% percentile CI, BH q-values within model; {src}. Mondrian arms are compared "
            "with split CP on the same subjects."
        ),
    )
    # prevalence diagnostic
    p = prev.copy()
    within_em = corr.get("within_site_mean_abs_em_error", float("nan"))
    for col in ("pi_target_true", "pi_source", "true_shift", "cil_test", "domain_auc"):
        p[f"{col}_fmt"] = _fmt(p[col], p.get(f"{col}_sd"))
    for col in ("pi_target_em", "em_error", "cil_test_em", "cil_label_shift_pred", "cil_excess"):
        p[f"{col}_fmt"] = _fmt(p[col], p[f"{col}_sd"])
    p["ess_fmt"] = _fmt(p.get("ess_weighted", pd.Series(np.nan, index=p.index)), digits=1)
    write_table(
        p,
        tables / f"prevalence_shift{sfx}",
        markdown_columns=[
            "test_site",
            "n_test",
            "pi_target_true_fmt",
            "pi_source_fmt",
            "true_shift_fmt",
            "pi_target_em_fmt",
            "em_error_fmt",
            "cil_test_fmt",
            "cil_label_shift_pred_fmt",
            "cil_excess_fmt",
            "cil_test_em_fmt",
            "domain_auc_fmt",
            "ess_fmt",
        ],
        title=(
            "Per-site prevalence shift (logistic regression, LOSO), mean ± SD over seeds. "
            "pi = ASD fraction (target true, source = calibration sites, target EM estimate "
            "from unlabelled deployment data); cil = mean P(ASD) minus true ASD fraction, "
            "before and after the EM adjustment; cil_label_shift_pred = the offset pure label "
            "shift would produce, -(true shift) x (1 - discrimination slope); cil_excess = "
            "observed minus that; ESS = effective sample size of the "
            f"covariate-shift weights on calibration half B. Across sites (Spearman): EM vs "
            f"true shift rho = {corr['em_vs_true_shift']['rho']:.2f}; cil vs true shift "
            f"rho = {corr['cil_vs_true_shift']['rho']:.2f}; mean |cil| "
            f"{corr['mean_abs_cil_before_em']:.3f} before vs "
            f"{corr['mean_abs_cil_after_em']:.3f} after EM; pure label shift accounts "
            f"for {corr['r2_cil_explained_by_label_shift']:.0%} of the summed squared cil; "
            f"{corr['n_sites_offset_beyond_label_shift']} sites have an offset larger than "
            "pure label shift allows (by > 2 SD). EM sanity check on within-site folds "
            f"(no shift): mean |EM error| {within_em:.3f}."
        ),
    )
    figs = reports / "figures"
    cf.figure_coverage_per_site(sites, figs / f"fig3_coverage_per_site{sfx}", alpha)
    cf.figure_class_coverage(sites, figs / f"fig3b_class_coverage{sfx}", alpha)
    cf.figure_tradeoff(head, sites, figs / f"fig4_target_labels{sfx}", alpha)
    cf.figure_abstention_per_site(sites, figs / f"fig5_abstention_per_site{sfx}")
    cf.figure_prevalence(prev, figs / f"figS1_prevalence_shift{sfx}")
    return head
