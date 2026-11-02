"""Summaries and inference for phase 5 (conformal prediction under site shift).

Unit of analysis
    A deployment site. Each site is evaluated with 10 seeds (different train /
    calibration splits); per-site rates are averaged over seeds.

Per-site inference
    Clopper-Pearson 95% interval and one-sided exact binomial test of
    ``coverage < 1 - alpha`` on the seed-averaged count (``round(mean covered)`` out of the
    site's evaluated subjects): the uncertainty of a site of that size. Benjamini-Hochberg
    across the sites of each (model, method) for marginal coverage, and across sites x
    classes for per-class coverage. The binomial test treats coverage at a site as a
    fixed probability; conformal coverage given one calibration set is itself random
    (Beta-distributed), so the tests are approximate for a single seed and conservative
    once averaged.

Between methods
    Paired bootstrap over sites (2000 resamples) of the difference to split CP in mean
    site coverage, mean shortfall ``max(0, 1 - alpha - coverage)`` and abstention, with
    BH across comparisons (primary). Secondary: a two-stage paired bootstrap that also
    resamples subjects within each drawn site (columns ``*_2stage``). Mondrian arms are
    compared with split CP on the same subjects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from shiftguard.evaluation.statistics import (
    benjamini_hochberg,
    beta_binomial_eb,
    binomial_test_below,
    clopper_pearson,
    paired_bootstrap,
    two_stage_paired_bootstrap,
)

GROUP = ["protocol", "model", "method"]


def per_seed_site(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Count covered, abstained and per-class cases per (protocol, model, method, seed, site)."""
    o = outcomes.copy()
    o["covered"] = np.where(o["label"] == 1, o["in1"], o["in0"]).astype(int)
    o["size"] = o["in0"].astype(int) + o["in1"].astype(int)
    o["abstain"] = (o["size"] != 1).astype(int)
    o["single"] = 1 - o["abstain"]
    o["single_correct"] = o["single"] * o["covered"]  # a singleton that covers is correct
    o["asd"] = (o["label"] == 1).astype(int)
    o["cov_asd"] = o["covered"] * o["asd"]
    o["cov_tdc"] = o["covered"] * (1 - o["asd"])
    agg = o.groupby([*GROUP, "seed", "test_site"], sort=True).agg(
        n=("covered", "size"),
        k=("covered", "sum"),
        n_asd=("asd", "sum"),
        k_asd=("cov_asd", "sum"),
        k_tdc=("cov_tdc", "sum"),
        abstain=("abstain", "sum"),
        single=("single", "sum"),
        single_correct=("single_correct", "sum"),
        size=("size", "sum"),
    )
    agg["n_tdc"] = agg["n"] - agg["n_asd"]
    return agg.reset_index()


def _rate(k: pd.Series, n: pd.Series) -> pd.Series:
    return k / n.where(n > 0)


def site_table(per_seed: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Seed-averaged coverage per site, overall and per class, with CIs and BH q-values.

    ``coverage_eb`` is the empirical-Bayes (beta-binomial) posterior mean of each site's
    coverage, fitted across the sites of the same (protocol, model, method): small sites
    are shrunk towards the across-site mean. ``selective_accuracy`` is the accuracy on the
    non-abstained (singleton) cases.
    """
    target = 1.0 - alpha
    ps = per_seed.assign(
        coverage=_rate(per_seed["k"], per_seed["n"]),
        coverage_asd=_rate(per_seed["k_asd"], per_seed["n_asd"]),
        coverage_tdc=_rate(per_seed["k_tdc"], per_seed["n_tdc"]),
        abstention=_rate(per_seed["abstain"], per_seed["n"]),
        mean_set_size=_rate(per_seed["size"], per_seed["n"]),
        selective_accuracy=_rate(per_seed["single_correct"], per_seed["single"]),
    )
    rows = []
    for key, g in ps.groupby([*GROUP, "test_site"], sort=True):
        row = dict(zip([*GROUP, "test_site"], key, strict=True))
        n = round(g["n"].mean())
        row |= {
            "n": n,
            "n_asd": round(g["n_asd"].mean()),
            "n_tdc": round(g["n_tdc"].mean()),
            "n_seeds": g["seed"].nunique(),
        }
        for c in (
            "coverage",
            "coverage_asd",
            "coverage_tdc",
            "abstention",
            "mean_set_size",
            "selective_accuracy",
        ):
            row[c] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else np.nan
        for suffix, nn, kk in (
            ("", n, g["k"].mean()),
            ("_asd", row["n_asd"], g["k_asd"].mean()),
            ("_tdc", row["n_tdc"], g["k_tdc"].mean()),
        ):
            if nn >= 1:
                k_int = int(min(nn, round(kk)))
                lo, hi = clopper_pearson(k_int, nn)
                p = binomial_test_below(k_int, nn, target)
            else:
                lo = hi = p = np.nan
            row |= {f"ci_low{suffix}": lo, f"ci_high{suffix}": hi, f"p_under{suffix}": p}
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["q_under"] = np.nan
    tab["q_under_asd"] = np.nan
    tab["q_under_tdc"] = np.nan
    tab["coverage_eb"] = np.nan
    for _, idx in tab.groupby(GROUP).groups.items():
        if len(idx) >= 2 and tab.loc[idx, "n"].min() >= 1:
            nn = tab.loc[idx, "n"].to_numpy()
            kk = np.minimum(nn, np.round(tab.loc[idx, "coverage"].to_numpy() * nn))
            tab.loc[idx, "coverage_eb"] = beta_binomial_eb(kk, nn).shrunk
        tab.loc[idx, "q_under"] = benjamini_hochberg(tab.loc[idx, "p_under"])
        both = benjamini_hochberg(np.r_[tab.loc[idx, "p_under_asd"], tab.loc[idx, "p_under_tdc"]])
        tab.loc[idx, "q_under_asd"] = both[: len(idx)]
        tab.loc[idx, "q_under_tdc"] = both[len(idx) :]
    return tab


def headline_table(
    per_seed: pd.DataFrame, sites: pd.DataFrame, alpha: float, fdr: float = 0.05
) -> pd.DataFrame:
    """One row per (protocol, model, method)."""
    target = 1.0 - alpha
    rows = []
    for key, g in per_seed.groupby(GROUP, sort=False):
        pooled = g.groupby("seed")[
            [
                "n",
                "k",
                "n_asd",
                "k_asd",
                "n_tdc",
                "k_tdc",
                "abstain",
                "size",
                "single",
                "single_correct",
            ]
        ].sum()
        s = sites[(sites[GROUP] == pd.Series(key, index=GROUP)).all(axis=1)]
        row = dict(zip(GROUP, key, strict=True))
        row |= {
            "n_sites": len(s),
            "coverage": float((pooled["k"] / pooled["n"]).mean()),
            "coverage_asd": float((pooled["k_asd"] / pooled["n_asd"]).mean()),
            "coverage_tdc": float((pooled["k_tdc"] / pooled["n_tdc"]).mean()),
            "abstention": float((pooled["abstain"] / pooled["n"]).mean()),
            "mean_set_size": float((pooled["size"] / pooled["n"]).mean()),
            "selective_accuracy": float(
                (pooled["single_correct"] / pooled["single"].where(pooled["single"] > 0)).mean()
            ),
        }
        if key[0] == "loso":
            shortfall = np.maximum(0.0, target - s["coverage"])
            class_cov = np.r_[s["coverage_asd"].dropna(), s["coverage_tdc"].dropna()]
            row |= {
                "mean_site_coverage": float(s["coverage"].mean()),
                "worst_site_coverage": float(s["coverage"].min()),
                "worst_site": s.loc[s["coverage"].idxmin(), "test_site"],
                "coverage_gap": float(shortfall.mean()),
                "worst_site_class_coverage": float(class_cov.min()) if class_cov.size else np.nan,
                "n_sites_under": int((s["q_under"] < fdr).sum()),
                "n_site_classes_under": int(
                    (s["q_under_asd"] < fdr).sum() + (s["q_under_tdc"] < fdr).sum()
                ),
            }
            k_avg = np.minimum(s["n"], np.round(s["coverage"] * s["n"]))
            if len(s) >= 2 and s["n"].min() >= 1:
                eb = beta_binomial_eb(k_avg.to_numpy(), s["n"].to_numpy())
                row["between_site_sd"] = eb.between_site_sd
        rows.append(row)
    return pd.DataFrame(rows)


def _site_vectors(
    per_seed: pd.DataFrame, outcomes: pd.DataFrame, model: str, method: str, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed-averaged site metrics for ``method`` and for split CP on the same subjects."""
    target = 1.0 - alpha
    o = outcomes[(outcomes["protocol"] == "loso") & (outcomes["model"] == model)]
    m = o[o["method"] == method]
    split = o[o["method"] == "split"]
    if method.startswith("mondrian_"):
        split = split.merge(m[["seed", "test_site", "SUB_ID"]], on=["seed", "test_site", "SUB_ID"])

    def site_metrics(x: pd.DataFrame) -> pd.DataFrame:
        ps = per_seed_site(x)
        ps["cov"] = ps["k"] / ps["n"]
        ps["abst"] = ps["abstain"] / ps["n"]
        v = ps.groupby("test_site")[["cov", "abst"]].mean()
        v["shortfall"] = np.maximum(0.0, target - v["cov"])
        return v

    a, b = site_metrics(m), site_metrics(split)
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def _subject_groups(
    outcomes: pd.DataFrame, model: str, method: str
) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
    """Per-site, per-subject paired values (method, split CP), averaged over seeds.

    Returns ``{"cov": (method groups, split groups), "abst": (...)}`` with one array per
    site, subjects in the same order in both. Mondrian arms are paired with split CP on
    the subjects and seeds they were evaluated on.
    """
    o = outcomes[(outcomes["protocol"] == "loso") & (outcomes["model"] == model)]
    keys = ["seed", "test_site", "SUB_ID"]
    m = o[o["method"] == method]
    split = o[o["method"] == "split"].merge(m[keys], on=keys)

    def per_subject(x: pd.DataFrame) -> pd.DataFrame:
        x = x.assign(
            cov=np.where(x["label"] == 1, x["in1"], x["in0"]).astype(float),
            abst=((x["in0"] + x["in1"]) != 1).astype(float),
        )
        return x.groupby(["test_site", "SUB_ID"])[["cov", "abst"]].mean()

    both = per_subject(m).join(per_subject(split), lsuffix="_m", rsuffix="_s", how="inner")
    out: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {"cov": ([], []), "abst": ([], [])}
    for _, g in both.groupby(level="test_site", sort=True):
        for k in ("cov", "abst"):
            out[k][0].append(g[f"{k}_m"].to_numpy())
            out[k][1].append(g[f"{k}_s"].to_numpy())
    return out


def comparisons_vs_split(
    per_seed: pd.DataFrame,
    outcomes: pd.DataFrame,
    alpha: float,
    n_boot: int = 2000,
    seed: int = 0,
    two_stage: bool = True,
) -> pd.DataFrame:
    """Paired site bootstrap of every LOSO method against split CP (same model).

    With ``two_stage``, the two-stage (sites, then subjects) bootstrap is added as a
    secondary analysis, with its own BH q-values within model.
    """
    rows = []
    loso = per_seed[per_seed["protocol"] == "loso"]
    target = 1.0 - alpha

    def shortfall(v: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, target - v)

    for (model, method), _ in loso.groupby(["model", "method"], sort=False):
        if method == "split":
            continue
        a, b = _site_vectors(per_seed, outcomes, model, method, alpha)
        groups = _subject_groups(outcomes, model, method) if two_stage else None
        for metric, col, key, transform in (
            ("coverage", "cov", "cov", None),
            ("shortfall", "shortfall", "cov", shortfall),
            ("abstention", "abst", "abst", None),
        ):
            res = paired_bootstrap(a[col], b[col], n_boot=n_boot, seed=seed)
            row = {"model": model, "method": method, "metric": metric, "n_sites": len(a)} | res
            if groups is not None:
                gx, gy = groups[key]
                two = two_stage_paired_bootstrap(gx, gy, transform, n_boot=n_boot, seed=seed)
                row |= {
                    "ci_low_2stage": two["ci_low"],
                    "ci_high_2stage": two["ci_high"],
                    "p_2stage": two["p_value"],
                }
            rows.append(row)
    tab = pd.DataFrame(rows)
    if len(tab):
        for p_col, q_col in (("p_value", "q_value"), ("p_2stage", "q_2stage")):
            if p_col not in tab:
                continue
            tab[q_col] = np.nan
            for _, idx in tab.groupby("model").groups.items():
                tab.loc[idx, q_col] = benjamini_hochberg(tab.loc[idx, p_col])
    return tab


def prevalence_table(diag: pd.DataFrame, model: str = "logreg") -> pd.DataFrame:
    """Per-site prevalence-shift diagnostic (no ComBat), mean ± SD over seeds."""
    d = diag[(diag["protocol"] == "loso") & (diag["model"] == model) & (diag["variant"] == "none")]
    d = d.assign(
        true_shift=d["pi_target_true"] - d["pi_source"],
        em_shift=d["pi_target_em"] - d["pi_source"],
        em_error=d["pi_target_em"] - d["pi_target_true"],
        cil_excess=d["cil_test"] - d["cil_label_shift_pred"],
    )
    cols = [
        "n_test",
        "pi_target_true",
        "pi_source",
        "true_shift",
        "pi_target_em",
        "em_shift",
        "em_error",
        "cil_test",
        "cil_test_em",
        "discrimination_slope",
        "cil_label_shift_pred",
        "cil_excess",
        "domain_auc",
        "ess_weighted",
        "ess_weighted_clip",
        "inf_rate_weighted",
        "inf_rate_weighted_clip",
        "n_cal_B",
    ]
    cols = [c for c in cols if c in d.columns]
    mean = d.groupby("test_site")[cols].mean()
    sd = d.groupby("test_site")[cols].std(ddof=1).add_suffix("_sd")
    return pd.concat([mean, sd], axis=1).reset_index()


def offset_vs_class_gap(
    sites: pd.DataFrame, prev: pd.DataFrame, model: str = "logreg", alpha: float = 0.1
) -> dict[str, Any]:
    """Test whether a site's calibration offset predicts which class loses coverage.

    Spearman correlation across LOSO sites between calibration-in-the-large and the
    per-class coverage gap (ASD minus TDC), for split and class-conditional CP. A positive
    offset (ASD over-predicted) should favour ASD and cost TDC coverage.

    Also the EM-estimated prevalence shift (no target label) against the split-CP
    coverage drop below target and against the class gap; the full label-free analysis
    (all signals, bootstrap CIs, BH, size adjustment) is in ``evaluation/detection.py``.
    """
    out: dict[str, Any] = {}
    loso = sites[(sites["protocol"] == "loso") & (sites["model"] == model)]

    def _rho(x: pd.Series, y: pd.Series) -> dict[str, Any]:
        r = stats.spearmanr(x, y)
        return {"rho": float(r.statistic), "p_value": float(r.pvalue), "n_sites": len(x)}

    cols = ["test_site", "cil_test", "em_shift"]
    for method in ("split", "class_conditional"):
        m = loso[loso["method"] == method].merge(prev[cols], on="test_site")
        if len(m) < 3:
            continue
        gap = m["coverage_asd"] - m["coverage_tdc"]
        out[f"offset_vs_class_gap_{method}"] = _rho(m["cil_test"], gap)
        if method == "split":
            out["em_shift_vs_class_gap_split"] = _rho(m["em_shift"], gap)
            out["abs_em_shift_vs_coverage_drop_split"] = _rho(
                m["em_shift"].abs(), (1 - alpha) - m["coverage"]
            )
    return out


def prevalence_correlations(
    prev: pd.DataFrame, diag: pd.DataFrame | None = None, model: str = "logreg"
) -> dict[str, Any]:
    """Across-site summaries of the prevalence diagnostic used in the text.

    ``n_sites_offset_beyond_label_shift`` counts sites whose offset exceeds the largest one
    pure label shift could produce (``|cil| > |true_shift|``) by more than twice its SD over
    seeds. With ``diag``, the EM error on within-site (no shift) folds is a sanity check.
    """
    out: dict[str, Any] = {}
    for name, x, y in (
        ("em_vs_true_shift", "em_shift", "true_shift"),
        ("cil_vs_true_shift", "cil_test", "true_shift"),
        ("cil_after_em_vs_true_shift", "cil_test_em", "true_shift"),
    ):
        r = stats.spearmanr(prev[x], prev[y])
        out[name] = {"rho": float(r.statistic), "p_value": float(r.pvalue), "n_sites": len(prev)}
    out["mean_abs_cil_before_em"] = float(prev["cil_test"].abs().mean())
    out["mean_abs_cil_after_em"] = float(prev["cil_test_em"].abs().mean())
    out["mean_abs_em_error"] = float(prev["em_error"].abs().mean())
    beyond = prev["cil_test"].abs() - prev["true_shift"].abs() > 2 * prev["cil_test_sd"]
    out["n_sites_offset_beyond_label_shift"] = int(beyond.sum())
    out["sites_offset_beyond_label_shift"] = prev.loc[beyond, "test_site"].tolist()
    # share of the across-site sum of squared offsets that pure label shift accounts for
    resid = prev["cil_test"] - prev["cil_label_shift_pred"]
    out["r2_cil_explained_by_label_shift"] = float(
        1.0 - (resid**2).sum() / (prev["cil_test"] ** 2).sum()
    )
    if diag is not None:
        w = diag[(diag["protocol"] == "within_site") & (diag["model"] == model)]
        if len(w):
            err = (w["pi_target_em"] - w["pi_target_true"]).abs()
            out["within_site_mean_abs_em_error"] = float(err.mean())
            out["within_site_max_abs_em_error"] = float(err.max())
    return out
