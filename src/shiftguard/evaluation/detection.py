"""Phase 6: label-free detection of a coverage loss at a new deployment site.

Exploratory by design: there are as many data points as deployment sites (18), and the
coverage of each site is itself measured with binomial noise (25-172 subjects).

Signals (computed per fold in ``conformal_runner.label_free_signals`` and the domain
classifier, from the deployment site's *features and predictions only*), averaged over
seeds per site:

``domain_auc``           out-of-fold AUC of a classifier separating training and deployment
                         features (covariate shift in the connectome)
``ks_prob``              KS statistic, P(ASD) on calibration half B vs deployment site
``ks_maxprob``           KS statistic, max-probability (confidence), same samples
``abs_confidence_shift`` |mean max-probability (deployment) - (calibration)|
``abstention``           split-CP abstention rate at the deployment site
``abs_abstention_shift`` |abstention (deployment) - abstention (calibration half B)|
``abs_mean_p_shift``     |mean P(ASD) (deployment) - (calibration)|
``abs_em_shift``         |EM-estimated deployment ASD fraction - source ASD fraction|

Targets (from the labels, for evaluation only), per site, seed-averaged:

``coverage_drop``        (1 - alpha) - split-CP coverage (positive = below target)
``worst_class_drop``     (1 - alpha) - min(coverage of ASD, coverage of TDC)

The signed signals ``mean_p_shift`` and ``em_shift`` are also related to the signed class
gap (coverage of ASD - coverage of TDC): they could say *which* class loses coverage.

Statistics: Spearman rho across sites; 95% percentile bootstrap CI over sites (2000
resamples); two-sided permutation p-value (10,000 permutations); Benjamini-Hochberg
across all signal x target tests of a setting; the range of rho over leave-one-site-out;
and a partial Spearman rho given site size (small sites have larger KS statistics and
more extreme coverage by chance alone). Within-site folds (no shift) give the
reference value of every signal when calibration and deployment data are exchangeable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from shiftguard.evaluation.statistics import benjamini_hochberg

UNSIGNED = [
    "domain_auc",
    "ks_prob",
    "ks_maxprob",
    "abs_confidence_shift",
    "abstention",
    "abs_abstention_shift",
    "abs_mean_p_shift",
    "abs_em_shift",
]
SIGNED = ["mean_p_shift", "em_shift"]
SIGNAL_LABELS = {
    "domain_auc": "Domain-classifier AUC",
    "ks_prob": "KS statistic, P(ASD)",
    "ks_maxprob": "KS statistic, confidence",
    "abs_confidence_shift": "|Confidence shift|",
    "abstention": "Abstention rate (split CP)",
    "abs_abstention_shift": "|Abstention shift|",
    "abs_mean_p_shift": "|Mean P(ASD) shift|",
    "abs_em_shift": "|EM prevalence shift|",
    "mean_p_shift": "Mean P(ASD) shift",
    "em_shift": "EM prevalence shift",
}
TARGET_LABELS = {
    "coverage_drop": "Coverage drop below target",
    "worst_class_drop": "Worst-class coverage drop",
    "class_gap": "Coverage ASD - TDC",
}


def fold_signals(diag: pd.DataFrame) -> pd.DataFrame:
    """Derive the label-free signals of every fold from the conformal fold diagnostics."""
    d = diag.copy()
    d["abs_confidence_shift"] = (d["mean_maxprob_test"] - d["mean_maxprob_cal"]).abs()
    d["abstention"] = d["abstain_split_test"]
    d["abs_abstention_shift"] = (d["abstain_split_test"] - d["abstain_split_cal"]).abs()
    d["mean_p_shift"] = d["mean_p_test"] - d["mean_p_cal"]
    d["abs_mean_p_shift"] = d["mean_p_shift"].abs()
    d["em_shift"] = d["pi_target_em"] - d["pi_source"]
    d["abs_em_shift"] = d["em_shift"].abs()
    if "domain_auc" not in d:
        d["domain_auc"] = np.nan
    return d


def site_signals(
    diag: pd.DataFrame, sites: pd.DataFrame, alpha: float, model: str, variant: str
) -> pd.DataFrame:
    """One row per deployment site: seed-averaged signals and coverage targets."""
    d = fold_signals(diag)
    d = d[(d["protocol"] == "loso") & (d["model"] == model) & (d["variant"] == variant)]
    sig = d.groupby("test_site")[UNSIGNED + SIGNED].mean()
    method = "split" if variant == "none" else f"{variant}+split"
    s = sites[(sites["protocol"] == "loso") & (sites["model"] == model)]
    s = s[s["method"] == method].set_index("test_site")
    target = 1.0 - alpha
    out = sig.join(
        pd.DataFrame(
            {
                "n": s["n"],
                "coverage": s["coverage"],
                "coverage_drop": target - s["coverage"],
                "worst_class_drop": target - s[["coverage_asd", "coverage_tdc"]].min(axis=1),
                "class_gap": s["coverage_asd"] - s["coverage_tdc"],
            }
        ),
        how="inner",
    )
    return out.reset_index().assign(model=model, variant=variant)


def within_site_reference(diag: pd.DataFrame, model: str) -> dict[str, float]:
    """Mean of every fold-level signal on within-site (exchangeable) folds."""
    d = fold_signals(diag)
    d = d[(d["protocol"] == "within_site") & (d["model"] == model)]
    return {c: float(d[c].mean()) for c in UNSIGNED + SIGNED if c in d and d[c].notna().any()}


def _resid(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    return r - np.polyval(np.polyfit(z, r, 1), z)


def spearman_with_ci(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    n_boot: int = 2000,
    n_perm: int = 10_000,
    seed: int = 0,
    covariate: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Spearman rho, percentile bootstrap CI over sites and a permutation p-value.

    ``rho_loo_min`` / ``rho_loo_max``: the range of rho when each site in turn is left
    out (how much a single site drives the result). With ``covariate`` (site size), the
    partial Spearman correlation given the covariate is added (``rho_partial``: Pearson
    correlation of the rank residuals after regressing each rank on the covariate's
    rank) with a permutation p-value (``p_partial``, residuals of ``y`` permuted).
    """
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    zz = np.zeros_like(xx) if covariate is None else np.asarray(covariate, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz)
    xx, yy, zz = xx[ok], yy[ok], zz[ok]
    n = len(xx)
    empty = {
        "rho": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "rho_loo_min": np.nan,
        "rho_loo_max": np.nan,
        "n_sites": n,
    }
    if covariate is not None:
        empty |= {"rho_partial": np.nan, "p_partial": np.nan}
    if n < 4 or np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return empty
    rho = float(stats.spearmanr(xx, yy).statistic)
    loo = [stats.spearmanr(np.delete(xx, i), np.delete(yy, i)).statistic for i in range(n)]
    rng = np.random.default_rng(seed)
    rx, ry = stats.rankdata(xx), stats.rankdata(yy)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(xx[i]) > 0 and np.ptp(yy[i]) > 0:
            boots.append(stats.spearmanr(xx[i], yy[i]).statistic)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    p = (1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / (n_perm + 1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    out = {
        "rho": rho,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p),
        "rho_loo_min": float(np.nanmin(loo)),
        "rho_loo_max": float(np.nanmax(loo)),
        "n_sites": n,
    }
    if covariate is not None:
        rz = stats.rankdata(zz)
        ex, ey = _resid(rx, rz), _resid(ry, rz)
        rp = float(np.corrcoef(ex, ey)[0, 1])
        pp = np.array([np.corrcoef(ex, rng.permutation(ey))[0, 1] for _ in range(n_perm)])
        out |= {
            "rho_partial": rp,
            "p_partial": float((1 + np.sum(np.abs(pp) >= abs(rp) - 1e-12)) / (n_perm + 1)),
        }
    return out


def correlation_table(
    site_sig: pd.DataFrame, n_boot: int = 2000, n_perm: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """Every signal against every target, per (model, variant) setting, with BH q-values.

    Unsigned signals are tested against ``coverage_drop`` and ``worst_class_drop``; signed
    signals against ``class_gap``. BH runs across all tests of a setting, separately for
    the plain and the size-adjusted (partial, given the site's number of deployment
    subjects) correlations: under no shift both the KS statistics and the spread of a
    site's coverage grow as the site gets smaller, which can create a correlation by
    itself.
    """
    rows = []
    for (model, variant), g in site_sig.groupby(["model", "variant"], sort=False):
        pairs = [(s, t) for t in ("coverage_drop", "worst_class_drop") for s in UNSIGNED]
        pairs += [(s, "class_gap") for s in SIGNED]
        for sig, target in pairs:
            res = spearman_with_ci(
                g[sig], g[target], n_boot=n_boot, n_perm=n_perm, seed=seed, covariate=g["n"]
            )
            rows.append({"model": model, "variant": variant, "signal": sig, "target": target} | res)
    tab = pd.DataFrame(rows)
    for p_col, q_col in (("p_value", "q_value"), ("p_partial", "q_partial")):
        tab[q_col] = np.nan
        for _, idx in tab.groupby(["model", "variant"]).groups.items():
            ok = tab.loc[idx, p_col].notna().to_numpy()
            tab.loc[idx[ok], q_col] = benjamini_hochberg(tab.loc[idx[ok], p_col])
    return tab
