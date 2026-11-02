r"""Statistical inference for coverage.

Clopper-Pearson interval
------------------------
For :math:`k` covered cases out of :math:`n`, the exact (conservative) two-sided
:math:`1 - \gamma` confidence interval for the coverage probability is

.. math::

    \Big[\, B\big(\tfrac{\gamma}{2};\, k,\, n - k + 1\big),\;
           B\big(1 - \tfrac{\gamma}{2};\, k + 1,\, n - k\big) \Big],

where :math:`B(q; a, b)` is the :math:`q` quantile of a Beta(a, b) distribution, with
the lower bound set to 0 when :math:`k = 0` and the upper bound to 1 when :math:`k = n`.

Test of under-coverage
----------------------
:math:`H_0: \text{coverage} \ge 1 - \alpha` against :math:`H_1: \text{coverage} < 1 - \alpha`,
using the exact one-sided binomial test, with p-value :math:`P\{\mathrm{Bin}(n, 1 - \alpha)
\le k\}`.

Multiplicity
------------
Per-site tests are adjusted with the Benjamini-Hochberg step-up procedure (FDR).

Between-site heterogeneity (empirical Bayes)
--------------------------------------------
Site coverages are modelled as :math:`k_s \sim \mathrm{Bin}(n_s, \theta_s)`,
:math:`\theta_s \sim \mathrm{Beta}(a, b)`. :math:`(a, b)` are fitted by maximum marginal
(beta-binomial) likelihood; each site's coverage is shrunk to its posterior mean
:math:`(a + k_s) / (a + b + n_s)`, and :math:`\sqrt{ab / ((a+b)^2 (a+b+1))}` is the
between-site SD of the true coverage.

Paired comparisons
------------------
Methods are compared on the same deployment sites with a paired bootstrap that
resamples sites (the unit of deployment) with replacement (primary). A two-stage
(hierarchical) paired bootstrap, which resamples sites and then subjects within each
drawn site, is a secondary sensitivity analysis: it adds the within-site sampling noise
of each site's estimate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for a binomial proportion.

    Parameters
    ----------
    k
        Number of successes.
    n
        Number of trials (``n >= 1``).
    level
        Confidence level, e.g. 0.95.

    Returns
    -------
    tuple of float
        ``(lower, upper)``.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n >= 1 and 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    tail = (1.0 - level) / 2.0
    lower = 0.0 if k == 0 else float(stats.beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1.0 - tail, k + 1, n - k))
    return lower, upper


def binomial_test_below(k: int, n: int, target: float) -> float:
    """One-sided exact binomial p-value for ``H1: coverage < target``."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n >= 1 and 0 <= k <= n, got k={k}, n={n}")
    return float(stats.binomtest(k, n, target, alternative="less").pvalue)


def benjamini_hochberg(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs are kept and ignored."""
    p = np.asarray(pvalues, dtype=np.float64)
    q = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = ok.size
    if m == 0:
        return q
    order = ok[np.argsort(p[ok], kind="stable")]
    ranked = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


@dataclass(frozen=True)
class BetaBinomialFit:
    """Empirical-Bayes beta-binomial fit across sites."""

    a: float
    b: float
    mean: float
    between_site_sd: float
    shrunk: NDArray[np.float64]


def beta_binomial_eb(k: ArrayLike, n: ArrayLike, max_precision: float = 1e6) -> BetaBinomialFit:
    """Fit ``Beta(a, b)`` to site coverages by marginal maximum likelihood.

    The precision ``a + b`` is bounded by ``max_precision``; reaching the bound means no
    detectable between-site heterogeneity (the shrunk estimates equal the pooled mean).
    """
    kk = np.asarray(k, dtype=np.float64)
    nn = np.asarray(n, dtype=np.float64)
    if kk.shape != nn.shape or kk.size < 2 or np.any(kk < 0) or np.any(kk > nn):
        raise ValueError("need at least two sites with 0 <= k <= n")
    pooled = np.clip(kk.sum() / nn.sum(), 1e-6, 1 - 1e-6)

    def nll(theta: NDArray[np.float64]) -> float:
        mu, log_prec = 1.0 / (1.0 + np.exp(-theta[0])), theta[1]
        prec = np.exp(log_prec)
        return -float(np.sum(stats.betabinom.logpmf(np.round(kk), nn, mu * prec, (1 - mu) * prec)))

    start = np.array([np.log(pooled / (1 - pooled)), np.log(50.0)])
    res = optimize.minimize(
        nll,
        start,
        method="L-BFGS-B",
        bounds=[(-15, 15), (np.log(1e-2), np.log(max_precision))],
    )
    mu = 1.0 / (1.0 + np.exp(-res.x[0]))
    prec = float(np.exp(res.x[1]))
    a, b = mu * prec, (1 - mu) * prec
    sd = float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
    return BetaBinomialFit(float(a), float(b), float(mu), sd, (a + kk) / (a + b + nn))


def paired_bootstrap(
    x: ArrayLike,
    y: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of ``statistic(x) - statistic(y)`` over shared units (sites).

    Returns the observed difference, a percentile interval and a two-sided p-value
    (twice the smaller tail of the bootstrap distribution around zero).
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("x and y must be 1-D, paired, with at least two units")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diffs = np.array([statistic(a[i]) - statistic(b[i]) for i in idx])
    tail = (1.0 - level) / 2.0
    p = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "diff": float(statistic(a) - statistic(b)),
        "ci_low": float(np.quantile(diffs, tail)),
        "ci_high": float(np.quantile(diffs, 1 - tail)),
        "p_value": float(min(1.0, p)),
    }


def two_stage_paired_bootstrap(
    groups_x: list[ArrayLike],
    groups_y: list[ArrayLike],
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Two-stage (hierarchical) paired bootstrap: sites, then subjects within sites.

    ``groups_x[s]`` and ``groups_y[s]`` hold paired per-subject values of site ``s``
    (e.g. seed-averaged coverage indicators of two methods). Each replicate draws sites
    with replacement and then, independently for every drawn site (also for repeated
    draws of the same site), its subjects with replacement. The statistic is the mean
    over drawn sites of ``transform(site mean of x) - transform(site mean of y)``
    (``transform`` = identity, or e.g. the shortfall ``max(0, target - coverage)``).
    Unlike the site-level bootstrap, the interval also carries the within-site sampling
    noise of each site's estimate. Returns the observed statistic, a percentile interval
    and a two-sided p-value as :func:`paired_bootstrap`.
    """
    xs = [np.asarray(g, dtype=np.float64) for g in groups_x]
    ys = [np.asarray(g, dtype=np.float64) for g in groups_y]
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two paired sites")
    if any(a.shape != b.shape or a.ndim != 1 or a.size < 1 for a, b in zip(xs, ys, strict=True)):
        raise ValueError("every site needs paired 1-D subject values")
    f = (lambda v: v) if transform is None else transform
    n_sites = len(xs)
    rng = np.random.default_rng(seed)
    site_idx = rng.integers(0, n_sites, size=(n_boot, n_sites))
    total = np.zeros(n_boot)
    for s in range(n_sites):
        rep, _ = np.nonzero(site_idx == s)  # replicate of every draw of site s
        if rep.size == 0:
            continue
        sub = rng.integers(0, xs[s].size, size=(rep.size, xs[s].size))
        contrib = f(xs[s][sub].mean(axis=1)) - f(ys[s][sub].mean(axis=1))
        np.add.at(total, rep, contrib)
    diffs = total / n_sites
    observed = float(np.mean([f(np.mean(a)) - f(np.mean(b)) for a, b in zip(xs, ys, strict=True)]))
    tail = (1.0 - level) / 2.0
    p = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "diff": observed,
        "ci_low": float(np.quantile(diffs, tail)),
        "ci_high": float(np.quantile(diffs, 1 - tail)),
        "p_value": float(min(1.0, p)),
    }
