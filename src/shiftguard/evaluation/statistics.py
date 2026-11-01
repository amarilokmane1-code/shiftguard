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
"""

from __future__ import annotations

from scipy import stats


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
