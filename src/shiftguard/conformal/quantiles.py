r"""Conformal quantiles, unweighted and weighted.

Split conformal quantile
------------------------
Given calibration scores :math:`S_1, \dots, S_n` and a miscoverage level
:math:`\alpha \in (0, 1)`,

.. math::

    \hat q = S_{(k)}, \qquad k = \lceil (n + 1)(1 - \alpha) \rceil,

the :math:`k`-th smallest score, with :math:`\hat q = +\infty` when :math:`k > n`. If the
calibration and test points are exchangeable, the set
:math:`C(x) = \{y : s(x, y) \le \hat q\}` satisfies

.. math:: 1 - \alpha \;\le\; P\{Y_{n+1} \in C(X_{n+1})\} \;\le\; 1 - \alpha + \frac{1}{n+1},

where the upper bound needs continuous scores (no ties). With :math:`k \le n` and
continuous scores the coverage is exactly :math:`k / (n + 1)`.

Weighted conformal quantile (Tibshirani, Barber, Candes and Ramdas, 2019)
-------------------------------------------------------------------------
Under covariate shift, :math:`P_X` changes between calibration and test but
:math:`P_{Y \mid X}` does not. With likelihood ratio :math:`w(x) = dP^{test}_X / dP^{cal}_X
(x)`, give calibration point :math:`i` and the test point :math:`x` the normalised weights

.. math::

    p_i(x) = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(x)}, \qquad
    p_{n+1}(x) = \frac{w(x)}{\sum_{j=1}^n w(X_j) + w(x)},

and take the :math:`(1 - \alpha)` quantile of the weighted distribution
:math:`\sum_i p_i(x)\,\delta_{S_i} + p_{n+1}(x)\,\delta_{+\infty}`:

.. math::

    \hat q(x) = \inf\Big\{ s : \sum_{i : S_i \le s} p_i(x) \ge 1 - \alpha \Big\},

with :math:`\hat q(x) = +\infty` when the calibration mass alone never reaches
:math:`1 - \alpha`. Then :math:`P\{Y \in C(X)\} \ge 1 - \alpha` on the test distribution.
The quantile depends on the test point through :math:`w(x)`, and it is invariant to
rescaling all weights by a constant. With :math:`w \equiv 1` it reduces exactly to the split
conformal quantile.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Guards ceil() against floating-point error, e.g. 0.9 * 100 = 90.00000000000001.
_REL_EPS = 1e-10


def check_alpha(alpha: float) -> float:
    """Validate a miscoverage level and return it as a float."""
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return a


def conformal_rank(n: int, alpha: float) -> int:
    """Return ``k = ceil((n + 1) * (1 - alpha))``, the rank of the conformal quantile."""
    x = (n + 1) * (1.0 - check_alpha(alpha))
    return math.ceil(x * (1.0 - _REL_EPS))


def min_calibration_size(alpha: float) -> int:
    """Smallest calibration size giving a finite conformal quantile.

    ``k <= n`` holds iff ``n >= (1 - alpha) / alpha``. For ``alpha = 0.1`` this is 9:
    with fewer calibration points every prediction set is the full label set.
    """
    a = check_alpha(alpha)
    return math.ceil((1.0 - a) / a * (1.0 - _REL_EPS))


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Split conformal quantile ``S_(k)`` with ``k = ceil((n + 1)(1 - alpha))``.

    Parameters
    ----------
    scores
        One-dimensional calibration scores.
    alpha
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    float
        The quantile, or ``inf`` when ``k > n`` (including ``n = 0``).
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        raise ValueError("calibration scores must be finite")
    n = s.size
    k = conformal_rank(n, alpha)
    if k > n:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])


def weighted_conformal_quantile(
    scores: ArrayLike,
    weights: ArrayLike,
    test_weights: ArrayLike,
    alpha: float,
) -> NDArray[np.float64]:
    """Weighted conformal quantile ``q_hat(x)`` for each test weight ``w(x)``.

    Parameters
    ----------
    scores
        Calibration scores ``S_1..S_n``.
    weights
        Non-negative likelihood-ratio weights ``w(X_i)`` of the calibration points.
    test_weights
        Weights ``w(x)`` of the test points (any shape; flattened).
    alpha
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        One quantile per test point (``inf`` where the calibration mass is too small).
    """
    a = check_alpha(alpha)
    s = np.asarray(scores, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    wt = np.asarray(test_weights, dtype=np.float64).ravel()
    if s.shape != w.shape:
        raise ValueError(f"scores and weights differ in length: {s.size} vs {w.size}")
    for name, arr in (("weights", w), ("test_weights", wt)):
        if not np.all(np.isfinite(arr)) or (arr.size and arr.min() < 0):
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.all(np.isfinite(s)):
        raise ValueError("calibration scores must be finite")
    if s.size == 0:
        return np.full(wt.shape, np.inf)

    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    cum = np.cumsum(w[order])
    total = cum[-1] + wt
    if np.any(total <= 0):
        raise ValueError("total weight must be positive")
    threshold = (1.0 - a) * total * (1.0 - _REL_EPS)
    idx = np.searchsorted(cum, threshold, side="left")
    finite = idx < s.size
    q = np.full(wt.shape, np.inf)
    q[finite] = s_sorted[idx[finite]]
    return q
