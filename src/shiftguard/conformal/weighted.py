r"""Weighted conformal prediction for covariate shift (Tibshirani et al., 2019).

Each calibration point is weighted by the likelihood ratio
:math:`w(x) = dP^{test}_X / dP^{cal}_X (x)`, and each test point gets its own quantile
:math:`\hat q(x)` (see :func:`~shiftguard.conformal.quantiles.weighted_conformal_quantile`).
With the true ratio, coverage on the test distribution is at least :math:`1 - \alpha`.
In practice :math:`w` is estimated (:mod:`shiftguard.shift.weights`), so the guarantee is
only approximate.

Effective sample size
---------------------
Heavy-tailed weights concentrate the calibration mass on a few points. Kish's effective
sample size of the calibration weights,

.. math:: n_{\mathrm{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}
          \in [1, n],

tells how many equally weighted points the weighted calibration set is worth. A small
:math:`n_{\mathrm{eff}}` means unstable quantiles (and, below
:func:`~shiftguard.conformal.quantiles.min_calibration_size`, mostly infinite ones).

Weight clipping
---------------
Optionally, weights are clipped to ``[w_lo, w_hi]``: the given quantiles of the
calibration weights. The same bounds are applied to the test weights. Clipping trades a
small bias (the guarantee no longer holds exactly) for much lower variance. It is a
heuristic and is reported as such.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shiftguard.conformal.base import BaseConformal
from shiftguard.conformal.quantiles import weighted_conformal_quantile
from shiftguard.conformal.scores import ScoreName, true_label_scores


def effective_sample_size(weights: ArrayLike) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` (0 for all-zero weights)."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size and (not np.all(np.isfinite(w)) or w.min() < 0):
        raise ValueError("weights must be finite and non-negative")
    denom = float(np.sum(w**2))
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0


def weight_clip_bounds(
    reference: ArrayLike, quantiles: tuple[float, float] = (0.01, 0.99)
) -> tuple[float, float]:
    """Clipping bounds: the ``quantiles`` of the reference (calibration) weights."""
    lo_q, hi_q = quantiles
    if not 0.0 <= lo_q < hi_q <= 1.0:
        raise ValueError(f"need 0 <= lower < upper <= 1, got {quantiles}")
    ref = np.asarray(reference, dtype=np.float64).ravel()
    if ref.size == 0:
        raise ValueError("reference weights are empty")
    lo, hi = np.quantile(ref, [lo_q, hi_q])
    return float(lo), float(hi)


def clip_weights(weights: ArrayLike, lower: float, upper: float) -> NDArray[np.float64]:
    """Clip weights to ``[lower, upper]``."""
    if lower > upper:
        raise ValueError(f"lower ({lower}) > upper ({upper})")
    return np.clip(np.asarray(weights, dtype=np.float64), lower, upper)


class WeightedConformal(BaseConformal):
    """Weighted split conformal classifier for covariate shift.

    Parameters
    ----------
    alpha, score, randomized, random_state
        See :class:`~shiftguard.conformal.base.BaseConformal`.
    clip_quantiles
        If given, e.g. ``(0.01, 0.99)``, clip calibration and test weights to these
        quantiles of the calibration weights.

    Attributes
    ----------
    weights_ : numpy.ndarray
        Calibration weights after clipping.
    clip_bounds_ : tuple or None
        The clipping bounds used.
    ess_ : float
        Effective sample size of ``weights_``.
    n_calibration_ : int
        Number of calibration points.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score: ScoreName = "lac",
        *,
        randomized: bool = False,
        random_state: int | None = None,
        clip_quantiles: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(alpha, score, randomized=randomized, random_state=random_state)
        self.clip_quantiles = clip_quantiles

    def fit(self, probs: ArrayLike, labels: ArrayLike, weights: ArrayLike) -> Self:
        """Store calibration scores and (clipped) likelihood-ratio weights."""
        scores = self._score_matrix(probs)
        cal = true_label_scores(scores, labels)
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != cal.shape:
            raise ValueError(f"weights must have shape {cal.shape}, got {w.shape}")
        if not np.all(np.isfinite(w)) or w.min(initial=0.0) < 0:
            raise ValueError("weights must be finite and non-negative")
        self.clip_bounds_: tuple[float, float] | None = None
        if self.clip_quantiles is not None:
            self.clip_bounds_ = weight_clip_bounds(w, self.clip_quantiles)
            w = clip_weights(w, *self.clip_bounds_)
        self.n_classes_ = scores.shape[1]
        self.n_calibration_ = cal.size
        self.cal_scores_ = cal
        self.weights_ = w
        self.ess_ = effective_sample_size(w)
        return self

    def quantiles(self, test_weights: ArrayLike) -> NDArray[np.float64]:
        """Per-test-point quantiles ``q_hat(x)`` given test weights ``w(x)``."""
        self._check_fitted()
        wt = np.asarray(test_weights, dtype=np.float64).ravel()
        if self.clip_bounds_ is not None:
            wt = clip_weights(wt, *self.clip_bounds_)
        return weighted_conformal_quantile(self.cal_scores_, self.weights_, wt, self.alpha)

    def predict_sets(self, probs: ArrayLike, test_weights: ArrayLike) -> NDArray[np.bool_]:
        """Prediction sets ``{y : s(x, y) <= q_hat(x)}``."""
        self._check_fitted()
        scores = self._score_matrix(probs)
        q = self.quantiles(test_weights)
        if q.shape != (scores.shape[0],):
            raise ValueError("need exactly one test weight per test point")
        return scores <= q[:, None]
