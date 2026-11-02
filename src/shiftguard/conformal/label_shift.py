r"""Weighted split conformal prediction under label shift.

Under label shift the likelihood ratio between target and source depends on the label
only, :math:`w(x, y) = \pi_t(y) / \pi_s(y)`. Following Podkopaev and Ramdas (2021), the
candidate label :math:`y` enters the prediction set when

.. math::

    s(x, y) \le \mathrm{Quantile}_{1-\alpha}\Big(
        \sum_{i=1}^n p_i^{y}\, \delta_{S_i} + p_{n+1}^{y}\, \delta_{+\infty}\Big),
    \qquad
    p_i^{y} = \frac{w(Y_i)}{\sum_{j} w(Y_j) + w(y)},\;
    p_{n+1}^{y} = \frac{w(y)}{\sum_{j} w(Y_j) + w(y)} .

The test-point weight depends on the candidate label, not on :math:`x`, so there is one
threshold per class. With the true priors the marginal coverage is at least
:math:`1-\alpha` under pure label shift; with estimated priors (EM) it is approximate.
Class-conditional conformal prediction gives the stronger per-class guarantee under pure
label shift without any weights.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shiftguard.conformal.base import BaseConformal
from shiftguard.conformal.quantiles import weighted_conformal_quantile
from shiftguard.conformal.scores import true_label_scores, validate_labels
from shiftguard.conformal.weighted import effective_sample_size


class LabelShiftConformal(BaseConformal):
    """Split conformal classifier reweighted by per-class prior ratios.

    Attributes
    ----------
    class_weights_ : numpy.ndarray
        ``w(y) = pi_t(y) / pi_s(y)`` for each class.
    thresholds_ : numpy.ndarray
        One score threshold per candidate class.
    ess_ : float
        Effective sample size of the calibration weights.
    """

    def fit(self, probs: ArrayLike, labels: ArrayLike, class_weights: ArrayLike) -> Self:
        """Compute one weighted-quantile threshold per candidate class."""
        scores = self._score_matrix(probs)
        y = validate_labels(labels, scores.shape[0], scores.shape[1])
        w = np.asarray(class_weights, dtype=np.float64).ravel()
        if w.shape != (scores.shape[1],) or not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError(f"class_weights must be {scores.shape[1]} positive finite values")
        cal = true_label_scores(scores, y)
        cal_w = w[y]
        self.n_classes_ = scores.shape[1]
        self.class_weights_ = w
        self.thresholds_ = weighted_conformal_quantile(cal, cal_w, w, self.alpha)
        self.ess_ = effective_sample_size(cal_w)
        return self

    def predict_sets(self, probs: ArrayLike) -> NDArray[np.bool_]:
        """Prediction sets ``{y : s(x, y) <= threshold_y}``."""
        self._check_fitted()
        return self._score_matrix(probs) <= self.thresholds_[None, :]
