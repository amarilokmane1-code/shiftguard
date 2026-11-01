r"""Split conformal prediction: marginal and class-conditional.

Marginal split conformal uses one quantile :math:`\hat q` of all calibration scores
(:func:`~shiftguard.conformal.quantiles.conformal_quantile`) and returns
:math:`C(x) = \{y : s(x, y) \le \hat q\}`. Coverage is guaranteed on average over
calibration and test draws, *if* they are exchangeable.

Class-conditional (label-conditional) conformal computes one quantile per class from the
calibration points of that class,

.. math:: \hat q_k = \text{conformal quantile of } \{S_i : Y_i = k\},
          \qquad C(x) = \{k : s(x, k) \le \hat q_k\},

which guarantees :math:`P\{Y \in C(X) \mid Y = k\} \ge 1 - \alpha` for every class
(Vovk, 2012; Sadinle et al., 2019). Under class imbalance, marginal coverage can hide
under-coverage of the minority class; the class-conditional version cannot.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shiftguard.conformal.base import BaseConformal
from shiftguard.conformal.quantiles import conformal_quantile
from shiftguard.conformal.scores import true_label_scores, validate_labels


class SplitConformal(BaseConformal):
    """Marginal split conformal classifier.

    Attributes
    ----------
    qhat_ : float
        The conformal quantile (``inf`` if the calibration set is too small).
    n_calibration_ : int
        Number of calibration points.
    """

    def fit(self, probs: ArrayLike, labels: ArrayLike) -> Self:
        """Compute the conformal quantile from calibration probabilities and labels."""
        scores = self._score_matrix(probs)
        cal = true_label_scores(scores, labels)
        self.n_classes_ = scores.shape[1]
        self.n_calibration_ = cal.size
        self.qhat_ = conformal_quantile(cal, self.alpha)
        return self

    def predict_sets(self, probs: ArrayLike) -> NDArray[np.bool_]:
        """Prediction sets ``{y : s(x, y) <= q_hat}`` as a boolean matrix."""
        self._check_fitted()
        return self._score_matrix(probs) <= self.qhat_


class ClassConditionalConformal(BaseConformal):
    """Class-conditional split conformal classifier (one quantile per class).

    Attributes
    ----------
    qhat_ : numpy.ndarray
        Quantile per class, shape ``(n_classes,)``.
    n_per_class_ : numpy.ndarray
        Calibration points per class.
    """

    def fit(self, probs: ArrayLike, labels: ArrayLike) -> Self:
        """Compute one conformal quantile per class."""
        scores = self._score_matrix(probs)
        n_classes = scores.shape[1]
        y = validate_labels(labels, scores.shape[0], n_classes)
        cal = true_label_scores(scores, y)
        self.n_classes_ = n_classes
        self.n_per_class_ = np.bincount(y, minlength=n_classes)
        self.qhat_ = np.array(
            [conformal_quantile(cal[y == k], self.alpha) for k in range(n_classes)]
        )
        return self

    def predict_sets(self, probs: ArrayLike) -> NDArray[np.bool_]:
        """Prediction sets ``{k : s(x, k) <= q_hat_k}``."""
        self._check_fitted()
        return self._score_matrix(probs) <= self.qhat_[None, :]
