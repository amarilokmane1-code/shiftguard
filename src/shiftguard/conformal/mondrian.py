r"""Mondrian (group-conditional) conformal prediction.

Calibration points are partitioned by a group variable :math:`G` known at test time
(here: the acquisition site). Each group gets its own quantile,

.. math:: \hat q_g = \text{conformal quantile of } \{S_i : G_i = g\},
          \qquad C(x, g) = \{y : s(x, y) \le \hat q_g\},

so that :math:`P\{Y \in C(X, G) \mid G = g\} \ge 1 - \alpha` for every group, provided
the points *within* a group are exchangeable (Vovk et al., 2005). The price is data: a
group with fewer than :func:`~shiftguard.conformal.quantiles.min_calibration_size`
calibration points has :math:`\hat q_g = +\infty`, so every set is the full label set
(the predictor abstains on that whole group). That is the honest answer, not a bug.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shiftguard.conformal.base import BaseConformal
from shiftguard.conformal.quantiles import conformal_quantile
from shiftguard.conformal.scores import ScoreName, true_label_scores

UnseenPolicy = Literal["full", "error"]


def _as_groups(groups: ArrayLike, n: int) -> NDArray[np.object_]:
    arr = np.asarray(groups, dtype=object).ravel()
    if arr.shape != (n,):
        raise ValueError(f"groups must have shape ({n},), got {arr.shape}")
    return arr


class MondrianConformal(BaseConformal):
    """Group-conditional split conformal classifier.

    Parameters
    ----------
    alpha, score, randomized, random_state
        See :class:`~shiftguard.conformal.base.BaseConformal`.
    unseen
        What to do with a test group that has no calibration data: ``"full"`` returns the
        full label set (valid, conservative); ``"error"`` raises.

    Attributes
    ----------
    qhat_ : dict
        Quantile per group.
    n_per_group_ : dict
        Calibration points per group.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score: ScoreName = "lac",
        *,
        randomized: bool = False,
        random_state: int | None = None,
        unseen: UnseenPolicy = "full",
    ) -> None:
        super().__init__(alpha, score, randomized=randomized, random_state=random_state)
        if unseen not in ("full", "error"):
            raise ValueError(f"unseen must be 'full' or 'error', got {unseen!r}")
        self.unseen: UnseenPolicy = unseen

    def fit(self, probs: ArrayLike, labels: ArrayLike, groups: ArrayLike) -> Self:
        """Compute one conformal quantile per group."""
        scores = self._score_matrix(probs)
        cal = true_label_scores(scores, labels)
        g = _as_groups(groups, cal.size)
        self.n_classes_ = scores.shape[1]
        self.qhat_: dict[Hashable, float] = {}
        self.n_per_group_: dict[Hashable, int] = {}
        for group in dict.fromkeys(g.tolist()):
            mask = g == group
            self.qhat_[group] = conformal_quantile(cal[mask], self.alpha)
            self.n_per_group_[group] = int(mask.sum())
        return self

    def group_quantiles(self, groups: ArrayLike) -> NDArray[np.float64]:
        """Quantile to use for each test point, according to its group."""
        self._check_fitted()
        g = np.asarray(groups, dtype=object).ravel()
        unseen = [x for x in dict.fromkeys(g.tolist()) if x not in self.qhat_]
        if unseen and self.unseen == "error":
            raise KeyError(f"groups without calibration data: {unseen}")
        return np.array([self.qhat_.get(x, math.inf) for x in g], dtype=np.float64)

    def predict_sets(self, probs: ArrayLike, groups: ArrayLike) -> NDArray[np.bool_]:
        """Prediction sets ``{y : s(x, y) <= q_hat_g}`` using each point's group."""
        self._check_fitted()
        scores = self._score_matrix(probs)
        q = self.group_quantiles(_as_groups(groups, scores.shape[0]))
        return scores <= q[:, None]
