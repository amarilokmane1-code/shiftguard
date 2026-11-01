"""Shared machinery for the conformal predictors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shiftguard.conformal.quantiles import check_alpha
from shiftguard.conformal.scores import SCORES, ScoreName, score_matrix


class BaseConformal:
    """Base class: stores the target level and the nonconformity score.

    Parameters
    ----------
    alpha
        Miscoverage level; the target coverage is ``1 - alpha``.
    score
        ``"lac"`` or ``"aps"`` (see :mod:`shiftguard.conformal.scores`).
    randomized
        Use the randomised APS score. Ignored for LAC.
    random_state
        Seed for the APS uniform draws.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score: ScoreName = "lac",
        *,
        randomized: bool = False,
        random_state: int | None = None,
    ) -> None:
        if score not in SCORES:
            raise ValueError(f"unknown score {score!r}; expected one of {SCORES}")
        self.alpha = check_alpha(alpha)
        self.score: ScoreName = score
        self.randomized = randomized
        self.random_state = random_state
        self._rng: np.random.Generator | None = None
        self.n_classes_: int | None = None

    @property
    def target_coverage(self) -> float:
        """The nominal coverage ``1 - alpha``."""
        return 1.0 - self.alpha

    def _score_matrix(self, probs: ArrayLike) -> NDArray[np.float64]:
        """Score matrix for ``probs``, checking the number of classes once fitted."""
        if self.randomized and self._rng is None:  # created lazily: only randomised APS needs it
            self._rng = np.random.default_rng(self.random_state)
        scores = score_matrix(probs, self.score, randomized=self.randomized, rng=self._rng)
        if self.n_classes_ is not None and scores.shape[1] != self.n_classes_:
            raise ValueError(
                f"fitted with {self.n_classes_} classes, got probs with {scores.shape[1]}"
            )
        return scores

    def _check_fitted(self) -> None:
        if self.n_classes_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
