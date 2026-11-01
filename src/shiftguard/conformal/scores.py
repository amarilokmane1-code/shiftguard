r"""Nonconformity scores for classification.

A nonconformity score :math:`s(x, y)` is large when label :math:`y` looks atypical for
input :math:`x`. Both scores below are computed from estimated class probabilities
:math:`\hat p(\cdot \mid x)`.

LAC, the least ambiguous set-valued classifier (Sadinle, Lei and Wasserman, 2019):

.. math:: s_{\mathrm{LAC}}(x, y) = 1 - \hat p(y \mid x)

APS, adaptive prediction sets (Romano, Sesia and Candes, 2020). Sort the classes by
decreasing probability; the score is the probability mass ranked strictly above
:math:`y`, plus a (possibly randomised) share of :math:`\hat p(y \mid x)` itself:

.. math::

    s_{\mathrm{APS}}(x, y) = \sum_{j \,:\, \pi_x(j) < \pi_x(y)} \hat p(j \mid x)
                             + u \, \hat p(y \mid x),

where :math:`\pi_x` is the rank of each class and :math:`u \sim \mathrm{U}(0, 1)` for the
randomised score or :math:`u = 1` for the deterministic one. Ties in probability are
broken by class index (stable sort), so the ranking is always well defined.

All functions return a score *matrix* of shape ``(n_samples, n_classes)``: entry
``[i, k]`` is the score of candidate label ``k`` for sample ``i``. The calibration score
of a labelled sample is the entry at its true label (:func:`true_label_scores`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScoreName = Literal["lac", "aps"]
SCORES: tuple[str, ...] = ("lac", "aps")


def validate_probs(probs: ArrayLike, atol: float = 1e-6) -> NDArray[np.float64]:
    """Check that ``probs`` is an ``(n, K)`` matrix of probability vectors.

    Parameters
    ----------
    probs
        Class probabilities, one row per sample.
    atol
        Tolerance on the row sums.

    Returns
    -------
    numpy.ndarray
        ``probs`` as a float64 array.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"probs must have shape (n_samples, n_classes>=2), got {arr.shape}")
    if arr.size == 0:
        return arr
    lo, hi = arr.min(), arr.max()  # NaN propagates, so this also catches non-finite values
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("probs contains non-finite values")
    if lo < -atol or hi > 1 + atol:
        raise ValueError("probs must lie in [0, 1]")
    if np.abs(arr.sum(axis=1) - 1.0).max() > atol:
        raise ValueError("each row of probs must sum to 1")
    return arr


def validate_labels(labels: ArrayLike, n_samples: int, n_classes: int) -> NDArray[np.intp]:
    """Check that ``labels`` are integers in ``[0, n_classes)``, one per sample."""
    arr = np.asarray(labels)
    if arr.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},), got {arr.shape}")
    if (
        arr.size
        and not np.issubdtype(arr.dtype, np.integer)
        and not np.all(np.equal(np.mod(arr, 1), 0))
    ):
        raise ValueError("labels must be integer class indices")
    arr = arr.astype(np.intp)
    if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
        raise ValueError(f"labels must be in [0, {n_classes})")
    return arr


def lac_score_matrix(probs: ArrayLike) -> NDArray[np.float64]:
    """LAC scores ``1 - p_hat(k | x_i)`` for every sample ``i`` and class ``k``."""
    return 1.0 - validate_probs(probs)


def aps_score_matrix(probs: ArrayLike, u: ArrayLike | None = None) -> NDArray[np.float64]:
    """APS scores for every sample and candidate class.

    Parameters
    ----------
    probs
        Class probabilities of shape ``(n, K)``.
    u
        Optional uniform draws of shape ``(n,)`` for the randomised score (one draw per
        sample, shared by all candidate labels). ``None`` gives the deterministic score.

    Returns
    -------
    numpy.ndarray
        Score matrix of shape ``(n, K)`` with values in ``[0, 1]``.
    """
    p = validate_probs(probs)
    order = np.argsort(-p, axis=1, kind="stable")
    p_sorted = np.take_along_axis(p, order, axis=1)
    mass_above = np.cumsum(p_sorted, axis=1) - p_sorted
    if u is None:
        own = p_sorted
    else:
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.shape != (p.shape[0],):
            raise ValueError(f"u must have shape ({p.shape[0]},), got {u_arr.shape}")
        own = u_arr[:, None] * p_sorted
    scores = np.empty_like(p)
    np.put_along_axis(scores, order, mass_above + own, axis=1)
    return np.clip(scores, 0.0, 1.0)


def score_matrix(
    probs: ArrayLike,
    score: ScoreName = "lac",
    *,
    randomized: bool = False,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Dispatch to the requested score.

    Parameters
    ----------
    probs
        Class probabilities of shape ``(n, K)``.
    score
        ``"lac"`` or ``"aps"``.
    randomized
        Use the randomised APS score (ignored for LAC, which is already continuous).
    rng
        Generator for the APS uniform draws (required when ``randomized`` is true).
    """
    if score == "lac":
        return lac_score_matrix(probs)
    if score == "aps":
        if not randomized:
            return aps_score_matrix(probs)
        if rng is None:
            raise ValueError("randomized APS needs a numpy Generator `rng`")
        n = np.asarray(probs).shape[0]
        return aps_score_matrix(probs, u=rng.uniform(size=n))
    raise ValueError(f"unknown score {score!r}; expected one of {SCORES}")


def true_label_scores(scores: NDArray[np.float64], labels: ArrayLike) -> NDArray[np.float64]:
    """Pick, for every row of a score matrix, the score of the true label."""
    y = validate_labels(labels, scores.shape[0], scores.shape[1])
    return scores[np.arange(scores.shape[0]), y]
