r"""Prior (label) shift: estimate the target class prior without labels and adjust.

Under pure label shift the class-conditional feature distributions are shared,
:math:`p_t(x \mid y) = p_s(x \mid y)`, and only the class prior changes,
:math:`\pi_t \ne \pi_s`. Bayes' rule then gives the target posterior from the source one,

.. math::

    p_t(y \mid x) = \frac{\tfrac{\pi_t(y)}{\pi_s(y)}\, p_s(y \mid x)}
                         {\sum_k \tfrac{\pi_t(k)}{\pi_s(k)}\, p_s(k \mid x)} ,

and the EM algorithm of Saerens, Latinne and Decaestecker (2002) estimates
:math:`\pi_t` from unlabelled target data by alternating this adjustment (E step) with
:math:`\pi_t(y) \leftarrow \tfrac{1}{m}\sum_j p_t(y \mid x_j)` (M step). It needs
*calibrated* source posteriors, which is why ShiftGuard applies it to the
temperature-scaled probabilities. If the shift is not pure label shift (for example a
site-wide shift of the features), the EM estimate absorbs part of that shift as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shiftguard.conformal.scores import validate_probs


def _check_prior(prior: ArrayLike, n_classes: int, name: str) -> NDArray[np.float64]:
    p = np.asarray(prior, dtype=np.float64).ravel()
    if p.shape != (n_classes,) or np.any(p <= 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError(f"{name} must be {n_classes} positive values summing to 1")
    return p


def adjust_to_prior(
    probs: ArrayLike, source_prior: ArrayLike, target_prior: ArrayLike
) -> NDArray[np.float64]:
    """Reweight source posteriors to a new class prior (Bayes' rule above)."""
    p = validate_probs(probs)
    ps = _check_prior(source_prior, p.shape[1], "source_prior")
    pt = _check_prior(target_prior, p.shape[1], "target_prior")
    q = p * (pt / ps)
    return q / q.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class PriorShiftEstimate:
    """Result of :func:`em_prior_shift`."""

    target_prior: NDArray[np.float64]
    adjusted_probs: NDArray[np.float64]
    n_iter: int
    converged: bool


def em_prior_shift(
    probs_target: ArrayLike,
    source_prior: ArrayLike,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
    min_prior: float = 1e-3,
) -> PriorShiftEstimate:
    """Estimate the target class prior by EM (Saerens et al., 2002).

    Parameters
    ----------
    probs_target
        Calibrated source-model posteriors for the **unlabelled** target samples.
    source_prior
        Class prior of the data the posteriors are calibrated to.
    max_iter, tol
        Stop when the prior changes by less than ``tol`` (max norm).
    min_prior
        Each class prior is kept at least ``min_prior`` (then renormalised).
    """
    p = validate_probs(probs_target)
    ps = _check_prior(source_prior, p.shape[1], "source_prior")
    pt = ps.copy()
    converged = False
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        q = p * (pt / ps)
        q /= q.sum(axis=1, keepdims=True)
        new = np.maximum(q.mean(axis=0), min_prior)
        new /= new.sum()
        if np.max(np.abs(new - pt)) < tol:
            pt, converged = new, True
            break
        pt = new
    return PriorShiftEstimate(pt, adjust_to_prior(p, ps, pt), n_iter, converged)


def label_shift_weights(source_prior: ArrayLike, target_prior: ArrayLike) -> NDArray[np.float64]:
    """Per-class importance weights ``w(y) = pi_t(y) / pi_s(y)``."""
    ps = np.asarray(source_prior, dtype=np.float64).ravel()
    pt = _check_prior(target_prior, ps.size, "target_prior")
    ps = _check_prior(ps, ps.size, "source_prior")
    return pt / ps
