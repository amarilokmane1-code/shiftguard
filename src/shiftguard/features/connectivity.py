r"""Functional-connectivity features from ROI time series.

For each subject, a connectivity matrix is estimated with
:class:`nilearn.connectome.ConnectivityMeasure` (Ledoit-Wolf covariance), and its strict
upper triangle is vectorised: :math:`R (R - 1) / 2` features for :math:`R` ROIs (19,900
for CC200).

- ``correlation`` and ``partial correlation`` are **subject-wise**: each subject's
  features depend only on its own time series. Values are Fisher z-transformed,
  :math:`z = \operatorname{arctanh}(r)`, after clipping :math:`|r| \le 1 - 10^{-7}`.
- ``tangent`` projects each covariance onto the tangent space at a group reference
  (geometric mean of covariances; Varoquaux et al., 2010). **The reference depends on
  the subjects used to fit it**, so it must be fitted on training subjects only and is
  computed inside each fold. :func:`subjectwise_features` refuses to precompute it, and
  :func:`fit_transform_fold` checks that the fitting set is disjoint from evaluation sets.
  Fisher z does not apply to tangent coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

KINDS: tuple[str, ...] = ("correlation", "partial correlation", "tangent")
SUBJECTWISE_KINDS: tuple[str, ...] = ("correlation", "partial correlation")
_R_CLIP = 1.0 - 1e-7


def n_edges(n_rois: int) -> int:
    """Return the number of strict upper-triangle entries of an ``n_rois x n_rois`` matrix."""
    return n_rois * (n_rois - 1) // 2


def fisher_z(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fisher z-transform ``arctanh(r)`` with clipping to keep values finite."""
    return np.arctanh(np.clip(r, -_R_CLIP, _R_CLIP))


class ConnectivityFeatures(BaseEstimator, TransformerMixin):
    """Vectorised connectivity, as a scikit-learn transformer over lists of time series.

    Parameters
    ----------
    kind
        ``"correlation"``, ``"partial correlation"`` or ``"tangent"``.
    fisher_z
        Apply the Fisher z-transform (correlation kinds only).
    cov_estimator
        ``"ledoit_wolf"`` (default, well conditioned) or ``"empirical"``.

    Attributes
    ----------
    n_fit_subjects_ : int
        Number of subjects the transformer was fitted on (the tangent reference set).
    n_rois_ : int
        Number of ROIs.
    """

    def __init__(
        self, kind: str = "correlation", fisher_z: bool = True, cov_estimator: str = "ledoit_wolf"
    ) -> None:
        self.kind = kind
        self.fisher_z = fisher_z
        self.cov_estimator = cov_estimator

    def _measure(self):
        from nilearn.connectome import ConnectivityMeasure

        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        estimators = {"ledoit_wolf": LedoitWolf, "empirical": EmpiricalCovariance}
        if self.cov_estimator not in estimators:
            raise ValueError(f"cov_estimator must be one of {tuple(estimators)}")
        return ConnectivityMeasure(
            cov_estimator=estimators[self.cov_estimator](store_precision=False),
            kind=self.kind,
            vectorize=True,
            discard_diagonal=True,
            standardize=False,
        )

    def fit(self, time_series: Sequence[NDArray[np.float64]], y: object = None) -> Self:
        """Fit on training subjects (for ``tangent``: estimate the group reference)."""
        ts = _check_list(time_series)
        self.measure_ = self._measure().fit(ts)
        self.n_fit_subjects_ = len(ts)
        self.n_rois_ = ts[0].shape[1]
        return self

    def transform(self, time_series: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Features of shape ``(n_subjects, n_rois * (n_rois - 1) / 2)``."""
        if not hasattr(self, "measure_"):
            raise RuntimeError("ConnectivityFeatures is not fitted; call fit() first")
        ts = _check_list(time_series)
        if ts[0].shape[1] != self.n_rois_:
            raise ValueError(f"fitted with {self.n_rois_} ROIs, got {ts[0].shape[1]}")
        if self.kind == "tangent" and len(ts) == 1:
            # nilearn refuses one-subject groups in tangent space, but once the reference is
            # fitted the projection is per subject: transform a duplicated pair, keep one row.
            X = np.asarray(self.measure_.transform([ts[0], ts[0]]), dtype=np.float64)[:1]
        else:
            X = np.asarray(self.measure_.transform(ts), dtype=np.float64)
        if self.fisher_z and self.kind in SUBJECTWISE_KINDS:
            X = fisher_z(X)
        if not np.all(np.isfinite(X)):
            raise ValueError("non-finite connectivity features (check zero-variance ROIs)")
        return X


def _check_list(time_series: Sequence[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    ts = [np.asarray(t, dtype=np.float64) for t in time_series]
    if not ts:
        raise ValueError("no time series given")
    n_rois = {t.shape[1] for t in ts if t.ndim == 2}
    if any(t.ndim != 2 for t in ts) or len(n_rois) != 1:
        raise ValueError("time series must be 2-D arrays with the same number of ROIs")
    return ts


def subjectwise_features(
    time_series: Sequence[NDArray[np.float64]],
    kind: str = "correlation",
    fisher_z: bool = True,
    cov_estimator: str = "ledoit_wolf",
) -> NDArray[np.float64]:
    """Precompute subject-wise features (correlation kinds only; no cross-subject fitting).

    Raises
    ------
    ValueError
        For ``kind="tangent"``, whose reference must be fitted inside each fold.
    """
    if kind not in SUBJECTWISE_KINDS:
        raise ValueError(
            f"{kind!r} features depend on the training subjects (group reference); "
            "compute them inside each fold with fit_transform_fold()"
        )
    transformer = ConnectivityFeatures(kind=kind, fisher_z=fisher_z, cov_estimator=cov_estimator)
    return transformer.fit(time_series).transform(time_series)


def fit_transform_fold(
    time_series: Sequence[NDArray[np.float64]],
    fit_idx: NDArray[np.intp],
    eval_idx: Sequence[NDArray[np.intp]],
    *,
    kind: str = "correlation",
    fisher_z: bool = True,
    cov_estimator: str = "ledoit_wolf",
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Fit the connectivity transformer on ``fit_idx`` only and transform every split.

    Parameters
    ----------
    time_series
        All subjects' time series.
    fit_idx
        Indices of the training subjects (the only ones used to fit the reference).
    eval_idx
        Index arrays of the other splits (calibration, test, ...). Each must be disjoint
        from ``fit_idx``.

    Returns
    -------
    tuple
        Training features and a list of features for each evaluation split.
    """
    fit_set = set(np.asarray(fit_idx).tolist())
    for i, idx in enumerate(eval_idx):
        overlap = fit_set.intersection(np.asarray(idx).tolist())
        if overlap:
            raise ValueError(
                f"evaluation split {i} overlaps the fitting split: {sorted(overlap)[:5]}"
            )
    ts = list(time_series)
    transformer = ConnectivityFeatures(kind=kind, fisher_z=fisher_z, cov_estimator=cov_estimator)
    transformer.fit([ts[i] for i in fit_idx])
    X_fit = transformer.transform([ts[i] for i in fit_idx])
    return X_fit, [transformer.transform([ts[i] for i in idx]) for idx in eval_idx]
