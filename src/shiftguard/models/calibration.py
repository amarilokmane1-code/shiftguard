r"""Post-hoc probability calibration and calibration metrics (binary).

Calibrators act on the model's logit :math:`z` and are fitted on the **calibration split
only**:

- **Temperature scaling** (Guo et al., 2017): :math:`\hat p = \sigma(z / T)`, with
  :math:`T > 0` minimising the negative log-likelihood. It is a monotone transform of the
  margin, so ranking metrics (AUC) and binary LAC conformal sets are unchanged.
- **Platt scaling** (Platt, 1999): :math:`\hat p = \sigma(a z + b)`.

Expected calibration error (positive class, equal-width bins :math:`B_1..B_M` on
:math:`[0, 1]`):

.. math:: \mathrm{ECE} = \sum_{m=1}^M \frac{|B_m|}{n}\,
          \big|\, \bar y_{B_m} - \bar p_{B_m} \big|,

where :math:`\bar y_{B_m}` is the observed ASD frequency and :math:`\bar p_{B_m}` the mean
predicted ASD probability in bin :math:`m`.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

CALIBRATORS: tuple[str, ...] = ("temperature", "platt", "none")


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -z))


def _nll(z: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Mean binary negative log-likelihood of logits ``z``."""
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


class TemperatureScaler:
    """Fit a single temperature ``T`` on calibration logits."""

    def fit(self, logits: ArrayLike, y: ArrayLike) -> Self:
        """Minimise the NLL over ``log T`` in ``[-5, 5]``."""
        z = np.asarray(logits, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        res = minimize_scalar(
            lambda log_t: _nll(z / np.exp(log_t), yy), bounds=(-5.0, 5.0), method="bounded"
        )
        self.temperature_ = float(np.exp(res.x))
        return self

    def predict_proba(self, logits: ArrayLike) -> NDArray[np.float64]:
        """Calibrated ``P(ASD)``."""
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature_)


class PlattScaler:
    """Fit ``sigma(a z + b)`` on calibration logits (unpenalised)."""

    def fit(self, logits: ArrayLike, y: ArrayLike) -> Self:
        """Fit the two Platt parameters."""
        z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y).astype(int))
        return self

    def predict_proba(self, logits: ArrayLike) -> NDArray[np.float64]:
        """Calibrated ``P(ASD)``."""
        z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        return self.model_.predict_proba(z)[:, 1]


class IdentityCalibrator:
    """No calibration: ``sigma(z)``."""

    def fit(self, logits: ArrayLike, y: ArrayLike) -> Self:
        """Nothing to fit."""
        return self

    def predict_proba(self, logits: ArrayLike) -> NDArray[np.float64]:
        """Uncalibrated ``P(ASD)``."""
        return sigmoid(logits)


def make_calibrator(method: str) -> TemperatureScaler | PlattScaler | IdentityCalibrator:
    """Return an unfitted calibrator by name."""
    if method == "temperature":
        return TemperatureScaler()
    if method == "platt":
        return PlattScaler()
    if method == "none":
        return IdentityCalibrator()
    raise ValueError(f"calibration method must be one of {CALIBRATORS}, got {method!r}")


def _bins(p: NDArray[np.float64], n_bins: int) -> NDArray[np.intp]:
    return np.minimum((p * n_bins).astype(int), n_bins - 1)


def expected_calibration_error(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> float:
    """ECE of the positive-class probability with equal-width bins."""
    yy = np.asarray(y, dtype=np.float64)
    pp = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    if yy.size == 0:
        return float("nan")
    b = _bins(pp, n_bins)
    ece = 0.0
    for k in np.unique(b):
        m = b == k
        ece += m.mean() * abs(yy[m].mean() - pp[m].mean())
    return float(ece)


def null_ece(p: ArrayLike, n_bins: int = 10, n_draws: int = 200, seed: int = 0) -> float:
    """Return the expected ECE of a *perfectly calibrated* model with these probabilities.

    Labels are drawn as ``y ~ Bernoulli(p)`` ``n_draws`` times and the ECE is averaged.
    Binned ECE is biased upward in small samples, so an observed ECE is only evidence of
    miscalibration when it clearly exceeds this size-matched reference.
    """
    pp = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    if pp.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.uniform(size=(n_draws, pp.size)) < pp
    return float(np.mean([expected_calibration_error(d, pp, n_bins) for d in draws]))


def reliability_table(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin mean predicted probability, observed frequency and count (non-empty bins)."""
    yy = np.asarray(y, dtype=np.float64)
    pp = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    b = _bins(pp, n_bins)
    rows = [
        {
            "bin": int(k),
            "mean_pred": pp[b == k].mean(),
            "frac_pos": yy[b == k].mean(),
            "count": int((b == k).sum()),
        }
        for k in np.unique(b)
    ]
    return pd.DataFrame(rows, columns=["bin", "mean_pred", "frac_pos", "count"])
