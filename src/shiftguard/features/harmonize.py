r"""ComBat harmonisation of connectivity features (neuroHarmonize).

ComBat (Johnson et al., 2007; Fortin et al., 2018) models every feature as
:math:`x_{ij} = \alpha + Z_i\beta + \gamma_{s(i)} + \delta_{s(i)}\varepsilon_{ij}`, with an
additive (:math:`\gamma_s`) and a multiplicative (:math:`\delta_s`) effect per site and
feature, shrunk towards a common prior by empirical Bayes, and removes the site terms.

Under leave-one-site-out, the deployment site's parameters can only be estimated from that
site's own features. ShiftGuard therefore fits ComBat on **all subjects without labels**:
the covariates are age and sex only, and the diagnosis is never used (it would have to be
known at the deployment site). Because no label enters the fit and every subject belongs to
the training, calibration or deployment part of every fold, this single fit is identical to
a per-fold transductive fit. The price of leaving the diagnosis out is that site
differences in case/control ratio are partly removed as "site effect" (documented in
docs/LIMITATIONS.md).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from shiftguard.data.dataset import Dataset

FORBIDDEN_COVARIATES = ("label", "DX_GROUP", "diagnosis")


def combat_harmonize(
    features: NDArray[np.float64],
    sites: pd.Series | NDArray[np.object_],
    covariates: pd.DataFrame,
) -> NDArray[np.float64]:
    """Return ComBat-harmonised features (empirical Bayes, parametric priors).

    Parameters
    ----------
    features
        ``(n_subjects, n_features)`` matrix.
    sites
        Site of each subject.
    covariates
        Numeric covariates whose effects are preserved (e.g. age, sex). Must not contain
        the diagnosis.
    """
    from neuroHarmonize import harmonizationLearn

    bad = [c for c in covariates.columns if c in FORBIDDEN_COVARIATES]
    if bad:
        raise ValueError(f"diagnosis columns must never be ComBat covariates: {bad}")
    X = np.asarray(features, dtype=np.float64)
    covars = covariates.reset_index(drop=True).astype(float)
    if covars.isna().any().any():
        raise ValueError("ComBat covariates contain missing values")
    covars.insert(0, "SITE", np.asarray(sites, dtype=str))
    if len(covars) != len(X):
        raise ValueError("features and covariates differ in length")
    with contextlib.redirect_stdout(io.StringIO()):  # neuroHarmonize prints progress
        _, X_adj = harmonizationLearn(X, covars, eb=True)
    return np.asarray(X_adj, dtype=np.float64)


def harmonize_dataset(ds: Dataset, config: dict[str, Any]) -> Dataset:
    """Apply ``features.harmonization`` (``none`` or ``combat``) to a loaded dataset."""
    method = config.get("features", {}).get("harmonization", "none")
    if method == "none":
        return ds
    if method != "combat":
        raise ValueError(f"unknown harmonization {method!r}; expected 'none' or 'combat'")
    if ds.features is None:
        raise ValueError("ComBat needs subject-wise features (not the tangent space)")
    cov_names = list(config["harmonization"]["covariates"])
    X = combat_harmonize(ds.features, ds.sites, ds.phenotypes[cov_names])
    return replace(ds, features=X, feature_kind=f"{ds.feature_kind}+combat")
