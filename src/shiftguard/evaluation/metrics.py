"""Discrimination and calibration metrics for binary predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from shiftguard.models.calibration import expected_calibration_error


def binary_metrics(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> dict[str, Any]:
    """AUC, balanced accuracy (threshold 0.5), ECE, Brier score and log loss.

    ``cal_in_large`` is calibration-in-the-large, mean predicted P(ASD) minus the observed
    ASD fraction (positive = ASD over-predicted). AUC is NaN when only one class is present.
    """
    yy = np.asarray(y).astype(int)
    pp = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    both = len(np.unique(yy)) == 2
    return {
        "n": int(yy.size),
        "prevalence": float(yy.mean()) if yy.size else float("nan"),
        "auc": float(roc_auc_score(yy, pp)) if both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(yy, (pp >= 0.5).astype(int)))
        if both
        else float("nan"),
        "ece": expected_calibration_error(yy, pp, n_bins),
        "cal_in_large": float(pp.mean() - yy.mean()) if yy.size else float("nan"),
        "brier": float(brier_score_loss(yy, pp)),
        "log_loss": float(log_loss(yy, pp, labels=[0, 1])),
    }
