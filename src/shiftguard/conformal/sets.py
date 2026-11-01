"""Prediction-set utilities and abstention.

A prediction set is stored as a boolean matrix of shape ``(n_samples, n_classes)``.
A case is **abstained** when its set is not a single class, i.e. it is empty or it
contains several classes (for binary ASD/control: both classes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from shiftguard.conformal.scores import validate_labels

NO_DECISION = -1


def _as_sets(sets: ArrayLike) -> NDArray[np.bool_]:
    arr = np.asarray(sets)
    if arr.ndim != 2 or arr.dtype != np.bool_:
        raise ValueError("sets must be a boolean matrix of shape (n_samples, n_classes)")
    return arr


def covered(sets: ArrayLike, labels: ArrayLike) -> NDArray[np.bool_]:
    """Whether each true label lies in its prediction set."""
    s = _as_sets(sets)
    y = validate_labels(labels, s.shape[0], s.shape[1])
    return s[np.arange(s.shape[0]), y]


def set_sizes(sets: ArrayLike) -> NDArray[np.intp]:
    """Return the number of labels in each prediction set."""
    return _as_sets(sets).sum(axis=1)


def abstained(sets: ArrayLike) -> NDArray[np.bool_]:
    """Flag sets that are empty or contain more than one class (abstentions)."""
    return set_sizes(sets) != 1


def decisions(sets: ArrayLike) -> NDArray[np.intp]:
    """Return the predicted class for singleton sets, ``NO_DECISION`` (-1) otherwise."""
    s = _as_sets(sets)
    out = np.full(s.shape[0], NO_DECISION, dtype=np.intp)
    single = s.sum(axis=1) == 1
    out[single] = np.argmax(s[single], axis=1)
    return out


def summarize_sets(sets: ArrayLike, labels: ArrayLike) -> dict[str, Any]:
    """Coverage, set size and abstention statistics for one batch of sets.

    Returns
    -------
    dict
        ``n``, ``n_covered``, ``coverage``, ``mean_set_size``, ``empty_rate``,
        ``full_rate``, ``abstention_rate``, ``n_decided`` and ``accuracy_decided``
        (accuracy on non-abstained cases; NaN when every case is abstained).
    """
    s = _as_sets(sets)
    y = validate_labels(labels, s.shape[0], s.shape[1])
    n = s.shape[0]
    sizes = s.sum(axis=1)
    cov = s[np.arange(n), y]
    dec = decisions(s)
    decided = dec != NO_DECISION
    n_decided = int(decided.sum())

    def _rate(mask: NDArray[np.bool_]) -> float:
        return float(mask.mean()) if n else float("nan")

    return {
        "n": n,
        "n_covered": int(cov.sum()),
        "coverage": _rate(cov),
        "mean_set_size": float(sizes.mean()) if n else float("nan"),
        "empty_rate": _rate(sizes == 0),
        "full_rate": _rate(sizes == s.shape[1]),
        "abstention_rate": _rate(~decided),
        "n_decided": n_decided,
        "accuracy_decided": float((dec[decided] == y[decided]).mean())
        if n_decided
        else float("nan"),
    }
