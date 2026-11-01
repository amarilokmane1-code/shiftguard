"""Loading, describing and caching ROI time series."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shiftguard.data.download import AbideSpec, timeseries_path
from shiftguard.utils.log import get_logger

log = get_logger(__name__)

TimeSeries = NDArray[np.float64]


def read_1d(path: Path) -> TimeSeries:
    """Read one PCP ``.1D`` file as a ``(n_timepoints, n_rois)`` array (``#`` lines skipped)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an empty file must raise, not warn
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.size == 0:
        raise ValueError(f"empty time series: {path}")
    return arr


def load_timeseries(
    data_dir: Path, spec: AbideSpec, file_ids: Iterable[str]
) -> dict[str, TimeSeries | None]:
    """Load the time series of ``file_ids``; missing or unreadable files map to ``None``."""
    out: dict[str, TimeSeries | None] = {}
    for fid in file_ids:
        path = timeseries_path(data_dir, spec, fid)
        try:
            out[fid] = read_1d(path)
        except (OSError, ValueError, UserWarning) as exc:
            log.warning("cannot read time series %s: %s", fid, exc)
            out[fid] = None
    return out


def timeseries_info(ts: Mapping[str, TimeSeries | None], atol: float = 1e-8) -> pd.DataFrame:
    """One row per subject: readability, length, number of ROIs, zero-variance ROIs.

    A zero-variance ROI (constant signal, usually an ROI outside the subject's
    functional field of view) has no defined correlation with any other ROI.
    """
    rows = []
    for fid, arr in ts.items():
        if arr is None:
            rows.append({"FILE_ID": fid, "readable": False})
            continue
        std = arr.std(axis=0)
        rows.append(
            {
                "FILE_ID": fid,
                "readable": True,
                "n_timepoints": arr.shape[0],
                "n_rois": arr.shape[1],
                "n_zero_variance_rois": int((std <= atol).sum()),
                "has_nonfinite": bool(~np.isfinite(arr).all()),
            }
        )
    info = pd.DataFrame(
        rows,
        columns=[
            "FILE_ID",
            "readable",
            "n_timepoints",
            "n_rois",
            "n_zero_variance_rois",
            "has_nonfinite",
        ],
    )
    return info


def save_timeseries(path: Path, ts: Mapping[str, TimeSeries]) -> Path:
    """Cache time series (keyed by ``FILE_ID``) in one compressed ``.npz`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in ts.items()})
    return path


def read_timeseries_cache(
    path: Path, file_ids: Iterable[str] | None = None
) -> dict[str, TimeSeries]:
    """Read a cache written by :func:`save_timeseries` (optionally a subset, in order)."""
    with np.load(path) as npz:
        keys = list(file_ids) if file_ids is not None else list(npz.files)
        return {k: npz[k].astype(np.float64) for k in keys}
