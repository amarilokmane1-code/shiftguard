r"""Simulated multi-site cohort for fast, download-free end-to-end runs (CI).

Each site ``s`` gets a random location offset :math:`\mu_s \sim N(0, \tau^2 I)` (the
"site effect", ``site_shift_scale = tau``) and its own case prevalence in [0.35, 0.65].
Features are :math:`x = \mu_s + \delta\, y\, w + \varepsilon`, with a fixed unit
signal direction :math:`w`, ``signal_strength = delta`` and :math:`\varepsilon \sim N(0, I)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shiftguard.data.dataset import Dataset


def make_synthetic_dataset(data_cfg: dict[str, Any], seed: int = 0) -> Dataset:
    """Generate a :class:`~shiftguard.data.dataset.Dataset` from the ``data`` config."""
    rng = np.random.default_rng(seed)
    n_sites = int(data_cfg["n_sites"])
    n_per = int(data_cfg["n_subjects_per_site"])
    p = int(data_cfg["n_features"])
    w = rng.normal(size=p)
    w /= np.linalg.norm(w)
    rows, blocks = [], []
    sub = 0
    for s in range(n_sites):
        mu = rng.normal(scale=data_cfg["site_shift_scale"], size=p)
        prev = rng.uniform(0.35, 0.65)
        y = (rng.uniform(size=n_per) < prev).astype(int)
        y[:2], y[2:4] = 1, 0  # both classes present at every site
        X = mu + data_cfg["signal_strength"] * y[:, None] * w + rng.normal(size=(n_per, p))
        blocks.append(X)
        for yi in y:
            sub += 1
            rows.append(
                {
                    "SUB_ID": sub,
                    "FILE_ID": f"SYN_{sub:05d}",
                    "SITE_ID": f"SITE_{s:02d}",
                    "DX_GROUP": 1 if yi == 1 else 2,
                    "label": int(yi),
                    "diagnosis": "ASD" if yi == 1 else "TDC",
                    "AGE_AT_SCAN": float(rng.uniform(8, 30)),
                    "SEX": int(rng.uniform() < 0.2) + 1,
                }
            )
    return Dataset(pd.DataFrame(rows), [], np.vstack(blocks), "correlation")


def load_any_dataset(config: dict[str, Any]) -> Dataset:
    """ABIDE (from the ``features`` cache) or the synthetic cohort, per ``data.source``.

    ``features.harmonization: combat`` applies label-free ComBat to the loaded features.
    """
    from shiftguard.features.harmonize import harmonize_dataset

    if config["data"].get("source") == "synthetic":
        ds = make_synthetic_dataset(config["data"], seed=config["experiment"]["base_seed"])
    else:
        from shiftguard.data.dataset import load_dataset

        ds = load_dataset(config)
    return harmonize_dataset(ds, config)
