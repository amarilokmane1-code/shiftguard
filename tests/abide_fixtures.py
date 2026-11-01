"""A tiny fake ABIDE tree (phenotype CSV + .1D files) for offline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shiftguard.data.download import AbideSpec, phenotype_path, timeseries_path

QC_OK = {
    "qc_rater_1": "OK",
    "qc_anat_rater_2": "OK",
    "qc_func_rater_2": "OK",
    "qc_anat_rater_3": "OK",
    "qc_func_rater_3": "OK",
}


def make_phenotypes(sites: dict[str, tuple[int, int]], start_id: int = 50000) -> pd.DataFrame:
    """Phenotype rows: ``sites`` maps SITE_ID -> (n_ASD, n_TDC). All pass QC."""
    rows, sub = [], start_id
    for site, (n_asd, n_tdc) in sites.items():
        for dx, n in ((1, n_asd), (2, n_tdc)):
            for _ in range(n):
                sub += 1
                rows.append(
                    {
                        "SUB_ID": sub,
                        "FILE_ID": f"{site}_{sub:07d}",
                        "SITE_ID": site,
                        "DX_GROUP": dx,
                        "AGE_AT_SCAN": 10.0 + (sub % 20),
                        "SEX": 1 + (sub % 4 == 0),
                        "FIQ": 100.0 if sub % 5 else -9999,
                        "func_mean_fd": 0.1,
                        **QC_OK,
                    }
                )
    return pd.DataFrame(rows)


def write_fake_abide(
    data_dir: Path,
    pheno: pd.DataFrame,
    spec: AbideSpec,
    *,
    n_timepoints: int = 60,
    n_rois: int = 8,
    seed: int = 0,
    skip: tuple[str, ...] = (),
) -> None:
    """Write the phenotype CSV and one random .1D file per subject (except ``skip``)."""
    rng = np.random.default_rng(seed)
    path = phenotype_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pheno.to_csv(path, index=False)
    for fid in pheno["FILE_ID"]:
        if fid == "no_filename" or fid in skip:
            continue
        ts_path = timeseries_path(data_dir, spec, fid)
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        header = "\t".join(f"#{i + 1}" for i in range(n_rois))
        np.savetxt(ts_path, rng.normal(size=(n_timepoints, n_rois)), header=header, comments="")
