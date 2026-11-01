"""ABIDE I phenotypes: loading, cleaning and label encoding.

Label convention used everywhere in the package: ``label = 1`` for autism spectrum
disorder (``DX_GROUP == 1``) and ``label = 0`` for typical controls (``DX_GROUP == 2``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ASD_CODE, TDC_CODE = 1, 2
LABEL_NAMES = {1: "ASD", 0: "TDC"}
MISSING_SENTINEL = -9999

QC_COLUMNS: tuple[str, ...] = (
    "qc_rater_1",
    "qc_anat_rater_2",
    "qc_func_rater_2",
    "qc_anat_rater_3",
    "qc_func_rater_3",
)
REQUIRED_COLUMNS: tuple[str, ...] = ("SUB_ID", "FILE_ID", "SITE_ID", "DX_GROUP")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "AGE_AT_SCAN",
    "SEX",
    "FIQ",
    "EYE_STATUS_AT_SCAN",
    "func_mean_fd",
    *QC_COLUMNS,
)


def load_phenotypes(path: str | Path, extra_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Read the PCP phenotypic CSV and return a tidy table, one row per subject.

    The ``-9999`` sentinel becomes NaN; ``label`` (1 = ASD, 0 = TDC) and ``diagnosis``
    ("ASD"/"TDC") are added; rows are sorted by ``SUB_ID``.

    Parameters
    ----------
    path
        Path to ``Phenotypic_V1_0b_preprocessed1.csv``.
    extra_columns
        Further columns to keep (e.g. from the config's ``data.phenotypes``).
    """
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"phenotype file lacks required columns: {missing}")
    keep = list(dict.fromkeys([*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, *extra_columns]))
    df = raw[[c for c in keep if c in raw.columns]].copy()

    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].mask(df[numeric] == MISSING_SENTINEL)

    bad_dx = ~df["DX_GROUP"].isin([ASD_CODE, TDC_CODE])
    if bad_dx.any():
        raise ValueError(
            f"unexpected DX_GROUP values: {sorted(df.loc[bad_dx, 'DX_GROUP'].unique())}"
        )
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["FILE_ID"] = df["FILE_ID"].astype(str)
    df["SITE_ID"] = df["SITE_ID"].astype(str)
    df["label"] = (df["DX_GROUP"] == ASD_CODE).astype(int)
    df["diagnosis"] = np.where(df["label"] == 1, "ASD", "TDC")
    if df["SUB_ID"].duplicated().any():
        raise ValueError("duplicate SUB_ID in phenotype file")
    return df.sort_values("SUB_ID").reset_index(drop=True)


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-site composition (n, % ASD, % male, age median and range, FIQ median), plus ALL.

    Sites differ in age range and sex ratio as well as in scanner; this table documents
    those differences for the data card.
    """

    def _summ(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(g),
                "pct_ASD": 100.0 * g["label"].mean(),
                "pct_male": 100.0 * (g["SEX"] == 1).mean(),
                "age_median": g["AGE_AT_SCAN"].median(),
                "age_min": g["AGE_AT_SCAN"].min(),
                "age_max": g["AGE_AT_SCAN"].max(),
                "FIQ_median": g["FIQ"].median() if "FIQ" in g else np.nan,
                "FIQ_missing": int(g["FIQ"].isna().sum()) if "FIQ" in g else len(g),
            }
        )

    rows = {site: _summ(g) for site, g in df.groupby("SITE_ID")}
    rows["ALL"] = _summ(df)
    out = pd.DataFrame(rows).T
    out.index.name = "SITE_ID"
    return out.astype({"n": int, "FIQ_missing": int}).reset_index()


def counts_by_site(df: pd.DataFrame, prefix: str = "n") -> pd.DataFrame:
    """Subjects per site: ``<prefix>`` (total), ``<prefix>_ASD`` and ``<prefix>_TDC``."""
    table = pd.crosstab(df["SITE_ID"], df["diagnosis"]).reindex(
        columns=["ASD", "TDC"], fill_value=0
    )
    table.columns = [f"{prefix}_{c}" for c in table.columns]
    table.insert(0, prefix, table.sum(axis=1))
    return table
