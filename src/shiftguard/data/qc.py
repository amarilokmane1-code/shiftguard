"""Quality control with a full audit trail.

Exclusion reasons are applied **in this order**; each excluded subject is attributed to
the *first* reason that applies, so the per-reason counts add up to the total exclusions:

1. ``no_preprocessed_data``  -- PCP provides no file (``FILE_ID == "no_filename"``).
2. ``rater_qc_fail``         -- fails the configured manual-rater rule (below).
3. ``missing_covariates``    -- age or sex missing (needed for ComBat; never diagnosis).
4. ``missing_timeseries``    -- file absent or unreadable after download.
5. ``too_few_timepoints``    -- fewer than ``min_timepoints`` volumes.
6. ``bad_rois``              -- non-finite values, or more than ``max_zero_variance_rois``
                                constant (zero-variance) ROIs, whose correlations are undefined.
7. ``small_site``            -- after steps 1-6, the subject's site has fewer than
                                ``min_subjects_per_class_per_site`` subjects in either class.
                                The whole site is dropped (it could not be a meaningful
                                deployment site, nor support calibration per class).

Zero-variance policy
--------------------
``exclude_subjects`` (default, main analysis)
    Rule 6 above: subjects with more than ``max_zero_variance_rois`` constant ROIs are
    excluded.
``drop_rois`` (sensitivity analysis)
    Constant ROIs never exclude a subject. Instead, every ROI that is constant in any
    subject of the final cohort is removed for **all** subjects (same feature set
    everywhere); see :func:`shiftguard.data.dataset.build_dataset`.

Rater rules (PCP manual QC fields)
----------------------------------
``none``
    No rater-based exclusion.
``rater_1_pass`` (default)
    Exclude subjects that rater 1 marked ``fail``.
``all_raters_pass``
    Exclude subjects that *any* rater (1, 2 anat/func, 3 anat/func) marked ``fail``.
``nilearn_quality_checked``
    The filter of ``nilearn.datasets.fetch_abide_pcp(quality_checked=True)``: rater 1
    ``OK``, rater 2 anat/func ``OK`` or ``maybe``, rater 3 anat/func ``OK``. Common in the
    literature; included for comparability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from shiftguard.data.phenotypes import QC_COLUMNS, counts_by_site

REASONS: tuple[str, ...] = (
    "no_preprocessed_data",
    "rater_qc_fail",
    "missing_covariates",
    "missing_timeseries",
    "too_few_timepoints",
    "bad_rois",
    "small_site",
)
RATER_RULES: tuple[str, ...] = (
    "none",
    "rater_1_pass",
    "all_raters_pass",
    "nilearn_quality_checked",
)


ZERO_VARIANCE_POLICIES: tuple[str, ...] = ("exclude_subjects", "drop_rois")


@dataclass(frozen=True)
class QCConfig:
    """QC parameters (see the module docstring)."""

    rater_rule: str = "rater_1_pass"
    min_subjects_per_class_per_site: int = 10
    min_timepoints: int = 50
    max_zero_variance_rois: int = 0
    zero_variance_policy: str = "exclude_subjects"
    covariates: tuple[str, ...] = ("AGE_AT_SCAN", "SEX")

    def __post_init__(self) -> None:
        if self.rater_rule not in RATER_RULES:
            raise ValueError(f"rater_rule must be one of {RATER_RULES}, got {self.rater_rule!r}")
        if self.zero_variance_policy not in ZERO_VARIANCE_POLICIES:
            raise ValueError(
                f"zero_variance_policy must be one of {ZERO_VARIANCE_POLICIES}, "
                f"got {self.zero_variance_policy!r}"
            )

    @classmethod
    def from_config(
        cls, qc_cfg: dict[str, Any], covariates: Sequence[str] | None = None
    ) -> QCConfig:
        """Build from the ``qc`` section of a config."""
        kwargs = {
            k: qc_cfg[k]
            for k in (
                "rater_rule",
                "min_subjects_per_class_per_site",
                "min_timepoints",
                "max_zero_variance_rois",
                "zero_variance_policy",
            )
            if k in qc_cfg
        }
        if covariates is not None:
            kwargs["covariates"] = tuple(covariates)
        return cls(**kwargs)


def rater_pass(pheno: pd.DataFrame, rule: str) -> pd.Series:
    """Boolean mask of subjects that pass the manual-rater rule."""
    if rule not in RATER_RULES:
        raise ValueError(f"unknown rater rule {rule!r}")
    if rule == "none":
        return pd.Series(True, index=pheno.index)
    missing = [c for c in QC_COLUMNS if c not in pheno.columns]
    if missing:
        raise ValueError(f"phenotypes lack QC columns {missing}")
    q = pheno[list(QC_COLUMNS)].astype("string").apply(lambda s: s.str.strip().str.lower())
    if rule == "rater_1_pass":
        return ~(q["qc_rater_1"] == "fail").fillna(False)
    if rule == "all_raters_pass":
        return ~(q == "fail").fillna(False).any(axis=1)
    ok = (q == "ok").fillna(False)
    ok_or_maybe = q.isin(["ok", "maybe"]).fillna(False)
    return (
        ok["qc_rater_1"]
        & ok_or_maybe["qc_anat_rater_2"]
        & ok_or_maybe["qc_func_rater_2"]
        & ok["qc_anat_rater_3"]
        & ok["qc_func_rater_3"]
    )


@dataclass
class QCResult:
    """Outcome of :func:`apply_qc`."""

    included: pd.DataFrame
    exclusions: pd.DataFrame
    summary: pd.DataFrame
    config: QCConfig
    small_sites: list[str] = field(default_factory=list)

    @property
    def reason_counts(self) -> pd.Series:
        """Number of excluded subjects per reason (all reasons, in order)."""
        return self.exclusions["reason"].value_counts().reindex(list(REASONS), fill_value=0)


def apply_qc(pheno: pd.DataFrame, ts_info: pd.DataFrame, config: QCConfig) -> QCResult:
    """Apply the QC rules and return included subjects, exclusions and a per-site summary.

    Parameters
    ----------
    pheno
        Output of :func:`shiftguard.data.phenotypes.load_phenotypes` (all subjects).
    ts_info
        Output of :func:`shiftguard.data.timeseries.timeseries_info` for the subjects
        whose files were looked for (subjects absent from it count as missing).
    config
        QC parameters.
    """
    df = pheno.merge(ts_info, on="FILE_ID", how="left")
    df["readable"] = df["readable"].astype("boolean").fillna(False).astype(bool)
    reason = pd.Series(pd.NA, index=df.index, dtype="object")

    def _flag(mask: pd.Series, name: str) -> None:
        reason[mask & reason.isna()] = name

    _flag(df["FILE_ID"] == "no_filename", "no_preprocessed_data")
    _flag(~rater_pass(df, config.rater_rule), "rater_qc_fail")
    cov = [c for c in config.covariates if c in df.columns]
    if len(cov) != len(config.covariates):
        raise ValueError(f"covariates {config.covariates} not all in phenotypes")
    _flag(df[cov].isna().any(axis=1), "missing_covariates")
    _flag(~df["readable"], "missing_timeseries")
    _flag(df["n_timepoints"].fillna(0) < config.min_timepoints, "too_few_timepoints")
    bad = df["has_nonfinite"].astype("boolean").fillna(False).astype(bool)
    if config.zero_variance_policy == "exclude_subjects":
        bad |= df["n_zero_variance_rois"].fillna(0) > config.max_zero_variance_rois
    _flag(bad, "bad_rois")

    remaining = reason.isna()
    per_class = pd.crosstab(df.loc[remaining, "SITE_ID"], df.loc[remaining, "label"])
    per_class = per_class.reindex(columns=[0, 1], fill_value=0)
    small = sorted(
        per_class.index[(per_class < config.min_subjects_per_class_per_site).any(axis=1)]
    )
    _flag(df["SITE_ID"].isin(small), "small_site")

    excluded = reason.notna()
    exclusions = df.loc[excluded, ["SUB_ID", "FILE_ID", "SITE_ID", "DX_GROUP", "diagnosis"]].copy()
    exclusions["reason"] = reason[excluded].astype(str)
    included = df.loc[~excluded].drop(columns=["readable"]).reset_index(drop=True)
    return QCResult(
        included=included,
        exclusions=exclusions.reset_index(drop=True),
        summary=qc_summary(df, reason),
        config=config,
        small_sites=small,
    )


def qc_summary(df: pd.DataFrame, reason: pd.Series) -> pd.DataFrame:
    """Per-site counts before QC, per exclusion reason, and after QC, plus a TOTAL row."""
    before = counts_by_site(df, prefix="n_before")
    after = counts_by_site(df[reason.isna()], prefix="n_after").reindex(before.index, fill_value=0)
    excl = pd.crosstab(df["SITE_ID"], reason.fillna("kept"))
    excl = excl.reindex(index=before.index, columns=list(REASONS), fill_value=0)
    excl.columns = [f"excl_{c}" for c in excl.columns]
    table = pd.concat([before, excl, after], axis=1)
    table.loc["TOTAL"] = table.sum(axis=0)
    table.index.name = "SITE_ID"
    return table.astype(int).reset_index()
