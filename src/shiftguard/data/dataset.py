"""Build and load the QC'd ABIDE dataset (the ``shiftguard features`` stage).

Outputs, for one pipeline/strategy/atlas ``<tag>``:

``data/derived/<tag>/``  (git-ignored: subject-level, derived from ABIDE)
    ``phenotypes_qc.csv``     included subjects with phenotypes and time-series info
    ``exclusions.csv``        every excluded subject with its reason
    ``timeseries.npz``        time series of the included subjects (float32)
    ``features_<kind>.npz``   subject-wise connectivity features (correlation kinds only)

``reports/``  (committed: site-level aggregates only)
    ``qc_summary.{csv,md}``               counts per site before/after QC and per reason
    ``figures/fig1_data_map.{png,pdf}``   subjects per site and diagnosis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shiftguard.data.download import AbideSpec, available_file_ids, phenotype_path
from shiftguard.data.phenotypes import cohort_summary, load_phenotypes
from shiftguard.data.qc import QCConfig, QCResult, apply_qc
from shiftguard.data.timeseries import (
    load_timeseries,
    read_timeseries_cache,
    save_timeseries,
    timeseries_info,
)
from shiftguard.features.connectivity import SUBJECTWISE_KINDS, subjectwise_features
from shiftguard.utils.log import get_logger
from shiftguard.utils.paths import data_dir, reports_dir
from shiftguard.utils.tables import write_table

log = get_logger(__name__)


def variant_suffix(config: dict[str, Any]) -> str:
    """``""`` for the main QC, ``"_droprois"`` for the ROI-exclusion sensitivity analysis."""
    policy = config.get("qc", {}).get("zero_variance_policy", "exclude_subjects")
    return "" if policy == "exclude_subjects" else "_droprois"


def derived_dir(config: dict[str, Any]) -> Path:
    """``<data_dir>/derived/<pipeline>_<strategy>_<derivative>[_droprois]``."""
    tag = AbideSpec.from_config(config["data"]).tag
    return data_dir(config) / "derived" / f"{tag}{variant_suffix(config)}"


def constant_rois(ts: list[NDArray[np.float64]], atol: float = 1e-8) -> NDArray[np.intp]:
    """Return the indices of ROIs that are constant in at least one time series."""
    if not ts:
        return np.array([], dtype=np.intp)
    return np.flatnonzero(np.any([t.std(axis=0) <= atol for t in ts], axis=0))


def _feature_file(kind: str) -> str:
    return f"features_{kind.replace(' ', '_')}.npz"


def build_dataset(config: dict[str, Any]) -> QCResult:
    """Run QC, cache time series and subject-wise features, write the QC report and Fig. 1."""
    from shiftguard.viz.figures import figure_data_map

    spec = AbideSpec.from_config(config["data"])
    root = data_dir(config)
    pheno_csv = phenotype_path(root)
    if not pheno_csv.is_file():
        raise FileNotFoundError(f"{pheno_csv} not found; run `shiftguard download` first")

    pheno = load_phenotypes(pheno_csv, extra_columns=config["data"].get("phenotypes", ()))
    file_ids = available_file_ids(pheno_csv)
    log.info("Loading %d time series (%s)", len(file_ids), spec.tag)
    ts = load_timeseries(root, spec, file_ids)
    info = timeseries_info(ts)

    covariates = config.get("harmonization", {}).get("covariates", ("AGE_AT_SCAN", "SEX"))
    qc = apply_qc(pheno, info, QCConfig.from_config(config["qc"], covariates))
    log.info(
        "QC kept %d/%d subjects from %d sites; exclusions by reason: %s",
        len(qc.included),
        len(pheno),
        qc.included["SITE_ID"].nunique(),
        qc.reason_counts.to_dict(),
    )

    out = derived_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    kept_ids = qc.included["FILE_ID"].tolist()
    kept_ts = {fid: ts[fid] for fid in kept_ids}
    dropped: list[int] = []
    if qc.config.zero_variance_policy == "drop_rois":
        dropped = constant_rois(list(kept_ts.values())).tolist()  # type: ignore[arg-type]
        keep = np.setdiff1d(np.arange(next(iter(kept_ts.values())).shape[1]), dropped)
        kept_ts = {fid: t[:, keep] for fid, t in kept_ts.items()}  # type: ignore[index]
        log.info("drop_rois: removing %d constant ROIs for every subject", len(dropped))
    qc.included.to_csv(out / "phenotypes_qc.csv", index=False, lineterminator="\n")
    qc.exclusions.to_csv(out / "exclusions.csv", index=False, lineterminator="\n")
    save_timeseries(out / "timeseries.npz", kept_ts)  # type: ignore[arg-type]

    kind = config["features"]["kind"]
    if kind in SUBJECTWISE_KINDS:
        X = subjectwise_features(
            [kept_ts[f] for f in kept_ids],
            kind=kind,
            fisher_z=config["features"]["fisher_z"],
            cov_estimator=config["features"].get("cov_estimator", "ledoit_wolf"),
        )
        np.savez_compressed(out / _feature_file(kind), X=X.astype(np.float32), file_ids=kept_ids)
        log.info("Cached %s features: %d subjects x %d edges", kind, *X.shape)
    else:
        log.info("%s features are fitted inside each fold (training subjects only)", kind)

    (out / "qc_config.json").write_text(
        json.dumps(
            {
                "qc": vars(qc.config) | {"covariates": list(qc.config.covariates)},
                "small_sites": qc.small_sites,
                "dropped_rois": dropped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    reports = reports_dir(config)
    reports.mkdir(parents=True, exist_ok=True)
    sfx = variant_suffix(config)
    write_table(
        qc.summary,
        reports / f"qc_summary{sfx}",
        title=f"QC summary ({spec.tag}, rater rule: {qc.config.rater_rule})",
    )
    write_table(
        cohort_summary(qc.included),
        reports / "tables" / f"cohort_summary{sfx}",
        digits=1,
        title=f"Cohort after QC, per site ({spec.tag})",
    )
    figure_data_map(
        qc.summary,
        reports / "figures" / f"fig1_data_map{sfx}",
        min_per_class=qc.config.min_subjects_per_class_per_site,
    )
    return qc


@dataclass
class Dataset:
    """The QC'd cohort, ready for the evaluation protocols."""

    phenotypes: pd.DataFrame
    time_series: list[NDArray[np.float64]]
    features: NDArray[np.float64] | None
    feature_kind: str

    @property
    def labels(self) -> NDArray[np.intp]:
        """1 = ASD, 0 = TDC."""
        return self.phenotypes["label"].to_numpy(dtype=np.intp)

    @property
    def sites(self) -> NDArray[np.object_]:
        """Acquisition site of each subject."""
        return self.phenotypes["SITE_ID"].to_numpy(dtype=object)


def load_dataset(config: dict[str, Any]) -> Dataset:
    """Load the cohort written by :func:`build_dataset` (subject order is preserved)."""
    out = derived_dir(config)
    pheno_path = out / "phenotypes_qc.csv"
    if not pheno_path.is_file():
        raise FileNotFoundError(f"{pheno_path} not found; run `shiftguard features` first")
    pheno = pd.read_csv(pheno_path, dtype={"FILE_ID": str, "SITE_ID": str})
    file_ids = pheno["FILE_ID"].tolist()
    ts = read_timeseries_cache(out / "timeseries.npz", file_ids)
    kind = config["features"]["kind"]
    X = None
    if kind in SUBJECTWISE_KINDS:
        with np.load(out / _feature_file(kind)) as npz:
            if npz["file_ids"].tolist() != file_ids:
                raise ValueError(
                    "feature cache and phenotype table are out of sync; rerun features"
                )
            X = npz["X"].astype(np.float64)
    return Dataset(pheno, [ts[f] for f in file_ids], X, kind)
