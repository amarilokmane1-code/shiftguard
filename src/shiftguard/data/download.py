"""Download ABIDE I Preprocessed (PCP) ROI time series and phenotypes.

Primary route: :func:`nilearn.datasets.fetch_abide_pcp`. Fallback: the same files fetched
directly over HTTPS from the public PCP S3 bucket, in parallel. Both routes write the
same on-disk layout, so the rest of the pipeline does not care which one ran::

    <data_dir>/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv
    <data_dir>/ABIDE_pcp/<pipeline>/<strategy>/<FILE_ID>_<derivative>.1D

All subjects with a preprocessed file are downloaded (``quality_checked=False``); the
QC rule is applied afterwards by :mod:`shiftguard.data.qc` so every exclusion is logged.

Setting ``SHIFTGUARD_OFFLINE=1`` (as CI and the test suite do) makes every function here
refuse to touch the network.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from shiftguard.utils.log import get_logger

log = get_logger(__name__)

S3_ROOT = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENOTYPE_FILE = "Phenotypic_V1_0b_preprocessed1.csv"
DATASET_DIRNAME = "ABIDE_pcp"
PIPELINES = ("cpac", "ccs", "dparsf", "niak")
STRATEGIES = ("filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal")
OFFLINE_ENV = "SHIFTGUARD_OFFLINE"

Method = Literal["auto", "nilearn", "s3"]


class OfflineError(RuntimeError):
    """Raised when a download is attempted while ``SHIFTGUARD_OFFLINE`` is set."""


def ensure_online() -> None:
    """Raise :class:`OfflineError` if downloads are disabled."""
    if os.environ.get(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes"}:
        raise OfflineError(
            f"Downloads are disabled ({OFFLINE_ENV} is set). CI and tests never download ABIDE."
        )


@dataclass(frozen=True)
class AbideSpec:
    """Which PCP derivative to use."""

    pipeline: str = "cpac"
    strategy: str = "filt_noglobal"
    derivative: str = "rois_cc200"

    def __post_init__(self) -> None:
        if self.pipeline not in PIPELINES:
            raise ValueError(f"pipeline must be one of {PIPELINES}, got {self.pipeline!r}")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if not self.derivative.startswith("rois_"):
            raise ValueError(
                f"only ROI time-series derivatives are supported, got {self.derivative!r}"
            )

    @classmethod
    def from_config(cls, data_cfg: dict) -> AbideSpec:
        """Build from the ``data`` section of a config."""
        return cls(data_cfg["pipeline"], data_cfg["strategy"], data_cfg["derivative"])

    @property
    def band_pass_filtering(self) -> bool:
        """Whether the strategy includes band-pass filtering."""
        return self.strategy.startswith("filt")

    @property
    def global_signal_regression(self) -> bool:
        """Whether the strategy includes global signal regression."""
        return self.strategy.endswith("_global")

    @property
    def tag(self) -> str:
        """Short identifier used in cache file names."""
        return f"{self.pipeline}_{self.strategy}_{self.derivative}"

    def filename(self, file_id: str) -> str:
        """File name of one subject's time series."""
        return f"{file_id}_{self.derivative}.1D"

    def url(self, file_id: str) -> str:
        """Public HTTPS URL of one subject's time series."""
        return (
            f"{S3_ROOT}/Outputs/{self.pipeline}/{self.strategy}/"
            f"{self.derivative}/{self.filename(file_id)}"
        )


def dataset_dir(data_dir: Path) -> Path:
    """``<data_dir>/ABIDE_pcp``."""
    return Path(data_dir) / DATASET_DIRNAME


def phenotype_path(data_dir: Path) -> Path:
    """Path of the PCP phenotypic CSV."""
    return dataset_dir(data_dir) / PHENOTYPE_FILE


def timeseries_dir(data_dir: Path, spec: AbideSpec) -> Path:
    """Directory holding the ``.1D`` files of one pipeline/strategy."""
    return dataset_dir(data_dir) / spec.pipeline / spec.strategy


def timeseries_path(data_dir: Path, spec: AbideSpec, file_id: str) -> Path:
    """Path of one subject's ``.1D`` time-series file."""
    return timeseries_dir(data_dir, spec) / spec.filename(file_id)


# --------------------------------------------------------------------------- HTTPS fallback
def _http_get(url: str, dest: Path, *, retries: int = 3, timeout: float = 60.0) -> int:
    """Download ``url`` to ``dest`` atomically and return the number of bytes written."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            tmp.replace(dest)
            return dest.stat().st_size
        except Exception as exc:  # network errors are retried, then reported
            last_error = exc
            log.debug("attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def fetch_phenotypes(data_dir: Path) -> Path:
    """Download the phenotypic CSV if it is not already present."""
    path = phenotype_path(data_dir)
    if path.is_file() and path.stat().st_size > 0:
        return path
    ensure_online()
    _http_get(f"{S3_ROOT}/{PHENOTYPE_FILE}", path)
    log.info("Downloaded phenotypes to %s", path)
    return path


def available_file_ids(pheno_csv: Path) -> list[str]:
    """``FILE_ID`` of every subject that has preprocessed data (not ``no_filename``)."""
    pheno = pd.read_csv(pheno_csv, usecols=["FILE_ID"])
    return [f for f in pheno["FILE_ID"].astype(str) if f != "no_filename"]


def missing_files(data_dir: Path, spec: AbideSpec, file_ids: Iterable[str]) -> list[str]:
    """File IDs whose time series is absent or empty on disk."""
    out = []
    for fid in file_ids:
        path = timeseries_path(data_dir, spec, fid)
        if not path.is_file() or path.stat().st_size == 0:
            out.append(fid)
    return out


def download_s3(
    data_dir: Path, spec: AbideSpec, file_ids: Iterable[str], *, n_jobs: int = 8
) -> list[str]:
    """Download time series over HTTPS in parallel; return the IDs that failed."""
    ensure_online()
    todo = list(file_ids)
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        futures = {
            pool.submit(_http_get, spec.url(fid), timeseries_path(data_dir, spec, fid)): fid
            for fid in todo
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            fid = futures[fut]
            try:
                fut.result()
            except RuntimeError as exc:
                failed.append(fid)
                log.warning("%s", exc)
            if i % 100 == 0 or i == len(todo):
                log.info("S3 download: %d/%d files", i, len(todo))
    return sorted(failed)


def download_nilearn(data_dir: Path, spec: AbideSpec, n_subjects: int | None = None) -> None:
    """Fetch through :func:`nilearn.datasets.fetch_abide_pcp` (no QC filtering)."""
    ensure_online()
    from nilearn.datasets import fetch_abide_pcp

    fetch_abide_pcp(
        data_dir=str(data_dir),
        n_subjects=n_subjects,
        pipeline=spec.pipeline,
        band_pass_filtering=spec.band_pass_filtering,
        global_signal_regression=spec.global_signal_regression,
        derivatives=[spec.derivative],
        quality_checked=False,
        verbose=1,
    )


@dataclass(frozen=True)
class DownloadReport:
    """What is on disk after a download."""

    n_expected: int
    n_present: int
    missing: tuple[str, ...]
    method_used: str
    bytes_on_disk: int


def directory_size(path: Path) -> int:
    """Total size in bytes of all files below ``path``."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def download_abide(
    data_dir: Path,
    spec: AbideSpec,
    *,
    method: Method = "auto",
    n_jobs: int = 8,
    n_subjects: int | None = None,
) -> DownloadReport:
    """Make sure the phenotypes and every available time series are on disk.

    Parameters
    ----------
    data_dir
        Root data directory (git-ignored).
    spec
        Pipeline, strategy and derivative.
    method
        ``"nilearn"``, ``"s3"`` or ``"auto"`` (nilearn first, then the HTTPS fallback for
        anything still missing).
    n_jobs
        Parallel connections for the HTTPS fallback.
    n_subjects
        Optional cap, for quick trials.
    """
    data_dir = Path(data_dir)
    pheno_csv = fetch_phenotypes(data_dir)
    file_ids = available_file_ids(pheno_csv)
    if n_subjects is not None:
        file_ids = file_ids[:n_subjects]
    todo = missing_files(data_dir, spec, file_ids)
    used = "cache"
    if todo:
        ensure_online()
        log.info("%d of %d time-series files to download (%s)", len(todo), len(file_ids), spec.tag)
        if method in ("auto", "nilearn"):
            try:
                download_nilearn(data_dir, spec, n_subjects=n_subjects)
                used = "nilearn"
            except Exception as exc:  # any nilearn failure triggers the documented fallback
                if method == "nilearn":
                    raise
                log.warning("nilearn fetch_abide_pcp failed (%s); using the S3 fallback", exc)
            todo = missing_files(data_dir, spec, file_ids)
        if todo and method in ("auto", "s3"):
            download_s3(data_dir, spec, todo, n_jobs=n_jobs)
            used = "s3" if used == "cache" else f"{used}+s3"
    missing = missing_files(data_dir, spec, file_ids)
    report = DownloadReport(
        n_expected=len(file_ids),
        n_present=len(file_ids) - len(missing),
        missing=tuple(missing),
        method_used=used,
        bytes_on_disk=directory_size(dataset_dir(data_dir)),
    )
    log.info(
        "ABIDE on disk: %d/%d time series, %.1f MB (method: %s)",
        report.n_present,
        report.n_expected,
        report.bytes_on_disk / 1e6,
        report.method_used,
    )
    return report
