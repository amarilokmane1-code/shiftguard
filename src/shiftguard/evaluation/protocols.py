"""Evaluation protocols: leave-one-site-out (site shift) and within-site (no shift).

Leave-one-site-out (LOSO)
    For each site ``s`` and seed: all subjects of ``s`` form the test (deployment) set.
    The other sites are split by subject into train / calibration, stratified by
    site x diagnosis (default 70 / 30). Site ``s`` is never used for training,
    calibration or hyperparameter tuning.

Within-site (exchangeable reference)
    For each seed: a random subject-level split of the pooled cohort into test (default
    20%), then train / calibration, all stratified by site x diagnosis, so every site is
    present in every split.

Every split is checked by :func:`check_split` before use: the three index sets are
disjoint, no subject ID appears in two of them, and (LOSO) the test site never appears
in train or calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split

from shiftguard.utils.seeding import derive_seed

PROTOCOLS: tuple[str, ...] = ("within_site", "loso")
POOLED = "pooled"


class LeakageError(AssertionError):
    """Raised when a split violates the no-leakage rules."""


@dataclass(frozen=True)
class Split:
    """Row indices of one train / calibration / test split."""

    protocol: str
    seed: int
    test_site: str
    train: NDArray[np.intp]
    cal: NDArray[np.intp]
    test: NDArray[np.intp]


def _strata(sites: NDArray[np.object_], labels: NDArray[np.intp]) -> NDArray[np.str_]:
    return np.array([f"{s}|{y}" for s, y in zip(sites, labels, strict=True)])


def _split(
    idx: NDArray[np.intp], frac: float, strata: NDArray[np.str_], seed: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    a, b = train_test_split(idx, test_size=frac, stratify=strata[idx], random_state=seed)
    return np.sort(a), np.sort(b)


def loso_splits(
    sites: ArrayLike,
    labels: ArrayLike,
    seeds: Sequence[int],
    calibration_fraction: float = 0.3,
    base_seed: int = 0,
) -> list[Split]:
    """One split per (seed, held-out site)."""
    s = np.asarray(sites, dtype=object)
    y = np.asarray(labels).astype(int)
    strata = _strata(s, y)
    out = []
    for seed in seeds:
        for site in sorted(set(s.tolist())):
            test = np.flatnonzero(s == site)
            rest = np.flatnonzero(s != site)
            train, cal = _split(
                rest, calibration_fraction, strata, derive_seed(base_seed, "loso", seed, site)
            )
            out.append(Split("loso", int(seed), str(site), train, cal, test))
    return out


def within_site_splits(
    sites: ArrayLike,
    labels: ArrayLike,
    seeds: Sequence[int],
    calibration_fraction: float = 0.3,
    test_fraction: float = 0.2,
    base_seed: int = 0,
) -> list[Split]:
    """One pooled random split per seed, stratified by site x diagnosis."""
    s = np.asarray(sites, dtype=object)
    y = np.asarray(labels).astype(int)
    strata = _strata(s, y)
    idx = np.arange(len(y))
    out = []
    for seed in seeds:
        rs = derive_seed(base_seed, "within_site", seed)
        rest, test = _split(idx, test_fraction, strata, rs)
        train, cal = _split(rest, calibration_fraction, strata, rs + 1)
        out.append(Split("within_site", int(seed), POOLED, train, cal, test))
    return out


def make_splits(protocol: str, sites: ArrayLike, labels: ArrayLike, exp_cfg: dict) -> list[Split]:
    """Dispatch on ``protocol`` using the ``experiment`` config section."""
    seeds = list(range(exp_cfg["n_seeds"]))
    common = {
        "calibration_fraction": exp_cfg["calibration_fraction"],
        "base_seed": exp_cfg["base_seed"],
    }
    if protocol == "loso":
        return loso_splits(sites, labels, seeds, **common)
    if protocol == "within_site":
        return within_site_splits(
            sites, labels, seeds, test_fraction=exp_cfg["test_fraction"], **common
        )
    raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")


def check_split(split: Split, subject_ids: ArrayLike, sites: ArrayLike) -> None:
    """Assert the no-leakage rules for one split.

    Raises
    ------
    LeakageError
        If two parts share a row or a subject ID, a part is empty, or (LOSO) the test site
        appears in train/calibration or the test set contains another site.
    """
    ids = np.asarray(subject_ids)
    s = np.asarray(sites, dtype=object)
    parts = {"train": split.train, "cal": split.cal, "test": split.test}
    for name, idx in parts.items():
        if len(idx) == 0:
            raise LeakageError(f"{split.protocol}/{split.test_site}: empty {name} split")
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(ids[parts[a]].tolist()) & set(ids[parts[b]].tolist())
            if shared:
                raise LeakageError(
                    f"{split.protocol}/{split.test_site}/seed {split.seed}: subjects in both "
                    f"{a} and {b}: {sorted(shared)[:5]}"
                )
    if split.protocol == "loso":
        fit_sites = set(s[split.train].tolist()) | set(s[split.cal].tolist())
        if split.test_site in fit_sites:
            raise LeakageError(f"test site {split.test_site} appears in train/calibration")
        if set(s[split.test].tolist()) != {split.test_site}:
            raise LeakageError(f"LOSO test set for {split.test_site} contains other sites")
