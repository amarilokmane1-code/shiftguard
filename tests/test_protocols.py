"""Leakage tests for the evaluation protocols."""

import numpy as np
import pytest

from shiftguard.data.synthetic import make_synthetic_dataset
from shiftguard.evaluation.protocols import (
    LeakageError,
    Split,
    check_split,
    loso_splits,
    make_splits,
    within_site_splits,
)

DATA = {
    "n_sites": 5,
    "n_subjects_per_site": 40,
    "n_features": 5,
    "site_shift_scale": 1.0,
    "signal_strength": 1.0,
}
EXP = {"n_seeds": 3, "calibration_fraction": 0.3, "test_fraction": 0.2, "base_seed": 0}


@pytest.fixture(scope="module")
def cohort():
    ds = make_synthetic_dataset(DATA, seed=1)
    return ds.phenotypes["SUB_ID"].to_numpy(), ds.sites, ds.labels


def test_loso_no_subject_in_two_parts_and_test_site_never_in_training(cohort):
    ids, sites, labels = cohort
    splits = make_splits("loso", sites, labels, EXP)
    assert len(splits) == EXP["n_seeds"] * len(set(sites))
    for sp in splits:
        check_split(sp, ids, sites)
        parts = [set(ids[sp.train]), set(ids[sp.cal]), set(ids[sp.test])]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert sp.test_site not in set(sites[sp.train]) | set(sites[sp.cal])
        assert set(sites[sp.test]) == {sp.test_site}
        assert len(sp.train) + len(sp.cal) + len(sp.test) == len(ids)  # nobody dropped


def test_loso_test_sets_partition_the_cohort_per_seed(cohort):
    ids, sites, labels = cohort
    for seed in range(EXP["n_seeds"]):
        tests = [sp.test for sp in loso_splits(sites, labels, [seed])]
        joined = np.sort(np.concatenate(tests))
        np.testing.assert_array_equal(joined, np.arange(len(ids)))


def test_within_site_splits_cover_every_site_in_every_part(cohort):
    ids, sites, labels = cohort
    for sp in within_site_splits(sites, labels, [0, 1], 0.3, 0.2):
        check_split(sp, ids, sites)
        for part in (sp.train, sp.cal, sp.test):
            assert set(sites[part]) == set(sites)
        assert len(sp.test) == pytest.approx(0.2 * len(ids), abs=len(set(sites)) * 2)


def test_splits_are_reproducible_and_seed_dependent(cohort):
    _, sites, labels = cohort
    a = loso_splits(sites, labels, [0, 1])
    b = loso_splits(sites, labels, [0, 1])
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.train, y.train)
    n_sites = len(set(sites))
    assert not np.array_equal(a[0].train, a[n_sites].train)  # seed 0 vs seed 1, same site


def test_calibration_split_is_stratified_by_site_and_diagnosis(cohort):
    _, sites, labels = cohort
    sp = loso_splits(sites, labels, [0], calibration_fraction=0.3)[0]
    for site in set(sites[sp.cal]):
        for y in (0, 1):
            n_cal = np.sum((sites[sp.cal] == site) & (labels[sp.cal] == y))
            n_all = np.sum((sites == site) & (labels == y))
            assert abs(n_cal - 0.3 * n_all) <= 1.5


def test_check_split_catches_leaks(cohort):
    ids, sites, labels = cohort
    good = loso_splits(sites, labels, [0])[0]
    overlap = Split(
        "loso", 0, good.test_site, good.train, np.r_[good.cal, good.train[:1]], good.test
    )
    with pytest.raises(LeakageError, match="both train and cal"):
        check_split(overlap, ids, sites)
    other_row = np.flatnonzero(sites == good.test_site)[0]
    site_leak = Split(
        "loso",
        0,
        good.test_site,
        np.r_[good.train, other_row],
        good.cal,
        np.setdiff1d(good.test, [other_row]),
    )
    with pytest.raises(LeakageError, match="test site"):
        check_split(site_leak, ids, sites)
    # Same subject ID on two different rows (e.g. a duplicated scan) is also caught
    dup_ids = ids.copy()
    dup_ids[good.test[0]] = ids[good.train[0]]
    with pytest.raises(LeakageError, match="subjects in both"):
        check_split(good, dup_ids, sites)
    with pytest.raises(LeakageError, match="empty"):
        check_split(
            Split("loso", 0, good.test_site, good.train, good.cal, good.test[:0]), ids, sites
        )


def test_unknown_protocol():
    with pytest.raises(ValueError):
        make_splits("kfold", ["a"], [0], EXP)
