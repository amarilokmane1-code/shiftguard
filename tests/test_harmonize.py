"""ComBat harmonisation: removes site effects, never uses labels."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neuroHarmonize")

from shiftguard.config import load_config
from shiftguard.data.synthetic import load_any_dataset, make_synthetic_dataset
from shiftguard.features.harmonize import combat_harmonize, harmonize_dataset

DATA = {
    "n_sites": 4,
    "n_subjects_per_site": 50,
    "n_features": 30,
    "site_shift_scale": 2.0,
    "signal_strength": 1.0,
}


def _site_mean_spread(X, sites):
    means = np.array([X[sites == s].mean(axis=0) for s in np.unique(sites)])
    return means.std(axis=0).mean()


def test_combat_removes_site_location_effects():
    ds = make_synthetic_dataset(DATA, seed=0)
    X = combat_harmonize(ds.features, ds.sites, ds.phenotypes[["AGE_AT_SCAN", "SEX"]])
    assert X.shape == ds.features.shape
    assert _site_mean_spread(X, ds.sites) < 0.25 * _site_mean_spread(ds.features, ds.sites)


def test_combat_output_does_not_depend_on_labels():
    ds = make_synthetic_dataset(DATA, seed=1)
    cov = ds.phenotypes[["AGE_AT_SCAN", "SEX"]]
    a = combat_harmonize(ds.features, ds.sites, cov)
    shuffled = ds.phenotypes.assign(label=np.random.default_rng(0).permutation(ds.labels))
    b = combat_harmonize(ds.features, shuffled["SITE_ID"], shuffled[["AGE_AT_SCAN", "SEX"]])
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="never"):
        combat_harmonize(ds.features, ds.sites, ds.phenotypes[["AGE_AT_SCAN", "label"]])
    with pytest.raises(ValueError, match="missing"):
        combat_harmonize(ds.features, ds.sites, cov.assign(SEX=np.nan))


def test_harmonize_dataset_from_config():
    cfg = load_config("configs/experiment/synthetic.yaml")
    plain = load_any_dataset(cfg)
    assert harmonize_dataset(plain, cfg) is plain  # default: none
    cfg["features"]["harmonization"] = "combat"
    harm = load_any_dataset(cfg)
    assert harm.feature_kind == "correlation+combat"
    assert not np.allclose(harm.features, plain.features)
    pd.testing.assert_frame_equal(harm.phenotypes, plain.phenotypes)
    cfg["features"]["harmonization"] = "zscore"
    with pytest.raises(ValueError, match="unknown"):
        harmonize_dataset(plain, cfg)
