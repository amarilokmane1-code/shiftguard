"""Site-effect description: site predictability and variance explained by site."""

import numpy as np
import pytest

from shiftguard.data.synthetic import make_synthetic_dataset
from shiftguard.evaluation.site_effect import describe_site_effect, variance_by_site

DATA = {
    "n_sites": 4,
    "n_subjects_per_site": 50,
    "n_features": 30,
    "site_shift_scale": 2.0,
    "signal_strength": 1.0,
}


def _cov(ds):
    return ds.phenotypes[["AGE_AT_SCAN", "SEX"]].assign(dx=ds.labels)


def test_strong_site_shift_is_detected():
    ds = make_synthetic_dataset(DATA, seed=0)
    eff = describe_site_effect(
        ds.features, ds.sites, _cov(ds), n_components_predict=10, n_components_variance=5
    )
    assert eff.site_balanced_accuracy > 0.9
    assert eff.chance == pytest.approx(0.25)
    assert 0.3 < eff.r2_site <= 1.0
    assert 0.0 <= eff.r2_site_partial <= eff.r2_site + 1e-9
    assert eff.embedding.shape == (len(ds.labels), 2)


def test_no_site_effect_gives_chance_and_small_r2():
    ds = make_synthetic_dataset(DATA | {"site_shift_scale": 0.0}, seed=1)
    eff = describe_site_effect(
        ds.features, ds.sites, _cov(ds), n_components_predict=10, n_components_variance=5
    )
    assert eff.site_balanced_accuracy < 0.45
    assert eff.r2_site < 0.1


def test_variance_table_columns():
    ds = make_synthetic_dataset(DATA, seed=2)
    comp, emb = variance_by_site(ds.features, ds.sites, _cov(ds), n_components=4)
    assert list(comp["component"]) == [1, 2, 3, 4]
    assert comp["explained_variance_ratio"].is_monotonic_decreasing
    assert np.all((comp[["r2_site", "r2_covariates"]] >= -1e-9).to_numpy())
    assert emb.shape[1] == 2


def test_cli_site_effect(tmp_path):
    pytest.importorskip("neuroHarmonize")
    import pandas as pd
    from typer.testing import CliRunner

    from shiftguard.cli import app

    res = CliRunner().invoke(
        app,
        [
            "site-effect",
            "-c",
            "configs/experiment/synthetic.yaml",
            "-s",
            f"paths.reports_dir={tmp_path.as_posix()}",
        ],
    )
    assert res.exit_code == 0, res.output
    tab = pd.read_csv(tmp_path / "tables" / "site_effect.csv").set_index("harmonization")
    assert tab.loc["none", "site_balanced_accuracy"] > tab.loc["combat", "site_balanced_accuracy"]
    assert tab.loc["combat", "r2_site"] < 0.05
    assert (tmp_path / "figures" / "fig2_site_effect.png").is_file()
