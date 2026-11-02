import numpy as np
import pytest
from scipy import stats

from shiftguard.evaluation.statistics import binomial_test_below, clopper_pearson


@pytest.mark.parametrize(("k", "n"), [(0, 10), (10, 10), (7, 10), (181, 201), (450, 500)])
def test_clopper_pearson_matches_scipy(k, n):
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    lo, hi = clopper_pearson(k, n, 0.95)
    assert lo == pytest.approx(ref.low)
    assert hi == pytest.approx(ref.high)


def test_clopper_pearson_edges_and_errors():
    assert clopper_pearson(0, 5)[0] == 0.0
    assert clopper_pearson(5, 5)[1] == 1.0
    for k, n in ((1, 0), (6, 5), (-1, 5)):
        with pytest.raises(ValueError):
            clopper_pearson(k, n)
    with pytest.raises(ValueError):
        clopper_pearson(1, 5, level=1.0)


def test_binomial_test_below():
    assert binomial_test_below(100, 100, 0.9) == pytest.approx(1.0)
    assert binomial_test_below(70, 100, 0.9) < 1e-6
    assert binomial_test_below(90, 100, 0.9) == pytest.approx(stats.binom.cdf(90, 100, 0.9))
    with pytest.raises(ValueError):
        binomial_test_below(5, 4, 0.9)


# ----------------------------------------------------------------------------- phase 5
from shiftguard.evaluation.statistics import (  # noqa: E402
    benjamini_hochberg,
    beta_binomial_eb,
    paired_bootstrap,
)


def test_benjamini_hochberg_matches_hand_computation():
    p = [0.01, 0.04, 0.03, 0.20, np.nan]
    q = benjamini_hochberg(p)
    # sorted 0.01, 0.03, 0.04, 0.20 -> 0.04, 0.0533, 0.0533, 0.20 (step-up monotone)
    np.testing.assert_allclose(q[:4], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20])
    assert np.isnan(q[4])
    assert np.isnan(benjamini_hochberg([np.nan])).all()


def test_beta_binomial_detects_heterogeneity_or_its_absence():
    rng = np.random.default_rng(0)
    n = np.full(30, 60)
    homo = beta_binomial_eb(rng.binomial(n, 0.9), n)
    assert homo.mean == pytest.approx(0.9, abs=0.02)
    assert homo.between_site_sd < 0.03
    theta = rng.beta(9, 1.5, size=30)  # mean 0.857, sd ~0.10
    het = beta_binomial_eb(rng.binomial(n, theta), n)
    assert 0.06 < het.between_site_sd < 0.16
    # shrinkage pulls extreme sites towards the mean
    k = rng.binomial(n, theta)
    fit = beta_binomial_eb(k, n)
    raw = k / n
    assert np.all(np.abs(fit.shrunk - fit.mean) <= np.abs(raw - fit.mean) + 1e-12)
    with pytest.raises(ValueError):
        beta_binomial_eb([5], [10])


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.7, 0.95, size=18)
    better = base + 0.05 + rng.normal(scale=0.005, size=18)
    res = paired_bootstrap(better, base, n_boot=1000)
    assert res["diff"] == pytest.approx(0.05, abs=0.01)
    assert res["ci_low"] > 0 and res["p_value"] < 0.01
    same = paired_bootstrap(base + rng.normal(scale=0.01, size=18), base, n_boot=1000)
    assert same["ci_low"] < 0 < same["ci_high"]
