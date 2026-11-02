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


def test_two_stage_bootstrap_adds_within_site_noise():
    from shiftguard.evaluation.statistics import paired_bootstrap, two_stage_paired_bootstrap

    rng = np.random.default_rng(0)
    site_effect = rng.normal(0.05, 0.02, size=12)
    # no within-site noise: subjects of a site all equal -> same as the site bootstrap
    flat_x = [np.full(30, 0.9 + d) for d in site_effect]
    flat_y = [np.full(30, 0.9) for _ in site_effect]
    two = two_stage_paired_bootstrap(flat_x, flat_y, n_boot=1000, seed=1)
    one = paired_bootstrap(0.9 + site_effect, np.full(12, 0.9), n_boot=1000, seed=1)
    assert two["diff"] == pytest.approx(one["diff"])
    assert two["ci_high"] - two["ci_low"] == pytest.approx(one["ci_high"] - one["ci_low"], rel=0.2)
    # binary subject outcomes: the two-stage interval is wider
    noisy_x = [rng.binomial(1, 0.9, size=30).astype(float) for _ in site_effect]
    noisy_y = [rng.binomial(1, 0.85, size=30).astype(float) for _ in site_effect]
    site_means = (np.array([x.mean() for x in noisy_x]), np.array([y.mean() for y in noisy_y]))
    w1 = paired_bootstrap(*site_means, n_boot=1000, seed=2)
    w2 = two_stage_paired_bootstrap(noisy_x, noisy_y, n_boot=1000, seed=2)
    assert w2["diff"] == pytest.approx(w1["diff"])
    assert (w2["ci_high"] - w2["ci_low"]) > (w1["ci_high"] - w1["ci_low"])


def test_two_stage_bootstrap_transform_and_errors():
    from shiftguard.evaluation.statistics import two_stage_paired_bootstrap

    x = [np.array([1.0, 1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])]
    y = [np.array([0.0, 1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0, 0.0])]
    short = two_stage_paired_bootstrap(x, y, lambda v: np.maximum(0.0, 0.9 - v), n_boot=200)
    # shortfalls: x (0.15, 0), y (0.4, 0.15) -> mean difference -0.2
    assert short["diff"] == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        two_stage_paired_bootstrap([np.ones(3)], [np.ones(3)])
    with pytest.raises(ValueError):
        two_stage_paired_bootstrap([np.ones(3), np.ones(2)], [np.ones(3), np.ones(3)])
