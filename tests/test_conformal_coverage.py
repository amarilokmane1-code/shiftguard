"""Synthetic validity tests of the conformal core.

Each trial draws a fresh calibration set and one test point, so the number of covered
trials is exactly Binomial(n_trials, coverage). Every simulation is seeded, so the tests
are deterministic. The acceptance regions are still set at the 0.1% level, so that a
correct implementation passes for any seed with probability >= 99.9%.
"""

import pytest
from scipy import stats

from shiftguard.conformal.quantiles import conformal_rank
from shiftguard.evaluation.simulation import simulate_covariate_shift, simulate_exchangeable

ALPHA = 0.1
N_TRIALS = 3000
N_CAL = 200
LEVEL = 1e-3  # two-sided acceptance region: 0.05% in each tail


# ------------------------------------------------------------------ (a) exchangeable
@pytest.mark.parametrize("n_cal", [19, N_CAL])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize(("score", "randomized"), [("lac", False), ("aps", True)])
def test_exchangeable_coverage_matches_theory_exactly(n_cal, n_classes, score, randomized):
    """Continuous scores: coverage is exactly k / (n + 1); both tails are tested.

    n_cal = 19 mimics a small site and makes an off-by-one error in the quantile rank
    detectable (18/20 = 0.90 vs 17/20 = 0.85); n_cal = 200 gives 181/201 = 0.9005.
    """
    res = simulate_exchangeable(
        n_classes=n_classes,
        n_calibration=n_cal,
        n_trials=N_TRIALS,
        alpha=ALPHA,
        score=score,
        randomized=randomized,
    )
    p_exact = conformal_rank(n_cal, ALPHA) / (n_cal + 1)
    lo = stats.binom.ppf(LEVEL / 2, N_TRIALS, p_exact)
    hi = stats.binom.ppf(1 - LEVEL / 2, N_TRIALS, p_exact)
    assert lo <= res.n_covered <= hi, (res.coverage, p_exact)
    assert res.p_below_target > LEVEL


def test_exchangeable_deterministic_aps_is_valid_but_conservative():
    """Deterministic APS has ties (score 1), so only the lower bound applies."""
    res = simulate_exchangeable(
        n_classes=3, n_calibration=N_CAL, n_trials=N_TRIALS, alpha=ALPHA, score="aps"
    )
    assert res.n_covered >= stats.binom.ppf(LEVEL, N_TRIALS, 1 - ALPHA)


# ------------------------------------------------------------------ (b) covariate shift
@pytest.fixture(scope="module")
def shift_results():
    results = simulate_covariate_shift(n_calibration=1000, n_trials=N_TRIALS, alpha=ALPHA)
    return {r.method: r for r in results}


def test_split_is_valid_on_the_source_distribution(shift_results):
    assert shift_results["split (source test)"].p_below_target > LEVEL


def test_covariate_shift_breaks_split_conformal(shift_results):
    res = shift_results["split"]
    assert res.p_below_target < 1e-6
    assert res.ci_high < 1 - ALPHA - 0.03


@pytest.mark.parametrize(
    "method",
    ["weighted (oracle w)", "weighted (estimated w)", "weighted (estimated w, clipped)"],
)
def test_weighted_conformal_restores_coverage(shift_results, method):
    res = shift_results[method]
    # not significantly below the target ...
    assert res.p_below_target > LEVEL
    # ... and not trivially valid through full sets (more informative than "always both")
    assert res.coverage < 1 - ALPHA + 0.03
    assert res.mean_set_size < 1.8
    # a meaningful effective sample size is reported
    assert 1 <= res.mean_ess <= res.n_calibration


def test_weighted_beats_split_under_shift(shift_results):
    assert shift_results["weighted (estimated w)"].ci_low > shift_results["split"].ci_high
