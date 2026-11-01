import math

import numpy as np
import pytest

from shiftguard.conformal.quantiles import (
    check_alpha,
    conformal_quantile,
    conformal_rank,
    min_calibration_size,
    weighted_conformal_quantile,
)


@pytest.mark.parametrize(
    ("n", "alpha", "k"),
    [
        (10, 0.1, 10),  # ceil(9.9)
        (10, 0.2, 9),  # ceil(8.8)
        (99, 0.1, 90),  # 0.9 * 100 = 90.00000000000001 in floating point: must stay 90
        (199, 0.1, 180),
        (200, 0.1, 181),  # ceil(180.9)
        (8, 0.1, 9),  # > n -> infinite quantile
    ],
)
def test_conformal_rank(n, alpha, k):
    assert conformal_rank(n, alpha) == k


@pytest.mark.parametrize(("alpha", "n_min"), [(0.1, 9), (0.05, 19), (0.2, 4), (0.5, 1)])
def test_min_calibration_size(alpha, n_min):
    assert min_calibration_size(alpha) == n_min
    assert math.isfinite(conformal_quantile(np.arange(n_min, dtype=float), alpha))
    if n_min > 1:
        assert math.isinf(conformal_quantile(np.arange(n_min - 1, dtype=float), alpha))


def test_conformal_quantile_hand_examples():
    s = np.arange(1, 11, dtype=float)[::-1]  # order must not matter
    assert conformal_quantile(s, 0.1) == 10
    assert conformal_quantile(s, 0.2) == 9
    assert conformal_quantile(s, 0.5) == 6  # ceil(5.5) = 6
    assert math.isinf(conformal_quantile([], 0.1))


def test_conformal_quantile_rejects_bad_input():
    with pytest.raises(ValueError):
        conformal_quantile([0.1, np.inf], 0.1)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            check_alpha(bad)


def test_weighted_hand_examples():
    s = [1.0, 2.0, 3.0]
    assert weighted_conformal_quantile(s, [1, 1, 1], [1], 0.25)[0] == 3  # threshold 3
    assert weighted_conformal_quantile(s, [1, 1, 1], [1], 0.5)[0] == 2  # threshold 2
    assert weighted_conformal_quantile(s, [3, 1, 1], [1], 0.5)[0] == 1  # mass 3/6 at s=1
    # test point carries most of the mass -> infinite quantile
    assert math.isinf(weighted_conformal_quantile(s, [1, 1, 1], [10], 0.5)[0])
    # zero calibration weight -> calibration mass never reaches 1 - alpha
    assert math.isinf(weighted_conformal_quantile(s, [0, 0, 0], [1], 0.1)[0])
    assert math.isinf(weighted_conformal_quantile([], [], [1.0], 0.1)[0])


@pytest.mark.parametrize("n", [1, 5, 9, 10, 37, 99, 100, 500])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.33])
def test_unit_weights_reduce_to_split_quantile(n, alpha):
    rng = np.random.default_rng(n)
    s = rng.uniform(size=n)
    q = weighted_conformal_quantile(s, np.ones(n), np.ones(3), alpha)
    assert np.all(q == conformal_quantile(s, alpha))


def test_weighted_quantile_scale_invariance_and_vectorisation():
    rng = np.random.default_rng(0)
    s, w, wt = rng.uniform(size=200), rng.exponential(size=200), rng.exponential(size=50)
    q = weighted_conformal_quantile(s, w, wt, 0.1)
    np.testing.assert_array_equal(q, weighted_conformal_quantile(s, 7.5 * w, 7.5 * wt, 0.1))
    single = [weighted_conformal_quantile(s, w, [x], 0.1)[0] for x in wt]
    np.testing.assert_array_equal(q, single)
    # a heavier test point never lowers its quantile
    order = np.argsort(wt)
    assert np.all(np.diff(q[order]) >= 0)


def test_weighted_quantile_validation():
    with pytest.raises(ValueError, match="differ"):
        weighted_conformal_quantile([1.0, 2.0], [1.0], [1.0], 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        weighted_conformal_quantile([1.0], [-1.0], [1.0], 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        weighted_conformal_quantile([1.0], [1.0], [np.nan], 0.1)
    with pytest.raises(ValueError, match="positive"):
        weighted_conformal_quantile([1.0], [0.0], [0.0], 0.1)
