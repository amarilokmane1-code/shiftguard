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
