import numpy as np
import pytest

from shiftguard.shift.weights import DomainClassifierRatio


def test_recovers_gaussian_mean_shift_ratio():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -0.5, 0.0])
    Xs = rng.normal(size=(4000, 3))
    Xt = rng.normal(size=(2000, 3)) + mu  # unequal sizes exercise the prior correction
    model = DomainClassifierRatio().fit(Xs, Xt)

    X_eval = rng.normal(size=(2000, 3))
    true_log = X_eval @ mu - mu @ mu / 2
    est_log = np.log(model.ratio(X_eval))
    assert np.corrcoef(true_log, est_log)[0, 1] > 0.99
    assert np.median(np.abs(est_log - true_log)) < 0.1
    # E_source[w(X)] = 1 for a true density ratio
    assert model.ratio(X_eval).mean() == pytest.approx(1.0, abs=0.1)


def test_no_shift_gives_ratio_near_one():
    rng = np.random.default_rng(1)
    model = DomainClassifierRatio().fit(rng.normal(size=(3000, 2)), rng.normal(size=(3000, 2)))
    w = model.ratio(rng.normal(size=(500, 2)))
    assert np.all(np.abs(np.log(w)) < 0.2)


def test_errors():
    with pytest.raises(ValueError):
        DomainClassifierRatio(prob_clip=0.6)
    with pytest.raises(RuntimeError):
        DomainClassifierRatio().ratio(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        DomainClassifierRatio().fit(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        DomainClassifierRatio().fit(np.zeros((0, 2)), np.zeros((3, 2)))
