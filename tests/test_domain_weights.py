"""Domain-classifier weights for one LOSO fold (PCA + CV logistic regression)."""

import numpy as np
from scipy.stats import spearmanr

from shiftguard.shift.domain import domain_classifier_weights


def _gaussians(rng, n_s, n_c, n_t, p, shift):
    mu = np.zeros(p)
    mu[:3] = shift
    return rng.normal(size=(n_s, p)), rng.normal(size=(n_c, p)), rng.normal(size=(n_t, p)) + mu


def test_weights_track_the_true_density_ratio():
    rng = np.random.default_rng(0)
    Xs, Xc, Xt = _gaussians(rng, 400, 200, 150, 30, 0.6)
    dw = domain_classifier_weights(Xs, Xc, Xt, components_grid=(5, 30), random_state=0)
    mu = np.zeros(30)
    mu[:3] = 0.6
    true_cal = np.exp(Xc @ mu - mu @ mu / 2)  # N(mu, I) / N(0, I)
    assert spearmanr(dw.cal_weights, true_cal).statistic > 0.7
    assert 0.65 < dw.auc < 0.95
    assert dw.test_weights.shape == (150,) and np.all(dw.test_weights > 0)
    # weights estimate a density ratio: their source mean is close to 1
    assert 0.6 < dw.cal_weights.mean() < 1.6


def test_no_shift_gives_flat_weights_and_chance_auc():
    rng = np.random.default_rng(1)
    Xs, Xc, Xt = _gaussians(rng, 400, 200, 150, 30, 0.0)
    dw = domain_classifier_weights(Xs, Xc, Xt, components_grid=(5, 10), random_state=0)
    assert dw.auc < 0.62
    assert 0.5 < np.median(dw.cal_weights) < 1.5


def test_separable_sites_give_extreme_weights():
    rng = np.random.default_rng(2)
    Xs, Xc, Xt = _gaussians(rng, 300, 100, 80, 20, 4.0)
    dw = domain_classifier_weights(Xs, Xc, Xt, components_grid=(5,), random_state=0)
    assert dw.auc > 0.99
    assert dw.n_components == 5
    assert np.median(dw.cal_weights) < 0.05  # source points look nothing like the target
