"""Label shift: EM prior estimation and label-shift-weighted conformal prediction."""

import numpy as np
import pytest
from scipy.stats import norm

from shiftguard.conformal import ClassConditionalConformal, LabelShiftConformal, SplitConformal
from shiftguard.conformal.sets import covered
from shiftguard.shift.label_shift import (
    adjust_to_prior,
    em_prior_shift,
    label_shift_weights,
)

MU = 1.0  # x | y ~ N(+-MU, 1): exact Bayes posteriors are known


def _sample(rng, n, prior1):
    y = (rng.uniform(size=n) < prior1).astype(int)
    x = rng.normal(loc=np.where(y == 1, MU, -MU))
    return x, y


def _posterior(x, prior1):
    l1, l0 = norm.pdf(x, MU), norm.pdf(x, -MU)
    p1 = prior1 * l1 / (prior1 * l1 + (1 - prior1) * l0)
    return np.c_[1 - p1, p1]


def test_adjust_to_prior_matches_bayes():
    x = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(
        adjust_to_prior(_posterior(x, 0.5), [0.5, 0.5], [0.2, 0.8]), _posterior(x, 0.8)
    )
    with pytest.raises(ValueError):
        adjust_to_prior(_posterior(x, 0.5), [0.5, 0.5], [0.9, 0.9])


def test_em_recovers_target_prior():
    rng = np.random.default_rng(0)
    x, _ = _sample(rng, 20000, 0.75)
    est = em_prior_shift(_posterior(x, 0.4), [0.6, 0.4])
    assert est.converged
    assert est.target_prior[1] == pytest.approx(0.75, abs=0.02)
    np.testing.assert_allclose(est.adjusted_probs.sum(axis=1), 1.0)
    # no shift: EM stays at the source prior
    x0, _ = _sample(rng, 20000, 0.4)
    assert em_prior_shift(_posterior(x0, 0.4), [0.6, 0.4]).target_prior[1] == pytest.approx(
        0.4, abs=0.02
    )


def test_label_shift_weights():
    np.testing.assert_allclose(label_shift_weights([0.5, 0.5], [0.25, 0.75]), [0.5, 1.5])


def _coverage(method, n_trials=600, n_cal=200, n_test=200, seed=1):
    """Calibrate on source prior 0.3, test on target prior 0.8 (strong label shift)."""
    rng = np.random.default_rng(seed)
    src, tgt = 0.3, 0.8
    cov, cov_class = [], {0: [], 1: []}
    for _ in range(n_trials):
        xc, yc = _sample(rng, n_cal, src)
        xt, yt = _sample(rng, n_test, tgt)
        pc, pt = _posterior(xc, src), _posterior(xt, src)
        if method == "split":
            sets = SplitConformal(0.1).fit(pc, yc).predict_sets(pt)
        elif method == "classcond":
            sets = ClassConditionalConformal(0.1).fit(pc, yc).predict_sets(pt)
        else:
            w = label_shift_weights([1 - src, src], [1 - tgt, tgt])
            sets = LabelShiftConformal(0.1).fit(pc, yc, w).predict_sets(pt)
        c = covered(sets, yt)
        cov.append(c.mean())
        for k in (0, 1):
            cov_class[k].append(c[yt == k].mean())
    return np.mean(cov), {k: np.mean(v) for k, v in cov_class.items()}


def test_split_cp_undercovers_under_label_shift_and_corrections_restore_it():
    split, split_class = _coverage("split")
    weighted, _ = _coverage("label_shift")
    _classcond, cc_class = _coverage("classcond")
    assert split < 0.87  # the up-weighted minority class is under-covered
    assert split_class[1] < 0.85
    assert weighted >= 0.895  # marginal guarantee restored (oracle priors)
    assert min(cc_class.values()) >= 0.895  # per-class guarantee, no weights needed


def test_label_shift_conformal_validation_and_ess():
    p = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    m = LabelShiftConformal(0.1).fit(p, [0, 1, 0], [1.0, 1.0])
    assert m.thresholds_.shape == (2,)
    assert m.ess_ == pytest.approx(3.0)
    with pytest.raises(ValueError):
        LabelShiftConformal(0.1).fit(p, [0, 1, 0], [1.0, 0.0])
    with pytest.raises(RuntimeError):
        LabelShiftConformal(0.1).predict_sets(p)


def test_calibration_offset_identity_under_pure_label_shift():
    """mean p_s(1|x) - pi_t = -(pi_t - pi_s) * (1 - slope) for calibrated source posteriors."""
    rng = np.random.default_rng(5)
    src, tgt = 0.45, 0.70
    xs, ys = _sample(rng, 200_000, src)
    xt, yt = _sample(rng, 200_000, tgt)
    ps, pt = _posterior(xs, src)[:, 1], _posterior(xt, src)[:, 1]
    slope = ps[ys == 1].mean() - ps[ys == 0].mean()
    observed = pt.mean() - yt.mean()
    predicted = -(yt.mean() - ys.mean()) * (1 - slope)
    assert observed == pytest.approx(predicted, abs=0.004)
    assert abs(observed) <= abs(yt.mean() - ys.mean())  # never larger than the prior shift
