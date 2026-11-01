import math

import numpy as np
import pytest

from shiftguard.conformal import (
    ClassConditionalConformal,
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    clip_weights,
    conformal_quantile,
    effective_sample_size,
    summarize_sets,
    weight_clip_bounds,
)
from shiftguard.conformal.sets import NO_DECISION, abstained, covered, decisions, set_sizes


def _binary(rng, n, slope=2.0, offset=0.0):
    """Correctly specified 1-D logistic data: (probs, labels)."""
    x = rng.normal(size=n)
    p1 = 1 / (1 + np.exp(-(slope * x + offset)))
    y = (rng.uniform(size=n) < p1).astype(int)
    return np.c_[1 - p1, p1], y


# ----------------------------------------------------------------------------- sets
def test_set_summaries_hand_example():
    sets = np.array([[True, False], [True, True], [False, False], [False, True]])
    labels = np.array([0, 1, 0, 0])
    np.testing.assert_array_equal(covered(sets, labels), [True, True, False, False])
    np.testing.assert_array_equal(set_sizes(sets), [1, 2, 0, 1])
    np.testing.assert_array_equal(abstained(sets), [False, True, True, False])
    np.testing.assert_array_equal(decisions(sets), [0, NO_DECISION, NO_DECISION, 1])
    summary = summarize_sets(sets, labels)
    assert summary["coverage"] == 0.5
    assert summary["mean_set_size"] == 1.0
    assert summary["empty_rate"] == 0.25
    assert summary["full_rate"] == 0.25
    assert summary["abstention_rate"] == 0.5
    assert summary["n_decided"] == 2
    assert summary["accuracy_decided"] == 0.5


def test_summary_all_abstained_gives_nan_accuracy():
    summary = summarize_sets(np.ones((3, 2), dtype=bool), [0, 1, 0])
    assert summary["coverage"] == 1.0
    assert math.isnan(summary["accuracy_decided"])


def test_sets_must_be_boolean():
    with pytest.raises(ValueError):
        set_sizes(np.ones((2, 2)))


# ----------------------------------------------------------------------------- split
def test_split_quantile_and_sets():
    rng = np.random.default_rng(0)
    p, y = _binary(rng, 300)
    cp = SplitConformal(alpha=0.1).fit(p, y)
    assert cp.qhat_ == conformal_quantile(1 - p[np.arange(300), y], 0.1)
    assert cp.n_calibration_ == 300
    assert cp.target_coverage == pytest.approx(0.9)
    sets = cp.predict_sets(p)
    np.testing.assert_array_equal(sets, (1 - p) <= cp.qhat_)


def test_split_tiny_calibration_gives_full_sets():
    p = np.array([[0.9, 0.1]] * 5)
    cp = SplitConformal(alpha=0.1).fit(p, [0] * 5)
    assert math.isinf(cp.qhat_)
    assert cp.predict_sets(p).all()


def test_split_errors():
    cp = SplitConformal()
    with pytest.raises(RuntimeError, match="not fitted"):
        cp.predict_sets([[0.5, 0.5]])
    cp.fit([[0.5, 0.5]] * 20, [0] * 20)
    with pytest.raises(ValueError, match="classes"):
        cp.predict_sets([[0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match="unknown score"):
        SplitConformal(score="raps")  # type: ignore[arg-type]


def test_randomized_aps_is_reproducible():
    rng = np.random.default_rng(3)
    p, y = _binary(rng, 200)
    a = SplitConformal(score="aps", randomized=True, random_state=7).fit(p, y).predict_sets(p)
    b = SplitConformal(score="aps", randomized=True, random_state=7).fit(p, y).predict_sets(p)
    np.testing.assert_array_equal(a, b)


def test_binary_deterministic_aps_degenerates_when_error_rate_exceeds_alpha():
    # In binary problems the non-top label always has deterministic APS score 1. If more
    # than alpha of the calibration labels are not the model's top class, q_hat = 1 and
    # every set is full. This is why randomised APS (or LAC) is used for binary tasks.
    p = np.array([[0.8, 0.2]] * 100)
    y = np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)]  # 20% "errors" > alpha
    cp = SplitConformal(alpha=0.1, score="aps").fit(p, y)
    assert cp.qhat_ == 1.0
    assert cp.predict_sets(p).all()
    assert SplitConformal(alpha=0.1, score="lac").fit(p, y).predict_sets(p).all()


# ----------------------------------------------------------------------------- class-conditional
def test_class_conditional_quantiles_match_subsets():
    rng = np.random.default_rng(1)
    p, y = _binary(rng, 400, offset=-1.5)
    cp = ClassConditionalConformal(alpha=0.1).fit(p, y)
    scores = 1 - p[np.arange(400), y]
    for k in (0, 1):
        assert cp.qhat_[k] == conformal_quantile(scores[y == k], 0.1)
    np.testing.assert_array_equal(cp.n_per_class_, np.bincount(y))
    np.testing.assert_array_equal(cp.predict_sets(p), (1 - p) <= cp.qhat_[None, :])


def test_class_conditional_fixes_minority_undercoverage():
    # Imbalanced data (~10% positives). Marginal split CP under-covers the minority class;
    # class-conditional CP covers each class at the target level.
    rng = np.random.default_rng(2)
    p_cal, y_cal = _binary(rng, 5000, offset=-3.0)
    p_te, y_te = _binary(rng, 50000, offset=-3.0)
    minority = y_te == 1
    split_cov = SplitConformal(0.1).fit(p_cal, y_cal).predict_sets(p_te)[minority, 1].mean()
    cc_sets = ClassConditionalConformal(0.1).fit(p_cal, y_cal).predict_sets(p_te)
    assert split_cov < 0.8
    assert cc_sets[minority, 1].mean() >= 0.88
    assert cc_sets[~minority, 0].mean() >= 0.88


# ----------------------------------------------------------------------------- mondrian
def test_mondrian_per_group_quantiles_and_unseen_policy():
    rng = np.random.default_rng(4)
    p, y = _binary(rng, 300)
    g = np.array(["A"] * 150 + ["B"] * 145 + ["C"] * 5)
    cp = MondrianConformal(alpha=0.1).fit(p, y, g)
    scores = 1 - p[np.arange(300), y]
    for group in ("A", "B"):
        assert cp.qhat_[group] == conformal_quantile(scores[g == group], 0.1)
    assert math.isinf(cp.qhat_["C"])  # 5 < min_calibration_size(0.1) = 9
    assert cp.n_per_group_ == {"A": 150, "B": 145, "C": 5}

    test_g = np.array(["A", "C", "Z"])
    sets = cp.predict_sets(p[:3], test_g)
    assert sets[1].all() and sets[2].all()  # small group and unseen group: full sets
    np.testing.assert_array_equal(sets[0], (1 - p[0]) <= cp.qhat_["A"])

    strict = MondrianConformal(alpha=0.1, unseen="error").fit(p, y, g)
    with pytest.raises(KeyError, match="Z"):
        strict.predict_sets(p[:3], test_g)
    with pytest.raises(ValueError):
        MondrianConformal(unseen="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="groups"):
        cp.predict_sets(p[:3], ["A"])


def test_mondrian_protects_a_noisy_group():
    # Group B has 30% label noise the model ignores; pooled split CP under-covers B,
    # Mondrian CP covers both groups.
    rng = np.random.default_rng(5)

    def sample(n):
        p, y = _binary(rng, n, slope=4.0)
        g = rng.choice(["A", "B"], size=n)
        flip = (g == "B") & (rng.uniform(size=n) < 0.3)
        return p, np.where(flip, 1 - y, y), g

    p_cal, y_cal, g_cal = sample(6000)
    p_te, y_te, g_te = sample(40000)
    idx = np.arange(len(y_te))
    split_sets = SplitConformal(0.1).fit(p_cal, y_cal).predict_sets(p_te)
    mond_sets = MondrianConformal(0.1).fit(p_cal, y_cal, g_cal).predict_sets(p_te, g_te)
    b = g_te == "B"
    assert split_sets[idx, y_te][b].mean() < 0.85
    for mask in (b, ~b):
        assert mond_sets[idx, y_te][mask].mean() >= 0.88


# ----------------------------------------------------------------------------- weighted
def test_ess_and_clipping_helpers():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert effective_sample_size([0, 0, 5, 0]) == pytest.approx(1)
    assert effective_sample_size(np.zeros(3)) == 0.0
    with pytest.raises(ValueError):
        effective_sample_size([-1.0, 1.0])
    lo, hi = weight_clip_bounds(np.arange(101, dtype=float), (0.1, 0.9))
    assert (lo, hi) == (10.0, 90.0)
    np.testing.assert_array_equal(clip_weights([0, 50, 100], lo, hi), [10, 50, 90])
    with pytest.raises(ValueError):
        weight_clip_bounds([1.0], (0.9, 0.1))
    with pytest.raises(ValueError):
        clip_weights([1.0], 2.0, 1.0)


def test_weighted_with_unit_weights_equals_split():
    rng = np.random.default_rng(6)
    p, y = _binary(rng, 250)
    p_te, _ = _binary(rng, 100)
    split = SplitConformal(0.1).fit(p, y).predict_sets(p_te)
    weighted = WeightedConformal(0.1).fit(p, y, np.ones(250))
    np.testing.assert_array_equal(weighted.predict_sets(p_te, np.ones(100)), split)
    assert weighted.ess_ == pytest.approx(250)


def test_weighted_clipping_applies_to_test_weights():
    rng = np.random.default_rng(7)
    p, y = _binary(rng, 200)
    w = rng.exponential(size=200)
    cp = WeightedConformal(0.1, clip_quantiles=(0.05, 0.95)).fit(p, y, w)
    lo, hi = cp.clip_bounds_
    assert cp.weights_.min() >= lo and cp.weights_.max() <= hi
    # an extreme test weight is clipped to `hi`, so it gives the same quantile as `hi`
    np.testing.assert_array_equal(cp.quantiles([1e9]), cp.quantiles([hi]))
    assert cp.ess_ > effective_sample_size(w)


def test_weighted_validation():
    p = np.array([[0.5, 0.5]] * 10)
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="shape"):
        WeightedConformal().fit(p, y, np.ones(9))
    with pytest.raises(ValueError, match="non-negative"):
        WeightedConformal().fit(p, y, -np.ones(10))
    cp = WeightedConformal().fit(p, y, np.ones(10))
    with pytest.raises(ValueError, match="one test weight"):
        cp.predict_sets(p[:2], np.ones(3))
    with pytest.raises(RuntimeError):
        WeightedConformal().quantiles([1.0])
