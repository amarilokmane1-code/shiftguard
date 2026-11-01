import numpy as np
import pytest

from shiftguard.conformal.scores import (
    aps_score_matrix,
    lac_score_matrix,
    score_matrix,
    true_label_scores,
    validate_labels,
    validate_probs,
)


def test_lac_values():
    np.testing.assert_allclose(lac_score_matrix([[0.7, 0.3], [0.1, 0.9]]), [[0.3, 0.7], [0.9, 0.1]])


def test_aps_deterministic_hand_example():
    # sorted order already: mass above = [0, .5, .8], own = [.5, .3, .2]
    np.testing.assert_allclose(aps_score_matrix([[0.5, 0.3, 0.2]]), [[0.5, 0.8, 1.0]])
    # same probabilities, permuted classes: scores follow the classes
    np.testing.assert_allclose(aps_score_matrix([[0.2, 0.5, 0.3]]), [[1.0, 0.5, 0.8]])


def test_aps_randomized_hand_example():
    np.testing.assert_allclose(aps_score_matrix([[0.5, 0.3, 0.2]], u=[0.5]), [[0.25, 0.65, 0.9]])


def test_aps_ties_broken_by_class_index():
    np.testing.assert_allclose(aps_score_matrix([[0.5, 0.5]]), [[0.5, 1.0]])


def test_aps_properties_random():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=500)
    s = aps_score_matrix(p)
    assert s.min() >= 0 and s.max() <= 1
    # the most probable class has score equal to its own probability
    top = p.argmax(axis=1)
    np.testing.assert_allclose(s[np.arange(500), top], p.max(axis=1))
    # the least probable class always has score 1
    np.testing.assert_allclose(s[np.arange(500), p.argmin(axis=1)], 1.0)
    # randomised scores never exceed deterministic ones
    assert np.all(aps_score_matrix(p, u=rng.uniform(size=500)) <= s + 1e-12)


def test_score_dispatch_and_errors():
    p = np.array([[0.6, 0.4]])
    np.testing.assert_allclose(score_matrix(p, "lac"), lac_score_matrix(p))
    np.testing.assert_allclose(score_matrix(p, "aps"), aps_score_matrix(p))
    out = score_matrix(p, "aps", randomized=True, rng=np.random.default_rng(1))
    assert out.shape == (1, 2)
    with pytest.raises(ValueError, match="Generator"):
        score_matrix(p, "aps", randomized=True)
    with pytest.raises(ValueError, match="unknown score"):
        score_matrix(p, "raps")  # type: ignore[arg-type]


def test_true_label_scores():
    s = np.array([[0.1, 0.9], [0.8, 0.2]])
    np.testing.assert_allclose(true_label_scores(s, [1, 0]), [0.9, 0.8])


@pytest.mark.parametrize(
    "bad",
    [
        [[0.5, 0.6]],  # row sum
        [[1.2, -0.2]],  # range
        [[np.nan, 1.0]],  # non-finite
        [0.5, 0.5],  # 1-D
        [[1.0]],  # single class
    ],
)
def test_validate_probs_rejects(bad):
    with pytest.raises(ValueError):
        validate_probs(bad)


def test_validate_labels():
    np.testing.assert_array_equal(validate_labels([0.0, 1.0], 2, 2), [0, 1])
    with pytest.raises(ValueError):
        validate_labels([0, 2], 2, 2)
    with pytest.raises(ValueError):
        validate_labels([0.5, 1], 2, 2)
    with pytest.raises(ValueError):
        validate_labels([0, 1, 1], 2, 2)
