import numpy as np
import pytest

from shiftguard.features.connectivity import (
    ConnectivityFeatures,
    fisher_z,
    fit_transform_fold,
    n_edges,
    subjectwise_features,
)


@pytest.fixture
def ts():
    rng = np.random.default_rng(0)
    mixing = rng.normal(size=(6, 6))
    return [
        rng.normal(size=(120, 6)) @ (mixing + rng.normal(scale=0.5, size=(6, 6))) for _ in range(12)
    ]


def test_correlation_features_match_numpy(ts):
    X = subjectwise_features(ts, cov_estimator="empirical")
    assert X.shape == (12, n_edges(6)) == (12, 15)
    iu = np.triu_indices(6, k=1)
    r = np.corrcoef(ts[3], rowvar=False)
    # nilearn vectorises the lower triangle row by row: compare as sets of values
    np.testing.assert_allclose(np.sort(X[3]), np.sort(np.arctanh(r[iu])), atol=1e-8)


def test_fisher_z_is_clipped_and_finite():
    assert np.isfinite(fisher_z(np.array([-1.0, 0.0, 1.0]))).all()
    assert fisher_z(np.array([0.5]))[0] == pytest.approx(np.arctanh(0.5))


def test_fisher_z_flag(ts):
    raw = subjectwise_features(ts, fisher_z=False)
    np.testing.assert_allclose(subjectwise_features(ts), fisher_z(raw))
    assert np.abs(raw).max() < 1


def test_subjectwise_features_are_independent_of_the_cohort(ts):
    full = subjectwise_features(ts)
    np.testing.assert_allclose(subjectwise_features(ts[:3]), full[:3])


def test_tangent_cannot_be_precomputed(ts):
    with pytest.raises(ValueError, match="inside each fold"):
        subjectwise_features(ts, kind="tangent")


def test_tangent_reference_is_fitted_on_training_subjects_only(ts):
    train, test = np.arange(0, 8), np.arange(8, 12)
    X_train, (X_test,) = fit_transform_fold(ts, train, [test], kind="tangent")
    # Test features depend only on the training fit, not on which other test subjects exist
    t = ConnectivityFeatures("tangent").fit([ts[i] for i in train])
    np.testing.assert_allclose(t.transform([ts[8]])[0], X_test[0])
    assert t.n_fit_subjects_ == 8
    # ... and fitting the reference on train + test would change them: the guard matters
    leaky = ConnectivityFeatures("tangent").fit(ts).transform([ts[i] for i in test])
    assert not np.allclose(leaky, X_test)
    assert X_train.shape == (8, 15)


def test_fit_transform_fold_rejects_overlap(ts):
    with pytest.raises(ValueError, match="overlaps"):
        fit_transform_fold(ts, np.arange(0, 8), [np.arange(9, 12), np.arange(7, 9)])


def test_transformer_errors(ts):
    with pytest.raises(RuntimeError):
        ConnectivityFeatures().transform(ts)
    with pytest.raises(ValueError, match="kind"):
        ConnectivityFeatures(kind="covariance").fit(ts)
    with pytest.raises(ValueError, match="cov_estimator"):
        ConnectivityFeatures(cov_estimator="oas").fit(ts)
    fitted = ConnectivityFeatures().fit(ts)
    with pytest.raises(ValueError, match="ROIs"):
        fitted.transform([np.ones((50, 4))])
    with pytest.raises(ValueError):
        ConnectivityFeatures().fit([])
    with pytest.raises(ValueError):
        ConnectivityFeatures().fit([ts[0], ts[1][:, :3]])
