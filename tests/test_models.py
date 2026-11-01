import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from shiftguard.evaluation.metrics import binary_metrics
from shiftguard.models import make_model
from shiftguard.models.calibration import (
    PlattScaler,
    TemperatureScaler,
    expected_calibration_error,
    make_calibrator,
    null_ece,
    reliability_table,
    sigmoid,
)
from shiftguard.models.logreg import NestedLogisticRegression


def _data(rng, n=120, p=400, signal=1.5):
    w = np.zeros(p)
    w[:10] = signal / np.sqrt(10)
    X = rng.normal(size=(n, p))
    y = (rng.uniform(size=n) < sigmoid(X @ w)).astype(int)
    return X, y


# ----------------------------------------------------------------------------- logreg
def test_full_rank_rotation_is_exact_for_l2():
    """n << p: fitting on PCA scores gives the same predictions as on all features."""
    rng = np.random.default_rng(0)
    X, y = _data(rng, n=60, p=500)
    X_new = rng.normal(size=(30, 500))
    C = 0.05
    ours = NestedLogisticRegression(Cs=(C,), inner_cv_folds=3, random_state=0).fit(X, y)
    sc = StandardScaler().fit(X)
    ref = LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(sc.transform(X), y)
    np.testing.assert_allclose(
        ours.predict_proba(X_new)[:, 1], ref.predict_proba(sc.transform(X_new))[:, 1], atol=2e-3
    )


def test_nested_cv_selects_from_grid_and_learns():
    rng = np.random.default_rng(1)
    X, y = _data(rng, n=200, p=50, signal=3.0)
    m = NestedLogisticRegression(Cs=(1e-4, 1e-2, 1.0), inner_cv_folds=4, random_state=0).fit(X, y)
    assert m.best_params_["C"] in (1e-4, 1e-2, 1.0)
    assert len(m.cv_log_loss_) == 3
    X_te, y_te = _data(rng, n=400, p=50, signal=3.0)
    assert roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]) > 0.75
    np.testing.assert_allclose(m.predict_proba(X_te).sum(axis=1), 1.0)


def test_elasticnet_and_errors():
    rng = np.random.default_rng(2)
    X, y = _data(rng, n=80, p=20, signal=3.0)
    m = NestedLogisticRegression(
        Cs=(0.1,), penalty="elasticnet", l1_ratios=(0.5,), inner_cv_folds=3, random_state=0
    ).fit(X, y)
    assert m.best_params_ == {"C": 0.1, "l1_ratio": 0.5}
    with pytest.raises(ValueError, match="binary"):
        NestedLogisticRegression().fit(X, np.zeros(len(y)))
    with pytest.raises(ValueError, match="penalty"):
        NestedLogisticRegression(penalty="l1", inner_cv_folds=2).fit(X, y)


def test_elasticnet_is_fitted_on_original_features_never_rotated():
    rng = np.random.default_rng(8)
    X, y = _data(rng, n=60, p=200, signal=3.0)  # n < p: a rotation would give 60 dims
    enet = NestedLogisticRegression(
        Cs=(0.5,), penalty="elasticnet", l1_ratios=(0.9,), inner_cv_folds=3, random_state=0
    ).fit(X, y)
    assert [name for name, _ in enet.preprocessor_.steps] == ["scale"]  # no PCA step
    assert enet.classifier_.coef_.shape == (1, 200)  # one weight per original feature
    assert np.sum(enet.classifier_.coef_ == 0) > 0  # L1 sparsity in the original basis

    l2 = NestedLogisticRegression(Cs=(0.5,), inner_cv_folds=3, random_state=0).fit(X, y)
    assert [name for name, _ in l2.preprocessor_.steps] == ["scale", "rotate"]
    assert l2.classifier_.coef_.shape == (1, 60)  # rotated basis: n_train components

    with pytest.raises(ValueError, match="elastic net"):
        NestedLogisticRegression(penalty="elasticnet", pca_components=10).fit(X, y)


def test_pca_basis_is_fitted_on_training_rows_only(monkeypatch):
    """Every scaler/PCA fit sees inner-training rows or the full training split, never more."""
    rng = np.random.default_rng(9)
    X, y = _data(rng, n=90, p=120, signal=3.0)
    fit_sizes = []

    class SpyLogReg(NestedLogisticRegression):
        def _preprocessor(self):
            pipe = super()._preprocessor()
            original_fit = pipe.fit

            def recording_fit(Xf, yf=None, **kw):
                fit_sizes.append(len(Xf))
                return original_fit(Xf, yf, **kw)

            pipe.fit = recording_fit
            return pipe

    import shiftguard.models.logreg as lr

    # fit() clones its template pipeline, which would drop the spy: keep it instead
    monkeypatch.setattr(lr, "clone", lambda est: est)
    m = SpyLogReg(Cs=(0.1,), inner_cv_folds=3, random_state=0).fit(X, y)
    assert fit_sizes[:-1] == [60, 60, 60]  # inner folds: 2/3 of the 90 training rows
    assert fit_sizes[-1] == 90  # final basis: exactly the training rows
    pca = m.preprocessor_.named_steps["rotate"]
    assert pca.n_samples_ == 90 and pca.n_components_ == 90


def test_runner_fits_model_on_fold_training_rows_only(monkeypatch):
    import shiftguard.evaluation.runner as runner
    from shiftguard.config import load_config
    from shiftguard.data.synthetic import make_synthetic_dataset
    from shiftguard.evaluation.protocols import make_splits

    cfg = load_config("configs/experiment/synthetic.yaml")
    cfg["experiment"]["n_seeds"] = 1
    ds = make_synthetic_dataset(cfg["data"], seed=0)
    split = make_splits("loso", ds.sites, ds.labels, cfg["experiment"])[0]
    seen = {}
    real_make_model = runner.make_model

    def spy_make_model(name, models_cfg, seed):
        model = real_make_model(name, models_cfg, seed)
        real_fit = model.fit

        def fit(X, y):
            seen["X"] = np.array(X)
            return real_fit(X, y)

        model.fit = fit
        return model

    monkeypatch.setattr(runner, "make_model", spy_make_model)
    runner.run_fold(ds, split, "logreg", cfg, single_thread=False)
    np.testing.assert_array_equal(seen["X"], ds.features[split.train])


def test_make_model_factory():
    cfg = {"logreg": {"grid": {"C": [0.1]}, "inner_cv_folds": 2}}
    assert make_model("logreg", cfg, 0).Cs == (0.1,)
    with pytest.raises(ValueError):
        make_model("svm", {"svm": {}}, 0)


# ----------------------------------------------------------------------------- mlp
@pytest.mark.torch
def test_mlp_learns_and_is_reproducible():
    pytest.importorskip("torch")
    from shiftguard.models.mlp import TorchMLPClassifier

    rng = np.random.default_rng(3)
    X, y = _data(rng, n=300, p=30, signal=4.0)
    X_te, y_te = _data(rng, n=300, p=30, signal=4.0)
    kw = dict(hidden_sizes=(16,), max_epochs=60, early_stopping_patience=10, random_state=7)
    a = TorchMLPClassifier(**kw).fit(X, y)
    b = TorchMLPClassifier(**kw).fit(X, y)
    np.testing.assert_allclose(a.decision_function(X_te), b.decision_function(X_te), atol=1e-6)
    assert roc_auc_score(y_te, a.predict_proba(X_te)[:, 1]) > 0.8
    assert 1 <= a.n_epochs_ <= 60


# ----------------------------------------------------------------------------- calibration
def test_temperature_scaling_recovers_true_temperature():
    rng = np.random.default_rng(4)
    z = rng.normal(scale=3.0, size=20000)
    y = (rng.uniform(size=z.size) < sigmoid(z / 2.5)).astype(int)  # model overconfident by 2.5
    t = TemperatureScaler().fit(z, y)
    assert t.temperature_ == pytest.approx(2.5, rel=0.05)
    # monotone: ranking unchanged
    np.testing.assert_array_equal(np.argsort(t.predict_proba(z)), np.argsort(sigmoid(z)))


def test_platt_and_identity():
    rng = np.random.default_rng(5)
    z = rng.normal(size=5000)
    y = (rng.uniform(size=z.size) < sigmoid(2 * z + 1)).astype(int)
    p = PlattScaler().fit(z, y).predict_proba(np.array([0.0]))
    assert p[0] == pytest.approx(sigmoid(1.0), abs=0.05)
    np.testing.assert_allclose(make_calibrator("none").fit(z, y).predict_proba(z), sigmoid(z))
    with pytest.raises(ValueError):
        make_calibrator("isotonic")


def test_ece_hand_example_and_reliability_table():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.85])  # bins 0, 0, 9, 8
    # |0 - .05| * 2/4 + |1 - .95| * 1/4 + |1 - .85| * 1/4
    assert expected_calibration_error(y, p) == pytest.approx(0.025 + 0.0125 + 0.0375)
    rel = reliability_table(y, p)
    assert rel["count"].sum() == 4 and list(rel["bin"]) == [0, 8, 9]


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(6)
    p = rng.uniform(size=100000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_null_ece_matches_calibrated_ece_and_shrinks_with_n():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.2, 0.8, size=200)
    # observed ECE of truly calibrated labels, averaged, equals the null reference
    observed = np.mean(
        [
            expected_calibration_error((rng.uniform(size=p.size) < p).astype(int), p)
            for _ in range(400)
        ]
    )
    assert null_ece(p, n_draws=400, seed=1) == pytest.approx(observed, rel=0.08)
    assert null_ece(np.tile(p, 10)) < 0.5 * null_ece(p)  # small-sample bias ~ 1/sqrt(n)
    assert null_ece(p, seed=3) == null_ece(p, seed=3)  # deterministic
    assert np.isnan(null_ece([]))


def test_binary_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["auc"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["cal_in_large"] == pytest.approx(0.5 - 0.5)
    assert binary_metrics([0, 0, 0, 1], [0.5] * 4)["cal_in_large"] == pytest.approx(0.25)
    assert np.isnan(binary_metrics([1, 1], [0.2, 0.9])["auc"])
