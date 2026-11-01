r"""Penalised logistic regression with nested cross-validation.

Hyperparameters are chosen by stratified K-fold CV **on the training data passed to
``fit`` only** (the inner loop of the nested design), minimising the mean inner-fold log
loss, which rewards good probabilities rather than just ranking. The model is then refit
on all training data with the selected values.

Exact speed-up for L2
---------------------
With :math:`n \ll p` (about 650 training subjects, 19,900 edges), the L2-penalised
solution lies in the row space of the centred training matrix (representer theorem), and
the L2 penalty is rotation invariant. Fitting on the full-rank PCA scores of the training
data (no truncation, no whitening) therefore gives *identical* predictions to fitting on all
features, while each fit works in :math:`n` dimensions instead of :math:`p`. This is
verified in the tests. It does **not** hold for the elastic net, whose L1 term is not
rotation invariant: the elastic net is always fitted on the original (standardised)
features, and combining it with ``pca_components`` raises an error.

Every PCA basis (and the scaler) is fitted on training rows only: on the inner-training
folds during hyperparameter search, then on all rows passed to ``fit``. Calibration and
test rows are only ever transformed.
"""

from __future__ import annotations

from itertools import product
from typing import Self

import numpy as np
import sklearn
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

_SKLEARN_GE_1_8 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


class NestedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary L2 or elastic-net logistic regression, tuned by inner stratified CV.

    Parameters
    ----------
    Cs
        Inverse regularisation strengths to try.
    penalty
        ``"l2"`` or ``"elasticnet"``.
    l1_ratios
        Elastic-net mixing values to try (ignored for L2).
    pca_components
        ``None``: full-rank rotation for L2 (exact) or no PCA for the elastic net. An
        integer truncates to that many components (an approximation, fitted on training
        data only).
    standardize
        Z-score features (fitted on training data only).
    inner_cv_folds
        Number of inner folds.
    max_iter
        Solver iterations.
    random_state
        Seed for the inner fold assignment.

    Attributes
    ----------
    best_params_ : dict
        Selected ``C`` (and ``l1_ratio``).
    cv_log_loss_ : dict
        Mean inner-fold log loss of every candidate.
    """

    def __init__(
        self,
        Cs: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        penalty: str = "l2",
        l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9),
        pca_components: int | None = None,
        standardize: bool = True,
        inner_cv_folds: int = 5,
        max_iter: int = 5000,
        random_state: int | None = None,
    ) -> None:
        self.Cs = Cs
        self.penalty = penalty
        self.l1_ratios = l1_ratios
        self.pca_components = pca_components
        self.standardize = standardize
        self.inner_cv_folds = inner_cv_folds
        self.max_iter = max_iter
        self.random_state = random_state

    def _preprocessor(self) -> Pipeline:
        steps: list[tuple[str, object]] = []
        steps.append(("scale", StandardScaler() if self.standardize else FunctionTransformer()))
        if self.pca_components is not None:
            steps.append(("pca", PCA(n_components=self.pca_components, svd_solver="full")))
        elif self.penalty == "l2":
            steps.append(("rotate", PCA(n_components=None, svd_solver="full")))
        return Pipeline(steps)

    def _classifier(self, C: float, l1_ratio: float | None) -> LogisticRegression:
        if self.penalty == "l2":
            return LogisticRegression(C=C, max_iter=self.max_iter)
        if self.penalty == "elasticnet":
            # scikit-learn >= 1.8 infers the elastic net from l1_ratio (`penalty` deprecated)
            legacy = {} if _SKLEARN_GE_1_8 else {"penalty": "elasticnet"}
            return LogisticRegression(
                C=C, l1_ratio=l1_ratio, solver="saga", max_iter=self.max_iter, **legacy
            )
        raise ValueError(f"penalty must be 'l2' or 'elasticnet', got {self.penalty!r}")

    def _grid(self) -> list[tuple[float, float | None]]:
        if self.penalty == "elasticnet":
            return list(product(self.Cs, self.l1_ratios))
        return [(C, None) for C in self.Cs]

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Select hyperparameters by inner CV on ``(X, y)``, then refit on all of it."""
        if self.penalty == "elasticnet" and self.pca_components is not None:
            # The L1 part is not rotation invariant: the elastic net must see the original
            # (standardised) features, never a PCA basis, truncated or not.
            raise ValueError("pca_components must be None for the elastic net")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("NestedLogisticRegression needs binary labels {0, 1} (both present)")
        grid = self._grid()
        losses = np.zeros(len(grid))
        skf = StratifiedKFold(self.inner_cv_folds, shuffle=True, random_state=self.random_state)
        for tr, va in skf.split(X, y):
            pre = clone(self._preprocessor()).fit(X[tr])
            Z_tr, Z_va = pre.transform(X[tr]), pre.transform(X[va])
            for j, (C, l1) in enumerate(grid):
                clf = self._classifier(C, l1).fit(Z_tr, y[tr])
                losses[j] += log_loss(y[va], clf.predict_proba(Z_va)[:, 1], labels=[0, 1])
        losses /= self.inner_cv_folds
        best_C, best_l1 = grid[int(np.argmin(losses))]
        self.best_params_ = {"C": best_C} | ({"l1_ratio": best_l1} if best_l1 is not None else {})
        self.cv_log_loss_ = {str(g): float(v) for g, v in zip(grid, losses, strict=True)}
        self.preprocessor_ = clone(self._preprocessor()).fit(X)
        self.classifier_ = self._classifier(best_C, best_l1).fit(self.preprocessor_.transform(X), y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Logit ``log p(1|x) / p(0|x)``."""
        Z = self.preprocessor_.transform(np.asarray(X, dtype=np.float64))
        return self.classifier_.decision_function(Z)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Class probabilities, shape ``(n, 2)``."""
        p1 = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.c_[1.0 - p1, p1]
