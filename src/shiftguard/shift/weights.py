r"""Likelihood-ratio estimation with a probabilistic domain classifier.

Train a classifier to tell source (calibration) features from target (deployment)
features, giving :math:`c(x) = \hat P(\text{target} \mid x)`. By Bayes' rule,

.. math::

    \hat w(x) = \frac{dP^{target}_X}{dP^{source}_X}(x)
              = \frac{c(x)}{1 - c(x)} \cdot \frac{n_{source}}{n_{target}},

where the last factor corrects for the class balance of the domain-classification data.
A logistic-regression domain classifier gives a log-linear ratio; that is exact for
Gaussian mean shifts with a shared covariance, and only an approximation otherwise.

The target features used here are **unlabelled**: no diagnosis from the deployment site
is ever used to estimate weights.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def default_domain_classifier(C: float = 1.0, random_state: int | None = None) -> Pipeline:
    """Standardised, L2-penalised logistic regression."""
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=random_state)
    )


class DomainClassifierRatio:
    """Estimate ``w(x) = p_target(x) / p_source(x)`` from unlabelled features.

    Parameters
    ----------
    classifier
        Any scikit-learn classifier with ``predict_proba``. Defaults to
        :func:`default_domain_classifier`.
    prob_clip
        Domain probabilities are clipped to ``[prob_clip, 1 - prob_clip]`` before taking
        odds, which bounds the ratio and avoids division by zero.
    """

    def __init__(
        self, classifier: ClassifierMixin | None = None, *, prob_clip: float = 1e-6
    ) -> None:
        if not 0.0 < prob_clip < 0.5:
            raise ValueError("prob_clip must be in (0, 0.5)")
        self.classifier = classifier
        self.prob_clip = prob_clip

    def fit(self, X_source: ArrayLike, X_target: ArrayLike) -> Self:
        """Fit the domain classifier (label 0 = source, 1 = target)."""
        Xs = np.asarray(X_source, dtype=np.float64)
        Xt = np.asarray(X_target, dtype=np.float64)
        if Xs.ndim != 2 or Xt.ndim != 2 or Xs.shape[1] != Xt.shape[1]:
            raise ValueError("X_source and X_target must be 2-D with the same n_features")
        if len(Xs) == 0 or len(Xt) == 0:
            raise ValueError("need at least one source and one target sample")
        X = np.vstack([Xs, Xt])
        d = np.r_[np.zeros(len(Xs), dtype=int), np.ones(len(Xt), dtype=int)]
        base = self.classifier if self.classifier is not None else default_domain_classifier()
        self.classifier_ = clone(base).fit(X, d)
        self.n_source_ = len(Xs)
        self.n_target_ = len(Xt)
        return self

    def predict_proba_target(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimated probability that each row comes from the target domain."""
        if not hasattr(self, "classifier_"):
            raise RuntimeError("DomainClassifierRatio is not fitted; call fit() first")
        return self.classifier_.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]

    def ratio(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimated likelihood ratio ``w(x)`` for each row of ``X``."""
        c = np.clip(self.predict_proba_target(X), self.prob_clip, 1.0 - self.prob_clip)
        return c / (1.0 - c) * (self.n_source_ / self.n_target_)
