"""Covariate-shift weights for one leave-one-site-out fold.

Pipeline (choices confirmed in phase 2, see docs/METHODS.md §4.5):

1. Standardise and fit PCA on the training rows **plus the unlabelled deployment-site
   rows** (transductive: no label is used).
2. Domain classifier = L2 logistic regression on the first ``k`` components, separating
   training rows (source, 0) from deployment rows (target, 1). ``k`` and ``C`` are chosen
   by stratified K-fold CV on the domain task (log loss); the out-of-fold AUC at the
   chosen values is reported (AUC near 1 means the sites are almost separable, so the
   weights will be extreme).
3. Calibration rows are never used to fit the classifier, so their weights are
   out-of-sample. Deployment rows get **cross-fitted** weights: each fold of target rows
   is scored by a classifier that did not see it.
4. ``w(x) = c(x) / (1 - c(x)) * n_source / n_target`` with the counts of the fit that
   produced ``c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DomainWeights:
    """Weights and diagnostics of one fold's domain classifier."""

    cal_weights: NDArray[np.float64]
    test_weights: NDArray[np.float64]
    auc: float
    n_components: int
    C: float


def _odds_weights(c: NDArray[np.float64], n_source: int, n_target: int, clip: float) -> NDArray:
    c = np.clip(c, clip, 1.0 - clip)
    return c / (1.0 - c) * (n_source / n_target)


def domain_classifier_weights(
    X_train: ArrayLike,
    X_cal: ArrayLike,
    X_target: ArrayLike,
    *,
    components_grid: tuple[int, ...] = (10, 20, 50, 100),
    C_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
    cv_folds: int = 5,
    random_state: int | None = None,
    prob_clip: float = 1e-6,
) -> DomainWeights:
    """Likelihood-ratio weights for calibration and (unlabelled) deployment rows."""
    Xs = np.asarray(X_train, dtype=np.float64)
    Xc = np.asarray(X_cal, dtype=np.float64)
    Xt = np.asarray(X_target, dtype=np.float64)
    pool = np.vstack([Xs, Xt])  # transductive: unlabelled target features only
    scaler = StandardScaler().fit(pool)
    n_max = min(max(components_grid), len(pool) - 1, pool.shape[1])
    pca = PCA(n_components=n_max, svd_solver="randomized", random_state=random_state)
    pca.fit(scaler.transform(pool))
    Zs, Zc, Zt = (pca.transform(scaler.transform(a)) for a in (Xs, Xc, Xt))
    Z = np.vstack([Zs, Zt])
    d = np.r_[np.zeros(len(Zs), dtype=int), np.ones(len(Zt), dtype=int)]
    grid = [(k, C) for k in sorted({min(k, n_max) for k in components_grid}) for C in C_grid]
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
    folds = list(skf.split(Z, d))

    def fit(k: int, C: float, rows: NDArray[np.intp]) -> LogisticRegression:
        return LogisticRegression(C=C, max_iter=5000).fit(Z[rows, :k], d[rows])

    best, best_loss, best_oof = grid[0], np.inf, None
    for k, C in grid:
        oof = np.empty(len(Z))
        for tr, va in folds:
            oof[va] = fit(k, C, tr).predict_proba(Z[va, :k])[:, 1]
        loss = log_loss(d, oof, labels=[0, 1])
        if loss < best_loss:
            best, best_loss, best_oof = (k, C), loss, oof
    k, C = best
    n_s, n_t = len(Zs), len(Zt)
    full = LogisticRegression(C=C, max_iter=5000).fit(Z[:, :k], d)
    cal_w = _odds_weights(full.predict_proba(Zc[:, :k])[:, 1], n_s, n_t, prob_clip)
    # deployment rows: cross-fitted, reusing the CV folds restricted to target rows
    test_w = np.empty(n_t)
    for tr, va in folds:
        va_t = va[va >= n_s]
        if len(va_t) == 0:
            continue
        c = fit(k, C, tr).predict_proba(Z[va_t, :k])[:, 1]
        n_t_fit = int(np.sum(d[tr]))
        test_w[va_t - n_s] = _odds_weights(c, len(tr) - n_t_fit, n_t_fit, prob_clip)
    auc = float(roc_auc_score(d, best_oof))
    return DomainWeights(cal_w, test_w, auc, int(k), float(C))
