r"""Synthetic experiments that check the conformal core against theory.

Both experiments use **one test point per trial** with a fresh calibration set, so the
number of covered trials is exactly Binomial(``n_trials``, :math:`p`), where :math:`p` is
the marginal coverage. Exact Clopper-Pearson intervals and binomial tests then apply
without approximation.

Experiment A: exchangeable data
-------------------------------
Features :math:`X \sim N(0, I_d)`, labels from a multinomial-logistic model, and a
deliberately *misspecified* model (shrunken, perturbed coefficients) providing
:math:`\hat p`. Conformal validity does not need a good model. For continuous scores
(LAC, randomised APS) theory gives coverage exactly :math:`k / (n + 1)` with
:math:`k = \lceil (n+1)(1-\alpha) \rceil`.

Experiment B: controlled covariate shift
----------------------------------------
Two features. Source :math:`X \sim N(0, I_2)`, target :math:`X \sim N((\mu, 0), I_2)`.
The label depends on :math:`x_2` through :math:`\sigma(\beta x_2)`, but labels are flipped
with probability :math:`\varepsilon(x_1) = \varepsilon_{max}\, \sigma(\gamma (x_1 - c))`,
which grows along the shifted direction. The model ignores the flips,
:math:`\hat p(1 \mid x) = \sigma(\beta x_2)`, so it is overconfident exactly where the
target puts its mass. :math:`P(Y \mid X)` is identical in both domains: this is pure
covariate shift, and the true ratio is :math:`w(x) = \exp(\mu x_1 - \mu^2 / 2)`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shiftguard.conformal.quantiles import conformal_rank
from shiftguard.conformal.scores import ScoreName
from shiftguard.conformal.split import SplitConformal
from shiftguard.conformal.weighted import WeightedConformal
from shiftguard.evaluation.statistics import binomial_test_below, clopper_pearson
from shiftguard.shift.weights import DomainClassifierRatio


@dataclass(frozen=True)
class CoverageResult:
    """Coverage of one method in one synthetic experiment."""

    experiment: str
    method: str
    score: str
    n_classes: int
    n_calibration: int
    n_trials: int
    target: float
    n_covered: int
    coverage: float
    ci_low: float
    ci_high: float
    p_below_target: float
    mean_set_size: float
    abstention_rate: float
    mean_ess: float = math.nan
    theory: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, for tables."""
        return asdict(self)


def results_frame(results: list[CoverageResult]) -> pd.DataFrame:
    """Stack results into a DataFrame."""
    return pd.DataFrame([r.as_dict() for r in results])


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sample_labels(p: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.intp]:
    """Draw one label per row of the probability tensor ``p`` (last axis = classes)."""
    u = rng.uniform(size=(*p.shape[:-1], 1))
    labels = (np.cumsum(p, axis=-1) < u).sum(axis=-1)
    return np.minimum(labels, p.shape[-1] - 1).astype(np.intp)


def _summarize(
    *,
    experiment: str,
    method: str,
    score: str,
    covered: NDArray[np.bool_],
    sizes: NDArray[np.intp],
    n_classes: int,
    n_calibration: int,
    alpha: float,
    ess: NDArray[np.float64] | None = None,
    theory: str = "",
) -> CoverageResult:
    n_trials = covered.size
    k = int(covered.sum())
    lo, hi = clopper_pearson(k, n_trials)
    return CoverageResult(
        experiment=experiment,
        method=method,
        score=score,
        n_classes=n_classes,
        n_calibration=n_calibration,
        n_trials=n_trials,
        target=1.0 - alpha,
        n_covered=k,
        coverage=k / n_trials,
        ci_low=lo,
        ci_high=hi,
        p_below_target=binomial_test_below(k, n_trials, 1.0 - alpha),
        mean_set_size=float(sizes.mean()),
        abstention_rate=float((sizes != 1).mean()),
        mean_ess=float(np.mean(ess)) if ess is not None else math.nan,
        theory=theory,
    )


# --------------------------------------------------------------------------- experiment A
@dataclass(frozen=True)
class LogisticProblem:
    """Multinomial-logistic data with a misspecified probability model."""

    true_coef: NDArray[np.float64]
    model_coef: NDArray[np.float64]

    @classmethod
    def make(cls, n_classes: int = 2, n_features: int = 5, *, seed: int = 0) -> LogisticProblem:
        """Draw true coefficients and a shrunken, perturbed model."""
        rng = np.random.default_rng(seed)
        true = rng.normal(scale=1.5, size=(n_features, n_classes))
        model = 0.5 * true + rng.normal(scale=0.5, size=true.shape)
        return cls(true_coef=true, model_coef=model)

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return self.true_coef.shape[1]

    def sample(
        self, shape: tuple[int, ...], rng: np.random.Generator
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Return labels and model probabilities for a batch of the given shape."""
        X = rng.normal(size=(*shape, self.true_coef.shape[0]))
        y = _sample_labels(_softmax(X @ self.true_coef), rng)
        return y, _softmax(X @ self.model_coef)


def simulate_exchangeable(
    *,
    n_classes: int = 2,
    n_calibration: int = 200,
    n_trials: int = 4000,
    alpha: float = 0.1,
    score: ScoreName = "lac",
    randomized: bool = False,
    seed: int = 0,
) -> CoverageResult:
    """Experiment A: split conformal coverage when data are exchangeable."""
    problem = LogisticProblem.make(n_classes=n_classes, seed=seed)
    rng = np.random.default_rng(seed + 1)
    y_cal, p_cal = problem.sample((n_trials, n_calibration), rng)
    y_te, p_te = problem.sample((n_trials, 1), rng)

    covered = np.empty(n_trials, dtype=bool)
    sizes = np.empty(n_trials, dtype=np.intp)
    cp = SplitConformal(alpha, score, randomized=randomized, random_state=seed + 2)
    for r in range(n_trials):
        sets = cp.fit(p_cal[r], y_cal[r]).predict_sets(p_te[r])
        covered[r] = sets[0, y_te[r, 0]]
        sizes[r] = sets[0].sum()

    k = conformal_rank(n_calibration, alpha)
    continuous = score == "lac" or randomized
    theory = (
        f"exact {k}/{n_calibration + 1} = {k / (n_calibration + 1):.4f}"
        if continuous
        else f">= {1 - alpha:.2f} (ties: upper bound does not apply)"
    )
    name = "split" + (" (randomised)" if randomized and score == "aps" else "")
    return _summarize(
        experiment="A_exchangeable",
        method=name,
        score=score,
        covered=covered,
        sizes=sizes,
        n_classes=n_classes,
        n_calibration=n_calibration,
        alpha=alpha,
        theory=theory,
    )


# --------------------------------------------------------------------------- experiment B
@dataclass(frozen=True)
class CovariateShiftProblem:
    """Pure covariate shift along ``x1`` with label noise that grows along ``x1``."""

    shift: float = 1.0
    signal: float = 3.0
    noise_max: float = 0.45
    noise_center: float = 1.0
    noise_slope: float = 3.0

    def sample_x(
        self, shape: tuple[int, ...], rng: np.random.Generator, *, target: bool
    ) -> NDArray[np.float64]:
        """Features from the source (``target=False``) or target distribution."""
        X = rng.normal(size=(*shape, 2))
        if target:
            X[..., 0] += self.shift
        return X

    def model_probs(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the (misspecified) model's probabilities, which ignore the label flips."""
        p1 = _sigmoid(self.signal * X[..., 1])
        return np.stack([1.0 - p1, p1], axis=-1)

    def true_probs(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the true ``P(Y | X)``, identical in source and target."""
        clean = _sigmoid(self.signal * X[..., 1])
        eps = self.noise_max * _sigmoid(self.noise_slope * (X[..., 0] - self.noise_center))
        p1 = (1.0 - eps) * clean + eps * (1.0 - clean)
        return np.stack([1.0 - p1, p1], axis=-1)

    def true_ratio(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Exact likelihood ratio ``p_target(x) / p_source(x)``."""
        return np.exp(self.shift * X[..., 0] - self.shift**2 / 2.0)

    def sample(
        self, shape: tuple[int, ...], rng: np.random.Generator, *, target: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64]]:
        """Features, labels and model probabilities."""
        X = self.sample_x(shape, rng, target=target)
        y = _sample_labels(self.true_probs(X), rng)
        return X, y, self.model_probs(X)


def simulate_covariate_shift(
    *,
    problem: CovariateShiftProblem | None = None,
    n_calibration: int = 1000,
    n_trials: int = 4000,
    alpha: float = 0.1,
    score: ScoreName = "lac",
    n_domain: int = 2000,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    seed: int = 0,
) -> list[CoverageResult]:
    """Experiment B: split vs weighted conformal under a controlled covariate shift.

    Methods
    -------
    ``split (source test)``
        Sanity check: split conformal evaluated on *source* test points.
    ``split``
        Split conformal calibrated on the source, evaluated on the target.
    ``weighted (oracle w)``
        Weighted CP with the exact likelihood ratio.
    ``weighted (estimated w)``
        Weighted CP with a logistic domain classifier fitted once on ``n_domain``
        *unlabelled* source and target samples, independent of the calibration data.
    ``weighted (estimated w, clipped)``
        As above, with weights clipped to ``clip_quantiles`` of the calibration weights.
    """
    prob = problem or CovariateShiftProblem()
    rng = np.random.default_rng(seed)

    ratio_model = DomainClassifierRatio().fit(
        prob.sample_x((n_domain,), rng, target=False),
        prob.sample_x((n_domain,), rng, target=True),
    )

    X_cal, y_cal, p_cal = prob.sample((n_trials, n_calibration), rng, target=False)
    _, y_src, p_src = prob.sample((n_trials,), rng, target=False)
    X_te, y_te, p_te = prob.sample((n_trials,), rng, target=True)

    w_cal_true, w_te_true = prob.true_ratio(X_cal), prob.true_ratio(X_te)
    w_cal_hat = ratio_model.ratio(X_cal.reshape(-1, 2)).reshape(n_trials, n_calibration)
    w_te_hat = ratio_model.ratio(X_te)

    methods: dict[str, dict[str, Any]] = {
        "split (source test)": {"kind": "split", "test": (p_src, y_src)},
        "split": {"kind": "split"},
        "weighted (oracle w)": {"kind": "weighted", "w": (w_cal_true, w_te_true)},
        "weighted (estimated w)": {"kind": "weighted", "w": (w_cal_hat, w_te_hat)},
        "weighted (estimated w, clipped)": {
            "kind": "weighted",
            "w": (w_cal_hat, w_te_hat),
            "clip": clip_quantiles,
        },
    }

    results: list[CoverageResult] = []
    for name, spec in methods.items():
        p_test, y_test = spec.get("test", (p_te, y_te))
        covered = np.empty(n_trials, dtype=bool)
        sizes = np.empty(n_trials, dtype=np.intp)
        ess = np.full(n_trials, np.nan) if spec["kind"] == "weighted" else None
        split = SplitConformal(alpha, score)
        weighted = WeightedConformal(alpha, score, clip_quantiles=spec.get("clip"))
        for r in range(n_trials):
            if ess is None:
                sets = split.fit(p_cal[r], y_cal[r]).predict_sets(p_test[r : r + 1])
            else:
                w_cal, w_te = spec["w"]
                weighted.fit(p_cal[r], y_cal[r], w_cal[r])
                sets = weighted.predict_sets(p_test[r : r + 1], w_te[r : r + 1])
                ess[r] = weighted.ess_
            covered[r] = sets[0, y_test[r]]
            sizes[r] = sets[0].sum()
        theory = {
            "split (source test)": "exchangeable: >= target",
            "split": "no guarantee under shift",
            "weighted (oracle w)": ">= target (exact weights)",
        }.get(name, "approximate (estimated weights)")
        results.append(
            _summarize(
                experiment="B_covariate_shift",
                method=name,
                score=score,
                covered=covered,
                sizes=sizes,
                n_classes=2,
                n_calibration=n_calibration,
                alpha=alpha,
                ess=ess,
                theory=theory,
            )
        )
    return results


@dataclass
class SimulationReport:
    """All synthetic results, for the CLI and the docs."""

    results: list[CoverageResult] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        """Results as a DataFrame."""
        return results_frame(self.results)


def run_all_simulations(
    *, n_trials: int = 4000, alpha: float = 0.1, seed: int = 0
) -> SimulationReport:
    """Run experiment A (LAC, APS and randomised APS; K = 2 and 3) and experiment B."""
    report = SimulationReport()
    for n_classes in (2, 3):
        for score, randomized in (("lac", False), ("aps", False), ("aps", True)):
            report.results.append(
                simulate_exchangeable(
                    n_classes=n_classes,
                    n_trials=n_trials,
                    alpha=alpha,
                    score=score,  # type: ignore[arg-type]
                    randomized=randomized,
                    seed=seed,
                )
            )
    report.results.extend(simulate_covariate_shift(n_trials=n_trials, alpha=alpha, seed=seed))
    return report
