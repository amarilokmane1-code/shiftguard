# Methods

*Sections 1, 4 and 7 are written; the others are completed as each phase is implemented.*

## 1. Data and features

**Data.** ABIDE I Preprocessed (PCP): C-PAC pipeline, band-pass filtered, no global
signal regression, CC200 atlas (200 ROIs). The PCP phenotypic file supplies site,
diagnosis, age, sex, FIQ and manual QC ratings. Download, QC rules and the resulting
cohort (944 subjects, 18 sites) are described in [DATA_CARD.md](DATA_CARD.md).

**Connectivity.** For each subject with ROI time series $T \in \mathbb{R}^{t \times R}$
($R = 200$), the covariance is estimated with Ledoit-Wolf shrinkage and converted to a
correlation matrix $C$ (`nilearn.connectome.ConnectivityMeasure`). The strict upper
triangle gives $R(R-1)/2 = 19{,}900$ features, Fisher z-transformed:
$z_{jk} = \operatorname{arctanh}(C_{jk})$, with $|C_{jk}|$ clipped at $1 - 10^{-7}$.
Correlation and partial correlation are **subject-wise**: a subject's features depend
only on its own time series, so they are computed once and cached.

**Tangent space (optional).** Each covariance $\Sigma_i$ is mapped to
$\log\big(\bar\Sigma^{-1/2}\Sigma_i\bar\Sigma^{-1/2}\big)$, where $\bar\Sigma$ is the
geometric mean of the covariances of the **training subjects only**. This reference
depends on who is in the training set, so tangent features are never precomputed:
`subjectwise_features(kind="tangent")` raises an error, and `fit_transform_fold()` fits the
reference on the training indices and checks that every evaluation split is disjoint from
them. A test checks that test-subject features are unchanged by the composition of the
test set, and that fitting the reference on train + test would change them.

## 2. Evaluation protocols (leave-one-site-out and within-site)
TBD (phase 4).

## 3. Models and probability calibration
TBD (phase 4).

## 4. Conformal prediction

Implementation: `src/shiftguard/conformal/` (written from scratch, unit tested). Notation: calibration pairs $(X_i, Y_i)_{i=1}^n$, a test point $(X_{n+1},
Y_{n+1})$, estimated class probabilities $\hat p(\cdot \mid x)$, miscoverage level
$\alpha$ (default 0.10, target coverage 90%).

### 4.1 Nonconformity scores

- **LAC** (Sadinle, Lei and Wasserman, 2019): $s(x, y) = 1 - \hat p(y \mid x)$.
- **APS** (Romano, Sesia and Candès, 2020): with classes sorted by decreasing
  $\hat p(\cdot \mid x)$ (ties broken by class index),
  $s(x, y) = \sum_{j \text{ ranked above } y} \hat p(j \mid x) + u\,\hat p(y \mid x)$,
  where $u \sim \mathrm{U}(0,1)$ (randomised) or $u = 1$ (deterministic).

### 4.2 Split conformal

With calibration scores $S_i = s(X_i, Y_i)$,

$$
\hat q = S_{(k)}, \qquad k = \lceil (n+1)(1-\alpha) \rceil,
\qquad C(x) = \{\, y : s(x, y) \le \hat q \,\},
$$

where $S_{(k)}$ is the $k$-th smallest score and $\hat q = +\infty$ when $k > n$. Under
exchangeability, $1-\alpha \le P\{Y_{n+1} \in C(X_{n+1})\} \le 1-\alpha+\tfrac{1}{n+1}$
(upper bound for continuous scores). A finite quantile needs
$n \ge (1-\alpha)/\alpha$, i.e. **at least 9 calibration points for $\alpha = 0.1$**. Below
that, every set is the full label set.

### 4.3 Class-conditional conformal

One quantile per class, $\hat q_k$ = conformal quantile of $\{S_i : Y_i = k\}$, and
$C(x) = \{k : s(x,k) \le \hat q_k\}$. This guarantees $P\{Y \in C(X) \mid Y = k\} \ge
1-\alpha$ for each diagnosis group.

### 4.4 Mondrian (group-conditional) conformal

One quantile per group $g$ known at test time (the acquisition site),
$\hat q_g$ = conformal quantile of $\{S_i : G_i = g\}$, and $C(x, g) = \{y : s(x,y) \le
\hat q_g\}$, which guarantees coverage within each group. A deployment site has
calibration data only in the *few labelled target subjects* scenario (0, 5, 10 or 20
labelled subjects moved to calibration). Groups without calibration data, or with fewer
than $(1-\alpha)/\alpha$ points, get full sets.

> **At $\alpha = 0.1$, per-site conformal needs at least 9 labelled target subjects.** The
> 5-label arm is kept on purpose as an honest demonstration: by construction, it
> abstains on every subject of the deployment site.

### 4.5 Weighted conformal prediction (covariate shift)

Following Tibshirani, Barber, Candès and Ramdas (2019): if $P_X$ changes but
$P_{Y \mid X}$ does not, weight calibration points by the likelihood ratio
$w(x) = dP_X^{\text{test}} / dP_X^{\text{cal}}(x)$. For a test point $x$,

$$
p_i(x) = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(x)}, \qquad
p_{n+1}(x) = \frac{w(x)}{\sum_{j=1}^n w(X_j) + w(x)},
$$

$$
\hat q(x) = \inf\Big\{ s : \sum_{i : S_i \le s} p_i(x) \ge 1-\alpha \Big\},
$$

that is, the $(1-\alpha)$ quantile of $\sum_i p_i(x)\,\delta_{S_i} + p_{n+1}(x)\,\delta_{+\infty}$
($+\infty$ if the calibration mass never reaches $1-\alpha$). With the exact $w$, coverage
on the test distribution is $\ge 1-\alpha$. With $w \equiv 1$ this reduces exactly to
§4.2 (unit-tested).

**Likelihood-ratio estimation.** A logistic-regression domain classifier
$c(x) = \hat P(\text{target} \mid x)$ is fitted on *unlabelled* source and target
features, and
$\hat w(x) = \frac{c(x)}{1-c(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}$.

**Effective sample size.** Kish's $n_{\text{eff}} = (\sum_i w_i)^2 / \sum_i w_i^2$ of
the calibration weights is reported with every weighted result.

**Clipping (optional).** Weights are clipped to the [1%, 99%] quantiles of the
calibration weights, and the same bounds are applied to test weights. This is a
variance-reduction heuristic that voids the exact guarantee, so it is reported as a
separate method.

### 4.6 Abstention

A case is abstained when its prediction set is empty or contains both classes. We report
the abstention rate and the accuracy on non-abstained (singleton) cases.

### 4.7 Validation on synthetic data

`shiftguard simulate` writes
[`reports/tables/conformal_synthetic.md`](../reports/tables/conformal_synthetic.md).
Each trial uses a fresh calibration set and **one** test point, so the number of covered
trials is exactly Binomial(trials, coverage) and exact inference applies.

- **A (exchangeable):** multinomial-logistic data with a deliberately misspecified model,
  $K \in \{2, 3\}$. For continuous scores, coverage must equal $k/(n+1)$ (both tails
  tested at the 0.1% level). A version with $n = 19$ makes an off-by-one error in $k$
  detectable (0.90 vs 0.85). We checked that a deliberately injected off-by-one fails it.
- **B (covariate shift):** source $X \sim N(0, I_2)$, target $X \sim N((1, 0), I_2)$. Label
  noise $\varepsilon(x_1) = 0.45\,\sigma(3(x_1 - 1))$ grows along the shifted direction and
  is ignored by the model, while $P(Y \mid X)$ is the same in both domains (pure
  covariate shift). Tests require split CP to fall significantly below target, and
  weighted CP (oracle, estimated, clipped) to be not significantly below target without
  degenerating to full sets.

A side finding the tests document: **in binary problems, deterministic APS degenerates.**
The non-top label always has score 1, so whenever the model's top-class error rate
exceeds $\alpha$, $\hat q = 1$ and every set is full. Randomised APS and LAC do not
have this problem. **Score choice for ABIDE:** LAC is the primary score and randomised
APS the secondary one. Deterministic APS is reported once, as this documented failure case.

## 5. Harmonisation
TBD (phase 5).

## 6. Label-free shift detection
TBD (phase 6).

## 7. Statistical inference

- **Clopper–Pearson interval** for $k$ covered out of $n$ (level $1-\gamma$):
  $\big[B(\tfrac{\gamma}{2}; k, n-k+1),\; B(1-\tfrac{\gamma}{2}; k+1, n-k)\big]$, with
  $B(q; a, b)$ the Beta quantile, lower bound 0 if $k=0$ and upper bound 1 if $k=n$.
- **Under-coverage test:** exact one-sided binomial test of
  $H_0: \text{coverage} \ge 1-\alpha$ against $H_1: \text{coverage} < 1-\alpha$, with
  p-value $P\{\mathrm{Bin}(n, 1-\alpha) \le k\}$.
- TBD (phase 5): Benjamini–Hochberg across sites, paired bootstrap between methods,
  empirical-Bayes beta-binomial for small sites.
