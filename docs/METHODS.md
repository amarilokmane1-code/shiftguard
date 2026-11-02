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

Implementation: `src/shiftguard/evaluation/protocols.py`. Seeds 0–9; every random
choice is derived from `(base_seed, protocol, seed, site)`.

- **Leave-one-site-out (LOSO, site shift).** For each of the 18 sites and each seed, every
  subject of that site is the test (deployment) set. The remaining sites are split by
  subject into train (70%) and calibration (30%), stratified by site × diagnosis. The
  deployment site is never used for training, calibration, hyperparameter tuning or
  feature fitting. 180 splits.
- **Within-site (exchangeable reference, no shift).** For each seed, a random subject-level
  split of the pooled cohort into test (20%), train and calibration (70/30 of the rest),
  stratified by site × diagnosis, so every site appears in every split. 10 splits.
- **Leakage guard.** Before any model is fitted, each split must pass `check_split`: the
  three parts are disjoint, no subject ID appears in two parts, and (LOSO) the test site
  is absent from train and calibration and the test set contains only that site. Unit
  tests also check that the LOSO test sets partition the cohort for every seed.
- **Fitted on training rows only:** feature standardisation, the PCA rotation,
  hyperparameter selection, MLP early stopping, and (tangent kind) the tangent
  reference. Calibrators are fitted on the calibration rows only.

## 3. Models and probability calibration

**L2 logistic regression** (`models/logreg.py`). Features are z-scored, then C is chosen
from $\{10^{-6}, \dots, 10^{0}\}$ by 5-fold stratified CV on the training rows,
minimising the mean inner-fold log loss, and the model is refit on all training rows.
With $n \approx 650 \ll p = 19{,}900$, the L2 solution lies in the row space of the
centred training matrix and the penalty is rotation invariant. So the model is fitted on
the full-rank PCA scores of the training rows (no truncation, no whitening), which gives
identical predictions to fitting on all edges, much faster (a unit test checks this).
The shortcut is used **only for pure L2**. The elastic net's L1 term is not rotation
invariant, so the elastic net is always fitted on the original standardised features, and
combining it with any PCA basis raises an error (tested). Every scaler and PCA basis is
fitted on training rows only: on each inner-training fold during the C search, then on
the fold's training split. Tests check the exact row counts of every fit and that the
runner passes the model nothing but the fold's training rows.

**MLP** (`models/mlp.py`, PyTorch). 19,900 → 256 → 64 → 1, ReLU, dropout 0.5, Adam
(learning rate $10^{-3}$, weight decay $10^{-3}$), batch size 64. Early stopping (patience
20, at most 200 epochs) uses an inner stratified 15% validation split of the training rows,
and the best weights are restored. The architecture is fixed in the config, not tuned:
only the stopping epoch is data-driven.

**Calibration** (`models/calibration.py`). Temperature scaling,
$\hat p = \sigma(z/T)$, with $T$ minimising the calibration-split NLL; Platt scaling is
available as an option. Temperature scaling is monotone in the logit, so AUC is unchanged.
For binary LAC, the true-label score is a monotone function of the signed margin
$\pm z$ for every $T$, so **conformal sets are also unchanged** and reusing the
calibration split for conformal prediction (phase 5) is harmless. That is not true for
Platt scaling, for APS or for the EM prior adjustment. **Phase 5 (confirmed):** the
temperature and the conformal quantile are fitted on disjoint halves of the calibration
split, for every method (§4.7).

**Metrics.** AUC; balanced accuracy at threshold 0.5; ECE of $P(\text{ASD})$ with 10
equal-width bins, $\sum_m \tfrac{|B_m|}{n}|\bar y_{B_m} - \bar p_{B_m}|$; Brier score.
LOSO reports pooled-test metrics (all deployment sites' predictions for one seed
together), plus the mean and the minimum of the per-site AUC. All are summarised as
mean ± SD over the 10 seeds.

Binned ECE is biased upward in small samples, so ECE values from test sets of different
sizes are not comparable (189 subjects per within-site test set, 944 pooled LOSO
predictions, 25–172 per LOSO site). Every ECE is therefore reported next to a
**size-matched null ECE**: the mean ECE over 200 label draws $y_i \sim
\text{Bernoulli}(\hat p_i)$ using the same probabilities, i.e. what a perfectly calibrated
model would show at that size (`null_ece`). An ECE is read as miscalibration only when it
clearly exceeds its null. **Calibration-in-the-large** per site is
$\bar{\hat p} - \bar y$ (mean predicted P(ASD) minus the observed ASD fraction). Under
perfect calibration its SD is $\sqrt{\sum_i \hat p_i(1-\hat p_i)}/n$, which gives the
grey 95% band in Figure 6. Pooled LOSO calibration can look perfect while site offsets
of opposite sign cancel, so the per-site view is the one that matters for phase 5.
`shiftguard summarize <run_dir>` rebuilds every table and Figure 6 from a run's saved
predictions without refitting.

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
calibration data only in the *few labelled target subjects* scenario: $k \in \{0, 5, 10,
20\}$ labelled deployment subjects form that site's calibration group. The labelled
subjects are nested random draws (the 5 are among the 10, which are among the 20), and
every arm is evaluated on the subjects outside the 20-subject draw, so all $k$ (and split
CP, in the paired comparison) are scored on the same people. Groups without calibration
data, or with fewer than $(1-\alpha)/\alpha$ points, get full sets.

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

**Likelihood-ratio estimation** (`shift/domain.py`). A domain classifier
$c(x) = \hat P(\text{target} \mid x)$ separates the fold's training rows (source) from the
*unlabelled* deployment rows (target):

1. the features are standardised and reduced by PCA fitted on the training rows **plus
   the unlabelled deployment rows** (transductive, no label used; randomised SVD);
2. L2 logistic regression on the first $k$ components, with $k \in \{10, 20, 50, 100\}$
   and $C \in \{0.01, 0.1, 1\}$ chosen by 5-fold stratified CV log loss on the domain
   task; the out-of-fold **domain AUC** at the chosen values is reported for every site;
3. calibration rows are never used to fit the classifier, so their weights are
   out-of-sample; deployment rows get **cross-fitted** weights (each fold of target rows
   is scored by a classifier that did not see it);
4. $\hat w(x) = \frac{c(x)}{1-c(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}$, with the
   counts of the fit that produced $c$.

**Effective sample size.** Kish's $n_{\text{eff}} = (\sum_i w_i)^2 / \sum_i w_i^2$ of
the calibration weights is reported with every weighted result.

**Clipping (optional).** Weights are clipped to the [1%, 99%] quantiles of the
calibration weights, and the same bounds are applied to test weights. This is a
variance-reduction heuristic that voids the exact guarantee, so it is reported as a
separate method.

### 4.6 Label shift (prior shift)

Under pure label shift, $p_t(x \mid y) = p_s(x \mid y)$ and only the class prior changes.

**EM prior adjustment** (Saerens, Latinne and Decaestecker, 2002; `shift/label_shift.py`).
Starting from the source prior $\pi_s$ (ASD fraction of the calibration sites), alternate

$$
\hat p_t(y \mid x_j) \propto \frac{\pi_t(y)}{\pi_s(y)}\, \hat p_s(y \mid x_j), \qquad
\pi_t(y) \leftarrow \frac{1}{m}\sum_{j=1}^m \hat p_t(y \mid x_j)
$$

over the $m$ **unlabelled** deployment subjects until the prior changes by less than
$10^{-8}$. EM needs calibrated source posteriors, so it is applied to the
temperature-scaled probabilities. When the shift is not pure label shift, the estimate
absorbs part of the other shift too.

**Label-shift weighted CP** (Podkopaev and Ramdas, 2021; `conformal/label_shift.py`).
The likelihood ratio depends on the label only, $w(y) = \hat\pi_t(y)/\pi_s(y)$, so the
candidate label $y$ is in the set when $s(x, y)$ is at most the $(1-\alpha)$ quantile of
$\sum_i p^y_i \delta_{S_i} + p^y_{n+1}\delta_{+\infty}$ with
$p^y_i = w(Y_i) / (\sum_j w(Y_j) + w(y))$ and $p^y_{n+1} = w(y) / (\sum_j w(Y_j) + w(y))$:
one threshold per class. Scores use the EM-adjusted probabilities (applied to calibration
and deployment points alike). With the true priors, marginal coverage is $\ge 1-\alpha$
under pure label shift; with EM priors it is approximate. An **oracle** arm with the
site's true ASD fraction is reported as a reference only (it uses target labels).

**Class-conditional CP under label shift.** Per-class coverage $P(Y \in C(X) \mid Y = k)$
depends only on $p(x \mid y)$, so class-conditional CP keeps its per-class guarantee under
pure label shift without any weights; marginal coverage is then a mixture of per-class
coverages, each $\ge 1-\alpha$. The EM adjustment is a monotone transform of each class's
probability, so it leaves class-conditional sets unchanged. This is why coverage is
reported **per class (ASD and TDC) at every site**.

### 4.7 Phase 5 protocol on ABIDE

The baseline predictions of phase 4 are reused (no refit). In every fold, the calibration
split is divided into two halves stratified by site x diagnosis: **half A** fits the
temperature and **half B** is the conformal calibration set of every method. No
calibration point is used twice; this matters for randomised APS and for the EM
adjustment, which depend on the calibrated probability values (binary LAC sets do not).
Primary model: L2 logistic regression; the MLP is secondary. Primary score: LAC.

| Method | Uses target labels? | Uses target features? |
|---|---|---|
| split, class-conditional (LAC; APS variants) | no | no |
| weighted, weighted (clipped) | no | yes (domain classifier) |
| label-shift (EM) | no | yes (EM prior) |
| label-shift (oracle) | **yes** (true prior), reference only | yes |
| Mondrian, $k$ labels | **yes** ($k$ subjects, not evaluated) | no |
| ComBat + split / class-conditional | no | yes (ComBat, §5) |

Unit tests permute the deployment site's labels and check that the sets of every
method marked "no" are unchanged. For the ComBat arm, the test reruns harmonisation and
the refit as well (§5).

**Coverage level.** The main analysis uses α = 0.1. A sensitivity run at α = 0.2
(`shiftguard conformal <run> --alpha 0.2`, outputs tagged `_alpha0p2`) reports the
coverage–abstention trade-off. At α = 0.2, Mondrian needs only $\lceil 0.8/0.2 \rceil = 4$
labels, so the 5-label arm is no longer a full-set arm.

### 4.8 Abstention

A case is abstained when its prediction set is empty or contains both classes. We report
the abstention rate and the accuracy on non-abstained (singleton) cases.

### 4.9 Validation on synthetic data

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

**ComBat** (Johnson et al., 2007; Fortin et al., 2018; `features/harmonize.py`, via
neuroHarmonize, empirical Bayes) removes an additive and a multiplicative site effect per
edge while preserving age and sex.

What the fit sees, exactly:

- **Inputs:** the connectivity features of the training sites **plus the unlabelled
  features of the deployment site**, the site of each subject, and two protected
  covariates, `AGE_AT_SCAN` and `SEX`.
- **Never an input:** the diagnosis of any subject, at any site. `combat_harmonize`
  raises an error if a diagnosis column (`label`, `DX_GROUP`, `diagnosis`) is passed as a
  covariate. No target label enters the fit.
- **One fit for all folds:** ComBat is fitted once on all 944 subjects. In every LOSO fold,
  each subject is either training, calibration or deployment data, and no label is used,
  so this single fit is identical to a per-fold fit on "training sites + unlabelled
  deployment site". The deployment site's batch parameters can only come from its own
  features, so the step is **transductive**: it assumes the deployment site's unlabelled
  data are available as a batch.

The L2 logistic regression is then refitted on the harmonised features in every LOSO fold
(`configs/experiment/loso_combat.yaml`). Only training-site labels are used for the fit,
and only calibration-site labels for the temperature and the conformal quantile. Split
and class-conditional CP are applied with the same calibration halves as the main
analysis.

**Leakage tests.**
- `test_combat_output_does_not_depend_on_labels` checks that ComBat ignores labels.
- `test_combat_arm_never_reads_deployment_labels` permutes the deployment site's labels
  and reruns the whole chain (ComBat → LOSO refit → temperature → `combat+split` /
  `combat+class_conditional` sets). The harmonised features, the logits and the
  prediction sets must be identical.

Leaving the diagnosis out has a cost: a site's case/control ratio is partly removed as
"site effect".

**ComBat + weighted CP.** For the combined arm, the domain classifier (§4.5) is refitted
on the harmonised features of the same folds. ComBat removes site means and variances
per edge, but the joint (multivariate) distribution of a site can still differ, so the
weights are not trivially 1.

**Describing the site effect** (`evaluation/site_effect.py`, `shiftguard site-effect`,
Figure 2). This is a description on all subjects, not part of any evaluation fold:

- *Site predictability:* balanced accuracy of predicting the site from the connectome
  (standardise → PCA 50 → multinomial L2 logistic regression, all inside 5-fold CV
  stratified by site; chance = 1/18).
- *Variance explained by site:* the 10 leading principal components of the standardised
  features are each regressed on site dummies. We report the variance-weighted mean
  $R^2$, plus the partial $R^2$ of site beyond age, sex and diagnosis. The diagnosis enters
  this description only.

Both are computed before and after ComBat. One caveat for the post-ComBat accuracy:
ComBat equalises site means on the full sample, so each held-out CV fold is shifted the
opposite way. Post-ComBat accuracy can therefore fall slightly *below* chance; that
means "no site signal left", not anti-information.

## 6. Label-free shift detection

`evaluation/detection.py`, `shiftguard detect <conformal run>`, Figure 7. **Exploratory**:
there are 18 data points (deployment sites), and each site's coverage is itself a
binomial estimate from 25–172 subjects.

**Signals.** For every LOSO fold, each signal uses only the deployment site's features
and predictions, never its labels. A unit test permutes the deployment labels and checks
that every signal is unchanged. The reference is calibration half B, which is
exchangeable with the source.

| Signal | Definition |
|---|---|
| domain AUC | out-of-fold AUC of the domain classifier (§4.5), training vs deployment features |
| KS, P(ASD) | two-sample Kolmogorov–Smirnov statistic between half B and the deployment site |
| KS, confidence | the same on the max-probability $\max_y \hat p(y \mid x)$ |
| \|confidence shift\| | \|mean max-probability (deployment) − (half B)\| |
| abstention | split-CP abstention rate at the deployment site |
| \|abstention shift\| | \|abstention (deployment) − abstention (half B, in-sample)\| |
| \|mean P(ASD) shift\| | \|mean P(ASD) (deployment) − (half B)\| |
| \|EM prevalence shift\| | \|EM-estimated deployment ASD fraction (§4.6) − source fraction\| |

Signals are averaged over the 10 seeds per site. Their mean on within-site folds (no
shift) is reported as a reference.

**Targets** (from the labels, for evaluation only), seed-averaged per site:
- coverage drop = $(1-\alpha)$ − split-CP coverage;
- worst-class drop = $(1-\alpha)$ − min(ASD coverage, TDC coverage).

The signed signals (mean P(ASD) shift, EM prevalence shift) are also compared with the
signed class gap (ASD − TDC coverage), to ask whether a signal can tell *which* class
loses coverage.

**Statistics.**
- Spearman $\rho$ across sites.
- 95% percentile bootstrap CI, resampling sites (2000 resamples).
- Two-sided permutation p-value (10,000 permutations of the target ranks).
- Benjamini–Hochberg across all signal × target tests of a setting.
- Settings: logistic regression (primary), MLP, and logistic regression after ComBat.

Two robustness checks are reported for every test:
- **Leave-one-site-out range.** The range of $\rho$ when each site is removed in turn.
- **Partial Spearman correlation given site size** ($n$ deployment subjects). Under no
  shift, KS statistics scale like $1/\sqrt{n}$ (larger for small sites), and a small site's coverage, especially
  its worst class, is more extreme by chance, so site size alone can create a
  correlation. The partial correlation is the Pearson correlation of the rank residuals
  after regressing the ranks of signal and target on the rank of $n$. It has its own
  permutation p-value (residuals permuted) and BH q-value.

A permutation test of $n = 18$ has a minimum detectable $|\rho|$ of about 0.47 at
$p = 0.05$ (before BH).

**What "detected" would mean.** A signal that ranks sites by their coverage drop could
trigger extra labelling (Mondrian) or a refusal to deploy. A null result is reported as
such.

## 7. Statistical inference

- **Clopper–Pearson interval** for $k$ covered out of $n$ (level $1-\gamma$):
  $\big[B(\tfrac{\gamma}{2}; k, n-k+1),\; B(1-\tfrac{\gamma}{2}; k+1, n-k)\big]$, with
  $B(q; a, b)$ the Beta quantile, lower bound 0 if $k=0$ and upper bound 1 if $k=n$.
- **Under-coverage test:** exact one-sided binomial test of
  $H_0: \text{coverage} \ge 1-\alpha$ against $H_1: \text{coverage} < 1-\alpha$, with
  p-value $P\{\mathrm{Bin}(n, 1-\alpha) \le k\}$.
- **Per site** (`evaluation/conformal_summary.py`): each deployment site is evaluated
  with 10 seeds; rates are averaged over seeds. The interval and the test use the
  seed-averaged count $\mathrm{round}(\overline{k})$ out of the site's $n$ evaluated
  subjects, i.e. the uncertainty of a site of that size. Per-class coverage (ASD, TDC)
  is tested the same way within each class.
- **Benjamini–Hochberg** (FDR 5%) across the sites of each (model, method) for marginal
  coverage, and across sites x classes for per-class coverage.
- **Paired bootstrap** between each method and split CP (same model): sites resampled
  with replacement (2000 resamples), statistics = mean site coverage, mean shortfall
  $\max(0, 1-\alpha-\text{coverage})$ and abstention; 95% percentile interval, two-sided
  bootstrap p-value, BH across comparisons within a model. This is the **primary**
  analysis: the site is the unit of deployment and of shift.
- **Two-stage (hierarchical) paired bootstrap** (secondary sensitivity analysis):
  1. draw sites with replacement;
  2. for every drawn site, independently also for repeated draws, draw its subjects with
     replacement;
  3. compute each method's site mean from the subjects' seed-averaged outcomes;
  4. take the statistic (the same three as above) as the mean over drawn sites.

  This also carries the binomial noise of each site's estimate, so its intervals are
  wider. Its p-values get their own BH correction. A plain subject-level bootstrap is
  deliberately not used: it would ignore the between-site variation that the study is
  about.
- **Empirical-Bayes beta-binomial**: $k_s \sim \mathrm{Bin}(n_s, \theta_s)$,
  $\theta_s \sim \mathrm{Beta}(a, b)$, $(a, b)$ by maximum marginal likelihood. Reported as
  the between-site SD of true coverage, $\sqrt{ab/((a+b)^2(a+b+1))}$, and as each site's
  posterior mean $(a + k_s)/(a + b + n_s)$ next to the raw estimate (`coverage_eb` in
  the per-site table).
- Caveat: given one calibration set, conformal coverage is itself random (Beta
  distributed), so a per-site binomial test at $1-\alpha$ is approximate.

## 8. Sensitivity analysis: ROI-level exclusion

`configs/experiment/loso_droprois.yaml` replaces subject exclusion for constant ROIs
(§1) by ROI exclusion: every ROI that is constant in any subject of the final cohort is
removed for all subjects (`qc.zero_variance_policy: drop_rois`), and no subject is
excluded for constant ROIs. This keeps 979 subjects from 19 sites (CMU stays; CALTECH is
still lost to rater-1 failures) with 175 of 200 ROIs (15,225 edges). The primary model
and the conformal methods are rerun on this cohort.
