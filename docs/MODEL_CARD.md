# Model card

Models and conformal predictors of ShiftGuard, as evaluated on ABIDE I (CC200
connectomes, 944 subjects, 18 sites). All numbers below come from the generated tables
in `reports/tables/` (see [RESULTS.md](RESULTS.md) for the full set and their run
folders).

## Intended use

- **For:** research on whether conformal coverage guarantees and abstention survive
  acquisition-site shift in multi-site neuroimaging, as a benchmark and as the pilot of a
  PhD proposal.
- **Not for:** clinical diagnosis, screening, triage, or any decision about an
  individual. The models have modest accuracy, and their coverage at a new site is not
  guaranteed (see below). Nothing in this repository has been validated for clinical
  use.

## Models

| Model | Input | Fitting | Status |
|---|---|---|---|
| L2 logistic regression (primary) | Fisher-z correlation connectome, 19,900 edges, standardised | C from {1e-6, ..., 1} by nested 5-fold CV (log loss) on the training split only; fitted on a full-rank PCA rotation of the training data (exact for L2, faster) | used |
| Elastic-net logistic regression | same, original basis (never the PCA basis) | C and l1_ratio by nested CV | implemented, not in the main runs |
| MLP (secondary) | same | 256-64 hidden units, dropout 0.5, weight decay 1e-3, Adam (lr 1e-3), early stopping on a 15% inner split | used |
| Graph neural network | — | — | **not implemented** (optional in the plan; the CPU pipeline does not need it) |

Every model is then:

1. **calibrated** by temperature scaling on half A of the calibration split (sites seen in
   training, subjects not used for fitting);
2. wrapped in **conformal prediction** (LAC score, α = 0.1) calibrated on half B. The
   prediction set is {ASD}, {TDC} or {ASD, TDC}. The model **abstains** when the set is not
   a single class.

Corrections for shift that were evaluated:
- class-conditional CP;
- Mondrian CP with 5–20 labelled subjects from the new site;
- covariate-shift weighted CP;
- label-shift weighted CP with an EM-estimated prior;
- ComBat harmonisation (age and sex preserved, diagnosis never used).

See [METHODS.md](METHODS.md).

## Training and evaluation data

ABIDE I, PCP preprocessing (CPAC, band-pass, no global signal regression), Craddock 200
atlas; 944 subjects (460 ASD, 484 controls) from 18 sites after QC. See
[DATA_CARD.md](DATA_CARD.md). Two protocols:

- **Within-site:** random splits stratified by site and diagnosis. Calibration and test
  data are exchangeable.
- **Leave-one-site-out (LOSO):** each site in turn is the unseen deployment site. It is
  never used for training, calibration or tuning, except in the explicit Mondrian arm.

Ten seeds per protocol.

## Performance

**Discrimination and calibration** ([baselines](../reports/tables/baselines.md)):
- LOSO logistic regression: AUC 0.731 ± 0.008, balanced accuracy 0.672 ± 0.010.
- Worst single deployment site: AUC 0.528 ± 0.043.
- MLP: AUC 0.690 ± 0.014.
- Within-site logistic regression: AUC 0.732 ± 0.030.

After temperature scaling, pooled ECE is 0.032, at the level of a perfectly calibrated
model of the same size (0.034). At 4 of 18 sites, however, the mean predicted P(ASD) is
off by more than chance (up to ±0.22), with offsets of opposite sign that cancel in the
pooled curve.

**Coverage and abstention** ([headline](../reports/tables/conformal_headline.md),
[per site and class](../reports/tables/conformal_sites.md); logistic regression, LOSO,
target 0.90):

| Method | Coverage (pooled) | Worst site | Worst site-class | Abstention | Accuracy when not abstaining |
|---|---|---|---|---|---|
| Split CP | 0.898 | 0.750 | 0.550 | 0.554 | 0.771 |
| ComBat + split CP | 0.910 | 0.856 | 0.793 | 0.564 | 0.793 |
| Mondrian, 20 target labels | 0.910 | 0.820 | 0.741 | 0.579 | 0.788 |

At α = 0.2 (target 0.80), split-CP abstention falls to 0.277, but the worst site-class
coverage falls to 0.357.

## Failure modes (what a user must know)

1. **Coverage is guaranteed on average, not at your site.** Pooled coverage is on target.
   At individual sites it is as low as 0.75, and within one diagnosis at one site as low
   as 0.55.
2. **The failure is class-specific.** The class that loses coverage at a site follows the
   sign of that site's calibration offset (ρ = 0.91 across sites).
3. **Most subjects get no single label.** With AUC ≈ 0.73, more than half of the
   deployment subjects receive both labels at α = 0.1. This is the honest behaviour of a
   conformal predictor with a weak model, not a defect to tune away.
4. **Some corrections make things worse.**
   - Label-shift CP with an EM-estimated prior under-covers badly (0.729), because the
     shift is mostly not a prior shift.
   - Covariate-shift weighting returns near-full sets (abstention 0.86), because every
     site is almost perfectly separable from the training sites.
5. **A label-free warning exists for the direction only.** Without labels from the new
   site, the shift in mean P(ASD) relative to the calibration data predicts *which* class
   will lose coverage (ρ = 0.83). No label-free signal predicted *how much* coverage would
   drop. The direction signal was identified post hoc on the same sites; see
   [LIMITATIONS.md](LIMITATIONS.md) §5.

## Subgroups and known biases

Training data are 85% male, mostly adolescents and young adults (median age 14), and
mostly cognitively able (median FIQ 108). Sites barely overlap in age. Performance and
coverage for women, young children and people with intellectual disability are **not
established**. Per-site results are in the tables; per-subgroup coverage was not
analysed.

## Ethical considerations

Autism is a heterogeneous, lifelong condition. A binary ASD/control label from a brain
scan does not capture it, and such a model must not be used to label individuals. The
data are retrospective research data shared under the ABIDE terms. No attempt is made to
re-identify participants, and no subject-level data or predictions are published.

## Caveats and recommendations

- Report coverage per site and per class, never only pooled.
- Before deploying a conformal predictor at a new site, collect at least
  ⌈(1 − α)/α⌉ labelled subjects there (9 at α = 0.1) for site-conditional calibration,
  or treat the guarantee as void.
- The results come from one dataset (ABIDE I) and one atlas and pipeline. They have not
  been replicated on an independent multi-site cohort.
