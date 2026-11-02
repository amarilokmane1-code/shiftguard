# ShiftGuard

**Does conformal abstention survive site shift? A multi-site autism neuroimaging benchmark.**

[![CI](https://github.com/amarilokmane1-code/shiftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/amarilokmane1-code/shiftguard/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status: work in progress (phase 5 of 9: conformal prediction under site shift).**
> Baselines and conformal methods have been run on ABIDE; label-free detection (phase 6)
> is still `TBD`. Every number is taken from saved run outputs.

> **Main finding so far: pooled numbers hide site-level failure.** Pooled over
> deployment sites, the leave-one-site-out model is calibrated (ECE 0.032 vs 0.034 for a
> perfectly calibrated model of the same size) and split conformal prediction covers
> 89.8% of subjects at a 90% target. Per site, calibration offsets of opposite sign
> (KKI +0.22, LEUVEN_1 −0.21) **cancel out in the pooled curve**, coverage ranges from
> 0.75 to 0.95, and coverage within one diagnostic class drops to 0.55. The sign of a
> site's offset predicts which class loses coverage (Spearman ρ = 0.91, 18 sites).

![Coverage per deployment site](reports/figures/fig3_coverage_per_site.png)

*Figure 3. Coverage per unseen deployment site (logistic regression, LOSO, 10 seeds) with
Clopper–Pearson 95% intervals; vertical line = 90% target; filled = significantly
below target after Benjamini–Hochberg.*

## The question

A classifier predicts autism spectrum disorder (ASD) versus typical controls from
resting-state functional connectivity (ABIDE I). It is wrapped in **conformal prediction**,
so instead of a single label it returns a *prediction set* that should contain the true
label with a chosen probability (for example 90%), and it **abstains** when that set is not a
single class.

That guarantee rests on one assumption: calibration and deployment data are
*exchangeable*. A new hospital, a new scanner or a new protocol breaks that assumption.
ShiftGuard asks:

1. Does the coverage guarantee still hold when the model is deployed on a **site it has
   never seen** (leave-one-site-out)?
2. Which corrections actually protect it: **ComBat harmonisation**, **Mondrian
   (site-conditional) conformal** with a few labelled target subjects, **weighted
   conformal prediction** for covariate shift, or combinations of these?
3. Can a loss of coverage on a new site be **detected without labels**?

## Why it matters

Clinical prediction models are trained on data from a few centres and then deployed
somewhere else. A model that says "I am 90% sure" has to keep that promise at the new
site, or it has to know when to abstain. This project is the pilot study for a PhD
proposal on *abstention that survives distribution shift* in autism research.

## What is done

| Component | Status |
|---|---|
| Scaffold, configs, CLI, CI | done (phase 1) |
| Conformal core + synthetic tests | done (phase 2): see [synthetic validity checks](reports/tables/conformal_synthetic.md) |
| ABIDE download, QC, connectomes | done (phase 3): 944 subjects, 18 sites, see [QC summary](reports/qc_summary.md) and [Figure 1](reports/figures/fig1_data_map.png) |
| Baselines: within-site vs LOSO, calibration | done (phase 4): see [baselines](reports/tables/baselines.md), [per-site LOSO](reports/tables/loso_sites.md) and [Figure 6](reports/figures/fig6_reliability.png) |
| Conformal under shift + statistics | done (phase 5): see [site effect](reports/tables/site_effect.md) and [Figure 2](reports/figures/fig2_site_effect.png), [headline](reports/tables/conformal_headline.md), [per site and class](reports/tables/conformal_sites.md), [paired comparisons](reports/tables/conformal_comparisons.md), [prevalence shift](reports/tables/prevalence_shift.md), Figures [3](reports/figures/fig3_coverage_per_site.png), [3b](reports/figures/fig3b_class_coverage.png), [4](reports/figures/fig4_target_labels.png), [5](reports/figures/fig5_abstention_per_site.png), [S1](reports/figures/figS1_prevalence_shift.png) |
| Label-free shift detection | TBD (phase 6) |
| Report and documentation | TBD (phase 7) |
| Kaggle notebook | TBD (phase 8) |

## Key results

*Copied from [conformal_headline.md](reports/tables/conformal_headline.md) (run
`2026-11-02_130158_conformal`) and the baseline runs. Nothing here is estimated by hand.*

L2 logistic regression, LAC score, α = 0.1 (target coverage 0.90), 10 seeds. Coverage gap
= mean over sites of max(0, 0.90 − site coverage). Worst site-class = lowest coverage
within one diagnosis at one site. Abstention = prediction set is not a single class.
Accuracy (kept) = accuracy on the subjects that received a single label.

| Setting | Method | AUC | Balanced acc. | Marginal coverage | Worst-site coverage | Worst site-class | Coverage gap | Abstention | Accuracy (kept) |
|---|---|---|---|---|---|---|---|---|---|
| Within-site | Split CP | 0.732 | 0.668 | 0.911 | – | – | – | 0.593 | 0.791 |
| LOSO | Split CP | 0.731 | 0.672 | 0.898 | 0.750 | 0.550 | 0.031 | 0.554 | 0.771 |
| LOSO | Class-conditional CP | 0.731 | 0.672 | 0.901 | 0.782 | 0.607 | 0.030 | 0.564 | 0.773 |
| LOSO | Label-shift CP (EM prior) | 0.731 | 0.672 | **0.729** | 0.417 | 0.000 | 0.209 | 0.324 | 0.626 |
| LOSO | Weighted CP (covariate shift, clipped) | 0.731 | 0.672 | 0.964 | 0.892 | 0.869 | 0.001 | 0.858 | 0.752 |
| LOSO | ComBat + split CP | 0.749 | 0.684 | 0.910 | 0.856 | 0.793 | 0.014 | 0.564 | 0.793 |
| LOSO | ComBat + weighted CP (clipped) | 0.749 | 0.684 | 0.911 | 0.843 | 0.764 | 0.015 | 0.571 | 0.793 |
| LOSO | Mondrian (20 target labels) | 0.731 | 0.672 | 0.910 | 0.820 | 0.741 | 0.010 | 0.579 | 0.788 |

Mondrian is scored on the subjects outside its 20 labelled ones (as is the matching split
CP in the paired comparison). With 0 or 5 target labels it returns full sets (100%
abstention), because a group needs at least 9 points at α = 0.1.

**Why so much abstention?** A conformal predictor only returns a single label when the
model is confident enough to meet the coverage target. With AUC ≈ 0.73, most subjects
sit near the decision boundary, so at 90% coverage more than half get both labels. Asking
for less coverage buys fewer abstentions. The α = 0.2 sensitivity run
([headline](reports/tables/conformal_headline_alpha0p2.md),
run `2026-11-02_132458_conformal`) halves abstention, but site failures are not removed:

| α (target) | Method | Marginal coverage | Worst-site coverage | Worst site-class | Abstention |
|---|---|---|---|---|---|
| 0.1 (0.90) | Split CP | 0.898 | 0.750 | 0.550 | 0.554 |
| 0.1 (0.90) | Class-conditional CP | 0.901 | 0.782 | 0.607 | 0.564 |
| 0.1 (0.90) | ComBat + split CP | 0.910 | 0.856 | 0.793 | 0.564 |
| 0.2 (0.80) | Split CP | 0.798 | 0.636 | 0.357 | 0.277 |
| 0.2 (0.80) | Class-conditional CP | 0.800 | 0.646 | 0.357 | 0.285 |
| 0.2 (0.80) | ComBat + split CP | 0.804 | 0.684 | 0.607 | 0.264 |

At α = 0.2, split CP leaves two site-classes significantly below target (LEUVEN_1 ASD
0.36, KKI controls 0.48; BH q < 0.05). ComBat again reduces the mean shortfall (−0.018,
95% CI [−0.036, −0.002]) at no abstention cost (−0.000).

### Conformal prediction under site shift (phase 5)

Per-site and per-class tests use the seed-averaged count with Clopper–Pearson intervals
and Benjamini–Hochberg across sites; method differences use a paired bootstrap over
sites (2000 resamples, 95% CI, BH q).

1. **Split CP holds on average but not per site or per class.** No single site is
   significantly below 90% after BH (sites have 25–172 subjects, so power is low), but
   11 of 18 sites are below target, the empirical-Bayes between-site SD of true coverage
   is 0.030, and within-class coverage reaches 0.55 (ASD at LEUVEN_1). Pooled coverage
   hides this in the same way the pooled calibration curve does.
2. **The shift is mostly *not* a prior (label) shift.** The true ASD fraction at KKI,
   LEUVEN_1 and PITT is within 0.08 of the training sites, yet their calibration offsets
   are +0.22, −0.21 and −0.13: about 4, 21 and 4 times what a prior shift of that size
   can produce. At 6 of 18 sites the offset even has the opposite sign (mostly small
   offsets). For source-calibrated probabilities, pure label shift can move the mean
   predicted P(ASD) by at most (true shift) × (1 − discrimination slope). That bound is
   exceeded by more than 2 SD at 5 sites (KKI, LEUVEN_1, PITT, STANFORD, UCLA_2), and it
   accounts for 33% of the summed squared offsets. **USM is the one clear prior shift**
   (ASD fraction 0.65 vs 0.47; offset −0.16, predicted −0.15). UM_2 is also consistent.
3. **Label-shift correction fails (negative result).** The EM prior estimate (Saerens et
   al., 2002) on unlabelled deployment data does not track the true ASD fraction
   (Spearman ρ = 0.11 across sites). The classifier separates the classes weakly
   (discrimination slope 0.14–0.22), so EM reads a site-wide logit shift as a huge
   prior shift and runs to within 0.05 of 0 or 1 at 5 of 18 sites (KKI: 0.999). Mean
   |offset| grows from 0.090 to 0.276 after adjustment. Label-shift-weighted CP with EM priors falls to 0.729
   coverage (−0.188 vs split, 95% CI [−0.257, −0.121]), significantly below target at
   12 of 18 sites. With the *true* prior (oracle, uses target labels) it equals split CP
   (+0.004, n.s.), because the true prior shift is small at most sites.
4. **Class-conditional CP does not rescue per-class coverage.** Its per-class guarantee
   holds under pure label shift only. KKI controls (0.69) and LEUVEN_1 ASD (0.61) are
   significantly below target after BH, as expected when the class-conditional feature
   distribution itself shifts.
5. **ComBat is the best label-free correction here.** Before harmonisation, the site
   can be predicted from the connectome with balanced accuracy 0.55 (chance 0.06), and
   site explains 13% of the variance of the 10 leading components. After ComBat (age and
   sex preserved, no diagnosis, fitted on the training sites plus the unlabelled
   deployment site) both are at chance level (0.05; 0.5%) ([Figure 2](reports/figures/fig2_site_effect.png)).
   ComBat + split CP raises the worst site from 0.750 to 0.856 and the worst site-class
   from 0.55 to 0.79. Mean shortfall changes by −0.017 [−0.035, −0.002] (q = 0.043), and
   abstention by +0.031 (n.s.). Adding covariate-shift weights on top changes nothing:
   after ComBat the domain classifier is at chance (median AUC 0.45), so the weights are
   close to uniform (ESS 38–134).
6. **A few target labels help, at a price.** Mondrian CP with 20 labelled deployment
   subjects adds +0.033 coverage [0.009, 0.059] for +0.091 abstention; with 10 labels it
   adds +0.028 for +0.112 abstention.
7. **Covariate-shift weighting degenerates.** A domain classifier separates every
   deployment site from the training sites (out-of-fold AUC 0.78–1.00). The effective
   sample size of the weights is 3.7–35 calibration points, so the sets are nearly full
   (abstention 0.83 unclipped, 0.86 clipped). Its high coverage comes from abstaining,
   not from correcting the shift.
8. **Sensitivity analysis: ROI-level exclusion.** Dropping the 25 ROIs that are constant
   in any subject, instead of excluding subjects, keeps 979 subjects and 19 sites (CMU
   returns). The conclusions do not change
   ([headline](reports/tables/conformal_headline_droprois.md)):
   - LOSO AUC is 0.725; split CP covers 0.895 (worst site LEUVEN_1 at 0.721, worst
     site-class 0.45);
   - EM label-shift CP falls to 0.725;
   - the offset predicts the class gap (ρ = 0.90), and label shift accounts for 31% of
     the squared offsets;
   - Mondrian with 20 labels adds +0.030 coverage;
   - CMU itself is covered at 0.930.
9. **Preview for phase 6 (label-free detection).** The EM-estimated prevalence shift,
   computed without labels, strongly predicts *which class* loses coverage at a site
   (ρ = 0.82 with the ASD − TDC coverage gap). It does **not** predict how far marginal
   coverage drops (ρ = 0.17, p = 0.49).

The MLP gives the same picture (headline table). Within-site split CP covers 0.911.

> These are coverage properties of a weak classifier (AUC 0.73), with more than half of
> subjects receiving both labels. Nothing here supports clinical use.

### Baselines (phase 4)

944 subjects, 18 sites, CC200 correlation connectomes (19,900 edges), 10 seeds, mean ± SD
over seeds. Run `2026-11-01_223647_loso`; the full table is in
[reports/tables/baselines.md](reports/tables/baselines.md).

| Protocol | Model | AUC | Worst-site AUC | Balanced acc. | ECE (calibrated) | Null ECE at same n |
|---|---|---|---|---|---|---|
| Within-site | L2 logistic regression | 0.732 ± 0.030 | – | 0.668 ± 0.022 | 0.080 ± 0.017 | 0.074 |
| Within-site | MLP | 0.692 ± 0.030 | – | 0.620 ± 0.031 | 0.078 ± 0.021 | 0.071 |
| LOSO | L2 logistic regression | 0.731 ± 0.008 | 0.528 ± 0.043 | 0.672 ± 0.010 | 0.032 ± 0.006 | 0.034 |
| LOSO | MLP | 0.690 ± 0.014 | 0.468 ± 0.061 | 0.628 ± 0.018 | 0.029 ± 0.008 | 0.033 |

Pooled over sites, leaving a site out costs almost nothing: logistic-regression AUC is
0.731 under LOSO vs 0.732 within-site. The shift shows up **per site**:
- AUC ranges from 0.53 (OHSU) to 0.84 (USM).
- At 4 of 18 deployment sites, the mean predicted ASD probability is off from the
  observed ASD fraction by more than chance (KKI +0.22, LEUVEN_1 −0.21).

**The offsets cancel in the pooled curve**: positive and negative site offsets average
out, so the pooled LOSO model looks calibrated while individual sites are not (see
[LIMITATIONS §7](docs/LIMITATIONS.md)). Phase 5 shows the same cancellation for coverage. The MLP has lower AUC and balanced accuracy and a
higher Brier score than logistic regression. After temperature scaling, both are
calibrated to within their null ECE.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). In short: ABIDE classification accuracy
is modest in the literature, sites differ in age range and acquisition protocol, ComBat
makes distributional assumptions, and the label-free detection part is exploratory.
**This is research software. It is not a diagnostic tool.**

## Reproduce

```bash
pip install -e ".[harmonize]"
shiftguard all --config configs/experiment/loso.yaml
shiftguard report
```

On a CPU-only machine, install PyTorch from the CPU index first so pip does not fetch the
much larger CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For development (lint, tests, pre-commit):

```bash
pip install -e ".[dev,harmonize]"
python tasks.py lint
python tasks.py test
```

Kaggle notebook: *link TBD*.

## Repository layout

```
src/shiftguard/   data, features, models, conformal, shift, evaluation, viz, cli.py
configs/          YAML configs (data, model, conformal, experiment)
tests/            unit, leakage and synthetic end-to-end tests
notebooks/        exploration, results, Kaggle
reports/          generated tables and figures
docs/             data card, model card, methods, results, limitations
```

## Data

ABIDE I Preprocessed (Preprocessed Connectomes Project). The data are downloaded by the
code, **never committed**, and subject to their own terms of use. See
[docs/DATA_CARD.md](docs/DATA_CARD.md).

## Citation

See [CITATION.cff](CITATION.cff).

## Author

**Lokmane Amari**, ORCID [0009-0006-2358-5912](https://orcid.org/0009-0006-2358-5912).

Related work:

> Amari L., Boukhiar N., Aissiou I., Bensefia Y. "Immun-AI: A Leakage-Safe, Explainable
> Multimodal Decision-Support Platform for Systemic Lupus Erythematosus Diagnosis."
> MICAD 2026, Springer Lecture Notes in Electrical Engineering (proceedings forthcoming).
