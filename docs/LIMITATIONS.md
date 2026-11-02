# Limitations

*Items 1–6 were written before any experiment ran, as the known risks; they have since been
updated with what the experiments showed. Items 7–12 come from the results.*

1. **Modest discriminative performance.** ASD vs control classification from resting-state
   connectivity on ABIDE is modest in the literature, especially under leave-one-site-out
   evaluation. This project studies the *reliability of the coverage guarantee*, not
   state-of-the-art accuracy. Low accuracy makes prediction sets larger and abstention
   frequent. That is the honest behaviour of a conformal predictor with a weak model.
2. **Sites differ in more than the scanner.** Age range, sex ratio, IQ, diagnostic
   procedure and eyes-open/closed instructions vary by site. A "site shift" therefore mixes
   covariate shift, possible label shift (case/control ratio) and possible concept shift.
   Weighted conformal prediction only corrects covariate shift.
3. **ComBat assumptions.** ComBat assumes additive and multiplicative site effects per
   feature, with empirical-Bayes pooling. For an unseen site its parameters must be
   estimated from that site's own unlabelled features (a transductive step, labelled as such).
4. **Small sites.** Some sites contribute few subjects per class, so per-site coverage
   estimates are noisy. Exact Clopper–Pearson intervals and a hierarchical (empirical-Bayes)
   estimate are reported, but power to detect under-coverage at a single site is limited.
5. **Label-free detection is exploratory.** There are only as many data points as sites
   (18), so any correlation needs $|\rho| \gtrsim 0.47$ to reach p < 0.05 before
   correction, and the bootstrap intervals are wide. Each site's coverage is itself a
   binomial estimate (25–172 subjects), and small sites have larger KS statistics and
   more extreme class coverage by chance, so a size-adjusted (partial) correlation is
   reported next to each one. Result:
   - **Negative:** no label-free signal predicts the marginal coverage drop.
   - **Robust:** the signed mean P(ASD) shift (and the EM prevalence shift) predicts
     which class loses coverage.
   - **Not robust to site size:** the KS-on-P(ASD) association with the worst-class drop.

   Caveat on the positive result: the mean P(ASD) shift was added *after* phase 5 had
   shown, on the same sites and using labels, that the calibration offset predicts the
   class gap. It is the label-free counterpart of that offset. Its success therefore
   confirms a hypothesis formed on the same data and is not an independent test. The
   EM shift was requested in advance, and the other signals follow the original plan.
   Nothing here has been replicated on an independent multi-site cohort.
6. **Sites lost to QC.** CALTECH (rater-1 failures) and CMU (constant ROIs) are excluded,
   so 18 of 20 PCP sites remain. Constant ROIs are not zero-filled: that would create an
   artificial site signal. Instead, affected subjects are excluded, because 121 distinct
   ROIs are involved, too many to drop for everyone. Partial field-of-view coverage is
   itself a site property, so the cohort under-represents sites with that problem. A
   sensitivity analysis with ROI-level exclusion (25 ROIs dropped; 979 subjects, 19
   sites, CMU kept) gives the same conclusions: split CP coverage 0.895 vs 0.898, worst
   site LEUVEN_1 in both, and EM label-shift CP fails in both.
7. **Calibration is site-specific** (phase 4 finding; phase 5 shows it is mostly *not*
   label shift). Pooled LOSO probabilities are well calibrated (temperature-scaled logistic
   regression: ECE 0.032 vs a size-matched null of 0.034). Per deployment site, however,
   the mean predicted P(ASD) misses the observed ASD fraction by more than a perfectly
   calibrated model would at that site size, at 4 of 18 sites for logistic regression
   (KKI +0.22, LEUVEN_1 −0.21, USM −0.16, PITT −0.13) and 2 of 18 for the MLP (KKI,
   USM). These are descriptive counts, not multiplicity-corrected tests. A
   calibration-in-the-large offset is what a change in case/control ratio or a site-wide
   shift in the logit produces. Weighted conformal prediction corrects covariate shift
   only, so these sites are where it is most likely to fail. Per-site ECE values (25–172
   test subjects) are dominated by small-sample bias and are only interpreted next to
   their null.
   Phase 5 tests the label-shift reading directly. For source-calibrated probabilities,
   pure label shift moves the mean predicted P(ASD) by $-\Delta\pi\,(1 - \text{slope})$,
   which is at most the true prevalence change. The observed offsets exceed that bound by
   more than 2 SD at 5 of 18 sites (KKI, LEUVEN_1, PITT, STANFORD, UCLA_2), and label
   shift accounts for only 33% of the summed squared offsets. Only USM (and UM_2) look like
   a prior shift. The rest is a shift in $p(x \mid y)$ (acquisition, cohort or
   diagnostic practice), which neither label-shift nor class-conditional CP corrects.
8. **EM prior estimation is unstable with a weak classifier.** The Saerens et al. (2002)
   EM estimate of the deployment ASD fraction assumes pure label shift and calibrated
   source posteriors. With a discrimination slope of 0.14–0.22, any site-wide logit shift
   is read as a prior shift about five times larger. The estimate runs to within 0.05 of
   0 or 1 at 5 of 18 sites. It is noisy even without shift (mean |error| 0.081 on
   within-site folds). Label-shift-weighted CP built on it under-covers (0.729). This
   is a negative result for this correction **on this data**, not for the method under
   its assumptions (the oracle-prior arm behaves as expected).
9. **Covariate-shift weights degenerate.** Every deployment site is almost perfectly
   separable from the training sites (domain AUC 0.78–1.00). The likelihood-ratio weights
   leave an effective sample size of 3.7–35 calibration points, so weighted CP mostly
   returns full sets (abstention 0.83–0.86). Its coverage is bought by abstention.
   After ComBat the domain classifier is at chance (median AUC 0.45), so the weights are
   near-uniform and ComBat + weighted CP equals ComBat + split CP.
10. **ComBat is transductive and removes some signal.** Batch parameters for the
    deployment site are estimated from that site's unlabelled features, so the
    deployment data must be available as a batch. Because the diagnosis cannot be a
    covariate, a site's case/control mix is partly absorbed into its "site effect". In
    this benchmark, ComBat removed the site signal (site balanced accuracy 0.55 → 0.05,
   chance 0.06), raised the refitted model's LOSO AUC from 0.731 to 0.749 and
    gave the best worst-site coverage among label-free methods. The improvement in mean
    shortfall is modest: −0.017, 95% CI [−0.035, −0.002] with the site bootstrap. It is
    not significant under the two-stage bootstrap, which also resamples subjects within
    sites: [−0.038, +0.000], q = 0.076.
11. **Abstention is high because the classifier is weak.** At α = 0.1, more than half
    of the deployment subjects receive both labels (split CP abstention 0.554). At
    α = 0.2 this halves (0.277), but the worst site-class coverage falls to 0.36. Lower
    abstention does not repair the site-level failures.
12. **Per-site power is low.** With 25–172 deployment subjects per site (and fewer per
    class), no site is significantly below 90% marginal coverage after
    Benjamini–Hochberg under split CP, although 11 of 18 are below target. Site-level
    statements rely on the paired bootstrap across sites and the empirical-Bayes
    between-site SD (0.030), not on single-site tests.
13. **Not a clinical tool.** No result here supports clinical use.
