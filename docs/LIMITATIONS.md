# Limitations

*This file is written before any experiment runs and is updated with concrete findings
in phase 7. It states known risks up front so the results can be read against them.*

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
   (about 17), so correlations between label-free signals and coverage drop have wide
   intervals. A negative result is a valid outcome.
6. **Sites lost to QC.** CALTECH (rater-1 failures) and CMU (constant ROIs) are excluded,
   so 18 of 20 PCP sites remain. Constant ROIs are not zero-filled: that would create an
   artificial site signal. Instead, affected subjects are excluded, because 121 distinct
   ROIs are involved, too many to drop for everyone. Partial field-of-view coverage is
   itself a site property, so the cohort under-represents sites with that problem. A
   sensitivity analysis with ROI-level exclusion (keeping CMU) is planned for phase 5.
7. **Calibration is site-specific, and part of it looks like label shift** (phase 4
   finding). Pooled LOSO probabilities are well calibrated (temperature-scaled logistic
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
8. **Not a clinical tool.** No result here supports clinical use.
