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
6. **Not a clinical tool.** No result here supports clinical use.
