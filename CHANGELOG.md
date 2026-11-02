# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Phase 6: label-free shift detection (`shiftguard detect <conformal run>`,
  `evaluation/detection.py`). Eight signals per deployment site, computed from features
  and predictions only (a permutation test checks they ignore deployment labels):
  domain AUC, KS on P(ASD) and on confidence, confidence shift, abstention and its shift,
  mean P(ASD) shift, EM prevalence shift. They are related to the marginal coverage drop,
  the worst-class drop and the signed class gap. Statistics: Spearman with bootstrap CI,
  permutation p, BH, leave-one-site-out range, and partial correlation given site size.
  Within-site folds give the no-shift reference. Settings: logistic regression, MLP,
  ComBat. Tables `detection_correlations`, `detection_signals`; Figures 7 and 7b.
- Phase 6 results (run `2026-11-02_150802_detection`):
  - No signal predicts the marginal coverage drop (all q >= 0.23).
  - The signed mean P(ASD) shift predicts which class loses coverage (rho = 0.83,
    q = 0.001, size-adjusted 0.82; MLP 0.86); the EM shift does the same (0.82).
  - KS on P(ASD) vs worst-class drop (rho = 0.67, q = 0.019) does not survive
    adjustment for site size (partial 0.53, q = 0.14).
  - Abstention rate is not an alarm (rho = -0.22).
- Phase 5: conformal prediction under site shift. `shiftguard conformal <baseline run>
  [--combat <run>]` reuses saved predictions. The calibration split is divided into
  halves stratified by site × diagnosis: one fits the temperature, the other is the
  conformal calibration set. Methods: split, class-conditional (LAC; randomised and
  deterministic APS), Mondrian with 0/5/10/20 labelled target subjects (nested draws,
  common evaluation set), covariate-shift weighted CP (PCA + L2 domain classifier with
  CV, cross-fitted target weights, clipped variant), label-shift weighted CP (Podkopaev &
  Ramdas) with EM priors (Saerens et al., 2002) and an oracle-prior reference, and ComBat
  (neuroHarmonize, empirical Bayes, age + sex only; diagnosis forbidden) followed by a
  refit (`configs/experiment/loso_combat.yaml`). A permutation test checks that no
  label-free method uses deployment labels.
- Phase 5 statistics: Clopper–Pearson per site and per site × class, Benjamini–Hochberg,
  paired site bootstrap vs split CP, beta-binomial empirical Bayes (between-site SD).
  Prevalence diagnostic: EM prior vs true prior, the pure-label-shift bound on
  calibration-in-the-large (tested identity), offset vs class-coverage gap, and the EM
  shift as a preview label-free signal. Tables `conformal_headline`, `conformal_sites`,
  `conformal_comparisons`, `prevalence_shift`; Figures 3, 4, 5, 7.
- `shiftguard conformal --alpha`: miscoverage sensitivity analysis with tagged outputs.
  At α = 0.2 (run `2026-11-02_132458_conformal`), split CP abstention falls from 0.554 to
  0.277, but the worst site-class coverage falls to 0.36. ComBat + split: 0.804 coverage,
  worst site 0.684, abstention 0.264.
- Leakage test for the ComBat arm. It permutes the deployment labels and reruns ComBat,
  the LOSO refit and the `combat+` sets, which must be identical.
- `report_tag` keeps variant runs (ComBat, drop-ROIs) from overwriting main tables;
  `qc.zero_variance_policy: drop_rois` for the ROI-level exclusion sensitivity analysis
  (`configs/experiment/loso_droprois.yaml`).
- Brief items completed in phase 5:
  - Site-effect description and Figure 2 (`shiftguard site-effect`): site balanced
    accuracy 0.55 → 0.05 after ComBat (chance 0.06); R^2 of site on the 10 leading
    components 0.13 → 0.005.
  - ComBat + weighted CP arms (domain weights refitted on harmonised features).
  - Accuracy on non-abstained cases.
  - Per-site empirical-Bayes coverage next to the raw estimate.
  - Figure 5: abstention and accuracy per site.
  - Figures renumbered to the brief: 3b = class coverage, 4 = target labels, S1 =
    prevalence shift.
- Label-free signals recorded in every conformal fold (KS tests, confidence, abstention,
  mean P(ASD) on calibration half B vs the deployment site), for phase 6.
- Drop-ROIs sensitivity analysis (run `2026-11-02_131805_conformal`, 979 subjects, 19
  sites): same conclusions as the main cohort.
- Phase 5 results (run `2026-11-02_130158_conformal`): split CP 0.898 pooled, worst site
  0.750, worst site-class 0.55; the site offset predicts which class loses coverage
  (ρ = 0.91). The shift is mostly not a prior shift (only USM clearly is). EM-based
  label-shift CP fails (0.729 coverage). ComBat + split gives the best label-free
  worst-site coverage (0.856) at similar abstention. Covariate-shift weights degenerate
  (ESS 3.7–35). Mondrian with 20 labels: +0.033 coverage for +0.091 abstention.
- Phase 4: baselines. Nested-CV L2 logistic regression (exact full-rank PCA rotation
  speed-up, tested); PyTorch MLP with early stopping; temperature / Platt calibration;
  ECE and reliability tables. LOSO and within-site protocols with a leakage guard on
  every split. Synthetic multi-site cohort for CI. `shiftguard run` (per-subject
  predictions in the run folder, `reports/tables/baselines.*`, `reports/tables/loso_sites.*`,
  Figure 6 reliability diagrams). Configs restructured (`models:` section, `calibration.yaml`).
- Phase 4 calibration diagnostics: size-matched null ECE (`null_ece`), per-site
  calibration-in-the-large, and a per-site LOSO panel in Figure 6. New command
  `shiftguard summarize <run_dir>` rebuilds tables and figures from saved predictions.
- Phase 4 checks: the elastic net is never fitted in the PCA-rotated basis (an error if
  combined with `pca_components`), and every scaler/PCA basis is fitted on training rows
  only (tests count the rows of every fit).
- Phase 4 results (run `2026-11-01_223647_loso`, 944 subjects, 18 sites, 10 seeds): LOSO
  AUC 0.731 ± 0.008 (logistic regression) and 0.690 ± 0.014 (MLP); within-site 0.732 ±
  0.030 and 0.692 ± 0.030. Calibration offsets beyond chance at 4/18 (logistic
  regression) and 2/18 (MLP) deployment sites.
- Zero-variance ROI decision: 121 distinct constant ROIs among the 26 affected subjects
  (> 5), so the strict subject rule stays the main analysis; ROI-level exclusion is
  planned as a phase 5 sensitivity analysis. Constant ROIs are never zero-filled.

- Phase 3: ABIDE data pipeline. Downloader (`nilearn.fetch_abide_pcp`, plus a parallel
  HTTPS fallback with retries to the same layout; refuses to run when `SHIFTGUARD_OFFLINE`
  is set). Phenotype cleaning (label 1 = ASD). Audited QC with seven ordered exclusion
  reasons and four rater rules. Connectivity features (correlation, partial correlation,
  tangent), where the tangent reference can only be fitted inside a fold.
  `shiftguard download` and `shiftguard features`. `reports/qc_summary.{csv,md}`,
  `reports/tables/cohort_summary.{csv,md}`, Figure 1 (`reports/figures/fig1_data_map`).
  Shared figure style (colour-blind-safe categorical order, PNG 300 dpi + PDF).
- Phase 3 decisions recorded: LAC primary / randomised APS secondary; the Mondrian
  5-label arm is kept (needs >= 9 labelled target subjects at alpha = 0.1); domain
  classifier = PCA (train + unlabelled target, transductive) + L2 logistic regression.
- Phase 2 — conformal core written from scratch (`shiftguard.conformal`): LAC and APS
  scores (deterministic and randomised), split conformal, class-conditional, Mondrian
  (group-conditional) and weighted conformal prediction with weight clipping and Kish
  effective sample size; abstention metrics. Logistic domain-classifier likelihood-ratio
  estimator (`shiftguard.shift.weights`). Exact Clopper–Pearson interval and one-sided
  binomial test (`shiftguard.evaluation.statistics`).
- Synthetic validity experiments (`shiftguard.evaluation.simulation`) and the
  `shiftguard simulate` command, writing `reports/tables/conformal_synthetic.{csv,md}`.
- Tests: exact exchangeable coverage (k/(n+1), both tails), covariate shift breaking split
  CP and weighted CP restoring it, class-conditional and Mondrian protection of a
  minority/noisy group, hand-computed quantile examples.
- docs/METHODS.md §4 (conformal) and §7 (coverage inference) with equations.
- Phase 1 — project scaffold: `src/` package layout, `pyproject.toml` (Python 3.11),
  YAML configs with inheritance and `--set` overrides, Typer CLI skeleton, timestamped
  logging, seeding, run provenance (config, seeds, git commit, package versions),
  ruff + pytest + pre-commit, GitHub Actions CI (never downloads ABIDE), docs skeleton.

### Fixed
- `.gitignore` now anchors `/data/` (previously also ignored `src/shiftguard/data/` and
  `configs/data/`).
