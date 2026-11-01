# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
