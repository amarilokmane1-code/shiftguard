# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
