# ShiftGuard

**Does conformal abstention survive site shift? A multi-site autism neuroimaging benchmark.**

[![CI](https://github.com/amarilokmane1-code/shiftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/amarilokmane1-code/shiftguard/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status: work in progress (phase 4 of 9: baselines).** The baseline classifiers have
> been run on ABIDE. The conformal results are still `TBD`. Every number is taken from
> saved run outputs.

<!-- Main figure (coverage per deployment site) goes here once phase 5 has run:
![Coverage per deployment site](reports/figures/fig3_coverage_per_site.png) -->

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
| Conformal under shift + statistics | TBD (phase 5) |
| Label-free shift detection | TBD (phase 6) |
| Report and documentation | TBD (phase 7) |
| Kaggle notebook | TBD (phase 8) |

## Key results

*Filled automatically from real runs. Nothing here is estimated by hand.*

| Setting | Method | AUC | Balanced acc. | Marginal coverage | Worst-site coverage | Coverage gap | Abstention |
|---|---|---|---|---|---|---|---|
| Within-site | Split CP | TBD | TBD | TBD | TBD | TBD | TBD |
| LOSO | Split CP | TBD | TBD | TBD | TBD | TBD | TBD |
| LOSO | ComBat + split CP | TBD | TBD | TBD | TBD | TBD | TBD |
| LOSO | Weighted CP | TBD | TBD | TBD | TBD | TBD | TBD |
| LOSO | Mondrian (20 target labels) | TBD | TBD | TBD | TBD | TBD | TBD |

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

These offsets are what phase 5 has to survive (see
[LIMITATIONS §7](docs/LIMITATIONS.md)). The MLP has lower AUC and balanced accuracy and a
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
