# Model card

*Skeleton. Completed in phase 7 with measured performance.*

## Intended use

Research only: studying whether conformal coverage guarantees survive acquisition-site
shift in multi-site neuroimaging. **Not for clinical diagnosis, screening or any decision
about an individual.**

## Models

- L2 / elastic-net logistic regression on Fisher-z functional connectivity.
- Small multilayer perceptron (PyTorch), dropout, early stopping.
- Optional graph neural network (behind a flag).

All are wrapped in probability calibration (fitted on a held-out calibration split) and
conformal prediction with abstention.

## Performance

Discrimination and calibration (phase 4): [baselines](../reports/tables/baselines.md)
and [per-site LOSO results](../reports/tables/loso_sites.md), mean ± SD over 10 seeds.
Coverage and abstention (phase 5): [headline](../reports/tables/conformal_headline.md)
and [per deployment site and class](../reports/tables/conformal_sites.md). At α = 0.1,
LOSO split CP (logistic regression) covers 0.898 of subjects pooled. Per site, coverage
is 0.75–0.95, and per class at a site it falls to 0.55. More than half of subjects get
both labels (abstention 0.554). Coverage at a new site is **not** guaranteed;
see [LIMITATIONS](LIMITATIONS.md) §7–12.

## Subgroups and known biases

See [DATA_CARD.md](DATA_CARD.md). Training data are mostly male, older children/adults,
and cognitively able. Performance and coverage on under-represented groups are not
established.

## Ethical considerations

Autism is a heterogeneous, lifelong condition. A binary ASD/control label from a brain
scan does not capture it, and such a model must not be used to label individuals.
