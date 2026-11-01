# Baselines: mean ± SD over seeds (run 2026-11-01_223647_loso). AUC, balanced accuracy and Brier use temperature-scaled probabilities. ece_null = expected ECE of a perfectly calibrated model with the same probabilities and test size; cal_in_large = mean P(ASD) minus observed ASD fraction.

| protocol | model | n_seeds | auc | auc_site_mean | auc_site_worst | balanced_accuracy | ece_raw | ece_cal | ece_null | cal_in_large | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loso | logreg | 10 | 0.731 ± 0.008 | 0.745 ± 0.008 | 0.528 ± 0.043 | 0.672 ± 0.010 | 0.039 ± 0.009 | 0.032 ± 0.006 | 0.034 ± 0.000 | -0.003 ± 0.008 | 0.210 ± 0.003 |
| loso | mlp | 10 | 0.690 ± 0.014 | 0.700 ± 0.014 | 0.468 ± 0.061 | 0.628 ± 0.018 | 0.059 ± 0.011 | 0.029 ± 0.008 | 0.033 ± 0.001 | 0.009 ± 0.007 | 0.221 ± 0.004 |
| within_site | logreg | 10 | 0.732 ± 0.030 | - | - | 0.668 ± 0.022 | 0.081 ± 0.017 | 0.080 ± 0.017 | 0.074 ± 0.002 | 0.002 ± 0.013 | 0.211 ± 0.010 |
| within_site | mlp | 10 | 0.692 ± 0.030 | - | - | 0.620 ± 0.031 | 0.099 ± 0.024 | 0.078 ± 0.021 | 0.071 ± 0.003 | 0.017 ± 0.032 | 0.222 ± 0.009 |
