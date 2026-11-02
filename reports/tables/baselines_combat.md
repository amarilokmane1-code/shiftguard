# Baselines: mean ± SD over seeds (run 2026-11-02_101130_loso_combat). AUC, balanced accuracy and Brier use temperature-scaled probabilities. ece_null = expected ECE of a perfectly calibrated model with the same probabilities and test size; cal_in_large = mean P(ASD) minus observed ASD fraction.

| protocol | model | n_seeds | auc | auc_site_mean | auc_site_worst | balanced_accuracy | ece_raw | ece_cal | ece_null | cal_in_large | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loso | logreg | 10 | 0.749 ± 0.006 | 0.745 ± 0.008 | 0.538 ± 0.029 | 0.684 ± 0.007 | 0.036 ± 0.009 | 0.029 ± 0.008 | 0.034 ± 0.001 | -0.001 ± 0.002 | 0.203 ± 0.002 |
