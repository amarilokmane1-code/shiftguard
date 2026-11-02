# Baselines: mean ± SD over seeds (run 2026-11-02_105846_loso_droprois). AUC, balanced accuracy and Brier use temperature-scaled probabilities. ece_null = expected ECE of a perfectly calibrated model with the same probabilities and test size; cal_in_large = mean P(ASD) minus observed ASD fraction.

| protocol | model | n_seeds | auc | auc_site_mean | auc_site_worst | balanced_accuracy | ece_raw | ece_cal | ece_null | cal_in_large | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loso | logreg | 10 | 0.725 ± 0.003 | 0.739 ± 0.005 | 0.545 ± 0.058 | 0.669 ± 0.008 | 0.047 ± 0.009 | 0.034 ± 0.009 | 0.033 ± 0.001 | -0.002 ± 0.007 | 0.212 ± 0.001 |
