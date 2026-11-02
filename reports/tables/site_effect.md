# Site effect on the connectome (run 2026-11-02_155829_site_effect). Site balanced accuracy: 5-fold CV (scaler, PCA 50, multinomial L2 logistic regression), chance = 1 / sites. r2_site: R^2 of site dummies on the 10 leading principal components, weighted by explained variance; r2_site_partial: beyond age, sex and diagnosis. ComBat uses age and sex only.

| harmonization | n_subjects | n_sites | site_balanced_accuracy | chance | r2_site | r2_site_partial |
|---|---|---|---|---|---|---|
| none | 944 | 18 | 0.549 | 0.056 | 0.129 | 0.116 |
| combat | 944 | 18 | 0.051 | 0.056 | 0.005 | 0.002 |
