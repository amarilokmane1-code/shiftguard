# Synthetic conformal validity checks (4000 trials, seed 0); run 2026-11-01_175354_simulate

| experiment | method | score | n_classes | n_calibration | target | coverage_95ci | p_below | mean_set_size | abstention_rate | mean_ess | theory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_exchangeable | split | lac | 2 | 200 | 0.900 | 0.901 [0.892, 0.910] | 0.61 | 1.702 | 0.702 | - | exact 181/201 = 0.9005 |
| A_exchangeable | split | aps | 2 | 200 | 0.900 | 1.000 [0.999, 1.000] | 1 | 2.000 | 1.000 | - | >= 0.90 (ties: upper bound does not apply) |
| A_exchangeable | split (randomised) | aps | 2 | 200 | 0.900 | 0.906 [0.897, 0.915] | 0.92 | 1.726 | 0.726 | - | exact 181/201 = 0.9005 |
| A_exchangeable | split | lac | 3 | 200 | 0.900 | 0.899 [0.890, 0.909] | 0.47 | 1.679 | 0.583 | - | exact 181/201 = 0.9005 |
| A_exchangeable | split | aps | 3 | 200 | 0.900 | 0.902 [0.892, 0.911] | 0.67 | 1.889 | 0.899 | - | >= 0.90 (ties: upper bound does not apply) |
| A_exchangeable | split (randomised) | aps | 3 | 200 | 0.900 | 0.902 [0.892, 0.911] | 0.67 | 1.817 | 0.715 | - | exact 181/201 = 0.9005 |
| B_covariate_shift | split (source test) | lac | 2 | 1000 | 0.900 | 0.895 [0.885, 0.904] | 0.15 | 1.313 | 0.313 | - | exchangeable: >= target |
| B_covariate_shift | split | lac | 2 | 1000 | 0.900 | 0.828 [0.815, 0.839] | 7.9e-45 | 1.319 | 0.319 | - | no guarantee under shift |
| B_covariate_shift | weighted (oracle w) | lac | 2 | 1000 | 0.900 | 0.903 [0.893, 0.912] | 0.73 | 1.574 | 0.574 | 377.848 | >= target (exact weights) |
| B_covariate_shift | weighted (estimated w) | lac | 2 | 1000 | 0.900 | 0.904 [0.895, 0.913] | 0.82 | 1.581 | 0.581 | 359.920 | approximate (estimated weights) |
| B_covariate_shift | weighted (estimated w, clipped) | lac | 2 | 1000 | 0.900 | 0.901 [0.892, 0.910] | 0.61 | 1.571 | 0.571 | 435.777 | approximate (estimated weights) |
