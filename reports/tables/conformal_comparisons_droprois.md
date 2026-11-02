# Method minus split CP (LOSO), paired bootstrap over sites (2000 resamples), 95% percentile CI, BH q-values within model; run 2026-11-02_131805_conformal; baseline 2026-11-02_105846_loso_droprois. Mondrian arms are compared with split CP on the same subjects.

| model | method | metric | n_sites | difference | q |
|---|---|---|---|---|---|
| logreg | class_conditional | coverage | 19 | +0.002 [-0.005, +0.008] | 0.530 |
| logreg | class_conditional | shortfall | 19 | -0.001 [-0.005, +0.005] | 0.743 |
| logreg | class_conditional | abstention | 19 | +0.007 [-0.006, +0.019] | 0.299 |
| logreg | class_conditional_aps | coverage | 19 | +0.010 [-0.002, +0.022] | 0.128 |
| logreg | class_conditional_aps | shortfall | 19 | -0.012 [-0.022, -0.003] | 0.019 |
| logreg | class_conditional_aps | abstention | 19 | +0.114 [+0.085, +0.143] | 0.000 |
| logreg | label_shift_em | coverage | 19 | -0.199 [-0.265, -0.136] | 0.000 |
| logreg | label_shift_em | shortfall | 19 | +0.189 [+0.127, +0.257] | 0.000 |
| logreg | label_shift_em | abstention | 19 | -0.257 [-0.337, -0.183] | 0.000 |
| logreg | label_shift_oracle | coverage | 19 | +0.002 [-0.011, +0.013] | 0.817 |
| logreg | label_shift_oracle | shortfall | 19 | +0.001 [-0.007, +0.011] | 0.920 |
| logreg | label_shift_oracle | abstention | 19 | -0.014 [-0.035, +0.006] | 0.215 |
| logreg | mondrian_k0 | coverage | 19 | +0.129 [+0.099, +0.161] | 0.000 |
| logreg | mondrian_k0 | shortfall | 19 | -0.043 [-0.068, -0.021] | 0.000 |
| logreg | mondrian_k0 | abstention | 19 | +0.495 [+0.440, +0.550] | 0.000 |
| logreg | mondrian_k10 | coverage | 19 | +0.024 [-0.002, +0.057] | 0.104 |
| logreg | mondrian_k10 | shortfall | 19 | -0.017 [-0.043, +0.004] | 0.157 |
| logreg | mondrian_k10 | abstention | 19 | +0.113 [+0.053, +0.190] | 0.000 |
| logreg | mondrian_k20 | coverage | 19 | +0.030 [+0.005, +0.057] | 0.022 |
| logreg | mondrian_k20 | shortfall | 19 | -0.029 [-0.054, -0.007] | 0.006 |
| logreg | mondrian_k20 | abstention | 19 | +0.089 [+0.026, +0.168] | 0.005 |
| logreg | mondrian_k5 | coverage | 19 | +0.129 [+0.099, +0.161] | 0.000 |
| logreg | mondrian_k5 | shortfall | 19 | -0.043 [-0.068, -0.021] | 0.000 |
| logreg | mondrian_k5 | abstention | 19 | +0.495 [+0.440, +0.550] | 0.000 |
| logreg | split_aps | coverage | 19 | +0.011 [+0.001, +0.022] | 0.058 |
| logreg | split_aps | shortfall | 19 | -0.013 [-0.022, -0.006] | 0.000 |
| logreg | split_aps | abstention | 19 | +0.110 [+0.080, +0.139] | 0.000 |
| logreg | split_aps_det | coverage | 19 | +0.121 [+0.099, +0.148] | 0.000 |
| logreg | split_aps_det | shortfall | 19 | -0.032 [-0.055, -0.016] | 0.000 |
| logreg | split_aps_det | abstention | 19 | +0.475 [+0.427, +0.526] | 0.000 |
| logreg | weighted | coverage | 19 | +0.051 [+0.021, +0.082] | 0.002 |
| logreg | weighted | shortfall | 19 | -0.019 [-0.042, +0.000] | 0.071 |
| logreg | weighted | abstention | 19 | +0.272 [+0.202, +0.346] | 0.000 |
| logreg | weighted_clip | coverage | 19 | +0.069 [+0.047, +0.094] | 0.000 |
| logreg | weighted_clip | shortfall | 19 | -0.027 [-0.049, -0.012] | 0.000 |
| logreg | weighted_clip | abstention | 19 | +0.278 [+0.211, +0.344] | 0.000 |
