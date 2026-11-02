# Conformal prediction at alpha = 0.1 (target coverage 0.90); run 2026-11-02_131805_conformal; baseline 2026-11-02_105846_loso_droprois. Coverage pooled over sites, mean over seeds. Site metrics use seed-averaged site coverage; coverage_gap = mean over sites of max(0, target - coverage); n_*_under = BH q < 0.05 for coverage < target; abstention = set is not a single class; selective_accuracy = accuracy on the non-abstained cases; between_site_sd = empirical-Bayes SD of true site coverage.

| protocol | model | method | coverage | coverage_asd | coverage_tdc | mean_site_coverage | worst_site_coverage | worst_site | coverage_gap | worst_site_class_coverage | n_sites_under | n_site_classes_under | abstention | mean_set_size | selective_accuracy | between_site_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loso | logreg | class_conditional | 0.897 | 0.899 | 0.894 | 0.881 | 0.732 | LEUVEN_1 | 0.032 | 0.500 | 0 | 1 | 0.564 | 1.564 | 0.763 | 0.034 |
| loso | logreg | class_conditional_aps | 0.896 | 0.896 | 0.896 | 0.889 | 0.775 | LEUVEN_1 | 0.021 | 0.629 | 0 | 0 | 0.653 | 1.650 | 0.705 | 0.001 |
| loso | logreg | label_shift_em | 0.725 | 0.728 | 0.722 | 0.680 | 0.417 | KKI | 0.222 | 0.000 | 13 | 14 | 0.327 | 1.288 | 0.621 | 0.143 |
| loso | logreg | label_shift_oracle | 0.897 | 0.877 | 0.917 | 0.880 | 0.732 | LEUVEN_1 | 0.033 | 0.479 | 0 | 1 | 0.541 | 1.541 | 0.777 | 0.030 |
| loso | logreg | mondrian_k0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | CMU | 0.000 | 1.000 | 0 | 0 | 1.000 | 2.000 | - | 0.000 |
| loso | logreg | mondrian_k10 | 0.918 | 0.912 | 0.924 | 0.895 | 0.760 | OHSU | 0.026 | 0.665 | 0 | 0 | 0.662 | 1.658 | 0.765 | 0.014 |
| loso | logreg | mondrian_k20 | 0.913 | 0.906 | 0.919 | 0.901 | 0.817 | UCLA_2 | 0.014 | 0.638 | 0 | 0 | 0.609 | 1.609 | 0.779 | 0.000 |
| loso | logreg | mondrian_k5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | CMU | 0.000 | 1.000 | 0 | 0 | 1.000 | 2.000 | - | 0.000 |
| loso | logreg | split | 0.895 | 0.876 | 0.913 | 0.879 | 0.721 | LEUVEN_1 | 0.032 | 0.450 | 0 | 1 | 0.558 | 1.558 | 0.762 | 0.033 |
| loso | logreg | split_aps | 0.897 | 0.883 | 0.910 | 0.890 | 0.771 | LEUVEN_1 | 0.019 | 0.571 | 0 | 0 | 0.648 | 1.646 | 0.710 | 0.002 |
| loso | logreg | split_aps_det | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | CMU | 0.000 | 1.000 | 0 | 0 | 1.000 | 2.000 | - | 0.000 |
| loso | logreg | weighted | 0.934 | 0.930 | 0.938 | 0.930 | 0.795 | YALE | 0.014 | 0.768 | 0 | 0 | 0.806 | 1.775 | 0.739 | 0.049 |
| loso | logreg | weighted_clip | 0.953 | 0.950 | 0.957 | 0.948 | 0.832 | OHSU | 0.006 | 0.808 | 0 | 0 | 0.825 | 1.821 | 0.746 | 0.035 |
