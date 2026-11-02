"""Phase 6: label-free signals, rank correlations with bootstrap CIs, BH."""

import numpy as np
import pandas as pd
import pytest

from shiftguard.evaluation import detection as det


def _diag(n_sites=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        for sd in range(3):
            rows.append(
                {
                    "protocol": "loso",
                    "model": "logreg",
                    "variant": "none",
                    "seed": sd,
                    "test_site": f"S{s}",
                    "mean_p_cal": 0.5,
                    "mean_p_test": 0.5 + 0.02 * s,
                    "mean_maxprob_cal": 0.7,
                    "mean_maxprob_test": 0.7 - 0.01 * s,
                    "ks_prob": 0.05 * s + 0.01 * rng.random(),
                    "ks_maxprob": 0.1,
                    "abstain_split_cal": 0.5,
                    "abstain_split_test": 0.5 + 0.01 * s,
                    "pi_target_em": 0.5 - 0.03 * s,
                    "pi_source": 0.5,
                    "domain_auc": 0.8,
                }
            )
    return pd.DataFrame(rows)


def _sites(n_sites=8):
    return pd.DataFrame(
        {
            "protocol": "loso",
            "model": "logreg",
            "method": "split",
            "test_site": [f"S{s}" for s in range(n_sites)],
            "n": 40,
            "coverage": [0.95 - 0.02 * s for s in range(n_sites)],
            "coverage_asd": [0.9 + 0.01 * s for s in range(n_sites)],
            "coverage_tdc": [0.9 - 0.03 * s for s in range(n_sites)],
        }
    )


def test_fold_signals_derivations():
    d = det.fold_signals(_diag())
    r = d[(d.test_site == "S3")].iloc[0]
    assert r["mean_p_shift"] == pytest.approx(0.06)
    assert r["abs_em_shift"] == pytest.approx(0.09)
    assert r["em_shift"] == pytest.approx(-0.09)
    assert r["abs_confidence_shift"] == pytest.approx(0.03)
    assert r["abs_abstention_shift"] == pytest.approx(0.03)
    assert r["abstention"] == pytest.approx(0.53)


def test_site_signals_and_correlations():
    sig = det.site_signals(_diag(), _sites(), alpha=0.1, model="logreg", variant="none")
    assert len(sig) == 8
    assert sig.set_index("test_site").loc["S0", "coverage_drop"] == pytest.approx(-0.05)
    assert sig.set_index("test_site").loc["S7", "worst_class_drop"] == pytest.approx(0.21)
    corr = det.correlation_table(sig, n_boot=200, n_perm=2000)
    c = corr.set_index(["signal", "target"])
    # ks_prob increases with s, as does the coverage drop: rho = 1
    assert c.loc[("ks_prob", "coverage_drop"), "rho"] == pytest.approx(1.0)
    assert c.loc[("ks_prob", "coverage_drop"), "p_value"] < 0.01
    # constant signals give no correlation (NaN), not an error
    assert np.isnan(c.loc[("ks_maxprob", "coverage_drop"), "rho"])
    # signed EM shift decreases while ASD - TDC gap increases: rho = -1
    assert c.loc[("em_shift", "class_gap"), "rho"] == pytest.approx(-1.0)
    assert corr["q_value"].dropna().between(0, 1).all()


def test_spearman_with_ci_is_calibrated_under_independence():
    rng = np.random.default_rng(3)
    ps = [
        det.spearman_with_ci(
            rng.normal(size=18), rng.normal(size=18), n_boot=50, n_perm=500, seed=i
        )["p_value"]
        for i in range(200)
    ]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10
    res = det.spearman_with_ci(np.arange(18.0), np.arange(18.0) ** 2, n_boot=200, n_perm=1000)
    assert res["rho"] == pytest.approx(1.0) and res["ci_low"] > 0.9
    assert res["rho_loo_min"] == pytest.approx(1.0) == res["rho_loo_max"]


def test_within_site_reference_uses_only_exchangeable_folds():
    d = _diag()
    w = d.iloc[:2].assign(protocol="within_site", ks_prob=0.04)
    ref = det.within_site_reference(pd.concat([d, w]), "logreg")
    assert ref["ks_prob"] == pytest.approx(0.04)


def test_partial_correlation_removes_a_pure_size_effect():
    rng = np.random.default_rng(0)
    n = rng.integers(25, 175, size=18).astype(float)
    x = 1.0 / np.sqrt(n) + 0.002 * rng.normal(size=18)  # both driven by size only
    y = 1.0 / np.sqrt(n) + 0.002 * rng.normal(size=18)
    res = det.spearman_with_ci(x, y, n_boot=100, n_perm=2000, covariate=n)
    assert res["rho"] > 0.6
    assert abs(res["rho_partial"]) < 0.5 and res["p_partial"] > 0.05
