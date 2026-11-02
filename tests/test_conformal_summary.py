"""Phase 5 summaries on hand-made outcomes with known coverage."""

import numpy as np
import pandas as pd
import pytest

from shiftguard.evaluation import conformal_summary as cs


def _outcomes(cov_by_site, n=40, seeds=(0, 1), method="split"):
    """Half ASD; ``cov_by_site`` = (ASD coverage, TDC coverage); covered sets are singletons."""
    rows = []
    for site, (c_asd, c_tdc) in cov_by_site.items():
        for seed in seeds:
            for i in range(n):
                y = int(i < n // 2)
                j = i if y else i - n // 2
                cov = j < round((c_asd if y else c_tdc) * n // 2)
                in1 = int(y == 1) if cov else int(y == 0)
                rows.append(
                    dict(
                        protocol="loso",
                        model="logreg",
                        method=method,
                        seed=seed,
                        test_site=site,
                        SUB_ID=f"{site}{i}",
                        SITE_ID=site,
                        label=y,
                        in0=1 - in1,
                        in1=in1,
                    )
                )
    return pd.DataFrame(rows)


def test_site_and_headline_tables():
    out = _outcomes({"A": (1.0, 1.0), "B": (0.5, 0.9)})
    ps = cs.per_seed_site(out)
    sites = cs.site_table(ps, alpha=0.1).set_index("test_site")
    assert sites.loc["A", "coverage"] == 1.0
    assert sites.loc["B", "coverage"] == pytest.approx(0.7)
    assert sites.loc["B", "coverage_asd"] == pytest.approx(0.5)
    assert sites.loc["B", "abstention"] == 0.0
    assert sites.loc["B", "ci_low"] < 0.7 < sites.loc["B", "ci_high"]
    assert sites.loc["B", "q_under"] < 0.05 and sites.loc["A", "q_under"] > 0.5
    head = cs.headline_table(ps, sites.reset_index(), alpha=0.1).iloc[0]
    assert head["coverage"] == pytest.approx(0.85)
    assert head["worst_site"] == "B" and head["worst_site_coverage"] == pytest.approx(0.7)
    assert head["coverage_gap"] == pytest.approx((0.0 + 0.2) / 2)
    assert head["worst_site_class_coverage"] == pytest.approx(0.5)
    assert head["n_sites_under"] == 1


def test_comparisons_and_offset_gap():
    base = _outcomes({s: (0.8, 0.8) for s in "ABCDEF"})
    better = _outcomes({s: (1.0, 1.0) for s in "ABCDEF"}, method="class_conditional")
    out = pd.concat([base, better])
    ps = cs.per_seed_site(out)
    comp = cs.comparisons_vs_split(ps, out, 0.1, n_boot=200).set_index("metric")
    assert comp.loc["coverage", "diff"] == pytest.approx(0.2)
    assert comp.loc["shortfall", "diff"] == pytest.approx(-0.1)
    # offsets that favour ASD where positive -> strong positive correlation
    gaps = {s: (0.9 + d, 0.9 - d) for s, d in zip("ABCDEF", np.linspace(-0.1, 0.1, 6), strict=True)}
    sites = cs.site_table(cs.per_seed_site(_outcomes(gaps)), 0.1)
    shift = np.linspace(-0.2, 0.2, 6)
    prev = pd.DataFrame({"test_site": list("ABCDEF"), "cil_test": shift, "em_shift": shift})
    corr = cs.offset_vs_class_gap(sites, prev)
    assert corr["offset_vs_class_gap_split"]["rho"] > 0.9
    assert corr["em_shift_vs_class_gap_split"]["rho"] > 0.9
    assert corr["abs_em_shift_vs_coverage_drop_split"]["n_sites"] == 6
