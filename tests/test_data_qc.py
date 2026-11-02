import numpy as np
import pandas as pd
import pytest

from abide_fixtures import make_phenotypes
from shiftguard.data.phenotypes import cohort_summary, counts_by_site, load_phenotypes
from shiftguard.data.qc import REASONS, QCConfig, apply_qc, rater_pass


def _info(pheno, n_timepoints=100, n_rois=8):
    """Time-series info saying every subject with a file is readable and clean."""
    ids = [f for f in pheno["FILE_ID"] if f != "no_filename"]
    return pd.DataFrame(
        {
            "FILE_ID": ids,
            "readable": True,
            "n_timepoints": n_timepoints,
            "n_rois": n_rois,
            "n_zero_variance_rois": 0,
            "has_nonfinite": False,
        }
    )


def _load(tmp_path, pheno):
    path = tmp_path / "pheno.csv"
    pheno.to_csv(path, index=False)
    return load_phenotypes(path)


def test_load_phenotypes_cleans_and_encodes(tmp_path):
    pheno = _load(tmp_path, make_phenotypes({"A": (3, 2)}))
    assert pheno["label"].tolist() == [1, 1, 1, 0, 0]
    assert pheno["diagnosis"].tolist() == ["ASD"] * 3 + ["TDC"] * 2
    assert pheno["FIQ"].isna().sum() == 1  # the -9999 sentinel became NaN
    assert counts_by_site(pheno).loc["A"].tolist() == [5, 3, 2]


def test_cohort_summary(tmp_path):
    pheno = _load(tmp_path, make_phenotypes({"A": (3, 1), "B": (1, 1)}))
    s = cohort_summary(pheno).set_index("SITE_ID")
    assert s.loc["A", "n"] == 4 and s.loc["ALL", "n"] == 6
    assert s.loc["A", "pct_ASD"] == pytest.approx(75.0)
    assert s.loc["ALL", "age_min"] == pheno["AGE_AT_SCAN"].min()
    assert s.loc["ALL", "FIQ_missing"] == pheno["FIQ"].isna().sum()


def test_load_phenotypes_rejects_bad_files(tmp_path):
    bad = make_phenotypes({"A": (1, 1)})
    bad.loc[0, "DX_GROUP"] = 3
    with pytest.raises(ValueError, match="DX_GROUP"):
        _load(tmp_path, bad)
    with pytest.raises(ValueError, match="required"):
        _load(tmp_path, bad.drop(columns=["SITE_ID"]))
    dup = make_phenotypes({"A": (1, 1)})
    dup.loc[1, "SUB_ID"] = dup.loc[0, "SUB_ID"]
    with pytest.raises(ValueError, match="duplicate"):
        _load(tmp_path, dup)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("none", [True, True, True, True, True]),
        ("rater_1_pass", [True, False, True, True, True]),  # only rater-1 "fail" excluded
        ("all_raters_pass", [True, False, False, True, True]),
        ("nilearn_quality_checked", [True, False, False, False, True]),  # "maybe"/NaN fail
    ],
)
def test_rater_rules(tmp_path, rule, expected):
    pheno = _load(tmp_path, make_phenotypes({"A": (3, 2)}))
    pheno.loc[1, "qc_rater_1"] = "fail"
    pheno.loc[2, "qc_func_rater_3"] = "fail"
    pheno.loc[3, "qc_rater_1"] = "maybe"
    pheno.loc[4, "qc_anat_rater_2"] = "maybe"  # allowed for rater 2 by nilearn
    assert rater_pass(pheno, rule).tolist() == expected


def test_apply_qc_attributes_first_reason_and_counts_add_up(tmp_path):
    raw = make_phenotypes({"BIG": (20, 12), "SMALL": (12, 4)})  # BIG keeps 13 ASD after QC
    raw.loc[0, "FILE_ID"] = "no_filename"
    raw.loc[1, "qc_rater_1"] = "fail"
    raw.loc[2, "AGE_AT_SCAN"] = np.nan
    raw.loc[2, "qc_rater_1"] = "fail"  # also fails QC: QC wins (it comes first)
    raw.loc[3, "AGE_AT_SCAN"] = np.nan
    pheno = _load(tmp_path, raw)
    info = _info(pheno)
    missing_id, short_id, bad_id = pheno["FILE_ID"].iloc[[4, 5, 6]]
    info = info[info["FILE_ID"] != missing_id]
    info.loc[info["FILE_ID"] == short_id, "n_timepoints"] = 20
    info.loc[info["FILE_ID"] == bad_id, "n_zero_variance_rois"] = 2

    qc = apply_qc(pheno, info, QCConfig(min_subjects_per_class_per_site=10, min_timepoints=50))
    counts = qc.reason_counts
    assert list(counts.index) == list(REASONS)
    assert counts.to_dict() == {
        "no_preprocessed_data": 1,
        "rater_qc_fail": 2,
        "missing_covariates": 1,
        "missing_timeseries": 1,
        "too_few_timepoints": 1,
        "bad_rois": 1,
        "small_site": 16,  # SMALL has only 4 TDC -> whole site dropped
    }
    assert qc.small_sites == ["SMALL"]
    assert len(qc.included) + len(qc.exclusions) == len(pheno)
    assert set(qc.included["SITE_ID"]) == {"BIG"}
    assert not set(qc.included["SUB_ID"]) & set(qc.exclusions["SUB_ID"])

    summary = qc.summary.set_index("SITE_ID")
    total = summary.loc["TOTAL"]
    assert total["n_before"] == len(pheno)
    assert total["n_after"] == len(qc.included)
    excl_cols = [c for c in summary.columns if c.startswith("excl_")]
    sites = summary.drop(index="TOTAL")
    assert (sites["n_before"] - sites[excl_cols].sum(axis=1) == sites["n_after"]).all()
    assert (sites["n_after_ASD"] + sites["n_after_TDC"] == sites["n_after"]).all()


def test_bad_rois_threshold_is_configurable(tmp_path):
    pheno = _load(tmp_path, make_phenotypes({"A": (3, 3)}))
    info = _info(pheno)
    info.loc[0, "n_zero_variance_rois"] = 2
    info.loc[1, "has_nonfinite"] = True
    cfg = dict(min_subjects_per_class_per_site=1)
    assert apply_qc(pheno, info, QCConfig(**cfg)).reason_counts["bad_rois"] == 2
    lenient = QCConfig(max_zero_variance_rois=2, **cfg)
    assert apply_qc(pheno, info, lenient).reason_counts["bad_rois"] == 1  # non-finite still out


def test_qc_config_validation():
    with pytest.raises(ValueError):
        QCConfig(rater_rule="whatever")
    cfg = QCConfig.from_config({"rater_rule": "none", "min_timepoints": 10}, ["AGE_AT_SCAN"])
    assert cfg.rater_rule == "none" and cfg.min_timepoints == 10
    assert cfg.covariates == ("AGE_AT_SCAN",)


def test_drop_rois_policy_never_excludes_for_constant_rois(tmp_path):
    pheno = _load(tmp_path, make_phenotypes({"A": (3, 3)}))
    info = _info(pheno)
    info.loc[0, "n_zero_variance_rois"] = 40
    info.loc[1, "has_nonfinite"] = True
    cfg = QCConfig(min_subjects_per_class_per_site=1, zero_variance_policy="drop_rois")
    assert apply_qc(pheno, info, cfg).reason_counts["bad_rois"] == 1  # non-finite only
    with pytest.raises(ValueError, match="zero_variance_policy"):
        QCConfig(zero_variance_policy="zero_fill")
