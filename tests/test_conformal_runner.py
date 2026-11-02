"""Phase 5 evaluator: calibration halves, label-free methods, Mondrian arms, CLI."""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from shiftguard.cli import app
from shiftguard.config import load_config
from shiftguard.data.synthetic import make_synthetic_dataset
from shiftguard.evaluation import conformal_runner as cr
from shiftguard.evaluation.runner import run_experiment

LABEL_FREE = [
    "split",
    "class_conditional",
    "split_aps",
    "class_conditional_aps",
    "split_aps_det",
    "label_shift_em",
    "weighted",
    "weighted_clip",
]

LABEL_FREE_SIGNALS = [
    "mean_p_cal",
    "mean_p_test",
    "mean_maxprob_cal",
    "mean_maxprob_test",
    "ks_prob",
    "ks_maxprob",
    "abstain_split_cal",
    "abstain_split_test",
]


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("configs/experiment/synthetic.yaml")
    cfg["experiment"]["n_seeds"] = 1
    ds = make_synthetic_dataset(cfg["data"], seed=0)
    preds, _ = run_experiment(ds, cfg, n_jobs=1)
    train_rows = cr.train_rows_from_splits(cfg, ds.sites, ds.labels)
    w, info = cr.compute_domain_weights(ds.features, preds, train_rows, cfg)
    return cfg, ds, preds, w, info


def test_calibration_halves_are_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    sites = np.array(["a"] * 41 + ["b"] * 20, dtype=object)
    labels = rng.integers(0, 2, size=61)
    a, b = cr.calibration_halves(sites, labels, seed=3)
    assert not set(a) & set(b) and len(a) + len(b) == 61
    for s in ("a", "b"):
        for y in (0, 1):
            n = np.sum((sites == s) & (labels == y))
            assert np.sum((sites[a] == s) & (labels[a] == y)) == n // 2
    a2, _ = cr.calibration_halves(sites, labels, seed=3)
    np.testing.assert_array_equal(a, a2)


def _fold(preds, site, protocol="loso"):
    f = preds[(preds.protocol == protocol) & (preds.test_site == site) & (preds.seed == 0)]
    return f.reset_index(drop=True)


def test_label_free_methods_never_read_deployment_labels(setup):
    cfg, ds, preds, w, _ = setup
    site = sorted(set(ds.sites))[0]
    fold = _fold(preds, site)
    weights = w[w.test_site == site].set_index("row")["weight"]
    out, info = cr.conformal_fold(fold, cfg, weights)
    shuffled = fold.copy()
    t = shuffled["part"] == "test"
    shuffled.loc[t, "label"] = np.random.default_rng(1).permutation(shuffled.loc[t, "label"])
    assert (shuffled["label"] != fold["label"]).any()
    out2, info2 = cr.conformal_fold(shuffled, cfg, weights)
    for m in LABEL_FREE:
        a = out[out.method == m].sort_values("SUB_ID")[["in0", "in1"]].to_numpy()
        b = out2[out2.method == m].sort_values("SUB_ID")[["in0", "in1"]].to_numpy()
        np.testing.assert_array_equal(a, b, err_msg=m)
    # phase 6 signals (and the EM prior) are label-free too
    for key in [*LABEL_FREE_SIGNALS, "pi_target_em", "pi_source"]:
        assert info[key] == info2[key], key


def test_mondrian_arms_share_evaluation_subjects_and_abstain_below_nine_labels(setup):
    cfg, ds, preds, _w, _ = setup
    site = sorted(set(ds.sites))[1]
    out, info = cr.conformal_fold(_fold(preds, site), cfg, None)
    ids = {k: set(out.loc[out.method == f"mondrian_k{k}", "SUB_ID"]) for k in (0, 5, 10, 20)}
    assert ids[0] == ids[5] == ids[10] == ids[20]
    assert len(ids[0]) == info["n_mondrian_eval"] == np.sum(ds.sites == site) - 20
    for k in (0, 5):  # fewer than ceil(0.9 / 0.1) = 9 labels: the full set for everyone
        m = out[out.method == f"mondrian_k{k}"]
        assert (m.in0 == 1).all() and (m.in1 == 1).all()
    assert "weighted" not in set(out.method)  # no weights given


def test_within_site_folds_get_only_exchangeable_methods(setup):
    cfg, _, preds, _, _ = setup
    out, info = cr.conformal_fold(_fold(preds, "pooled", "within_site"), cfg, None)
    assert set(out.method) == {
        "split",
        "class_conditional",
        "split_aps",
        "class_conditional_aps",
        "split_aps_det",
    }
    assert info["n_cal_A"] + info["n_cal_B"] == np.sum(
        (preds.protocol == "within_site") & (preds.part == "cal") & (preds.seed == 0)
    )


def test_domain_weights_cover_every_loso_cal_and_test_row(setup):
    _, ds, preds, w, info = setup
    assert len(info) == len(set(ds.sites))
    loso = preds[(preds.protocol == "loso") & (preds.model == "logreg")]
    for (seed, site), g in loso.groupby(["seed", "test_site"]):
        rows = set(w[(w.seed == seed) & (w.test_site == site)].row)
        assert rows == set(g.row)
    assert (w.weight > 0).all() and info.domain_auc.between(0, 1).all()


def test_cli_conformal_end_to_end(tmp_path):
    runner = CliRunner()
    common = ["-s", f"paths.reports_dir={tmp_path.as_posix()}", "-s", "experiment.n_seeds=1"]
    assert (
        runner.invoke(app, ["run", "-c", "configs/experiment/synthetic.yaml", *common]).exit_code
        == 0
    )
    (base,) = (tmp_path / "runs").iterdir()
    res = runner.invoke(app, ["conformal", str(base), "--combat", str(base)])
    assert res.exit_code == 0, res.output
    tables = tmp_path / "tables"
    for name in (
        "conformal_headline",
        "conformal_sites",
        "conformal_comparisons",
        "prevalence_shift",
    ):
        assert (tables / f"{name}.md").is_file(), name
    for fig in (
        "fig3_coverage_per_site",
        "fig3b_class_coverage",
        "fig4_target_labels",
        "fig5_abstention_per_site",
        "figS1_prevalence_shift",
    ):
        assert (tmp_path / "figures" / f"{fig}.png").stat().st_size > 0, fig
    head = pd.read_csv(tables / "conformal_headline.csv")
    assert {"combat+split", "label_shift_em", "mondrian_k20"} <= set(head.method)
    assert list((tmp_path / "cache").glob("domain_weights_*.csv.gz"))
    # second call reuses the cached domain weights
    assert runner.invoke(app, ["conformal", str(base)]).exit_code == 0
    assert runner.invoke(app, ["conformal", str(tmp_path)]).exit_code == 3
    # sensitivity run at another alpha: tagged outputs, main tables untouched
    before = (tables / "conformal_headline.csv").read_bytes()
    assert runner.invoke(app, ["conformal", str(base), "--alpha", "0.2"]).exit_code == 0
    assert (tables / "conformal_headline.csv").read_bytes() == before
    h2 = pd.read_csv(tables / "conformal_headline_alpha0p2.csv")
    assert len(h2) > 0
    assert (tmp_path / "figures" / "fig3_coverage_per_site_alpha0p2.png").is_file()
    assert runner.invoke(app, ["conformal", str(base), "--alpha", "1.5"]).exit_code == 2


def test_combat_arm_never_reads_deployment_labels():
    """Permuting the deployment site's labels leaves the whole ComBat chain unchanged.

    The chain is: ComBat on all subjects (age and sex only) -> LOSO refit -> temperature
    -> ``combat+split`` / ``combat+class_conditional`` sets on that site.
    """
    pytest.importorskip("neuroHarmonize")
    from dataclasses import replace

    from shiftguard.features.harmonize import harmonize_dataset

    cfg = load_config("configs/experiment/synthetic.yaml")
    cfg["experiment"] |= {"n_seeds": 1, "protocols": ["loso"]}
    cfg["features"]["harmonization"] = "combat"
    ds = make_synthetic_dataset(cfg["data"], seed=0)
    site = sorted(set(ds.sites))[2]
    t = ds.sites == site
    ph = ds.phenotypes.copy()
    ph.loc[t, "label"] = np.random.default_rng(5).permutation(ph.loc[t, "label"].to_numpy())
    ph["DX_GROUP"] = np.where(ph["label"] == 1, 1, 2)
    assert (ph["label"].to_numpy() != ds.labels).any()
    shuffled = replace(ds, phenotypes=ph)

    sets = []
    for d in (ds, shuffled):
        h = harmonize_dataset(d, cfg)
        preds, _ = run_experiment(h, cfg, n_jobs=1)
        fold = _fold(preds, site)
        out, _ = cr.conformal_fold(fold, cfg, prefix="combat+", methods="basic")
        sets.append((h.features, fold, out))
    np.testing.assert_array_equal(sets[0][0], sets[1][0])  # ComBat features
    a, b = (s[1].sort_values("SUB_ID") for s in sets)
    np.testing.assert_array_equal(a["logit"].to_numpy(), b["logit"].to_numpy())
    assert {"combat+split", "combat+class_conditional"} <= set(sets[0][2].method)
    for m in ("combat+split", "combat+class_conditional"):
        x, y = (s[2][s[2].method == m].sort_values("SUB_ID")[["in0", "in1"]] for s in sets)
        np.testing.assert_array_equal(x.to_numpy(), y.to_numpy(), err_msg=m)
