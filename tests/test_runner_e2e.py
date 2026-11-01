"""Synthetic end-to-end test of the baseline pipeline (no download; runs in CI)."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from shiftguard.cli import app
from shiftguard.config import load_config
from shiftguard.data.synthetic import make_synthetic_dataset
from shiftguard.evaluation.runner import fold_metrics, run_experiment


@pytest.fixture(scope="module")
def run_output():
    cfg = load_config("configs/experiment/synthetic.yaml")
    cfg["experiment"]["n_seeds"] = 1
    ds = make_synthetic_dataset(cfg["data"], seed=0)
    preds, info = run_experiment(ds, cfg, n_jobs=1)
    return ds, preds, info


def test_predictions_have_no_leakage(run_output):
    ds, preds, info = run_output
    n_sites = len(set(ds.sites))
    assert len(info) == n_sites + 1  # LOSO folds + one within-site split (1 seed, 1 model)
    for (protocol, site), g in preds.groupby(["protocol", "test_site"]):
        cal, test = g[g.part == "cal"], g[g.part == "test"]
        assert not set(cal.SUB_ID) & set(test.SUB_ID)
        if protocol == "loso":
            assert set(test.SITE_ID) == {site}
            assert site not in set(cal.SITE_ID)
    loso_test = preds[(preds.protocol == "loso") & (preds.part == "test")]
    assert sorted(loso_test.SUB_ID) == sorted(ds.phenotypes.SUB_ID)  # every subject tested once


def test_probabilities_and_metrics(run_output):
    _, preds, _ = run_output
    assert preds[["p_raw", "p_cal"]].apply(lambda c: c.between(0, 1).all()).all()
    folds = fold_metrics(preds)
    assert {"auc_cal", "ece_raw", "ece_cal"} <= set(folds.columns)
    # temperature scaling never changes the ranking, so AUC is identical
    pd.testing.assert_series_equal(folds["auc_raw"], folds["auc_cal"], check_names=False)


def test_cli_run_writes_outputs(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "run",
            "-c",
            "configs/experiment/synthetic.yaml",
            "-s",
            f"paths.reports_dir={tmp_path.as_posix()}",
            "-s",
            "experiment.n_seeds=1",
        ],
    )
    assert result.exit_code == 0, result.output
    (run_dir,) = (tmp_path / "runs").iterdir()
    for name in ("predictions.csv.gz", "fold_metrics.csv", "baselines.md", "metadata.json"):
        assert (run_dir / name).is_file(), name
    assert (tmp_path / "tables" / "baselines.md").is_file()
    assert (tmp_path / "tables" / "loso_sites.md").is_file()
    assert (tmp_path / "figures" / "fig6_reliability.png").stat().st_size > 0
    table = pd.read_csv(tmp_path / "tables" / "baselines.csv")
    assert {"ece_null", "cal_in_large"} <= set(table.columns)

    # `summarize` rebuilds the same tables from the saved predictions, without refitting
    preds_before = (run_dir / "predictions.csv.gz").stat().st_mtime_ns
    (tmp_path / "tables" / "baselines.csv").unlink()
    result = CliRunner().invoke(app, ["summarize", str(run_dir)])
    assert result.exit_code == 0, result.output
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "tables" / "baselines.csv"), table)
    assert (run_dir / "predictions.csv.gz").stat().st_mtime_ns == preds_before
    assert CliRunner().invoke(app, ["summarize", str(tmp_path)]).exit_code == 3
