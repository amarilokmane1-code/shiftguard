"""Phase 7: run manifest, regeneration from saved runs, RESULTS.md and the README block."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from shiftguard.cli import app
from shiftguard.evaluation import report as rep


def test_manifest_round_trip_and_unknown_keys(tmp_path):
    path = tmp_path / "final_runs.yaml"
    rep.save_manifest(path, {"detection": "d", "baseline": "b"})
    text = path.read_text(encoding="utf-8")
    assert text.index("baseline:") < text.index("detection:")  # canonical order
    assert rep.load_manifest(path) == {"baseline": "b", "detection": "d"}
    path.write_text("baselin: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown manifest keys"):
        rep.load_manifest(path)


def test_readme_block_needs_markers(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(
        {
            "protocol": ["loso"],
            "model": ["logreg"],
            "method": ["split"],
            "coverage": [0.9],
            "abstention": [0.5],
        }
    ).to_csv(tables / "conformal_headline.csv", index=False)
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    assert not rep.update_readme(readme, tables)
    readme.write_text(f"a\n{rep.README_START}\nold\n{rep.README_END}\nb\n", encoding="utf-8")
    assert rep.update_readme(readme, tables)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "| LOSO | Split CP |" in text and text.endswith("b\n")


@pytest.mark.slow
def test_all_then_report_end_to_end(tmp_path):
    pytest.importorskip("neuroHarmonize")
    runner = CliRunner()
    common = [
        "-c",
        "configs/experiment/synthetic.yaml",
        "-s",
        f"paths.reports_dir={tmp_path.as_posix()}",
        "-s",
        "experiment.n_seeds=1",
    ]
    res = runner.invoke(app, ["all", *common, "--simulate-trials", "50", "--sensitivity"])
    assert res.exit_code == 0, res.output
    manifest = rep.load_manifest(tmp_path / rep.MANIFEST)
    expected = {"simulate", "baseline", "baseline_combat", "conformal", "conformal_alpha0p2"}
    assert expected | {"site_effect", "detection"} <= set(manifest)
    assert "baseline_droprois" not in manifest  # synthetic data: no ROI-level QC
    results = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    for section in ("## Provenance", "## Baselines", "## Conformal prediction", "## Label-free"):
        assert section in results, section
    # `report` rebuilds every table from the run folders without refitting
    head_before = (tmp_path / "tables" / "conformal_headline.csv").read_bytes()
    (tmp_path / "tables" / "conformal_headline.csv").unlink()
    readme = tmp_path / "README.md"
    readme.write_text(f"{rep.README_START}\n{rep.README_END}\n", encoding="utf-8")
    out = tmp_path / "docs" / "RESULTS.md"
    res = runner.invoke(app, ["report", *common, "--results", str(out), "--readme", str(readme)])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "tables" / "conformal_headline.csv").read_bytes() == head_before
    assert out.read_text(encoding="utf-8") == results
    assert "| LOSO | Split CP |" in readme.read_text(encoding="utf-8")
    # a missing manifest is a clear error
    empty = ["-s", f"paths.reports_dir={(tmp_path / 'none').as_posix()}"]
    assert runner.invoke(app, ["report", *empty, "--readme", str(readme)]).exit_code == 3


def test_results_md_cohort_ignores_the_total_row(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(
        {
            "SITE_ID": ["A", "B", "ALL"],
            "n": [10, 30, 40],
            "pct_ASD": [50.0, 40.0, 42.5],
            "pct_male": [80.0, 100.0, 95.0],
        }
    ).to_csv(tables / "cohort_summary.csv", index=False)
    text = rep.build_results_md(tmp_path, {})
    assert "**40 subjects** from **2 sites** (42.5% ASD, 95.0% male; 10\N{EN DASH}30" in text
