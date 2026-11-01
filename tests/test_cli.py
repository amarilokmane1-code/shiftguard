import pytest
from typer.testing import CliRunner

from shiftguard import __version__
from shiftguard.cli import app

runner = CliRunner()


def test_help_lists_every_stage():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("download", "features", "run", "report", "all", "info"):
        assert command in result.output


def test_info_reports_version():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_simulate_writes_tables_and_provenance(tmp_path):
    result = runner.invoke(
        app, ["simulate", "--n-trials", "40", "--seed", "3", "--reports-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tables" / "conformal_synthetic.csv").is_file()
    md = (tmp_path / "tables" / "conformal_synthetic.md").read_text(encoding="utf-8")
    assert "weighted (oracle w)" in md
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert {"config.yaml", "metadata.json", "conformal_synthetic.csv"} <= {
        p.name for p in run_dir.iterdir()
    }


@pytest.mark.parametrize("stage", ["download", "features"])
def test_data_stages_fail_cleanly_without_data_or_network(stage, tmp_path):
    # empty data dir + SHIFTGUARD_OFFLINE=1: no download attempt, clear exit code 3
    result = runner.invoke(app, [stage, "-s", f"paths.data_dir={tmp_path.as_posix()}"])
    assert result.exit_code == 3, result.output


@pytest.mark.parametrize("stage", ["run", "report", "all"])
def test_unimplemented_stages_fail_loudly(stage):
    result = runner.invoke(app, [stage])
    assert result.exit_code == 2
