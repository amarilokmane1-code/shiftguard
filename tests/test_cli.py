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


@pytest.mark.parametrize("stage", ["download", "features", "run", "report", "all"])
def test_unimplemented_stages_fail_loudly(stage):
    result = runner.invoke(app, [stage])
    assert result.exit_code == 2
