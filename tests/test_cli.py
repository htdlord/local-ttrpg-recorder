from typer.testing import CliRunner

from sessionscribe import __version__
from sessionscribe.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output and "record-probe" in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


def test_record_probe_rejects_bad_channels():
    result = runner.invoke(app, ["record-probe", "--channels", "0-3"])
    assert result.exit_code != 0
