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


def test_config_show_and_validate(tmp_path):
    result = runner.invoke(app, ["config", "show", "--set", "decoding.language=hu"])
    assert result.exit_code == 0 and 'language = "hu"' in result.output

    path = tmp_path / "s.toml"
    assert (
        runner.invoke(app, ["config", "init", str(path), "--preset", "fast-draft"]).exit_code == 0
    )
    assert runner.invoke(app, ["config", "init", str(path)]).exit_code == 1  # no overwrite
    result = runner.invoke(app, ["config", "validate", str(path)])
    assert result.exit_code == 0 and "mode fast" in result.output


def test_config_reports_invalid_settings():
    result = runner.invoke(app, ["config", "show", "--set", "decoding.beam_size=0"])
    assert result.exit_code == 2 and "decoding.beam_size" in result.output


def test_presets_list():
    result = runner.invoke(app, ["presets", "list"])
    assert (
        result.exit_code == 0 and "conversation" in result.output and "fast-draft" in result.output
    )
