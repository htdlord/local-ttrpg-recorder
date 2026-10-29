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


def test_bench_reference_workflow(tmp_path):
    import json

    import numpy as np
    import soundfile as sf

    clip = tmp_path / "clip01"
    (clip / "tracks").mkdir(parents=True)
    (clip / "source").mkdir()
    for name in ("ch01_gm", "ch03_seat_1"):
        sf.write(clip / "tracks" / f"{name}.wav", np.zeros(48_000 * 5, np.float32), 48_000)
    rec = {"id": 1, "start": 0.5, "end": 1.5, "channel": 1, "text": "Szia, ő az.", "flags": []}
    (clip / "source" / "session.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    def ok(*args):
        result = runner.invoke(app, ["bench", *args])
        assert result.exit_code == 0, result.output
        return result.output

    ok("clip", "init", str(clip), "--language", "hu")
    ok("ref", "import", str(clip))
    ok("ref", "labels", str(clip))
    assert runner.invoke(app, ["bench", "ref", "labels", str(clip)]).exit_code == 1  # no overwrite
    out = ok(
        "ref",
        "build",
        str(clip),
        "--labels",
        str(clip / "labels" / "gm.txt"),
        "--labels",
        str(clip / "labels" / "seat_1.txt"),
    )
    assert "1 utterances" in out and "Szia" not in out  # counts, never text
    ok("ref", "check", str(clip))
    result = runner.invoke(app, ["bench", "run", "--refs", str(tmp_path), "--matrix", "nope"])
    assert result.exit_code == 1
