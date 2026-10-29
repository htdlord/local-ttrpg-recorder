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
    assert runner.invoke(app, ["config", "init", str(path), "--preset", "fast"]).exit_code == 0
    assert runner.invoke(app, ["config", "init", str(path)]).exit_code == 1  # no overwrite
    result = runner.invoke(app, ["config", "validate", str(path)])
    assert result.exit_code == 0 and "mode fast" in result.output


def test_config_reports_invalid_settings():
    result = runner.invoke(app, ["config", "show", "--set", "decoding.beam_size=0"])
    assert result.exit_code == 2 and "decoding.beam_size" in result.output


def test_presets_list():
    result = runner.invoke(app, ["presets", "list"])
    assert result.exit_code == 0 and "conversation" in result.output and "fast" in result.output


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


def test_session_init(tmp_path):
    import numpy as np
    import soundfile as sf

    from sessionscribe.core.session import Session

    tracks = tmp_path / "rec"
    tracks.mkdir()
    for name in ("ch01_gm", "ch02_anna"):
        sf.write(str(tracks / f"{name}.wav"), np.zeros(480, np.float32), 48_000)
    out = tmp_path / "s" / "session.toml"
    result = runner.invoke(app, ["session", "init", str(tracks), "--out", str(out)])
    assert result.exit_code == 0, result.output
    session = Session.load(out)
    assert [p.id for p in session.participants] == ["gm", "anna"]
    assert (
        session.source_path(session.participants[1]).resolve()
        == (tracks / "ch02_anna.wav").resolve()
    )
    assert runner.invoke(app, ["session", "init", str(tracks), "--out", str(out)]).exit_code == 1


def _fake_pipeline(monkeypatch):
    from sessionscribe.core.pipeline import run
    from tests.fakes import FakeFactory

    real_process, real_export = run.process_session, run.export_session

    def identity(settings):
        return {"id": "fake", "revision": "r1", "device": "cpu", "compute_type": "int8"}

    def process(session, settings, **kw):
        kw.update(isolation="inline", factory=FakeFactory(), model_identity=identity)
        return real_process(session, settings, **kw)

    def export(session, settings, **kw):
        return real_export(session, settings, model_identity=identity, **kw)

    monkeypatch.setattr(run, "process_session", process)
    monkeypatch.setattr(run, "export_session", export)


def _session_folder(tmp_path):
    import numpy as np
    import soundfile as sf

    folder = tmp_path / "rec"
    folder.mkdir()
    x = np.zeros(48_000 * 3, np.float32)
    x[48_000:72_000] = 0.4 * np.sin(np.arange(24_000) * 2 * np.pi * 500 / 48_000)
    for name in ("ch01_gm", "ch02_anna"):
        sf.write(str(folder / f"{name}.wav"), x, 48_000, subtype="FLOAT")
    assert runner.invoke(app, ["session", "init", str(folder)]).exit_code == 0
    return folder


def test_transcribe_session_export_and_clean(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    folder = _session_folder(tmp_path)
    out = tmp_path / "out"
    args = [
        "transcribe",
        "--session",
        str(folder),
        "-l",
        "hu",
        "--out",
        str(out),
        "--formats",
        "json,txt,srt,vtt,md",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "2 chunks: 0 from cache, 2 transcribed" in result.output
    assert sorted(p.suffix for p in out.iterdir()) == [".json", ".md", ".srt", ".txt", ".vtt"]
    again = runner.invoke(app, ["export", "--session", str(folder), "--out", str(out)])
    assert again.exit_code == 0, again.output
    assert "2 from cache, 0 transcribed" in again.output
    assert (folder / "work").is_dir()
    assert runner.invoke(app, ["clean", "--session", str(folder), "--yes"]).exit_code == 0
    assert not (folder / "work").exists()
    missing = runner.invoke(app, ["export", "--session", str(folder), "--out", str(out)])
    assert missing.exit_code == 1 and "no cached ASR result" in missing.output


def test_transcribe_single_file_and_argument_check(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    folder = _session_folder(tmp_path)
    file = folder / "ch01_gm.wav"
    result = runner.invoke(app, ["transcribe", str(file), "-l", "hu", "--out", str(tmp_path / "o")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "o" / "ch01_gm.json").is_file()
    both = runner.invoke(app, ["transcribe", str(file), "--session", str(folder)])
    assert both.exit_code == 2 and "either FILE or --session" in both.output
