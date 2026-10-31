"""bench clip cut: aligned tracks from a recorded session (segments, offsets)."""

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from ttrpg_recorder.cli import app
from ttrpg_recorder.core.bench.clip import Clip, cut_clip
from ttrpg_recorder.core.session import Session, SessionConfig, write_session

R = 48_000
MARK_S = 7.0  # session time of a pulse on every track


def pulse_track(seconds: float, at_s: float) -> np.ndarray:
    x = np.zeros(round(seconds * R), np.float32)
    x[round(at_s * R)] = 0.5
    return x


def make_session(folder):
    folder.mkdir()
    seg = folder / "rec" / "gm"
    seg.mkdir(parents=True)
    whole = pulse_track(12.0, MARK_S)
    sf.write(str(seg / "seg0001.flac"), whole[: 6 * R], R, subtype="PCM_24")  # boundary at 6 s
    sf.write(str(seg / "seg0002.flac"), whole[6 * R :], R, subtype="PCM_24")
    # starts 2 s into the session: the pulse is at track-local 5 s
    sf.write(str(folder / "anna.flac"), pulse_track(8.0, MARK_S - 2.0), R, subtype="PCM_24")
    config = SessionConfig.model_validate(
        {
            "name": "t",
            "participants": [
                {"id": "gm", "name": "Mesélő", "source": "rec/gm"},
                {"id": "anna", "name": "Anna", "source": "anna.flac", "offset_s": 2.0},
            ],
        }
    )
    write_session(folder / "session.toml", config)
    return Session(folder, config)


def test_cut_aligns_segments_and_offsets(tmp_path):
    session = make_session(tmp_path / "s")
    config = cut_clip(session, tmp_path / "clip02", start_s=5.0, duration_s=6.0, language="hu")
    clip = Clip.load(tmp_path / "clip02")
    assert [t.file for t in config.tracks] == ["tracks/ch01_gm.flac", "tracks/ch02_anna.flac"]
    assert [t.name for t in clip.config.tracks] == ["Mesélő", "Anna"]
    assert config.duration_s == 6.0
    for track in clip.config.tracks:
        x, rate = sf.read(str(clip.track_path(track)), dtype="float32")
        assert rate == R and len(x) == 6 * R
        assert int(np.argmax(np.abs(x))) == round((MARK_S - 5.0) * R)


def test_cut_pads_before_the_offset_and_after_the_end(tmp_path):
    session = make_session(tmp_path / "s")
    cut_clip(session, tmp_path / "c", start_s=0.0, duration_s=12.0, language="hu")
    anna, _ = sf.read(str(tmp_path / "c" / "tracks" / "ch02_anna.flac"), dtype="float32")
    assert len(anna) == 12 * R  # 2 s silence + 8 s track + 2 s silence
    assert int(np.argmax(np.abs(anna))) == round(MARK_S * R)


def test_cli_cut_refuses_an_existing_clip(tmp_path):
    session = make_session(tmp_path / "s")
    args = ["bench", "clip", "cut", str(tmp_path / "c"), "-s", str(session.folder)]
    first = CliRunner().invoke(app, [*args, "--start", "0:05", "--minutes", "0.1"])
    assert first.exit_code == 0, first.output
    assert "2 tracks" in first.output and "Anna" not in first.output  # names stay in the file
    again = CliRunner().invoke(app, [*args, "--start", "0:05", "--minutes", "0.1"])
    assert again.exit_code == 1
