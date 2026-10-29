import shutil

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from sessionscribe.core.audio.reader import AudioReadError
from sessionscribe.core.audio.track import VirtualTrack, file_fingerprint
from sessionscribe.core.config import load_settings
from sessionscribe.core.session import (
    Session,
    SessionConfig,
    suggest_session,
    write_session,
)


def ramp(n, start=0):
    return ((np.arange(start, start + n) % 1000) / 1000.0).astype(np.float32)


def write(path, data, rate=48_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, rate, subtype="PCM_24" if path.suffix == ".flac" else "FLOAT")
    return path


def test_virtual_track_reads_across_segment_files(tmp_path):
    seg = tmp_path / "anna"
    write(seg / "000.wav", ramp(1000))
    write(seg / "001.wav", ramp(500, 1000))
    write(seg / "002.wav", ramp(700, 1500))
    track = VirtualTrack.open(seg)
    assert track.frames == 2200 and track.samplerate == 48_000
    np.testing.assert_allclose(track.read(900, 800), ramp(800, 900))
    np.testing.assert_allclose(track.read(2100, 500), ramp(100, 2100))  # clipped at the end
    assert [p.info.path.name for p in track.parts_in(900, 200)] == ["000.wav", "001.wav"]
    blocks = list(track.blocks(0, None, block_frames=300))
    np.testing.assert_allclose(np.concatenate(blocks), ramp(2200))


def test_virtual_track_rejects_mixed_formats(tmp_path):
    write(tmp_path / "s" / "a.wav", ramp(100))
    write(tmp_path / "s" / "b.wav", ramp(100), rate=44_100)
    with pytest.raises(AudioReadError, match="differs"):
        VirtualTrack.open(tmp_path / "s")


def test_fingerprint_ignores_mtime_and_touches_only_range(tmp_path):
    seg = tmp_path / "s"
    write(seg / "a.wav", ramp(1000))
    write(seg / "b.wav", ramp(1000, 1000))
    track = VirtualTrack.open(seg)
    first, whole = track.fingerprint(0, 500), track.fingerprint(0, 2000)
    copy = tmp_path / "copy"
    shutil.copytree(seg, copy)  # new mtimes
    assert VirtualTrack.open(copy).fingerprint(0, 500) == first
    write(seg / "b.wav", ramp(1000, 7))  # changes the second file only
    again = VirtualTrack.open(seg)
    assert again.fingerprint(0, 500) == first
    assert again.fingerprint(0, 2000) != whole


def test_file_fingerprint_edges(tmp_path):
    big = tmp_path / "big.bin"
    data = bytearray(3 << 20)
    big.write_bytes(bytes(data))
    edge, full = file_fingerprint(big), file_fingerprint(big, full=True)
    data[len(data) // 2] = 1  # middle change: only the full hash sees it
    big.write_bytes(bytes(data))
    assert file_fingerprint(big) == edge
    assert file_fingerprint(big, full=True) != full
    data[0] = 1
    big.write_bytes(bytes(data))
    assert file_fingerprint(big) != edge


def test_session_round_trip_and_suggestion(tmp_path):
    write(tmp_path / "tracks" / "ch01_gm.wav", ramp(100))
    write(tmp_path / "tracks" / "ch03_Ádám.wav", ramp(100))
    write(tmp_path / "tracks" / "seat_x" / "000.flac", ramp(100))
    config = suggest_session(tmp_path)
    assert [(p.id, p.source) for p in config.participants] == [
        ("gm", "tracks/ch01_gm.wav"),
        ("Adam", "tracks/ch03_Ádám.wav"),
        ("seat_x", "tracks/seat_x"),
    ]
    assert config.participants[1].name == "Ádám" and config.name == tmp_path.name
    config.participants[1].offset_s = 1.5
    write_session(tmp_path / "session.toml", config)
    assert (tmp_path / "session.toml").read_bytes()[:3] != b"\xef\xbb\xbf"
    session = Session.load(tmp_path)
    assert session.config == config
    assert session.participants[1].offset_samples == 72_000
    assert session.open_track(session.participants[2]).frames == 100
    assert session.work_dir() == tmp_path / "work"


@pytest.mark.parametrize(
    "participants",
    [
        [{"id": "a", "source": "x.wav"}, {"id": "a", "source": "y.wav"}],
        [{"id": "a b", "source": "x.wav"}],
        [{"id": "a", "source": "x.wav", "offset_s": -1.0}],
        [{"id": f"p{i}", "source": "x.wav"} for i in range(9)],
        [],
    ],
)
def test_session_validation(participants):
    with pytest.raises(ValidationError):
        SessionConfig(participants=participants)


def test_session_settings_layer(tmp_path):
    cfg = tmp_path / "s.toml"
    cfg.write_text('[decoding]\nlanguage = "en"\nbeam_size = 3\n', encoding="utf-8")
    s = load_settings(
        preset=None,
        config_path=cfg,
        session={"decoding": {"language": "hu"}},
        session_preset="fast",
        overrides=["decoding.beam_size=2"],
    )
    assert s.asr.mode == "fast"  # session preset
    assert s.decoding.language == "hu"  # session beats --config
    assert s.decoding.beam_size == 2  # --set beats session
