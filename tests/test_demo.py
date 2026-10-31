"""Demo session: assembly from synthetic voices (no download)."""

import itertools

import numpy as np
import pytest

from sessionscribe.core import demo
from sessionscribe.core.session import Session

R = demo.RATE


def voices(n=4, seconds=200.0, seed=1):
    """Speech-like noise bursts (0.4 s on, 0.15 s off), a different level per voice."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * R)) / R
    gate = ((t % 0.55) < 0.4).astype(np.float32)
    return [(rng.normal(0, 0.05, len(t)) * gate).astype(np.float32) for _ in range(n)]


def db(x):
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-20)


def test_script_has_a_mic_check_one_overlap_and_alternating_turns():
    lines = demo.script(4)
    assert [line.speaker for line in lines[:4]] == [0, 1, 2, 3]
    assert sum(1 for line in lines if line.gap_s < 0) == 1
    assert all(a.speaker != b.speaker for a, b in itertools.pairwise(lines[4:]))
    assert demo.script(4) == lines  # deterministic


def test_assembly_places_lines_and_bleeds_with_the_given_coupling():
    lines = demo.script(4)
    tracks, placed = demo.assemble(voices(), lines, noise_dbfs=-100.0)
    assert len(tracks) == 4 and len({len(t) for t in tracks}) == 1
    assert len(placed) == len(lines)
    # cuts snap to pauses, so lengths stay near the wanted ones
    for line, p in zip(lines, placed, strict=True):
        assert abs((p.end - p.start) / R - line.target_s) <= 0.8
    overlaps = [(a, b) for a, b in itertools.pairwise(placed) if b.start < a.end]
    assert len(overlaps) == 1
    # mic check: speaker 1 alone; mic 0 hears it at C[0][1] (room() keeps the level)
    solo = placed[1]
    own = tracks[1][solo.start : solo.end]
    other = tracks[0][solo.start : solo.end]
    assert db(other) - db(own) == pytest.approx(demo.COUPLING_DB[0][1], abs=1.5)


def test_create_demo_writes_a_loadable_session(tmp_path):
    path = demo.create_demo(tmp_path / "demo", voices=voices())
    session = Session.load(path)
    assert [p.id for p in session.participants] == [v.id for v in demo.VOICES]
    assert [p.name for p in session.participants][1:] == ["Bodor", "Kira", "Lin"]
    lin = session.participants[3]
    assert lin.offset_s == demo.LATE_START_S["lin"]
    lengths = [session.open_track(p).frames / R + p.offset_s for p in session.participants]
    assert max(lengths) - min(lengths) < 0.01  # same session length once offsets apply
    starts = [session.config.mic_check[v.id][0] for v in demo.VOICES]
    assert starts == sorted(starts)
    assert session.config.settings == {"decoding": {"language": "en"}}
    credits = (path.parent / "SOURCES.md").read_text(encoding="utf-8")
    assert all(v.reader in credits and v.url in credits for v in demo.VOICES)
    with pytest.raises(FileExistsError):
        demo.create_demo(tmp_path / "demo", voices=voices())


def test_offline_without_sources_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline"):
        demo.create_demo(tmp_path / "demo", offline=True)
