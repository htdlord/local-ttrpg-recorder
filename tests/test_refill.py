"""Re-transcription of VAD speech that got no words."""

import json

import numpy as np
import soundfile as sf

from tests.fakes import FakeFactory
from ttrpg_recorder.core.config import load_settings
from ttrpg_recorder.core.exporters.formats import to_json, to_md
from ttrpg_recorder.core.pipeline.asr import Segment, SpeechInterval, Word
from ttrpg_recorder.core.pipeline.refill import uncovered
from ttrpg_recorder.core.pipeline.run import process_session
from ttrpg_recorder.core.quality import POSSIBLE_MISSED_SPEECH, REFILLED
from ttrpg_recorder.core.session import Participant, Session, SessionConfig, write_session

R = 48_000
IDENTITY = {"id": "fake", "revision": "r1", "device": "cpu", "compute_type": "int8"}


def seg(a, b, words=True):
    w = (Word(int(a * R), int(b * R), " x", 0.9),) if words else ()
    return Segment(0, int(a * R), int(b * R), "x", -0.2, 0.0, 1.0, 0.0, w)


def test_uncovered_stretches():
    vad = [SpeechInterval(0, 10 * R), SpeechInterval(20 * R, 21 * R)]
    holes = uncovered(vad, [seg(1, 2), seg(5, 9)], int(1.5 * R))
    assert holes == [(2 * R, 5 * R)]  # 0-1 s and 9-10 s are shorter than 1.5 s; 20-21 too
    assert uncovered(vad, [seg(0, 10, words=False)], R) == [(20 * R, 21 * R)]


def session_with_two_bursts(tmp_path):
    x = np.zeros(12 * R, np.float32)
    t = np.arange(2 * R) / R
    for start in (2, 7):
        x[start * R : (start + 2) * R] = 0.4 * np.sin(2 * np.pi * 500 * t)
    sf.write(str(tmp_path / "a.wav"), x, R, subtype="FLOAT")
    write_session(
        tmp_path / "session.toml",
        SessionConfig(participants=[Participant(id="a", name="Anna", source="a.wav")]),
    )
    return Session.load(tmp_path)


def run(session, factory, *overrides):
    return process_session(
        session,
        load_settings(overrides=list(overrides)),
        isolation="inline",
        factory=factory,
        model_identity=lambda s: dict(IDENTITY),
    )


def test_dropped_speech_is_refilled_and_flagged(tmp_path):
    session = session_with_two_bursts(tmp_path)
    result, stats = run(session, FakeFactory(drop_first_above_s=5.0))
    segs = result.transcript.segments
    assert len(segs) == 2 and stats.refilled == 1 and stats.missed == 0
    refilled = segs[0]
    assert REFILLED in refilled.quality_flags
    assert refilled.start == 2 * R  # placed at its true track time
    assert refilled.id == "a-c000-0001"  # raw index after the chunk's own segment
    assert result.transcript.missed == []


def test_still_silent_speech_is_reported_for_review(tmp_path):
    session = session_with_two_bursts(tmp_path)
    result, stats = run(session, FakeFactory(drop_first_above_s=5.0, deaf=True))
    assert stats.missed == 1 and len(result.transcript.segments) == 1
    (missed,) = result.transcript.missed
    assert missed.quality_flags == (POSSIBLE_MISSED_SPEECH,)
    assert (missed.start, missed.end) == (2 * R, 4 * R)
    data = json.loads(json.dumps(to_json(result)))
    assert data["possible_missed_speech"][0]["start"] == 2.0
    assert "**[00:00:02] Anna:** ⚠ _(possible missed speech, 2.0 s)_" in to_md(result)


def test_refill_can_be_switched_off(tmp_path):
    session = session_with_two_bursts(tmp_path)
    result, stats = run(session, FakeFactory(drop_first_above_s=5.0), "asr.refill_holes=false")
    assert stats.refilled == 0 and len(result.transcript.segments) == 1
    assert result.transcript.missed == []  # not checked when refill is off
