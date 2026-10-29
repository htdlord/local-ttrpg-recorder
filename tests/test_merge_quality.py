from dataclasses import replace

import pytest

from sessionscribe.core import quality as q
from sessionscribe.core.config import QualitySettings
from sessionscribe.core.merge import (
    BOUNDARY_DUPLICATE,
    RecordingGap,
    TrackInput,
    join_same_speaker,
    merge_tracks,
    segment_id,
)
from sessionscribe.core.pipeline.asr import Segment, SpeechInterval, Word
from sessionscribe.core.pipeline.chunks import KeptSegment

R = 48_000
QS = QualitySettings()


def seg(index, start_s, end_s, text="szia", **kw):
    a, b = int(start_s * R), int(end_s * R)
    base = Segment(index, a, b, text, -0.2, 0.01, 1.2, 0.0, (Word(a, b, f" {text}", 0.9),))
    return replace(base, **kw)


def test_merge_sorts_on_session_time_with_offsets():
    tracks = [
        TrackInput("gm", 0, [KeptSegment(0, seg(0, 5, 6)), KeptSegment(1, seg(3, 20, 21))], None),
        TrackInput("anna", 2 * R, [KeptSegment(0, seg(0, 3, 4))], None),  # 5 s session time
        TrackInput("bela", 0, [KeptSegment(0, seg(7, 1, 2))], None),
    ]
    out = merge_tracks(tracks, QS)
    assert [s.id for s in out.segments] == [
        "bela-c000-0007",
        "gm-c000-0000",  # same start as anna: participant order decides
        "anna-c000-0000",
        "gm-c001-0003",
    ]
    anna = out.segments[2]
    assert anna.start == 5 * R and anna.track_start == 3 * R and anna.words[0].start == 5 * R


def test_ids_do_not_depend_on_offsets_or_names():
    kept = [KeptSegment(2, seg(4, 1, 2))]
    a = merge_tracks([TrackInput("p1", 0, kept, None)], QS).segments[0]
    b = merge_tracks([TrackInput("p1", 9 * R, kept, None)], QS).segments[0]
    assert a.id == b.id == segment_id("p1", 2, 4) == "p1-c002-0004"


def test_boundary_duplicates_are_suppressed_with_reason():
    kept = [KeptSegment(0, seg(0, 1, 2)), KeptSegment(0, seg(1, 9, 10), 1, True)]
    out = merge_tracks([TrackInput("p", 0, kept, None)], QS)
    assert [s.id for s in out.segments] == ["p-c000-0000"]
    assert out.suppressed[0].suppression_reason == BOUNDARY_DUPLICATE


def test_join_same_speaker():
    tracks = [
        TrackInput("a", 0, [KeptSegment(0, seg(0, 0, 1)), KeptSegment(0, seg(1, 1.5, 2))], None),
        TrackInput("b", 0, [KeptSegment(0, seg(0, 5, 6))], None),
    ]
    segs = merge_tracks(tracks, QS).segments
    assert [len(g) for g in join_same_speaker(segs, R)] == [2, 1]
    assert [len(g) for g in join_same_speaker(segs, 0)] == [1, 1, 1]


@pytest.mark.parametrize(
    ("changes", "flag"),
    [
        ({"avg_logprob": -1.5}, q.LOW_LOGPROB),
        ({"no_speech_prob": 0.9, "avg_logprob": -0.8}, q.HIGH_NO_SPEECH),
        ({"compression_ratio": 3.0}, q.HIGH_COMPRESSION),
        ({"text": "nem nem nem " * 4}, q.REPETITION),
        ({"text": "á" * 80}, q.TOO_LONG),
        ({"text": "á" * 40, "end": int(0.3 * R)}, q.TOO_LONG),
    ],
)
def test_quality_flags_positive_and_negative(changes, flag):
    vad = [SpeechInterval(0, 3 * R)]
    assert q.quality_flags(seg(0, 0, 2), QS, vad=vad) == []
    assert flag in q.quality_flags(seg(0, 0, 2, **changes), QS, vad=vad)


def test_window_level_no_speech_alone_is_not_flagged():
    # no_speech_prob is shared by a whole 30 s window; a confident decode is not silence
    assert q.quality_flags(seg(0, 0, 2, no_speech_prob=0.9), QS, vad=None) == []
    # a short word with tight timestamps is not "too long"
    assert q.quality_flags(seg(0, 0, 0.3, text="szervusz"), QS, vad=None) == []


def test_vad_and_gap_flags():
    vad = [SpeechInterval(0, R // 2)]
    assert q.OUTSIDE_VAD in q.quality_flags(seg(0, 0, 2), QS, vad=vad)
    assert q.OUTSIDE_VAD not in q.quality_flags(seg(0, 0, 2), QS, vad=None)  # VAD off
    gaps = [(int(1.5 * R), 2 * R)]
    assert q.OVERLAPS_GAP in q.quality_flags(seg(0, 0, 2), QS, vad=None, gaps=gaps)
    assert q.OVERLAPS_GAP not in q.quality_flags(seg(0, 3, 4), QS, vad=None, gaps=gaps)
    assert (
        q.quality_flags(seg(0, 0, 2, avg_logprob=-9), QualitySettings(enabled=False), vad=None)
        == []
    )


def test_gap_for_one_participant_only():
    tracks = [
        TrackInput("a", 0, [KeptSegment(0, seg(0, 1, 2))], None),
        TrackInput("b", 0, [KeptSegment(0, seg(0, 1, 2))], None),
    ]
    out = merge_tracks(tracks, QS, [RecordingGap(R, R, "input_overflow", "b")])
    assert {s.speaker: s.quality_flags for s in out.segments} == {"a": (), "b": (q.OVERLAPS_GAP,)}
