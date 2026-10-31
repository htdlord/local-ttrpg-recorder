import json

from ttrpg_recorder.core.config import load_settings
from ttrpg_recorder.core.exporters.formats import (
    output_stem,
    to_json,
    to_md,
    to_srt,
    to_txt,
    to_vtt,
    write_session_outputs,
)
from ttrpg_recorder.core.merge import RecordingGap, TrackInput, merge_tracks
from ttrpg_recorder.core.pipeline.asr import Segment, Word
from ttrpg_recorder.core.pipeline.chunks import KeptSegment
from ttrpg_recorder.core.result import SessionResult
from ttrpg_recorder.core.session import Participant

R = 48_000


def seg(index, a, b, text, logprob=-0.2):
    a, b = int(a * R), int(b * R)
    return Segment(index, a, b, text, logprob, 0.01, 1.2, 0.0, (Word(a, b, f" {text}", 0.9),))


def result(**output):
    settings = load_settings(overrides=[f"output.{k}={v}" for k, v in output.items()])
    tracks = [
        TrackInput(
            "gm",
            0,
            [
                KeptSegment(0, seg(0, 1.25, 2.5, "Árvíztűrő")),
                KeptSegment(0, seg(1, 3.0, 4.0, "tükörfúrógép")),
            ],
            None,
        ),
        TrackInput("anna", R, [KeptSegment(0, seg(0, 5.0, 6.0, "ő ű á", logprob=-3.0))], None),
    ]
    return SessionResult(
        name="Kampány 3",
        date="2026-10-29",
        participants=[
            Participant(id="gm", name="Mesélő", source="ch01_gm.wav"),
            Participant(id="anna", name="Anna", source="anna/", offset_s=1.0),
        ],
        settings=settings,
        transcript=merge_tracks(tracks, settings.quality),
        duration=10 * R,
        model={"model_id": "large-v3", "compute_type": "float16", "revision": "r1"},
        languages={"gm": "hu", "anna": "hu"},
        gaps=[RecordingGap(8 * R, R // 2, "input_overflow")],
    )


def test_txt_and_same_speaker_join():
    assert to_txt(result()) == (
        "[00:00:01] Mesélő: Árvíztűrő\n[00:00:03] Mesélő: tükörfúrógép\n[00:00:06] Anna: ő ű á\n"
    )
    joined = to_txt(result(merge_same_speaker_gap_s=1.0, timestamp_style='"hh:mm:ss.mmm"'))
    assert joined.splitlines()[0] == "[00:00:01.250] Mesélő: Árvíztűrő tükörfúrógép"


def test_srt_and_vtt_millisecond_formats():
    srt = to_srt(result())
    assert srt.startswith("1\n00:00:01,250 --> 00:00:02,500\nMesélő: Árvíztűrő\n\n2\n")
    vtt = to_vtt(result())
    assert vtt.startswith("WEBVTT\n\n00:00:01.250 --> 00:00:02.500\nMesélő: Árvíztűrő\n")
    assert "00:00:06.000 --> 00:00:07.000\nAnna: ő ű á" in vtt


def test_md_marks_flags_and_gaps():
    md = to_md(result())
    assert md.startswith("# Kampány 3\n")
    assert "> **Recording gaps:** 00:00:08 (0.5 s, input_overflow)" in md
    assert "**[00:00:06] Anna:** ⚠ ő ű á" in md  # low logprob flag
    assert "**[00:00:01] Mesélő:** Árvíztűrő" in md


def test_json_schema_fields():
    data = to_json(result())
    assert data["session"]["timeline_rate"] == R and data["bleed_mode_applied"] == "off"
    anna = data["segments"][-1]
    assert anna["id"] == "anna-c000-0000" and anna["start"] == 6.0
    assert anna["audio_start_sample"] == 5 * R  # track-local
    assert anna["quality_flags"] == ["low_logprob"]
    assert data["participants"][1] == {
        "id": "anna",
        "name": "Anna",
        "source": "anna/",
        "offset_s": 1.0,
    }
    assert data["recording_gaps"][0]["start_sample"] == 8 * R


def test_write_outputs_without_bom(tmp_path):
    res = result(formats='["json","txt","srt","vtt","md"]')
    paths = write_session_outputs(res, tmp_path)
    assert sorted(p.name for p in paths) == [
        f"Kampány 3.{ext}" for ext in sorted(["json", "md", "srt", "txt", "vtt"])
    ]
    for p in paths:
        assert p.read_bytes()[:3] != b"\xef\xbb\xbf"
    assert json.loads((tmp_path / "Kampány 3.json").read_text("utf-8"))["segments"]


def test_output_stem_is_filename_safe():
    res = result(filename_template='"{date} {stem}: a/b"')
    assert output_stem(res) == "2026-10-29 Kampány 3_ a_b"
