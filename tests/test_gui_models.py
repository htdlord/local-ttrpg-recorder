"""GUI view models without Qt."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from tests.test_attribution import seg
from ttrpg_recorder.core.config import Settings, load_settings
from ttrpg_recorder.core.merge import RecordingGap, Transcript
from ttrpg_recorder.core.result import SessionResult
from ttrpg_recorder.core.session import Participant, SessionConfig
from ttrpg_recorder.gui.models import colors, participants, results
from ttrpg_recorder.gui.models import settings_form as sf

R = 48_000


def test_field_specs_cover_every_setting_with_kinds_and_bounds():
    specs = {s.path: s for s in sf.field_specs()}
    assert specs["asr.mode"].kind == "choice" and specs["asr.mode"].choices == ("quality", "fast")
    assert specs["output.formats"].kind == "multichoice"
    assert specs["decoding.temperature"].kind == "floats"
    assert specs["asr.batch_size"].optional and specs["asr.batch_size"].minimum == 1
    assert specs["bleed.suppress_below"].maximum == 1
    assert specs["model.id"].editable_choices and "large-v3" in specs["model.id"].choices
    assert {s.split(".")[0] for s in specs} >= {"model", "asr", "bleed", "quality", "output"}
    assert next(s.path for s in sf.basic_specs()) == "model.id"


def test_mode_awareness():
    specs = {s.path: s for s in sf.field_specs()}
    assert sf.is_disabled(specs["decoding.best_of"], "fast")
    assert not sf.is_disabled(specs["decoding.best_of"], "quality")
    assert (
        sf.mode_note(specs["decoding.temperature"], "fast")
        == "only the first value is used in this mode"
    )
    assert sf.is_disabled(specs["asr.batch_size"], "quality")


def test_values_diff_and_text_parsing():
    base = load_settings()
    changed = sf.with_value(base, "decoding.beam_size", 3)
    changed = sf.with_value(changed, "bleed.mode", "pre")
    assert sf.settings_diff(base, changed) == {
        "decoding": {"beam_size": 3},
        "bleed": {"mode": "pre"},
    }
    assert sf.changed_paths(sf.settings_diff(base, changed)) == ["decoding.beam_size", "bleed.mode"]
    with pytest.raises(ValidationError):
        sf.with_value(base, "decoding.beam_size", 0)
    specs = {s.path: s for s in sf.field_specs()}
    assert sf.parse_text(specs["decoding.temperature"], "0 0,2 0.4") == [0.0, 0.2, 0.4]
    assert sf.parse_text(specs["asr.batch_size"], "") is None
    assert sf.format_value(specs["decoding.temperature"], [0.0, 0.2]) == "0 0.2"
    assert sf.get_value(Settings(), "asr.mode") == "quality"


def test_participant_rows_validate_per_row():
    base = SessionConfig(participants=[Participant(id="gm", source="a.wav")])
    rows = [
        participants.ParticipantRow(id="gm", name="Mesélő", source="a.wav"),
        participants.ParticipantRow(id="anna", source="b.wav", offset_s="1,5", channel="2"),
    ]
    ok = participants.validate(base, rows)
    assert (
        ok.ok
        and ok.config.participants[1].offset_s == 1.5
        and ok.config.participants[1].channel == 2
    )
    bad = participants.validate(
        base, [participants.ParticipantRow(id="gm", source="a.wav", offset_s="-1")]
    )
    assert not bad.ok and "offset_s" in bad.row_errors[0][0]
    dup = participants.validate(
        base,
        [
            participants.ParticipantRow(id="gm", source="a.wav"),
            participants.ParticipantRow(id="gm", source="b.wav"),
        ],
    )
    assert not dup.ok and dup.errors == ["participant ids must be unique"]
    back = participants.ParticipantRow.from_participant(ok.config.participants[1])
    assert (back.channel, back.offset_s) == ("2", "1.5")


def make_result():
    t = Transcript(
        segments=[
            seg("gm", 1, 2, "egy", sid="a"),
            seg("anna", 3, 4, "kettő", sid="b"),
        ],
        suppressed=[seg("anna", 1, 2, "egy", sid="c")],
    )
    t.segments[1] = replace(
        t.segments[1], quality_flags=("possible_wrong_speaker",), probable_speaker="unknown"
    )
    t.suppressed[0] = replace(t.suppressed[0], suppression_reason="bleed", bleed_source="gm")
    return SessionResult(
        name="s",
        date=None,
        participants=[Participant(id="gm", source="a.wav"), Participant(id="anna", source="b.wav")],
        settings=Settings(),
        transcript=t,
        duration=10 * R,
        gaps=[RecordingGap(5 * R, R // 2, "buffer_overflow")],
    )


def test_result_rows_filters_and_counts():
    rows = results.rows_from_result(make_result())
    assert [r.kind for r in rows] == ["segment", "suppressed", "segment", "gap"]
    f = results.ResultFilter()
    assert [r.id for r in results.filter_rows(rows, f)] == ["a", "b", "gap-0"]
    f.show_suppressed = True
    assert "c" in [r.id for r in results.filter_rows(rows, f)]
    assert [r.id for r in results.filter_rows(rows, results.ResultFilter(only_review=True))] == [
        "b"
    ]
    assert [r.id for r in results.filter_rows(rows, results.ResultFilter(text="KETT"))] == ["b"]
    assert [
        r.id
        for r in results.filter_rows(rows, results.ResultFilter(speakers={"gm"}, show_gaps=False))
    ] == ["a"]
    assert results.counts(rows) == {"segments": 2, "review": 1, "suppressed": 1, "gaps": 1}
    b = next(r for r in rows if r.id == "b")
    assert b.can_reassign and b.other_speaker == "unknown"
    assert not next(r for r in rows if r.id == "c").can_reassign


def test_eta_and_duration_format():
    assert results.eta_seconds(0, 100, 10) is None
    assert results.eta_seconds(25, 100, 10) == pytest.approx(30)
    assert results.format_duration(3725) == "1:02:05" and results.format_duration(65) == "1:05"


@pytest.mark.parametrize("tokens", [colors.LIGHT, colors.DARK])
def test_theme_contrast(tokens):
    for c in tokens.speakers:
        assert colors.contrast(c, tokens.surface) >= 4.5, c
    assert colors.contrast(tokens.text, tokens.surface) >= 7
    assert colors.contrast(tokens.text_muted, tokens.surface) >= 4.5
    assert colors.contrast(tokens.accent_text, tokens.accent) >= 4.5
    for c in (tokens.warning, tokens.error, tokens.success, tokens.accent):
        assert colors.contrast(c, tokens.surface) >= 4.5, c
