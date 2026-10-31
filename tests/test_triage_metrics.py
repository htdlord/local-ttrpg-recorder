import pytest

from ttrpg_recorder.core.bench.triage import TriageFile, TriageUnit
from ttrpg_recorder.core.bench.triage_metrics import locate, score_triage


def unit(uid, speaker, a, b, label, source=None):
    return TriageUnit(id=uid, speaker=speaker, start=a, end=b, label=label, source=source)


def seg(uid, speaker, a, b, **kw):
    return {"id": uid, "speaker": speaker, "start": a, "end": b, **kw}


TRIAGE = TriageFile(
    clip="c",
    speakers=["gm", "anna"],
    units=[
        unit("gm-1", "gm", 0, 2, "s"),  # kept
        unit("gm-2", "gm", 3, 4, "v"),  # mixed: speech; suppressed -> false suppression
        unit("anna-1", "anna", 0, 2, "b", "gm"),  # bleed kept -> residual
        unit("anna-2", "anna", 5, 6, "b", "gm"),  # bleed suppressed
        unit("anna-3", "anna", 7, 9, "o"),  # overlap kept + marked
        unit("anna-4", "anna", 10, 11, "w", "gm"),  # wrong speaker, kept under anna
        unit("gm-3", "gm", 12, 13, "n"),  # hallucination kept
        unit("gm-4", "gm", 14, 15, "?"),  # unsure: not scored
        unit("gm-m000", "gm", 16, 18, "s"),  # missed speech, still absent
    ],
)
OUTPUT = {
    "segments": [
        seg("gm-1", "gm", 0, 2),
        seg("anna-1", "anna", 0, 2),
        seg("other-id", "anna", 7.1, 9, overlap=True),  # pre mode: new id, same time
        seg("anna-4", "anna", 10, 11),
        seg("gm-3", "gm", 12, 13),
    ],
    "suppressed": [seg("gm-2", "gm", 3, 4), seg("anna-2", "anna", 5, 6)],
}


def test_locate_by_id_then_time():
    assert locate(TRIAGE.units[0], OUTPUT)[0] == "kept"
    assert locate(TRIAGE.units[4], OUTPUT)[1]["id"] == "other-id"
    assert locate(TRIAGE.units[1], OUTPUT)[0] == "suppressed"
    assert locate(TRIAGE.units[8], OUTPUT)[0] == "absent"
    # same time on another speaker's track is not a match
    assert locate(unit("x", "gm", 7, 9, "s"), OUTPUT)[0] == "absent"


def test_score():
    s = score_triage(TRIAGE, OUTPUT)
    assert (s.labelled, s.unsure, s.speech_units) == (9, 1, 5)  # s, v, o, w, missed s
    assert s.speech_recall == pytest.approx(3 / 5)  # gm-1, anna-3, anna-4
    assert s.speech_recall_s == pytest.approx(5 / 8)
    assert s.false_suppression == pytest.approx(1 / 5) and s.speech_absent == 1
    assert (s.bleed_units, s.residual_duplicates) == (2, 0.5)  # v is not counted as bleed
    assert (s.overlap_kept, s.overlap_marked) == (1.0, 1.0)
    assert s.hallucination_residue == 1.0
    assert (s.wrong_speaker_units, s.attribution) == (1, 0.0)


def test_empty_groups_give_none():
    s = score_triage(TriageFile(clip="c", speakers=["gm"]), {"segments": []})
    assert s.speech_recall is None and s.residual_duplicates is None


def test_residual_split_by_source_mic():
    tri = TriageFile(
        clip="c",
        speakers=["gm", "anna", "bela"],
        units=[
            unit("a1", "anna", 0, 1, "b", "gm"),
            unit("a2", "anna", 2, 3, "b", "bela"),
            unit("g1", "gm", 4, 5, "b", "bela"),
        ],
    )
    out = {
        "silent_mics": ["bela"],
        "segments": [seg("a2", "anna", 2, 3), seg("g1", "gm", 4, 5)],
        "suppressed": [seg("a1", "anna", 0, 1)],
    }
    s = score_triage(tri, out)
    assert (s.bleed_units_working, s.residual_duplicates_working) == (1, 0.0)
    assert (s.bleed_units_silent, s.residual_duplicates_silent) == (2, 1.0)
    assert score_triage(tri, out, silent_mics=[]).residual_duplicates_working == pytest.approx(
        2 / 3, abs=1e-3
    )
