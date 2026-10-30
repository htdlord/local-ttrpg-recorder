"""Triage reference: units, suggestions, labelling keys, rebuild, CSV."""

import numpy as np
import pytest
import soundfile as sf

from sessionscribe.core.bench import triage as t
from sessionscribe.core.bench.clip import Clip, init_clip
from sessionscribe.core.merge import TranscriptSegment
from sessionscribe.core.pipeline.asr import Word
from tests.fakes import FakeFactory

R = 48_000
FRAME = 960
IDENTITY = {"id": "fake", "revision": "r1", "device": "cpu", "compute_type": "int8"}


def env(seconds, level=-80.0, loud=()):
    e = np.full(int(seconds * 50), level, np.float32)
    for a, b, db in loud:
        e[int(a * 50) : int(b * 50)] = db
    return e


def tseg(speaker, a, b, text, sid=None):
    s, e = int(a * R), int(b * R)
    return TranscriptSegment(
        sid or f"{speaker}-c000-0000",
        speaker,
        f"{speaker}-c000",
        s,
        e,
        s,
        text,
        -0.2,
        0.0,
        1.0,
        0.0,
        (Word(s, e, f" {text}", 0.9),),
    )


def levels(own, other):
    return t.Levels(own, other, None if other is None else "anna")


@pytest.mark.parametrize(
    ("kind", "lv", "sim", "has", "expected"),
    [
        ("segment", levels(-2, -20), 0, True, ("s", None)),  # own clearly louder (relative)
        ("segment", levels(-2, -20), 95, True, ("s", None)),  # ... and another track echoes it
        ("segment", levels(-20, -2), 0, True, ("b", "anna")),  # source louder and has it too
        ("segment", levels(-20, -2), 0, False, (None, None)),  # louder but no segment there
        ("segment", levels(-8, -6), 90, True, ("b", "bela")),  # same text, other a bit louder
        ("segment", levels(-3, -4), 10, True, ("o", None)),  # both speak, different text
        ("segment", levels(-12, -8), 10, True, (None, None)),  # in between: a person decides
        ("missed", levels(-30, -40), 0, False, ("n", None)),
        ("missed", levels(-15, -2), 0, True, ("b", "anna")),
        ("missed", levels(-5, -8), 0, True, (None, None)),
        ("segment", levels(-2, None), 0, False, ("s", None)),  # single-track clip
    ],
)
def test_suggestion_rules(kind, lv, sim, has, expected):
    assert t.suggest(kind, lv, sim, "bela", other_has_segment=has) == expected


def test_speech_level_is_relative_to_each_track():
    quiet_mic = env(10, loud=[(0, 8, -40)])
    loud_mic = env(10, loud=[(0, 8, -15)])
    rel = t.relative_envelopes({"a": quiet_mic, "b": loud_mic})
    assert t.speech_level(quiet_mic) == pytest.approx(-40)
    assert float(rel["a"][10]) == pytest.approx(0) and float(rel["b"][10]) == pytest.approx(0)
    assert t.speech_level(env(2)) == -80.0  # silent track: its maximum


def test_unit_levels_use_loud_own_frames():
    envs = {
        "gm": env(4, loud=[(1, 2, -20)]),
        "anna": env(4, loud=[(1, 2, -35)]),
        "bela": env(4, loud=[(0, 4, -50)]),
    }
    lv = t.unit_levels(envs, "gm", int(0.5 * R), int(2.5 * R), FRAME)
    assert lv.own_db == pytest.approx(-20, abs=0.1)
    assert lv.other_speaker == "anna" and lv.other_db == pytest.approx(-35, abs=0.1)


def test_similarity_only_with_nearby_other_tracks():
    a = tseg("gm", 1, 2, "Menjünk a kocsmába!")
    near = tseg("anna", 1.2, 2.1, "menjünk a kocsmába", "anna-c000-0000")
    far = tseg("bela", 9, 10, "Menjünk a kocsmába!", "bela-c000-0000")
    score, who = t.best_similarity(a, [a, near, far])
    assert who == "anna" and score > 95
    assert t.best_similarity(a, [a, far]) == (0.0, None)


def labeller(n=3, speakers=("gm", "anna", "bela")):
    units = [
        t.TriageUnit(id=f"u{i}", speaker="gm", start=i, end=i + 1, suggestion="s" if i else None)
        for i in range(n)
    ]
    tf = t.TriageFile(clip="c", speakers=list(speakers), units=units)
    clock = iter(range(100))
    return tf, t.Labeller(tf, list(range(n)), clock=lambda: float(next(clock)))


def test_labeller_keys_sources_undo_and_status():
    tf, lab = labeller()
    assert lab.key("\r") == "none"  # unit 0 has no suggestion: a person must decide
    assert lab.key("b") == "need_source"
    assert lab.key("2") == "moved"
    assert (tf.units[0].label, tf.units[0].source) == ("b", "anna")
    assert lab.key("\r") == "moved" and tf.units[1].accepted_suggestion
    assert lab.key("u") == "moved" and lab.unit.id == "u1"  # back one
    assert lab.key("v") == "need_source" and lab.key("\r") == "moved"
    assert (tf.units[1].label, tf.units[1].source) == ("v", None)
    assert lab.key("w") == "need_source" and lab.key("\x1b") == "none" and lab.pending is None
    assert lab.key("w") == "need_source" and lab.key("9") == "none"  # no participant 9
    assert lab.key("3") == "moved" and tf.units[2].source == "bela"
    assert lab.done and tf.first_unlabelled() == 3
    tf.refresh_status()
    assert tf.status == "complete"
    assert lab.key("q") == "quit"


def test_labeller_other_keys():
    _, lab = labeller()
    assert [lab.key(k) for k in (" ", "m", "x", "h", "z")] == [
        "play",
        "play_mix",
        "play_other",
        "help",
        "none",
    ]
    assert "ONLY this track" in t.HELP and "THEIR track" in t.HELP


def test_visit_order():
    tf, _ = labeller(4, speakers=("gm", "anna"))
    tf.units[0].label = "s"
    tf.units[2].speaker = "anna"
    assert t.visit_order(tf, None, None) == [1, 2, 3]
    assert t.visit_order(tf, "all", 3) == [2, 3]
    assert t.visit_order(tf, "anna", None) == [2]
    assert t.visit_order(tf, "suggested", None) == [1, 2, 3]


def test_carry_labels_by_id_then_time():
    old = [
        t.TriageUnit(id="gm-c000-0000", speaker="gm", start=1, end=2, label="s"),
        t.TriageUnit(id="gm-c000-0001", speaker="gm", start=5, end=7, label="b", source="anna"),
        t.TriageUnit(id="gm-c000-0002", speaker="gm", start=20, end=21, label="n"),
    ]
    new = [
        t.TriageUnit(id="gm-c000-0000", speaker="gm", start=1, end=2),
        t.TriageUnit(id="gm-c001-0000", speaker="gm", start=5.2, end=7.1),  # new id, same time
        t.TriageUnit(id="gm-c001-0001", speaker="gm", start=30, end=31),
    ]
    lost = t.carry_labels(old, new)
    assert [(u.label, u.source) for u in new] == [("s", None), ("b", "anna"), (None, None)]
    assert [u.id for u in lost] == ["gm-c000-0002"]


def test_csv_round_trip(tmp_path):
    tf, _ = labeller(2, speakers=("gm", "Ádám"))
    tf.units[0].label, tf.units[0].source = "w", "Ádám"
    path = tmp_path / "t.csv"
    t.write_csv(path, tf)
    assert path.read_bytes()[:3] != b"\xef\xbb\xbf"
    text = path.read_text(encoding="utf-8").replace(
        "u1,gm,segment,1.000,2.000,s,,", "u1,gm,segment,1.000,2.000,s,o,"
    )
    fresh, _ = labeller(2, speakers=("gm", "Ádám"))
    updated, problems = t.apply_csv(
        fresh, "﻿" + text + "zz,gm,segment,0,1,,s,\nu0,gm,segment,0,1,,q,\n"
    )
    assert updated == 2
    assert [(u.label, u.source) for u in fresh.units] == [("w", "Ádám"), ("o", None)]
    assert problems == ["line 4: unknown id", "line 5: unknown label 'q'"]


def test_stats():
    tf, lab = labeller()
    lab.key("s")
    lab.key("\r")
    st = t.triage_stats(tf)
    assert (st.units, st.labelled, st.labels["s"], st.accepted, st.suggested) == (3, 2, 2, 1, 2)
    assert st.mean_decide_s == 1.0


def make_clip(tmp_path):
    folder = tmp_path / "clip"
    (folder / "tracks").mkdir(parents=True)
    x = np.zeros(6 * R, np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 500 * np.arange(R) / R)
    gm, anna = x.copy(), x.copy()
    gm[R : 2 * R] = tone  # gm speaks at 1 s; anna's mic hears it 20 dB lower
    anna[R : 2 * R] = 0.1 * tone
    anna[4 * R : 5 * R] = tone  # anna speaks at 4 s
    sf.write(str(folder / "tracks" / "ch01_gm.wav"), gm, R, subtype="FLOAT")
    sf.write(str(folder / "tracks" / "ch02_anna.wav"), anna, R, subtype="FLOAT")
    init_clip(folder, "hu")
    return Clip.load(folder)


def test_units_from_the_session_pipeline(tmp_path):
    clip = make_clip(tmp_path)
    kw = {"isolation": "inline", "factory": FakeFactory(), "model_identity": lambda s: IDENTITY}
    session, result = t.clip_result(clip, **kw)
    assert result.settings.bleed.mode == "off"
    tf = t.new_triage(clip, session, result)
    assert tf.speakers == ["gm", "anna"] and tf.asr["language"] == "hu"
    got = [(u.speaker, u.start, u.suggestion, u.suggestion_source) for u in tf.units]
    assert got == [
        ("gm", 1.0, "s", None),
        ("anna", 1.0, "b", "gm"),  # the fake engine transcribes the bleed too
        ("anna", 4.0, "s", None),
    ]
    path = t.triage_path(clip)
    t.write_triage(path, tf)
    assert "ő" not in path.read_text(encoding="utf-8")  # no transcript text in the file
    assert t.read_triage(path) == tf
    assert t.texts_by_id(result)[tf.units[0].id].startswith("ő0")


def test_mutual_bleed_suggestions_are_dropped():
    units = [
        t.TriageUnit(
            id="a", speaker="gm", start=0, end=2, suggestion="b", suggestion_source="anna"
        ),
        t.TriageUnit(
            id="b", speaker="anna", start=1, end=3, suggestion="b", suggestion_source="gm"
        ),
        t.TriageUnit(
            id="c", speaker="anna", start=5, end=6, suggestion="b", suggestion_source="gm"
        ),
    ]
    assert t.drop_contradictions(units) == 2
    assert [u.suggestion for u in units] == [None, None, "b"]
