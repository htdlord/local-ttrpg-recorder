"""Bleed attribution, dedup and overlap on hand-made ownership masks (session time)."""

import numpy as np

from sessionscribe.core.config import BleedSettings, DedupSettings
from sessionscribe.core.merge import Transcript, TranscriptSegment
from sessionscribe.core.pipeline.asr import Word
from sessionscribe.core.pipeline.attribution import (
    apply_bleed_handling,
    attribute_bleed,
    dedup,
    mark_overlaps,
)
from sessionscribe.core.pipeline.coupling import manual_coupling
from sessionscribe.core.pipeline.ownership import Ownership
from sessionscribe.core.quality import LOW_OWNERSHIP, POSSIBLE_DUPLICATE, POSSIBLE_WRONG_SPEAKER

R = 48_000
SPK = ["gm", "anna", "bela"]


def own(spans, seconds=20):
    """spans: speaker -> [(a, b), ...] seconds owned."""
    masks = np.zeros((len(SPK), seconds * 50), bool)
    for spk, ranges in spans.items():
        for a, b in ranges:
            masks[SPK.index(spk), int(a * 50) : int(b * 50)] = True
    return Ownership(SPK, 960, masks, manual_coupling(SPK, np.zeros((3, 3))), [])


def seg(speaker, a, b, text="menjünk tovább a hídon", sid=None, n_words=None):
    words = text.split()
    step = (b - a) / len(words)
    ws = tuple(
        Word(int((a + k * step) * R), int((a + (k + 1) * step) * R), f" {w}", 0.9)
        for k, w in enumerate(words)
    )
    return TranscriptSegment(
        sid or f"{speaker}-c000-{int(a * 10):04d}",
        speaker,
        f"{speaker}-c000",
        int(a * R),
        int(b * R),
        int(a * R),
        text,
        -0.2,
        0.0,
        1.0,
        0.0,
        ws,
    )


def run(segments, owned, **bleed):
    t = Transcript(segments=sorted(segments, key=lambda s: s.start))
    return apply_bleed_handling(t, owned, BleedSettings(**bleed), DedupSettings())


def test_single_speaker_with_bleed():
    out = run([seg("gm", 1, 3), seg("anna", 1, 3)], own({"gm": [(1, 3)]}))
    assert [s.speaker for s in out.segments] == ["gm"]
    (bleed,) = out.suppressed
    assert (bleed.speaker, bleed.suppression_reason, bleed.bleed_source) == ("anna", "bleed", "gm")
    assert out.segments[0].ownership == 1.0 and bleed.ownership == 0.0


def test_only_copy_is_kept_as_possible_wrong_speaker():
    # gm's mic owns the frames, but gm's track has no words there (the triage "w" case)
    out = run([seg("anna", 5, 7)], own({"gm": [(5, 7)]}))
    (kept,) = out.segments
    assert POSSIBLE_WRONG_SPEAKER in kept.quality_flags and kept.probable_speaker == "gm"
    assert not out.suppressed


def test_mixed_segment_with_own_words_is_kept():
    # anna's quiet remark (2 words owned) inside a stretch of gm's bleed on her mic
    words = "a b c d e f g h i j"
    out = run(
        [seg("gm", 0, 10, words), seg("anna", 0, 10, words, sid="anna-x")],
        own({"gm": [(0, 10)], "anna": [(4, 6)]}),
    )
    anna = [s for s in out.segments if s.speaker == "anna"]
    assert anna and LOW_OWNERSHIP in anna[0].quality_flags  # never dropped, flagged
    assert anna[0].ownership == 0.2


def test_quiet_speaker_owned_frames_keep_speech():
    out = run([seg("gm", 0, 6), seg("bela", 2, 3, "igen")], own({"gm": [(0, 6)], "bela": [(2, 3)]}))
    assert {s.speaker for s in out.segments} == {"gm", "bela"}
    bela = next(s for s in out.segments if s.speaker == "bela")
    assert bela.ownership == 1.0 and LOW_OWNERSHIP not in bela.quality_flags


def test_true_overlap_is_marked_and_kept():
    out = run(
        [seg("gm", 1, 4, "erre menjünk"), seg("anna", 2, 5, "nem arra")],
        own({"gm": [(1, 4)], "anna": [(2, 5)]}),
    )
    assert [s.overlap for s in out.segments] == [True, True] and not out.suppressed


def test_two_people_saying_the_same_short_word():
    out = run(
        [seg("gm", 1, 1.4, "nem"), seg("anna", 1.1, 1.5, "nem")],
        own({"gm": [(1, 1.4)], "anna": [(1.1, 1.5)]}),
    )
    assert len(out.segments) == 2 and not out.suppressed  # both owned: two speakers


def test_near_duplicate_text_needs_ownership_evidence():
    a = seg("gm", 1, 3, "menjünk tovább a hídon")
    b = seg("anna", 1.2, 3.1, "menjünk tovább a hídon át")
    # anna owns 40 %: above suppress_below, so not bleed; dedup removes it with evidence
    owned = own({"gm": [(1, 3)], "anna": [(1.2, 1.96)]})
    t = attribute_bleed(Transcript(segments=[a, b]), owned, BleedSettings())
    assert len(t.segments) == 2
    out = dedup(t, DedupSettings())
    assert [s.speaker for s in out.segments] == ["gm"]
    assert out.suppressed[0].suppression_reason == "duplicate"
    assert out.suppressed[0].duplicate_of == a.id
    # without ownership values (no evidence): kept and flagged
    flagged = dedup(Transcript(segments=[a, b]), DedupSettings())
    assert [POSSIBLE_DUPLICATE in s.quality_flags for s in flagged.segments] == [True, True]


def test_short_duplicate_needs_strong_evidence():
    a = seg("gm", 1, 1.5, "igen")
    b = seg("anna", 1.0, 1.5, "igen")
    owned = own({"gm": [(1, 1.5)], "anna": [(1.0, 1.2)]})  # anna 40 %: weak evidence
    t = attribute_bleed(Transcript(segments=[a, b]), owned, BleedSettings())
    out = dedup(t, DedupSettings())
    assert len(out.segments) == 2
    assert all(POSSIBLE_DUPLICATE in s.quality_flags for s in out.segments)


def test_overlap_needs_both_owned():
    t = Transcript(segments=[seg("gm", 1, 4), seg("anna", 2, 5, "más szöveg itt")])
    out = mark_overlaps(t, own({"gm": [(1, 4)]}))
    assert [s.overlap for s in out.segments] == [False, False]


def test_own_interjection_with_other_text_is_not_bleed():
    # gm talks; anna's short "várj" on her own mic is quiet (low ownership) but its text is
    # not gm's: no text evidence, so it is kept
    out = run(
        [seg("gm", 0, 4, "átkelünk a folyón hajnalban"), seg("anna", 1, 1.4, "várj")],
        own({"gm": [(0, 4)]}),
    )
    anna = [s for s in out.segments if s.speaker == "anna"]
    assert anna and LOW_OWNERSHIP in anna[0].quality_flags and not out.suppressed


def test_third_person_heard_on_two_mics_is_flagged_not_removed():
    # someone whose own mic missed it: gm's and anna's mics both "own" the same sentence
    text = "nem megyek veletek a barlangba"
    out = run(
        [seg("gm", 1, 3, text), seg("anna", 1.1, 3.1, text)],
        own({"gm": [(1, 3)], "anna": [(1.1, 3.1)]}),
    )
    assert len(out.segments) == 2 and not out.suppressed
    assert all(POSSIBLE_DUPLICATE in s.quality_flags for s in out.segments)


def own_with_silent(spans, silent, relative=None, seconds=20):
    o = own(spans, seconds)
    o.silent = list(silent)
    if relative is not None:
        rel = np.full(o.masks.shape, -30.0, np.float32)
        for spk, (a, b, db) in relative.items():
            rel[SPK.index(spk), int(a * 50) : int(b * 50)] = db
        o.relative = rel
    return o


def test_silent_mic_third_person_keeps_louder_copy_as_unknown_speaker():
    # bela's mic records nothing; his sentence is heard on gm's and anna's mics
    text = "nem megyek veletek a barlangba"
    owned = own_with_silent(
        {"gm": [(1, 3)], "anna": [(1.1, 3.1)]},
        silent=["bela"],
        relative={"gm": (1, 3, -26.0), "anna": (1.1, 3.1, -21.0)},  # louder on anna's mic
    )
    out = run([seg("gm", 1, 3, text), seg("anna", 1.1, 3.1, text)], owned)
    (kept,) = out.segments
    assert kept.speaker == "anna" and POSSIBLE_WRONG_SPEAKER in kept.quality_flags
    assert kept.probable_speaker == "unknown"  # no automatic reassignment to bela
    (dup,) = out.suppressed
    assert dup.speaker == "gm" and dup.suppression_reason == "duplicate"
    assert dup.duplicate_of == kept.id


def test_without_a_silent_mic_both_copies_stay_flagged():
    text = "nem megyek veletek a barlangba"
    owned = own_with_silent({"gm": [(1, 3)], "anna": [(1.1, 3.1)]}, silent=[])
    out = run([seg("gm", 1, 3, text), seg("anna", 1.1, 3.1, text)], owned)
    assert len(out.segments) == 2 and not out.suppressed


def test_silent_mic_rule_leaves_short_words_alone():
    owned = own_with_silent({"gm": [(1, 1.4)], "anna": [(1.1, 1.5)]}, silent=["bela"])
    out = run([seg("gm", 1, 1.4, "nem"), seg("anna", 1.1, 1.5, "nem")], owned)
    assert len(out.segments) == 2 and not out.suppressed


def test_silent_mic_rule_skips_pairs_where_one_wearer_speaks():
    # anna's copy is at her normal speaking level: she said it; gm's is bleed
    text = "nem megyek veletek a barlangba"
    owned = own_with_silent(
        {"gm": [(1, 3)], "anna": [(1.1, 3.1)]},
        silent=["bela"],
        relative={"gm": (1, 3, -16.0), "anna": (1.1, 3.1, -1.0)},
    )
    out = run([seg("gm", 1, 3, text), seg("anna", 1.1, 3.1, text)], owned)
    assert len(out.segments) == 2 and not out.suppressed
    assert all(POSSIBLE_DUPLICATE in s.quality_flags for s in out.segments)
