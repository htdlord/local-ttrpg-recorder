"""Bench metrics, scoring of saved outputs, mode recommendation and reports (synthetic)."""

import json

import pytest

from sessionscribe.core.bench.metrics import (
    HypSegment,
    HypWord,
    score_track,
    segments_from_json,
    summarize,
    words_of,
)
from sessionscribe.core.bench.reference import Reference, Utterance
from sessionscribe.core.bench.resources import peak_rss_mb
from sessionscribe.core.bench.runner import (
    MATRICES,
    Cell,
    derived,
    recommend_mode,
    results_markdown,
)


def ref(*utts):
    return Reference(clip="c", language="hu", utterances=list(utts))


def utt(uid, speaker, start, end, text, tags=()):
    return Utterance(id=uid, speaker=speaker, start=start, end=end, text=text, tags=list(tags))


def seg(start, end, *words):
    """words: (start, end, text)."""
    ws = tuple(HypWord(a, b, t) for a, b, t in words)
    return HypSegment(start, end, " ".join(w.text for w in ws), ws)


REF = ref(
    utt("r001", "gm", 1.0, 3.0, "Jó ember, ő bízik."),
    utt("r002", "gm", 5.0, 6.0, "Igen"),
    utt("r003", "gm", 8.0, 9.0, "", tags=("laugh",)),  # unscored
    utt("r004", "seat_1", 1.0, 3.0, "Árvíztűrő tükörfúrógép"),
)


def test_perfect_track():
    hyp = [
        seg(1.0, 3.0, (1.0, 1.5, "jó"), (1.5, 2.0, "ember"), (2.0, 2.4, "ő"), (2.4, 3.0, "bízik")),
        seg(5.0, 6.0, (5.1, 5.9, "igen.")),
    ]
    s = score_track("gm", REF, hyp)
    assert s.ref_utterances == 2 and s.ref_words == 5
    assert s.word_errors == 0 and s.char_errors == 0 and s.recalled == 2
    assert s.hyp_words_outside == 0 and not s.silent
    assert s.start_errors_s == [0.0, pytest.approx(0.1)]


def test_errors_bleed_outside_and_missed_utterance():
    hyp = [
        seg(1.0, 3.0, (1.0, 1.5, "jó"), (1.5, 2.0, "ember"), (2.0, 3.0, "bízott")),
        seg(12.0, 13.0, (12.0, 12.5, "bleed"), (12.5, 13.0, "szöveg")),  # outside windows
    ]
    s = score_track("gm", REF, hyp)
    # ref "jó ember ő bízik igen" vs hyp "jó ember bízott": 1 sub + 2 del
    assert (s.substitutions, s.deletions, s.insertions) == (1, 2, 0)
    assert s.recalled == 1  # r001 has 2/4 hits (= 50 %), r002 missed
    assert s.hyp_words_outside == 2 and s.hyp_seconds_outside == pytest.approx(1.0)


def test_window_padding():
    hyp = [seg(4.75, 4.9, (4.75, 4.9, "igen"))]  # mid 4.825 is within 0.3 s of r002
    assert score_track("gm", REF, hyp).recalled == 1


def test_silent_track():
    hyp = [seg(1.0, 2.0, (1.0, 1.5, "valami"), (1.5, 2.0, "hallucináció"))]
    s = score_track("seat_2", REF, hyp)
    # Not part of WER (no reference window): reported only as the silent-track metric.
    assert s.silent and s.hyp_words_outside == 2 and s.insertions == 0
    summary = summarize([score_track("gm", REF, []), s])
    assert summary.silent_tracks == 1 and summary.silent_track_words == 2
    assert summary.silent_track_segments == 1 and summary.hyp_words_outside == 0
    assert summary.wer == 1.0 and summary.speech_recall == 0.0  # gm said 5 words, none found


def test_interpolated_word_times():
    words, interpolated = words_of([HypSegment(0.0, 2.0, "egy kettő", ())])
    assert interpolated and [(w.start, w.end) for w in words] == [(0.0, 1.0), (1.0, 2.0)]
    s = score_track("seat_1", REF, [HypSegment(1.0, 3.0, "Árvíztűrő tükörfúrógép", ())])
    assert s.word_errors == 0 and s.interpolated
    assert summarize([s]).interpolated_word_times


def test_summary_micro_average():
    a = score_track("gm", REF, [seg(1.0, 3.0, (1.0, 3.0, "jó"))])
    b = score_track("seat_1", REF, [seg(1.0, 3.0, (1.0, 2.0, "árvíztűrő"), (2.0, 3.0, "x"))])
    summary = summarize([a, b])
    assert summary.ref_words == 7
    assert summary.wer == round((a.word_errors + b.word_errors) / 7, 4)


def test_segments_from_our_json():
    data = {
        "segments": [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "ő",
                "words": [{"start": 1.0, "end": 2.0, "word": " ő", "prob": 0.9}],
            }
        ]
    }
    (s,) = segments_from_json(json.loads(json.dumps(data, ensure_ascii=False)))
    assert s.words[0].text == " ő"


def test_matrix_m1_shape():
    cells = MATRICES["m1"]()
    assert len(cells) == 32
    assert len({c.name for c in cells}) == 32
    quality = [c for c in cells if c.mode == "quality"]
    assert len(quality) == 8 and all(c.batch_size is None for c in quality)
    assert {c.batch_size for c in cells if c.mode == "fast"} == {4, 8, 16}
    assert Cell("large-v3", "float16", "fast", 8, False).name == "large-v3_float16_fast_b8_nowts"


def entry(mode, wer, batch=None, transcribe_s=60.0, speech_s=600.0, ok=True):
    cell = Cell("large-v3", "float16", mode, batch, True)
    run = {"ok": ok, "transcribe_s": transcribe_s, "audio_s": 3000.0, "speech_s": speech_s}
    e = {"name": cell.name, "cell": cell.__dict__, "run": run, "derived": derived(run)}
    if ok:
        e["metrics"] = {"summary": {"wer": wer}}
    return e


def test_derived_projection():
    d = derived({"ok": True, "transcribe_s": 60.0, "audio_s": 3000.0, "speech_s": 600.0})
    assert d == {"rtf": 0.02, "speech_rtf": 0.1, "projected_min": 60.0}


def test_recommendation_keeps_quality_within_tolerance():
    cells = [entry("quality", 0.205), entry("fast", 0.20, 8), entry("fast", 0.22, 16)]
    rec = recommend_mode(cells, 109.5)
    assert rec["decision"] == "quality" and rec["provisional"] is True
    assert rec["best_fast_cell"].endswith("b8_wts")


def test_recommendation_revisits_when_worse_or_slow():
    assert recommend_mode([entry("quality", 0.22), entry("fast", 0.20, 8)], 1)["decision"] == (
        "revisit"
    )
    slow = [entry("quality", 0.20, transcribe_s=70.0), entry("fast", 0.20, 8)]
    assert recommend_mode(slow, 1)["decision"] == "revisit"
    assert recommend_mode([entry("fast", 0.2, 8)], 1)["decision"] is None


def test_results_markdown_numbers_only():
    cells = [entry("quality", 0.2), entry("fast", 0.0, 8, ok=False)]
    cells[0]["metrics"]["summary"] |= {
        "cer": 0.1,
        "speech_recall": 0.9,
        "hyp_words_outside": 3,
        "silent_track_words": 1,
        "silent_track_segments": 1,
        "timestamp_start_median_ms": 120.0,
        "timestamp_end_median_ms": 80.0,
    }
    cells[0]["run"] |= {"load_s": 2.0, "peak_vram_mb": 4000, "vram_method": "device-delta"}
    cells[1]["run"]["error"] = "out of memory"
    md = results_markdown(
        {
            "run": "x",
            "clips": [{"id": "clip01", "seconds": 109.5, "tracks": 7, "status": "draft"}],
            "cells": cells,
            "recommendation": recommend_mode(cells, 109.5),
        }
    )
    assert "UNVERIFIED" in md and "| 20.0 | 10.0 | 90.0 | 3 | 1 / 1 |" in md
    assert "4000*" in md and "failed: out of memory" in md


def test_peak_rss():
    value = peak_rss_mb()
    assert value is None or value > 1
