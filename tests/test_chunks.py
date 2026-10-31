import itertools
from dataclasses import replace

import numpy as np
import pytest

from ttrpg_recorder.core.pipeline.asr import Segment, Word
from ttrpg_recorder.core.pipeline.chunks import (
    Chunk,
    keep_owned,
    plan_chunks,
    quietest_point,
    resolve_chunks,
)

R = 48_000
FRAME = 960


def seg(index, words, text=None):
    ws = tuple(Word(int(a * R), int(b * R), f" {t}", 0.9) for a, b, t in words)
    return Segment(
        index=index,
        start=ws[0].start,
        end=ws[-1].end,
        text=text or "".join(w.text for w in ws).strip(),
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        compression_ratio=1.2,
        temperature=0.0,
        words=ws,
    )


def loud_envelope(seconds, quiet=()):
    env = np.full(int(seconds * R / FRAME), -20.0, np.float32)
    for a, b in quiet:
        env[int(a * R / FRAME) : int(b * R / FRAME)] = -80.0
    return env


def test_short_track_is_one_chunk():
    chunks = plan_chunks(100 * R, owned_s=600, context_s=5)
    assert chunks == [Chunk(0, 0, 100 * R, 0, 100 * R)]


def test_boundaries_move_to_silence_deterministically():
    env = loud_envelope(70, quiet=[(27.0, 28.0), (52.0, 53.0)])
    a = plan_chunks(70 * R, env, owned_s=25, context_s=2, search_s=5)
    b = plan_chunks(70 * R, env.copy(), owned_s=25, context_s=2, search_s=5)
    assert a == b
    assert [c.owned_start for c in a] == [0, a[1].owned_start, a[2].owned_start]
    assert 27.0 * R <= a[1].owned_start <= 28.0 * R
    assert 50.0 * R <= a[2].owned_start <= 53.0 * R
    assert a[1].ctx_start == a[1].owned_start - 2 * R and a[-1].owned_end == 70 * R
    assert all(x.owned_end == y.owned_start for x, y in itertools.pairwise(a))


def test_tiny_last_chunk_is_merged():
    chunks = plan_chunks(int(105 * R), owned_s=50, search_s=0)
    assert [c.owned for c in chunks] == [(0, 50 * R), (50 * R, 105 * R)]


def test_quietest_point_prefers_earliest_tie():
    env = loud_envelope(40, quiet=[(10, 11), (20, 21)])
    point = quietest_point(env, FRAME, 15 * R, 10 * R, 0, 40 * R)
    assert 10 * R <= point < 11 * R


def test_midpoint_rule():
    chunk = Chunk(0, 0, 10 * R, 0, 12 * R)
    segs = [seg(0, [(8.0, 9.0, "a")]), seg(1, [(9.5, 10.8, "b")]), seg(2, [(10.2, 11, "c")])]
    assert [s.index for s in keep_owned(chunk, segs, last=False)] == [0]
    assert [s.index for s in keep_owned(chunk, segs[1:], last=False)] == []  # midpoint 10.15


def test_boundary_duplicate_words_are_trimmed_unique_kept():
    c0, c1 = Chunk(0, 0, 10 * R, 0, 12 * R), Chunk(1, 10 * R, 20 * R, 8 * R, 20 * R)
    # chunk 0 sees a sentence running over the boundary; chunk 1 heard the last words too
    left = seg(
        0, [(8.0, 8.5, "egy"), (8.6, 9.6, "kettő"), (9.8, 10.3, "három"), (10.4, 10.9, "négy")]
    )
    right = seg(0, [(10.4, 10.9, "négy"), (11.0, 11.5, "öt")])
    kept = resolve_chunks([c0, c1], [[left], [right]])
    assert [k.segment.text for k in kept] == ["egy kettő három", "négy öt"]
    assert kept[0].trimmed_words == 1 and kept[0].segment.end == int(10.3 * R)
    assert kept[0].segment.index == 0 and kept[0].chunk == 0


def test_boundary_word_kept_when_neighbour_missed_it():
    c0, c1 = Chunk(0, 0, 10 * R, 0, 12 * R), Chunk(1, 10 * R, 20 * R, 8 * R, 20 * R)
    left = seg(0, [(9.0, 9.6, "ő"), (9.7, 10.4, "ű"), (10.5, 10.9, "á")])
    right = seg(0, [(14.0, 14.5, "más")])
    kept = resolve_chunks([c0, c1], [[left], [right]])
    assert kept[0].segment == left and kept[0].trimmed_words == 0


def test_fully_duplicated_segment_is_reported_not_lost():
    c0, c1 = Chunk(0, 0, 10 * R, 0, 12 * R), Chunk(1, 10 * R, 20 * R, 8 * R, 20 * R)
    # segment midpoint 9.5 s is owned by chunk 0, but its only word lies past the boundary
    left = replace(seg(0, [(10.1, 10.9, "y")]), start=9 * R, end=10 * R)
    right = seg(0, [(10.1, 10.9, "y")])
    kept = resolve_chunks([c0, c1], [[left], [right]])
    assert [(k.chunk, k.boundary_duplicate, k.trimmed_words) for k in kept] == [
        (0, True, 1),
        (1, False, 0),
    ]
    assert kept[0].segment == left  # reported unchanged, as suppressed


def test_results_must_match_chunks():
    with pytest.raises(ValueError):
        resolve_chunks([Chunk(0, 0, R, 0, R)], [])
