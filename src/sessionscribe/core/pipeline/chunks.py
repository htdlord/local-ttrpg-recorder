"""Chunk planning and boundary resolution in track-local time (SPEC §5.1).

A chunk is an *owned* range plus *context* on both sides. ASR sees the context range;
only segments whose midpoint lies in the owned range are kept. Words of a kept segment
that stick out past the owned range are removed only when the neighbouring chunk's own
words cover the same time; otherwise they stay (losing speech is worse than a duplicate).
All positions are track-local samples at the 48 kHz timeline rate.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from sessionscribe.core.pipeline.asr import Segment, Word
from sessionscribe.core.timeline import TIMELINE_RATE

SMOOTH_S = 0.5
MIN_LAST_FRACTION = 0.25  # a shorter last chunk is merged into the previous one
COVER_FRACTION = 0.5


@dataclass(frozen=True)
class Chunk:
    index: int
    owned_start: int
    owned_end: int
    ctx_start: int
    ctx_end: int

    @property
    def owned(self) -> tuple[int, int]:
        return self.owned_start, self.owned_end

    def owns(self, position: int, *, last: bool) -> bool:
        if last:
            return self.owned_start <= position <= self.owned_end
        return self.owned_start <= position < self.owned_end


def quietest_point(
    envelope: np.ndarray | None, frame: int, centre: int, search: int, lo: int, hi: int
) -> int:
    """Start of the quietest smoothed stretch within ``centre ± search`` (earliest on ties)."""
    if envelope is None or envelope.size == 0 or search <= 0:
        return centre
    first = max(lo, centre - search) // frame
    last = min(hi, centre + search) // frame
    if last <= first:
        return centre
    width = max(1, round(SMOOTH_S * TIMELINE_RATE / frame))
    power = np.power(10.0, envelope.astype(np.float64) / 10.0)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(first, last + 1)
    lo_idx = np.clip(starts - width // 2, 0, power.size)
    hi_idx = np.clip(starts + (width - width // 2), 0, power.size)
    mean = (csum[hi_idx] - csum[lo_idx]) / np.maximum(hi_idx - lo_idx, 1)
    # Round so float noise cannot break ties differently on another machine.
    best = int(np.argmin(np.round(10.0 * np.log10(np.maximum(mean, 1e-30)), 3)))
    return int(starts[best]) * frame


def plan_chunks(
    length: int,
    envelope: np.ndarray | None = None,
    *,
    frame: int = 960,
    owned_s: float = 600.0,
    context_s: float = 5.0,
    search_s: float = 15.0,
) -> list[Chunk]:
    """Deterministic chunk plan for a track of ``length`` samples."""
    owned = round(owned_s * TIMELINE_RATE)
    context = round(context_s * TIMELINE_RATE)
    search = min(round(search_s * TIMELINE_RATE), owned // 4)
    if owned <= 0:
        raise ValueError("owned range must be positive")
    count = max(1, -(-length // owned))
    if count > 1 and length - (count - 1) * owned < MIN_LAST_FRACTION * owned:
        count -= 1
    bounds = [0]
    for k in range(1, count):
        nominal = k * owned
        point = quietest_point(envelope, frame, nominal, search, bounds[-1] + 1, length - 1)
        bounds.append(point)
    bounds.append(length)
    return [
        Chunk(i, a, b, max(0, a - context), min(length, b + context))
        for i, (a, b) in enumerate(itertools.pairwise(bounds))
    ]


@dataclass(frozen=True)
class KeptSegment:
    chunk: int
    segment: Segment  # after boundary trimming; ``segment.index`` is the raw ASR index
    trimmed_words: int = 0
    # Every word is a duplicate of the neighbouring chunk's: reported as suppressed.
    boundary_duplicate: bool = False


def midpoint(start: int, end: int) -> int:
    return (start + end) // 2


def keep_owned(chunk: Chunk, segments: Sequence[Segment], *, last: bool) -> list[Segment]:
    return [s for s in segments if chunk.owns(midpoint(s.start, s.end), last=last)]


def _covered(word: Word, cover: Sequence[Word]) -> bool:
    length = word.end - word.start
    if length <= 0:
        return any(c.start <= word.start <= c.end for c in cover)
    overlap = sum(max(0, min(word.end, c.end) - max(word.start, c.start)) for c in cover)
    return overlap >= COVER_FRACTION * length


def _rebuild(seg: Segment, words: list[Word]) -> Segment:
    """Segment from its remaining words; only a trimmed side moves its boundary."""
    return replace(
        seg,
        start=seg.start if words[0] is seg.words[0] else words[0].start,
        end=seg.end if words[-1] is seg.words[-1] else words[-1].end,
        text="".join(w.text for w in words).strip(),
        words=tuple(words),
    )


def resolve_chunks(
    chunks: Sequence[Chunk], results: Sequence[Sequence[Segment]]
) -> list[KeptSegment]:
    """Midpoint rule per chunk, then trim words duplicated across each boundary."""
    if len(chunks) != len(results):
        raise ValueError("one result per chunk expected")
    last = len(chunks) - 1
    kept = [
        keep_owned(c, r, last=i == last)
        for i, (c, r) in enumerate(zip(chunks, results, strict=True))
    ]

    def inside_words(i: int) -> list[Word]:
        c = chunks[i]
        return [
            w for s in kept[i] for w in s.words if c.owns(midpoint(w.start, w.end), last=i == last)
        ]

    inside = [inside_words(i) for i in range(len(chunks))]
    out: list[KeptSegment] = []
    for i, chunk in enumerate(chunks):
        neighbours = [inside[j] for j in (i - 1, i + 1) if 0 <= j <= last]
        cover = [w for ws in neighbours for w in ws]
        for seg in kept[i]:
            if not seg.words or not cover:
                out.append(KeptSegment(chunk.index, seg))
                continue
            words = [
                w
                for w in seg.words
                if chunk.owns(midpoint(w.start, w.end), last=i == last) or not _covered(w, cover)
            ]
            dropped = len(seg.words) - len(words)
            if not dropped:
                out.append(KeptSegment(chunk.index, seg))
            elif words:
                out.append(KeptSegment(chunk.index, _rebuild(seg, words), dropped))
            else:
                out.append(KeptSegment(chunk.index, seg, dropped, boundary_duplicate=True))
    return out
