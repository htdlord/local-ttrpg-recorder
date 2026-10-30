"""Bleed attribution, dedup and overlap marking after ASR (SPEC §5.4-5.5).

Everything here works on the merged transcript (session time) and the ownership masks;
nothing re-runs ASR. Losing real speech is worse than a duplicate, so a segment is
suppressed as bleed only with four pieces of evidence: its own track hardly owns its
frames, another track does, that track has words there itself, and those words read
like this segment's text. Without the source's words it is the only copy (the triage
``w`` case): kept and flagged ``possible_wrong_speaker``. Without matching text (e.g. a
short own interjection while the other person talks) it is kept and flagged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from sessionscribe.core.bench.text import normalize
from sessionscribe.core.config import BleedSettings, DedupSettings
from sessionscribe.core.merge import Transcript, TranscriptSegment
from sessionscribe.core.pipeline.ownership import Ownership
from sessionscribe.core.quality import (
    LOW_OWNERSHIP,
    POSSIBLE_DUPLICATE,
    POSSIBLE_WRONG_SPEAKER,
)
from sessionscribe.core.timeline import TIMELINE_RATE

SOURCE_OWNS = 0.5  # another track owns at least this share of the segment's frames
SOURCE_WORDS = 0.5  # ... and its kept words cover at least this share of the segment
DEDUP_LOW, DEDUP_HIGH = 0.5, 0.5  # ownership evidence for removing a duplicate
SHORT_LOW, SHORT_HIGH = 0.2, 0.8  # stronger evidence for utterances under min_words
OVERLAP_OWNED = 0.5
UNKNOWN_SPEAKER = "unknown"  # probable_speaker when loudness cannot tell who it was
THIRD_PARTY_REL_DB = -20.0  # both copies this far below their track's speech level (clip01)
OVERLAP_MIN_S = 0.3
OVERLAP_TEXT_MAX = 60.0  # texts at least this similar are copies, not an overlap
OWNED_WORD = 0.5  # a word counts as own speech when its track owns this share of it
OWNED_SPEECH_S = 0.5  # this much owned speech inside a segment keeps it (mixed: own + bleed)


def segment_ownership(seg: TranscriptSegment, own: Ownership) -> float:
    """Share of the segment the speaker owns; words weighted by duration."""
    if seg.words:
        total = sum(max(1, w.end - w.start) for w in seg.words)
        return (
            sum(
                own.owned(seg.speaker, w.start, max(w.end, w.start + 1)) * max(1, w.end - w.start)
                for w in seg.words
            )
            / total
        )
    return own.owned(seg.speaker, seg.start, max(seg.end, seg.start + 1))


def owned_speech_samples(seg: TranscriptSegment, own: Ownership) -> int:
    """Duration of the segment's words its own track owns (mixed segments keep these)."""
    return sum(
        w.end - w.start
        for w in seg.words
        if w.end > w.start and own.owned(seg.speaker, w.start, w.end) >= OWNED_WORD
    )


def _add_flag(seg: TranscriptSegment, flag: str) -> TranscriptSegment:
    if flag in seg.quality_flags:
        return seg
    return replace(seg, quality_flags=(*seg.quality_flags, flag))


def source_text(seg: TranscriptSegment, segments: Sequence[TranscriptSegment], source: str) -> str:
    """The source's words around the segment (1 s margin), for the text evidence."""
    lo, hi = seg.start - TIMELINE_RATE, seg.end + TIMELINE_RATE
    parts: list[str] = []
    for s in segments:
        if s.speaker != source or s.end <= lo or s.start >= hi:
            continue
        if s.words:
            parts += [w.text for w in s.words if w.end > lo and w.start < hi]
        else:
            parts.append(s.text)
    return " ".join(parts)


def text_matches(seg: TranscriptSegment, source_words: str, threshold: float) -> bool:
    from rapidfuzz import fuzz

    a, b = normalize(seg.text), normalize(source_words)
    return bool(a and b) and fuzz.partial_ratio(a, b) >= threshold


def _covered(start: int, end: int, segments: Sequence[TranscriptSegment]) -> float:
    """Share of ``[start, end)`` covered by the words (or spans) of ``segments``."""
    spans = sorted((w.start, w.end) for s in segments for w in (s.words or ())) + [
        (s.start, s.end) for s in segments if not s.words
    ]
    covered, cur = 0, start
    for a, b in sorted(spans):
        a, b = max(a, cur), min(b, end)
        if b > a:
            covered += b - a
            cur = b
    return covered / max(1, end - start)


def attribute_bleed(
    transcript: Transcript, own: Ownership, settings: BleedSettings, min_words: int = 3
) -> Transcript:
    """Set ``ownership``; suppress evidenced bleed; flag low ownership / wrong speaker.

    Utterances shorter than ``min_words`` need ownership under ``SHORT_LOW`` (two people
    can say the same short word at once).
    """
    segs = [replace(s, ownership=round(segment_ownership(s, own), 3)) for s in transcript.segments]
    candidates: dict[int, str] = {}
    for n, seg in enumerate(segs):
        short = len(normalize(seg.text).split()) < min_words
        limit = min(settings.suppress_below, SHORT_LOW) if short else settings.suppress_below
        if seg.ownership is None or seg.ownership >= limit:
            continue
        if owned_speech_samples(seg, own) >= OWNED_SPEECH_S * TIMELINE_RATE:
            continue  # own words inside (mixed): never suppressed, flagged low_ownership below
        others = [
            (own.owned(k, seg.start, max(seg.end, seg.start + 1)), k)
            for k in own.speakers
            if k != seg.speaker
        ]
        if others:
            share, likely = max(others)
            if share >= SOURCE_OWNS:
                candidates[n] = likely
    out: list[TranscriptSegment] = []
    suppressed = list(transcript.suppressed)
    for n, seg in enumerate(segs):
        source = candidates.get(n)
        if source is not None:
            source_segments = [
                s for m, s in enumerate(segs) if s.speaker == source and m not in candidates
            ]
            if _covered(seg.start, seg.end, source_segments) < SOURCE_WORDS:
                seg = replace(_add_flag(seg, POSSIBLE_WRONG_SPEAKER), probable_speaker=source)
            elif text_matches(
                seg, source_text(seg, source_segments, source), settings.text_similarity
            ):
                suppressed.append(replace(seg, suppression_reason="bleed", bleed_source=source))
                continue
        if seg.ownership is not None and seg.ownership < settings.low_ownership_below:
            seg = _add_flag(seg, LOW_OWNERSHIP)
        out.append(seg)
    suppressed.sort(key=lambda s: (s.start, s.id))
    return replace(transcript, segments=out, suppressed=suppressed)


def _gap(a: TranscriptSegment, b: TranscriptSegment) -> int:
    return max(0, max(a.start, b.start) - min(a.end, b.end))


def dedup(
    transcript: Transcript, settings: DedupSettings, own: Ownership | None = None
) -> Transcript:
    """Remove a cross-track duplicate only with ownership evidence; else keep and flag.

    A weaker copy that contains owned speech of its own (mixed) is never removed.

    Silent mics (owner decision 2026-10-30): when the session has a mic that records
    (almost) nothing, a longer identical text on two other tracks that the evidence rules
    cannot resolve is probably that participant, heard on both. One copy is kept, on the
    louder track (relative to its own speech level), flagged ``possible_wrong_speaker``
    with ``probable_speaker = "unknown"`` - loudness cannot tell who it was, so there is
    no automatic reassignment (manual in the GUI); the other copy is a duplicate.
    """
    if not settings.enabled:
        return transcript
    from rapidfuzz import fuzz

    segs = list(transcript.segments)
    texts = [normalize(s.text) for s in segs]
    max_gap = round(settings.max_time_gap_s * TIMELINE_RATE)
    remove: dict[int, int] = {}
    flag: set[int] = set()
    third_party: list[tuple[int, int]] = []
    silent = set(own.silent) if own is not None else set()
    for i, a in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if b.start > a.end + max_gap:
                break  # sorted by start: every later segment is further away
            if a.speaker == b.speaker or _gap(a, b) > max_gap or not texts[i] or not texts[j]:
                continue
            if fuzz.ratio(texts[i], texts[j]) < settings.similarity:
                continue
            oa, ob = a.ownership, b.ownership
            if oa is None or ob is None:
                flag.update((i, j))
                continue
            low, high = (i, j) if oa <= ob else (j, i)
            lo, hi = min(oa, ob), max(oa, ob)
            short = min(len(texts[i].split()), len(texts[j].split())) < settings.min_words
            need_low, need_high = (SHORT_LOW, SHORT_HIGH) if short else (DEDUP_LOW, DEDUP_HIGH)
            if silent and not short and a.speaker not in silent and b.speaker not in silent:
                unresolved = (hi >= OVERLAP_OWNED and lo >= OVERLAP_OWNED) or not (
                    lo < need_low and hi >= need_high
                )
                if unresolved:
                    third_party.append((i, j))
                    continue
            if hi >= OVERLAP_OWNED and lo >= OVERLAP_OWNED:
                # Both own their frames. A short word: two people said it. A longer
                # identical sentence at once is unlikely: a third person whose own mic
                # missed it, heard on both - keep both, flag for review.
                if not short:
                    flag.update((i, j))
                continue
            mixed = own is not None and (
                owned_speech_samples(segs[low], own) >= OWNED_SPEECH_S * TIMELINE_RATE
            )
            if lo < need_low and hi >= need_high and not mixed and low not in remove:
                remove[low] = high
            else:
                flag.update((i, j))
    wrong: set[int] = set()
    for i, j in third_party:
        if i in remove or j in remove:
            continue
        assert own is not None
        level_i = own.relative_level(segs[i].speaker, segs[i].start, segs[i].end)
        level_j = own.relative_level(segs[j].speaker, segs[j].start, segs[j].end)
        if max(level_i, level_j) >= THIRD_PARTY_REL_DB:
            # One copy is at its own track's speaking level: that wearer probably said
            # it, so this is not an absent third person; keep both, flagged.
            flag.update((i, j))
            continue
        keep, drop = (i, j) if level_i >= level_j else (j, i)
        remove[drop] = keep
        wrong.add(keep)
    out, suppressed = [], list(transcript.suppressed)
    for n, seg in enumerate(segs):
        if n in wrong:
            seg = replace(_add_flag(seg, POSSIBLE_WRONG_SPEAKER), probable_speaker=UNKNOWN_SPEAKER)
        if n in remove:
            suppressed.append(
                replace(seg, suppression_reason="duplicate", duplicate_of=segs[remove[n]].id)
            )
        else:
            out.append(_add_flag(seg, POSSIBLE_DUPLICATE) if n in flag else seg)
    suppressed.sort(key=lambda s: (s.start, s.id))
    return replace(transcript, segments=out, suppressed=suppressed)


def mark_overlaps(transcript: Transcript, own: Ownership) -> Transcript:
    """Two people speaking at once: different speakers, overlapping in time by at least
    ``OVERLAP_MIN_S``, both owning the shared frames, and saying different things.

    Two copies of the same words are bleed or a duplicate, not an overlap (on clip01 the
    ownership test alone marked most segments of two strongly coupled mics).
    """
    from rapidfuzz import fuzz

    segs = list(transcript.segments)
    texts = [normalize(s.text) for s in segs]
    min_len = round(OVERLAP_MIN_S * TIMELINE_RATE)
    marked: set[int] = set()
    for i, a in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if b.start >= a.end:
                break
            if a.speaker == b.speaker:
                continue
            lo, hi = max(a.start, b.start), min(a.end, b.end)
            if hi - lo < min_len:
                continue
            if texts[i] and texts[j] and fuzz.ratio(texts[i], texts[j]) >= OVERLAP_TEXT_MAX:
                continue
            if (
                own.owned(a.speaker, lo, hi) >= OVERLAP_OWNED
                and own.owned(b.speaker, lo, hi) >= OVERLAP_OWNED
            ):
                marked.update((i, j))
    return replace(
        transcript,
        segments=[replace(s, overlap=True) if n in marked else s for n, s in enumerate(segs)],
    )


def apply_bleed_handling(
    transcript: Transcript, own: Ownership, bleed: BleedSettings, dedup_settings: DedupSettings
) -> Transcript:
    out = attribute_bleed(transcript, own, bleed, dedup_settings.min_words)
    out = dedup(out, dedup_settings, own)
    if dedup_settings.mark_overlaps:
        out = mark_overlaps(out, own)
    return out
