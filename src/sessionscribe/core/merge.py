"""Merge per-track results into one session transcript (SPEC §5).

Input is each participant's kept segments in track-local time; output is sorted on the
session timeline. Bleed attribution, dedup and overlap marking arrive in M3; the
``suppressed`` list already carries chunk-boundary duplicates with their reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from sessionscribe.core.config import QualitySettings
from sessionscribe.core.pipeline.asr import SpeechInterval, Word
from sessionscribe.core.pipeline.chunks import KeptSegment
from sessionscribe.core.quality import POSSIBLE_MISSED_SPEECH, quality_flags

BOUNDARY_DUPLICATE = "chunk_boundary_duplicate"


@dataclass(frozen=True)
class RecordingGap:
    """Lost recording data in track-local samples; ``participant=None`` means every track."""

    start: int
    length: int
    reason: str
    participant: str | None = None


@dataclass(frozen=True)
class TrackInput:
    participant: str
    offset: int  # session timeline samples
    kept: Sequence[KeptSegment]
    vad: Sequence[SpeechInterval] | None  # None: VAD off
    # VAD speech still without words after re-transcription (track-local samples).
    missed: Sequence[tuple[int, int, int]] = ()  # (chunk, start, end)


@dataclass(frozen=True)
class TranscriptSegment:
    id: str
    speaker: str
    chunk_id: str
    start: int  # session timeline samples
    end: int
    track_start: int  # track-local samples, to find the passage in the original file
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float | None
    words: tuple[Word, ...]  # session timeline samples
    quality_flags: tuple[str, ...] = ()
    overlap: bool = False
    ownership: float | None = None
    trimmed_words: int = 0
    suppression_reason: str | None = None
    duplicate_of: str | None = None
    bleed_source: str | None = None  # participant whose voice this probably is (bleed)
    probable_speaker: str | None = None  # possible_wrong_speaker: who probably said it
    reassigned_from: str | None = None  # speaker before a manual reassignment


@dataclass
class Transcript:
    segments: list[TranscriptSegment] = field(default_factory=list)
    suppressed: list[TranscriptSegment] = field(default_factory=list)
    # Empty-text entries flagged possible_missed_speech, for review next to the segments.
    missed: list[TranscriptSegment] = field(default_factory=list)


def chunk_id(participant: str, chunk: int) -> str:
    return f"{participant}-c{chunk:03d}"


def segment_id(participant: str, chunk: int, index: int) -> str:
    """Stable id: participant, chunk and the raw ASR index within that chunk."""
    return f"{chunk_id(participant, chunk)}-{index:04d}"


def merge_tracks(
    tracks: Sequence[TrackInput],
    quality: QualitySettings,
    gaps: Sequence[RecordingGap] = (),
) -> Transcript:
    order: Mapping[str, int] = {t.participant: i for i, t in enumerate(tracks)}
    out = Transcript()
    for track in tracks:
        own_gaps = [
            (g.start, g.start + max(g.length, 1))  # unknown length (0): still a point in time
            for g in gaps
            if g.participant in (None, track.participant)
        ]
        for kept in track.kept:
            seg = kept.segment
            sid = segment_id(track.participant, kept.chunk, seg.index)
            words = tuple(
                Word(w.start + track.offset, w.end + track.offset, w.text, w.probability)
                for w in seg.words
            )
            item = TranscriptSegment(
                id=sid,
                speaker=track.participant,
                chunk_id=chunk_id(track.participant, kept.chunk),
                start=seg.start + track.offset,
                end=seg.end + track.offset,
                track_start=seg.start,
                text=seg.text,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                compression_ratio=seg.compression_ratio,
                temperature=seg.temperature,
                words=words,
                quality_flags=tuple(quality_flags(seg, quality, vad=track.vad, gaps=own_gaps)),
                trimmed_words=kept.trimmed_words,
                suppression_reason=BOUNDARY_DUPLICATE if kept.boundary_duplicate else None,
            )
            (out.suppressed if kept.boundary_duplicate else out.segments).append(item)
        for n, (chunk, start, end) in enumerate(track.missed):
            out.missed.append(
                TranscriptSegment(
                    id=f"{chunk_id(track.participant, chunk)}-m{n:03d}",
                    speaker=track.participant,
                    chunk_id=chunk_id(track.participant, chunk),
                    start=start + track.offset,
                    end=end + track.offset,
                    track_start=start,
                    text="",
                    avg_logprob=0.0,
                    no_speech_prob=0.0,
                    compression_ratio=0.0,
                    temperature=None,
                    words=(),
                    quality_flags=(POSSIBLE_MISSED_SPEECH,),
                )
            )

    def key(s: TranscriptSegment) -> tuple[int, int, str]:
        return s.start, order[s.speaker], s.id

    out.segments.sort(key=key)
    out.suppressed.sort(key=key)
    out.missed.sort(key=key)
    return out


REASSIGNED = "reassigned"


def apply_reassignments(
    transcript: Transcript, reassignments: Mapping[str, str], order: Sequence[str]
) -> tuple[Transcript, list[str]]:
    """Manual speaker changes by segment id; returns ids that were not found among kept
    segments (suppressed or no longer produced). Text and timing are untouched."""
    out: list[TranscriptSegment] = []
    found: set[str] = set()
    for seg in transcript.segments:
        to = reassignments.get(seg.id)
        if to is None or to == seg.speaker:
            if to is not None:
                found.add(seg.id)
            out.append(seg)
            continue
        found.add(seg.id)
        flags = (
            seg.quality_flags
            if REASSIGNED in seg.quality_flags
            else (*seg.quality_flags, REASSIGNED)
        )
        out.append(replace(seg, speaker=to, reassigned_from=seg.speaker, quality_flags=flags))
    rank = {pid: i for i, pid in enumerate(order)}
    out.sort(key=lambda s: (s.start, rank.get(s.speaker, len(rank)), s.id))
    missing = sorted(set(reassignments) - found)
    return replace(transcript, segments=out), missing


def join_same_speaker(
    segments: Sequence[TranscriptSegment], max_gap: int
) -> list[list[TranscriptSegment]]:
    """Group consecutive lines of one speaker less than ``max_gap`` samples apart (txt/md)."""
    groups: list[list[TranscriptSegment]] = []
    for seg in segments:
        if (
            max_gap > 0
            and groups
            and groups[-1][-1].speaker == seg.speaker
            and seg.start - groups[-1][-1].end <= max_gap
        ):
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return groups
