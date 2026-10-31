"""Quality flags (SPEC §5.6): markers worth a human look, never deletions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ttrpg_recorder.core.config import QualitySettings
from ttrpg_recorder.core.pipeline.asr import Segment, SpeechInterval
from ttrpg_recorder.core.timeline import TIMELINE_RATE

LOW_LOGPROB = "low_logprob"
HIGH_NO_SPEECH = "high_no_speech"
HIGH_COMPRESSION = "high_compression"
REPETITION = "repetition"
TOO_LONG = "too_long_for_duration"
OUTSIDE_VAD = "outside_vad"
OVERLAPS_GAP = "overlaps_gap"
REFILLED = "refilled"  # text from re-transcribing word-less VAD speech; check it
POSSIBLE_MISSED_SPEECH = "possible_missed_speech"  # VAD speech still without words
# Bleed attribution and dedup (M3).
POSSIBLE_DUPLICATE = "possible_duplicate"
LOW_OWNERSHIP = "low_ownership"
POSSIBLE_WRONG_SPEAKER = "possible_wrong_speaker"  # another mic owns it, only copy here

MIN_TEXT_BASIS_S = 1.0  # short words have tight timestamps; do not flag them as too long


def overlap(a: int, b: int, intervals: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, min(b, e) - max(a, s)) for s, e in intervals)


def has_repetition(text: str, n: int, times: int) -> bool:
    words = text.lower().split()
    if len(words) < n * times:
        return False
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()) >= times


def quality_flags(
    seg: Segment,
    settings: QualitySettings,
    *,
    vad: Sequence[SpeechInterval] | None,
    gaps: Sequence[tuple[int, int]] = (),
) -> list[str]:
    """Flags for one segment in track-local time; ``vad=None`` means VAD was off."""
    flags: list[str] = [REFILLED] if seg.refilled else []  # provenance: always shown
    if not settings.enabled:
        return flags
    if seg.avg_logprob < settings.low_logprob:
        flags.append(LOW_LOGPROB)
    if (
        seg.no_speech_prob > settings.high_no_speech
        and seg.avg_logprob < settings.high_no_speech_logprob
    ):
        flags.append(HIGH_NO_SPEECH)
    if seg.compression_ratio > settings.high_compression:
        flags.append(HIGH_COMPRESSION)
    if has_repetition(seg.text, settings.repeated_ngram, settings.repeated_ngram_times):
        flags.append(REPETITION)
    length = max(0, seg.end - seg.start)
    voiced = length
    if vad is not None:
        voiced = overlap(seg.start, seg.end, [(v.start, v.end) for v in vad])
        if length and voiced < settings.min_vad_fraction * length:
            flags.append(OUTSIDE_VAD)
    chars = len(seg.text.replace(" ", ""))
    basis = max(voiced, MIN_TEXT_BASIS_S * TIMELINE_RATE)
    if chars > settings.max_chars_per_s * basis / TIMELINE_RATE:
        flags.append(TOO_LONG)
    if any(overlap(seg.start, max(seg.end, seg.start + 1), [g]) for g in gaps):
        flags.append(OVERLAPS_GAP)
    return flags
