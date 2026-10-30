"""Re-transcribe VAD speech that got no words (DECISIONS 2026-10-30).

Whisper in quality mode sometimes skips speech after a decoding window starts: VAD marks
it as speech, but no segment covers it. Each such stretch of at least ``refill_min_s`` is
transcribed again as its own short window. New segments carry ``refilled=True`` (and go
through the normal quality flags); stretches that still get no words are reported as
possible missed speech. Positions are chunk-relative timeline samples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from sessionscribe.core.config import Settings
from sessionscribe.core.pipeline.asr import Segment, SpeechInterval, TranscriptionResult
from sessionscribe.core.pipeline.chunks import midpoint
from sessionscribe.core.timeline import TIMELINE_RATE, WHISPER_RATE, rescale, seconds_to_samples

if TYPE_CHECKING:
    from sessionscribe.core.pipeline.worker import Engine


def covered_spans(segments: Sequence[Segment]) -> list[tuple[int, int]]:
    """Time with recognised words (segment span when there are no word timestamps)."""
    spans = [(w.start, w.end) for s in segments for w in s.words]
    spans += [(s.start, s.end) for s in segments if not s.words]
    return sorted(spans)


def uncovered(
    vad: Sequence[SpeechInterval], segments: Sequence[Segment], min_len: int
) -> list[tuple[int, int]]:
    """Stretches of VAD speech of at least ``min_len`` samples without any word."""
    spans = covered_spans(segments)
    holes: list[tuple[int, int]] = []
    for v in vad:
        cur = v.start
        for a, b in spans:
            if b <= cur or a >= v.end:
                continue
            if a - cur >= min_len:
                holes.append((cur, a))
            cur = max(cur, b)
        if v.end - cur >= min_len:
            holes.append((cur, v.end))
    return holes


@dataclass
class Refill:
    segments: list[Segment] = field(default_factory=list)
    missed: list[tuple[int, int]] = field(default_factory=list)
    holes: int = 0
    transcribe_s: float = 0.0


def _shift(seg: Segment, offset: int) -> Segment:
    return replace(
        seg,
        start=seg.start + offset,
        end=seg.end + offset,
        words=tuple(replace(w, start=w.start + offset, end=w.end + offset) for w in seg.words),
    )


def refill_holes(
    engine: Engine,
    audio: np.ndarray,
    result: TranscriptionResult,
    settings: Settings,
    *,
    source_id: str,
    owned: tuple[int, int],
    cancel: Callable[[], bool] | None = None,
) -> Refill:
    """Transcribe word-less VAD stretches overlapping ``owned`` (chunk-relative samples)."""
    out = Refill()
    if not settings.asr.refill_holes or not settings.vad.enabled:
        return out
    min_len = seconds_to_samples(settings.asr.refill_min_s)
    pad = seconds_to_samples(settings.asr.refill_pad_s)
    length = rescale(len(audio), WHISPER_RATE, TIMELINE_RATE)
    holes = [
        h
        for h in uncovered(result.vad_intervals, result.segments, min_len)
        if h[1] > owned[0] and h[0] < owned[1]
    ]
    out.holes = len(holes)
    next_index = len(result.segments)
    for a, b in holes:
        lo, hi = max(0, a - pad), min(length, b + pad)
        piece = audio[
            rescale(lo, TIMELINE_RATE, WHISPER_RATE) : rescale(hi, TIMELINE_RATE, WHISPER_RATE)
        ]
        found = engine.transcribe(piece, source_id=source_id, cancel=cancel)
        out.transcribe_s += found.transcribe_s + found.vad_s
        got: list[Segment] = []
        for raw in found.segments:
            seg = _shift(raw, lo)
            if seg.words:
                # Only the hole itself: words in the padding were recognised already.
                words = [w for w in seg.words if a <= midpoint(w.start, w.end) <= b]
                if not words:
                    continue
                if len(words) != len(seg.words):
                    seg = replace(
                        seg,
                        start=words[0].start,
                        end=words[-1].end,
                        text="".join(w.text for w in words).strip(),
                        words=tuple(words),
                    )
            elif not a <= midpoint(seg.start, seg.end) <= b:
                continue
            got.append(replace(seg, index=next_index, refilled=True))
            next_index += 1
        out.segments += got
        out.missed += uncovered([SpeechInterval(a, b)], got, min_len)
    return out
