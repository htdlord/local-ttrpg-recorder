"""M1 metrics per track against a reference (SPEC §11.2).

Recognition is scored only inside the speaker's own reference windows, so bleed on a
track does not count against WER here (that is M3's job). Everything outside the
windows is reported separately, and tracks without any reference speech get their own
"silent track" figures: whatever ASR produces there is hallucination or bleed.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from sessionscribe.core.bench.reference import Reference, Utterance
from sessionscribe.core.bench.text import normalize, require_jiwer

WINDOW_PAD_S = 0.3
RECALL_MIN_HIT_RATIO = 0.5


@dataclass(frozen=True)
class HypWord:
    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class HypSegment:
    start: float
    end: float
    text: str
    words: tuple[HypWord, ...] = ()


def words_of(segments: Iterable[HypSegment]) -> tuple[list[HypWord], bool]:
    """Hypothesis words; without word timestamps they are spread evenly over the segment.

    Returns the words and whether any timing had to be interpolated.
    """
    out: list[HypWord] = []
    interpolated = False
    for seg in segments:
        if seg.words:
            out.extend(seg.words)
            continue
        tokens = seg.text.split()
        if not tokens:
            continue
        interpolated = True
        step = max(seg.end - seg.start, 1e-3) / len(tokens)
        out.extend(
            HypWord(seg.start + i * step, seg.start + (i + 1) * step, tok)
            for i, tok in enumerate(tokens)
        )
    return sorted(out, key=lambda w: w.start), interpolated


@dataclass
class TrackScore:
    speaker: str
    ref_utterances: int = 0
    ref_words: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_chars: int = 0
    char_errors: int = 0
    recalled: int = 0
    hyp_words_in_windows: int = 0
    hyp_words_outside: int = 0
    hyp_seconds_outside: float = 0.0
    hyp_segments: int = 0
    start_errors_s: list[float] = field(default_factory=list)
    end_errors_s: list[float] = field(default_factory=list)
    interpolated: bool = False

    @property
    def silent(self) -> bool:
        return self.ref_utterances == 0

    @property
    def word_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def _windows(utts: list[Utterance]) -> list[tuple[float, float, Utterance]]:
    return [(u.start - WINDOW_PAD_S, u.end + WINDOW_PAD_S, u) for u in utts]


def _owner(word: HypWord, windows: list[tuple[float, float, Utterance]]) -> Utterance | None:
    """The utterance whose (padded) window contains the word's midpoint; nearest centre wins."""
    best: Utterance | None = None
    best_d = float("inf")
    for lo, hi, u in windows:
        if lo <= word.mid <= hi:
            d = abs(word.mid - (u.start + u.end) / 2)
            if d < best_d:
                best, best_d = u, d
    return best


def score_track(speaker: str, reference: Reference, segments: list[HypSegment]) -> TrackScore:
    jiwer = require_jiwer()
    utts = sorted((u for u in reference.for_speaker(speaker) if u.scored), key=lambda u: u.start)
    score = TrackScore(speaker, ref_utterances=len(utts), hyp_segments=len(segments))
    words, score.interpolated = words_of(segments)
    windows = _windows(utts)
    assigned: dict[str, list[HypWord]] = {u.id: [] for u in utts}
    inside: list[HypWord] = []
    for w in words:
        owner = _owner(w, windows)
        if owner is None:
            score.hyp_words_outside += 1
            score.hyp_seconds_outside += max(w.end - w.start, 0.0)
        else:
            assigned[owner.id].append(w)
            inside.append(w)
    score.hyp_words_in_windows = len(inside)

    ref_text = normalize(" ".join(u.text for u in utts))
    hyp_text = normalize(" ".join(w.text for w in inside))
    score.ref_words = len(ref_text.split())
    score.ref_chars = len(ref_text)
    if ref_text:
        out = jiwer.process_words(ref_text, hyp_text)
        score.substitutions, score.deletions, score.insertions = (
            out.substitutions,
            out.deletions,
            out.insertions,
        )
        chars = jiwer.process_characters(ref_text, hyp_text)
        score.char_errors = chars.substitutions + chars.deletions + chars.insertions
    else:
        score.insertions = len(hyp_text.split())

    for u in utts:
        ref_u = normalize(u.text)
        mine = assigned[u.id]
        if not ref_u:
            continue
        hyp_u = normalize(" ".join(w.text for w in mine))
        hits = jiwer.process_words(ref_u, hyp_u).hits if hyp_u else 0
        if hits / len(ref_u.split()) >= RECALL_MIN_HIT_RATIO:
            score.recalled += 1
            score.start_errors_s.append(abs(mine[0].start - u.start))
            score.end_errors_s.append(abs(mine[-1].end - u.end))
    return score


def _median_ms(values: list[float]) -> float | None:
    return round(1000 * statistics.median(values), 1) if values else None


@dataclass
class Summary:
    tracks: int
    ref_utterances: int
    ref_words: int
    wer: float | None
    cer: float | None
    speech_recall: float | None
    hyp_words_outside: int
    hyp_seconds_outside: float
    silent_tracks: int
    silent_track_words: int
    silent_track_segments: int
    silent_track_seconds: float
    timestamp_start_median_ms: float | None
    timestamp_end_median_ms: float | None
    interpolated_word_times: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(scores: list[TrackScore]) -> Summary:
    """Micro-averaged over all tracks (and clips)."""
    ref_words = sum(s.ref_words for s in scores)
    ref_chars = sum(s.ref_chars for s in scores)
    ref_utts = sum(s.ref_utterances for s in scores)
    silent = [s for s in scores if s.silent]
    return Summary(
        tracks=len(scores),
        ref_utterances=ref_utts,
        ref_words=ref_words,
        wer=round(sum(s.word_errors for s in scores) / ref_words, 4) if ref_words else None,
        cer=round(sum(s.char_errors for s in scores) / ref_chars, 4) if ref_chars else None,
        speech_recall=round(sum(s.recalled for s in scores) / ref_utts, 4) if ref_utts else None,
        hyp_words_outside=sum(s.hyp_words_outside for s in scores if not s.silent),
        hyp_seconds_outside=round(sum(s.hyp_seconds_outside for s in scores if not s.silent), 2),
        silent_tracks=len(silent),
        silent_track_words=sum(s.hyp_words_outside for s in silent),
        silent_track_segments=sum(s.hyp_segments for s in silent),
        silent_track_seconds=round(sum(s.hyp_seconds_outside for s in silent), 2),
        timestamp_start_median_ms=_median_ms([e for s in scores for e in s.start_errors_s]),
        timestamp_end_median_ms=_median_ms([e for s in scores for e in s.end_errors_s]),
        interpolated_word_times=any(s.interpolated for s in scores),
    )


def segments_from_json(data: dict[str, Any]) -> list[HypSegment]:
    """Read our single-track JSON output (exporters.single)."""
    return [
        HypSegment(
            float(s["start"]),
            float(s["end"]),
            s["text"],
            tuple(HypWord(float(w["start"]), float(w["end"]), w["word"]) for w in s["words"]),
        )
        for s in data["segments"]
    ]
