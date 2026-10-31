"""Result rows and filters for the Result tab (no Qt)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from ttrpg_recorder.core.merge import TranscriptSegment
from ttrpg_recorder.core.quality import (
    LOW_OWNERSHIP,
    POSSIBLE_DUPLICATE,
    POSSIBLE_WRONG_SPEAKER,
)
from ttrpg_recorder.core.result import SessionResult
from ttrpg_recorder.core.timeline import TIMELINE_RATE

RowKind = Literal["segment", "suppressed", "missed", "gap"]
REVIEW_FLAGS = frozenset({POSSIBLE_WRONG_SPEAKER, POSSIBLE_DUPLICATE, LOW_OWNERSHIP})


@dataclass(frozen=True)
class ResultRow:
    kind: RowKind
    id: str
    speaker: str | None
    start: int  # session samples
    end: int
    text: str = ""
    flags: tuple[str, ...] = ()
    reason: str | None = None  # suppression reason / gap reason
    overlap: bool = False
    track_start: int | None = None
    other_speaker: str | None = None  # bleed source / probable speaker
    reassigned_from: str | None = None
    ownership: float | None = None

    @property
    def start_s(self) -> float:
        return self.start / TIMELINE_RATE

    @property
    def end_s(self) -> float:
        return self.end / TIMELINE_RATE

    @property
    def flagged(self) -> bool:
        return bool(self.flags) and self.kind != "gap"

    @property
    def needs_review(self) -> bool:
        return bool(REVIEW_FLAGS & set(self.flags)) or self.kind == "missed"

    @property
    def can_reassign(self) -> bool:
        return self.kind == "segment"


def _row(seg: TranscriptSegment, kind: RowKind) -> ResultRow:
    return ResultRow(
        kind=kind,
        id=seg.id,
        speaker=seg.speaker,
        start=seg.start,
        end=seg.end,
        text=seg.text,
        flags=tuple(seg.quality_flags),
        reason=seg.suppression_reason,
        overlap=seg.overlap,
        track_start=seg.track_start,
        other_speaker=seg.bleed_source or seg.probable_speaker,
        reassigned_from=seg.reassigned_from,
        ownership=seg.ownership,
    )


def rows_from_result(result: SessionResult) -> list[ResultRow]:
    t = result.transcript
    rows = [_row(s, "segment") for s in t.segments]
    rows += [_row(s, "suppressed") for s in t.suppressed]
    rows += [_row(s, "missed") for s in t.missed]
    rows += [
        ResultRow(
            kind="gap",
            id=f"gap-{i}",
            speaker=g.participant,
            start=g.start,
            end=g.start + g.length,
            reason=g.reason,
        )
        for i, g in enumerate(result.gaps)
    ]
    order = {p.id: i for i, p in enumerate(result.participants)}
    rows.sort(key=lambda r: (r.start, order.get(r.speaker or "", -1), r.id))
    return rows


@dataclass
class ResultFilter:
    show_suppressed: bool = False
    show_missed: bool = True
    show_gaps: bool = True
    only_review: bool = False  # flagged for review: wrong speaker, duplicate, missed, ...
    only_overlap: bool = False
    speakers: set[str] = field(default_factory=set)  # empty = all
    text: str = ""

    def accepts(self, row: ResultRow) -> bool:
        if row.kind == "suppressed" and not self.show_suppressed:
            return False
        if row.kind == "missed" and not self.show_missed:
            return False
        if row.kind == "gap":
            return self.show_gaps and not (self.only_review or self.only_overlap or self.text)
        if self.only_review and not (row.needs_review or row.flagged):
            return False
        if self.only_overlap and not row.overlap:
            return False
        if self.speakers and row.speaker not in self.speakers:
            return False
        return not (self.text and self.text.lower() not in row.text.lower())


def filter_rows(rows: Sequence[ResultRow], f: ResultFilter) -> list[ResultRow]:
    return [r for r in rows if f.accepts(r)]


def counts(rows: Sequence[ResultRow]) -> dict[str, int]:
    return {
        "segments": sum(r.kind == "segment" for r in rows),
        "review": sum(r.kind == "segment" and (r.needs_review or r.flagged) for r in rows)
        + sum(r.kind == "missed" for r in rows),
        "suppressed": sum(r.kind == "suppressed" for r in rows),
        "gaps": sum(r.kind == "gap" for r in rows),
    }


def eta_seconds(done: float, total: float, elapsed_s: float) -> float | None:
    """Remaining time from the processed share; None until there is a meaningful rate."""
    if total <= 0 or done <= 0 or elapsed_s < 2.0:
        return None
    rate = done / elapsed_s
    return max(0.0, (total - done) / rate)


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
