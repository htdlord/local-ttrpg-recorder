"""Score a session output (JSON) against a triage reference (SPEC §11.2, M3).

A unit is found in the output by segment id (post mode keeps the ASR, so ids match) or
by the same speaker and at least 50 % mutual time overlap (pre mode re-runs ASR). A unit
is *kept* (in ``segments``), *suppressed* (in ``suppressed``) or *absent*. ``?`` units
are not scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from sessionscribe.core.bench.triage import SPEECH, TriageFile, TriageUnit, overlap_fraction

Where = Literal["kept", "suppressed", "absent"]
REVIEW_FLAGS = frozenset({"possible_duplicate", "low_ownership", "possible_wrong_speaker"})


def _find(unit: TriageUnit, items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for item in items:
        if item.get("id") == unit.id and item.get("speaker") == unit.speaker:
            return item
    for item in items:
        if item.get("speaker") != unit.speaker:
            continue
        a, b = float(item["start"]), float(item["end"])
        if (
            overlap_fraction(unit.start, unit.end, a, b) >= 0.5
            and overlap_fraction(a, b, unit.start, unit.end) >= 0.5
        ):
            return item
    return None


def locate(unit: TriageUnit, output: Mapping[str, Any]) -> tuple[Where, Mapping[str, Any] | None]:
    item = _find(unit, output.get("segments", []))
    if item is not None:
        return "kept", item
    item = _find(unit, output.get("suppressed", []))
    if item is not None:
        return "suppressed", item
    return "absent", None


def _rate(part: float, whole: float) -> float | None:
    return round(part / whole, 4) if whole else None


@dataclass(frozen=True)
class TriageScore:
    labelled: int
    unsure: int
    speech_units: int
    speech_recall: float | None  # s/w/v/o kept
    speech_recall_s: float | None  # same, weighted by seconds
    false_suppression: float | None  # s/w/v/o suppressed
    speech_absent: int  # s/w/v/o neither kept nor suppressed (e.g. pre mode lost it)
    bleed_units: int
    residual_duplicates: float | None  # b kept
    residual_flagged: float | None  # b kept but flagged for review (duplicate/ownership)
    overlap_units: int
    overlap_kept: float | None  # o kept
    overlap_marked: float | None  # o kept and marked overlap
    no_speech_units: int
    hallucination_residue: float | None  # n kept
    wrong_speaker_units: int
    attribution: float | None  # w kept under its true speaker (informational before M3)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_triage(triage: TriageFile, output: Mapping[str, Any]) -> TriageScore:
    units = [u for u in triage.units if u.label is not None]
    scored = [u for u in units if u.label != "?"]
    where = {u.id: locate(u, output) for u in scored}

    def group(*labels: str) -> list[TriageUnit]:
        return [u for u in scored if u.label in labels]

    def count(us: Sequence[TriageUnit], state: Where) -> int:
        return sum(where[u.id][0] == state for u in us)

    speech = group(*SPEECH)
    bleed, overlap, nospeech, wrong = group("b"), group("o"), group("n"), group("w")
    kept_s = sum(u.duration for u in speech if where[u.id][0] == "kept")
    marked = sum(
        where[u.id][0] == "kept" and bool((where[u.id][1] or {}).get("overlap")) for u in overlap
    )
    attributed = sum(
        where[u.id][0] == "kept"
        and u.source is not None
        and (where[u.id][1] or {}).get("speaker") == u.source
        for u in wrong
    )
    return TriageScore(
        labelled=len(units),
        unsure=len(units) - len(scored),
        speech_units=len(speech),
        speech_recall=_rate(count(speech, "kept"), len(speech)),
        speech_recall_s=_rate(kept_s, sum(u.duration for u in speech)),
        false_suppression=_rate(count(speech, "suppressed"), len(speech)),
        speech_absent=count(speech, "absent"),
        bleed_units=len(bleed),
        residual_duplicates=_rate(count(bleed, "kept"), len(bleed)),
        residual_flagged=_rate(
            sum(
                where[u.id][0] == "kept"
                and bool(REVIEW_FLAGS & set((where[u.id][1] or {}).get("quality_flags", ())))
                for u in bleed
            ),
            len(bleed),
        ),
        overlap_units=len(overlap),
        overlap_kept=_rate(count(overlap, "kept"), len(overlap)),
        overlap_marked=_rate(marked, len(overlap)),
        no_speech_units=len(nospeech),
        hallucination_residue=_rate(count(nospeech, "kept"), len(nospeech)),
        wrong_speaker_units=len(wrong),
        attribution=_rate(attributed, sum(u.source is not None for u in wrong)),
    )
