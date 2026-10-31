"""Triage reference (M3 prerequisite, DECISIONS 2026-10-30).

Every segment of our own bleed-off session output, and every possible-missed-speech
stretch, becomes a *unit* that a person labels by ear; no text is corrected:

- ``s`` own speech of the track's participant;
- ``b`` bleed: another participant's voice that their own track also has (droppable);
- ``w`` wrong speaker: someone else said it, but only this track has it (reassign);
- ``v`` mixed: own speech and bleed in the same segment (counts as own speech);
- ``o`` true overlap: own speech while someone else speaks;
- ``n`` no speech (noise, hallucination);
- ``?`` unsure, not scored.

``triage.json`` stores units by their stable segment id plus speaker and time, never
text. Suggestions come from energy levels and text similarity between tracks and only
save keystrokes; metrics use the final labels.
"""

from __future__ import annotations

import csv
import io
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ttrpg_recorder.core.bench.clip import Clip
from ttrpg_recorder.core.bench.text import normalize
from ttrpg_recorder.core.config import Settings, load_settings
from ttrpg_recorder.core.fileio import atomic_write_json, atomic_write_text
from ttrpg_recorder.core.merge import TranscriptSegment
from ttrpg_recorder.core.pipeline.coupling import speech_level
from ttrpg_recorder.core.pipeline.features import SILENCE_DB, track_envelope
from ttrpg_recorder.core.result import SessionResult
from ttrpg_recorder.core.session import Participant, Session, SessionConfig
from ttrpg_recorder.core.timeline import TIMELINE_RATE

TRIAGE_FILE = "triage.json"
TRIAGE_SCHEMA_VERSION = 1
LABELS = ("s", "b", "w", "v", "o", "n", "?")
WITH_SOURCE = ("b", "w", "v")  # labels that may name another participant
SPEECH = ("s", "w", "v", "o")  # own (or reassignable) speech: recall, false suppression
LABEL_NAMES = {
    "s": "own speech",
    "b": "bleed (source track has it too)",
    "w": "wrong speaker (only copy, reassign)",
    "v": "mixed: own speech + bleed",
    "o": "true overlap",
    "n": "no speech",
    "?": "unsure",
}

# Suggestion thresholds. Levels are relative to each track's own typical speech level
# (95th percentile of its active frames), because mic gains differ by 15-20 dB in practice.
BLEED_DB = 6.0  # another track this much louder (relative) suggests bleed
OWN_DB = 6.0  # own track this much louder suggests own speech
SPEAKING_REL = -10.0  # both tracks this close to their speech level: both speak
QUIET_REL = -25.0  # a missed-speech stretch this far below own speech: no speech
SIMILAR = 80.0
DIFFERENT = 50.0
NEAR_S = 1.0  # other-track segments this close in time are compared by text

UnitKind = Literal["segment", "missed"]


class TriageUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    speaker: str
    kind: UnitKind = "segment"
    start: float
    end: float
    suggestion: str | None = None
    suggestion_source: str | None = None
    # Levels relative to each track's own typical speech level (0 dB = speaking normally).
    own_db: float | None = None
    other_db: float | None = None
    other_speaker: str | None = None
    similarity: float | None = None
    label: str | None = None
    source: str | None = Field(None, description="Source (b, v) or true speaker (w).")
    accepted_suggestion: bool | None = None
    decide_s: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class TriageFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = TRIAGE_SCHEMA_VERSION
    clip: str
    status: Literal["draft", "complete"] = "draft"
    speakers: list[str]
    asr: dict[str, Any] = Field(default_factory=dict, description="Model and mode used.")
    units: list[TriageUnit] = Field(default_factory=list)

    def refresh_status(self) -> None:
        done = all(u.label is not None for u in self.units)
        self.status = "complete" if done and self.units else "draft"

    def first_unlabelled(self) -> int:
        return next((i for i, u in enumerate(self.units) if u.label is None), len(self.units))


def triage_path(clip: Clip) -> Path:
    return clip.folder / TRIAGE_FILE


def read_triage(path: Path) -> TriageFile:
    return TriageFile.model_validate_json(path.read_text(encoding="utf-8"))


def write_triage(path: Path, triage: TriageFile) -> None:
    triage.refresh_status()
    atomic_write_json(path, triage.model_dump(mode="json"))


# --- the session behind the units ---------------------------------------------------------


def clip_session(clip: Clip) -> Session:
    participants = [
        Participant(id=t.speaker, name=t.name or t.speaker, source=t.file)
        for t in clip.config.tracks
    ]
    return Session(clip.folder, SessionConfig(name=clip.config.id, participants=participants))


def triage_settings(clip: Clip) -> Settings:
    """Default settings with bleed handling off: exactly what M3 will filter."""
    return load_settings(
        cli={"decoding": {"language": clip.config.language}, "bleed": {"mode": "off"}}
    )


# --- suggestions --------------------------------------------------------------------------


@dataclass(frozen=True)
class Levels:
    own_db: float
    other_db: float | None
    other_speaker: str | None


def _mean_db(env: np.ndarray) -> float:
    if env.size == 0:
        return SILENCE_DB
    power = np.power(10.0, env.astype(np.float64) / 10.0)
    return float(10.0 * np.log10(max(power.mean(), 1e-12)))


def unit_levels(
    envelopes: Mapping[str, np.ndarray], speaker: str, start: int, end: int, frame: int
) -> Levels:
    """Own level and the loudest other track over the unit's louder own frames."""
    a, b = start // frame, max(start // frame + 1, -(-end // frame))
    own = envelopes[speaker][a:b]
    if own.size == 0:
        return Levels(SILENCE_DB, None, None)
    loud = own >= np.median(own)
    best: tuple[float, str] | None = None
    for other, env in envelopes.items():
        if other == speaker:
            continue
        part = env[a:b]
        if part.size < own.size:
            part = np.concatenate([part, np.full(own.size - part.size, SILENCE_DB, np.float32)])
        level = _mean_db(part[loud])
        if best is None or level > best[0]:
            best = (level, other)
    own_db = _mean_db(own[loud])
    return Levels(own_db, None if best is None else best[0], None if best is None else best[1])


def best_similarity(
    seg: TranscriptSegment, others: Sequence[TranscriptSegment]
) -> tuple[float, str | None]:
    from rapidfuzz import fuzz

    text = normalize(seg.text)
    if not text:
        return 0.0, None
    near = NEAR_S * TIMELINE_RATE
    best: tuple[float, str | None] = (0.0, None)
    for o in others:
        if o.speaker == seg.speaker or o.end < seg.start - near or o.start > seg.end + near:
            continue
        score = float(fuzz.ratio(text, normalize(o.text)))
        if score > best[0]:
            best = (score, o.speaker)
    return best


def relative_envelopes(envelopes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: (v - speech_level(v)).astype(np.float32) for k, v in envelopes.items()}


def suggest(
    kind: UnitKind,
    levels: Levels,
    similarity: float,
    similar_speaker: str | None,
    *,
    other_has_segment: bool,
) -> tuple[str | None, str | None]:
    """(label, source) or (None, None) when a person has to decide; relative levels."""
    own, other = levels.own_db, levels.other_db
    diff = None if other is None else own - other
    if kind == "missed":
        if own < QUIET_REL:
            return "n", None
        if diff is not None and diff <= -BLEED_DB and other_has_segment:
            return "b", levels.other_speaker
        return None, None
    if diff is None:  # single track
        return ("s", None) if own >= QUIET_REL else (None, None)
    if diff <= -BLEED_DB and other_has_segment:
        return "b", levels.other_speaker
    if similarity >= SIMILAR and diff < 0:
        return "b", similar_speaker
    if diff >= OWN_DB:  # clearly louder here: the original, even if another track echoes it
        return "s", None
    if (
        abs(diff) < BLEED_DB
        and own >= SPEAKING_REL
        and other is not None
        and other >= SPEAKING_REL
        and similarity < DIFFERENT
    ):
        return "o", None
    return None, None


# --- building units -----------------------------------------------------------------------


def build_units(
    result: SessionResult, envelopes: Mapping[str, np.ndarray], frame: int
) -> list[TriageUnit]:
    """Units in session-time order: segments plus possible-missed-speech stretches."""
    segs = result.transcript.segments
    items: list[tuple[TranscriptSegment, UnitKind]] = [(s, "segment") for s in segs]
    items += [(m, "missed") for m in result.transcript.missed]
    order = {p.id: i for i, p in enumerate(result.participants)}
    envelopes = relative_envelopes(envelopes)
    items.sort(key=lambda it: (it[0].start, order.get(it[0].speaker, 0), it[0].id))
    units: list[TriageUnit] = []
    for seg, kind in items:
        levels = unit_levels(envelopes, seg.speaker, seg.start, seg.end, frame)
        sim, sim_speaker = best_similarity(seg, segs) if kind == "segment" else (0.0, None)
        has_segment = any(
            o.speaker == levels.other_speaker and o.end > seg.start and o.start < seg.end
            for o in segs
        )
        label, source = suggest(kind, levels, sim, sim_speaker, other_has_segment=has_segment)
        units.append(
            TriageUnit(
                id=seg.id,
                speaker=seg.speaker,
                kind=kind,
                start=round(seg.start / TIMELINE_RATE, 3),
                end=round(seg.end / TIMELINE_RATE, 3),
                suggestion=label,
                suggestion_source=source,
                own_db=round(levels.own_db, 1),
                other_db=None if levels.other_db is None else round(levels.other_db, 1),
                other_speaker=levels.other_speaker,
                similarity=round(sim, 1) if kind == "segment" else None,
            )
        )
    drop_contradictions(units)
    return units


def drop_contradictions(units: Sequence[TriageUnit]) -> int:
    """Clear "b" suggestions of overlapping units that name each other as the source.

    Level cues cannot tell which way the bleed goes there; a person decides.
    """
    bad: set[int] = set()
    for i, x in enumerate(units):
        if x.suggestion != "b":
            continue
        for j, y in enumerate(units):
            if (
                y.speaker == x.suggestion_source
                and y.suggestion == "b"
                and y.suggestion_source == x.speaker
                and min(x.end, y.end) > max(x.start, y.start)
            ):
                bad.update((i, j))
    for i in bad:
        units[i].suggestion = units[i].suggestion_source = None
    return len(bad)


def session_envelopes(
    session: Session, result: SessionResult, cache_root: Path
) -> dict[str, np.ndarray]:
    """Envelopes on the session frame grid (offsets applied by whole frames)."""
    from ttrpg_recorder.core.cache import Cache
    from ttrpg_recorder.core.pipeline.features import frame_samples

    frame_ms = result.settings.bleed.frame_ms
    frame = frame_samples(frame_ms)
    cache = Cache(cache_root)
    out: dict[str, np.ndarray] = {}
    for p in session.participants:
        env = track_envelope(session.open_track(p), cache, frame_ms)
        shift = round(p.offset_samples / frame)
        out[p.id] = np.concatenate([np.full(shift, SILENCE_DB, np.float32), env]) if shift else env
    return out


def overlap_fraction(a0: float, a1: float, b0: float, b1: float) -> float:
    length = a1 - a0
    if length <= 0:
        return 0.0
    return max(0.0, min(a1, b1) - max(a0, b0)) / length


def carry_labels(old: Sequence[TriageUnit], new: list[TriageUnit]) -> list[TriageUnit]:
    """Old labels onto rebuilt units: by id, else same speaker and >= 50 % time overlap.

    Returns the old labelled units that found no new unit.
    """
    by_id = {u.id: u for u in old if u.label is not None}
    used: set[str] = set()
    for unit in new:
        match = by_id.get(unit.id)
        if match is None or match.speaker != unit.speaker:
            match = next(
                (
                    u
                    for u in old
                    if u.label is not None
                    and u.id not in used
                    and u.speaker == unit.speaker
                    and overlap_fraction(unit.start, unit.end, u.start, u.end) >= 0.5
                    and overlap_fraction(u.start, u.end, unit.start, unit.end) >= 0.5
                ),
                None,
            )
        if match is not None and match.id not in used:
            used.add(match.id)
            unit.label, unit.source = match.label, match.source
            unit.accepted_suggestion, unit.decide_s = match.accepted_suggestion, match.decide_s
    return [u for u in old if u.label is not None and u.id not in used]


# --- labelling state machine --------------------------------------------------------------

Action = Literal["play", "play_mix", "play_other", "help", "quit", "moved", "need_source", "none"]


@dataclass
class Labeller:
    """Keys in, labels and actions out; no I/O (the CLI plays audio and saves)."""

    triage: TriageFile
    order: list[int]  # unit indices to visit
    pos: int = 0
    pending: str | None = None  # label waiting for a source digit
    history: list[int] = field(default_factory=list)
    clock: Callable[[], float] = time.monotonic
    shown_at: float = 0.0

    def __post_init__(self) -> None:
        self.shown_at = self.clock()

    @property
    def done(self) -> bool:
        return self.pos >= len(self.order)

    @property
    def unit(self) -> TriageUnit:
        return self.triage.units[self.order[self.pos]]

    def _set(self, label: str, source: str | None, accepted: bool) -> Action:
        unit = self.unit
        unit.label, unit.source = label, source
        unit.accepted_suggestion = accepted
        unit.decide_s = round(self.clock() - self.shown_at, 2)
        self.history.append(self.pos)
        self.pos += 1
        self.pending = None
        self.shown_at = self.clock()
        return "moved"

    def key(self, key: str) -> Action:
        if self.done:
            return "quit" if key == "q" else "none"
        if self.pending is not None:
            if key.isdigit() and 1 <= int(key) <= len(self.triage.speakers):
                return self._set(self.pending, self.triage.speakers[int(key) - 1], False)
            if key in ("\r", "\n"):
                return self._set(self.pending, None, False)
            if key == "\x1b":
                self.pending = None
            return "none"
        if key in ("\r", "\n"):
            unit = self.unit
            if unit.suggestion is None:
                return "none"
            return self._set(unit.suggestion, unit.suggestion_source, True)
        if key in LABELS:
            if key in WITH_SOURCE:
                self.pending = key
                return "need_source"
            return self._set(key, None, False)
        if key == "u" and self.history:
            self.pos = self.history.pop()
            self.shown_at = self.clock()
            return "moved"
        return {
            " ": "play",
            "m": "play_mix",
            "x": "play_other",
            "h": "help",
            "q": "quit",
        }.get(key, "none")  # type: ignore[return-value]


HELP = """\
Keys:  Enter accept suggestion   space replay   m all tracks mixed   x loudest other track
       u back one unit           h this help    q save and quit
Labels:
  s  own speech of this track's participant
  b  bleed: another participant's voice that THEIR track has too (it can be dropped)
  w  wrong speaker: someone else said it, but ONLY this track has it (reassign it)
  v  mixed: own speech and bleed in the same segment (counts as own speech)
  o  true overlap: this participant speaks while someone else speaks too
  n  no speech (noise, hallucination)
  ?  unsure (not scored)
  After b, w or v: digit 1-8 = source / true speaker (Enter = unknown, Esc = cancel).
Example: Anna says "Nem!" and Béla's mic picks it up faintly.
  On Béla's track: if Anna's track also has "Nem!" -> b (source Anna).
  If Anna's track missed it (only Béla's has it) -> w (true speaker Anna)."""


def visit_order(triage: TriageFile, only: str | None, start: int | None) -> list[int]:
    """Unit indices to visit: unlabelled by default, or a filter / start position."""
    idx = list(range(len(triage.units)))
    if only == "unlabelled" or (only is None and start is None):
        idx = [i for i in idx if triage.units[i].label is None]
    elif only == "suggested":
        idx = [i for i in idx if triage.units[i].suggestion is not None]
    elif only == "all":
        pass
    elif only:
        idx = [i for i in idx if triage.units[i].speaker == only]
    if start is not None:
        idx = [i for i in idx if i >= start - 1]
    return idx


# --- CSV fallback -------------------------------------------------------------------------

CSV_FIELDS = ("id", "speaker", "kind", "start", "end", "suggestion", "label", "source")


def to_csv(triage: TriageFile) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for u in triage.units:
        writer.writerow(
            {
                "id": u.id,
                "speaker": u.speaker,
                "kind": u.kind,
                "start": f"{u.start:.3f}",
                "end": f"{u.end:.3f}",
                "suggestion": (u.suggestion or "")
                + (f">{u.suggestion_source}" if u.suggestion_source else ""),
                "label": u.label or "",
                "source": u.source or "",
            }
        )
    return buf.getvalue()


def write_csv(path: Path, triage: TriageFile) -> None:
    atomic_write_text(path, to_csv(triage))


def apply_csv(triage: TriageFile, text: str) -> tuple[int, list[str]]:
    """Labels from a CSV (by id); returns (updated count, problems)."""
    units = {u.id: u for u in triage.units}
    problems: list[str] = []
    updated = 0
    for n, row in enumerate(csv.DictReader(io.StringIO(text.lstrip("﻿"))), start=2):
        unit = units.get((row.get("id") or "").strip())
        label = (row.get("label") or "").strip().lower()
        source = (row.get("source") or "").strip() or None
        if unit is None:
            problems.append(f"line {n}: unknown id")
            continue
        if not label:
            continue
        if label not in LABELS:
            problems.append(f"line {n}: unknown label {label!r}")
            continue
        if source is not None and (label not in WITH_SOURCE or source not in triage.speakers):
            problems.append(f"line {n}: source {source!r} not valid for label {label!r}")
            continue
        unit.label, unit.source, unit.accepted_suggestion = label, source, None
        updated += 1
    return updated, problems


# --- stats --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageStats:
    units: int
    labelled: int
    labels: dict[str, int]
    suggested: int
    accepted: int
    mean_decide_s: float | None


def triage_stats(triage: TriageFile) -> TriageStats:
    labelled = [u for u in triage.units if u.label is not None]
    times = [u.decide_s for u in labelled if u.decide_s is not None]
    return TriageStats(
        units=len(triage.units),
        labelled=len(labelled),
        labels={k: sum(u.label == k for u in labelled) for k in LABELS},
        suggested=sum(u.suggestion is not None for u in triage.units),
        accepted=sum(bool(u.accepted_suggestion) for u in labelled),
        mean_decide_s=round(sum(times) / len(times), 2) if times else None,
    )


# --- entry points -------------------------------------------------------------------------


def clip_result(
    clip: Clip, *, transcribe: bool = True, **kwargs: Any
) -> tuple[Session, SessionResult]:
    """Bleed-off session output of a clip (ASR only for chunks missing from the cache)."""
    from ttrpg_recorder.core.pipeline.run import export_session, process_session

    session = clip_session(clip)
    settings = triage_settings(clip)
    if transcribe:
        result, _ = process_session(session, settings, **kwargs)
    else:
        result, _ = export_session(session, settings, **kwargs)
    return session, result


OWN_SUGGEST = 0.8  # unit owned at least this much by its track: suggest "s"
BLEED_SUGGEST = 0.2  # owned less, and another track owns it and has words there: "b"


def ownership_suggestions(
    units: Sequence[TriageUnit], own: Any, segments: Sequence[TranscriptSegment]
) -> None:
    """Replace suggestions with ownership-based ones (clip01: 12 of 15 right vs 5 of 16).

    Where ownership is ambiguous (both tracks own the frames) there is no suggestion.
    """
    for u in units:
        a, b = round(u.start * TIMELINE_RATE), round(u.end * TIMELINE_RATE)
        mine = own.owned(u.speaker, a, b)
        others = [(own.owned(k, a, b), k) for k in own.speakers if k != u.speaker]
        share, source = max(others) if others else (0.0, None)
        has_words = any(x.speaker == source and x.end > a and x.start < b for x in segments)
        u.suggestion, u.suggestion_source = None, None
        if mine >= OWN_SUGGEST and share < 0.5:
            u.suggestion = "s"
        elif mine < BLEED_SUGGEST and share >= 0.5:
            u.suggestion = "b" if has_words else "w"
            u.suggestion_source = source


def new_triage(clip: Clip, session: Session, result: SessionResult) -> TriageFile:
    from ttrpg_recorder.core.cache import Cache
    from ttrpg_recorder.core.pipeline.features import frame_samples
    from ttrpg_recorder.core.pipeline.ownership import compute_ownership

    cache_root = session.work_dir(result.settings.output.work_dir) / "cache"
    envelopes = session_envelopes(session, result, cache_root)
    units = build_units(result, envelopes, frame_samples(result.settings.bleed.frame_ms))
    post = result.settings.model_copy(
        update={"bleed": result.settings.bleed.model_copy(update={"mode": "post"})}
    )
    own = compute_ownership(session, post, Cache(cache_root))
    ownership_suggestions(units, own, result.transcript.segments)
    return TriageFile(
        clip=clip.config.id,
        speakers=[p.id for p in session.participants],
        asr={
            "model": result.model.get("model_id"),
            "compute_type": result.model.get("compute_type"),
            "mode": result.settings.asr.mode,
            "language": result.settings.decoding.language,
        },
        units=units,
    )


def texts_by_id(result: SessionResult) -> dict[str, str]:
    """Unit texts for the terminal only; never written to a file or log."""
    return {s.id: s.text for s in result.transcript.segments}
