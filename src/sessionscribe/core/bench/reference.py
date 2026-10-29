"""Reference transcript model and validation.

A reference lists what each participant actually said (speaker, start, end, text, tags).
Everything that was considered but is not real speech of that speaker (bleed, deleted
lines) is kept in ``excluded`` with a reason, so nothing disappears without a trace.
Times are seconds with millisecond precision: this is a public, hand-edited format.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sessionscribe.core.fileio import atomic_write_text

REFERENCE_SCHEMA_VERSION = 1

# Final tags. "overlap": genuine simultaneous speech; "unsure": text uncertain;
# "inaudible"/"laugh": not scored; "short": short answer (dedup tests).
TAGS = frozenset({"overlap", "unsure", "inaudible", "laugh", "short"})
UNSCORED_TAGS = frozenset({"inaudible", "laugh"})
# Draft hints from importers; a verified reference must not contain them.
HINT_TAGS = frozenset({"overlap?", "dup?", "bleed?"})

ID_PATTERN = re.compile(r"^r\d{3,}$")
SELF_OVERLAP_TOLERANCE_S = 0.05
DURATION_TOLERANCE_S = 0.5


class Utterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    speaker: str
    start: float = Field(ge=0)
    end: float
    text: str
    tags: list[str] = Field(default_factory=list)
    origin: str = Field("", description="Where it came from, e.g. import:jsonl#12 or manual.")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def scored(self) -> bool:
        return not (UNSCORED_TAGS & set(self.tags))


class Excluded(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance: Utterance
    reason: str


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = REFERENCE_SCHEMA_VERSION
    clip: str
    language: str
    status: Literal["draft", "verified"] = "draft"
    source: str = ""
    utterances: list[Utterance] = Field(default_factory=list)
    excluded: list[Excluded] = Field(default_factory=list)

    def for_speaker(self, speaker: str) -> list[Utterance]:
        return [u for u in self.utterances if u.speaker == speaker]


def read_reference(path: Path) -> Reference:
    return Reference.model_validate_json(path.read_text(encoding="utf-8"))


def write_reference(path: Path, reference: Reference) -> None:
    data = reference.model_dump(mode="json")
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def next_id(used: set[str]) -> str:
    n = 1 + max((int(i[1:]) for i in used if ID_PATTERN.match(i)), default=0)
    return f"r{n:03d}"


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_reference(
    ref: Reference, speakers: list[str], duration_s: float, *, verified: bool = False
) -> Validation:
    """Structural checks; ``verified`` additionally forbids leftover draft hints."""
    v = Validation()
    known = set(speakers)
    seen: set[str] = set()
    for u in ref.utterances:
        where = f"{u.id} ({u.speaker} {u.start:.3f}-{u.end:.3f})"
        if u.id in seen:
            v.errors.append(f"{where}: duplicate id")
        seen.add(u.id)
        if not ID_PATTERN.match(u.id):
            v.errors.append(f"{where}: id must look like r001")
        if u.speaker not in known:
            v.errors.append(f"{where}: unknown speaker (known: {', '.join(speakers)})")
        if u.end <= u.start:
            v.errors.append(f"{where}: end must be after start")
        if u.end > duration_s + DURATION_TOLERANCE_S:
            v.errors.append(f"{where}: ends after the clip ({duration_s:.3f} s)")
        if not u.text.strip() and u.scored:
            v.errors.append(f"{where}: empty text (tag it [inaudible] if nothing is intelligible)")
        for tag in u.tags:
            if tag in HINT_TAGS:
                (v.errors if verified else v.warnings).append(
                    f"{where}: draft hint [{tag}] still present"
                )
            elif tag not in TAGS:
                v.errors.append(f"{where}: unknown tag [{tag}] (known: {', '.join(sorted(TAGS))})")
    for speaker in speakers:
        own = sorted(ref.for_speaker(speaker), key=lambda u: u.start)
        for a, b in itertools.pairwise(own):
            overlap = a.end - b.start
            if overlap > SELF_OVERLAP_TOLERANCE_S:
                v.errors.append(
                    f"{a.id}/{b.id} ({speaker}): overlap each other by {overlap:.3f} s; "
                    "one person cannot say two things at once - merge or move one"
                )
    for e in ref.excluded:
        if not e.reason:
            v.errors.append(f"excluded {e.utterance.id}: missing reason")
    return v
