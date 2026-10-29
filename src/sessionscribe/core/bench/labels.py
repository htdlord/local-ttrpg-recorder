"""Audacity label round-trip for hand-correcting a reference transcript.

Label files are UTF-8 without BOM, one label per line: ``start<TAB>end<TAB>text``.
The label text carries the speaker, because Audacity's "Export Labels" writes all
label tracks into one file without track names::

    seat_3 | r012 [overlap] text as spoken

- Change the speaker by editing the prefix; new labels may omit the id.
- A prefix starting with ``_`` (``_seat_3``) marks a line as *not* real speech of that
  speaker (bleed, noise); it is kept in the reference's ``excluded`` list.
- ``_diff`` lines are listening hints only and are ignored when building.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sessionscribe.core.bench.clip import Clip
from sessionscribe.core.bench.reference import (
    Excluded,
    Reference,
    Utterance,
    Validation,
    next_id,
    validate_reference,
)
from sessionscribe.core.fileio import atomic_write_text

DIFF_TRACK = "_diff"
BLEED_FILE = "_bleed.txt"
LABEL_TEXT = re.compile(
    r"^\s*(?P<speaker>[^|]+?)\s*\|\s*"
    r"(?:(?P<id>r\d{3,})(?:\s+|$))?"
    r"(?:\[(?P<tags>[^\]]*)\]\s*)?"
    r"(?P<text>.*)$"
)


@dataclass(frozen=True)
class Label:
    line: int
    start: float
    end: float
    speaker: str
    id: str | None
    tags: tuple[str, ...]
    text: str


def format_label(start: float, end: float, speaker: str, uid: str | None, tags, text: str) -> str:
    head = f"{speaker} | " + (f"{uid} " if uid else "")
    tag_text = f"[{','.join(tags)}] " if tags else ""
    clean = " ".join(text.split())  # labels are single-line
    return f"{start:.6f}\t{end:.6f}\t{head}{tag_text}{clean}".rstrip()


def utterance_label(u: Utterance, speaker: str | None = None) -> str:
    return format_label(u.start, u.end, speaker or u.speaker, u.id, u.tags, u.text)


def write_label_files(
    clip: Clip, reference: Reference, diff_lines: list[str] | None = None
) -> list[Path]:
    """One file per speaker (empty ones too, so every track gets a label track)."""
    folder = clip.labels_dir
    written: list[Path] = []
    for speaker in clip.config.speakers:
        lines = [utterance_label(u) for u in sorted(reference.for_speaker(speaker), key=_start)]
        written.append(_write(folder / f"{speaker}.txt", lines))
    bleed = [
        utterance_label(e.utterance, f"_{e.utterance.speaker}")
        for e in sorted(reference.excluded, key=lambda e: e.utterance.start)
    ]
    written.append(_write(folder / BLEED_FILE, bleed))
    if diff_lines is not None:
        written.append(_write(folder / f"{DIFF_TRACK}.txt", diff_lines))
    return written


def _start(u: Utterance) -> float:
    return u.start


def _write(path: Path, lines: list[str]) -> Path:
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return path


def parse_labels(text: str, errors: list[str], source: str = "labels") -> list[Label]:
    labels: list[Label] = []
    for n, raw in enumerate(text.lstrip("﻿").splitlines(), start=1):
        if not raw.strip() or raw.startswith("\\"):  # blank / spectral-selection line
            continue
        parts = raw.split("\t", 2)
        if len(parts) < 2:
            errors.append(f"{source}:{n}: expected start<TAB>end<TAB>text")
            continue
        try:
            start, end = (float(p.replace(",", ".")) for p in parts[:2])
        except ValueError:
            errors.append(f"{source}:{n}: start/end are not numbers")
            continue
        body = parts[2] if len(parts) == 3 else ""
        m = LABEL_TEXT.match(body)
        if not m:
            errors.append(f"{source}:{n}: missing 'speaker | ' prefix")
            continue
        tags = tuple(t.strip() for t in (m["tags"] or "").split(",") if t.strip())
        labels.append(Label(n, start, end, m["speaker"].strip(), m["id"], tags, m["text"].strip()))
    return labels


def read_labels(paths: list[Path], errors: list[str]) -> list[Label]:
    labels: list[Label] = []
    for path in paths:
        labels.extend(parse_labels(path.read_text(encoding="utf-8"), errors, path.name))
    return labels


def build_reference(
    clip: Clip,
    labels: list[Label],
    base: Reference | None,
    *,
    verified: bool = False,
    source: str = "",
) -> tuple[Reference, Validation]:
    """Turn edited labels into a reference; ids missing from the labels become excluded."""
    validation = Validation()
    base_items: dict[str, Utterance] = {}
    base_reasons: dict[str, str] = {}
    if base is not None:
        base_items = {u.id: u for u in base.utterances}
        base_items |= {e.utterance.id: e.utterance for e in base.excluded}
        base_reasons = {e.utterance.id: e.reason for e in base.excluded}
    used = set(base_items)
    seen: set[str] = set()
    pending: list[tuple[Label, bool]] = []  # (label, excluded)
    for label in labels:
        if label.speaker == DIFF_TRACK:
            continue
        if label.id is not None:
            if label.id in seen:
                validation.errors.append(f"line {label.line}: id {label.id} appears twice")
                continue
            seen.add(label.id)
            used.add(label.id)
        pending.append((label, label.speaker.startswith("_")))

    utterances: list[Utterance] = []
    excluded: list[Excluded] = []
    for label, is_excluded in pending:
        uid = label.id or next_id(used)
        used.add(uid)
        origin = base_items[uid].origin if uid in base_items else "manual"
        speaker = label.speaker.lstrip("_") if is_excluded else label.speaker
        utt = Utterance(
            id=uid,
            speaker=speaker,
            start=round(label.start, 3),
            end=round(label.end, 3),
            text=label.text,
            tags=list(label.tags),
            origin=origin,
        )
        if is_excluded:
            excluded.append(Excluded(utterance=utt, reason="marked as not this speaker's speech"))
        else:
            utterances.append(utt)
    for uid, utt in base_items.items():
        if uid not in seen:
            reason = base_reasons.get(uid, "")
            reason = f"{reason}; deleted while editing" if reason else "deleted while editing"
            excluded.append(Excluded(utterance=utt, reason=reason))

    reference = Reference(
        clip=clip.config.id,
        language=clip.config.language,
        status="verified" if verified else "draft",
        source=source,
        utterances=sorted(utterances, key=lambda u: (u.start, u.speaker)),
        excluded=sorted(excluded, key=lambda e: e.utterance.start),
    )
    checks = validate_reference(
        reference, clip.config.speakers, clip.config.duration_s, verified=verified
    )
    validation.errors += checks.errors
    validation.warnings += checks.warnings
    return reference, validation
