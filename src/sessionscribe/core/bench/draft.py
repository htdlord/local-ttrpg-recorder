"""Draft references: import an existing transcript, or generate one with our ASR.

Drafts are starting points for hand correction. Importers never drop anything: a
segment that looks like bleed goes to ``excluded`` with the reason, and shows up on
the ``_bleed`` label track where the corrector can bring it back.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sessionscribe.core.bench.clip import Clip
from sessionscribe.core.bench.reference import Excluded, Reference, Utterance
from sessionscribe.core.pipeline.asr import TranscriptionResult
from sessionscribe.core.timeline import samples_to_seconds

FLAG_HINTS = {
    "possible_crosstalk": "overlap?",
    "possible_duplicate_overlap": "dup?",
    "possible_duplicate": "dup?",
}


@dataclass
class ImportStats:
    records: int = 0
    utterances: int = 0
    excluded: int = 0
    hints: Counter[str] = field(default_factory=Counter)
    unknown_flags: Counter[str] = field(default_factory=Counter)


def import_jsonl(clip: Clip, path: Path) -> tuple[Reference, ImportStats]:
    """Import a segment-per-line JSON transcript (fields: start, end, channel, text, ...)."""
    stats = ImportStats()
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{n}: not valid JSON ({exc.msg})") from exc
    records.sort(key=lambda r: (float(r["start"]), r.get("id", 0)))

    utterances: list[Utterance] = []
    excluded: list[Excluded] = []
    for i, rec in enumerate(records, start=1):
        stats.records += 1
        channel = int(rec["channel"])
        track = clip.config.track_for_channel(channel)
        if track is None:
            raise ValueError(f"{path.name}: channel {channel} has no track in clip.toml")
        tags: list[str] = []
        for flag in rec.get("flags") or []:
            hint = FLAG_HINTS.get(flag)
            if hint is None:
                stats.unknown_flags[flag] += 1
            elif hint not in tags:
                tags.append(hint)
                stats.hints[hint] += 1
        source_id = rec.get("id", i)
        utt = Utterance(
            id=f"r{i:03d}",
            speaker=track.speaker,
            start=round(float(rec["start"]), 3),
            end=round(float(rec["end"]), 3),
            text=str(rec.get("text", "")).strip(),
            tags=tags,
            origin=f"import:{path.name}#{source_id}",
        )
        duplicate_of = rec.get("duplicate_of")
        if duplicate_of is not None:
            excluded.append(
                Excluded(utterance=utt, reason=f"bleed? (marked duplicate of #{duplicate_of})")
            )
            stats.excluded += 1
        else:
            utterances.append(utt)
            stats.utterances += 1
    reference = Reference(
        clip=clip.config.id,
        language=clip.config.language,
        source=f"import:{path.name}",
        utterances=utterances,
        excluded=excluded,
    )
    return reference, stats


def reference_from_asr(clip: Clip, results: dict[str, TranscriptionResult]) -> Reference:
    """Every ASR segment of every track as an utterance of that track's speaker."""
    items: list[tuple[float, str, Utterance]] = []
    for speaker, result in results.items():
        for seg in result.segments:
            if not seg.text:
                continue
            start = samples_to_seconds(seg.start)
            items.append(
                (
                    start,
                    speaker,
                    Utterance(
                        id="r000",
                        speaker=speaker,
                        start=start,
                        end=samples_to_seconds(seg.end),
                        text=seg.text,
                        origin=f"asr:{result.model.model_id}/{result.mode}#{seg.index}",
                    ),
                )
            )
    items.sort(key=lambda t: (t[0], t[1]))
    utterances = [u.model_copy(update={"id": f"r{n:03d}"}) for n, (_, _, u) in enumerate(items, 1)]
    model = next(iter(results.values())).model.model_id if results else "?"
    return Reference(
        clip=clip.config.id,
        language=clip.config.language,
        source=f"asr:{model}",
        utterances=utterances,
    )
