"""Where two drafts disagree: guides the corrector to the passages worth listening to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sessionscribe.core.bench.labels import DIFF_TRACK, format_label
from sessionscribe.core.bench.reference import Reference, Utterance
from sessionscribe.core.bench.text import cer

DIFF_CER_THRESHOLD = 0.15

Kind = Literal["match", "differs", "only_import", "only_asr"]


@dataclass
class DiffItem:
    kind: Kind
    speaker: str
    start: float
    end: float
    import_id: str | None
    import_excluded: bool
    import_text: str
    asr_ids: list[str] = field(default_factory=list)
    asr_text: str = ""
    cer: float | None = None


def _overlap(a: Utterance, b: Utterance) -> float:
    return min(a.end, b.end) - max(a.start, b.start)


def compare(imported: Reference, asr: Reference, speakers: list[str]) -> list[DiffItem]:
    """Pair items of the same speaker by time overlap; each ASR item joins its best match."""
    items: list[DiffItem] = []
    for speaker in speakers:
        ours = [(u, False) for u in imported.for_speaker(speaker)]
        ours += [(e.utterance, True) for e in imported.excluded if e.utterance.speaker == speaker]
        theirs = asr.for_speaker(speaker)
        matched: dict[int, list[Utterance]] = {i: [] for i in range(len(ours))}
        for a in theirs:
            best, best_ov = None, 0.0
            for i, (u, _) in enumerate(ours):
                ov = _overlap(u, a)
                if ov > best_ov:
                    best, best_ov = i, ov
            if best is None:
                items.append(
                    DiffItem("only_asr", speaker, a.start, a.end, None, False, "", [a.id], a.text)
                )
            else:
                matched[best].append(a)
        for i, (u, excluded) in enumerate(ours):
            parts = sorted(matched[i], key=lambda x: x.start)
            if not parts:
                items.append(
                    DiffItem("only_import", speaker, u.start, u.end, u.id, excluded, u.text)
                )
                continue
            text = " ".join(p.text for p in parts)
            rate = cer(u.text, text)
            kind: Kind = "differs" if rate > DIFF_CER_THRESHOLD else "match"
            items.append(
                DiffItem(
                    kind,
                    speaker,
                    min(u.start, parts[0].start),
                    max(u.end, parts[-1].end),
                    u.id,
                    excluded,
                    u.text,
                    [p.id for p in parts],
                    text,
                    rate,
                )
            )
    return sorted(items, key=lambda d: (d.start, d.speaker))


def diff_ids(d: DiffItem) -> str:
    """Both drafts number from r001, so name the side."""
    return f"import {d.import_id or '-'} / asr {','.join(d.asr_ids) or '-'}"


def diff_label_lines(items: list[DiffItem]) -> list[str]:
    lines = []
    for d in items:
        if d.kind == "match":
            continue
        what = d.kind if d.cer is None else f"{d.kind} CER {d.cer:.2f}"
        ids = diff_ids(d)
        lines.append(
            format_label(d.start, d.end, DIFF_TRACK, None, (), f"{d.speaker} {what} {ids}")
        )
    return lines


def diff_markdown(items: list[DiffItem]) -> str:
    """Side-by-side text; written only inside the (git-ignored) clip folder."""
    out = ["# Draft comparison: import vs ASR", ""]
    counts: dict[str, int] = {}
    for d in items:
        counts[d.kind] = counts.get(d.kind, 0) + 1
    out.append(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    out.append("")
    for d in items:
        if d.kind == "match":
            continue
        flag = " (excluded in import)" if d.import_excluded else ""
        rate = "" if d.cer is None else f", CER {d.cer:.2f}"
        out.append(f"## {d.start:.2f}-{d.end:.2f} {d.speaker}: {d.kind}{rate}{flag}")
        out.append(f"- import {d.import_id or '-'}: {d.import_text}")
        out.append(f"- asr {' '.join(d.asr_ids) or '-'}: {d.asr_text}")
        out.append("")
    return "\n".join(out)
