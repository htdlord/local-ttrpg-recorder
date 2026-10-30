"""Session exporters (SPEC §7): txt, srt, vtt, md and json; UTF-8 without BOM."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sessionscribe import __version__
from sessionscribe.core.exporters.single import _library_versions
from sessionscribe.core.fileio import atomic_write_json, atomic_write_text
from sessionscribe.core.merge import TranscriptSegment, join_same_speaker
from sessionscribe.core.pipeline import PROCESSING_VERSION
from sessionscribe.core.result import SessionResult
from sessionscribe.core.timeline import (
    TIMELINE_RATE,
    format_clock,
    samples_to_seconds,
    seconds_to_samples,
)

JSON_SCHEMA_VERSION = 1
FLAG_MARK = "⚠"
OVERLAP_MARK = "⟂"


def _millis(result: SessionResult) -> bool:
    return result.settings.output.timestamp_style == "hh:mm:ss.mmm"


def _groups(result: SessionResult) -> list[list[TranscriptSegment]]:
    gap = seconds_to_samples(result.settings.output.merge_same_speaker_gap_s)
    return join_same_speaker([s for s in result.transcript.segments if s.text], gap)


def _text(group: list[TranscriptSegment]) -> str:
    return " ".join(s.text for s in group)


def to_txt(result: SessionResult) -> str:
    millis = _millis(result)
    lines = [
        f"[{format_clock(g[0].start, millis=millis)}] {result.name_of(g[0].speaker)}: {_text(g)}"
        for g in _groups(result)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _cue_time(samples: int, sep: str) -> str:
    return format_clock(samples, millis=True).replace(".", sep)


def _cues(result: SessionResult, sep: str) -> list[str]:
    return [
        f"{_cue_time(s.start, sep)} --> {_cue_time(max(s.end, s.start + 1), sep)}\n"
        f"{result.name_of(s.speaker)}: {s.text}\n"
        for s in result.transcript.segments
        if s.text
    ]


def to_srt(result: SessionResult) -> str:
    return "\n".join(f"{i}\n{cue}" for i, cue in enumerate(_cues(result, ","), 1))


def to_vtt(result: SessionResult) -> str:
    return "WEBVTT\n\n" + "\n".join(_cues(result, "."))


def to_md(result: SessionResult) -> str:
    millis = _millis(result)
    model = result.model
    languages = sorted({lang for lang in result.languages.values() if lang})
    lines = [
        f"# {result.name}",
        "",
        f"- Date: {result.date or 'unknown'}",
        f"- Length: {format_clock(result.duration)}",
        "- Participants: " + ", ".join(p.display_name for p in result.participants),
        f"- Model: {model.get('model_id', result.settings.model.id)} "
        f"({model.get('compute_type', result.settings.model.compute_type)}), "
        f"{result.settings.asr.mode} mode",
        f"- Language: {', '.join(languages) or result.settings.decoding.language}",
        f"- Bleed handling: {result.bleed_mode_applied}",
    ]
    if result.gaps:
        lines += [
            "",
            "> **Recording gaps:** "
            + "; ".join(
                f"{format_clock(g.start)} ({g.length / TIMELINE_RATE:.1f} s, {g.reason}"
                + (f", {result.name_of(g.participant)})" if g.participant else ")")
                for g in result.gaps
            ),
        ]
    lines.append("")
    entries: list[tuple[int, str]] = []
    for group in _groups(result):
        marks = ""
        if any(s.overlap for s in group):
            marks += f" {OVERLAP_MARK}"
        if any(s.quality_flags for s in group):
            marks += f" {FLAG_MARK}"
        clock = format_clock(group[0].start, millis=millis)
        who = result.name_of(group[0].speaker)
        entries.append((group[0].start, f"**[{clock}] {who}:**{marks} {_text(group)}  "))
    for m in result.transcript.missed:
        clock = format_clock(m.start, millis=millis)
        seconds = (m.end - m.start) / TIMELINE_RATE
        entries.append(
            (
                m.start,
                f"**[{clock}] {result.name_of(m.speaker)}:** {FLAG_MARK} "
                f"_(possible missed speech, {seconds:.1f} s)_  ",
            )
        )
    lines += [text for _, text in sorted(entries, key=lambda e: e[0])]
    return "\n".join(lines).rstrip() + "\n"


def _segment_json(s: TranscriptSegment) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": s.id,
        "speaker": s.speaker,
        "source_chunk_id": s.chunk_id,
        "start": samples_to_seconds(s.start),
        "end": samples_to_seconds(s.end),
        "audio_start_sample": s.track_start,
        "text": s.text,
        "avg_logprob": round(s.avg_logprob, 4),
        "no_speech_prob": round(s.no_speech_prob, 4),
        "compression_ratio": round(s.compression_ratio, 3),
        "temperature": s.temperature,
        "ownership": s.ownership,
        "overlap": s.overlap,
        "quality_flags": list(s.quality_flags),
        "boundary_trimmed_words": s.trimmed_words,
        "words": [
            {
                "start": samples_to_seconds(w.start),
                "end": samples_to_seconds(w.end),
                "word": w.text,
                "prob": round(w.probability, 4),
            }
            for w in s.words
        ],
    }
    if s.suppression_reason:
        out["suppression_reason"] = s.suppression_reason
        out["duplicate_of"] = s.duplicate_of
    return out


def to_json(result: SessionResult) -> dict[str, Any]:
    model = result.model
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "app_version": __version__,
        "processing_version": PROCESSING_VERSION,
        "libraries": _library_versions(),
        "model": {
            "id": model.get("model_id", result.settings.model.id),
            "revision": model.get("revision"),
            "device": model.get("device"),
            "compute_type": model.get("compute_type"),
            "asr_mode": result.settings.asr.mode,
            "batch_sizes": sorted(set(result.batch_sizes)),
        },
        "session": {
            "name": result.name,
            "date": result.date,
            "duration_s": samples_to_seconds(result.duration),
            "timeline_rate": TIMELINE_RATE,
            "languages": result.languages,
        },
        "settings": result.settings.model_dump(mode="json"),
        "bleed_mode_applied": result.bleed_mode_applied,
        "participants": [
            {"id": p.id, "name": p.display_name, "source": p.source, "offset_s": p.offset_s}
            | ({"channel": p.channel} if p.channel else {})
            for p in result.participants
        ],
        "coupling_db": None,
        "recording_gaps": [
            {
                "start_sample": g.start,
                "length_samples": g.length,
                "reason": g.reason,
                "participant": g.participant,
            }
            for g in result.gaps
        ],
        "chunks": [
            {
                "id": f"{c.participant}-c{c.index:03d}",
                "owned": [samples_to_seconds(c.owned[0]), samples_to_seconds(c.owned[1])],
                "context": [samples_to_seconds(c.context[0]), samples_to_seconds(c.context[1])],
                "cache_key": c.cache_key,
                "cached": c.cached,
                "transcribe_s": round(c.transcribe_s, 3),
                "segments": c.segments,
            }
            for c in result.chunks
        ],
        "vad_intervals": {
            pid: None
            if vad is None
            else [[samples_to_seconds(v.start), samples_to_seconds(v.end)] for v in vad]
            for pid, vad in result.vad.items()
        },
        "processing_events": result.processing_events,
        "segments": [_segment_json(s) for s in result.transcript.segments],
        "suppressed": [_segment_json(s) for s in result.transcript.suppressed],
        "possible_missed_speech": [
            {
                "id": m.id,
                "speaker": m.speaker,
                "start": samples_to_seconds(m.start),
                "end": samples_to_seconds(m.end),
                "audio_start_sample": m.track_start,
                "quality_flags": list(m.quality_flags),
            }
            for m in result.transcript.missed
        ],
    }


TEXT_EXPORTERS: dict[str, Callable[[SessionResult], str]] = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "md": to_md,
}
FORMATS = ("json", *TEXT_EXPORTERS)


def output_stem(result: SessionResult) -> str:
    stem = result.settings.output.filename_template.format(stem=result.name, date=result.date or "")
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" .") or "session"


def write_session_outputs(
    result: SessionResult, out_dir: Path, formats: tuple[str, ...] | None = None
) -> list[Path]:
    formats = formats or tuple(result.settings.output.formats)
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown format(s) {', '.join(unknown)}; expected {', '.join(FORMATS)}")
    stem = output_stem(result)
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "json":
            atomic_write_json(path, to_json(result))
        else:
            atomic_write_text(path, TEXT_EXPORTERS[fmt](result))
        written.append(path)
    return written
