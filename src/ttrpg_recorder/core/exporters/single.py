"""Single-track outputs: seconds appear here for the first and only time (rounded once)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ttrpg_recorder import __version__
from ttrpg_recorder.core.config import Settings
from ttrpg_recorder.core.fileio import atomic_write_json, atomic_write_text
from ttrpg_recorder.core.pipeline import PROCESSING_VERSION
from ttrpg_recorder.core.pipeline.asr import Segment, TranscriptionResult
from ttrpg_recorder.core.timeline import TIMELINE_RATE, format_clock, samples_to_seconds

JSON_SCHEMA_VERSION = 1
SUPPORTED_FORMATS = ("json", "txt")


@dataclass(frozen=True)
class Speaker:
    id: str
    name: str
    source: str


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("faster-whisper", "ctranslate2", "av"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def segment_id(speaker: str, seg: Segment, chunk: int = 0) -> str:
    return f"{speaker}-c{chunk:03d}-{seg.index:04d}"


def to_json(result: TranscriptionResult, settings: Settings, speaker: Speaker) -> dict[str, Any]:
    def seg_json(seg: Segment) -> dict[str, Any]:
        return {
            "id": segment_id(speaker.id, seg),
            "speaker": speaker.id,
            "source_chunk_id": f"{speaker.id}-c000",
            "start": samples_to_seconds(seg.start),
            "end": samples_to_seconds(seg.end),
            "audio_start_sample": seg.start,
            "text": seg.text,
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "compression_ratio": round(seg.compression_ratio, 3),
            "temperature": seg.temperature,
            "quality_flags": [],
            "words": [
                {
                    "start": samples_to_seconds(w.start),
                    "end": samples_to_seconds(w.end),
                    "word": w.text,
                    "prob": round(w.probability, 4),
                }
                for w in seg.words
            ],
        }

    m = result.model
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "app_version": __version__,
        "processing_version": PROCESSING_VERSION,
        "libraries": _library_versions(),
        "model": {
            "id": m.model_id,
            "revision": m.revision,
            "device": m.device,
            "compute_type": m.compute_type,
            "asr_mode": result.mode,
            "batch_size": result.batch_size,
        },
        "session": {
            "name": Path(speaker.source).stem,
            "duration_s": samples_to_seconds(result.duration),
            "timeline_rate": TIMELINE_RATE,
            "language": result.language,
            "language_probability": (
                None
                if result.language_probability is None
                else round(result.language_probability, 3)
            ),
        },
        "settings": settings.model_dump(mode="json"),
        "participants": [
            {"id": speaker.id, "name": speaker.name, "source": speaker.source, "offset_s": 0.0}
        ],
        "recording_gaps": [],
        "vad_intervals": [
            {"start": samples_to_seconds(v.start), "end": samples_to_seconds(v.end)}
            for v in result.vad_intervals
        ],
        "segments": [seg_json(s) for s in result.segments],
        "suppressed": [],
    }


def to_txt(result: TranscriptionResult, speaker: Speaker, *, millis: bool = False) -> str:
    lines = [
        f"[{format_clock(seg.start, millis=millis)}] {speaker.name}: {seg.text}"
        for seg in result.segments
        if seg.text
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_outputs(
    result: TranscriptionResult,
    settings: Settings,
    speaker: Speaker,
    out_dir: Path,
    stem: str,
    formats: tuple[str, ...],
) -> list[Path]:
    unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
    if unknown:
        raise ValueError(
            f"format(s) {', '.join(unknown)} not available yet; single-file mode writes "
            f"{', '.join(SUPPORTED_FORMATS)}"
        )
    written: list[Path] = []
    if "json" in formats:
        path = out_dir / f"{stem}.json"
        atomic_write_json(path, to_json(result, settings, speaker))
        written.append(path)
    if "txt" in formats:
        path = out_dir / f"{stem}.txt"
        millis = settings.output.timestamp_style == "hh:mm:ss.mmm"
        atomic_write_text(path, to_txt(result, speaker, millis=millis))
        written.append(path)
    return written
