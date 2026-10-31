"""Check a recording against its manifest, and recover after a crash (SPEC §8.2).

``verify`` compares every manifest segment with the file on disk (existence, frame
count, SHA-256 for closed segments) and lists audio files the manifest does not know.
``recover`` handles segments left ``open`` by a crash or kill: whatever is decodable
is copied into a clean file of the same name (the original is kept as ``.partial``),
the channels of one segment are cut to the same length so they stay aligned, and the
segment becomes ``recovered`` (or ``damaged`` if nothing was readable). Raw audio is
only ever read; the original bytes stay on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ttrpg_recorder.core.audio.manifest import RecordingManifest, SegmentRecord, file_sha256
from ttrpg_recorder.core.audio.track import AUDIO_SUFFIXES
from ttrpg_recorder.core.session import Session, write_session

PARTIAL_SUFFIX = ".partial"


@dataclass
class SegmentCheck:
    file: str
    status: str
    ok: bool
    problem: str = ""


@dataclass
class VerifyReport:
    segments: list[SegmentCheck] = field(default_factory=list)
    unknown_files: list[str] = field(default_factory=list)
    frames_per_channel: dict[int, int] = field(default_factory=dict)
    gaps: int = 0
    gap_seconds: float = 0.0
    state: str = ""

    @property
    def ok(self) -> bool:
        aligned = len(set(self.frames_per_channel.values())) <= 1
        return all(s.ok for s in self.segments) and not self.unknown_files and aligned


def _readable_frames(path: Path) -> int:
    """Frames that decode, even from a file whose header was never finalised."""
    try:
        with sf.SoundFile(str(path)) as fh:
            total = 0
            while True:
                block = fh.read(1 << 16, dtype="int32")
                if not len(block):
                    break
                total += len(block)
            return total
    except (RuntimeError, sf.LibsndfileError):
        return 0


def verify(session_dir: Path) -> VerifyReport:
    session = Session.load(session_dir)
    rec = session.config.recording
    if rec is None:
        raise ValueError(f"{session_dir} has no [recording] manifest")
    report = VerifyReport(state=rec.state, gaps=len(rec.gaps))
    report.gap_seconds = round(sum(g.length_samples for g in rec.gaps) / rec.samplerate, 3)
    known: set[Path] = set()
    for seg in rec.segments:
        path = session.folder / seg.file
        known.add(path.resolve())
        if not path.is_file():
            partial = path.with_name(path.name + PARTIAL_SUFFIX)
            problem = (
                "no readable audio (killed right after it was opened); original kept as "
                f"{partial.name}"
                if seg.status == "damaged" and partial.is_file()
                else "missing"
            )
            report.segments.append(SegmentCheck(seg.file, seg.status, False, problem))
            continue
        if seg.status == "open":
            report.segments.append(
                SegmentCheck(seg.file, seg.status, False, "still open: run 'record recover'")
            )
            continue
        frames = sf.info(str(path)).frames
        if frames != seg.frames:
            report.segments.append(
                SegmentCheck(seg.file, seg.status, False, f"{frames} frames, manifest {seg.frames}")
            )
            continue
        if seg.sha256 and file_sha256(path) != seg.sha256:
            report.segments.append(SegmentCheck(seg.file, seg.status, False, "hash differs"))
            continue
        ok = seg.status != "damaged"
        report.segments.append(SegmentCheck(seg.file, seg.status, ok, "" if ok else "damaged"))
    rec_dir = session.folder / "rec"
    if rec_dir.is_dir():
        for p in sorted(rec_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES and p.resolve() not in known:
                report.unknown_files.append(p.relative_to(session.folder).as_posix())
    for ch in rec.channels:
        report.frames_per_channel[ch] = sum(s.frames for s in rec.segments_for(ch))
    return report


def recover(session_dir: Path) -> RecordingManifest:
    """Turn ``open`` segments into readable ``recovered`` ones; mark the run interrupted."""
    session = Session.load(session_dir)
    rec = session.config.recording
    if rec is None:
        raise ValueError(f"{session_dir} has no [recording] manifest")
    by_index: dict[int, list[SegmentRecord]] = {}
    for seg in rec.segments:
        if seg.status == "open":
            by_index.setdefault(seg.index, []).append(seg)
    for segs in by_index.values():
        frames = min(
            _readable_frames(session.folder / s.file) if (session.folder / s.file).is_file() else 0
            for s in segs
        )
        for seg in segs:
            path = session.folder / seg.file
            if not path.is_file():
                seg.status, seg.frames = "damaged", 0
                continue
            partial = path.with_name(path.name + PARTIAL_SUFFIX)
            path.replace(partial)
            if frames == 0:
                seg.status, seg.frames, seg.sha256 = "damaged", 0, None
                continue
            with sf.SoundFile(str(partial)) as src:
                data = src.read(frames, dtype="int32")
                rate, subtype, fmt = src.samplerate, src.subtype, src.format
            sf.write(str(path), np.ascontiguousarray(data), rate, subtype=subtype, format=fmt)
            seg.frames, seg.status, seg.sha256 = len(data), "recovered", file_sha256(path)
    rec.state = "interrupted"
    write_session(session.folder / "session.toml", session.config)
    return rec
