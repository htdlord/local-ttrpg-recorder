"""Virtual tracks: one file, or a folder of consecutive segment files read as one track.

Positions here are frames at the track's own sample rate; callers convert to the session
timeline with :func:`sessionscribe.core.timeline.rescale`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from sessionscribe.core.audio.reader import (
    READ_BLOCK_FRAMES,
    AudioReadError,
    TrackInfo,
    _channel_index,
    track_info,
)

AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".rf64", ".w64"})
FINGERPRINT_EDGE_BYTES = 1 << 20


def audio_files(folder: Path) -> list[Path]:
    """Audio files of a segment folder in name order (recorder names sort chronologically)."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES),
        key=lambda p: p.name,
    )


def file_fingerprint(path: Path, *, full: bool = False) -> str:
    """Size + SHA-256 of the first and last MB (or of the whole file).

    No mtime: copying a session folder must not invalidate its cache (DECISIONS).
    """
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        if full or size <= 2 * FINGERPRINT_EDGE_BYTES:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        else:
            h.update(fh.read(FINGERPRINT_EDGE_BYTES))
            fh.seek(size - FINGERPRINT_EDGE_BYTES)
            h.update(fh.read(FINGERPRINT_EDGE_BYTES))
    return h.hexdigest()


@dataclass(frozen=True)
class TrackPart:
    info: TrackInfo
    start: int  # first frame of this file within the virtual track

    @property
    def end(self) -> int:
        return self.start + self.info.frames


class VirtualTrack:
    """Mono view of one channel across one or more files with equal format."""

    def __init__(self, parts: list[TrackPart], channel: int | None, *, full_hash: bool = False):
        self.parts = parts
        self.channel = channel
        self.full_hash = full_hash
        self._column = _channel_index(parts[0].info, channel)
        self._fingerprints: dict[Path, str] = {}

    @classmethod
    def open(
        cls, source: Path, channel: int | None = None, *, full_hash: bool = False
    ) -> VirtualTrack:
        source = Path(source)
        if source.is_dir():
            files = audio_files(source)
            if not files:
                raise AudioReadError(f"no audio files in {source}")
        elif source.is_file():
            files = [source]
        else:
            raise AudioReadError(f"{source} does not exist")
        parts: list[TrackPart] = []
        pos = 0
        for path in files:
            info = track_info(path)
            if parts:
                first = parts[0].info
                if (info.samplerate, info.channels) != (first.samplerate, first.channels):
                    raise AudioReadError(
                        f"{path.name}: {info.samplerate} Hz / {info.channels} ch differs from "
                        f"{first.path.name} ({first.samplerate} Hz / {first.channels} ch)"
                    )
            parts.append(TrackPart(info, pos))
            pos += info.frames
        return cls(parts, channel, full_hash=full_hash)

    @property
    def samplerate(self) -> int:
        return self.parts[0].info.samplerate

    @property
    def frames(self) -> int:
        return self.parts[-1].end

    @property
    def files(self) -> list[Path]:
        return [p.info.path for p in self.parts]

    def parts_in(self, start: int, frames: int) -> list[TrackPart]:
        end = start + frames
        return [p for p in self.parts if p.start < end and p.end > start]

    def blocks(
        self, start: int = 0, frames: int | None = None, block_frames: int = READ_BLOCK_FRAMES
    ) -> Iterator[np.ndarray]:
        """Mono float32 blocks covering ``[start, start + frames)``, clipped to the track."""
        start = max(0, start)
        end = self.frames if frames is None else min(self.frames, start + frames)
        for part in self.parts_in(start, end - start):
            lo = max(start, part.start) - part.start
            hi = min(end, part.end) - part.start
            with sf.SoundFile(str(part.info.path)) as fh:
                fh.seek(lo)
                left = hi - lo
                while left > 0:
                    block = fh.read(min(block_frames, left), dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    left -= len(block)
                    yield np.ascontiguousarray(block[:, self._column])

    def read(self, start: int, frames: int) -> np.ndarray:
        pieces = list(self.blocks(start, frames))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def fingerprint(self, start: int = 0, frames: int | None = None) -> str:
        """Content fingerprint of the files a range touches, with their position."""
        frames = self.frames - start if frames is None else frames
        h = hashlib.sha256(f"ch{self.channel}".encode())
        for part in self.parts_in(start, frames):
            fp = self._fingerprints.get(part.info.path)
            if fp is None:
                fp = file_fingerprint(part.info.path, full=self.full_hash)
                self._fingerprints[part.info.path] = fp
            h.update(f"{part.start}:{fp};".encode())
        return h.hexdigest()
