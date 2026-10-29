"""Track readers. Raw audio is only ever read, never modified."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from sessionscribe.core.audio.resample import StreamResampler
from sessionscribe.core.timeline import WHISPER_RATE

READ_BLOCK_FRAMES = 1 << 16
# M1 loads a whole track at 16 kHz (about 230 MB per hour); chunking arrives in M2.
DEFAULT_MAX_DURATION_S = 2 * 3600.0


class AudioReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrackInfo:
    path: Path
    samplerate: int
    channels: int
    frames: int
    format: str
    subtype: str

    @property
    def duration_s(self) -> float:
        return self.frames / self.samplerate


def track_info(path: Path) -> TrackInfo:
    try:
        info = sf.info(str(path))
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise AudioReadError(f"cannot read {path}: {exc}") from exc
    return TrackInfo(
        Path(path),
        int(info.samplerate),
        int(info.channels),
        int(info.frames),
        info.format,
        info.subtype,
    )


def _channel_index(info: TrackInfo, channel: int | None) -> int:
    """0-based column for a 1-based ``channel``; multi-channel files need one."""
    if channel is None:
        if info.channels != 1:
            raise AudioReadError(
                f"{info.path.name} has {info.channels} channels; choose one with --channel"
            )
        return 0
    if not 1 <= channel <= info.channels:
        raise AudioReadError(f"{info.path.name} has no channel {channel} (1..{info.channels})")
    return channel - 1


def read_blocks(
    path: Path, channel: int | None = None, block_frames: int = READ_BLOCK_FRAMES
) -> Iterator[np.ndarray]:
    """Mono float32 blocks at the file's own sample rate."""
    info = track_info(path)
    column = _channel_index(info, channel)
    with sf.SoundFile(str(path)) as fh:
        for block in fh.blocks(blocksize=block_frames, dtype="float32", always_2d=True):
            yield np.ascontiguousarray(block[:, column])


def load_for_asr(
    path: Path,
    channel: int | None = None,
    max_duration_s: float | None = DEFAULT_MAX_DURATION_S,
) -> tuple[np.ndarray, TrackInfo]:
    """Whole track as mono float32 at 16 kHz, resampled once."""
    info = track_info(path)
    _channel_index(info, channel)
    if max_duration_s is not None and info.duration_s > max_duration_s:
        raise AudioReadError(
            f"{info.path.name} is {info.duration_s / 3600:.1f} h long; single-file mode "
            f"loads at most {max_duration_s / 3600:.1f} h (raise --max-duration)"
        )
    rs = StreamResampler(info.samplerate, WHISPER_RATE)
    expected = int(np.ceil(info.frames * WHISPER_RATE / info.samplerate)) + 16
    out = np.empty(expected, dtype=np.float32)
    filled = 0
    for block in read_blocks(path, channel):
        out, filled = _append(out, filled, rs.process(block))
    out, filled = _append(out, filled, rs.flush())
    return out[:filled], info


def _append(buffer: np.ndarray, filled: int, part: np.ndarray) -> tuple[np.ndarray, int]:
    end = filled + part.size
    if end > buffer.size:
        grown = np.empty(max(end, buffer.size * 2), dtype=buffer.dtype)
        grown[:filled] = buffer[:filled]
        buffer = grown
    buffer[filled:end] = part
    return buffer, end
