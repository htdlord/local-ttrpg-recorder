"""Energy envelopes: RMS level in dBFS per frame (default 20 ms), per track.

Used for chunk boundary placement (M2) and ownership masks (M3). Frame ``k`` covers
track-local timeline samples ``[k * frame, (k + 1) * frame)`` with
``frame = frame_ms * 48``. Envelopes are cached in fixed nominal blocks, independent of
where chunk boundaries end up.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sessionscribe.core.audio.track import VirtualTrack
from sessionscribe.core.cache import Cache, make_key
from sessionscribe.core.events import EventSink, ProgressEvent, null_sink
from sessionscribe.core.pipeline import PROCESSING_VERSION
from sessionscribe.core.timeline import TIMELINE_RATE

SILENCE_DB = -120.0
FEATURE_BLOCK_S = 600


def frame_samples(frame_ms: int, rate: int = TIMELINE_RATE) -> int:
    n = frame_ms * rate
    if n % 1000:
        raise ValueError(f"frame_ms={frame_ms} is not a whole number of samples at {rate} Hz")
    return n // 1000


def frames_db(samples: np.ndarray, frame: int) -> np.ndarray:
    """dBFS of each frame; a trailing partial frame uses the samples it has."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    n = -(-samples.size // frame)
    padded = np.zeros(n * frame, dtype=np.float64)
    padded[: samples.size] = samples
    sq = (padded.reshape(n, frame) ** 2).sum(axis=1)
    counts = np.full(n, frame, dtype=np.float64)
    counts[-1] = samples.size - (n - 1) * frame
    power = sq / counts
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(power)
    return np.maximum(db, SILENCE_DB).astype(np.float32)


class EnvelopeBuilder:
    """Streaming :func:`frames_db`: blocks of any size give the same result."""

    def __init__(self, frame: int) -> None:
        self.frame = frame
        self._rest = np.zeros(0, dtype=np.float32)
        self._parts: list[np.ndarray] = []

    def push(self, block: np.ndarray) -> None:
        data = np.concatenate([self._rest, block]) if self._rest.size else block
        whole = (data.size // self.frame) * self.frame
        if whole:
            self._parts.append(frames_db(data[:whole], self.frame))
        self._rest = np.array(data[whole:], dtype=np.float32)

    def finish(self) -> np.ndarray:
        if self._rest.size:
            self._parts.append(frames_db(self._rest, self.frame))
            self._rest = np.zeros(0, dtype=np.float32)
        return np.concatenate(self._parts) if self._parts else np.zeros(0, dtype=np.float32)


def envelope(blocks: Iterable[np.ndarray], frame: int) -> np.ndarray:
    builder = EnvelopeBuilder(frame)
    for block in blocks:
        builder.push(block)
    return builder.finish()


def track_envelope(
    track: VirtualTrack,
    cache: Cache | None,
    frame_ms: int,
    *,
    source_id: str = "",
    sink: EventSink = null_sink,
) -> np.ndarray:
    """Envelope of a whole track on the timeline frame grid, block by block from the cache."""
    frame = frame_samples(frame_ms, track.samplerate)
    frames_per_block = FEATURE_BLOCK_S * 1000 // frame_ms
    block = frames_per_block * frame
    parts: list[np.ndarray] = []
    for start in range(0, track.frames, block):
        length = min(block, track.frames - start)
        key = make_key(
            "features",
            {
                "fingerprint": track.fingerprint(start, length),
                "start": start,
                "frames": length,
                "samplerate": track.samplerate,
                "frame_ms": frame_ms,
                "processing_version": PROCESSING_VERSION,
            },
        )
        env = cache.load_array(key) if cache is not None else None
        if env is None:
            env = envelope(track.blocks(start, length), frame)
            if cache is not None:
                cache.store_array(key, env)
        parts.append(env)
        sink(ProgressEvent(f"features:{source_id}", start + length, track.frames, "frames"))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
