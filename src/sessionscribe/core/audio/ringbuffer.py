"""Preallocated, bounded single-producer/single-consumer ring buffer for audio frames.

The producer is the PortAudio callback: :meth:`RingBuffer.write` never allocates array
memory, never blocks and drops (and counts) a block that does not fit. The consumer is a
writer thread calling :meth:`RingBuffer.read`. Positions are monotonically increasing
frame counters; each side only advances its own counter, after the data copy is done.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class RingBuffer:
    def __init__(self, capacity_frames: int, channels: int, dtype: npt.DTypeLike = np.int32):
        if capacity_frames <= 0 or channels <= 0:
            raise ValueError("capacity and channel count must be positive")
        self._buf = np.zeros((capacity_frames, channels), dtype=dtype)
        self._capacity = capacity_frames
        self._write_pos = 0
        self._read_pos = 0
        self.dropped_blocks = 0
        self.dropped_frames = 0
        self.max_fill_frames = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return int(self._buf.shape[1])

    def available(self) -> int:
        """Frames written but not yet read."""
        return self._write_pos - self._read_pos

    def write(self, block: np.ndarray, columns: np.ndarray | None = None) -> bool:
        """Copy ``block`` (frames x channels) in; select ``columns`` of it if given.

        Returns False, and counts the drop, when the block does not fit.
        """
        frames = block.shape[0]
        write_pos = self._write_pos
        fill = write_pos - self._read_pos
        if frames > self._capacity - fill:
            self.dropped_blocks += 1
            self.dropped_frames += frames
            return False
        start = write_pos % self._capacity
        first = min(frames, self._capacity - start)
        self._copy(block[:first], start, columns)
        if first < frames:
            self._copy(block[first:], 0, columns)
        self._write_pos = write_pos + frames
        if fill + frames > self.max_fill_frames:
            self.max_fill_frames = fill + frames
        return True

    def _copy(self, src: np.ndarray, start: int, columns: np.ndarray | None) -> None:
        dst = self._buf[start : start + src.shape[0]]
        if columns is None:
            np.copyto(dst, src, casting="same_kind")
        else:
            # mode="clip" avoids the internal buffering that mode="raise" does with out=.
            np.take(src, columns, axis=1, out=dst, mode="clip")

    def read(self, max_frames: int | None = None) -> np.ndarray:
        """Return a copy of up to ``max_frames`` buffered frames (possibly empty)."""
        read_pos = self._read_pos
        frames = self._write_pos - read_pos
        if max_frames is not None:
            frames = min(frames, max_frames)
        start = read_pos % self._capacity
        first = min(frames, self._capacity - start)
        if first == frames:
            out = self._buf[start : start + frames].copy()
        else:
            out = np.concatenate((self._buf[start:], self._buf[: frames - first]))
        self._read_pos = read_pos + frames
        return out
