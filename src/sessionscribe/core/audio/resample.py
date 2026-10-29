"""Streaming resampler on PyAV (libswresample), see DECISIONS M1.

Block-invariant: feeding the same signal in different block sizes gives identical
output, and the output is time-aligned with the input (no filter delay) once
:meth:`StreamResampler.flush` has been called.
"""

from __future__ import annotations

import av
import numpy as np


class StreamResampler:
    """Mono float32 resampler; call :meth:`process` per block, then :meth:`flush` once."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._passthrough = in_rate == out_rate
        self._resampler = (
            None
            if self._passthrough
            else av.AudioResampler(format="flt", layout="mono", rate=out_rate)
        )

    def process(self, block: np.ndarray) -> np.ndarray:
        samples = np.ascontiguousarray(block, dtype=np.float32).reshape(-1)
        if self._passthrough or samples.size == 0:
            return samples.copy()
        frame = av.AudioFrame.from_ndarray(samples[None, :], format="flt", layout="mono")
        frame.sample_rate = self.in_rate
        assert self._resampler is not None
        return self._collect(self._resampler.resample(frame))

    def flush(self) -> np.ndarray:
        if self._passthrough:
            return np.zeros(0, dtype=np.float32)
        assert self._resampler is not None
        return self._collect(self._resampler.resample(None))

    @staticmethod
    def _collect(frames: list[av.AudioFrame]) -> np.ndarray:
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]).astype(np.float32)


def resample(signal: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    rs = StreamResampler(in_rate, out_rate)
    return np.concatenate([rs.process(signal), rs.flush()])
