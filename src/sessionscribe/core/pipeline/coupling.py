"""Session frame grid and coupling between microphones (SPEC §5.2).

``C[i][j]`` (dB) is how loud participant j's voice typically is on mic i relative to
mic j: bleed of j on track i is predicted at ``E_j + C[i][j]``. Levels are per 20 ms
frame (dBFS envelopes from ``features``) on the session grid; offsets shift a track by
whole frames. Mic gains differ by 15-20 dB in practice, so every per-track threshold is
relative to that track's own speech or noise level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sessionscribe.core.pipeline.features import SILENCE_DB

ACTIVE_DB = -60.0  # frames above this count when estimating a track's speech level
NOISE_ACTIVE_DB = -100.0  # frames above this (not digital silence) count for the noise level
DOMINANT_REL_DB = -6.0  # speaking: within this of its own speech level
ACTIVE_ABOVE_NOISE_DB = 15.0  # mic j "active": this far above its own noise level
PEAK_MIN_FRAMES = 50  # a histogram peak needs at least 1 s of frames
PEAK_SMOOTH_BINS = 5
PAIR_SUM_MAX_DB = -20.0  # C[i][j] + C[j][i]: both acoustic paths, provisional
GUIDED_REL_DB = -10.0  # mic check frames this close to the speaker's speech level count

CouplingSource = Literal["auto", "guided", "manual", "default", "clamped"]


def speech_level(env: np.ndarray) -> float:
    """A track's typical own-speech level: 95th percentile of its active frames."""
    active = env[env > ACTIVE_DB]
    if active.size == 0:
        return float(env.max()) if env.size else SILENCE_DB
    return float(np.percentile(active, 95))


def noise_level(env: np.ndarray) -> float:
    """A track's noise floor: 10th percentile of frames that are not digital silence."""
    active = env[env > NOISE_ACTIVE_DB]
    if active.size == 0:
        return SILENCE_DB
    return float(np.percentile(active, 10))


def align(envelopes: Sequence[np.ndarray], shifts: Sequence[int]) -> np.ndarray:
    """Stack envelopes on the session grid: shift each by whole frames, pad with silence."""
    length = max((len(e) + s for e, s in zip(envelopes, shifts, strict=True)), default=0)
    out = np.full((len(envelopes), length), SILENCE_DB, dtype=np.float32)
    for row, (env, shift) in enumerate(zip(envelopes, shifts, strict=True)):
        out[row, shift : shift + len(env)] = env
    return out


@dataclass
class Coupling:
    speakers: list[str]
    matrix: np.ndarray  # n x n, dB; diagonal 0
    frames: np.ndarray  # n x n, frames each value is based on (0 = default)
    source: dict[tuple[int, int], CouplingSource] = field(default_factory=dict)

    def to_json(self) -> list[list[float]]:
        return [[round(float(v), 1) for v in row] for row in self.matrix]

    def weak_pairs(self) -> list[tuple[str, str]]:
        """(mic, voice) pairs that use the default or a clamped value."""
        return [
            (self.speakers[i], self.speakers[j])
            for (i, j), src in sorted(self.source.items())
            if src in ("default", "clamped")
        ]


def _enforce_pair_sum(
    matrix: np.ndarray, source: dict[tuple[int, int], CouplingSource] | None = None
) -> None:
    """``C[i][j] + C[j][i]`` is the two acoustic paths (gains cancel), so clearly negative.

    A participant who hardly speaks can make the estimate of their voice pick up the
    reverse path (about ``-C[j][i]``); clamp the larger value of each pair so the sum
    stays at most ``PAIR_SUM_MAX_DB``, and mark it.
    """
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] + matrix[j, i] > PAIR_SUM_MAX_DB:
                hi, lo = ((i, j), (j, i)) if matrix[i, j] >= matrix[j, i] else ((j, i), (i, j))
                matrix[hi] = PAIR_SUM_MAX_DB - matrix[lo]
                if source is not None:
                    source[hi] = "clamped"


def lowest_peak(diff: np.ndarray, min_mass: int) -> float | None:
    """Lowest smoothed histogram peak (1 dB bins) holding at least ``min_mass`` frames.

    Returns the frame-weighted mean of the raw values around that peak.
    """
    edges = np.arange(-80.0, 81.0, 1.0)
    hist, _ = np.histogram(diff, bins=edges)
    half = PEAK_SMOOTH_BINS // 2
    smooth = np.convolve(hist, np.ones(PEAK_SMOOTH_BINS), mode="same")
    k = 1
    while k < len(smooth) - 1:
        if smooth[k] > smooth[k - 1]:
            end = k
            while end + 1 < len(smooth) and smooth[end + 1] == smooth[k]:
                end += 1
            centre = (k + end) // 2
            if (end + 1 >= len(smooth) or smooth[end + 1] < smooth[k]) and smooth[
                centre
            ] >= min_mass:
                lo, hi = edges[max(0, centre - half)], edges[min(len(edges) - 1, centre + half + 1)]
                near = diff[(diff >= lo) & (diff < hi)]
                return float(near.mean()) if near.size else float(edges[centre] + 0.5)
            k = end + 1
        else:
            k += 1
    return None


def estimate_coupling(
    speakers: Sequence[str],
    levels: np.ndarray,
    *,
    min_frames: int,
    default_db: float,
) -> Coupling:
    """Per ordered pair (mic i, voice j): the lowest peak of ``E_i - E_j`` while j is active.

    On frames where mic j is clearly above its noise, ``E_i - E_j`` clusters at
    ``C[i][j]`` (j speaking), at ``-C[j][i]`` (i speaking, heard on j) and at third-party
    values in between; the lowest significant cluster is j's voice. Measured on clip01,
    where a level-relative "j is speaking" test failed for low-gain and silent mics.
    Pairs with too few active frames or no clear peak use ``default_db``; pair sums are
    then kept physically plausible.
    """
    n = len(speakers)
    matrix = np.zeros((n, n), dtype=np.float64)
    frames = np.zeros((n, n), dtype=np.int64)
    source: dict[tuple[int, int], CouplingSource] = {}
    for j in range(n):
        active = levels[j] >= noise_level(levels[j]) + ACTIVE_ABOVE_NOISE_DB
        count = int(active.sum())
        for i in range(n):
            if i == j:
                continue
            frames[i, j] = count
            peak = (
                lowest_peak(levels[i, active] - levels[j, active], PEAK_MIN_FRAMES)
                if count >= min_frames
                else None
            )
            if peak is None:
                matrix[i, j] = default_db
                source[(i, j)] = "default"
            else:
                matrix[i, j] = peak
                source[(i, j)] = "auto"
    _enforce_pair_sum(matrix, source)
    return Coupling(list(speakers), matrix, frames, source)


def apply_guided(
    coupling: Coupling,
    levels: np.ndarray,
    ranges: Mapping[str, tuple[int, int]],
    *,
    min_frames: int,
) -> Coupling:
    """Mic check: in each participant's frame range only they speak; overrides auto."""
    rel = levels - np.array([[speech_level(row)] for row in levels], dtype=np.float32)
    for speaker, (a, b) in ranges.items():
        if speaker not in coupling.speakers:
            continue
        j = coupling.speakers.index(speaker)
        m = np.zeros(levels.shape[1], dtype=bool)
        m[max(0, a) : max(0, b)] = True
        m &= rel[j] >= GUIDED_REL_DB
        count = int(m.sum())
        if count < max(1, min_frames // 5):  # a mic check is short: 1/5 of the auto minimum
            continue
        for i in range(len(coupling.speakers)):
            if i != j:
                coupling.matrix[i, j] = float(np.median(levels[i, m] - levels[j, m]))
                coupling.frames[i, j] = count
                coupling.source[(i, j)] = "guided"
    return coupling


def manual_coupling(speakers: Sequence[str], matrix: Sequence[Sequence[float]]) -> Coupling:
    n = len(speakers)
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (n, n):
        raise ValueError(f"coupling_db must be {n} x {n} (participant order), got {arr.shape}")
    np.fill_diagonal(arr, 0.0)
    source: dict[tuple[int, int], CouplingSource] = {
        (i, j): "manual" for i in range(n) for j in range(n) if i != j
    }
    return Coupling(list(speakers), arr, np.zeros((n, n), dtype=np.int64), source)
