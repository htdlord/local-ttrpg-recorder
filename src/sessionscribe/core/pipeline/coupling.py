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
DOMINANCE_MARGIN_DB = 6.0  # ... and this much above every other track (relative)
GUIDED_REL_DB = -10.0  # mic check frames this close to the speaker's speech level count

CouplingSource = Literal["auto", "guided", "manual", "default"]


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
        """(mic, voice) pairs that fell back to the default value."""
        return [
            (self.speakers[i], self.speakers[j])
            for (i, j), src in sorted(self.source.items())
            if src == "default"
        ]


def _dominant(levels: np.ndarray, rel: np.ndarray, j: int) -> np.ndarray:
    others = np.delete(rel, j, axis=0)
    loudest_other = others.max(axis=0) if others.size else np.full(rel.shape[1], -np.inf)
    return (rel[j] >= DOMINANT_REL_DB) & (rel[j] - loudest_other >= DOMINANCE_MARGIN_DB)


def estimate_coupling(
    speakers: Sequence[str],
    levels: np.ndarray,
    *,
    min_frames: int,
    default_db: float,
    margin_db: float = DOMINANCE_MARGIN_DB,
) -> Coupling:
    """Median level difference on frames where exactly one track clearly dominates.

    One refinement pass uses the first estimate to find frames that only the speaker
    owns (a close neighbour's strong bleed can hide dominance on relative levels alone).
    """
    n = len(speakers)
    rel = levels - np.array([[speech_level(row)] for row in levels], dtype=np.float32)
    matrix = np.zeros((n, n), dtype=np.float64)
    frames = np.zeros((n, n), dtype=np.int64)
    source: dict[tuple[int, int], CouplingSource] = {}

    def fill(masks: list[np.ndarray]) -> None:
        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                m = masks[j]
                count = int(m.sum())
                frames[i, j] = count
                if count >= min_frames:
                    matrix[i, j] = float(np.median(levels[i, m] - levels[j, m]))
                    source[(i, j)] = "auto"
                else:
                    matrix[i, j] = default_db
                    source[(i, j)] = "default"

    fill([_dominant(levels, rel, j) for j in range(n)])
    if n > 1:
        refined = []
        for j in range(n):
            # j speaks and owns the frame, and every other track is explained by j's bleed.
            m = rel[j] >= DOMINANT_REL_DB
            for k in range(n):
                if k != j:
                    m &= levels[j] >= levels[k] + matrix[j, k] + margin_db
                    m &= levels[k] < levels[j] + matrix[k, j] + margin_db
            refined.append(m)
        fill(refined)
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
