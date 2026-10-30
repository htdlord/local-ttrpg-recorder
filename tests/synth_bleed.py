"""Synthetic levels and audio with known coupling, for bleed tests.

Each participant's voice reaches mic i with an acoustic attenuation ``acoustic[i][j]``
(0 on the diagonal); each mic adds its own gain and noise. The coupling the pipeline
should find is ``C[i][j] = acoustic[i][j] + gain_i - gain_j``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FPS = 50  # 20 ms frames


def db_to_power(db):
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def power_to_db(p):
    return (10.0 * np.log10(np.maximum(p, 1e-12))).astype(np.float32)


def levels(
    speech: Sequence[Sequence[tuple[float, float]]],
    acoustic: np.ndarray,
    gains: Sequence[float],
    noise: Sequence[float],
    seconds: float,
    voice_db: float | Sequence[float] = -20.0,
    rng_seed: int = 0,
) -> np.ndarray:
    """dB envelopes (n x frames): speech[j] = [(start_s, end_s), ...] of participant j."""
    rng = np.random.default_rng(rng_seed)
    n, frames = len(gains), int(seconds * FPS)
    voices = np.zeros((n, frames))
    per = [voice_db] * n if isinstance(voice_db, (int, float)) else list(voice_db)
    for j, spans in enumerate(speech):
        for a, b in spans:
            jitter = rng.normal(0, 1.5, int(b * FPS) - int(a * FPS))
            voices[j, int(a * FPS) : int(b * FPS)] = db_to_power(per[j] + jitter)
    out = np.zeros((n, frames))
    for i in range(n):
        heard = sum(voices[j] * db_to_power(acoustic[i][j]) for j in range(n))
        out[i] = heard * db_to_power(gains[i]) + db_to_power(noise[i] + rng.normal(0, 1, frames))
    return power_to_db(out)


def expected_coupling(acoustic: np.ndarray, gains: Sequence[float]) -> np.ndarray:
    g = np.asarray(gains, dtype=np.float64)
    c = np.asarray(acoustic, dtype=np.float64) + g[:, None] - g[None, :]
    np.fill_diagonal(c, 0.0)
    return c


def acoustic(n: int, bleed_db: float = -18.0) -> np.ndarray:
    a = np.full((n, n), bleed_db)
    np.fill_diagonal(a, 0.0)
    return a
