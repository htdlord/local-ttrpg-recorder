"""Audio for previewing a passage (no Qt): read from the original track, int16 PCM."""

from __future__ import annotations

import numpy as np

from sessionscribe.core.session import Session
from sessionscribe.core.timeline import TIMELINE_RATE, rescale, seconds_to_samples

CONTEXT_S = 0.3


def passage(
    session: Session, participant: str, start: int, end: int, context_s: float = CONTEXT_S
) -> tuple[np.ndarray, int]:
    """Session samples ``[start, end)`` of a participant's track, with context, as int16."""
    p = next(x for x in session.participants if x.id == participant)
    track = session.open_track(p)
    pad = seconds_to_samples(context_s)
    a = max(0, start - pad - p.offset_samples)
    b = max(a, end + pad - p.offset_samples)
    rate = track.samplerate
    frames = track.read(rescale(a, TIMELINE_RATE, rate), rescale(b - a, TIMELINE_RATE, rate))
    pcm = np.clip(frames * 32767.0, -32768, 32767).astype(np.int16)
    return pcm, rate
