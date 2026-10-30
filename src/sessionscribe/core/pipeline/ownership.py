"""Ownership masks (SPEC §5.3) on the session frame grid, cached in their own layer.

Track i owns frame t when ``E_i >= floor_i`` and ``E_i >= max_j (E_j + C[i][j]) +
margin``: it is louder than the bleed predicted from everyone else, so a quiet speaker
next to a loud one still owns their frames. ``floor_i`` is relative to the track's own
noise level (mic gains differ by 15-20 dB).
Ownership lasts ``hold_ms`` after an owned frame. If two tracks own a frame, both
participants are speaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sessionscribe.core.cache import Cache, make_key
from sessionscribe.core.config import Settings
from sessionscribe.core.pipeline.coupling import (
    Coupling,
    align,
    apply_guided,
    estimate_coupling,
    manual_coupling,
    noise_level,
    speech_level,
)
from sessionscribe.core.pipeline.features import frame_samples, track_envelope
from sessionscribe.core.session import Session
from sessionscribe.core.timeline import seconds_to_samples

ATTRIBUTION_VERSION = 1  # bump when masks, attribution or dedup output changes


def track_floors(levels: np.ndarray, above_noise_db: float) -> np.ndarray:
    """Per-track floor: noise level + ``above_noise_db``.

    No cap from the speech level: on clip01 a silent, noisy track's "speech level" sat a
    few dB above its noise, and a cap below it let the track own almost every frame.
    """
    return np.array([noise_level(row) + above_noise_db for row in levels], dtype=np.float32)


def silent_mics(speakers: list[str], levels: np.ndarray, range_db: float) -> list[str]:
    """Tracks whose speech level is within ``range_db`` of their noise level."""
    return [
        s
        for s, row in zip(speakers, levels, strict=True)
        if speech_level(row) - noise_level(row) < range_db
    ]


def hold_forward(mask: np.ndarray, frames: int) -> np.ndarray:
    """``out[t]`` is true if ``mask`` was true in ``[t - frames, t]``."""
    if frames <= 0 or not mask.any():
        return mask.copy()
    m = mask.astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(m)])
    t = np.arange(mask.size)
    lo = np.maximum(0, t - frames)
    return (csum[t + 1] - csum[lo]) > 0


def ownership_masks(
    levels: np.ndarray,
    coupling: np.ndarray,
    floors: np.ndarray,
    margin_db: float,
    hold_frames: int = 0,
) -> np.ndarray:
    n = levels.shape[0]
    masks = np.zeros(levels.shape, dtype=bool)
    for i in range(n):
        own = levels[i] >= floors[i]
        if n > 1:
            predicted = np.max([levels[j] + coupling[i, j] for j in range(n) if j != i], axis=0)
            own &= levels[i] >= predicted + margin_db
        masks[i] = hold_forward(own, hold_frames)
    return masks


@dataclass
class Ownership:
    speakers: list[str]
    frame: int  # timeline samples per frame
    masks: np.ndarray  # n x frames, bool, session grid
    coupling: Coupling
    floors: list[float]
    key: str = ""
    warnings: list[str] = field(default_factory=list)
    # Mics whose speech level stays within ``silent_mic_range_db`` of their noise: they
    # record (almost) nothing of their wearer, whose voice exists only as bleed elsewhere.
    silent: list[str] = field(default_factory=list)
    # Levels relative to each track's speech level (n x frames), for comparing copies.
    relative: np.ndarray | None = None

    def index(self, speaker: str) -> int:
        return self.speakers.index(speaker)

    def frames(self, start: int, end: int) -> tuple[int, int]:
        a = max(0, start // self.frame)
        return a, max(a + 1, -(-end // self.frame))

    def owned(self, speaker: str, start: int, end: int) -> float:
        """Fraction of the frames in ``[start, end)`` (session samples) the speaker owns."""
        a, b = self.frames(start, end)
        row = self.masks[self.index(speaker), a:b]
        return float(row.mean()) if row.size else 0.0

    def relative_level(self, speaker: str, start: int, end: int) -> float:
        """Mean level over ``[start, end)`` relative to the track's speech level (dB)."""
        if self.relative is None:
            return 0.0
        a, b = self.frames(start, end)
        part = self.relative[self.index(speaker), a:b]
        if not part.size:
            return -120.0
        return float(10 * np.log10(np.mean(np.power(10.0, part.astype(np.float64) / 10))))

    def owned_mask(self, speaker: str, start: int, end: int) -> np.ndarray:
        a, b = self.frames(start, end)
        return self.masks[self.index(speaker), a:b]


def _key_fields(session: Session, settings: Settings, fingerprints: list[str]) -> dict[str, Any]:
    b = settings.bleed
    return {
        "fingerprints": fingerprints,
        "offsets": [p.offset_samples for p in session.participants],
        "frame_ms": b.frame_ms,
        "coupling": b.coupling,
        "margin_db": b.ownership_margin_db,
        "hold_ms": b.hold_ms,
        "floor_above_noise_db": b.floor_above_noise_db,
        "coupling_min_frames": b.coupling_min_frames,
        "coupling_default_db": b.coupling_default_db,
        "silent_mic_range_db": b.silent_mic_range_db,
        "mic_check": {k: list(v) for k, v in sorted(session.config.mic_check.items())},
        "coupling_db": session.config.coupling_db,
        "attribution_version": ATTRIBUTION_VERSION,
    }


def compute_ownership(session: Session, settings: Settings, cache: Cache | None) -> Ownership:
    """Coupling and masks for a whole session (cached; features come from their own layer)."""
    b = settings.bleed
    frame = frame_samples(b.frame_ms)
    speakers = [p.id for p in session.participants]
    tracks = [session.open_track(p) for p in session.participants]
    key = make_key(
        "ownership",
        _key_fields(session, settings, [t.fingerprint() + f"@{t.samplerate}" for t in tracks]),
    )
    envs = [track_envelope(t, cache, b.frame_ms) for t in tracks]
    shifts = [round(p.offset_samples / frame) for p in session.participants]
    levels = align(envs, shifts)
    silent = silent_mics(speakers, levels, b.silent_mic_range_db)
    relative = (levels - np.array([[speech_level(r)] for r in levels])).astype(np.float32)
    if cache is not None:
        meta = cache.load_json(key)
        packed = cache.load_array(key)
        if meta is not None and packed is not None:
            n, length = len(speakers), int(meta["length"])
            masks = np.unpackbits(packed, axis=1, count=length).astype(bool).reshape(n, length)
            matrix = np.asarray(meta["matrix"], dtype=np.float64)
            counts = np.asarray(meta["frames"], dtype=np.int64)
            source = {(i, j): s for i, j, s in meta["source"]}
            coupling = Coupling(speakers, matrix, counts, source)
            return Ownership(
                speakers,
                frame,
                masks,
                coupling,
                meta["floors"],
                key.short,
                meta.get("warnings", []),
                silent,
                relative,
            )

    warnings: list[str] = []
    if silent:
        warnings.append(
            f"mic(s) {', '.join(silent)} record almost nothing above their noise; "
            "their wearers' speech only exists as bleed on other tracks"
        )
    if b.coupling == "manual":
        if session.config.coupling_db is None:
            raise ValueError("bleed.coupling = manual needs coupling_db in the session file")
        coupling = manual_coupling(speakers, session.config.coupling_db)
    else:
        coupling = estimate_coupling(
            speakers, levels, min_frames=b.coupling_min_frames, default_db=b.coupling_default_db
        )
        if b.coupling == "guided":
            if not session.config.mic_check:
                warnings.append("bleed.coupling = guided but the session has no [mic_check]")
            ranges = {
                pid: (
                    seconds_to_samples(a) // frame,
                    seconds_to_samples(e) // frame,
                )
                for pid, (a, e) in session.config.mic_check.items()
            }
            coupling = apply_guided(coupling, levels, ranges, min_frames=b.coupling_min_frames)
    weak = coupling.weak_pairs()
    if weak:
        warnings.append(
            f"coupling: {len(weak)} mic pair(s) had too little single-speaker audio; "
            f"using {b.coupling_default_db:g} dB for them"
        )
    floors = track_floors(levels, b.floor_above_noise_db)
    hold = round(b.hold_ms / b.frame_ms)
    masks = ownership_masks(levels, coupling.matrix, floors, b.ownership_margin_db, hold)
    result = Ownership(
        speakers,
        frame,
        masks,
        coupling,
        [round(float(f), 2) for f in floors],
        key.short,
        warnings,
        silent,
        relative,
    )
    if cache is not None:
        cache.store_array(key, np.packbits(masks, axis=1))
        cache.store_json(
            key,
            {
                "length": int(masks.shape[1]),
                "matrix": coupling.matrix.tolist(),
                "frames": coupling.frames.tolist(),
                "source": [[i, j, s] for (i, j), s in sorted(coupling.source.items())],
                "floors": result.floors,
                "warnings": warnings,
            },
        )
    return result


def track_local_mask(own: Ownership, speaker: str, offset: int, start: int, end: int) -> np.ndarray:
    """A participant's mask for track-local samples ``[start, end)`` (pre mode)."""
    a, b = own.frames(start + offset, end + offset)
    row = own.masks[own.index(speaker)]
    out = np.zeros(b - a, dtype=bool)
    part = row[a:b]
    out[: part.size] = part
    return out


def pre_gains(
    owned: np.ndarray, attenuation_db: float, *, mute: bool, release_frames: int
) -> np.ndarray:
    """Per-frame linear gain for pre mode: 1 on owned frames, attenuated elsewhere.

    After owned speech the gain falls back to the attenuation over ``release_frames``
    (linear in dB); ``mute`` (expert option) silences instead of attenuating.
    """
    floor_db = -120.0 if mute else -attenuation_db
    gain_db = np.where(owned, 0.0, floor_db).astype(np.float64)
    if release_frames > 0:
        since = np.full(owned.size, np.inf)
        last = -np.inf
        for t, own in enumerate(owned):
            if own:
                last = t
            since[t] = t - last
        ramp = (since > 0) & (since <= release_frames)
        gain_db[ramp] = np.maximum(gain_db[ramp], floor_db * since[ramp] / release_frames)
    gain = np.power(10.0, gain_db / 20.0)
    if mute:
        gain[gain_db <= -120.0] = 0.0
    return gain.astype(np.float32)


def apply_gains(audio: np.ndarray, gains: np.ndarray, frame: int, ramp: int) -> np.ndarray:
    """Multiply 16 kHz audio by per-frame gains with ``ramp``-sample smoothing (no clicks)."""
    per_sample = np.repeat(gains, frame)[: audio.size]
    if per_sample.size < audio.size:
        fill = gains[-1] if gains.size else 1.0
        per_sample = np.concatenate([per_sample, np.full(audio.size - per_sample.size, fill)])
    if ramp > 1:
        kernel = np.ones(ramp) / ramp
        per_sample = np.convolve(
            np.pad(per_sample, (ramp // 2, ramp - 1 - ramp // 2), mode="edge"), kernel, mode="valid"
        )
    return (audio * per_sample.astype(np.float32)).astype(np.float32)
