"""Guided mic check: alarms for assigned channels before (or at the start of) recording.

Phases: a few seconds of silence (everyone quiet) for each channel's noise level, then
each participant speaks in turn. Every comparison uses a channel's *rise above its own
noise*, because mic gains differ by 15-20 dB (owner decision); only the absolute
"too quiet" level and clipping use dBFS.

Alarms per channel:
- ``no_signal`` (FAIL): no rise during its own turn, or digital silence (muted,
  unplugged or wrongly assigned);
- ``wrong_channel`` (FAIL): another channel rises more during this participant's turn
  than their own does (swapped assignment);
- ``quiet`` (WARN): own-turn speech below an absolute floor, or its rise far below the
  other channels' rises (low gain, badly placed lavalier);
- ``clipping`` (WARN): the peak reached -1 dBFS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sessionscribe.core.audio.recorder import CLIP_DBFS, LevelEvent

NO_SIGNAL_RISE_DB = 10.0
DIGITAL_SILENCE_DBFS = -100.0
QUIET_DBFS = -50.0
QUIET_BELOW_OTHERS_DB = 20.0

Verdict = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class Turn:
    participant: str
    channel_index: int  # position in the assigned channel list
    start_s: float
    end_s: float


@dataclass
class ChannelCheck:
    channel: int
    participant: str
    noise_dbfs: float
    speech_dbfs: float
    rise_db: float
    peak_dbfs: float
    alarms: list[tuple[str, str]] = field(default_factory=list)  # (kind, message)

    @property
    def verdict(self) -> Verdict:
        kinds = {k for k, _ in self.alarms}
        if kinds & {"no_signal", "wrong_channel"}:
            return "FAIL"
        return "WARN" if kinds else "PASS"


def _window(events: Sequence[LevelEvent], start: float, end: float) -> np.ndarray:
    rows = [e.rms_dbfs for e in events if start <= e.time_s <= end]
    return np.asarray(rows, dtype=np.float64)


def analyze(
    events: Sequence[LevelEvent],
    channels: Sequence[int],
    participants: Sequence[str],
    turns: Sequence[Turn],
    noise_window: tuple[float, float] | None,
) -> list[ChannelCheck]:
    """Per-channel levels and alarms from the level events of a mic check."""
    if not events:
        raise ValueError("no audio was received during the mic check")
    all_rms = np.asarray([e.rms_dbfs for e in events], dtype=np.float64)
    peaks = np.asarray([e.peak_dbfs for e in events], dtype=np.float64)
    if noise_window is not None and len(_window(events, *noise_window)):
        noise = np.median(_window(events, *noise_window), axis=0)
    else:
        noise = np.percentile(all_rms, 10, axis=0)
    n = len(channels)
    own_speech = np.full(n, -120.0)
    rise_in_turn = np.full((n, n), -120.0)  # [speaker turn, channel]
    for turn in turns:
        w = _window(events, turn.start_s, turn.end_s)
        if not len(w):
            continue
        level = np.percentile(w, 90, axis=0)
        rise_in_turn[turn.channel_index] = level - noise
        own_speech[turn.channel_index] = level[turn.channel_index]
    checks = []
    for c in range(n):
        rise = own_speech[c] - noise[c]
        check = ChannelCheck(
            channel=channels[c],
            participant=participants[c],
            noise_dbfs=round(float(noise[c]), 1),
            speech_dbfs=round(float(own_speech[c]), 1),
            rise_db=round(float(rise), 1),
            peak_dbfs=round(float(np.max(peaks[:, c])), 1) if len(peaks) else -120.0,
        )
        has_turn = any(t.channel_index == c for t in turns)
        if float(np.max(all_rms[:, c])) <= DIGITAL_SILENCE_DBFS:
            check.alarms.append(("no_signal", "digital silence: muted, unplugged or not assigned"))
        elif has_turn and rise < NO_SIGNAL_RISE_DB:
            check.alarms.append(
                (
                    "no_signal",
                    f"rises only {rise:.0f} dB above its noise when {participants[c]} speaks",
                )
            )
        if has_turn:
            others = [k for k in range(n) if k != c and rise_in_turn[c, k] > rise_in_turn[c, c]]
            if others:
                k = max(others, key=lambda k: rise_in_turn[c, k])
                check.alarms.append(
                    (
                        "wrong_channel",
                        f"channel {channels[k]} rises more ({rise_in_turn[c, k]:.0f} dB) than "
                        f"channel {channels[c]} ({rise_in_turn[c, c]:.0f} dB) when "
                        f"{participants[c]} speaks: swapped assignment?",
                    )
                )
            other_rises = [
                rise_in_turn[t.channel_index, t.channel_index]
                for t in turns
                if t.channel_index != c
            ]
            if own_speech[c] < QUIET_DBFS and rise >= NO_SIGNAL_RISE_DB:
                check.alarms.append(
                    ("quiet", f"speech only {own_speech[c]:.0f} dBFS: raise the gain")
                )
            elif (
                other_rises
                and rise < float(np.median(other_rises)) - QUIET_BELOW_OTHERS_DB
                and rise >= NO_SIGNAL_RISE_DB
            ):
                check.alarms.append(
                    (
                        "quiet",
                        f"rises {rise:.0f} dB, the others about "
                        f"{np.median(other_rises):.0f} dB: low gain or badly placed mic?",
                    )
                )
        if check.peak_dbfs >= CLIP_DBFS:
            check.alarms.append(("clipping", f"peak {check.peak_dbfs:.1f} dBFS: lower the gain"))
        checks.append(check)
    return checks


def overall(checks: Sequence[ChannelCheck]) -> Verdict:
    verdicts = {c.verdict for c in checks}
    return "FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else "PASS"


def plan_turns(
    participants: Sequence[str], noise_s: float, turn_s: float, gap_s: float = 1.0
) -> tuple[tuple[float, float] | None, list[Turn]]:
    """Timeline of a mic check: optional silence, then one turn per participant."""
    t = 0.0
    noise = None
    if noise_s > 0:
        noise = (0.5, noise_s)  # skip the first half second (settling, the prompt)
        t = noise_s + gap_s
    turns = []
    for i, pid in enumerate(participants):
        turns.append(Turn(pid, i, t + 0.5, t + turn_s))
        t += turn_s + gap_s
    return noise, turns
