"""Integer-sample session timeline (SPEC §4).

All positions inside ``core`` are integer sample indices at :data:`TIMELINE_RATE`.
Float seconds exist only at the edges: library results coming in and exporters going
out, each converted exactly once.
"""

from __future__ import annotations

from fractions import Fraction

TIMELINE_RATE = 48_000
WHISPER_RATE = 16_000
EXPORT_DECIMALS = 3


def seconds_to_samples(seconds: float, rate: int = TIMELINE_RATE) -> int:
    return round(seconds * rate)


def samples_to_seconds(samples: int, rate: int = TIMELINE_RATE) -> float:
    """For export only: rounded to milliseconds."""
    return round(samples / rate, EXPORT_DECIMALS)


def rescale(samples: int, from_rate: int, to_rate: int) -> int:
    """Exact rate conversion of a sample position, rounded half away from zero."""
    value = Fraction(samples * to_rate, from_rate)
    rounded = int(abs(value) + Fraction(1, 2))
    return rounded if value >= 0 else -rounded


def format_clock(samples: int, rate: int = TIMELINE_RATE, *, millis: bool = False) -> str:
    """``hh:mm:ss`` (truncated) or ``hh:mm:ss.mmm`` for a non-negative position."""
    total_ms = rescale(max(samples, 0), rate, 1000)
    seconds, ms = divmod(total_ms, 1000)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{text}.{ms:03d}" if millis else text
