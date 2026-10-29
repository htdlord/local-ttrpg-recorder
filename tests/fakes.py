"""Fake ASR engines for pipeline tests; importable by spawned worker processes."""

from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sessionscribe.core.pipeline.asr import (
    LoadedModel,
    Segment,
    SpeechInterval,
    TranscriptionResult,
    Word,
)
from sessionscribe.core.timeline import TIMELINE_RATE, WHISPER_RATE

FRAME = WHISPER_RATE // 50  # 20 ms
ACTIVE_DB = -40.0
WORD = 0.4  # seconds per fake word

LOADED = LoadedModel("fake", "fake", "rev0", "cpu", "int8", True, 0.0)


def crash() -> None:
    sys.stdout.flush()
    if os.name == "nt":
        os._exit(0xC0000005 - (1 << 32))  # as a signed C int
    os.kill(os.getpid(), signal.SIGSEGV)


def active_regions(audio: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) in 16 kHz samples of stretches above -40 dBFS."""
    n = len(audio) // FRAME
    if n == 0:
        return []
    frames = audio[: n * FRAME].astype(np.float64).reshape(n, FRAME)
    with np.errstate(divide="ignore"):
        db = 10 * np.log10((frames**2).mean(axis=1))
    active = db > ACTIVE_DB
    regions, start = [], None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append((start * FRAME, i * FRAME))
            start = None
    if start is not None:
        regions.append((start * FRAME, n * FRAME))
    return regions


def to_timeline(samples16: int) -> int:
    return samples16 * (TIMELINE_RATE // WHISPER_RATE)


class FakeEngine:
    """One segment per loud stretch, one word per 0.4 s; text names the word's position."""

    def __init__(
        self,
        calls: Path | None = None,
        crash_state: Path | None = None,
        crash_on: str | None = None,
    ) -> None:
        self.calls, self.crash_state, self.crash_on = calls, crash_state, crash_on
        self.loaded = LOADED

    def transcribe(self, audio, *, source_id, sink=None, cancel=None):
        if self.calls is not None:
            with self.calls.open("a", encoding="utf-8") as fh:
                fh.write(f"{source_id}\n")
        if self.crash_state is not None and source_id == self.crash_on:
            left = int(self.crash_state.read_text(encoding="utf-8") or 0)
            if left > 0:
                self.crash_state.write_text(str(left - 1), encoding="utf-8")
                crash()
        segments = []
        for i, (a, b) in enumerate(active_regions(audio)):
            step = int(WORD * WHISPER_RATE)
            words = tuple(
                Word(to_timeline(s), to_timeline(min(b, s + step)), f" ő{k}", 0.9)
                for k, s in enumerate(range(a, b, step))
            )
            segments.append(
                Segment(
                    i,
                    to_timeline(a),
                    to_timeline(b),
                    "".join(w.text for w in words).strip(),
                    -0.2,
                    0.01,
                    1.2,
                    0.0,
                    words,
                )
            )
        return TranscriptionResult(
            source_id=source_id,
            model=LOADED,
            mode="quality",
            batch_size=None,
            language="hu",
            language_probability=0.99,
            duration=to_timeline(len(audio)),
            transcribe_s=0.01,
            vad_s=0.0,
            segments=segments,
            vad_intervals=[
                SpeechInterval(to_timeline(a), to_timeline(b)) for a, b in active_regions(audio)
            ],
        )


@dataclass(frozen=True)
class FakeFactory:
    """Picklable engine factory (spawned workers import this module)."""

    calls: Path | None = None
    crash_state: Path | None = None
    crash_on: str | None = None
    fail_load: bool = False

    def __call__(self, settings):
        if self.fail_load:
            raise RuntimeError("no model here")
        return FakeEngine(self.calls, self.crash_state, self.crash_on)


class OomBatched:
    """Batched pipeline that runs out of memory above ``limit``; records the sizes tried."""

    def __init__(self, limit: int, sizes: list[int]) -> None:
        self.limit, self.sizes = limit, sizes

    def transcribe(self, audio, batch_size, **kwargs):
        from types import SimpleNamespace

        self.sizes.append(batch_size)

        def gen():
            if batch_size > self.limit:
                raise RuntimeError("CUDA failed with error out of memory")
            yield from ()

        return gen(), SimpleNamespace(language="hu", language_probability=0.9)


@dataclass
class OomFactory:
    limit: int
    sizes: list[int]

    def __call__(self, settings):
        from sessionscribe.core.pipeline.asr import AsrEngine

        return AsrEngine(settings, object(), LOADED, lambda m: OomBatched(self.limit, self.sizes))
