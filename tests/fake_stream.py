"""A fake PortAudio input stream for recorder tests (no audio hardware).

A thread calls the recorder callback with synthetic int32 blocks, faster than real
time. Scenario hooks: an input-overflow status, a jump in ADC timestamps (lost audio
the driver reports only through time), and a device that dies after N blocks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np


def tone_block(start: int, frames: int, channels: int, amps: list[float], rate: int) -> np.ndarray:
    t = (np.arange(start, start + frames) / rate)[:, None]
    freqs = 300.0 + 50.0 * np.arange(channels)[None, :]
    x = np.sin(2 * np.pi * freqs * t) * np.asarray(amps)[None, :]
    return (x * (2**31 - 1)).astype(np.int32)


@dataclass
class Scenario:
    blocks: int = 200
    blocksize: int = 480
    channels: int = 3
    rate: int = 48_000
    amps: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])
    noise: float = 1e-4
    block_sleep: float = 0.0005
    overflow_at: int | None = None
    adc_jump_at: int | None = None
    adc_jump_s: float = 0.0
    die_at: int | None = None
    amps_fn: Callable[[int], list[float]] | None = None  # per block index


class FakeStream:
    def __init__(self, callback, scenario: Scenario, start_block: int = 0):
        self.callback = callback
        self.s = scenario
        self.samplerate = float(scenario.rate)
        self.latency = 0.01
        self.active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_block = start_block
        self.sent_frames = 0
        self.done = threading.Event()

    def start(self) -> None:
        self.active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        s = self.s
        rng = np.random.default_rng(0)
        adc = 0.0
        for b in range(self._start_block, s.blocks):
            if self._stop.is_set():
                break
            if s.die_at is not None and b == s.die_at:
                self.active = False
                return
            amps = s.amps_fn(b) if s.amps_fn else s.amps
            block = tone_block(b * s.blocksize, s.blocksize, s.channels, amps, s.rate)
            block = block + (rng.normal(0, s.noise, block.shape) * 2**31).astype(np.int32)
            if s.adc_jump_at is not None and b == s.adc_jump_at:
                adc += s.adc_jump_s
            status = SimpleNamespace(input_overflow=s.overflow_at == b)
            self.callback(block, s.blocksize, SimpleNamespace(inputBufferAdcTime=1.0 + adc), status)
            adc += s.blocksize / s.rate
            self.sent_frames += s.blocksize
            time.sleep(s.block_sleep)
        self.done.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.active = False

    def close(self) -> None:
        self.active = False


class FakeOpener:
    device_name = "Fake UMC"
    device_index = 7
    host_api = "ASIO"
    columns = None

    def __init__(self, scenarios: list[Scenario], fail_opens: int = 0):
        self.scenarios = list(scenarios)
        self.opened: list[FakeStream] = []
        self.refreshes = 0
        self.fail_opens = fail_opens

    def open(self, callback):
        if self.fail_opens > 0 and self.opened:
            self.fail_opens -= 1
            raise RuntimeError("device not found")
        s = self.scenarios[min(len(self.opened), len(self.scenarios) - 1)]
        stream = FakeStream(callback, s)
        self.opened.append(stream)
        return stream

    def refresh(self) -> None:
        self.refreshes += 1


def child_opener(setup) -> FakeOpener:
    """Device for the recorder child process in tests (``SESSIONSCRIBE_TEST_OPENER``).

    About real time, endless. ``SESSIONSCRIBE_TEST_SCENARIO``:
    - ``steady`` (default): every channel speaks;
    - ``mic_ok``: each channel speaks only in its own mic-check turn (1 s noise, 1 s
      turns, see ``plan_turns``), after that all of them;
    - ``mic_dead2``: like ``mic_ok`` but the second channel is never voiced.
    """
    import os

    from sessionscribe.core.audio.mic_check import plan_turns

    n = len(setup.channels)
    rate, blocksize = setup.samplerate, setup.blocksize
    scenario = os.environ.get("SESSIONSCRIBE_TEST_SCENARIO", "steady")
    _, turns = plan_turns(setup.ids, 1.0, 1.0)
    end = turns[-1].end_s + 1.0

    def amps(b: int) -> list[float]:
        t = b * blocksize / rate
        if scenario == "steady":
            return [0.3] * n
        out = [0.0] * n
        if t >= end:
            out = [0.3] * n
        for turn in turns:
            if turn.start_s - 0.3 <= t < turn.end_s:
                out[turn.channel_index] = 0.3
        if scenario == "mic_dead2" and n > 1:
            out[1] = 0.0
        return out

    return FakeOpener(
        [
            Scenario(
                blocks=10**9,
                blocksize=blocksize,
                channels=n,
                rate=rate,
                amps_fn=amps,
                block_sleep=blocksize / rate,
            )
        ]
    )
