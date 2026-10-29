"""Raw multi-channel recording probe (M0 hardware validation, SPEC §13).

One input stream for all selected channels. The PortAudio callback only copies into a
preallocated ring buffer and updates counters; a writer thread drains it, measures
levels and (optionally) writes one FLAC PCM_24 file per channel.

PASS criterion (DECISIONS 2026-10-29, M0): no overflow / error status flag, nothing dropped
by the ring buffer, equal frame counts on all channels, written == received, continuous
callback ADC timestamps (see TimestampMonitor). Wall-clock deviation is informational
(WARN above 200 ppm).
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import statistics
import threading
import time
from array import array
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from sessionscribe.core.audio.backend import (
    SAMPLE_DTYPE,
    AudioSetupError,
    InputDevice,
    StreamPlan,
    check_stream_settings,
    default_input_index,
    is_asio,
    list_input_devices,
    load_sounddevice,
    plan_input_stream,
    select_input_device,
)
from sessionscribe.core.audio.ringbuffer import RingBuffer

logger = logging.getLogger(__name__)

FULL_SCALE = float(2**31)  # int32 samples
BYTES_PER_SAMPLE_24BIT = 3
DISK_RESERVE = 1.10
CLOCK_WARN_PPM = 200.0
CLOCK_MIN_SECONDS = 60.0  # shorter runs: callback jitter dominates the ppm figure
STALL_TIMEOUT_S = 2.0
TIMESTAMP_TOLERANCE_BLOCKS = 0.5
TIMESTAMP_QUEUE_BLOCKS = 1 << 16  # (adc, frames) pairs between callback and writer


class Verdict(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ProbeConfig:
    channels: tuple[int, ...]
    duration_s: float
    device: str | None = None
    host_api: str | None = None
    samplerate: int = 48_000
    blocksize: int = 0  # 0 = let the host API choose
    buffer_s: float = 10.0
    write: bool = True
    out_dir: Path = Path("work/probe")
    wasapi_exclusive: bool = False


@dataclass(frozen=True)
class ProbeProgress:
    elapsed_s: float
    frames_received: int
    buffer_fill: float  # 0..1
    input_overflows: int
    dropped_blocks: int


@dataclass(frozen=True)
class ChannelLevel:
    channel: int  # 1-based input channel
    peak_dbfs: float
    rms_dbfs: float


@dataclass
class ProbeReport:
    device: str
    host_api: str
    device_index: int
    device_auto_selected: bool  # True when no --audio-device was given
    channels: list[int]
    samplerate: int
    actual_samplerate: float | None
    dtype: str
    blocksize: int
    latency_s: float | None
    requested_duration_s: float
    wall_elapsed_s: float
    interrupted: bool
    frames_received: int
    blocks: int
    frames_consumed: int
    frames_per_channel: list[int]
    frames_written: list[int] | None
    input_overflows: int
    input_underflows: int
    other_status_flags: int
    ring_dropped_blocks: int
    ring_dropped_frames: int
    ring_capacity_s: float
    ring_max_fill_s: float
    writer_max_write_s: float
    timestamps_available: bool
    timestamp_discontinuities: int
    timestamp_jitter_median_ms: float | None
    timestamp_jitter_p99_ms: float | None
    timestamp_jitter_max_ms: float | None
    missing_timestamps: int
    timestamp_queue_dropped: int
    clock_deviation_ppm: float | None
    levels: list[ChannelLevel]
    output_dir: str | None = None
    files: list[str] = field(default_factory=list)
    fatal_error: str | None = None
    verdict: Verdict = Verdict.FAIL
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        def clean(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value

        return json.dumps(clean(asdict(self)), indent=2)


def to_dbfs(value: float) -> float:
    return 20.0 * math.log10(value / FULL_SCALE) if value > 0 else -math.inf


def clock_deviation_ppm(frames: int, samplerate: float, wall_seconds: float) -> float | None:
    """Interface clock vs. PC wall clock in ppm (positive = interface runs fast)."""
    if wall_seconds <= 0 or frames <= 0:
        return None
    return (frames / samplerate / wall_seconds - 1.0) * 1e6


def required_disk_bytes(channels: int, samplerate: int, duration_s: float) -> int:
    """Uncompressed 24-bit size plus reserve (FLAC is smaller, this is the upper bound)."""
    raw = channels * samplerate * BYTES_PER_SAMPLE_24BIT * duration_s
    return int(raw * DISK_RESERVE)


@dataclass(frozen=True)
class JitterStats:
    """Absolute timestamp jitter around the median-smoothed residual, in seconds."""

    median_s: float
    p99_s: float
    max_s: float


class TimestampMonitor:
    """Detects discontinuities in callback ADC timestamps relative to the sample count.

    The residual ``(adc - adc0) - frames / rate`` is smoothed with a running median over
    ``smooth_s``. While blocks are contiguous the smoothed residual stays near a slowly
    tracking baseline (which absorbs clock drift); jitter and single late timestamps
    are removed by the median. A lost or inserted block shifts it permanently by one
    block. A discontinuity is a smoothed shift of more than ``tolerance_blocks`` (half a
    block) that persists for at least ``persist_s`` — so one lost block is detected.

    Runs in the writer thread, not in the audio callback.
    """

    def __init__(
        self,
        samplerate: float,
        tolerance_blocks: float = TIMESTAMP_TOLERANCE_BLOCKS,
        smooth_s: float = 0.2,
        persist_s: float = 0.1,
        alpha: float = 0.01,
    ):
        self.samplerate = samplerate
        self.tolerance_blocks = tolerance_blocks
        self.smooth_s = smooth_s
        self.persist_s = persist_s
        self.alpha = alpha
        self.seen = 0
        self.missing = 0
        self.discontinuities = 0
        self._frames = 0
        self._anchor_adc: float | None = None
        self._anchor_frames = 0
        self._window: deque[float] | None = None
        self._baseline = 0.0
        self._excess_s = 0.0
        self._jitter = array("d")

    def update(self, adc: float, frames: int) -> None:
        if adc <= 0.0:
            self.missing += 1
            self._frames += frames
            return
        self.seen += 1
        block_s = frames / self.samplerate
        if self._anchor_adc is None or self._window is None:
            self._anchor_adc = adc
            self._anchor_frames = self._frames
            length = max(5, math.ceil(self.smooth_s / block_s)) if block_s > 0 else 5
            self._window = deque(maxlen=length)
        residual = (adc - self._anchor_adc) - (self._frames - self._anchor_frames) / self.samplerate
        self._window.append(residual)
        smoothed = statistics.median(self._window)
        self._jitter.append(abs(residual - smoothed))
        shift = smoothed - self._baseline
        if abs(shift) > self.tolerance_blocks * block_s:
            self._excess_s += block_s
            if self._excess_s >= self.persist_s:
                self.discontinuities += 1
                self._baseline = smoothed
                self._excess_s = 0.0
        else:
            self._excess_s = 0.0
            self._baseline += self.alpha * shift
        self._frames += frames

    def jitter_stats(self) -> JitterStats | None:
        if not self._jitter:
            return None
        values = np.frombuffer(self._jitter, dtype=np.float64)
        median, p99 = np.percentile(values, [50, 99])
        return JitterStats(float(median), float(p99), float(np.max(values)))


class ProbeEngine:
    """Callback + consumer state for one probe run. Independent of sounddevice."""

    def __init__(
        self,
        channels: int,
        samplerate: int,
        ring_capacity_frames: int,
        columns: np.ndarray | None = None,
        writers: list[Any] | None = None,
    ):
        self.samplerate = samplerate
        self.ring = RingBuffer(ring_capacity_frames, channels, np.int32)
        self.columns = columns
        self.writers = writers
        # Callback-side counters (written only by the callback).
        self.frames_received = 0
        self.blocks = 0
        self.input_overflows = 0
        self.input_underflows = 0
        self.other_status_flags = 0
        self.first_block_frames = 0
        self.first_wall: float | None = None
        self.last_wall: float | None = None
        self.timestamp_queue = RingBuffer(TIMESTAMP_QUEUE_BLOCKS, 2, np.float64)
        self._timestamp_row = np.zeros((1, 2), dtype=np.float64)  # preallocated
        self.fatal_error: str | None = None
        # Consumer-side state.
        self.frames_consumed = 0
        self.channel_frames = np.zeros(channels, dtype=np.int64)
        self.peak = np.zeros(channels, dtype=np.int64)
        self.sumsq = np.zeros(channels, dtype=np.float64)
        self.writer_max_write_s = 0.0
        self.timestamps = TimestampMonitor(samplerate)

    # --- producer (PortAudio thread) ---------------------------------------------------

    def callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        try:
            now = time.perf_counter()
            if status:
                if status.input_overflow:
                    self.input_overflows += 1
                if status.input_underflow:
                    self.input_underflows += 1
                if not (status.input_overflow or status.input_underflow):
                    self.other_status_flags += 1
            row = self._timestamp_row
            row[0, 0] = time_info.inputBufferAdcTime
            row[0, 1] = frames
            self.timestamp_queue.write(row)
            self.ring.write(indata, self.columns)
            if self.blocks == 0:
                self.first_wall = now
                self.first_block_frames = frames
            self.last_wall = now
            self.frames_received += frames
            self.blocks += 1
        except Exception as exc:  # surfaced by the main loop
            self.fatal_error = f"callback: {exc!r}"

    # --- consumer (writer thread) ------------------------------------------------------

    def drain(self, max_frames: int = 1 << 16) -> int:
        """Move buffered frames to levels/files. Returns the number of frames handled."""
        for adc, block_frames in self.timestamp_queue.read():
            self.timestamps.update(float(adc), int(block_frames))
        data = self.ring.read(max_frames)
        frames = data.shape[0]
        if frames == 0:
            return 0
        wide = data.astype(np.int64)
        np.maximum(self.peak, np.abs(wide).max(axis=0), out=self.peak)
        as_float = data.astype(np.float64)
        self.sumsq += np.einsum("ij,ij->j", as_float, as_float)
        if self.writers is not None:
            t0 = time.perf_counter()
            for c, writer in enumerate(self.writers):
                writer.write(np.ascontiguousarray(data[:, c]))
            self.writer_max_write_s = max(self.writer_max_write_s, time.perf_counter() - t0)
        self.channel_frames += frames
        self.frames_consumed += frames
        return frames

    def run_writer(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                if self.drain() == 0:
                    time.sleep(0.01)
            while self.drain():
                pass
        except Exception as exc:
            self.fatal_error = f"writer: {exc!r}"

    # --- results -----------------------------------------------------------------------

    def levels(self, channels: tuple[int, ...]) -> list[ChannelLevel]:
        result = []
        for i, ch in enumerate(channels):
            n = int(self.channel_frames[i])
            rms = math.sqrt(self.sumsq[i] / n) if n else 0.0
            result.append(ChannelLevel(ch, to_dbfs(float(self.peak[i])), to_dbfs(rms)))
        return result

    def wall_elapsed(self) -> float:
        if self.first_wall is None or self.last_wall is None:
            return 0.0
        return self.last_wall - self.first_wall

    def clock_deviation_ppm(self) -> float | None:
        # Frames delivered between the first and the last callback.
        return clock_deviation_ppm(
            self.frames_received - self.first_block_frames, self.samplerate, self.wall_elapsed()
        )


def evaluate(report: ProbeReport) -> tuple[Verdict, list[str]]:
    """Apply the M0 PASS criterion. Returns verdict and human-readable reasons."""
    fails: list[str] = []
    warns: list[str] = []
    if report.fatal_error:
        fails.append(f"fatal error: {report.fatal_error}")
    if report.frames_received == 0:
        fails.append("no audio frames received")
    if report.actual_samplerate is not None and report.actual_samplerate != report.samplerate:
        fails.append(
            f"stream runs at {report.actual_samplerate:g} Hz, requested {report.samplerate}"
        )
    if report.input_overflows:
        fails.append(f"{report.input_overflows} input overflow(s)")
    if report.input_underflows or report.other_status_flags:
        fails.append(
            f"{report.input_underflows + report.other_status_flags} other error status flag(s)"
        )
    if report.ring_dropped_blocks:
        fails.append(
            f"ring buffer dropped {report.ring_dropped_blocks} block(s) "
            f"({report.ring_dropped_frames} frames)"
        )
    if len(set(report.frames_per_channel)) > 1:
        fails.append(f"channel frame counts differ: {report.frames_per_channel}")
    if report.frames_consumed != report.frames_received:
        fails.append(f"consumed {report.frames_consumed} frames, received {report.frames_received}")
    if report.frames_written is not None and any(
        w != report.frames_received for w in report.frames_written
    ):
        fails.append(f"written frames {report.frames_written} != received {report.frames_received}")
    if report.timestamp_discontinuities:
        fails.append(
            f"{report.timestamp_discontinuities} timestamp discontinuity(ies) "
            "(lost or inserted audio)"
        )
    if report.timestamp_queue_dropped:
        fails.append(f"timestamp queue dropped {report.timestamp_queue_dropped} entries")
    if not report.timestamps_available:
        warns.append("host API provides no ADC timestamps; continuity not verified")
    elif report.missing_timestamps:
        warns.append(f"{report.missing_timestamps} block(s) without ADC timestamp")
    ppm = report.clock_deviation_ppm
    if ppm is not None and report.wall_elapsed_s >= CLOCK_MIN_SECONDS and abs(ppm) > CLOCK_WARN_PPM:
        warns.append(f"interface clock deviates {ppm:+.0f} ppm from wall clock")
    if fails:
        return Verdict.FAIL, fails + warns
    if warns:
        return Verdict.WARN, warns
    return Verdict.PASS, []


def _open_writers(
    out_dir: Path, channels: tuple[int, ...], samplerate: int
) -> tuple[list[Any], list[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"ch{ch:02d}.flac" for ch in channels]
    writers = []
    try:
        for path in paths:
            writers.append(
                sf.SoundFile(
                    path, "w", samplerate=samplerate, channels=1, subtype="PCM_24", format="FLAC"
                )
            )
    except Exception:
        for w in writers:
            w.close()
        raise
    return writers, paths


def run_probe(
    config: ProbeConfig,
    *,
    on_progress: Callable[[ProbeProgress], None] | None = None,
    on_start: Callable[[InputDevice, bool], None] | None = None,
    progress_interval_s: float = 5.0,
    stop: threading.Event | None = None,
) -> ProbeReport:
    """Record ``config.duration_s`` seconds and return the evaluated report.

    Raises :class:`AudioSetupError` when the device/stream cannot be set up (before
    anything is recorded). Ctrl+C ends the run early with a normal report.
    """
    sd = load_sounddevice(enable_asio=is_asio(config.host_api))
    devices = list_input_devices(sd)
    device = select_input_device(
        devices,
        config.device,
        config.host_api,
        default_input_index(sd, config.host_api),
        min_channels=max(config.channels),
    )
    plan: StreamPlan = plan_input_stream(
        sd, device, config.channels, wasapi_exclusive=config.wasapi_exclusive
    )
    check_stream_settings(sd, plan, config.samplerate)
    if on_start is not None:
        on_start(device, config.device is None)

    out_dir: Path | None = None
    writers: list[Any] | None = None
    paths: list[Path] = []
    if config.write:
        out_dir = config.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        disk_root = next(p for p in (out_dir, *out_dir.parents) if p.exists())
        free = shutil.disk_usage(disk_root).free
        need = required_disk_bytes(len(config.channels), config.samplerate, config.duration_s)
        if free < need:
            raise AudioSetupError(
                f"not enough free space in {disk_root}: {free / 1e9:.1f} GB free, "
                f"{need / 1e9:.1f} GB needed (uncompressed 24-bit + 10 %)"
            )

    capacity = max(int(config.buffer_s * config.samplerate), 4096)
    engine = ProbeEngine(len(config.channels), config.samplerate, capacity, plan.columns)
    try:
        stream = sd.InputStream(
            device=device.index,
            channels=plan.open_channels,
            samplerate=config.samplerate,
            dtype=SAMPLE_DTYPE,
            blocksize=config.blocksize,
            callback=engine.callback,
            extra_settings=plan.extra_settings,
        )
    except Exception as exc:
        raise AudioSetupError(
            f"cannot open {plan.open_channels} channel(s) at {config.samplerate} Hz on "
            f"{device.label}: {exc}"
        ) from exc
    if out_dir is not None:
        try:
            writers, paths = _open_writers(out_dir, config.channels, config.samplerate)
        except Exception:
            stream.close()
            raise
        engine.writers = writers

    writer_stop = threading.Event()
    writer_thread = threading.Thread(
        target=engine.run_writer, args=(writer_stop,), name="probe-writer", daemon=True
    )
    stop = stop or threading.Event()
    interrupted = False
    actual_rate: float | None = None
    latency: float | None = None
    try:
        writer_thread.start()
        with stream:
            actual_rate = float(stream.samplerate)
            latency = float(stream.latency)
            start = time.perf_counter()
            next_progress = start + progress_interval_s
            try:
                while not stop.is_set():
                    now = time.perf_counter()
                    if now - start >= config.duration_s:
                        break
                    if engine.fatal_error:
                        break
                    last = engine.last_wall if engine.last_wall is not None else start
                    if now - last > STALL_TIMEOUT_S:
                        engine.fatal_error = f"no audio blocks for {STALL_TIMEOUT_S:.0f} s"
                        break
                    if not stream.active:
                        engine.fatal_error = "stream stopped unexpectedly"
                        break
                    if on_progress is not None and now >= next_progress:
                        next_progress += progress_interval_s
                        on_progress(
                            ProbeProgress(
                                elapsed_s=now - start,
                                frames_received=engine.frames_received,
                                buffer_fill=engine.ring.available() / engine.ring.capacity,
                                input_overflows=engine.input_overflows,
                                dropped_blocks=engine.ring.dropped_blocks,
                            )
                        )
                    time.sleep(0.05)
            except KeyboardInterrupt:
                interrupted = True
    finally:
        writer_stop.set()
        writer_thread.join()
        if writers is not None:
            for w in writers:
                w.close()

    frames_written = [sf.info(str(p)).frames for p in paths] if writers is not None else None
    jitter = engine.timestamps.jitter_stats()
    report = ProbeReport(
        device=device.name,
        host_api=device.host_api,
        device_index=device.index,
        device_auto_selected=config.device is None,
        channels=list(config.channels),
        samplerate=config.samplerate,
        actual_samplerate=actual_rate,
        dtype=SAMPLE_DTYPE,
        blocksize=config.blocksize,
        latency_s=latency,
        requested_duration_s=config.duration_s,
        wall_elapsed_s=engine.wall_elapsed(),
        interrupted=interrupted,
        frames_received=engine.frames_received,
        blocks=engine.blocks,
        frames_consumed=engine.frames_consumed,
        frames_per_channel=frames_written or [int(n) for n in engine.channel_frames],
        frames_written=frames_written,
        input_overflows=engine.input_overflows,
        input_underflows=engine.input_underflows,
        other_status_flags=engine.other_status_flags,
        ring_dropped_blocks=engine.ring.dropped_blocks,
        ring_dropped_frames=engine.ring.dropped_frames,
        ring_capacity_s=capacity / config.samplerate,
        ring_max_fill_s=engine.ring.max_fill_frames / config.samplerate,
        writer_max_write_s=engine.writer_max_write_s,
        timestamps_available=engine.timestamps.seen > 0,
        timestamp_discontinuities=engine.timestamps.discontinuities,
        timestamp_jitter_median_ms=jitter.median_s * 1000.0 if jitter else None,
        timestamp_jitter_p99_ms=jitter.p99_s * 1000.0 if jitter else None,
        timestamp_jitter_max_ms=jitter.max_s * 1000.0 if jitter else None,
        missing_timestamps=engine.timestamps.missing,
        timestamp_queue_dropped=engine.timestamp_queue.dropped_blocks,
        clock_deviation_ppm=engine.clock_deviation_ppm(),
        levels=engine.levels(config.channels),
        output_dir=str(out_dir) if out_dir is not None else None,
        files=[str(p) for p in paths],
        fatal_error=engine.fatal_error,
    )
    report.verdict, report.reasons = evaluate(report)
    if out_dir is not None:
        (out_dir / "probe-report.json").write_text(report.to_json(), encoding="utf-8")
    return report
