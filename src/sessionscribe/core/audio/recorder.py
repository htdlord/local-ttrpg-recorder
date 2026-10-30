"""Multi-channel session recorder (SPEC §8).

One input stream for all assigned channels. The PortAudio callback only copies into a
preallocated, bounded ring buffer and updates counters; it never does I/O, allocates or
blocks. A writer thread drains the buffer into consecutive, individually closed segment
files per channel (``rec/chNN_<participant>/NNNN.flac``) and keeps the manifest in the
session file up to date (atomically, after every closed segment, gap and state change).

Lost audio never shifts the timeline: frames the ring buffer had to drop (exact count),
a timestamp discontinuity reported by the driver (estimated) and the time an unplugged
device was away (estimated from wall time) are filled with silence in every channel
and recorded as gaps. Real audio is never modified.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from sessionscribe.core.audio.manifest import (
    AlarmRecord,
    GapRecord,
    RecordingManifest,
    SegmentRecord,
    file_sha256,
)
from sessionscribe.core.audio.probe import FULL_SCALE, TimestampMonitor, required_disk_bytes
from sessionscribe.core.audio.ringbuffer import RingBuffer
from sessionscribe.core.session import Participant, SessionConfig, write_session

log = logging.getLogger(__name__)

CLIP_DBFS = -1.0
NO_AUDIO_S = 1.0  # no callback for this long while running: alarm
DISK_CHECK_S = 5.0
IDLE_RISE_DB = 10.0  # a channel within this of its noise level is idle
NOISE_LEARN_S = 120  # no idle warnings before the noise floor has been seen (seconds)
RECONNECT_EVERY_S = 2.0
DROP_QUEUE = 4096


# --- events ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelEvent:
    time_s: float
    peak_dbfs: tuple[float, ...]
    rms_dbfs: tuple[float, ...]
    clipped: tuple[bool, ...]


@dataclass(frozen=True)
class RecorderAlarm:
    time_s: float
    kind: str
    message: str
    channel: int | None = None
    fatal: bool = False


@dataclass(frozen=True)
class RecorderStatus:
    time_s: float
    frames: int
    buffer_fill: float
    disk_left_s: float | None
    gaps: int
    state: str


RecorderEvent = LevelEvent | RecorderAlarm | RecorderStatus
RecorderSink = Callable[[RecorderEvent], None]


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


# --- stream abstraction ---------------------------------------------------------------------


class InputStream(Protocol):
    samplerate: float
    latency: float
    active: bool

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class StreamOpener(Protocol):
    """Opens the configured stream; ``refresh`` re-reads the device list (after unplug)."""

    device_name: str
    device_index: int
    host_api: str
    columns: np.ndarray | None

    def open(self, callback: Callable[..., None]) -> InputStream: ...
    def refresh(self) -> None: ...


# --- callback side --------------------------------------------------------------------------


class CaptureEngine:
    """State shared by the audio callback (producer) and the writer (consumer)."""

    def __init__(self, channels: int, samplerate: int, capacity: int, columns: np.ndarray | None):
        self.ring = RingBuffer(capacity, channels, np.int32)
        self.columns = columns
        self.samplerate = samplerate
        self.timestamp_queue = RingBuffer(1 << 16, 2, np.float64)
        self.drop_queue = RingBuffer(DROP_QUEUE, 2, np.int64)
        self._ts_row = np.zeros((1, 2), np.float64)
        self._drop_row = np.zeros((1, 2), np.int64)
        self.frames_received = 0
        self.blocks = 0
        self.input_overflows = 0
        self.last_wall = 0.0
        self.fatal: str | None = None

    def callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        try:
            if status and getattr(status, "input_overflow", False):
                self.input_overflows += 1
            adc = getattr(time_info, "inputBufferAdcTime", 0.0) if time_info is not None else 0.0
            self._ts_row[0, 0] = adc
            self._ts_row[0, 1] = frames
            self.timestamp_queue.write(self._ts_row)
            if not self.ring.write(indata, self.columns):
                self._drop_row[0, 0] = self.frames_received
                self._drop_row[0, 1] = frames
                self.drop_queue.write(self._drop_row)
            self.frames_received += frames
            self.blocks += 1
            self.last_wall = time.monotonic()
        except Exception as exc:  # surfaced by the main loop
            self.fatal = f"callback: {exc!r}"


# --- writer side ----------------------------------------------------------------------------


class SegmentWriter:
    """Per-channel segment files; rotates every ``segment_frames``; silence for gaps."""

    def __init__(
        self,
        folder: Path,
        manifest: RecordingManifest,
        on_change: Callable[[], None],
        segment_frames: int,
    ) -> None:
        self.folder = folder
        self.manifest = manifest
        self.on_change = on_change
        self.segment_frames = segment_frames
        self.position = max(
            (s.start_sample + s.frames for s in manifest.segments), default=0
        )  # frames written per channel (timeline)
        self._files: list[sf.SoundFile] = []
        self._records: list[SegmentRecord] = []
        self._in_segment = 0
        self._since_flush = 0
        self.flush_frames = manifest.samplerate  # flush to disk every second (crash safety)

    @property
    def n(self) -> int:
        return len(self.manifest.channels)

    def _path(self, c: int, index: int) -> Path:
        ch, pid = self.manifest.channels[c], self.manifest.participants[c]
        ext = "flac" if self.manifest.format == "flac" else "wav"
        return self.folder / "rec" / f"ch{ch:02d}_{pid}" / f"{index:04d}.{ext}"

    def _open(self) -> None:
        index = 1 + max((s.index for s in self.manifest.segments), default=-1)
        for c in range(self.n):
            path = self._path(c, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            fmt = "FLAC" if self.manifest.format == "flac" else "RF64"
            self._files.append(
                sf.SoundFile(
                    str(path),
                    "w",
                    samplerate=self.manifest.samplerate,
                    channels=1,
                    subtype=self.manifest.subtype,
                    format=fmt,
                )
            )
            record = SegmentRecord(
                channel=self.manifest.channels[c],
                participant=self.manifest.participants[c],
                index=index,
                file=path.relative_to(self.folder).as_posix(),
                start_sample=self.position,
            )
            self._records.append(record)
            self.manifest.segments.append(record)
        self._in_segment = 0
        self.on_change()

    def close_segment(self) -> None:
        if not self._files:
            return
        for f, record in zip(self._files, self._records, strict=True):
            f.close()
            record.frames = self._in_segment
            record.sha256 = file_sha256(self.folder / record.file)
            record.status = "closed"
        self._files, self._records = [], []
        self.on_change()

    def write(self, block: np.ndarray) -> None:
        """Frames x channels int32; split at segment boundaries."""
        start = 0
        while start < block.shape[0]:
            if not self._files:
                self._open()
            room = self.segment_frames - self._in_segment
            part = block[start : start + room]
            for c, f in enumerate(self._files):
                f.write(np.ascontiguousarray(part[:, c]))
            self._in_segment += part.shape[0]
            self.position += part.shape[0]
            start += part.shape[0]
            self._since_flush += part.shape[0]
            if self._in_segment >= self.segment_frames:
                self.close_segment()
                self._since_flush = 0
            elif self._since_flush >= self.flush_frames:
                for f in self._files:
                    f.flush()
                self._since_flush = 0

    def pad(self, frames: int) -> None:
        chunk = 1 << 16
        while frames > 0:
            n = min(chunk, frames)
            self.write(np.zeros((n, self.n), np.int32))
            frames -= n


@dataclass
class ChannelLevels:
    """Writer-side level accounting (peaks/RMS since the last event, idle tracking)."""

    n: int
    samplerate: int
    peak: np.ndarray = field(init=False)
    sumsq: np.ndarray = field(init=False)
    count: int = 0
    sec_sumsq: np.ndarray = field(init=False)
    sec_count: int = 0
    noise_db: np.ndarray = field(init=False)
    idle_s: np.ndarray = field(init=False)
    warned: np.ndarray = field(init=False)
    seconds: int = 0

    def __post_init__(self) -> None:
        self.peak = np.zeros(self.n)
        self.sumsq = np.zeros(self.n)
        self.sec_sumsq = np.zeros(self.n)
        self.noise_db = np.full(self.n, np.inf)
        self.idle_s = np.zeros(self.n)
        self.warned = np.zeros(self.n, bool)

    def add(self, block: np.ndarray) -> list[np.ndarray]:
        """Accumulate; returns the per-channel dB of every completed second."""
        x = block.astype(np.float64) / FULL_SCALE
        np.maximum(self.peak, np.abs(x).max(axis=0), out=self.peak)
        sq = np.einsum("ij,ij->j", x, x)
        self.sumsq += sq
        self.count += x.shape[0]
        done: list[np.ndarray] = []
        offset = 0
        while offset < x.shape[0]:
            take = min(self.samplerate - self.sec_count, x.shape[0] - offset)
            part = x[offset : offset + take]
            self.sec_sumsq += np.einsum("ij,ij->j", part, part)
            self.sec_count += take
            offset += take
            if self.sec_count >= self.samplerate:
                done.append(10 * np.log10(np.maximum(self.sec_sumsq / self.sec_count, 1e-12)))
                self.sec_sumsq[:] = 0
                self.sec_count = 0
        return done

    def take_event(self, time_s: float) -> LevelEvent | None:
        if not self.count:
            return None
        rms = np.sqrt(self.sumsq / self.count)
        event = LevelEvent(
            time_s,
            tuple(_db(p) for p in self.peak),
            tuple(_db(r) for r in rms),
            tuple(bool(_db(p) >= CLIP_DBFS) for p in self.peak),
        )
        self.peak[:] = 0
        self.sumsq[:] = 0
        self.count = 0
        return event

    def idle_check(self, second_db: np.ndarray, warn_s: float) -> list[int]:
        """Channels that just reached ``warn_s`` at noise level while another is active."""
        self.seconds += 1
        # Noise floor: the quietest second so far (conversation always has pauses).
        self.noise_db = np.minimum(self.noise_db, second_db)
        if self.seconds <= NOISE_LEARN_S:
            return []
        active = second_db >= self.noise_db + IDLE_RISE_DB
        out = []
        for c in range(self.n):
            if active[c]:
                self.idle_s[c] = 0
                self.warned[c] = False
            elif active.any():
                self.idle_s[c] += 1
                if self.idle_s[c] >= warn_s and not self.warned[c]:
                    self.warned[c] = True
                    out.append(c)
        return out


# --- recorder -------------------------------------------------------------------------------


@dataclass
class RecorderConfig:
    session_dir: Path
    channels: tuple[int, ...]
    participants: tuple[str, ...]
    names: tuple[str, ...] = ()
    samplerate: int = 48_000
    blocksize: int = 480
    segment_s: float = 300.0
    format: str = "flac"
    buffer_s: float = 20.0
    planned_hours: float = 8.0
    idle_warn_s: float = 600.0
    disk_warn_s: float = 1800.0
    disk_stop_s: float = 120.0
    reconnect: bool = True
    level_interval_s: float = 0.25
    session_name: str = ""


class RecorderError(RuntimeError):
    pass


class Recorder:
    def __init__(
        self,
        config: RecorderConfig,
        opener: StreamOpener,
        *,
        sink: RecorderSink = lambda e: None,
        disk_free: Callable[[Path], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        inhibit: Callable[[], Any] | None = None,
    ) -> None:
        if len(config.channels) != len(config.participants):
            raise ValueError("one participant per channel")
        self.config = config
        self.opener = opener
        self.sink = sink
        self.disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
        self.clock = clock
        self.inhibit = inhibit
        self.session_path = config.session_dir / "session.toml"
        self._lock = threading.Lock()
        self.session = self._prepare_session()
        self.manifest = self.session.recording
        assert self.manifest is not None
        n = len(config.channels)
        self.engine = CaptureEngine(
            n,
            config.samplerate,
            max(4096, int(config.buffer_s * config.samplerate)),
            opener.columns,
        )
        self.writer = SegmentWriter(
            config.session_dir,
            self.manifest,
            self._save,
            round(config.segment_s * config.samplerate),
        )
        self.levels = ChannelLevels(n, config.samplerate)
        self.timestamps = TimestampMonitor(config.samplerate)
        self._seen_discontinuities = 0
        self._writer_stop = threading.Event()
        self._writer_paused = threading.Event()
        self._writer_error: str | None = None
        self._drain_lock = threading.Lock()
        self._written_real = 0
        self._received_offset = 0
        self._t0 = clock()

    # --- session file ---

    def _prepare_session(self) -> SessionConfig:
        c = self.config
        c.session_dir.mkdir(parents=True, exist_ok=True)
        if self.session_path.is_file():
            from sessionscribe.core.session import Session

            existing = Session.load(self.session_path).config
            rec = existing.recording
            if rec is None or rec.state == "recording":
                raise RecorderError(
                    f"{self.session_path} exists"
                    + (
                        " and is still marked as recording; run 'record recover' first"
                        if rec is not None
                        else " without a recording"
                    )
                )
            if list(c.channels) != rec.channels or list(c.participants) != rec.participants:
                raise RecorderError("resuming needs the same channels and participants")
            rec.state = "recording"
            return existing
        names = c.names or c.participants
        participants = [
            Participant(id=pid, name=name, source=f"rec/ch{ch:02d}_{pid}")
            for ch, pid, name in zip(c.channels, c.participants, names, strict=True)
        ]
        manifest = RecordingManifest(
            device_name=self.opener.device_name,
            device_index=self.opener.device_index,
            host_api=self.opener.host_api,
            channels=list(c.channels),
            participants=list(c.participants),
            samplerate=c.samplerate,
            format="flac" if c.format == "flac" else "rf64",
            blocksize=c.blocksize,
            started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            segment_s=c.segment_s,
        )
        return SessionConfig(
            name=c.session_name or c.session_dir.name,
            date=datetime.now().date().isoformat(),
            participants=participants,
            recording=manifest,
        )

    def _save(self) -> None:
        with self._lock:
            m = self.manifest
            assert m is not None
            e = self.engine
            m.metrics.frames = self.writer.position
            m.metrics.blocks = e.blocks
            m.metrics.input_overflows = e.input_overflows
            m.metrics.max_buffer_fill = round(e.ring.max_fill_frames / e.ring.capacity, 4)
            m.metrics.timestamp_discontinuities = self.timestamps.discontinuities
            write_session(self.session_path, self.session)

    def audio_time(self) -> float:
        """Recording time in seconds: frames written (incl. gaps) / sample rate."""
        return self.writer.position / self.config.samplerate

    def set_mic_check(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Mic check turns in recording time (guided coupling, SPEC §5.2)."""
        with self._lock:
            self.session.mic_check = dict(ranges)
        self._save()

    @property
    def rec_dir(self) -> Path:
        return self.config.session_dir / "rec"

    # --- events ---

    def _time(self) -> float:
        return round(self.clock() - self._t0, 3)

    def _alarm(
        self, kind: str, message: str, channel: int | None = None, fatal: bool = False
    ) -> None:
        assert self.manifest is not None
        self.manifest.alarms.append(
            AlarmRecord(time_s=self._time(), kind=kind, channel=channel, message=message)
        )
        self.sink(RecorderAlarm(self._time(), kind, message, channel, fatal))
        self._save()

    def _gap(self, length: int, reason: str, *, estimated: bool) -> None:
        assert self.manifest is not None
        start = self.writer.position
        if length > 0:
            self.writer.pad(length)
            self.manifest.metrics.padded_frames += length
        self.manifest.gaps.append(
            GapRecord(
                start_sample=start,
                length_samples=max(0, length),
                reason=reason,
                padded=length > 0,
                estimated=estimated,
            )
        )
        self._save()

    # --- writer thread ---

    def _drain_once(self, max_frames: int = 1 << 16) -> int:
        with self._drain_lock:
            return self._drain_locked(max_frames)

    def _drain_locked(self, max_frames: int) -> int:
        e = self.engine
        for adc, frames in e.timestamp_queue.read():
            self.timestamps.update(float(adc), int(frames))
        drops = e.drop_queue.read()
        # With drops pending, take everything so each drop's position is inside ``data``.
        data = e.ring.read(None if drops.size else max_frames)
        if drops.size:
            # Dropped blocks are not in the ring: insert silence at their place.
            for pos, frames in drops:
                self.manifest.metrics.dropped_frames += int(frames)  # type: ignore[union-attr]
                cut = max(0, min(data.shape[0], int(pos) - self._consumed_received()))
                if cut:
                    self._write(data[:cut])
                    data = data[cut:]
                self._gap(int(frames), "buffer_overflow", estimated=False)
                self._received_offset += int(frames)
        if data.shape[0]:
            self._write(data)
        if self.timestamps.discontinuities > self._seen_discontinuities:
            self._seen_discontinuities = self.timestamps.discontinuities
            lost = round(max(0.0, self.timestamps.last_shift_s) * self.config.samplerate)
            self._gap(lost, "input_overflow", estimated=True)
        return int(data.shape[0])

    def _consumed_received(self) -> int:
        """Frames of the callback's count that the writer has handled (excl. padding)."""
        return self._written_real + self._received_offset

    def _write(self, data: np.ndarray) -> None:
        t0 = time.perf_counter()
        self.writer.write(data)
        self._written_real += data.shape[0]
        m = self.manifest
        assert m is not None
        m.metrics.max_write_s = max(m.metrics.max_write_s, round(time.perf_counter() - t0, 4))
        for second in self.levels.add(data):
            for c in self.levels.idle_check(second, self.config.idle_warn_s):
                ch = self.config.channels[c]
                self._alarm(
                    "idle_channel",
                    f"channel {ch} ({self.config.participants[c]}) has stayed at its noise "
                    f"level for {self.config.idle_warn_s / 60:.0f} min while others are "
                    "active: muted or unplugged mic?",
                    ch,
                )

    def _writer_loop(self) -> None:
        try:
            while not self._writer_stop.is_set():
                if self._writer_paused.is_set() or self._drain_once() == 0:
                    time.sleep(0.01)
            while self._drain_once():
                pass
        except Exception as exc:
            self._writer_error = f"writer: {exc!r}"
            log.exception("writer failed")

    # --- main loop ---

    def check_disk(self) -> float | None:
        need = required_disk_bytes(
            len(self.config.channels), self.config.samplerate, self.config.planned_hours * 3600
        )
        free = self.disk_free(self.config.session_dir)
        if free < need:
            raise RecorderError(
                f"not enough free space: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed "
                f"for {self.config.planned_hours:g} h (24-bit + 10 %)"
            )
        return None

    def disk_left_s(self) -> float:
        per_s = required_disk_bytes(len(self.config.channels), self.config.samplerate, 1.0)
        return self.disk_free(self.config.session_dir) / max(per_s, 1)

    def run(self, stop: threading.Event, max_seconds: float | None = None) -> RecordingManifest:
        """Record until ``stop`` is set (or ``max_seconds``); always leaves files closed."""
        self.check_disk()
        assert self.manifest is not None
        if self.manifest.segments:  # resumed session: the time in between is a gap
            self._gap(0, "process_killed", estimated=True)
        inhibitor = self.inhibit() if self.inhibit else None
        try:
            stream = self.opener.open(self.engine.callback)
        except BaseException:
            if inhibitor is not None:
                inhibitor.release()
            raise
        self.manifest.actual_samplerate = float(
            getattr(stream, "samplerate", self.config.samplerate)
        )
        self.manifest.latency_s = getattr(stream, "latency", None)
        self._save()
        writer = threading.Thread(target=self._writer_loop, name="recorder-writer", daemon=True)
        writer.start()
        warned_disk = False
        last_level = last_disk = self.clock()
        state = "stopped"
        try:
            stream.start()
            self.engine.last_wall = time.monotonic()
            while not stop.is_set():
                time.sleep(0.05)
                now = self.clock()
                if max_seconds is not None and now - self._t0 >= max_seconds:
                    break
                if self.engine.fatal or self._writer_error:
                    self._alarm("fatal", self.engine.fatal or self._writer_error or "", fatal=True)
                    state = "interrupted"
                    break
                lost = not getattr(stream, "active", True) or (
                    time.monotonic() - self.engine.last_wall > NO_AUDIO_S
                )
                if lost:
                    new_stream = self._handle_device_lost(stream, stop)
                    if new_stream is None:
                        state = "interrupted"
                        break
                    stream = new_stream
                if now - last_level >= self.config.level_interval_s:
                    last_level = now
                    event = self.levels.take_event(round(self.audio_time(), 3))
                    if event:
                        self.sink(event)
                if now - last_disk >= DISK_CHECK_S:
                    last_disk = now
                    left = self.disk_left_s()
                    self.sink(
                        RecorderStatus(
                            self._time(),
                            self.writer.position,
                            self.engine.ring.available() / self.engine.ring.capacity,
                            left,
                            len(self.manifest.gaps),
                            "recording",
                        )
                    )
                    if left < self.config.disk_stop_s:
                        self._alarm(
                            "disk_full", f"only {left:.0f} s of disk left: stopping", fatal=True
                        )
                        break
                    if left < self.config.disk_warn_s and not warned_disk:
                        warned_disk = True
                        self._alarm("disk_low", f"about {left / 60:.0f} min of disk space left")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.debug("stream close failed", exc_info=True)
            self._writer_stop.set()
            writer.join()
            self.writer.close_segment()
            self.manifest.state = "interrupted" if state == "interrupted" else "stopped"
            self._save()
            if inhibitor is not None:
                inhibitor.release()
        return self.manifest

    def _handle_device_lost(self, stream: InputStream, stop: threading.Event) -> InputStream | None:
        """Close what we have; reconnect if allowed; pad the time away as a gap."""
        lost_at = time.monotonic()
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.debug("closing the lost stream failed", exc_info=True)
        # pause the writer, write what arrived, close the open segment
        self._writer_paused.set()
        while self._drain_once():
            pass
        with self._drain_lock:
            self.writer.close_segment()
        self._alarm("device_lost", "audio stopped arriving (device unplugged?)")
        if not self.config.reconnect:
            self._writer_paused.clear()
            return None
        while not stop.is_set():
            time.sleep(RECONNECT_EVERY_S)
            try:
                self.opener.refresh()
                new = self.opener.open(self.engine.callback)
                new.start()
            except Exception as exc:
                log.info("reconnect failed: %s", exc)
                continue
            away = round((time.monotonic() - lost_at) * self.config.samplerate)
            with self._drain_lock:
                self._gap(away, "device_lost", estimated=True)
            self._received_offset = self.engine.frames_received - self._written_real
            assert self.manifest is not None
            self.manifest.metrics.reconnects += 1
            self.engine.last_wall = time.monotonic()
            self._writer_paused.clear()
            self._alarm("reconnected", "audio is back; recording continues")
            return new
        self._writer_paused.clear()
        return None
