import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from sessionscribe.core.audio.probe import (
    ProbeEngine,
    ProbeReport,
    TimestampMonitor,
    Verdict,
    clock_deviation_ppm,
    evaluate,
    required_disk_bytes,
    to_dbfs,
)

RATE = 48_000
BLOCK = 480


class Flags:
    """Stand-in for sounddevice.CallbackFlags."""

    def __init__(self, overflow=False, underflow=False, other=False):
        self.input_overflow = overflow
        self.input_underflow = underflow
        self._any = overflow or underflow or other

    def __bool__(self):
        return self._any


def tone(amplitude_dbfs: float, n: int, offset: int = 0, freq: float = 440.0) -> np.ndarray:
    t = (np.arange(n) + offset) / RATE
    amp = 10 ** (amplitude_dbfs / 20) * (2**31 - 1)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int32)


def feed(engine, blocks, channels, levels_db=None, *, adc0=1.0, gap_at=None, flags_at=None):
    adc = adc0
    for b in range(blocks):
        if levels_db is None:
            data = np.zeros((BLOCK, channels), np.int32)
        else:
            data = np.stack([tone(db, BLOCK, offset=b * BLOCK) for db in levels_db], axis=1)
        if gap_at is not None and b == gap_at:
            adc += BLOCK / RATE  # one block lost before this one
        flags = (flags_at or {}).get(b, Flags())
        engine.callback(data, BLOCK, SimpleNamespace(inputBufferAdcTime=adc), flags)
        adc += BLOCK / RATE


def drain_all(engine: ProbeEngine) -> None:
    while engine.drain():
        pass


def make_report(engine: ProbeEngine, channels, frames_written=None) -> ProbeReport:
    report = ProbeReport(
        device="fake",
        host_api="ASIO",
        device_index=0,
        device_auto_selected=False,
        channels=list(channels),
        samplerate=RATE,
        actual_samplerate=float(RATE),
        dtype="int32",
        blocksize=BLOCK,
        latency_s=0.01,
        requested_duration_s=1.0,
        wall_elapsed_s=engine.wall_elapsed(),
        interrupted=False,
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
        ring_capacity_s=1.0,
        ring_max_fill_s=0.1,
        writer_max_write_s=0.0,
        timestamps_available=engine.timestamps.seen > 0,
        timestamp_discontinuities=engine.timestamps.discontinuities,
        timestamp_jitter_median_ms=None,
        timestamp_jitter_p99_ms=None,
        timestamp_jitter_max_ms=None,
        missing_timestamps=engine.timestamps.missing,
        timestamp_queue_dropped=engine.timestamp_queue.dropped_blocks,
        clock_deviation_ppm=engine.clock_deviation_ppm(),
        levels=engine.levels(tuple(channels)),
    )
    report.verdict, report.reasons = evaluate(report)
    return report


def test_clean_run_passes_and_levels_identify_channels():
    engine = ProbeEngine(3, RATE, RATE)
    feed(engine, 100, 3, levels_db=[-6.0, -20.0, -40.0])
    drain_all(engine)
    report = make_report(engine, (1, 2, 3))
    assert report.verdict is Verdict.PASS, report.reasons
    assert [lv.peak_dbfs for lv in report.levels] == pytest.approx([-6.0, -20.0, -40.0], abs=0.1)
    # Sine RMS is 3.01 dB below its peak.
    assert [lv.rms_dbfs for lv in report.levels] == pytest.approx([-9.01, -23.01, -43.01], abs=0.1)


def test_writes_pcm24_flac_with_matching_frames(tmp_path):
    paths = [tmp_path / f"ch{c:02d}.flac" for c in (1, 2)]
    writers = [sf.SoundFile(p, "w", RATE, 1, subtype="PCM_24", format="FLAC") for p in paths]
    engine = ProbeEngine(2, RATE, RATE, writers=writers)
    feed(engine, 50, 2, levels_db=[-6.0, -30.0])
    drain_all(engine)
    for w in writers:
        w.close()
    written = [sf.info(str(p)).frames for p in paths]
    assert written == [50 * BLOCK] * 2
    assert sf.info(str(paths[0])).subtype == "PCM_24"
    data, _ = sf.read(paths[0], dtype="int32")
    assert to_dbfs(float(np.abs(data.astype(np.int64)).max())) == pytest.approx(-6.0, abs=0.1)
    report = make_report(engine, (1, 2), frames_written=written)
    assert report.verdict is Verdict.PASS, report.reasons


def test_overflow_flag_fails():
    engine = ProbeEngine(1, RATE, RATE)
    feed(engine, 20, 1, flags_at={5: Flags(overflow=True)})
    drain_all(engine)
    report = make_report(engine, (1,))
    assert report.verdict is Verdict.FAIL
    assert any("overflow" in r for r in report.reasons)


def test_other_status_flag_fails():
    engine = ProbeEngine(1, RATE, RATE)
    feed(engine, 20, 1, flags_at={3: Flags(other=True)})
    drain_all(engine)
    assert make_report(engine, (1,)).verdict is Verdict.FAIL


def test_ring_drop_fails():
    engine = ProbeEngine(1, RATE, BLOCK * 3)
    feed(engine, 10, 1)  # never drained while feeding -> drops
    drain_all(engine)
    report = make_report(engine, (1,))
    assert report.verdict is Verdict.FAIL
    assert any("dropped" in r for r in report.reasons)


def test_written_mismatch_fails():
    engine = ProbeEngine(2, RATE, RATE)
    feed(engine, 10, 2)
    drain_all(engine)
    report = make_report(engine, (1, 2), frames_written=[10 * BLOCK, 10 * BLOCK - 1])
    assert report.verdict is Verdict.FAIL


def test_column_selection_in_callback():
    engine = ProbeEngine(2, RATE, RATE, columns=np.asarray([1, 3], dtype=np.intp))
    feed(engine, 10, 4, levels_db=[-50.0, -6.0, -50.0, -12.0])
    drain_all(engine)
    peaks = [lv.peak_dbfs for lv in engine.levels((2, 4))]
    assert peaks == pytest.approx([-6.0, -12.0], abs=0.1)


def test_lost_block_is_a_timestamp_discontinuity():
    engine = ProbeEngine(1, RATE, RATE)
    feed(engine, 200, 1, gap_at=100)
    drain_all(engine)
    report = make_report(engine, (1,))
    assert report.timestamp_discontinuities == 1
    assert report.verdict is Verdict.FAIL


def run_monitor(blocks, *, block=BLOCK, jitter_s=0.0, drift_ppm=0.0, lost_at=(), offsets=None):
    """Feed a TimestampMonitor; ``lost_at`` block indices are preceded by one lost block."""
    rng = np.random.default_rng(1)
    mon = TimestampMonitor(RATE)
    adc = 1.0
    for b in range(blocks):
        if b in lost_at:
            adc += block / RATE * (1 + drift_ppm * 1e-6)
        extra = offsets(b) if offsets else 0.0
        mon.update(adc + rng.uniform(-jitter_s, jitter_s) + extra, block)
        adc += block / RATE * (1 + drift_ppm * 1e-6)
    return mon


def test_single_lost_block_is_detected():
    # 10 ms blocks: one lost block is exactly 2x the half-block threshold.
    assert run_monitor(500, lost_at={250}).discontinuities == 1


def test_single_lost_block_is_detected_under_jitter_and_drift():
    mon = run_monitor(3000, jitter_s=0.004, drift_ppm=500, lost_at={1500})
    assert mon.discontinuities == 1


def test_single_lost_block_with_large_asio_blocks():
    assert run_monitor(300, block=2048, jitter_s=0.002, lost_at={150}).discontinuities == 1


def test_several_lost_blocks_are_counted_separately():
    assert run_monitor(1000, jitter_s=0.002, lost_at={200, 600}).discontinuities == 2


def test_jitter_and_drift_alone_are_tolerated():
    # ±4 ms on 10 ms blocks with 500 ppm clock drift.
    assert run_monitor(3000, jitter_s=0.004, drift_ppm=500).discontinuities == 0


def test_single_late_timestamp_is_not_a_discontinuity():
    mon = run_monitor(300, offsets=lambda b: 0.03 if b == 150 else 0.0)
    assert mon.discontinuities == 0


def test_shift_shorter_than_100_ms_is_not_a_discontinuity():
    # 8 blocks (80 ms) of late stamps, then back on track.
    mon = run_monitor(300, offsets=lambda b: 0.01 if 100 <= b < 108 else 0.0)
    assert mon.discontinuities == 0


def test_jitter_statistics():
    stats = run_monitor(2000, jitter_s=0.002).jitter_stats()
    assert stats is not None
    assert 0.0005 < stats.median_s < 0.0015  # median |uniform(±2 ms)| is 1 ms
    assert stats.median_s < stats.p99_s <= stats.max_s <= 0.0045
    assert TimestampMonitor(RATE).jitter_stats() is None


def test_missing_timestamps_warn():
    engine = ProbeEngine(1, RATE, RATE)
    feed(engine, 20, 1, adc0=0.0)  # first stamp is 0 = missing, the rest are valid
    drain_all(engine)
    report = make_report(engine, (1,))
    assert report.missing_timestamps == 1
    assert report.verdict is Verdict.WARN
    verdict, reasons = evaluate(replace(report, timestamps_available=False))
    assert verdict is Verdict.WARN and "continuity not verified" in reasons[0]


def test_clock_deviation_warns_only_on_long_runs():
    engine = ProbeEngine(1, RATE, RATE)
    feed(engine, 10, 1)
    drain_all(engine)
    base = make_report(engine, (1,))
    assert evaluate(replace(base, wall_elapsed_s=1800.0, clock_deviation_ppm=350.0))[0] is (
        Verdict.WARN
    )
    assert evaluate(replace(base, wall_elapsed_s=5.0, clock_deviation_ppm=350.0))[0] is (
        Verdict.PASS
    )
    assert evaluate(replace(base, wall_elapsed_s=1800.0, clock_deviation_ppm=-150.0))[0] is (
        Verdict.PASS
    )


def test_clock_deviation_ppm():
    assert clock_deviation_ppm(48_048, 48_000, 1.0) == pytest.approx(1000.0)
    assert clock_deviation_ppm(0, 48_000, 1.0) is None


def test_required_disk_bytes():
    assert required_disk_bytes(8, 48_000, 3600) == int(8 * 48_000 * 3 * 3600 * 1.1)


def test_silence_is_minus_inf_and_json_stays_valid():
    assert to_dbfs(0.0) == -math.inf
    engine = ProbeEngine(1, RATE, RATE)
    feed(engine, 5, 1)
    drain_all(engine)
    text = make_report(engine, (1,)).to_json()
    assert "Infinity" not in text and "NaN" not in text
