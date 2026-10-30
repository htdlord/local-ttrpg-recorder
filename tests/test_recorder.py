"""Recorder with a fake stream: segments, manifest, gaps, device loss, disk, idle."""

import threading
import time

import numpy as np
import pytest

from sessionscribe.core.audio import recorder as rec
from sessionscribe.core.audio.manifest import file_sha256
from sessionscribe.core.audio.recorder import Recorder, RecorderConfig, RecorderError
from sessionscribe.core.audio.track import VirtualTrack
from sessionscribe.core.session import Session
from tests.fake_stream import FakeOpener, Scenario

RATE = 48_000


def config(tmp_path, **kw):
    base = dict(
        session_dir=tmp_path / "s",
        channels=(1, 3, 5),
        participants=("gm", "anna", "bela"),
        segment_s=1.0,
        planned_hours=0.01,
        buffer_s=2.0,
    )
    base.update(kw)
    return RecorderConfig(**base)


def record(cfg, opener, *, until=None, sink=None, **kw):
    events = []
    r = Recorder(
        cfg,
        opener,
        sink=sink or events.append,
        disk_free=kw.pop("disk_free", lambda p: 10**12),
        **kw,
    )
    stop = threading.Event()

    def stopper():
        while not (until or (lambda: opener.opened and opener.opened[-1].done.is_set()))():
            time.sleep(0.01)
        time.sleep(0.2)
        stop.set()

    threading.Thread(target=stopper, daemon=True).start()
    manifest = r.run(stop, max_seconds=30)
    return r, manifest, events


def test_segments_manifest_and_read_back(tmp_path):
    cfg = config(tmp_path)
    _, m, events = record(cfg, FakeOpener([Scenario(blocks=250)]))  # 2.5 s
    assert m.state == "stopped" and m.metrics.frames == 250 * 480
    gm = m.segments_for(1)
    assert [s.frames for s in gm] == [48_000, 48_000, 24_000]
    assert all(s.status == "closed" and s.sha256 for s in m.segments)
    assert file_sha256(cfg.session_dir / gm[0].file) == gm[0].sha256
    session = Session.load(cfg.session_dir)
    assert [p.source for p in session.participants] == [
        "rec/ch01_gm",
        "rec/ch03_anna",
        "rec/ch05_bela",
    ]
    assert session.config.recording.device_name == "Fake UMC"
    track = VirtualTrack.open(cfg.session_dir / "rec/ch01_gm")
    assert track.frames == 250 * 480
    x = track.read(0, 4800)
    assert 0.25 < np.abs(x).max() < 0.35  # the 0.3 tone, 24-bit
    assert any(isinstance(e, rec.LevelEvent) for e in events)
    assert not m.gaps


def test_ring_overflow_is_padded_and_keeps_the_timeline(tmp_path, monkeypatch):
    real_write = rec.SegmentWriter.write
    slow = {"n": 0}

    def slow_write(self, block):
        slow["n"] += 1
        if slow["n"] == 3:
            time.sleep(0.6)  # the writer stalls; the small ring overflows
        real_write(self, block)

    monkeypatch.setattr(rec.SegmentWriter, "write", slow_write)
    cfg = config(tmp_path, buffer_s=0.1)
    opener = FakeOpener([Scenario(blocks=300, block_sleep=0.002)])
    _, m, _ = record(cfg, opener)
    gaps = [g for g in m.gaps if g.reason == "buffer_overflow"]
    assert gaps and all(g.padded and not g.estimated for g in gaps)
    assert m.metrics.dropped_frames == sum(g.length_samples for g in gaps) > 0
    assert m.metrics.frames == opener.opened[0].sent_frames  # dropped audio became silence
    lengths = {c: sum(s.frames for s in m.segments_for(c)) for c in (1, 3, 5)}
    assert len(set(lengths.values())) == 1


def test_timestamp_jump_is_an_estimated_gap(tmp_path):
    cfg = config(tmp_path)
    opener = FakeOpener([Scenario(blocks=200, adc_jump_at=100, adc_jump_s=0.05)])
    _, m, _ = record(cfg, opener)
    (gap,) = [g for g in m.gaps if g.reason == "input_overflow"]
    assert gap.estimated and gap.padded and abs(gap.length_samples - 2400) <= 480
    assert m.metrics.frames == 200 * 480 + gap.length_samples


def test_device_lost_and_reconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "RECONNECT_EVERY_S", 0.05)
    cfg = config(tmp_path)
    opener = FakeOpener([Scenario(blocks=400, die_at=100), Scenario(blocks=100)], fail_opens=1)
    _, m, _ = record(
        cfg, opener, until=lambda: len(opener.opened) == 2 and opener.opened[1].done.is_set()
    )
    (gap,) = [g for g in m.gaps if g.reason == "device_lost"]
    assert gap.estimated and gap.padded and gap.start_sample == 100 * 480
    assert m.metrics.reconnects == 1 and opener.refreshes >= 2
    assert [a.kind for a in m.alarms][:2] == ["device_lost", "reconnected"]
    assert m.state == "stopped"
    assert all(s.status == "closed" for s in m.segments)


def test_device_lost_without_reconnect_is_interrupted(tmp_path):
    cfg = config(tmp_path, reconnect=False)
    _, m, _ = record(cfg, FakeOpener([Scenario(blocks=400, die_at=60)]), until=lambda: False)
    assert m.state == "interrupted" and m.metrics.frames == 60 * 480


def test_disk_checks(tmp_path, monkeypatch):
    with pytest.raises(RecorderError, match="not enough free space"):
        Recorder(config(tmp_path), FakeOpener([Scenario()]), disk_free=lambda p: 1000).run(
            threading.Event()
        )
    monkeypatch.setattr(rec, "DISK_CHECK_S", 0.05)
    free = {"bytes": 10**12}
    cfg = config(tmp_path / "b", disk_warn_s=1800, disk_stop_s=120)
    opener = FakeOpener([Scenario(blocks=5000, block_sleep=0.001)])

    def disk(_):
        return free["bytes"]

    r = Recorder(cfg, opener, disk_free=disk, sink=lambda e: None)
    stop = threading.Event()

    def shrink():
        time.sleep(0.3)
        per_s = rec.required_disk_bytes(3, RATE, 1.0)
        free["bytes"] = int(per_s * 600)  # 10 min left: warning
        time.sleep(0.3)
        free["bytes"] = int(per_s * 60)  # 1 min left: stop

    threading.Thread(target=shrink, daemon=True).start()
    m = r.run(stop, max_seconds=20)
    assert [a.kind for a in m.alarms] == ["disk_low", "disk_full"]
    assert m.state == "stopped" and all(s.status == "closed" for s in m.segments)


def test_idle_channel_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "NOISE_LEARN_S", 2)
    cfg = config(tmp_path, idle_warn_s=3)
    # gm and anna take turns each second; bela (channel 5) only speaks in the first one

    def amps(b):
        second = b // 100
        return [0.3 * (second % 2), 0.3 * (1 - second % 2), 0.3 if second == 0 else 0.0]

    _, m, _ = record(cfg, FakeOpener([Scenario(blocks=800, amps_fn=amps)]))
    idle = [a for a in m.alarms if a.kind == "idle_channel"]
    assert [a.channel for a in idle] == [5]


def test_existing_recording_session_is_not_overwritten(tmp_path):
    cfg = config(tmp_path)
    record(cfg, FakeOpener([Scenario(blocks=50)]))
    m2 = record(cfg, FakeOpener([Scenario(blocks=50)]))[1]  # stopped: resume appends
    assert m2.metrics.frames == 100 * 480 and m2.gaps[0].reason == "process_killed"
    assert len({s.index for s in m2.segments}) == 2
    with pytest.raises(RecorderError, match="same channels"):
        Recorder(config(tmp_path, channels=(1, 2, 3)), FakeOpener([Scenario()]))


def test_manifest_gaps_reach_the_transcript(tmp_path):
    from sessionscribe.core.config import load_settings
    from sessionscribe.core.pipeline.run import process_session
    from tests.fakes import FakeFactory

    cfg = config(tmp_path)
    opener = FakeOpener([Scenario(blocks=200, adc_jump_at=100, adc_jump_s=0.05)])
    record(cfg, opener)
    session = Session.load(cfg.session_dir)
    assert [g.reason for g in session.gaps()] == ["input_overflow"]
    result, _ = process_session(
        session,
        load_settings(overrides=['bleed.mode="off"']),
        isolation="inline",
        factory=FakeFactory(),
        model_identity=lambda s: {"id": "fake"},
    )
    assert [g.reason for g in result.gaps] == ["input_overflow"]
    # (the overlaps_gap flag itself is covered in test_merge_quality; here the padded
    # silence may split the fake engine's segment exactly at the gap)
    from sessionscribe.core.exporters.formats import to_md

    assert "Recording gaps" in to_md(result) and "input_overflow" in to_md(result)
