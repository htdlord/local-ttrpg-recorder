"""verify / recover, including a real process kill during recording."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from sessionscribe.core.audio.recovery import PARTIAL_SUFFIX, recover, verify
from sessionscribe.core.audio.track import VirtualTrack
from sessionscribe.core.session import Session, write_session
from tests.fake_stream import FakeOpener, Scenario
from tests.test_recorder import config, record

ROOT = Path(__file__).resolve().parents[1]


def test_verify_clean_and_tampered(tmp_path):
    cfg = config(tmp_path)
    record(cfg, FakeOpener([Scenario(blocks=250)]))
    report = verify(cfg.session_dir)
    assert report.ok and report.state == "stopped"
    assert set(report.frames_per_channel.values()) == {250 * 480}
    session = Session.load(cfg.session_dir)
    first = cfg.session_dir / session.config.recording.segments[0].file
    sf.write(str(first), np.zeros(48_000, np.float32), 48_000, subtype="PCM_24")  # tampered
    (cfg.session_dir / "rec" / "ch01_gm" / "9999.flac").write_bytes(b"x")
    bad = verify(cfg.session_dir)
    assert not bad.ok and bad.unknown_files == ["rec/ch01_gm/9999.flac"]
    assert [s.problem for s in bad.segments if not s.ok] == ["hash differs"]


def test_recover_open_segments_keeps_originals(tmp_path):
    cfg = config(tmp_path)
    record(cfg, FakeOpener([Scenario(blocks=150)]))
    session = Session.load(cfg.session_dir)
    rec = session.config.recording
    last = [s for s in rec.segments if s.index == max(x.index for x in rec.segments)]
    for s in last:  # pretend the run died before the last segment was closed
        s.status, s.frames, s.sha256 = "open", 0, None
    rec.state = "recording"
    write_session(cfg.session_dir / "session.toml", session.config)
    assert not verify(cfg.session_dir).ok
    m = recover(cfg.session_dir)
    assert m.state == "interrupted"
    assert [s.status for s in m.segments if s.index == last[0].index] == ["recovered"] * 3
    assert {s.frames for s in m.segments if s.index == last[0].index} == {24_000}
    assert (cfg.session_dir / (last[0].file + PARTIAL_SUFFIX)).is_file()
    assert verify(cfg.session_dir).ok


def test_process_kill_then_recover(tmp_path):
    folder = tmp_path / "killed"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.kill_recorder", str(folder)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    deadline = time.time() + 30
    manifest = folder / "session.toml"
    while time.time() < deadline:
        try:
            text = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
        except PermissionError:  # Windows: the child is replacing the file right now
            text = ""
        if text.count('status = "closed"') >= 4:
            break
        time.sleep(0.1)
    time.sleep(0.6)  # part of the next segment is on disk (flushed every second)
    if os.name == "nt":
        proc.kill()
    else:
        os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    before = verify(folder)
    closed = [s for s in before.segments if s.status == "closed"]
    assert closed and all(s.ok for s in closed)  # closed segments survive the kill intact
    assert any(s.status == "open" for s in before.segments)
    m = recover(folder)
    assert m.state == "interrupted"
    assert all(s.status in ("closed", "recovered", "damaged") for s in m.segments)
    after = verify(folder)
    assert len(set(after.frames_per_channel.values())) == 1  # channels still aligned
    session = Session.load(folder)
    lengths = {p.id: VirtualTrack.open(session.source_path(p)).frames for p in session.participants}
    assert len(set(lengths.values())) == 1
