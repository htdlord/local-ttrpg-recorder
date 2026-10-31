"""Child process for the kill test: records with a fake stream until killed."""

import sys
import threading
from pathlib import Path

from tests.fake_stream import FakeOpener, Scenario
from ttrpg_recorder.core.audio.recorder import Recorder, RecorderConfig

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    cfg = RecorderConfig(
        session_dir=folder,
        channels=(1, 2),
        participants=("gm", "anna"),
        segment_s=1.0,
        planned_hours=0.01,
        buffer_s=2.0,
    )
    opener = FakeOpener([Scenario(blocks=100_000, channels=2, amps=[0.3, 0.3], block_sleep=0.01)])
    print("ready", flush=True)
    Recorder(cfg, opener, disk_free=lambda p: 10**12).run(threading.Event(), max_seconds=60)
