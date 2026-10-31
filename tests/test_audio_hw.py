"""Needs a real audio input device. Excluded from CI."""

import pytest

from ttrpg_recorder.core.audio.probe import ProbeConfig, Verdict, run_probe


@pytest.mark.audio_hw
def test_default_device_short_probe():
    report = run_probe(ProbeConfig(channels=(1,), duration_s=1.0, write=False))
    assert report.frames_received > 0
    assert report.verdict is not Verdict.FAIL, report.reasons


def _env(name, default=None):
    """TTRPG_RECORDER_TEST_<name>, or the SESSIONSCRIBE_TEST_<name> used before the rename."""
    import os

    return os.environ.get(f"TTRPG_RECORDER_TEST_{name}") or os.environ.get(
        f"SESSIONSCRIBE_TEST_{name}", default
    )


def _rig():
    """Real interface from the environment, e.g. TTRPG_RECORDER_TEST_DEVICE="UMC ASIO Driver",
    TTRPG_RECORDER_TEST_HOST_API=asio, TTRPG_RECORDER_TEST_CHANNELS=1-8."""
    device = _env("DEVICE")
    host_api = _env("HOST_API")
    if not device or not host_api:
        pytest.skip("set TTRPG_RECORDER_TEST_DEVICE and TTRPG_RECORDER_TEST_HOST_API")
    import sys

    from ttrpg_recorder.core.audio import backend

    if (
        "asio" in host_api.lower()
        and "sounddevice" in sys.modules
        and not backend._asio_requested_at_import
    ):
        pytest.skip(
            "sounddevice was loaded without ASIO earlier in this run; run with -k rig tests"
        )
    from ttrpg_recorder.core.audio.backend import parse_channel_spec

    return (
        device,
        host_api,
        parse_channel_spec(_env("CHANNELS", "1-8")),
    )


@pytest.mark.audio_hw
def test_two_minute_recording_verifies(tmp_path):
    import threading

    from ttrpg_recorder.core.audio.live import SoundDeviceOpener
    from ttrpg_recorder.core.audio.recorder import Recorder, RecorderConfig
    from ttrpg_recorder.core.audio.recovery import verify

    device, host_api, channels = _rig()
    opener = SoundDeviceOpener(device, host_api, channels, 48_000, 480)
    ids = tuple(f"ch{c:02d}" for c in channels)
    cfg = RecorderConfig(
        session_dir=tmp_path / "s",
        channels=channels,
        participants=ids,
        segment_s=60,
        planned_hours=0.1,
    )
    manifest = Recorder(cfg, opener).run(threading.Event(), max_seconds=120)
    assert manifest.state == "stopped"
    assert manifest.metrics.dropped_frames == 0 and not manifest.gaps
    report = verify(cfg.session_dir)
    assert report.ok, [s for s in report.segments if not s.ok]


@pytest.mark.audio_hw
def test_meter_and_device_refresh():
    from ttrpg_recorder.core.audio.live import Meter, SoundDeviceOpener

    device, host_api, channels = _rig()
    opener = SoundDeviceOpener(device, host_api, channels, 48_000, 480)
    events = []
    Meter(opener, len(channels), 48_000).run(1.0, events.append)
    assert events
    opener.refresh()  # PortAudio re-initialised; the same device is found again
    events.clear()
    Meter(opener, len(channels), 48_000).run(1.0, events.append)
    assert events
