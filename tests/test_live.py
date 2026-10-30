"""Device lookup, meter, sleep inhibitor (no hardware)."""

import pytest

from sessionscribe.core.audio.backend import AudioSetupError, InputDevice
from sessionscribe.core.audio.live import Meter, bar, find_device
from sessionscribe.core.audio.power import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, inhibit_sleep
from tests.fake_stream import FakeOpener, Scenario

DEVICES = [
    InputDevice(3, "UMC1820", "Windows WASAPI", 2, 48000),
    InputDevice(4, "IN 01-08 (UMC1820)", "Windows WASAPI", 8, 48000),
    InputDevice(57, "UMC ASIO Driver", "ASIO", 18, 48000),
    InputDevice(60, "Microphone (Webcam)", "Windows WASAPI", 1, 48000),
]


def test_find_device_by_name_and_host_api():
    assert find_device(DEVICES, "UMC ASIO Driver", "asio").index == 57
    assert find_device(DEVICES, "in 01-08", "wasapi").index == 4
    assert find_device(DEVICES, "UMC1820", "wasapi").index == 3  # exact name wins
    with pytest.raises(AudioSetupError, match="several devices"):
        find_device(DEVICES, "U", "wasapi")
    with pytest.raises(AudioSetupError, match="no input device named"):
        find_device(DEVICES, "Scarlett", "asio")
    with pytest.raises(AudioSetupError, match="no input devices on host API"):
        find_device(DEVICES, "UMC", "jack")


def test_meter_reports_levels():
    events = []
    Meter(FakeOpener([Scenario(blocks=200, amps=[0.5, 0.05, 0.0])]), 3, 48_000, 0.05).run(
        1.0, events.append
    )
    assert events
    last = events[-1]
    assert last.rms_dbfs[0] > last.rms_dbfs[1] > last.rms_dbfs[2]
    assert bar(-60) == "." * 20 and bar(0) == "#" * 20


def test_sleep_inhibitor_windows():
    calls = []
    inh = inhibit_sleep(platform="win32", set_state=lambda f: calls.append(f) or 1)
    assert inh.active and calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    inh.release()
    assert calls[-1] == ES_CONTINUOUS and not inh.active
    failed = inhibit_sleep(platform="win32", set_state=lambda f: 0)
    assert not failed.active and "failed" in failed.how


def test_sleep_inhibitor_linux():
    started = []

    class Proc:
        def terminate(self):
            started.append("terminated")

        def wait(self, timeout):
            return 0

    def popen(cmd, **kw):
        started.append(cmd)
        return Proc()

    inh = inhibit_sleep(platform="linux", which=lambda n: "/usr/bin/systemd-inhibit", popen=popen)
    assert inh.active and "--what=sleep:idle" in started[0]
    inh.release()
    assert started[-1] == "terminated"
    missing = inhibit_sleep(platform="linux", which=lambda n: None)
    assert not missing.active and "not found" in missing.how
    assert not inhibit_sleep(platform="darwin").active
